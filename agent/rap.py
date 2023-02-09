import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from sac_ae import  Actor, Critic, LOG_FREQ, gaussian_logprob
from transition_model import make_transition_model


EPSILON = 1e-9

def _sqrt(x, tol=0.):
    tol = torch.zeros_like(x)
    return torch.sqrt(torch.maximum(x, tol))

def cosine_distance(x, y):
    numerator = torch.sum(x * y, dim=-1, keepdim=True)
    # print("numerator", numerator.shape, numerator)
    denominator = torch.sqrt(
        torch.sum(x.pow(2.), dim=-1, keepdim=True)) * torch.sqrt(torch.sum(y.pow(2.), dim=-1, keepdim=True))
    cos_similarity = numerator / (denominator + EPSILON)

    return torch.atan2(_sqrt(1. - cos_similarity.pow(2.)), cos_similarity)


class StateRewardDecoder(nn.Module):
    def __init__(self, encoder_feature_dim, max_sigma=1e0, min_sigma=1e-4):
        super().__init__()
        self.trunck = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 2))

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

    def forward(self, x):
        y = self.trunck(x)
        sigma = y[..., 1:2]
        mu = y[..., 0:1]
        sigma = torch.sigmoid(sigma)  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma
       
    
    def loss(self, mu, sigma, r, reduce='mean'):
        diff = (mu - r.detach()) / sigma
        if reduce == 'none':
            loss = 0.5 * (0.5 * diff.pow(2) + torch.log(sigma))
        elif  reduce =='mean':
            loss = 0.5 * torch.mean(0.5 * diff.pow(2) + torch.log(sigma))
        else:
            raise NotImplementedError

        return loss

class RAPAgent(object):
    """RAP distance"""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        transition_model_type,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        encoder_stride=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        bisim_coef=0.5,
        rap_structural_distance='l1_smooth',
        rap_reward_dist=False,
        rap_square_target=False,
    ):
        print(__file__)
        print(rap_structural_distance, rap_reward_dist, rap_square_target)
        self.rap_structural_distance=rap_structural_distance
        self.rap_reward_dist=rap_reward_dist
        self.rap_square_target=rap_square_target


        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.transition_model_type = transition_model_type
        self.bisim_coef = bisim_coef

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride
        ).to(device)


        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape
            # transition_model_type, encoder_feature_dim, [0,]
        ).to(device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)

        self.state_reward_decoder = StateRewardDecoder(
            encoder_feature_dim).to(device)

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        # optimizer for decoder
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters())
            + list(self.state_reward_decoder.parameters()),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, log_pi, log_std = self.actor(obs, compute_log_pi=True)
            return pi.cpu().data.numpy().flatten(), log_pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
        # critic_loss = F.mse_loss(current_Q1,
        #                          target_Q) + F.mse_loss(current_Q2, target_Q)
        critic_loss = F.smooth_l1_loss(current_Q1,
                                 target_Q) + F.smooth_l1_loss(current_Q2, target_Q)
                                 
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 40.)
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 40.)
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_encoder(self, obs, action, behavioural_log_pi, reward, next_obs, L, step):
        h = self.critic.encoder(obs)            

        # Sample random states across episodes at random
        batch_size = obs.size(0)
        perm = np.random.permutation(batch_size)
        h2 = h[perm]

        

        with torch.no_grad():
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
            reward2 = reward[perm]

        if pred_next_latent_sigma1 is None:
            pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
        if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
        elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
        else:
            raise NotImplementedError

        # z_dist = F.smooth_l1_loss(h, h2, reduction='none').mean(dim=-1, keepdim=True)
        z_dist = self.metric_func(h, h2)
       

        if self.rap_reward_dist:
            reward_mu, reward_sigma = self.state_reward_decoder(h)
            loss_reward_decoder = self.state_reward_decoder.loss(
                reward_mu, reward_sigma, reward
            )
        if self.rap_reward_dist:
            reward_var = reward_sigma.detach().pow(2.)
            reward_var2 = reward_var[perm]
            r_var = reward_var
            r_var2 = reward_var2

            reward_mu2 = reward_mu[perm]
            r_mean = reward_mu
            r_mean2 = reward_mu2
        
        with torch.no_grad():
            if self.rap_reward_dist:
                r_dist = (reward - reward2).pow(2.)
                r_dist = F.relu(r_dist - r_var - r_var2)
                # r_dist = (r_dist - r_var - r_var2).abs()
                r_dist = r_dist.sqrt()
                # r_dist = F.smooth_l1_loss(r_mean, r_mean2, reduction='none')
            else:
                r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')
        if self.transition_model_type == '':
            transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none')
        else:
            if self.rap_structural_distance == 'x^2+y^2-xy' or self.rap_structural_distance == 'rap_angular':
                transition_dist = self.metric_func(pred_next_latent_mu1, pred_next_latent_mu2)
            else:
                transition_dist = torch.sqrt(
                    (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
                    (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
                ).mean(dim=-1, keepdim=True)
            
        if self.rap_square_target:
            assert self.rap_reward_dist
            with torch.no_grad():
                r_dist_square = (reward - reward2).pow(2.)
                r_dist_square_minus_var = r_dist_square - r_var - r_var2
            diff_square = (z_dist - self.discount * transition_dist).pow(2.)
            loss = F.smooth_l1_loss(diff_square, r_dist_square_minus_var, reduction='mean')
        else:
            rap_dist_target = r_dist + self.discount * transition_dist
            loss = F.smooth_l1_loss(z_dist, rap_dist_target, reduction='mean')
        if self.rap_reward_dist:
            loss = loss + loss_reward_decoder
        L.log('train_ae/encoder_loss', loss, step)
        return loss

    def metric_func(self, x, y):
        if self.rap_structural_distance == 'l2':
            dist = F.pairwise_distance(x, y, p=2, keepdim=True)
        elif self.rap_structural_distance == 'l1_smooth':
            dist = F.smooth_l1_loss(x, y, reduction='none')
            dist = dist.mean(dim=-1, keepdim=True)
        elif self.rap_structural_distance == 'mico_angular':
            beta = 1e-6 #1e-5 # #0.1
            base_distances = cosine_distance(x, y)
            # print("base_distances", base_distances)
            norm_average = (x.pow(2.).sum(dim=-1, keepdim=True)
            # norm_average = 0.5 * (x.pow(2.).sum(dim=-1, keepdim=True) 
                + y.pow(2.).sum(dim=-1, keepdim=True))
            dist = norm_average + beta * base_distances
        elif self.rap_structural_distance == 'x^2+y^2-xy':
            # beta = 1.0 # 0 < beta < 2
            k = 0.1 # 0 < k < 2
            base_distances = (x * y).sum(dim=-1, keepdim=True)
            # print("base_distances", base_distances)
            norm_average = (x.pow(2.).sum(dim=-1, keepdim=True) 
                + y.pow(2.).sum(dim=-1, keepdim=True))
            # dist = norm_average - (2. - beta) * base_distances
            dist = norm_average - k * base_distances
            # dist = dist.sqrt()
        else:
            raise NotImplementedError
        return dist

    def update_transition_reward_model(self, obs, action, next_obs, reward, L, step):
        h = self.critic.encoder(obs)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        # pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(h)
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.critic.encoder(next_obs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_ae/transition_loss', loss, step)

        pred_next_latent = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
        pred_next_reward = self.reward_decoder(pred_next_latent)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        total_loss = loss + reward_loss
        return total_loss

    def update(self, replay_buffer, L, step):
        obs, action, behavioural_log_pi, _, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        transition_reward_loss = self.update_transition_reward_model(obs, action, next_obs, reward, L, step)
        encoder_loss = self.update_encoder(obs, action, behavioural_log_pi, reward, next_obs, L, step)
        total_loss = self.bisim_coef * encoder_loss + 1e-4 * transition_reward_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # print("total_loss", total_loss.item())

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

            
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.reward_decoder.state_dict(),
            '%s/reward_decoder_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.reward_decoder.load_state_dict(
            torch.load('%s/reward_decoder_%s.pt' % (model_dir, step))
        )

