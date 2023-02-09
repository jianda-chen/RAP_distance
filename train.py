import numpy as np
import torch
torch.set_num_threads(10)
# import torchvision
import argparse
import os
import gym
import time
import json
import dmc2gym

from shutil import copyfile
import utils
from logger import Logger
from video import VideoRecorder



def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--resource_files', type=str)
    parser.add_argument('--eval_resource_files', type=str)
    parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
    parser.add_argument('--total_frames', default=1000, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    # train
    parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'RAP'])
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    parser.add_argument('--load_encoder', default=None, type=str)
    # eval
    parser.add_argument('--eval_freq', default=10, type=int)  # TODO: master had 10000
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str, choices=['pixel', 'pixelCarla096', 'pixelCarla098', 'identity', 'robo168'])
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--decoder_type', default='pixel', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=1e-3, type=float)
    parser.add_argument('--alpha_beta', default=0.9, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--port', default=2000, type=int)


    parser.add_argument('--rap_structural_distance', default='l1_smooth', type=str, choices=['l1_smooth', 'l2', 'mico_angular', 'x^2+y^2-xy'])
    parser.add_argument('--rap_reward_dist', default=False, action='store_true')
    parser.add_argument('--rap_square_target', default=False, action='store_true')
    
    
    parser.add_argument('--robosuite', default=False, action='store_true')
    parser.add_argument('--eleurent', default=False, action='store_true')

    parser.add_argument('--note', default=None, type=str)


    args = parser.parse_args()
    print(args)
    return args


def evaluate(env, agent, video, num_episodes, L, step, device=None, embed_viz_dir=None,):

    # embedding visualization
    obses = []
    values = []
    embeddings = []

    for i in range(num_episodes):
        # carla metrics:

        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)

            if embed_viz_dir:
                obses.append(obs)
                with torch.no_grad():
                    values.append(min(agent.critic(torch.Tensor(obs).to(device).unsqueeze(0), torch.Tensor(action).to(device).unsqueeze(0))).item())
                    embeddings.append(agent.critic.encoder(torch.Tensor(obs).unsqueeze(0).to(device)).cpu().detach().numpy())

            obs, reward, done, info = env.step(action)

            video.record(env)
            episode_reward += reward

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
        L.dump(step)

    if embed_viz_dir:
        dataset = {'obs': obses, 'values': values, 'embeddings': embeddings}
        torch.save(dataset, os.path.join(embed_viz_dir, 'train_dataset_{}.pt'.format(step)))

    # L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'baseline':
        from agent.baseline_agent import BaselineAgent
        agent = BaselineAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    elif args.agent == 'RAP':
        from agent.rap import RAPAgent
        import agent.rap
        source_code_file_path = agent.rap.__file__
        agent = RAPAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            bisim_coef=args.bisim_coef,
            rap_structural_distance=args.rap_structural_distance,
            rap_reward_dist=args.rap_reward_dist,
            rap_square_target=args.rap_square_target,
        )
        agent.source_code_file_path = source_code_file_path
    elif args.agent == 'bisim':
        import agent.bisim_agent
        source_code_file_path = agent.bisim_agent.__file__
        from agent.bisim_agent import BisimAgent
        agent = BisimAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            bisim_coef=args.bisim_coef
        )

    return agent


def main():
    args = parse_args()
    print(args.note)

    utils.set_seed_everywhere(args.seed)
    if args.domain_name == 'carla':
        from secant.envs.carla import make_carla
        train_port = args.port
        env = make_carla(
            f"town04", 
            client_port=train_port, 
            modalities=["rgb"],
            action_repeat=args.action_repeat,
            weather='random',
            weather_args=[
                "clear_noon",
                "wet_sunset",
                "wet_cloudy_noon",
                "soft_rain_sunset",
                "mid_rain_sunset",
                "hard_rain_noon",
            ],
            frame_stack=3,
        )

        eval_env = env

    elif args.robosuite:
        from secant.envs.robosuite import make_robosuite, ALL_TASKS, make_robosuite_random_sample

        env = make_robosuite(
            task=args.task_name,
            mode='random-domain',
            scene_id=1,
            image_height=128, # 168 # 84 # 128
            image_width=128,
            # obs_cameras='frontview',
        )

        eval_env = make_robosuite(
            task=args.task_name,
            mode='random-domain',
            scene_id=2,
            image_height=128, # 168 # 84 # 128
            image_width=128,
            # obs_cameras='frontview',
        )
    elif args.eleurent:
        from eleurent_env import EleurentHighWay
        env = EleurentHighWay(
            domain_name=args.domain_name,
            task_name=args.task_name,
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
            )
        eval_env = env

    else:
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )
        env.seed(args.seed)

        eval_env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.eval_resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )
        eval_env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel') and not args.robosuite and args.domain_name != 'carla':
        env = utils.FrameStack(env, k=args.frame_stack)
        eval_env = utils.FrameStack(eval_env, k=args.frame_stack)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    print(env.observation_space.shape)
    print(env.action_space.shape)

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        is_framestack=(args.domain_name != 'carla'),
    )

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    if hasattr(agent, 'source_code_file_path'):
        if agent.source_code_file_path is not None:
            code_dir = os.path.join(args.work_dir, 'code')
            os.makedirs(code_dir, exist_ok=True)
            copyfile(agent.source_code_file_path, os.path.join(code_dir, 
                    os.path.basename(agent.source_code_file_path)))

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if args.decoder_type == 'inverse':
                for i in range(1, args.k):  # fill k_obs with 0s if episode is done
                    replay_buffer.k_obses[replay_buffer.idx - i] = 0
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if episode % args.eval_freq == 0:
                L.log('eval/episode', episode, step)
                evaluate(eval_env, agent, video, args.num_eval_episodes, L, step)
                if args.save_model:
                    agent.save(model_dir, step=None)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

            L.log('train/episode', episode, step)
        # torchvision.utils.save_image(torch.tensor(obs[-3:]) / 255., "fig/obs{}.png".format(step))

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
            log_pi = utils.gym_action_space_log_prob(env.action_space, action)
        else:
            with utils.eval_mode(agent):
                action, log_pi = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        curr_reward = reward
        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        replay_buffer.add(obs, action, log_pi, curr_reward, reward, next_obs, done_bool,)
        if args.decoder_type == 'inverse':
            np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)

        obs = next_obs
        episode_step += 1


def collect_data(env, agent, num_rollouts, path_length, checkpoint_path):
    rollouts = []
    for i in range(num_rollouts):
        obses = []
        acs = []
        rews = []
        observation = env.reset()
        for j in range(path_length):
            action = agent.sample_action(observation)
            next_observation, reward, done, _ = env.step(action)
            obses.append(observation)
            acs.append(action)
            rews.append(reward)
            observation = next_observation
        obses.append(next_observation)
        rollouts.append((obses, acs, rews))

    from scipy.io import savemat

    savemat(
        os.path.join(checkpoint_path, "dynamics-data.mat"),
        {
            "trajs": np.array([path[0] for path in rollouts]),
            "acs": np.array([path[1] for path in rollouts]),
            "rews": np.array([path[2] for path in rollouts])
        }
    )


if __name__ == '__main__':
    main()
