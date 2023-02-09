import argparse
import os
import time
from pprint import pformat
import socket
from secant.envs.carla import make_carla
import subprocess as sp
# import psutil


def random_free_tcp_port() -> int:
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port

def check_port_in_use(port: int) -> bool:
    a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    location = ("127.0.0.1", 80)
    result_of_check = a_socket.connect_ex(location)
    a_socket.close()
    return result_of_check == 0

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--port",
    type=int,
    default=-1,
    help="If port == -1, randomly select a free port to use",
)
parser.add_argument("-m", "--map", type=int, default=1, help="town0x map, from 1 to 5")
parser.add_argument(
    "-ne", "--no-server", action="store_true", help="Do not launch a server subprocess"
)
parser.add_argument(
    "-e",
    "--server-executable",
    type=str,
    default="/usr/local/bin/carla-server",
    help="town0x map, from 1 to 5",
)
args = parser.parse_args()
assert 1 <= args.map <= 5
if args.port < 0:
    args.port = random_free_tcp_port()
    while check_port_in_use(args.port+1):
        args.port = random_free_tcp_port()

train_port = args.port
eval_port = args.port + 1
server_executable = '../CARLA_0.9.9.4/CarlaUE4.sh'

print(check_port_in_use(train_port))
print(check_port_in_use(eval_port))

train_proc = sp.Popen(
    f"DISPLAY=4.  {server_executable} "
    f"-opengl -nosound -carla-world-port={train_port}",
    shell=True,
)
print(f"Server starting at port {train_port}")
# eval_proc = sp.Popen(
#     f"DISPLAY=  {server_executable} "
#     f"-opengl -nosound -carla-world-port={eval_port}",
#     shell=True,
# )
# print(f"Server starting at port {eval_port}")

# time.sleep(15)

print("Server process ID:", train_proc.pid)
# print("Server process ID:", eval_proc.pid)

while True:
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        break