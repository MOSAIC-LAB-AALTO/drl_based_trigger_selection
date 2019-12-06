from environment import ENV



nb_mec = 3
nb_vnfs = 2
# MECs
min_cpu = 50
max_cpu = 100
min_ram = 50
max_ram = 100
min_disk = 131072
max_disk = 524288
# Containers
min_c_cpu = 1
max_c_cpu = 4
min_c_ram = 1
max_c_ram = 4
min_c_disk = 512
max_c_disk = 4096

# DQN_agent
episodes = 1000                        # Total episodes for the training
batch_size = 32                        # Total used memory in memory replay mode
max_env_steps = 100                    # Max steps per episode
batch_update = 4
# Generate the MEC environment
env = ENV(nb_mec, nb_vnfs, min_cpu, max_cpu, min_ram, max_ram, min_disk, max_disk, min_c_cpu, max_c_cpu, min_c_ram,
          max_c_ram, min_c_disk, max_c_disk)

action_size = env.action_space
print('action_size: {}'.format(action_size))

state_size = env.observation_space
print('state_size: {}'.format(state_size))

env.view_infrastructure()
state = env.get_state()
print(state)
env.delete_mec()
env.delete_vnfs()

env.generate_mec()
env.generate_vnfs()
env.view_infrastructure()
state = env.get_state()
print(state)