#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:51:00 2019

@author: RaMy
"""
from random import randint, sample, randrange
from mec import MEC
from c_vnf import VNF
import pickle
import math


class ENV:

    def __init__(self, nb_mec=0, nb_vnfs=0, min_cpu=50, max_cpu=100, min_ram=50, max_ram=100, min_disk=4096,
                 max_disk=122819, min_c_cpu=1, max_c_cpu=4, min_c_ram=1, max_c_ram=4, min_c_disk=512,
                 max_c_disk=4096):

        self.nb_mec = nb_mec
        self.nb_vnfs = nb_vnfs
        # MECs
        self.min_cpu = min_cpu
        self.max_cpu = max_cpu
        self.min_ram = min_ram
        self.max_ram = max_ram
        self.min_disk = min_disk
        self.max_disk = max_disk
        # Containers
        self.min_c_cpu = min_c_cpu
        self.max_c_cpu = max_c_cpu
        self.min_c_ram = min_c_ram
        self.max_c_ram = max_c_ram
        self.min_c_disk = min_c_disk
        self.max_c_disk = max_c_disk

        self.mec = {}
        self.vnfs = {}
        self.initial_state = []

    def view_infrastructure(self):
        """
        :return: a view of the current configuration of the environment
        """

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        for i in range(self.nb_mec):
            print("*********************************--  MEC number {} --******************************".format(i))
            print('MEC name: {}'.format(self.mec[i].mec_name))
            print('MEC Members: {}'.format(self.mec[i].get_member()))
            print('MEC cpu: {}'.format(self.mec[i].cpu))
            print('MEC ram: {}'.format(self.mec[i].ram))
            print('MEC disk: {}'.format(self.mec[i].disk))
            print('#################################-- MEC {} Detailed Members: --#################################'
                  .format(i))
            for j in range(self.nb_vnfs):
                for k in range(len(self.mec[i].get_member())):
                    if self.vnfs[j].vnf_name == self.mec[i].get_member()[k]:
                        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^--  VNF number {} --^^^^^^^^^^^^^^^^^^^^^^^^^^^".format(j))
                        print('VNF name: {}'.format(self.vnfs[j].vnf_name))
                        print('VNF cpu: {}'.format(self.vnfs[j].cpu))
                        print('VNF ram: {}'.format(self.vnfs[j].ram))
                        print('VNF disk: {}'.format(self.vnfs[j].disk))
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    def view_infrastructure_(self, name, boll=True):
        """
        :return: a view of the current configuration of the environment
        """

        with open(name, 'a') as en:
            if boll:
                en.write('@@@@@@@@@@@@-- The Current environment --@@@@@@@@@@@@')
            else:
                en.write('@@@@@@@@@@@@-- The Next environment --@@@@@@@@@@@@')
            en.write('\n')
            en.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            en.write('\n')
            for i in range(self.nb_mec):
                en.write("*********************************--  MEC number {} --******************************".format(i))
                en.write('\n')
                en.write('MEC name: {}'.format(self.mec[i].mec_name))
                en.write('\n')
                en.write('MEC Members: {}'.format(self.mec[i].get_member()))
                en.write('\n')
                en.write('MEC cpu: {}'.format(self.mec[i].cpu))
                en.write('\n')
                en.write('MEC ram: {}'.format(self.mec[i].ram))
                en.write('\n')
                en.write('MEC disk: {}'.format(self.mec[i].disk))
                en.write('\n')
                en.write('#################################-- MEC {} Detailed Members: --#################################'
                      .format(i))
                en.write('\n')
                for j in range(self.nb_vnfs):
                    for k in range(len(self.mec[i].get_member())):
                        if self.vnfs[j].vnf_name == self.mec[i].get_member()[k]:
                            en.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^--  VNF number {} --^^^^^^^^^^^^^^^^^^^^^^^^^^^".format(j))
                            en.write('\n')
                            en.write('VNF name: {}'.format(self.vnfs[j].vnf_name))
                            en.write('\n')
                            en.write('VNF cpu: {}'.format(self.vnfs[j].cpu))
                            en.write('\n')
                            en.write('VNF ram: {}'.format(self.vnfs[j].ram))
                            en.write('\n')
                            en.write('VNF disk: {}'.format(self.vnfs[j].disk))
                            en.write('\n')
            en.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            en.write('\n')

    def get_mec(self):
        """
        :return: a list of MECs in the network
        """
        return self.mec

    def generate_mec(self):
        """
        :return: generate MECs in the environment
        """

        for i in range(self.nb_mec):
            self.mec[i] = MEC(i, randint(self.min_cpu, self.max_cpu), randint(self.min_ram, self.max_ram),
                              randint(self.min_disk, self.max_disk))

    def generate_vnfs(self):
        """
        :return: generate VNFs in the environment
        """
        i = 0
        while i < self.nb_vnfs:
            for mec in sample(list(self.mec), 1):
                cpu = randint(self.min_c_cpu, self.max_c_cpu)
                ram = randint(self.min_c_ram, self.max_c_ram)
                disk = randint(self.min_c_disk, self.max_c_disk)
                if self.mec[mec].cpu_availability(cpu) and self.mec[mec].ram_availability(ram) and \
                        self.mec[mec].disk_availability(disk):
                    self.vnfs[i] = VNF(vnf_name=i, ethnicity=str(mec), cpu=cpu, ram=ram, disk=disk)
                    self.mec[mec].set_member(i)
                    i += 1

    def get_rat(self, mec_id):
        """
        :param mec_id:
        :return: data related to the RAT trigger
        """
        cpu_percentage = round((self.mec[mec_id].cpu_max - self.mec[mec_id].cpu) * 100 / self.mec[mec_id].cpu_max, 2)
        ram_percentage = round((self.mec[mec_id].ram_max - self.mec[mec_id].ram) * 100 / self.mec[mec_id].ram_max, 2)
        disk_percentage = round((self.mec[mec_id].disk_max - self.mec[mec_id].disk) * 100 / self.mec[mec_id].disk_max, 2)
        return cpu_percentage, ram_percentage, disk_percentage

    def get_sct(self, vnf_id):
        """
        :param vnf_id:
        :return: data related to the SCT trigger
        """
        return self.vnfs[vnf_id].get_live_cpu(), self.vnfs[vnf_id].get_live_ram()

    def get_state(self, initial_state=False):
        """
        :return: a given state of the environment in a given time-step 't'
        """
        state = []
        for i in range(self.nb_mec):
            mec_cpu_percentage, mec_ram_percentage, mec_disk_percentage = self.get_rat(i)
            state.extend([mec_cpu_percentage, mec_ram_percentage, mec_disk_percentage])
            for j in range(self.nb_vnfs):
                for k in range(len(self.mec[i].get_member())):
                    if self.vnfs[j].vnf_name == self.mec[i].get_member()[k]:
                        vnf_cpu_percentage, vnf_ram_percentage = self.get_sct(self.vnfs[j].vnf_name)
                        state.extend([vnf_cpu_percentage, vnf_ram_percentage])
        if initial_state:
            if not self.initial_state:
                self.initial_state = state
            else:
                return self.initial_state
        return state

    def migrate(self, vnf_id, mec_dest_id):
        """
        :param vnf_id:
        :param mec_dest_id:
        :return: migrate a given container from one MEC to another one, True if migrated otherwise False
        """
        # Control time
        failure_time = (self.max_c_disk / 1000) + 100
        staying_time = (self.min_c_disk / 4) / 1000
        with open('environment.txt', 'a') as en:
            en.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            en.write('\n')
            en.write('I\'m the migrate function I received an order to migrate VNF {} belonging to MEC {} to the MEC number {}'.
              format(vnf_id, self.vnfs[vnf_id].ethnicity, mec_dest_id))
            en.write('\n')
            en.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            en.write('\n')
            if int(self.vnfs[vnf_id].ethnicity) == mec_dest_id:
                en.write("Container cannot be migrated to the same host !!! == Do Nothing")
                en.write('\n')
                en.write('time for same-host: {}'.format(staying_time))
                en.write('\n')
                return False, (1 / staying_time)
            if self.mec[mec_dest_id].cpu_availability(self.vnfs[vnf_id].cpu) and \
                    self.mec[mec_dest_id].ram_availability(self.vnfs[vnf_id].ram) and \
                    self.mec[mec_dest_id].disk_availability(self.vnfs[vnf_id].disk):
                # Remove the container's details from the source MEC
                self.mec[int(self.vnfs[vnf_id].ethnicity)].del_member(vnf_id)
                self.mec[int(self.vnfs[vnf_id].ethnicity)].cpu += self.vnfs[vnf_id].cpu
                self.mec[int(self.vnfs[vnf_id].ethnicity)].ram += self.vnfs[vnf_id].ram
                self.mec[int(self.vnfs[vnf_id].ethnicity)].disk += self.vnfs[vnf_id].disk

                # Addition of the container's details to the destination MEC
                self.mec[mec_dest_id].set_member(vnf_id)

                # Updating VNF ethnicity
                self.vnfs[vnf_id].set_ethnicity(mec_dest_id)

                # Migration time based on the disk size
                migration_time = self.vnfs[vnf_id].disk / 1000
                # time.sleep(migration_time)
                en.write('migration time: {}'.format(migration_time))
                en.write('\n')
                return True, (1 / migration_time)

            # Roll-back procedure in case of migration's failure
            # Control time
            en.write('roll-back (failure) time: {}'.format(failure_time))
            en.write('\n')
            return False, (1 / failure_time)

    def scale_up(self, vnf_id, resource_type):
        """
        :param vnf_id:
        :param resource_type:
        :return: A Boolean giving the status of the scale up operation
        """
        scale_up_time = (self.min_c_disk / 2) / 1000
        failure_time = (self.max_c_disk / 1000) + 100
        with open('environment.txt', 'a') as en:
            en.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            en.write('\n')
            en.write('I\'m the scale UP function I received an order to scale UP {} resources for the VNF {}'.
              format(resource_type, vnf_id))
            en.write('\n')
            en.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            en.write('\n')
        if resource_type == "CPU":
            cpu_resource_unit = randint(self.min_c_cpu, self.max_c_cpu)
            if self.mec[int(self.vnfs[vnf_id].ethnicity)].cpu_availability(cpu_resource_unit):
                self.vnfs[vnf_id].cpu += cpu_resource_unit
                return True, (1 / scale_up_time)
        elif resource_type == "RAM":
            ram_resource_unit = randint(self.min_c_ram, self.max_c_ram)
            if self.mec[int(self.vnfs[vnf_id].ethnicity)].ram_availability(ram_resource_unit):
                self.vnfs[vnf_id].ram += ram_resource_unit
                return True, (1 / scale_up_time)
        elif resource_type == "DISK":
            disk_resource_unit = randint(self.min_c_disk, self.max_c_disk)
            if self.mec[int(self.vnfs[vnf_id].ethnicity)].disk_availability(disk_resource_unit):
                self.vnfs[vnf_id].disk += disk_resource_unit
                return True, (1 / scale_up_time)
        return False, (1 / failure_time)

    def scale_down(self, vnf_id, resource_type):
        """
        :param vnf_id:
        :param resource_type:
        :return: A Boolean giving the status of the scale down operation
        """
        scale_down_time = ((self.min_c_disk / 2) + 128) / 1000
        failure_time = (self.max_c_disk / 1000) + 100
        with open('environment.txt', 'a') as en:
            en.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            en.write('\n')
            en.write('I\'m the scale Down function I received an order to scale Down {} resources for the VNF {}'.
                     format(resource_type, vnf_id))
            en.write('\n')
            en.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            en.write('\n')

        if resource_type == "CPU":
            if self.vnfs[vnf_id].cpu == 1:
                # print("Container with 1 core CPU cannot scale down !!!")
                return False, (1 / failure_time)
            cpu_resource_unit = randint(self.min_c_cpu, self.max_c_cpu/2)
            if self.vnfs[vnf_id].cpu - cpu_resource_unit >= 1:
                self.mec[int(self.vnfs[vnf_id].ethnicity)].cpu += cpu_resource_unit
                self.vnfs[vnf_id].cpu -= cpu_resource_unit
                return True, (1 / scale_down_time)
        elif resource_type == "RAM":
            if self.vnfs[vnf_id].ram == 1:
                # print("Container with 1 GB of Memory cannot scale down !!!")
                return False, (1 / failure_time)
            ram_resource_unit = randint(self.min_c_ram, self.max_c_ram/2)
            if self.vnfs[vnf_id].ram - ram_resource_unit >= 1:
                self.mec[int(self.vnfs[vnf_id].ethnicity)].ram += ram_resource_unit
                self.vnfs[vnf_id].ram -= ram_resource_unit
                return True, (1 / scale_down_time)
        elif resource_type == "DISK":
            if self.vnfs[vnf_id].disk == 512:
                # print("Container with 512 MB of Disk cannot scale down !!!")
                return False, (1 / failure_time)
            disk_resource_unit = randrange(self.min_c_disk/4, self.max_c_disk, self.min_c_disk/4)
            if self.vnfs[vnf_id].disk - disk_resource_unit >= 512:
                self.mec[int(self.vnfs[vnf_id].ethnicity)].disk += disk_resource_unit
                self.vnfs[vnf_id].disk -= disk_resource_unit
                return True, (1 / scale_down_time)
        return False, (1 / failure_time)

    def reward(self, action_time):
        """
        :param action_time:
        :return: a reward for a given action in a given time-step at a given state
        """
        resource_usage = 0
        for i in range(self.nb_mec):
            if len(self.mec[i].get_member()) != 0:
                mec_cpu_percentage, mec_ram_percentage, mec_disk_percentage = self.get_rat(i)
                resource_usage += (1 / mec_cpu_percentage) + (1 / mec_ram_percentage) + (1 / mec_disk_percentage)
        # Normalization
        action_time = action_time / ((self.max_c_disk / 1000) + 100)
        resource_usage = resource_usage / 100
        # TODO: we can add a coefficient to promote one time over the usage or the opposite.
        alpha = 1
        beta = 1
        return alpha * action_time + beta * resource_usage

    def action(self):
        """
        :return: Select a random action value
        """
        return randint(1, (len(self.vnfs) * (len(self.mec) + 2 * 3)))

    def step(self, action):
        """
        :param action:
        :return: a step based on a received action to modify the environment, new state and a reward will be observed
        as well
        """
        # Populating the set of actions to be used later as a manual
        set_of_actions = {}
        i = 1
        while i <= len(self.mec):
            set_of_actions[i] = self.mec[i-1].mec_name
            i += 1
        set_of_actions[i], set_of_actions[i+3] = "CPU", "CPU"
        set_of_actions[i+1], set_of_actions[i+4] = "RAM", "RAM"
        set_of_actions[i+2], set_of_actions[i+5] = "DISK", "DISK"

        # Gathering vnf_id and type of the taken action
        vnf_id = math.ceil(action / (len(self.mec) + 2 * 3)) - 1
        action = (len(self.mec) + 2 * 3) * (1 - (vnf_id + 1)) + action
        if action <= len(self.mec):
            # Migrating or doing nothing is the source mec_id equal destination mec_id
            operation_success, action_time = self.migrate(vnf_id, set_of_actions[action])
        elif len(self.mec) < action <= len(self.mec) + 3:
            # Scaling Up
            operation_success, action_time = self.scale_up(vnf_id, set_of_actions[action])
        else:
            # Scaling Down print("scale down")
            operation_success, action_time = self.scale_down(vnf_id, set_of_actions[action])
        return self.get_state(), self.reward(action_time), False, False

    def save_topology(self, file_name):
        """
        :param file_name:
        :return: implemented in order to save the current object that contain all the required data (save the topology)
        """

        my_data = [self.nb_mec, self.nb_vnfs, self.min_cpu, self.max_cpu, self.min_ram, self.max_ram, self.min_disk,
                   self.max_disk, self.min_c_cpu, self.max_c_cpu, self.min_c_ram, self.max_c_ram, self.min_c_disk,
                   self.max_c_disk, self.mec, self.vnfs]

        with open(file_name + '.dat', 'wb') as fp:
            pickle.dump(my_data, fp)

    def restore_topology(self, file_name):
        """
        :param file_name:
        :return: the saved data from the previous created topology
        """
        with open(file_name + '.dat', 'rb') as fp:
            my_data = pickle.load(fp)

        self.nb_mec = my_data[0]
        self.nb_vnfs = my_data[1]
        self.min_cpu = my_data[2]
        self.max_cpu = my_data[3]
        self.min_ram = my_data[4]
        self.max_ram = my_data[5]
        self.min_disk = my_data[6]
        self.max_disk = my_data[7]
        self.min_c_cpu = my_data[8]
        self.max_c_cpu = my_data[9]
        self.min_c_ram = my_data[10]
        self.max_c_ram = my_data[11]
        self.min_c_disk = my_data[12]
        self.max_c_disk = my_data[13]
        self.mec = my_data[14]
        self.vnfs = my_data[15]


