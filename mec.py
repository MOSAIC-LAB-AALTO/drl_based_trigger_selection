#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:51:00 2019

@author: RaMy
"""


class MEC:
   
    def __init__(self, mec_name, cpu=0, ram=0, disk=0):
        self.mec_name = mec_name
        self.cpu_max = self.cpu = cpu
        self.ram_max = self.ram = ram
        self.disk_max = self.disk = disk
        self.list_of_c_vnfs = []

    def set_member(self, c_vnf):
        # print('The Current members are: {}'.format(self.list_of_c_vnfs))
        self.list_of_c_vnfs.append(c_vnf)
        # print('The Next members are: {}'.format(self.list_of_c_vnfs))

    def del_member(self, c_vnf):
        # print('The Current members are: {}'.format(self.list_of_c_vnfs))
        self.list_of_c_vnfs.remove(c_vnf)
        # print('The Next members are: {}'.format(self.list_of_c_vnfs))

    def get_member(self):
        return self.list_of_c_vnfs

    def cpu_availability(self, c_vnf_cpu):
        if c_vnf_cpu <= self.cpu:
            self.cpu -= c_vnf_cpu
            return True
        else:
            return False

    def ram_availability(self, c_vnf_ram):
        if c_vnf_ram <= self.ram:
            self.ram -= c_vnf_ram
            return True
        else:
            return False

    def disk_availability(self, c_vnf_disk):
        if c_vnf_disk <= self.disk:
            self.disk -= c_vnf_disk
            return True
        else:
            return False
