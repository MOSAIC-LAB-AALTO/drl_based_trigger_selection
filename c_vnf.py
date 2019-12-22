#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:51:00 2019

@author: RaMy
"""
import random


class VNF:
    def __init__(self, vnf_name, ethnicity="", cpu=0, ram=0, disk=0):
        self.vnf_name = vnf_name
        self.ethnicity = ethnicity
        self.cpu = self.initial_cpu = cpu
        self.ram = self.initial_ram = ram
        self.disk = disk
        self.initial_live_cpu = 0.0
        self.initial_live_ram = 0.0

    def set_ethnicity(self, ethnicity):
        # VNF ethnicity
        self.ethnicity = ethnicity

    def get_live_cpu(self):
        # method to randomly generate cpu load
        if self.initial_live_cpu == 0.0:
            live_cpu = 0
            for i in range(self.cpu):
                live_cpu += round(random.uniform(1, 99), 2)
            self.initial_live_cpu = round(live_cpu/self.cpu, 2)
            return self.initial_live_cpu
        else:
            self.initial_live_cpu = 100 * (self.initial_live_cpu * self.initial_cpu / (100 * self.cpu))
            self.initial_cpu = self.cpu
            return self.initial_live_cpu

    def get_live_ram(self):
        # method to randomly generate ram load
        if self.initial_live_ram == 0.0:
            live_ram = 0
            for i in range(self.ram):
                live_ram += round(random.uniform(1, 99), 2)
            self.initial_live_ram = round(live_ram/self.ram, 2)
            return self.initial_live_ram
        else:
            self.initial_live_ram = 100 * (self.initial_live_ram * self.initial_ram / (100 * self.ram))
            self.initial_ram = self.ram
            return self.initial_live_ram

    def get_sct_info(self):
        return self.initial_live_cpu, self.initial_live_ram
