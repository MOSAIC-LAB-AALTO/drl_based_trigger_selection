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
        self.cpu = cpu
        self.ram = ram
        self.disk = disk

    def set_ethnicity(self, ethnicity):
        self.ethnicity = ethnicity

    def get_live_cpu(self):
        live_cpu = 0
        for i in range(self.cpu):
            live_cpu += round(random.uniform(1, 99), 2)
        return round(live_cpu/self.cpu, 2)

    def get_live_ram(self):
        live_ram = 0
        for i in range(self.ram):
            live_ram += round(random.uniform(1, 99), 2)
        return round(live_ram/self.ram, 2)
