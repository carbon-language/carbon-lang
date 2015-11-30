#!/usr/bin/env python

"""
Shared functionality used by `client` and `server` when dealing with
remote transmission
"""

# Future imports
from __future__ import absolute_import
from __future__ import print_function

# Python modules
import json
import logging
import os
import socket
import struct
import sys

# LLDB modules
import use_lldb_suite

def generate_config(languages):
    config = {"languages": languages}
    return json.dumps(config)

def parse_config(json_reader):
    json_data = json_reader.read()
    options_dict = json.loads(json_data)
    return options_dict
