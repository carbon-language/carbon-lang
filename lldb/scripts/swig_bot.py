#!/usr/bin/env python

# Python modules
import sys

# LLDB modules
import use_lldb_suite

if __name__ == "__main__":
    from swig_bot_lib import client
    client.run(sys.argv[1:])
