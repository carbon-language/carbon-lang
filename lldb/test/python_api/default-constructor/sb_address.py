"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb

def fuzz_obj(obj):
    obj.GetFileAddress()
    obj.GetLoadAddress(lldb.SBTarget())
    obj.OffsetAddress(sys.maxint)
    obj.GetDescription(lldb.SBStream())
