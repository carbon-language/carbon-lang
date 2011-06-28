"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb

def fuzz_obj(obj):
    obj.Initialize(lldb.SBDebugger.Create(), None, None, 0, "$", "^", True)
    obj.IsActive()
    obj.IsDone()
    obj.SetIsDone(True)
    obj.GetGranularity()
