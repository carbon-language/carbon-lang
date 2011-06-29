"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb

def fuzz_obj(obj):
    obj.GetFileSpec()
    obj.GetPlatformFileSpec()
    obj.SetPlatformFileSpec(lldb.SBFileSpec())
    obj.GetUUIDBytes()
    obj.GetUUIDString()
    obj.ResolveFileAddress(sys.maxint, lldb.SBAddress())
    obj.ResolveSymbolContextForAddress(lldb.SBAddress(), 0)
    obj.GetDescription(lldb.SBStream())
    obj.GetNumSymbols()
    obj.GetSymbolAtIndex(sys.maxint)
    obj.FindFunctions("my_func", 0xffffffff, True, lldb.SBSymbolContextList())
    obj.FindGlobalVariables(lldb.SBTarget(), "my_global_var", 1)
