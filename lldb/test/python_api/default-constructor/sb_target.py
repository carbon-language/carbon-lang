"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb

def fuzz_obj(obj):
    obj.GetProcess()
    listener = lldb.SBListener()
    error = lldb.SBError()
    obj.Launch(listener, None, None, None, None, None, None, 0, True, error)
    obj.LaunchSimple(None, None, None)
    obj.AttachToProcessWithID(listener, 123, error)
    obj.AttachToProcessWithName(listener, 'lldb', False, error)
    obj.ConnectRemote(listener, "connect://to/here", None, error)
    obj.GetExecutable()
    obj.GetNumModules()
    obj.GetModuleAtIndex(0xffffffff)
    obj.GetDebugger()
    filespec = lldb.SBFileSpec()
    obj.FindModule(filespec)
    contextlist = lldb.SBSymbolContextList()
    obj.FindFunctions("the_func", 0xff, True, contextlist)
    address = lldb.SBAddress()
    obj.ResolveLoadAddress(0xffff, address)
    obj.ResolveSymbolContextForAddress(address, 0)
    obj.BreakpointCreateByLocation("filename", 20)
    obj.BreakpointCreateByLocation(filespec, 20)
    obj.BreakpointCreateByName("func", None)
    obj.BreakpointCreateByRegex("func.", None)
    obj.BreakpointCreateByAddress(0xf0f0)
    obj.GetNumBreakpoints()
    obj.GetBreakpointAtIndex(0)
    obj.BreakpointDelete(0)
    obj.FindBreakpointByID(0)
    obj.EnableAllBreakpoints()
    obj.DisableAllBreakpoints()
    obj.DeleteAllBreakpoints()
    obj.GetBroadcaster()
    obj.GetDescription(lldb.SBStream(), lldb.eDescriptionLevelBrief)
    obj.Clear()
