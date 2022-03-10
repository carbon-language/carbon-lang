"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.GetSize()
    obj.GetInstructionAtIndex(0xffffffff)
    obj.AppendInstruction(lldb.SBInstruction())
    try:
        obj.Print(None)
    except Exception:
        pass
    obj.GetDescription(lldb.SBStream())
    obj.DumpEmulationForAllInstructions("armv7")
    obj.Clear()
    for inst in obj:
        s = str(inst)
