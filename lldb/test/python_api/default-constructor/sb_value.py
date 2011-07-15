"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb

def fuzz_obj(obj):
    obj.GetError()
    obj.GetName()
    obj.GetTypeName()
    obj.GetByteSize()
    obj.IsInScope()
    obj.GetFormat()
    obj.SetFormat(lldb.eFormatBoolean)
    obj.GetValue()
    obj.GetValueType()
    obj.GetValueDidChange()
    obj.GetSummary()
    obj.GetObjectDescription()
    obj.GetLocation()
    obj.SetValueFromCString("my_new_value")
    obj.GetChildAtIndex(1)
    obj.GetChildAtIndex(2, lldb.eNoDynamicValues, False)
    obj.GetIndexOfChildWithName("my_first_child")
    obj.GetChildMemberWithName("my_first_child")
    obj.GetChildMemberWithName("my_first_child", lldb.eNoDynamicValues)
    obj.GetNumChildren()
    obj.GetOpaqueType()
    obj.Dereference()
    obj.TypeIsPointerType()
    stream = lldb.SBStream()
    obj.GetDescription(stream)
    obj.GetExpressionPath(stream)
    obj.GetExpressionPath(stream, True)
