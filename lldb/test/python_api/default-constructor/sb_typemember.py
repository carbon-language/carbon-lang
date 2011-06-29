"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb

def fuzz_obj(obj):
    obj.IsBaseClass()
    obj.IsBitfield()
    obj.GetBitfieldWidth()
    obj.GetBitfieldOffset()
    obj.GetOffset()
    obj.GetName()
    obj.GetType()
    obj.GetParentType()
    obj.SetName("my_type_member_name")
    obj.Clear()
