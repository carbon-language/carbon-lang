//===-- SBValueList.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBValueList_h_
#define LLDB_SBValueList_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

#ifdef SWIG
%feature("docstring",
"Represents a collection of SBValues.  Both SBFrame's GetVariables() and
GetRegisters() return a SBValueList.

SBValueList supports SBValue iteration. For example (from test/lldbutil.py),

def get_registers(frame, kind):
    '''Returns the registers given the frame and the kind of registers desired.

    Returns None if there's no such kind.
    '''
    registerSet = frame.GetRegisters() # Return type of SBValueList.
    for value in registerSet:
        if kind.lower() in value.GetName().lower():
            return value

    return None

def get_GPRs(frame):
    '''Returns the general purpose registers of the frame as an SBValue.

    The returned SBValue object is iterable.  An example:
        ...
        from lldbutil import get_GPRs
        regs = get_GPRs(frame)
        for reg in regs:
            print '%s => %s' % (reg.GetName(), reg.GetValue())
        ...
    '''
    return get_registers(frame, 'general purpose')

def get_FPRs(frame):
    '''Returns the floating point registers of the frame as an SBValue.

    The returned SBValue object is iterable.  An example:
        ...
        from lldbutil import get_FPRs
        regs = get_FPRs(frame)
        for reg in regs:
            print '%s => %s' % (reg.GetName(), reg.GetValue())
        ...
    '''
    return get_registers(frame, 'floating point')

def get_ESRs(frame):
    '''Returns the exception state registers of the frame as an SBValue.

    The returned SBValue object is iterable.  An example:
        ...
        from lldbutil import get_ESRs
        regs = get_ESRs(frame)
        for reg in regs:
            print '%s => %s' % (reg.GetName(), reg.GetValue())
        ...
    '''
    return get_registers(frame, 'exception state')
"
         ) SBValueList;
#endif
class SBValueList
{
#ifdef SWIG
    %feature("autodoc", "1");
#endif
public:

    SBValueList ();

    SBValueList (const lldb::SBValueList &rhs);

    ~SBValueList();

    bool
    IsValid() const;

    void
    Append (const lldb::SBValue &val_obj);

    void
    Append (const lldb::SBValueList& value_list);

    uint32_t
    GetSize() const;

    lldb::SBValue
    GetValueAtIndex (uint32_t idx) const;

    lldb::SBValue
    FindValueObjectByUID (lldb::user_id_t uid);


#ifndef SWIG
    const lldb::SBValueList &
    operator = (const lldb::SBValueList &rhs);

    lldb_private::ValueObjectList *
    operator -> ();

    lldb_private::ValueObjectList &
    operator* ();

    const lldb_private::ValueObjectList *
    operator -> () const;

    const lldb_private::ValueObjectList &
    operator* () const;
    
    lldb_private::ValueObjectList *
    get ();

    lldb_private::ValueObjectList &
    ref ();

#endif

private:
    friend class SBFrame;

    SBValueList (const lldb_private::ValueObjectList *lldb_object_ptr);

    void
    Append (lldb::ValueObjectSP& val_obj_sp);

    void
    CreateIfNeeded ();

    std::auto_ptr<lldb_private::ValueObjectList> m_opaque_ap;
};


} // namespace lldb

#endif  // LLDB_SBValueList_h_
