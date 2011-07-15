//===-- SBValue.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBValue_h_
#define LLDB_SBValue_h_

#include "lldb/API/SBDefines.h"

#include <stdio.h>

namespace lldb {

#ifdef SWIG
%feature("docstring",
"Represents the value of a variable, a register, or an expression.

SBValue supports iteration through its child, which in turn is represented
as an SBValue.  For example, we can get the general purpose registers of a
frame as an SBValue, and iterate through all the registers,

    registerSet = frame.GetRegisters() # Returns an SBValueList.
    for regs in registerSet:
        if 'general purpose registers' in regs.getName().lower():
            GPRs = regs
            break

    print '%s (number of children = %d):' % (GPRs.GetName(), GPRs.GetNumChildren())
    for reg in GPRs:
        print 'Name: ', reg.GetName(), ' Value: ', reg.GetValue()

produces the output:

General Purpose Registers (number of children = 21):
Name:  rax  Value:  0x0000000100000c5c
Name:  rbx  Value:  0x0000000000000000
Name:  rcx  Value:  0x00007fff5fbffec0
Name:  rdx  Value:  0x00007fff5fbffeb8
Name:  rdi  Value:  0x0000000000000001
Name:  rsi  Value:  0x00007fff5fbffea8
Name:  rbp  Value:  0x00007fff5fbffe80
Name:  rsp  Value:  0x00007fff5fbffe60
Name:  r8  Value:  0x0000000008668682
Name:  r9  Value:  0x0000000000000000
Name:  r10  Value:  0x0000000000001200
Name:  r11  Value:  0x0000000000000206
Name:  r12  Value:  0x0000000000000000
Name:  r13  Value:  0x0000000000000000
Name:  r14  Value:  0x0000000000000000
Name:  r15  Value:  0x0000000000000000
Name:  rip  Value:  0x0000000100000dae
Name:  rflags  Value:  0x0000000000000206
Name:  cs  Value:  0x0000000000000027
Name:  fs  Value:  0x0000000000000010
Name:  gs  Value:  0x0000000000000048
"
         ) SBValue;
#endif
class SBValue
{
#ifdef SWIG
    %feature("autodoc", "1");
#endif
public:
    SBValue ();

    SBValue (const SBValue &rhs);

#ifndef SWIG
    const SBValue &
    operator =(const SBValue &rhs);
#endif

    ~SBValue ();

    bool
    IsValid() const;
    
    SBError
    GetError();

    lldb::user_id_t
    GetID ();
    
    const char *
    GetName();

    const char *
    GetTypeName ();

    size_t
    GetByteSize ();

    bool
    IsInScope (const lldb::SBFrame &frame);  // DEPRECATED - SBValues know their own frames.

    bool
    IsInScope ();

    lldb::Format
    GetFormat () const;
    
    void
    SetFormat (lldb::Format format);

    const char *
    GetValue (const lldb::SBFrame &frame);   // DEPRECATED - SBValues know their own frames.

    const char *
    GetValue ();

    ValueType
    GetValueType ();

    bool
    GetValueDidChange (const lldb::SBFrame &frame);  // DEPRECATED - SBValues know their own frames.

    bool
    GetValueDidChange ();

    const char *
    GetSummary (const lldb::SBFrame &frame);  // DEPRECATED - SBValues know their own frames.
    
    const char *
    GetSummary ();
    
    const char *
    GetObjectDescription (const lldb::SBFrame &frame);  // DEPRECATED - SBValues know their own frames.

    const char *
    GetObjectDescription ();

    const char *
    GetLocation (const lldb::SBFrame &frame);  // DEPRECATED - SBValues know their own frames.

    const char *
    GetLocation ();

    bool
    SetValueFromCString (const lldb::SBFrame &frame, const char *value_str);  // DEPRECATED - SBValues know their own frames.

    bool
    SetValueFromCString (const char *value_str);

    lldb::SBValue
    GetChildAtIndex (uint32_t idx);

    lldb::SBValue
    GetChildAtIndex (uint32_t idx, lldb::DynamicValueType use_dynamic);

    // Matches children of this object only and will match base classes and
    // member names if this is a clang typed object.
    uint32_t
    GetIndexOfChildWithName (const char *name);

    // Matches child members of this object and child members of any base
    // classes.
    lldb::SBValue
    GetChildMemberWithName (const char *name);

    // Matches child members of this object and child members of any base
    // classes.
    lldb::SBValue
    GetChildMemberWithName (const char *name, lldb::DynamicValueType use_dynamic);
    
    // Expands nested expressions like .a->b[0].c[1]->d
    lldb::SBValue
    GetValueForExpressionPath(const char* expr_path);

    uint32_t
    GetNumChildren ();

    void *
    GetOpaqueType();


    lldb::SBValue
    Dereference ();

    bool
    TypeIsPointerType ();

    bool
    GetDescription (lldb::SBStream &description);

    bool
    GetExpressionPath (lldb::SBStream &description);
    
    bool
    GetExpressionPath (lldb::SBStream &description, bool qualify_cxx_base_classes);

    SBValue (const lldb::ValueObjectSP &value_sp);
    
protected:
    friend class SBValueList;
    friend class SBFrame;

#ifndef SWIG

    // Mimic shared pointer...
    lldb_private::ValueObject *
    get() const;

    lldb_private::ValueObject *
    operator->() const;

    lldb::ValueObjectSP &
    operator*();

    const lldb::ValueObjectSP &
    operator*() const;

#endif

private:
    lldb::ValueObjectSP m_opaque_sp;
};

} // namespace lldb

#endif  // LLDB_SBValue_h_
