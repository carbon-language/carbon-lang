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

class SBValue
{
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

    const char *
    GetName();

    const char *
    GetTypeName ();

    size_t
    GetByteSize ();

    bool
    IsInScope (const lldb::SBFrame &frame);

    lldb::Format
    GetFormat () const;
    
    void
    SetFormat (lldb::Format format);

    const char *
    GetValue (const lldb::SBFrame &frame);

    ValueType
    GetValueType ();

    bool
    GetValueDidChange (const lldb::SBFrame &frame);

    const char *
    GetSummary (const lldb::SBFrame &frame);
    
    const char *
    GetObjectDescription (const lldb::SBFrame &frame);

    const char *
    GetLocation (const lldb::SBFrame &frame);

    bool
    SetValueFromCString (const lldb::SBFrame &frame, const char *value_str);

    lldb::SBValue
    GetChildAtIndex (uint32_t idx);

    // Matches children of this object only and will match base classes and
    // member names if this is a clang typed object.
    uint32_t
    GetIndexOfChildWithName (const char *name);

    // Matches child members of this object and child members of any base
    // classes.
    lldb::SBValue
    GetChildMemberWithName (const char *name);

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

protected:
    friend class SBValueList;
    friend class SBFrame;

    SBValue (const lldb::ValueObjectSP &value_sp);

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
