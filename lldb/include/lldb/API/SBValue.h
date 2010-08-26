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

    ~SBValue ();

    bool
    IsValid() const;

    void
    Print (FILE *out_file, lldb::SBFrame *frame, bool print_type, bool print_value);

    const char *
    GetName();

    const char *
    GetTypeName ();

    size_t
    GetByteSize ();

    bool
    IsInScope (const lldb::SBFrame &frame);

    const char *
    GetValue (const lldb::SBFrame &frame);

    bool
    GetValueDidChange ();

    const char *
    GetSummary (const lldb::SBFrame &frame);

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

    bool
    ValueIsStale ();

    void *
    GetOpaqueType();

    //void
    //DumpType ();

    lldb::SBValue
    Dereference ();

    bool
    TypeIsPtrType ();


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
