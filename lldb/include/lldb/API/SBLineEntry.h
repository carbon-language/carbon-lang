//===-- SBLineEntry.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBLineEntry_h_
#define LLDB_SBLineEntry_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBFileSpec.h"

namespace lldb {

#ifdef SWIG
%feature("docstring",
         "Specifies an association with a contiguous range of instructions and"
         " a source file location. SBCompileUnit contains SBLineEntry(s)."
         ) SBLineEntry;
#endif
class SBLineEntry
{
#ifdef SWIG
    %feature("autodoc", "1");
#endif
public:

    SBLineEntry ();

    SBLineEntry (const lldb::SBLineEntry &rhs);

    ~SBLineEntry ();

#ifndef SWIG
    const lldb::SBLineEntry &
    operator = (const lldb::SBLineEntry &rhs);
#endif

    lldb::SBAddress
    GetStartAddress () const;

    lldb::SBAddress
    GetEndAddress () const;

    bool
    IsValid () const;

    lldb::SBFileSpec
    GetFileSpec () const;

    uint32_t
    GetLine () const;

    uint32_t
    GetColumn () const;

#ifndef SWIG
    bool
    operator == (const lldb::SBLineEntry &rhs) const;

    bool
    operator != (const lldb::SBLineEntry &rhs) const;

#endif

    bool
    GetDescription (lldb::SBStream &description);

protected:

    lldb_private::LineEntry *
    get ();
    
private:
    friend class SBCompileUnit;
    friend class SBFrame;
    friend class SBSymbolContext;

#ifndef SWIG

    const lldb_private::LineEntry *
    operator->() const;

    const lldb_private::LineEntry &
    operator*() const;

#endif


    SBLineEntry (const lldb_private::LineEntry *lldb_object_ptr);

    void
    SetLineEntry (const lldb_private::LineEntry &lldb_object_ref);

    std::auto_ptr<lldb_private::LineEntry> m_opaque_ap;
};


} // namespace lldb

#endif // LLDB_SBLineEntry_h_
