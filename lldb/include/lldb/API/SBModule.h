//===-- SBModule.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBModule_h_
#define LLDB_SBModule_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBModule
{
public:

    SBModule ();

    ~SBModule ();

    bool
    IsValid () const;

    lldb::SBFileSpec
    GetFileSpec () const;

    const uint8_t *
    GetUUIDBytes () const;

#ifndef SWIG
    bool
    operator == (const lldb::SBModule &rhs) const;

    bool
    operator != (const lldb::SBModule &rhs) const;

#endif


private:
    friend class SBSymbolContext;
    friend class SBTarget;
    friend class SBFrame;

    explicit SBModule (const lldb::ModuleSP& module_sp);

    void
    SetModule (const lldb::ModuleSP& module_sp);
#ifndef SWIG

    lldb::ModuleSP &
    operator *();


    lldb_private::Module *
    operator ->();

    const lldb_private::Module *
    operator ->() const;

    lldb_private::Module *
    get();

    const lldb_private::Module *
    get() const;

#endif

    lldb::ModuleSP m_lldb_object_sp;
};


} // namespace lldb

#endif // LLDB_SBModule_h_
