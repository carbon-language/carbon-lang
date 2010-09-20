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
#include "lldb/API/SBSymbolContext.h"

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

    bool
    ResolveFileAddress (lldb::addr_t vm_addr, 
                        lldb::SBAddress& addr);

    lldb::SBSymbolContext
    ResolveSymbolContextForAddress (const lldb::SBAddress& addr, 
                                    uint32_t resolve_scope);

    bool
    GetDescription (lldb::SBStream &description);

    // The following function gets called by Python when a user tries to print
    // an object of this class.  It takes no arguments and returns a
    // PyObject * representing a char * (and it must be named "__repr__");

    PyObject *
    __repr__ ();

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

    lldb::ModuleSP m_opaque_sp;
};


} // namespace lldb

#endif // LLDB_SBModule_h_
