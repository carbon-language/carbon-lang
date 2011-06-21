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

    SBModule (const SBModule &rhs);
    
#ifndef SWIG
    const SBModule &
    operator = (const SBModule &rhs);
#endif

    ~SBModule ();

    bool
    IsValid () const;

    lldb::SBFileSpec
    GetFileSpec () const;

    lldb::SBFileSpec
    GetPlatformFileSpec () const;

    bool
    SetPlatformFileSpec (const lldb::SBFileSpec &platform_file);

    const uint8_t *
    GetUUIDBytes () const;

    const char *
    GetUUIDString () const;

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

    size_t
    GetNumSymbols ();
    
    lldb::SBSymbol
    GetSymbolAtIndex (size_t idx);

    uint32_t
    FindFunctions (const char *name, 
                   uint32_t name_type_mask, // Logical OR one or more FunctionNameType enum bits
                   bool append, 
                   lldb::SBSymbolContextList& sc_list);

private:
    friend class SBAddress;
    friend class SBFrame;
    friend class SBSymbolContext;
    friend class SBTarget;

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
