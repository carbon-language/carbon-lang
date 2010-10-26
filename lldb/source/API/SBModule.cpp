//===-- SBModule.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBModule.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/STreamString.h"

using namespace lldb;
using namespace lldb_private;


SBModule::SBModule () :
    m_opaque_sp ()
{
}

SBModule::SBModule (const lldb::ModuleSP& module_sp) :
    m_opaque_sp (module_sp)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBModule::SBModule (module_sp=%p) => this.sp = %p", module_sp.get(), m_opaque_sp.get());
}

SBModule::~SBModule ()
{
}

bool
SBModule::IsValid () const
{
    return m_opaque_sp.get() != NULL;
}

SBFileSpec
SBModule::GetFileSpec () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBModule::GetFileSpec ()");

    SBFileSpec file_spec;
    if (m_opaque_sp)
        file_spec.SetFileSpec(m_opaque_sp->GetFileSpec());

    if (log)
    {
        SBStream sstr;
        file_spec.GetDescription (sstr);
        log->Printf ("SBModule::GetFileSpec (this.sp=%p) => SBFileSpec : this.ap = %p, 's'", m_opaque_sp.get(),
                     file_spec.get(), sstr.GetData());
    }

    return file_spec;
}

const uint8_t *
SBModule::GetUUIDBytes () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBModule::GetUUIDBytes ()");

    if (m_opaque_sp)
    {
        if (log)
        {
            StreamString sstr;
            m_opaque_sp->GetUUID().Dump (&sstr);
            log->Printf ("SBModule::GetUUIDBytes (this.sp=%p) => '%s'", m_opaque_sp.get(), sstr.GetData());
        }
        return (const uint8_t *)m_opaque_sp->GetUUID().GetBytes();
    }

    if (log)
        log->Printf ("SBModule::GetUUIDBytes (this.sp=%p) => NULL", m_opaque_sp.get());
    return NULL;
}


bool
SBModule::operator == (const SBModule &rhs) const
{
    if (m_opaque_sp)
        return m_opaque_sp.get() == rhs.m_opaque_sp.get();
    return false;
}

bool
SBModule::operator != (const SBModule &rhs) const
{
    if (m_opaque_sp)
        return m_opaque_sp.get() != rhs.m_opaque_sp.get();
    return false;
}

lldb::ModuleSP &
SBModule::operator *()
{
    return m_opaque_sp;
}

lldb_private::Module *
SBModule::operator ->()
{
    return m_opaque_sp.get();
}

const lldb_private::Module *
SBModule::operator ->() const
{
    return m_opaque_sp.get();
}

lldb_private::Module *
SBModule::get()
{
    return m_opaque_sp.get();
}

const lldb_private::Module *
SBModule::get() const
{
    return m_opaque_sp.get();
}


void
SBModule::SetModule (const lldb::ModuleSP& module_sp)
{
    m_opaque_sp = module_sp;
}


bool
SBModule::ResolveFileAddress (lldb::addr_t vm_addr, SBAddress& addr)
{
    if (m_opaque_sp)
        return m_opaque_sp->ResolveFileAddress (vm_addr, *addr);
    
    addr->Clear();
    return false;
}

SBSymbolContext
SBModule::ResolveSymbolContextForAddress (const SBAddress& addr, uint32_t resolve_scope)
{
    SBSymbolContext sb_sc;
    if (m_opaque_sp && addr.IsValid())
        m_opaque_sp->ResolveSymbolContextForAddress (*addr, resolve_scope, *sb_sc);
    return sb_sc;
}

bool
SBModule::GetDescription (SBStream &description)
{
    if (m_opaque_sp)
    {
        description.ref();
        m_opaque_sp->GetDescription (description.get());
    }
    else
        description.Printf ("No value");

    return true;
}
