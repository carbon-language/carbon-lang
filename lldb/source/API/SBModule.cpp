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
#include "lldb/API/SBStream.h"
#include "lldb/API/SBSymbolContextList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"

using namespace lldb;
using namespace lldb_private;


SBModule::SBModule () :
    m_opaque_sp ()
{
}

SBModule::SBModule (const lldb::ModuleSP& module_sp) :
    m_opaque_sp (module_sp)
{
}

SBModule::SBModule(const SBModule &rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

const SBModule &
SBModule::operator = (const SBModule &rhs)
{
    if (this != &rhs)
        m_opaque_sp = rhs.m_opaque_sp;
    return *this;
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
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBFileSpec file_spec;
    if (m_opaque_sp)
        file_spec.SetFileSpec(m_opaque_sp->GetFileSpec());

    if (log)
    {
        log->Printf ("SBModule(%p)::GetFileSpec () => SBFileSpec(%p)", 
        m_opaque_sp.get(), file_spec.get());
    }

    return file_spec;
}

lldb::SBFileSpec
SBModule::GetPlatformFileSpec () const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    SBFileSpec file_spec;
    if (m_opaque_sp)
        file_spec.SetFileSpec(m_opaque_sp->GetPlatformFileSpec());
    
    if (log)
    {
        log->Printf ("SBModule(%p)::GetPlatformFileSpec () => SBFileSpec(%p)", 
                     m_opaque_sp.get(), file_spec.get());
    }
    
    return file_spec;
    
}

bool
SBModule::SetPlatformFileSpec (const lldb::SBFileSpec &platform_file)
{
    bool result = false;
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    if (m_opaque_sp)
    {
        m_opaque_sp->SetPlatformFileSpec(*platform_file);
        result = true;
    }
    
    if (log)
    {
        log->Printf ("SBModule(%p)::SetPlatformFileSpec (SBFileSpec(%p (%s%s%s)) => %i", 
                     m_opaque_sp.get(), 
                     platform_file.get(),
                     platform_file->GetDirectory().GetCString(),
                     platform_file->GetDirectory() ? "/" : "",
                     platform_file->GetFilename().GetCString(),
                     result);
    }
    return result;
}



const uint8_t *
SBModule::GetUUIDBytes () const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    const uint8_t *uuid_bytes = NULL;
    if (m_opaque_sp)
        uuid_bytes = (const uint8_t *)m_opaque_sp->GetUUID().GetBytes();

    if (log)
    {
        if (uuid_bytes)
        {
            StreamString s;
            m_opaque_sp->GetUUID().Dump (&s);
            log->Printf ("SBModule(%p)::GetUUIDBytes () => %s", m_opaque_sp.get(), s.GetData());
        }
        else
            log->Printf ("SBModule(%p)::GetUUIDBytes () => NULL", m_opaque_sp.get());
    }
    return uuid_bytes;
}


const char *
SBModule::GetUUIDString () const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    static char uuid_string[80];
    const char * uuid_c_string = NULL;
    if (m_opaque_sp)
        uuid_c_string = (const char *)m_opaque_sp->GetUUID().GetAsCString(uuid_string, sizeof(uuid_string));

    if (log)
    {
        if (uuid_c_string)
        {
            StreamString s;
            m_opaque_sp->GetUUID().Dump (&s);
            log->Printf ("SBModule(%p)::GetUUIDString () => %s", m_opaque_sp.get(), s.GetData());
        }
        else
            log->Printf ("SBModule(%p)::GetUUIDString () => NULL", m_opaque_sp.get());
    }
    return uuid_c_string;
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

size_t
SBModule::GetNumSymbols ()
{
    if (m_opaque_sp)
    {
        ObjectFile *obj_file = m_opaque_sp->GetObjectFile();
        if (obj_file)
        {
            Symtab *symtab = obj_file->GetSymtab();
            if (symtab)
                return symtab->GetNumSymbols();
        }
    }
    return 0;
}

SBSymbol
SBModule::GetSymbolAtIndex (size_t idx)
{
    SBSymbol sb_symbol;
    if (m_opaque_sp)
    {
        ObjectFile *obj_file = m_opaque_sp->GetObjectFile();
        if (obj_file)
        {
            Symtab *symtab = obj_file->GetSymtab();
            if (symtab)
                sb_symbol.SetSymbol(symtab->SymbolAtIndex (idx));
        }
    }
    return sb_symbol;
}

uint32_t
SBModule::FindFunctions (const char *name, 
                         uint32_t name_type_mask, 
                         bool append, 
                         lldb::SBSymbolContextList& sc_list)
{
    if (!append)
        sc_list.Clear();
    if (m_opaque_sp)
    {
        const bool symbols_ok = true;
        return m_opaque_sp->FindFunctions (ConstString(name), 
                                           name_type_mask, 
                                           symbols_ok, 
                                           append, 
                                           *sc_list);
    }
    return 0;
}

