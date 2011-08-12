//===-- SBAddress.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBAddress.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;


SBAddress::SBAddress () :
    m_opaque_ap ()
{
}

SBAddress::SBAddress (const lldb_private::Address *lldb_object_ptr) :
    m_opaque_ap ()
{
    if (lldb_object_ptr)
        m_opaque_ap.reset (new lldb_private::Address(*lldb_object_ptr));
}

SBAddress::SBAddress (const SBAddress &rhs) :
    m_opaque_ap ()
{
    if (rhs.IsValid())
        m_opaque_ap.reset (new lldb_private::Address(*rhs.m_opaque_ap.get()));
}

// Create an address by resolving a load address using the supplied target
SBAddress::SBAddress (lldb::addr_t load_addr, lldb::SBTarget &target) :
    m_opaque_ap()
{    
    SetLoadAddress (load_addr, target);
}



SBAddress::~SBAddress ()
{
}

const SBAddress &
SBAddress::operator = (const SBAddress &rhs)
{
    if (this != &rhs && rhs.IsValid())
        m_opaque_ap.reset (new lldb_private::Address(*rhs.m_opaque_ap.get()));
    return *this;
}

bool
SBAddress::IsValid () const
{
    return m_opaque_ap.get() != NULL && m_opaque_ap->IsValid();
}

void
SBAddress::Clear ()
{
    m_opaque_ap.reset();
}

void
SBAddress::SetAddress (const lldb_private::Address *lldb_object_ptr)
{
    if (lldb_object_ptr)
    {
        if (m_opaque_ap.get())
            *m_opaque_ap = *lldb_object_ptr;
        else
            m_opaque_ap.reset (new lldb_private::Address(*lldb_object_ptr));
        return;
    }
    if (m_opaque_ap.get())
        m_opaque_ap->Clear();
}

lldb::addr_t
SBAddress::GetFileAddress () const
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetFileAddress();
    else
        return LLDB_INVALID_ADDRESS;
}

lldb::addr_t
SBAddress::GetLoadAddress (const SBTarget &target) const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    lldb::addr_t addr = LLDB_INVALID_ADDRESS;
    if (m_opaque_ap.get())
    {
        Mutex::Locker api_locker (target->GetAPIMutex());
        addr = m_opaque_ap->GetLoadAddress (target.get());
    }
    
    if (log)
    {
        if (addr == LLDB_INVALID_ADDRESS)
            log->Printf ("SBAddress::GetLoadAddress (SBTarget(%p)) => LLDB_INVALID_ADDRESS", target.get());
        else
            log->Printf ("SBAddress::GetLoadAddress (SBTarget(%p)) => 0x%llx", target.get(), addr);
    }

    return addr;
}

void
SBAddress::SetLoadAddress (lldb::addr_t load_addr, lldb::SBTarget &target)
{
    // Create the address object if we don't already have one
    ref();
    if (target.IsValid())
        *this = target.ResolveLoadAddress(load_addr);
    else
        m_opaque_ap->Clear();

    // Check if we weren't were able to resolve a section offset address.
    // If we weren't it is ok, the load address might be a location on the
    // stack or heap, so we should just have an address with no section and
    // a valid offset
    if (!m_opaque_ap->IsValid())
        m_opaque_ap->SetOffset(load_addr);
}

bool
SBAddress::OffsetAddress (addr_t offset)
{
    if (m_opaque_ap.get())
    {
        addr_t addr_offset = m_opaque_ap->GetOffset();
        if (addr_offset != LLDB_INVALID_ADDRESS)
        {
            m_opaque_ap->SetOffset(addr_offset + offset);
            return true;
        }
    }
    return false;
}

lldb_private::Address *
SBAddress::operator->()
{
    return m_opaque_ap.get();
}

const lldb_private::Address *
SBAddress::operator->() const
{
    return m_opaque_ap.get();
}

lldb_private::Address &
SBAddress::ref ()
{
    if (m_opaque_ap.get() == NULL)
        m_opaque_ap.reset (new lldb_private::Address);
    return *m_opaque_ap;
}

const lldb_private::Address &
SBAddress::ref () const
{
    // "const SBAddress &addr" should already have checked "addr.IsValid()" 
    // prior to calling this function. In case you didn't we will assert
    // and die to let you know.
    assert (m_opaque_ap.get());
    return *m_opaque_ap;
}

lldb_private::Address *
SBAddress::get ()
{
    return m_opaque_ap.get();
}

bool
SBAddress::GetDescription (SBStream &description)
{
    // Call "ref()" on the stream to make sure it creates a backing stream in
    // case there isn't one already...
    description.ref();
    if (m_opaque_ap.get())
        m_opaque_ap->Dump (description.get(), NULL, Address::DumpStyleModuleWithFileAddress, Address::DumpStyleInvalid, 4);
    else
        description.Printf ("No value");

    return true;
}

SectionType
SBAddress::GetSectionType ()
{
    if (m_opaque_ap.get())
    {
        const Section *section = m_opaque_ap->GetSection();
        if (section)
            return section->GetType();
    }
    return eSectionTypeInvalid;
}


SBModule
SBAddress::GetModule ()
{
    SBModule sb_module;
    if (m_opaque_ap.get())
    {
        const Module *module = m_opaque_ap->GetModule();
        if (module)
            *sb_module = module->GetSP();
    }
    return sb_module;
}

SBSymbolContext
SBAddress::GetSymbolContext (uint32_t resolve_scope)
{
    SBSymbolContext sb_sc;
    if (m_opaque_ap.get())
        m_opaque_ap->CalculateSymbolContext (&sb_sc.ref(), resolve_scope);
    return sb_sc;
}

SBCompileUnit
SBAddress::GetCompileUnit ()
{
    SBCompileUnit sb_comp_unit;
    if (m_opaque_ap.get())
        sb_comp_unit.reset(m_opaque_ap->CalculateSymbolContextCompileUnit());
    return sb_comp_unit;
}

SBFunction
SBAddress::GetFunction ()
{
    SBFunction sb_function;
    if (m_opaque_ap.get())
        sb_function.reset(m_opaque_ap->CalculateSymbolContextFunction());
    return sb_function;
}

SBBlock
SBAddress::GetBlock ()
{
    SBBlock sb_block;
    if (m_opaque_ap.get())
        sb_block.reset(m_opaque_ap->CalculateSymbolContextBlock());
    return sb_block;
}

SBSymbol
SBAddress::GetSymbol ()
{
    SBSymbol sb_symbol;
    if (m_opaque_ap.get())
        sb_symbol.reset(m_opaque_ap->CalculateSymbolContextSymbol());
    return sb_symbol;
}

SBLineEntry
SBAddress::GetLineEntry ()
{
    SBLineEntry sb_line_entry;
    if (m_opaque_ap.get())
    {
        LineEntry line_entry;
        if (m_opaque_ap->CalculateSymbolContextLineEntry (line_entry))
            sb_line_entry.SetLineEntry (line_entry);
    }
    return sb_line_entry;
}


