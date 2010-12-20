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
SBAddress::operator*()
{
    if (m_opaque_ap.get() == NULL)
        m_opaque_ap.reset (new lldb_private::Address);
    return *m_opaque_ap;
}

const lldb_private::Address &
SBAddress::operator*() const
{
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
