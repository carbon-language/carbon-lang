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

using namespace lldb;
using namespace lldb_private;


SBAddress::SBAddress () :
    m_opaque_ap ()
{
}

SBAddress::SBAddress (const lldb_private::Address *lldb_object_ptr) :
    m_opaque_ap ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (lldb_object_ptr)
        m_opaque_ap.reset (new lldb_private::Address(*lldb_object_ptr));

    if (log)
    {
        SBStream sstr;
        GetDescription (sstr);
        log->Printf ("SBAddress::SBAddress (lldb_object_ptr=%p) "
                     "=> this.ap = %p (%s)", lldb_object_ptr, m_opaque_ap.get(), sstr.GetData());
    }
}

SBAddress::SBAddress (const SBAddress &rhs) :
    m_opaque_ap ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (rhs.IsValid())
        m_opaque_ap.reset (new lldb_private::Address(*rhs.m_opaque_ap.get()));

    if (log)
        log->Printf ("SBAddress::SBAddress (rhs.m_opaque_ap = %p) => this.ap = %p",
                     (rhs.IsValid() ? rhs.m_opaque_ap.get() : NULL), m_opaque_ap.get());
}

SBAddress::~SBAddress ()
{
}

const SBAddress &
SBAddress::operator = (const SBAddress &rhs)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    
    if (this != &rhs)
    {
        if (rhs.IsValid())
            m_opaque_ap.reset (new lldb_private::Address(*rhs.m_opaque_ap.get()));
    }
    if (log)
        log->Printf ("SBAddress::operator= (rhs.ap = %p) => this.ap = %p", 
                     (rhs.IsValid() ? rhs.m_opaque_ap.get() : NULL), m_opaque_ap.get());

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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    
    //if (log)
    //    log->Printf ("SBAddress::GetLoadAddress");

    if (m_opaque_ap.get())
    {
        lldb::addr_t addr = m_opaque_ap->GetLoadAddress (target.get());
        if (log)
            log->Printf ("SBAddress::GetLoadAddress (target.sp=%p) => %p", target.get(), addr);
        return addr;
    }
    else
    {
        if (log)
            log->Printf ("SBAddress::GetLoadAddress (target.sp=%p) => LLDB_INVALID_ADDRESS", target.get());
        return LLDB_INVALID_ADDRESS;
    }
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
    description.ref();
    if (m_opaque_ap.get())
    {
        m_opaque_ap->DumpDebug (description.get());
    }
    else
        description.Printf ("No value");

    return true;
}
