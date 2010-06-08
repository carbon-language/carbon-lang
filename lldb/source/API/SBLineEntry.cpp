//===-- SBLineEntry.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBLineEntry.h"
#include "lldb/Symbol/LineEntry.h"

using namespace lldb;


SBLineEntry::SBLineEntry () :
    m_lldb_object_ap ()
{
}

SBLineEntry::SBLineEntry (const SBLineEntry &rhs) :
    m_lldb_object_ap ()
{
    if (rhs.IsValid())
    {
        m_lldb_object_ap.reset (new lldb_private::LineEntry (*rhs));
    }
}



SBLineEntry::SBLineEntry (const lldb_private::LineEntry *lldb_object_ptr) :
    m_lldb_object_ap ()
{
    if (lldb_object_ptr)
        m_lldb_object_ap.reset (new lldb_private::LineEntry(*lldb_object_ptr));
}

const SBLineEntry &
SBLineEntry::operator = (const SBLineEntry &rhs)
{
    if (this != &rhs)
    {
        if (rhs.IsValid())
            m_lldb_object_ap.reset (new lldb_private::LineEntry(*rhs));
    }
    return *this;
}

void
SBLineEntry::SetLineEntry (const lldb_private::LineEntry &lldb_object_ref)
{
    if (m_lldb_object_ap.get())
        (*m_lldb_object_ap.get()) = lldb_object_ref;
    else
        m_lldb_object_ap.reset (new lldb_private::LineEntry (lldb_object_ref));
}


SBLineEntry::~SBLineEntry ()
{
}


SBAddress
SBLineEntry::GetStartAddress () const
{
    SBAddress sb_address;
    if (m_lldb_object_ap.get())
        sb_address.SetAddress(&m_lldb_object_ap->range.GetBaseAddress());
    return sb_address;
}

SBAddress
SBLineEntry::GetEndAddress () const
{
    SBAddress sb_address;
    if (m_lldb_object_ap.get())
    {
        sb_address.SetAddress(&m_lldb_object_ap->range.GetBaseAddress());
        sb_address.OffsetAddress(m_lldb_object_ap->range.GetByteSize());
    }
    return sb_address;
}

bool
SBLineEntry::IsValid () const
{
    return m_lldb_object_ap.get() != NULL;
}


SBFileSpec
SBLineEntry::GetFileSpec () const
{
    SBFileSpec sb_file_spec;
    if (m_lldb_object_ap.get() && m_lldb_object_ap->file)
        sb_file_spec.SetFileSpec(m_lldb_object_ap->file);
    return sb_file_spec;
}

uint32_t
SBLineEntry::GetLine () const
{
    if (m_lldb_object_ap.get())
        return m_lldb_object_ap->line;
    return 0;
}


uint32_t
SBLineEntry::GetColumn () const
{
    if (m_lldb_object_ap.get())
        return m_lldb_object_ap->column;
    return 0;
}

bool
SBLineEntry::operator == (const SBLineEntry &rhs) const
{
    lldb_private::LineEntry *lhs_ptr = m_lldb_object_ap.get();
    lldb_private::LineEntry *rhs_ptr = rhs.m_lldb_object_ap.get();

    if (lhs_ptr && rhs_ptr)
        return lldb_private::LineEntry::Compare (*lhs_ptr, *rhs_ptr) == 0;

    return lhs_ptr == rhs_ptr;
}

bool
SBLineEntry::operator != (const SBLineEntry &rhs) const
{
    lldb_private::LineEntry *lhs_ptr = m_lldb_object_ap.get();
    lldb_private::LineEntry *rhs_ptr = rhs.m_lldb_object_ap.get();

    if (lhs_ptr && rhs_ptr)
        return lldb_private::LineEntry::Compare (*lhs_ptr, *rhs_ptr) != 0;

    return lhs_ptr != rhs_ptr;
}

const lldb_private::LineEntry *
SBLineEntry::operator->() const
{
    return m_lldb_object_ap.get();
}

const lldb_private::LineEntry &
SBLineEntry::operator*() const
{
    return *m_lldb_object_ap;
}





