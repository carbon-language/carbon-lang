//===-- SBLineEntry.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <limits.h>

#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Log.h"
#include "lldb/Symbol/LineEntry.h"

using namespace lldb;
using namespace lldb_private;


SBLineEntry::SBLineEntry () :
    m_opaque_ap ()
{
}

SBLineEntry::SBLineEntry (const SBLineEntry &rhs) :
    m_opaque_ap ()
{
    if (rhs.IsValid())
        m_opaque_ap.reset (new lldb_private::LineEntry (*rhs));
}



SBLineEntry::SBLineEntry (const lldb_private::LineEntry *lldb_object_ptr) :
    m_opaque_ap ()
{
    if (lldb_object_ptr)
        m_opaque_ap.reset (new lldb_private::LineEntry(*lldb_object_ptr));
}

const SBLineEntry &
SBLineEntry::operator = (const SBLineEntry &rhs)
{
    if (this != &rhs && rhs.IsValid())
        m_opaque_ap.reset (new lldb_private::LineEntry(*rhs));
    return *this;
}

void
SBLineEntry::SetLineEntry (const lldb_private::LineEntry &lldb_object_ref)
{
    if (m_opaque_ap.get())
        (*m_opaque_ap.get()) = lldb_object_ref;
    else
        m_opaque_ap.reset (new lldb_private::LineEntry (lldb_object_ref));
}


SBLineEntry::~SBLineEntry ()
{
}


SBAddress
SBLineEntry::GetStartAddress () const
{

    SBAddress sb_address;
    if (m_opaque_ap.get())
        sb_address.SetAddress(&m_opaque_ap->range.GetBaseAddress());

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        StreamString sstr;
        if (sb_address.get())
            sb_address->Dump (&sstr, NULL, Address::DumpStyleModuleWithFileAddress, Address::DumpStyleInvalid, 4);
        log->Printf ("SBLineEntry(%p)::GetStartAddress () => SBAddress (%p): %s", 
                     m_opaque_ap.get(), sb_address.get(), sstr.GetData());
    }

    return sb_address;
}

SBAddress
SBLineEntry::GetEndAddress () const
{
    SBAddress sb_address;
    if (m_opaque_ap.get())
    {
        sb_address.SetAddress(&m_opaque_ap->range.GetBaseAddress());
        sb_address.OffsetAddress(m_opaque_ap->range.GetByteSize());
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        StreamString sstr;
        if (sb_address.get())
            sb_address->Dump (&sstr, NULL, Address::DumpStyleModuleWithFileAddress, Address::DumpStyleInvalid, 4);
        log->Printf ("SBLineEntry(%p)::GetEndAddress () => SBAddress (%p): %s", 
                     m_opaque_ap.get(), sb_address.get(), sstr.GetData());
    }
    return sb_address;
}

bool
SBLineEntry::IsValid () const
{
    return m_opaque_ap.get() != NULL;
}


SBFileSpec
SBLineEntry::GetFileSpec () const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBFileSpec sb_file_spec;
    if (m_opaque_ap.get() && m_opaque_ap->file)
        sb_file_spec.SetFileSpec(m_opaque_ap->file);

    if (log)
    {
        SBStream sstr;
        sb_file_spec.GetDescription (sstr);
        log->Printf ("SBLineEntry(%p)::GetFileSpec () => SBFileSpec(%p): %s", m_opaque_ap.get(),
                     sb_file_spec.get(), sstr.GetData());
    }

    return sb_file_spec;
}

uint32_t
SBLineEntry::GetLine () const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    uint32_t line = 0;
    if (m_opaque_ap.get())
        line = m_opaque_ap->line;

    if (log)
        log->Printf ("SBLineEntry(%p)::GetLine () => %u", m_opaque_ap.get(), line);

    return line;
}


uint32_t
SBLineEntry::GetColumn () const
{
    if (m_opaque_ap.get())
        return m_opaque_ap->column;
    return 0;
}

bool
SBLineEntry::operator == (const SBLineEntry &rhs) const
{
    lldb_private::LineEntry *lhs_ptr = m_opaque_ap.get();
    lldb_private::LineEntry *rhs_ptr = rhs.m_opaque_ap.get();

    if (lhs_ptr && rhs_ptr)
        return lldb_private::LineEntry::Compare (*lhs_ptr, *rhs_ptr) == 0;

    return lhs_ptr == rhs_ptr;
}

bool
SBLineEntry::operator != (const SBLineEntry &rhs) const
{
    lldb_private::LineEntry *lhs_ptr = m_opaque_ap.get();
    lldb_private::LineEntry *rhs_ptr = rhs.m_opaque_ap.get();

    if (lhs_ptr && rhs_ptr)
        return lldb_private::LineEntry::Compare (*lhs_ptr, *rhs_ptr) != 0;

    return lhs_ptr != rhs_ptr;
}

const lldb_private::LineEntry *
SBLineEntry::operator->() const
{
    return m_opaque_ap.get();
}

const lldb_private::LineEntry &
SBLineEntry::operator*() const
{
    return *m_opaque_ap;
}

bool
SBLineEntry::GetDescription (SBStream &description)
{
    if (m_opaque_ap.get())
    {
        char file_path[PATH_MAX*2];
        m_opaque_ap->file.GetPath (file_path, sizeof (file_path));
        description.Printf ("%s:%u", file_path, GetLine());
        if (GetColumn() > 0)
            description.Printf (":%u", GetColumn());
    }
    else
        description.Printf ("No value");

    return true;
}

lldb_private::LineEntry *
SBLineEntry::get ()
{
    return m_opaque_ap.get();
}
