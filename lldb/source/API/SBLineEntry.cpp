//===-- SBLineEntry.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBStream.h"
#include "lldb/Symbol/LineEntry.h"
#include "lldb/Core/Log.h"

using namespace lldb;
using namespace lldb_private;


SBLineEntry::SBLineEntry () :
    m_opaque_ap ()
{
}

SBLineEntry::SBLineEntry (const SBLineEntry &rhs) :
    m_opaque_ap ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (rhs.IsValid())
    {
        m_opaque_ap.reset (new lldb_private::LineEntry (*rhs));
    }

    if (log)
        log->Printf ("SBLineEntry::SBLineEntry (rhs.ap=%p) => this.ap = %p ",
                     (rhs.IsValid() ? rhs.m_opaque_ap.get() : NULL), m_opaque_ap.get());

}



SBLineEntry::SBLineEntry (const lldb_private::LineEntry *lldb_object_ptr) :
    m_opaque_ap ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (lldb_object_ptr)
        m_opaque_ap.reset (new lldb_private::LineEntry(*lldb_object_ptr));

    if (log)
        log->Printf ("SBLineEntry::SBLineEntry (lldb_object_ptr=%p) => this.ap = %p", 
                     lldb_object_ptr, m_opaque_ap.get());
}

const SBLineEntry &
SBLineEntry::operator = (const SBLineEntry &rhs)
{
    if (this != &rhs)
    {
        if (rhs.IsValid())
            m_opaque_ap.reset (new lldb_private::LineEntry(*rhs));
    }
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBLineEntry::GetStartAddress ()");

    SBAddress sb_address;
    if (m_opaque_ap.get())
        sb_address.SetAddress(&m_opaque_ap->range.GetBaseAddress());

    if (log)
    {
        SBStream sstr;
        sb_address.GetDescription (sstr);
        log->Printf ("SBLineEntry::GetStartAddress (this.ap=%p) => SBAddress (this.ap = %p, (%s)", m_opaque_ap.get(),
                     sb_address.get(), sstr.GetData());
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBLineEntry::GetFileSpec ()");

    SBFileSpec sb_file_spec;
    if (m_opaque_ap.get() && m_opaque_ap->file)
        sb_file_spec.SetFileSpec(m_opaque_ap->file);

    if (log)
    {
        SBStream sstr;
        sb_file_spec.GetDescription (sstr);
        log->Printf ("SBLineEntry::GetFileSpec (this.ap=%p) => SBFileSpec : this.ap = %p, '%s'", m_opaque_ap.get(),
                     sb_file_spec.get(), sstr.GetData());
    }

    return sb_file_spec;
}

uint32_t
SBLineEntry::GetLine () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBLineEntry::GetLine ()");

    uint32_t line = 0;
    if (m_opaque_ap.get())
        line = m_opaque_ap->line;

    if (log)
        log->Printf ("SBLineEntry::GetLine (this.ap=%p) => %d", m_opaque_ap.get(), line);

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
        // Line entry:  File, line x {, column y}:  Addresses: <start_addr> - <end_addr>
        char file_path[PATH_MAX*2];
        m_opaque_ap->file.GetPath (file_path, sizeof (file_path));
        description.Printf ("Line entry: %s, line %d", file_path, GetLine());
        if (GetColumn() > 0)
            description.Printf (", column %d", GetColumn());
        description.Printf (":  Addresses:  0x%p - 0x%p", GetStartAddress().GetFileAddress() , 
                            GetEndAddress().GetFileAddress());
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
