//===-- SBCompileUnit.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBCompileUnit.h"
#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBStream.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineEntry.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Core/Log.h"

using namespace lldb;
using namespace lldb_private;


SBCompileUnit::SBCompileUnit () :
    m_opaque_ptr (NULL)
{
}

SBCompileUnit::SBCompileUnit (lldb_private::CompileUnit *lldb_object_ptr) :
    m_opaque_ptr (lldb_object_ptr)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    
    if (log)
    {
        SBStream sstr;
        GetDescription (sstr);
        log->Printf ("SBCompileUnit::SBCompileUnit (lldb_private::CompileUnit *lldb_object_ptr=%p)"
                     " => this.obj = %p (%s)", lldb_object_ptr, m_opaque_ptr, sstr.GetData());
    }
}

SBCompileUnit::~SBCompileUnit ()
{
    m_opaque_ptr = NULL;
}

SBFileSpec
SBCompileUnit::GetFileSpec () const
{
    SBFileSpec file_spec;
    if (m_opaque_ptr)
        file_spec.SetFileSpec(*m_opaque_ptr);
    return file_spec;
}

uint32_t
SBCompileUnit::GetNumLineEntries () const
{
    if (m_opaque_ptr)
    {
        LineTable *line_table = m_opaque_ptr->GetLineTable ();
        if (line_table)
            return line_table->GetSize();
    }
    return 0;
}

SBLineEntry
SBCompileUnit::GetLineEntryAtIndex (uint32_t idx) const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBCompileUnit::GetLineEntryAtIndex (this.obj=%p, idx=%d)", m_opaque_ptr, idx);

    SBLineEntry sb_line_entry;
    if (m_opaque_ptr)
    {
        LineTable *line_table = m_opaque_ptr->GetLineTable ();
        if (line_table)
        {
            LineEntry line_entry;
            if (line_table->GetLineEntryAtIndex(idx, line_entry))
                sb_line_entry.SetLineEntry(line_entry);
        }
    }
    
    if (log)
    {
        SBStream sstr;
        sb_line_entry.GetDescription (sstr);
        log->Printf ("SBCompileUnit::GetLineEntryAtIndex (this.obj=%p, idx=%d) => SBLineEntry: '%s'", m_opaque_ptr, 
                     idx, sstr.GetData());
    }

    return sb_line_entry;
}

uint32_t
SBCompileUnit::FindLineEntryIndex (uint32_t start_idx, uint32_t line, SBFileSpec *inline_file_spec) const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //{
    //    SBStream sstr;
    //    inline_file_spec->GetDescription (sstr);
    //    log->Printf ("SBCompileUnit::FindLineEntryIndex (this.obj=%p, start_idx=%d, line=%d, inline_file_spec='%s')",
    //                 m_opaque_ptr, start_idx, line, sstr.GetData());
    //}

    if (m_opaque_ptr)
    {
        FileSpec file_spec;
        if (inline_file_spec && inline_file_spec->IsValid())
            file_spec = inline_file_spec->ref();
        else
            file_spec = *m_opaque_ptr;

        
        uint32_t ret_value = m_opaque_ptr->FindLineEntry (start_idx,
                                                          line,
                                                          inline_file_spec ? inline_file_spec->get() : NULL,
                                                          NULL);
        if (log)
        {
            SBStream sstr;
            inline_file_spec->GetDescription (sstr);
            log->Printf ("SBCompileUnit::FindLineEntryIndex(this.obj=%p, start_idx=%d, line=%d, inline_file_spec='%s')"
                         "=> '%d'", m_opaque_ptr, start_idx, line, sstr.GetData(), ret_value);
        }

        return ret_value;
    }

    if (log)
    {
        SBStream sstr;
        inline_file_spec->GetDescription (sstr);
        log->Printf ("SBCompileUnit::FindLineEntryIndex (this.obj=%p, start_idx=%d, line=%d, inline_file_spec='%s')"
                     " => '%d'", m_opaque_ptr, start_idx, line, sstr.GetData(), UINT32_MAX);
    }

    return UINT32_MAX;
}

bool
SBCompileUnit::IsValid () const
{
    return m_opaque_ptr != NULL;
}

bool
SBCompileUnit::operator == (const SBCompileUnit &rhs) const
{
    return m_opaque_ptr == rhs.m_opaque_ptr;
}

bool
SBCompileUnit::operator != (const SBCompileUnit &rhs) const
{
    return m_opaque_ptr != rhs.m_opaque_ptr;
}

const lldb_private::CompileUnit *
SBCompileUnit::operator->() const
{
    return m_opaque_ptr;
}

const lldb_private::CompileUnit &
SBCompileUnit::operator*() const
{
    return *m_opaque_ptr;
}

const lldb_private::CompileUnit *
SBCompileUnit::get () const
{
    return m_opaque_ptr;
}
    
bool
SBCompileUnit::GetDescription (SBStream &description)
{
    if (m_opaque_ptr)
    {
        description.ref();
        m_opaque_ptr->Dump (description.get(), false);
    }
    else
        description.Printf ("No Value");
    
    return true;
}
