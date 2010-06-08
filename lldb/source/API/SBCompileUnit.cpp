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
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineEntry.h"
#include "lldb/Symbol/LineTable.h"

using namespace lldb;
using namespace lldb_private;


SBCompileUnit::SBCompileUnit () :
    m_lldb_object_ptr (NULL)
{
}

SBCompileUnit::SBCompileUnit (lldb_private::CompileUnit *lldb_object_ptr) :
    m_lldb_object_ptr (lldb_object_ptr)
{
}

SBCompileUnit::~SBCompileUnit ()
{
    m_lldb_object_ptr = NULL;
}

SBFileSpec
SBCompileUnit::GetFileSpec () const
{
    SBFileSpec file_spec;
    if (m_lldb_object_ptr)
        file_spec.SetFileSpec(*m_lldb_object_ptr);
    return file_spec;
}

uint32_t
SBCompileUnit::GetNumLineEntries () const
{
    if (m_lldb_object_ptr)
    {
        LineTable *line_table = m_lldb_object_ptr->GetLineTable ();
        if (line_table)
            return line_table->GetSize();
    }
    return 0;
}

SBLineEntry
SBCompileUnit::GetLineEntryAtIndex (uint32_t idx) const
{
    SBLineEntry sb_line_entry;
    if (m_lldb_object_ptr)
    {
        LineTable *line_table = m_lldb_object_ptr->GetLineTable ();
        if (line_table)
        {
            LineEntry line_entry;
            if (line_table->GetLineEntryAtIndex(idx, line_entry))
                sb_line_entry.SetLineEntry(line_entry);
        }
    }
    return sb_line_entry;
}

uint32_t
SBCompileUnit::FindLineEntryIndex (uint32_t start_idx, uint32_t line, SBFileSpec *inline_file_spec) const
{
    if (m_lldb_object_ptr)
    {
        FileSpec file_spec;
        if (inline_file_spec && inline_file_spec->IsValid())
            file_spec = inline_file_spec->ref();
        else
            file_spec = *m_lldb_object_ptr;

        return m_lldb_object_ptr->FindLineEntry (start_idx,
                                                 line,
                                                 inline_file_spec ? inline_file_spec->get() : NULL,
                                                 NULL);
    }
    return UINT32_MAX;
}

bool
SBCompileUnit::IsValid () const
{
    return m_lldb_object_ptr != NULL;
}

bool
SBCompileUnit::operator == (const SBCompileUnit &rhs) const
{
    return m_lldb_object_ptr == rhs.m_lldb_object_ptr;
}

bool
SBCompileUnit::operator != (const SBCompileUnit &rhs) const
{
    return m_lldb_object_ptr != rhs.m_lldb_object_ptr;
}

const lldb_private::CompileUnit *
SBCompileUnit::operator->() const
{
    return m_lldb_object_ptr;
}

const lldb_private::CompileUnit &
SBCompileUnit::operator*() const
{
    return *m_lldb_object_ptr;
}
