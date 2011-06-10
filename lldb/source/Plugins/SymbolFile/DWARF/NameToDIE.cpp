//===-- NameToDIE.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NameToDIE.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/RegularExpression.h"

void
NameToDIE::Insert (const lldb_private::ConstString& name, const Info &info)
{
    m_collection.insert (std::make_pair(name.AsCString(), info));
}

size_t
NameToDIE::Find (const lldb_private::ConstString &name, std::vector<Info> &info_array) const
{
    const char *name_cstr = name.AsCString();
    const size_t initial_info_array_size = info_array.size();
    collection::const_iterator pos, end = m_collection.end();
    for (pos = m_collection.lower_bound (name_cstr); pos != end && pos->first == name_cstr; ++pos)
    {
        info_array.push_back (pos->second);
    }
    return info_array.size() - initial_info_array_size;
}

size_t
NameToDIE::Find (const lldb_private::RegularExpression& regex, std::vector<Info> &info_array) const
{
    const size_t initial_info_array_size = info_array.size();
    collection::const_iterator pos, end = m_collection.end();
    for (pos = m_collection.begin(); pos != end; ++pos)
    {
        if (regex.Execute(pos->first))
            info_array.push_back (pos->second);
    }
    return info_array.size() - initial_info_array_size;
}

size_t
NameToDIE::FindAllEntriesForCompileUnitWithIndex (const uint32_t cu_idx, std::vector<Info> &info_array) const
{
    const size_t initial_info_array_size = info_array.size();
    collection::const_iterator pos, end = m_collection.end();
    for (pos = m_collection.begin(); pos != end; ++pos)
    {
        if (cu_idx == pos->second.cu_idx)
            info_array.push_back (pos->second);
    }
    return info_array.size() - initial_info_array_size;
}

void
NameToDIE::Dump (lldb_private::Stream *s)
{
    collection::const_iterator pos, end = m_collection.end();
    for (pos = m_collection.begin(); pos != end; ++pos)
    {
        s->Printf("%p: 0x%8.8x 0x%8.8x \"%s\"\n", pos->first, pos->second.cu_idx, pos->second.die_idx, pos->first);
    }
}
