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


static uint32_t
dl_new_hash (const char *s)
{
    uint32_t h = 5381;
    
    for (unsigned char c = *s; c; c = *++s)
        h = ((h << 5) + h) + c;
    
    return h;
}

struct HashEntry
{
    uint32_t hash;
    uint32_t cu_idx;
    uint32_t die_idx;
    const char *name;
};

typedef struct HashEntry HashEntryType;

void
NameToDIE::Hash (lldb_private::Stream *s)
{
    typedef std::vector<HashEntryType> hash_collection;
    hash_collection hash_entries;
    collection::const_iterator pos, end = m_collection.end();
    for (pos = m_collection.begin(); pos != end; ++pos)
    {
        HashEntry entry = { dl_new_hash (pos->first), pos->second.cu_idx, pos->second.die_idx, pos->first };
        hash_entries.push_back (entry); 
    }
    

    const uint32_t hash_entries_size = hash_entries.size();
    
    uint32_t i;

    for (uint32_t power_2 = 0x10; power_2 <= hash_entries_size; power_2 <<= 1)
    {
        const uint32_t size = power_2 - 1;
        if (size > 0x10 && size > hash_entries_size)
            break;

        s->Printf ("\nTrying size of %u for %u items:\n", size, hash_entries_size);
        std::vector<uint32_t> indexes(size, 0);
        for (i=0; i<hash_entries_size; ++i)
        {
            indexes[hash_entries[i].hash % size]++;
        }
        const uint32_t indexes_size = indexes.size();
        uint32_t empties = 0;
        uint32_t good = 0;
        uint32_t collisions = 0;
        uint64_t total = 0;
        for (i=0; i<indexes_size; ++i)
        {
            uint32_t c = indexes[i];
            total += c;
            if (c == 0)
                ++empties;
            else if (c == 1)
                ++good;
            else
                ++collisions;
        }
        s->Printf ("good       = %u\n", good);
        s->Printf ("empties    = %u\n", empties);
        s->Printf ("collisions = %u\n", collisions);
        s->Printf ("avg count  = %llu\n", total / indexes_size);
    }
}
