//===-- HashedNameToDIE.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HashedNameToDIE.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Symbol/ObjectFile.h"

#include "DWARFCompileUnit.h"
#include "DWARFDebugInfo.h"
#include "DWARFDebugInfoEntry.h"
#include "SymbolFileDWARF.h"
using namespace lldb;
using namespace lldb_private;

static uint32_t
dl_new_hash (const char *s)
{
    uint32_t h = 5381;
    
    for (unsigned char c = *s; c; c = *++s)
        h = ((h << 5) + h) + c;
    
    return h;
}

HashedNameToDIE::HashedNameToDIE (SymbolFileDWARF *dwarf, const DataExtractor &data) :
    m_dwarf (dwarf),
    m_data (data),
    m_header  ()
{
}

void
HashedNameToDIE::Initialize()
{
    uint32_t offset = 0;
    m_header.version = m_data.GetU16(&offset);
    if (m_header.version)
    {
        m_header.hash_type = m_data.GetU8(&offset);
        m_header.hash_index_bitsize = m_data.GetU8(&offset);
        m_header.num_buckets = m_data.GetU32(&offset);
        m_header.num_hashes = m_data.GetU32(&offset);
        m_header.die_offset_base = m_data.GetU32(&offset);
    }
}

size_t
HashedNameToDIE::Find (const ConstString &name, DIEArray &die_ofsets) const
{
    const size_t initial_size = die_ofsets.size();
    const char *name_cstr = name.GetCString();
    if (name_cstr && name_cstr[0])
    {
        // Hash the C string
        const uint32_t name_hash = dl_new_hash (name_cstr);
        
        // Find the correct bucket for the using the hash value
        const uint32_t bucket_idx = name_hash % m_header.num_buckets;
        
        // Calculate the offset for the bucket entry for the bucket index
        uint32_t offset = GetOffsetOfBucketEntry (bucket_idx);

        // Extract the bucket entry.
        const uint32_t bucket_entry = m_data.GetU32 (&offset);
        if (bucket_entry)
        {
            // The bucket entry is non-zero which means it isn't empty.
            // The bucket entry is made up of a hash index whose bit width
            // is m_header.hash_index_bitsize, and a hash count whose value
            // is the remaining bits in the 32 bit value. Below we extract
            // the hash index and the hash count
            const uint32_t hash_idx = bucket_entry & GetHashIndexMask();
            const uint32_t hash_count = bucket_entry >> m_header.hash_index_bitsize;
            const uint32_t hash_end_idx = hash_idx + hash_count;
            // Figure out the offset to the hash value by index
            uint32_t hash_offset = GetOffsetOfHashValue (hash_idx);
            for (uint32_t idx = hash_idx; idx < hash_end_idx; ++idx)
            {
                // Extract the hash value and see if it matches our string
                const uint32_t hash = m_data.GetU32 (&hash_offset);
                if (hash == name_hash)
                {
                    // The hash matches, but we still need to verify that the
                    // C string matches in case we have a hash collision. Figure
                    // out the offset for the data associated with this hash entry
                    offset = GetOffsetOfHashDataOffset (idx);
                    // Extract the first 32 bit value which is the .debug_str offset
                    // of the string we need
                    uint32_t hash_data_offset = m_data.GetU32 (&offset);
                    const uint32_t str_offset = m_data.GetU32 (&hash_data_offset);
                    // Extract the C string and comapare it
                    const char *cstr_name = m_dwarf->get_debug_str_data().PeekCStr(str_offset);
                    if (cstr_name)
                    {
                        if (strcmp(name_cstr, cstr_name) == 0)
                        {
                            // We have a match, now extract the DIE count
                            const uint32_t die_count = m_data.GetU32 (&hash_data_offset);
                            // Now extract "die_count" DIE offsets and put them into the
                            // results
                            for (uint32_t die_idx = 0; die_idx < die_count; ++die_idx)
                                die_ofsets.push_back(m_data.GetU32 (&hash_data_offset));
                        }
                    }
                }
            }
        }
    }
    return die_ofsets.size() - initial_size;
}

size_t
HashedNameToDIE::Find (const RegularExpression& regex, DIEArray &die_ofsets) const
{
//    const size_t initial_info_array_size = info_array.size();
//    collection::const_iterator pos, end = m_collection.end();
//    for (pos = m_collection.begin(); pos != end; ++pos)
//    {
//        if (regex.Execute(pos->first))
//            info_array.push_back (pos->second);
//    }
//    return info_array.size() - initial_info_array_size;
    return 0;
}

void
HashedNameToDIE::Dump (Stream *s)
{
//    collection::const_iterator pos, end = m_collection.end();
//    for (pos = m_collection.begin(); pos != end; ++pos)
//    {
//        s->Printf("%p: 0x%8.8x 0x%8.8x \"%s\"\n", pos->first, pos->second.cu_idx, pos->second.die_idx, pos->first);
//    }
}


