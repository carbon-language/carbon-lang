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
#include "DWARFDefines.h"
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


void
HashedNameToDIE::Header::Dump (Stream &s)
{
    s.Printf ("header.magic              = 0x%8.8x", magic);
    s.Printf ("header.version            = 0x%4.4x", version);
    s.Printf ("header.addr_bytesize      = 0x%2.2x", addr_bytesize);
    s.Printf ("header.hash_function      = 0x%2.2x", hash_function);
    s.Printf ("header.hash_count_bitsize = 0x%2.2x", hash_count_bitsize);
    s.Printf ("header.hash_index_bitsize = 0x%2.2x", hash_index_bitsize);
    s.Printf ("header.hash_bytesize      = 0x%2.2x", hash_bytesize);
    s.Printf ("header.offset_bytesize    = 0x%2.2x", offset_bytesize);
    s.Printf ("header.bucket_count       = 0x%8.8x %u", bucket_count, bucket_count);
    s.Printf ("header.hashes_count       = 0x%8.8x %u", hashes_count, hashes_count);
    s.Printf ("header.prologue_length    = 0x%8.8x %u", prologue_length, prologue_length);
}

uint32_t
HashedNameToDIE::Header::Read (const DataExtractor &data, uint32_t offset)
{
    magic = data.GetU32 (&offset);
    if (magic != HASH_MAGIC)
    {
        // Magic bytes didn't match
        version = 0;
        return UINT32_MAX;
    }
    
    version = data.GetU16 (&offset);
    if (version != 1)
    {
        // Unsupported version
        return UINT32_MAX;
    }
    addr_bytesize       = data.GetU8  (&offset);
    hash_function       = data.GetU8  (&offset);
    hash_count_bitsize  = data.GetU8  (&offset);
    hash_index_bitsize  = data.GetU8  (&offset);
    hash_bytesize       = data.GetU8  (&offset);
    offset_bytesize     = data.GetU8  (&offset);
    bucket_count        = data.GetU32 (&offset);
    hashes_count        = data.GetU32 (&offset);
    prologue_length     = data.GetU32 (&offset);
    return offset;
}

void
HashedNameToDIE::DWARF::Header::Dump (Stream &s)
{
    HashedNameToDIE::Header::Dump (s);
    dwarf_prologue.Dump (s);
}

uint32_t
HashedNameToDIE::DWARF::Header::Read (const DataExtractor &data, uint32_t offset)
{
    offset = HashedNameToDIE::Header::Read (data, offset);
    if (offset != UINT32_MAX)
        offset = dwarf_prologue.Read (data, offset);
    else
        dwarf_prologue.Clear();
    return offset;
}

void
HashedNameToDIE::DWARF::Prologue::Dump (Stream &s)
{
    s.Printf ("dwarf_prologue.die_base_offset    = 0x%8.8x\n", die_base_offset);
    const size_t num_atoms = atoms.size();
    for (size_t i = 0; i < num_atoms; ++i)
    {
        s.Printf ("dwarf_prologue.atom[%zi] = %17s %s\n", 
                  i, 
                  GetAtomTypeName (atoms[i].type), 
                  DW_FORM_value_to_name(atoms[i].form));
    }
}

uint32_t
HashedNameToDIE::DWARF::Prologue::Read (const DataExtractor &data, uint32_t offset)
{
    Clear();
    die_base_offset = data.GetU32 (&offset);
    Atom atom;
    while (offset != UINT32_MAX)
    {
        atom.type = data.GetU16 (&offset);
        atom.form = data.GetU16 (&offset);
        if (atom.type == eAtomTypeNULL)
            break;
        atoms.push_back(atom);
    }
    return offset;
}


HashedNameToDIE::MemoryTable::MemoryTable (SymbolFileDWARF *dwarf, 
                                           const lldb_private::DataExtractor &data,
                                           bool is_apple_names) :
    m_data (data),
    m_string_table (dwarf->get_debug_str_data ()),
    m_is_apple_names (is_apple_names),
    m_header  ()
{
}

bool
HashedNameToDIE::MemoryTable::Initialize ()
{
    uint32_t offset = 0;
    offset = m_header.Read (m_data, offset);
    return m_header.version == 1;
}


size_t
HashedNameToDIE::MemoryTable::Find (const char *name_cstr, DIEArray &die_ofsets) const
{
    if (m_header.version == 1)
    {
        const size_t initial_size = die_ofsets.size();
        if (name_cstr && name_cstr[0])
        {
            // Hash the C string
            const uint32_t name_hash = dl_new_hash (name_cstr);
            
            // Find the correct bucket for the using the hash value
            const uint32_t bucket_idx = name_hash % m_header.bucket_count;
            
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
                        uint32_t hash_data_offset = m_data.GetU32 (&offset);
                        uint32_t str_offset;
                        // Now we have the offset to the data for all strings that match
                        // our 32 bit hash. The format of the hash bucket is:
                        //
                        // uint32_t stroff;     // string offset in .debug_str table
                        // uint32_t num_dies;   // Number of DIEs in debug info that match the string that follow this
                        // uint32_t die_offsets[num_dies]; // An array of DIE offsets
                        //
                        // When a "stroff" is read and it is zero, then the data for this
                        // hash is terminated.
                        while ((str_offset = m_data.GetU32 (&hash_data_offset)) != 0)
                        {
                            // Extract the C string and comapare it
                            const char *cstr_name = m_string_table.PeekCStr(str_offset);
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
        }
        return die_ofsets.size() - initial_size;
    }
    return 0;
}

void
HashedNameToDIE::MemoryTable::Dump (Stream &s)
{
    if (m_header.version == 1)
    {

        bool verbose = s.GetVerbose();
        
        if (m_is_apple_names)
            s.PutCString (".apple_names contents:\n");
        else
            s.PutCString (".apple_types contents:\n");
        
        m_header.Dump (s);
        uint32_t i,j,k;
        uint32_t empty_bucket_count = 0;
        uint32_t hash_collisions = 0;
        uint32_t bucket_entry_offset = GetOffsetOfBucketEntry (0);
        for (i=0; i<m_header.bucket_count; ++i)
        {
            const uint32_t bucket_entry = m_data.GetU32 (&bucket_entry_offset);
            s.Printf("bucket[%u] 0x%8.8x", i, bucket_entry);
            if (bucket_entry)
            {
                const uint32_t hash_idx = bucket_entry & GetHashIndexMask();
                const uint32_t hash_count = bucket_entry >> m_header.hash_index_bitsize;
                
                s.Printf(" (hash_idx = %u, hash_count = %u)\n", hash_idx, hash_count);
                
                const uint32_t hash_end_idx = hash_idx + hash_count;
                uint32_t hash_offset = GetOffsetOfHashValue (hash_idx);
                uint32_t data_offset = GetOffsetOfHashDataOffset (hash_idx);
                
                for (j=hash_idx; j<hash_end_idx; ++j)
                {
                    const uint32_t hash = m_data.GetU32 (&hash_offset);
                    uint32_t hash_data_offset = m_data.GetU32 (&data_offset);
                    if (verbose)
                        s.Printf("  hash[%u] = 0x%8.8x, offset[%u] = 0x%8.8x\n", j, hash, j, hash_data_offset);
                    else
                        s.Printf("  hash[%u] = 0x%8.8x\n", j, hash);

                    uint32_t string_idx = 0;
                    uint32_t strp_offset;
                    while ((strp_offset = m_data.GetU32 (&hash_data_offset)) != 0)
                    {
                        const uint32_t num_die_offsets = m_data.GetU32 (&hash_data_offset);
                        s.Printf("    string[%u] = 0x%8.8x \"%s\", dies[%u] = {", 
                                 string_idx, 
                                 strp_offset, 
                                 m_string_table.PeekCStr(strp_offset),
                                 num_die_offsets);
                        ++string_idx;
                        
                        for (k=0; k<num_die_offsets; ++k)
                        {
                            const uint32_t die_offset = m_data.GetU32 (&hash_data_offset);
                            s.Printf(" 0x%8.8x", die_offset);
                        }
                        s.PutCString (" }\n");
                    }
                    if (string_idx > 1)
                        ++hash_collisions;
                }
            }
            else
            {
                s.PutCString(" (empty)\n");
                ++empty_bucket_count;
            }
        }
        
        s.Printf ("%u of %u buckets empty (%2.1f%%)\n",  empty_bucket_count, m_header.bucket_count, (((float)empty_bucket_count/(float)m_header.bucket_count)*100.0f));
        s.Printf ("Average hashes/non-empty bucket = %2.1f%%\n", ((float)m_header.hashes_count/(float)(m_header.bucket_count - empty_bucket_count)));
        s.Printf ("Hash collisions = %u\n", hash_collisions);
    }
}


