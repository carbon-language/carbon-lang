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
        if (name_cstr && name_cstr[0])
        {
            // Hash the C string
            const uint32_t name_hash = dl_new_hash (name_cstr);
            
            const uint32_t bucket_count = m_header.bucket_count;
            const uint32_t hashes_count = m_header.bucket_count;
            // Find the correct bucket for the using the hash value
            const uint32_t bucket_idx = name_hash % bucket_count;
            
            // Calculate the offset for the bucket entry for the bucket index
            uint32_t offset = GetOffsetOfBucketEntry (bucket_idx);
            
            // Extract the bucket entry which is a hash index. If the hash index
            // is UINT32_MAX, then the bucket is empty. If it isn't, it is the
            // index of the hash in the hashes array. We will then iterate through
            // all hashes as long as they match "bucket_idx" which was calculated
            // above
            uint32_t hash_idx = m_data.GetU32 (&offset);
            if (hash_idx != UINT32_MAX)
            {
                uint32_t hash_offset = GetOffsetOfHashValue (hash_idx);
                
                const size_t initial_size = die_ofsets.size();
                uint32_t hash;
                while (((hash = m_data.GetU32 (&hash_offset)) % bucket_count) == bucket_idx)
                {
                    if (hash_idx >= hashes_count)
                        break;
                    
                    if (hash == name_hash)
                    {
                        // The hash matches, but we still need to verify that the
                        // C string matches in case we have a hash collision. Figure
                        // out the offset for the data associated with this hash entry
                        offset = GetOffsetOfHashDataOffset (hash_idx);
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
                    ++hash_idx;
                }
                
                return die_ofsets.size() - initial_size;
            }
        }
    }
    return 0;
}

void
HashedNameToDIE::MemoryTable::Dump (Stream &s)
{
    if (m_header.version == 1)
    {        
        if (m_is_apple_names)
            s.PutCString (".apple_names contents:\n");
        else
            s.PutCString (".apple_types contents:\n");
        
        m_header.Dump (s);
        uint32_t empty_bucket_count = 0;
        uint32_t hash_collisions = 0;
        uint32_t hash_idx_offset = GetOffsetOfBucketEntry (0);
        const uint32_t bucket_count = m_header.bucket_count;
        const uint32_t hashes_count = m_header.hashes_count;
        for (uint32_t bucket_idx=0; bucket_idx<bucket_count; ++bucket_idx)
        {
            uint32_t hash_idx = m_data.GetU32 (&hash_idx_offset);
            s.Printf("bucket[%u] ", bucket_idx);

            if (hash_idx != UINT32_MAX)
            {                
                s.Printf(" => hash[%u]\n", hash_idx);
                
                uint32_t hash_offset = GetOffsetOfHashValue (hash_idx);
                uint32_t data_offset = GetOffsetOfHashDataOffset (hash_idx);
                
                uint32_t hash;
                while (((hash = m_data.GetU32 (&hash_offset)) % bucket_count) == bucket_idx)
                {
                    if (hash_idx >= hashes_count)
                        break;
                    
                    uint32_t hash_data_offset = m_data.GetU32 (&data_offset);
                    s.Printf("  hash[%u] = 0x%8.8x\n", hash_idx, hash);

                    uint32_t string_count = 0;
                    uint32_t strp_offset;
                    while ((strp_offset = m_data.GetU32 (&hash_data_offset)) != 0)
                    {
                        const uint32_t num_die_offsets = m_data.GetU32 (&hash_data_offset);
                        s.Printf("   str[%u] = 0x%8.8x \"%s\", dies[%u] = {", 
                                 string_count, 
                                 strp_offset, 
                                 m_string_table.PeekCStr(strp_offset),
                                 num_die_offsets);
                        ++string_count;
                        
                        for (uint32_t die_idx=0; die_idx<num_die_offsets; ++die_idx)
                        {
                            const uint32_t die_offset = m_data.GetU32 (&hash_data_offset);
                            s.Printf(" 0x%8.8x", die_offset);
                        }
                        s.PutCString (" }\n");
                    }
                    if (string_count > 1)
                        ++hash_collisions;
                }
            }
            else
            {
                s.PutCString(" EMPTY\n");
                ++empty_bucket_count;
            }
            s.EOL();
        }
        s.EOL();
        s.Printf ("%u of %u buckets empty (%2.1f%%)\n",  empty_bucket_count, bucket_count, (((float)empty_bucket_count/(float)m_header.bucket_count)*100.0f));
        s.Printf ("Average hashes/non-empty bucket = %2.1f%%\n", ((float)m_header.hashes_count/(float)(m_header.bucket_count - empty_bucket_count)));
        s.Printf ("Hash collisions = %u\n", hash_collisions);
    }
}


