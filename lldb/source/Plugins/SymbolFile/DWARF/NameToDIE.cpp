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

void
NameToDIE::Insert (const ConstString& name, const Info &info)
{
    m_collection.insert (std::make_pair(name.AsCString(), info));
}

size_t
NameToDIE::Find (const ConstString &name, std::vector<Info> &info_array) const
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
NameToDIE::Find (const RegularExpression& regex, std::vector<Info> &info_array) const
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
NameToDIE::Dump (Stream *s)
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
    
    bool
    operator < (const HashEntry &rhs) const
    {
        return hash < rhs.hash;
    }
};

struct HashHeader
{
    uint32_t version;
    uint32_t bucket_info_size; // The fixed data associated with this bucket
    uint32_t bucket_count;
    uint32_t flags;
};

struct HashBucketInfo
{
    uint32_t offset;    // Offset to the data for each bucket
};

struct HashBucketEntryStrp
{
    HashBucketEntryStrp () :
        str_offset (0)
    {
    }

    HashBucketEntryStrp (uint32_t s, uint32_t d) :
        str_offset (s)
    {
    }
    
    uint32_t
    GetByteSize ()
    {
        return  sizeof(uint32_t) + // String offset in .debug_str
                sizeof(uint32_t) + // Number of dies
                die_array.size() * sizeof(uint32_t);
    }

    uint32_t str_offset;
    std::vector<dw_offset_t> die_array;
};

typedef std::vector<dw_offset_t> DIEArray;
typedef std::map<const char *, DIEArray> NameToDIEArrayMap;

struct HashBucketEntryCStr
{
    uint32_t
    GetByteSize () const
    {
        uint32_t byte_size = 0;
        NameToDIEArrayMap::const_iterator pos, end = name_to_die.end();
        for (pos = name_to_die.begin(); pos != end; ++pos)
        {
            // Include the size of the and a length for the dies, and all dies
            byte_size += sizeof(uint32_t) + sizeof(uint32_t) * (pos->second.size() + 1);
        }
        return byte_size;
    }
    
    NameToDIEArrayMap name_to_die;
};

static uint32_t
closest_power_2_less_than_n (uint32_t n)
{  
    if (n)
        return 0x80000000u >> __builtin_clz (n);
    return 0;
}

typedef struct HashEntry HashEntryType;

void
NameToDIE::Hash (Stream *s, SymbolFileDWARF *dwarf)
{
//    if (m_collection.empty())
//        return;
//    
//    typedef std::vector<HashEntryType> hash_collection;
//    hash_collection hash_entries;
//    collection::const_iterator pos, end = m_collection.end();
//    for (pos = m_collection.begin(); pos != end; ++pos)
//    {
//        HashEntry entry = { dl_new_hash (pos->first), pos->second.cu_idx, pos->second.die_idx, pos->first };
//        hash_entries.push_back (entry); 
//    }
//    
////    const DataExtractor &debug_str_data = dwarf->get_debug_str_data();
//    
////    uint32_t collisions = 0;
////    for (i=1; i<hash_entries_size; ++i)
////    {
////        if (hash_entries[i-1].hash == hash_entries[i].hash &&
////            hash_entries[i-1].name != hash_entries[i].name)
////            ++collisions;
////    }
////    s->Printf("count = %u, collisions = %u\n", hash_entries_size, collisions);
//    
////    for (i=0; i<hash_entries_size; ++i)
////        s->Printf("0x%8.8x: cu = %8u, die = %8u, name = '%s'\n", 
////                  hash_entries[i].hash,
////                  hash_entries[i].cu_idx,
////                  hash_entries[i].die_idx,
////                  hash_entries[i].name);
//    DWARFDebugInfo *debug_info = dwarf->DebugInfo();
//
//    uint32_t num_buckets;
//    if (hash_entries_size > 1024)
//        num_buckets = closest_power_2_less_than_n (hash_entries_size/16);
//    else if (hash_entries_size > 128)
//        num_buckets = closest_power_2_less_than_n (hash_entries_size/8);
//    else
//        num_buckets = closest_power_2_less_than_n (hash_entries_size/4);
//    if (num_buckets == 0)
//        num_buckets = 1;
//    
//    //for (uint32_t power_2 = 0x10; power_2 <= hash_entries_size; power_2 <<= 1)
//    {
////        if (num_buckets > 0x10 && num_buckets > hash_entries_size)
////            break;
//
//        typedef std::vector<uint32_t> uint32_array;
//        typedef std::map<uint32_t, HashBucketEntryCStr> HashBucketEntryMap;
//        std::vector<HashBucketEntryMap> hash_buckets;
//        hash_buckets.resize(num_buckets);
//        
//        uint32_t bucket_entry_empties = 0;
//        uint32_t bucket_entry_single = 0;
//        uint32_t bucket_entry_collisions = 0;
//        uint32_t names_entry_single = 0;
//        uint32_t names_entry_collisions = 0;
//        //StreamString hash_file_data(Stream::eBinary, dwarf->GetObjectFile()->GetAddressByteSize(), dwarf->GetObjectFile()->GetByteSize());
//
//        // Write hash table header
////        hash_file_data.PutHex32 (1); // Version
////        hash_file_data.PutHex32 (4); // Sizeof bucket data
////        hash_file_data.PutHex32 (num_buckets);
////        hash_file_data.PutHex32 (0); // Flags
//        
//        s->Printf("HashHeader = { version = %u, bucket_info_size = %u, bucket_count = %u, flags = 0x%8.8x }\n", 1, (uint32_t)sizeof(HashBucketInfo), num_buckets, 0);
//        
//        for (i=0; i<hash_entries_size; ++i)
//        {
//            uint32_t hash = hash_entries[i].hash;
//            uint32_t bucket_idx = hash_entries[i].hash % num_buckets;
//            DWARFCompileUnit *cu = debug_info->GetCompileUnitAtIndex (hash_entries[i].cu_idx);
//            cu->ExtractDIEsIfNeeded(false);
//            DWARFDebugInfoEntry *die = cu->GetDIEAtIndexUnchecked(hash_entries[i].die_idx);
//            hash_buckets[bucket_idx][hash].name_to_die[hash_entries[i].name].push_back(die->GetOffset());
//        }
//        uint32_t byte_size = sizeof(HashHeader); // Header
//        uint32_t data_offset = 0;
//        uint32_t num_bucket_entries;
//        uint32_t bucket_data_size;
//        // Now for each bucket we write the offset to the data for each bucket
//        // The offset is currently a zero based offset from the end of this table
//        // which is header.num_buckets * sizeof(uint32_t) long.
//        for (i=0; i<num_buckets; ++i)
//        {
//            byte_size += sizeof(HashBucketInfo);
//            HashBucketEntryMap &bucket_entry = hash_buckets[i];
//            bucket_data_size = 0;
//            HashBucketEntryMap::const_iterator pos, end = bucket_entry.end();
//            for (pos = bucket_entry.begin(); pos != end; ++pos)
//            {
//                bucket_data_size += sizeof(pos->first) + pos->second.GetByteSize();
//            }
//            if (bucket_data_size > 0)
//            {
//                // Offset to bucket data
////                hash_file_data.PutHex32 (data_offset); 
//                s->Printf("bucket[%u] {0x%8.8x}\n", i, data_offset);
//                data_offset += bucket_data_size;
//            }
//            else
//            {
//                // Invalid offset that indicates an empty bucket
////                hash_file_data.PutHex32 (UINT32_MAX);
//                s->Printf("bucket[%u] {0xFFFFFFFF}\n", i);
//                ++bucket_entry_empties;
//            }
//        }
//
//        // Now we write the bucket data for each bucket that corresponds to each bucket
//        // offset from above.
//        data_offset = 0;
//        uint32_t total_num_name_entries = 0;
//        uint32_t total_num_bucket_entries = 0;
//        uint32_t total_non_empty_buckets = 0;
//        for (i=0; i<num_buckets; ++i)
//        {
//            HashBucketEntryMap &bucket_entry = hash_buckets[i];
//            bucket_data_size = 0;
//            if (bucket_entry.empty())
//                continue;
//
//            ++total_non_empty_buckets;
//            
//            s->Printf("0x%8.8x: BucketEntry:\n", data_offset, num_bucket_entries);                
//            bucket_data_size = 0;
//            uint32_t num_bucket_entries = 0;
//            HashBucketEntryMap::const_iterator pos, end = bucket_entry.end();
//            for (pos = bucket_entry.begin(); pos != end; ++pos)
//            {
//                ++num_bucket_entries;
//                uint32_t hash_data_len = pos->second.GetByteSize();
//                s->Printf("  hash = 0x%8.8x, length = 0x%8.8x:\n", pos->first, hash_data_len);
////                hash_file_data.PutHex32 (pos->first); // Write the hash
////                hash_file_data.PutHex32 (hash_data_len); // The length of the data for this hash not including the length itself
//                
//                const HashBucketEntryCStr &hash_entry = pos->second;
//                uint32_t num_name_entries = 0;
//                NameToDIEArrayMap::const_iterator name_pos, name_end = hash_entry.name_to_die.end();
//                for (name_pos = hash_entry.name_to_die.begin(); name_pos != name_end; ++name_pos)
//                {
//                    ++num_name_entries;
//                    ++total_num_name_entries;
//                    s->Printf("    name = %p '%s'\n", name_pos->first, name_pos->first);
////                    hash_file_data.PutHex32 (pos->first); // Write the hash
////                    hash_file_data.PutHex32 (hash_data_len); // The length of the data for this hash not including the length itself
//
//
//                    const uint32_t num_dies = name_pos->second.size();
//                    s->Printf("      dies[%u] = { ", num_dies);
//                    for (uint32_t j=0; j < num_dies; ++j)
//                        s->Printf("0x%8.8x ", name_pos->second[j]);
//                    s->PutCString("}\n");
//                }
//                if (num_name_entries == 1)
//                    ++names_entry_single;
//                else if (num_name_entries > 1)
//                    ++names_entry_collisions;
//                bucket_data_size += sizeof(pos->first) + hash_data_len;
//            }
//            data_offset += bucket_data_size;
//            byte_size += bucket_data_size;
//            total_num_bucket_entries += num_bucket_entries;
//            if (num_bucket_entries == 1)
//                ++bucket_entry_single;
//            else if (num_bucket_entries > 1)
//                ++bucket_entry_collisions;
//        }
//
//        s->Printf ("Trying size of %u buckets, %u items:\n", num_buckets, hash_entries_size);
//        s->Printf ("buckets: empty    = %u (%%%f)\n", bucket_entry_empties, ((float)bucket_entry_empties/(float)num_buckets) * 100.0f);
//        s->Printf ("buckets: single   = %u\n", bucket_entry_single);
//        s->Printf ("buckets: multiple = %u (avg = %f entries/bucket, avg = %f entries/non-empty bucket)\n",
//                   bucket_entry_collisions, 
//                   (float)total_num_bucket_entries / (float)num_buckets,
//                   (float)total_num_bucket_entries / (float)total_non_empty_buckets);
//        s->Printf ("names  : single   = %u of %u\n", names_entry_single, total_num_name_entries);
//        s->Printf ("names  : multiple = %u of %u\n", names_entry_collisions, total_num_name_entries);
//        s->Printf ("total byte size   = %u\n", byte_size);
//        s->PutCString ("\n----------------------------------------------------------------------\n\n");
//    }
}
