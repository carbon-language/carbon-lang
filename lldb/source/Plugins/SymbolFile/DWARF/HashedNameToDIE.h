//===-- HashedNameToDIE.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_HashedNameToDIE_h_
#define SymbolFileDWARF_HashedNameToDIE_h_

#include <vector>
#include "lldb/lldb-defines.h"
#include "lldb/Core/dwarf.h"

class SymbolFileDWARF;

typedef std::vector<dw_offset_t> DIEArray;

class HashedNameToDIE
{
public:
    struct Header
	{
		uint16_t version;
		 uint8_t hash_type;
		 uint8_t hash_index_bitsize;
		uint32_t num_buckets;
		uint32_t num_hashes;
		uint32_t die_offset_base;

		Header() :
            version(0),
            hash_type (0),
            hash_index_bitsize (0),
            num_buckets(0),
            num_hashes (0),
            die_offset_base(0)
		{
		}
	};
    

    HashedNameToDIE (SymbolFileDWARF *dwarf, 
                     const lldb_private::DataExtractor &data);
    
    ~HashedNameToDIE ()
    {
    }
    
    bool
    IsValid () const
    {
        return m_header.version > 0;
    }

    uint32_t
    GetHashIndexMask () const
    {
        return (1u << m_header.hash_index_bitsize) - 1u;
    }
    
    uint32_t
    GetOffsetOfBucketEntry (uint32_t idx) const
    {
        if (idx < m_header.num_buckets)
            return sizeof(Header) + 4 * idx;
        return UINT32_MAX;
    }

    uint32_t
    GetOffsetOfHashValue (uint32_t idx) const
    {
        if (idx < m_header.num_hashes)
            return  sizeof(Header) + 
                    4 * m_header.num_buckets + 
                    4 * idx;
        return UINT32_MAX;
    }

    uint32_t
    GetOffsetOfHashDataOffset (uint32_t idx) const
    {
        if (idx < m_header.num_hashes)
        {
            return  sizeof(Header) +
                    4 * m_header.num_buckets +
                    4 * m_header.num_hashes +
                    4 * idx;
        }
        return UINT32_MAX;
    }

    void
    Dump (lldb_private::Stream *s);

    size_t
    Find (const lldb_private::ConstString &name, 
          DIEArray &die_ofsets) const;
    
    size_t
    Find (const lldb_private::RegularExpression& regex,  
          DIEArray &die_ofsets) const;

    void
    Initialize();
    
protected:
    SymbolFileDWARF *m_dwarf;
    const lldb_private::DataExtractor &m_data;
    Header m_header;
};

#endif  // SymbolFileDWARF_HashedNameToDIE_h_
