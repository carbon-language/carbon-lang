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
    enum NameFlags
    {
        eNameFlagIsExternal         = (1u << 0),
        eNameFlagIsClassCXX         = (1u << 1),
        eNameFlagIsClassObjC        = (1u << 2),
        eNameFlagIsClassObjCMaster  = (1u << 3)
    };
    
    enum TypeFlags
    {
        eTypeFlagIsExternal         = (1u << 0)
    };
    
    enum HashFunctionType
    {
        eHashFunctionDJB        = 0u,   // Daniel J Bernstein hash function that is also used by the ELF GNU_HASH sections
    };
    
    static const uint32_t HASH_MAGIC = 0x48415348u;
    
    struct Header
	{
        uint32_t magic;              // 'HASH' magic value to allow endian detection        
        uint16_t version;            // Version number
        uint8_t  addr_bytesize;      // Size in bytes of an address
		uint8_t  hash_function;      // The hash function enumeration that was used
		uint8_t  hash_count_bitsize; // Size in bits of the hash count in each bucket entry
		uint8_t  hash_index_bitsize; // Size in bits of the hash index in each bucket entry
		uint8_t  hash_bytesize;      // Size in bytes of the hash that is stored in the hashes which must be <= 4
		uint8_t  offset_bytesize;    // Size in bytes of the hash data offsets
        
		uint32_t bucket_count;       // The number of buckets in this hash table
		uint32_t hashes_count;       // The total number of unique hash values and hash data offsets in this table
        uint32_t prologue_length;    // The length of the prologue
        
		Header (uint32_t _prologue_length) :
            magic (HASH_MAGIC),
            version (1),
            addr_bytesize (4),
            hash_count_bitsize (8),
            hash_index_bitsize (24),
            hash_function (eHashFunctionDJB),
            hash_bytesize (4),      // Store the entire 32 bit hash by default
            offset_bytesize (4),    // Store a 4 byte offset for every hash value
            bucket_count (0),
            hashes_count (0),
            prologue_length (_prologue_length)
		{
		}
        
        virtual
        ~Header ()
        {            
        }
        
        virtual size_t
        GetByteSize() const
        {
            return  sizeof(magic) + 
            sizeof(version) + 
            sizeof(addr_bytesize) + 
            sizeof(hash_function) +
            sizeof(hash_count_bitsize) +
            sizeof(hash_index_bitsize) +
            sizeof(hash_bytesize) +
            sizeof(offset_bytesize) +
            sizeof(bucket_count) +
            sizeof(hashes_count) +
            sizeof(prologue_length) +
            prologue_length;
        }
        
        virtual void
        Dump (lldb_private::Stream &s);

        virtual uint32_t
        Read (const lldb_private::DataExtractor &data, uint32_t offset);
	};

    struct DWARF
    {
        enum AtomType
        {
            eAtomTypeNULL       = 0u,
            eAtomTypeHashString = 1u,   // String value for hash, use DW_FORM_strp (preferred) or DW_FORM_string
            eAtomTypeHashLength = 2u,   // Length of data for the previous string refered by the last eAtomTypeHashString atom
            eAtomTypeArraySize  = 3u,   // A count that specifies a number of atoms that follow this entry, the next atom defines what the atom type for the array is
            eAtomTypeDIEOffset  = 4u,   // DIE offset, check form for encoding. If DW_FORM_ref1,2,4,8 or DW_FORM_ref_udata, then this value is added to the prologue
            eAtomTypeTag        = 5u,   // DW_TAG_xxx value, should be encoded as DW_FORM_data1 (if no tags exceed 255) or DW_FORM_data2
            eAtomTypeNameFlags  = 6u,   // Flags from enum NameFlags
            eAtomTypeTypeFlags  = 7u,   // Flags from enum TypeFlags
        };
        
        struct Atom
        {
            uint16_t type;
            dw_form_t form;
            
            Atom (uint16_t t = eAtomTypeNULL, dw_form_t f = 0) :
                type (t),
                form (f)
            {
            }
        };
        
        typedef std::vector<Atom> AtomArray;
        
        
        static const char *
        GetAtomTypeName (uint16_t atom)
        {
            switch (atom)
            {
                case eAtomTypeNULL:         return "NULL";
                case eAtomTypeHashString:   return "hash-string";
                case eAtomTypeHashLength:   return "hash-data-length";
                case eAtomTypeArraySize:    return "array-size";
                case eAtomTypeDIEOffset:    return "die-offset";
                case eAtomTypeTag:          return "die-tag";
                case eAtomTypeNameFlags:    return "name-flags";
                case eAtomTypeTypeFlags:    return "type-flags";
            }
            return "<invalid>";
        }
        struct Prologue
        {
            // DIE offset base so die offsets in hash_data can be CU relative
            dw_offset_t die_base_offset;
            AtomArray atoms;
            
            Prologue (dw_offset_t _die_base_offset = 0) :
                die_base_offset (_die_base_offset)
            {
                // Define an array of DIE offsets by first defining an array, 
                // and then define the atom type for the array, in this case
                // we have an array of DIE offsets
                atoms.push_back (Atom(eAtomTypeArraySize, DW_FORM_data4));
                atoms.push_back (Atom(eAtomTypeDIEOffset, DW_FORM_data4));
            }

            virtual
            ~Prologue ()
            {            
            }
            
            
            virtual void
            Clear ()
            {
                die_base_offset = 0;
                atoms.clear();
            }
            
            virtual void
            Dump (lldb_private::Stream &s);        
            
            virtual uint32_t
            Read (const lldb_private::DataExtractor &data, uint32_t offset);
            
            size_t
            GetByteSize () const
            {
                // Add an extra count to the atoms size for the zero termination Atom that gets
                // written to disk
                return sizeof(die_base_offset) + ((atoms.size() + 1) * sizeof(Atom));
            }
        };
        
        struct Header : public HashedNameToDIE::Header
        {
            Header (dw_offset_t _die_base_offset = 0) :
                HashedNameToDIE::Header (sizeof(Prologue)),
                dwarf_prologue (_die_base_offset)
            {
            }

            virtual
            ~Header ()
            {            
            }

            Prologue dwarf_prologue;
            
            virtual void
            Dump (lldb_private::Stream &s);        
            
            virtual uint32_t
            Read (const lldb_private::DataExtractor &data, uint32_t offset);
        };
        
    };
    

    class MemoryTable
    {
    public:
        
        MemoryTable (SymbolFileDWARF *dwarf, 
                     const lldb_private::DataExtractor &data,
                     bool is_apple_names);
        
        ~MemoryTable ()
        {
        }

        bool
        Initialize ();

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
            if (idx < m_header.bucket_count)
                return m_header.GetByteSize() + 4 * idx;
            return UINT32_MAX;
        }
        
        uint32_t
        GetOffsetOfHashValue (uint32_t idx) const
        {
            if (idx < m_header.hashes_count)
                return  m_header.GetByteSize() + 
                4 * m_header.bucket_count + 
                4 * idx;
            return UINT32_MAX;
        }
        
        uint32_t
        GetOffsetOfHashDataOffset (uint32_t idx) const
        {
            if (idx < m_header.hashes_count)
            {
                return  m_header.GetByteSize() +
                4 * m_header.bucket_count +
                4 * m_header.hashes_count +
                4 * idx;
            }
            return UINT32_MAX;
        }
        
        void
        Dump (lldb_private::Stream &s);
        
        size_t
        Find (const char *name, DIEArray &die_ofsets) const;
        
    protected:
        const lldb_private::DataExtractor &m_data;
        const lldb_private::DataExtractor &m_string_table;
        bool m_is_apple_names; // true => .apple_names, false => .apple_types
        DWARF::Header m_header;
    };    
};

#endif  // SymbolFileDWARF_HashedNameToDIE_h_
