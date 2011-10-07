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
#include "lldb/Core/RegularExpression.h"

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
		uint32_t bucket_count;       // The number of buckets in this hash table
		uint32_t hashes_count;       // The total number of unique hash values and hash data offsets in this table
        uint32_t prologue_length;    // The length of the prologue
        
		Header (uint32_t _prologue_length) :
            magic (HASH_MAGIC),
            version (1),
            addr_bytesize (4),
            hash_function (eHashFunctionDJB),
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

#include "lldb/Core/MappedHash.h"

struct DWARFMappedHash
{
    typedef std::vector<uint32_t> DIEArray;
    
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
        
        virtual ~Prologue()
        {
        }

        virtual void
        Clear ()
        {
            die_base_offset = 0;
            atoms.clear();
        }
        
//        void
//        Dump (std::ostream* ostrm_ptr);        
        
        uint32_t
        Read (const lldb_private::DataExtractor &data, uint32_t offset)
        {
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
        
//        virtual void
//        Write (BinaryStreamBuf &s);
        
        size_t
        GetByteSize () const
        {
            // Add an extra count to the atoms size for the zero termination Atom that gets
            // written to disk
            return sizeof(die_base_offset) + ((atoms.size() + 1) * sizeof(Atom));
        }
    };
    
    struct Header : public MappedHash::Header<Prologue>
    {
        Header (dw_offset_t _die_base_offset = 0)
        {
        }
        
        virtual 
        ~Header()
        {
        }

        virtual size_t
        GetByteSize (const HeaderData &header_data)
        {
            return header_data.GetByteSize();
        }
        
        
        //        virtual void
        //        Dump (std::ostream* ostrm_ptr);        
        //        
        virtual uint32_t
        Read (lldb_private::DataExtractor &data, uint32_t offset)
        {
            offset = MappedHash::Header<Prologue>::Read (data, offset);
            if (offset != UINT32_MAX)
            {
                offset = header_data.Read (data, offset);
            }
            return offset;
        }
        //        
        //        virtual void
        //        Write (BinaryStreamBuf &s);
    };
    
//    class ExportTable
//    {
//    public:
//        ExportTable ();
//        
//        void
//        AppendNames (DWARFDebugPubnamesSet &pubnames_set,
//                     StringTable &string_table);
//        
//        void
//        AppendNamesEntry (SymbolFileDWARF *dwarf2Data,
//                          const DWARFCompileUnit* cu,
//                          const DWARFDebugInfoEntry* die,
//                          StringTable &string_table);
//        
//        void
//        AppendTypesEntry (DWARFData *dwarf2Data,
//                          const DWARFCompileUnit* cu,
//                          const DWARFDebugInfoEntry* die,
//                          StringTable &string_table);
//        
//        size_t
//        Save (BinaryStreamBuf &names_data, const StringTable &string_table);
//        
//        void
//        AppendName (const char *name, 
//                    uint32_t die_offset, 
//                    StringTable &string_table,
//                    dw_offset_t name_debug_str_offset = DW_INVALID_OFFSET); // If "name" has already been looked up, then it can be supplied
//        void
//        AppendType (const char *name, 
//                    uint32_t die_offset, 
//                    StringTable &string_table);
//        
//        
//    protected:
//        struct Entry
//        {
//            uint32_t hash;
//            uint32_t str_offset;
//            uint32_t die_offset;
//        };
//        
//        // Map uniqued .debug_str offset to the corresponding DIE offsets
//        typedef std::map<uint32_t, DIEArray> NameInfo;
//        // Map a name hash to one or more name infos
//        typedef std::map<uint32_t, NameInfo> BucketEntry;
//        
//        static uint32_t
//        GetByteSize (const NameInfo &name_info);
//        
//        typedef std::vector<BucketEntry> BucketEntryColl;
//        typedef std::vector<Entry> EntryColl;
//        EntryColl m_entries;
//        
//    };
    
    
    // A class for reading and using a saved hash table from a block of data
    // in memory
    class MemoryTable : public MappedHash::MemoryTable<uint32_t, DWARFMappedHash::Header, DIEArray>
    {
    public:
        
        MemoryTable (lldb_private::DataExtractor &table_data, 
                     const lldb_private::DataExtractor &string_table,
                     bool is_apple_names) :
            MappedHash::MemoryTable<uint32_t, Header, DIEArray> (table_data),
            m_data (table_data),
            m_string_table (string_table),
            m_is_apple_names (is_apple_names)
        {
        }
    
        virtual 
        ~MemoryTable ()
        {
        }

        virtual const char *
        GetStringForKeyType (KeyType key) const
        {
            // The key in the DWARF table is the .debug_str offset for the string
            return m_string_table.PeekCStr (key);
        }
        
        virtual Result
        GetHashDataForName (const char *name,
                            uint32_t* hash_data_offset_ptr, 
                            Pair &pair) const
        {
            pair.key = m_data.GetU32 (hash_data_offset_ptr);
            // If the key is zero, this terminates our chain of HashData objects
            // for this hash value.
            if (pair.key == 0)
                return eResultEndOfHashData;

            // There definitely should be a string for this string offset, if
            // there isn't, there is something wrong, return and error
            const char *strp_cstr = m_string_table.PeekCStr (pair.key);
            if (strp_cstr == NULL)
                return eResultError;

            const uint32_t count = m_data.GetU32 (hash_data_offset_ptr);
            const uint32_t data_size = count * sizeof(uint32_t);
            if (count > 0 && m_data.ValidOffsetForDataOfSize (*hash_data_offset_ptr, data_size))
            {
                if (strcmp (name, strp_cstr) == 0)
                {
                    pair.value.clear();
                    for (uint32_t i=0; i<count; ++i)
                        pair.value.push_back (m_data.GetU32 (hash_data_offset_ptr));
                    return eResultKeyMatch;
                }
                else
                {
                    // Skip the data so we are ready to parse another HashData
                    // for this hash value
                    *hash_data_offset_ptr += data_size;
                    // The key doesn't match
                    return eResultKeyMismatch;
                }
            }
            else
            {
                *hash_data_offset_ptr = UINT32_MAX;
                return eResultError;
            }
        }

        virtual Result
        AppendHashDataForRegularExpression (const lldb_private::RegularExpression& regex,
                                            uint32_t* hash_data_offset_ptr, 
                                            Pair &pair) const
        {
            pair.key = m_data.GetU32 (hash_data_offset_ptr);
            // If the key is zero, this terminates our chain of HashData objects
            // for this hash value.
            if (pair.key == 0)
                return eResultEndOfHashData;
            
            // There definitely should be a string for this string offset, if
            // there isn't, there is something wrong, return and error
            const char *strp_cstr = m_string_table.PeekCStr (pair.key);
            if (strp_cstr == NULL)
                return eResultError;
            
            const uint32_t count = m_data.GetU32 (hash_data_offset_ptr);
            const uint32_t data_size = count * sizeof(uint32_t);
            if (count > 0 && m_data.ValidOffsetForDataOfSize (*hash_data_offset_ptr, data_size))
            {
                if (regex.Execute(strp_cstr))
                {
                    for (uint32_t i=0; i<count; ++i)
                        pair.value.push_back (m_data.GetU32 (hash_data_offset_ptr));
                    return eResultKeyMatch;
                }
                else
                {
                    // Skip the data so we are ready to parse another HashData
                    // for this hash value
                    *hash_data_offset_ptr += data_size;
                    // The key doesn't match
                    return eResultKeyMismatch;
                }
            }
            else
            {
                *hash_data_offset_ptr = UINT32_MAX;
                return eResultError;
            }
        }

        size_t
        AppendAllDIEsThatMatchingRegex (const lldb_private::RegularExpression& regex, 
                                        DIEArray &die_offsets) const
        {
            const uint32_t hash_count = m_header.hashes_count;
            Pair pair;
            for (uint32_t offset_idx=0; offset_idx<hash_count; ++offset_idx)
            {
                uint32_t hash_data_offset = GetHashDataOffset (offset_idx);
                while (hash_data_offset != UINT32_MAX)
                {
                    const uint32_t prev_hash_data_offset = hash_data_offset;
                    Result hash_result = AppendHashDataForRegularExpression (regex, &hash_data_offset, pair);
                    if (prev_hash_data_offset == hash_data_offset)
                        break;

                    // Check the result of getting our hash data
                    switch (hash_result)
                    {
                        case eResultKeyMatch:
                        case eResultKeyMismatch:
                            // Whether we matches or not, it doesn't matter, we
                            // keep looking.
                            break;
                            
                        case eResultEndOfHashData:
                        case eResultError:
                            hash_data_offset = UINT32_MAX;
                            break;
                    }
                }
            }
            die_offsets.swap (pair.value);
            return die_offsets.size();
        }
        
        size_t
        AppendAllDIEsInRange (const uint32_t die_offset_start, 
                              const uint32_t die_offset_end, 
                              DIEArray &die_offsets) const
        {
            const uint32_t hash_count = m_header.hashes_count;
            for (uint32_t offset_idx=0; offset_idx<hash_count; ++offset_idx)
            {
                bool done = false;
                uint32_t hash_data_offset = GetHashDataOffset (offset_idx);
                while (!done && hash_data_offset != UINT32_MAX)
                {
                    KeyType key = m_data.GetU32 (&hash_data_offset);
                    // If the key is zero, this terminates our chain of HashData objects
                    // for this hash value.
                    if (key == 0)
                        break;
                    
                    const uint32_t count = m_data.GetU32 (&hash_data_offset);
                    for (uint32_t i=0; i<count; ++i)
                    {
                        uint32_t die_offset = m_data.GetU32 (&hash_data_offset);
                        if (die_offset == 0)
                            done = true;
                        if (die_offset_start <= die_offset && die_offset < die_offset_end)
                            die_offsets.push_back(die_offset);
                    }
                }
            }
            return die_offsets.size();
        }

        size_t 
        FindByName (const char *name, DIEArray &die_offsets)
        {
            Pair kv_pair;
            size_t old_size = die_offsets.size();
            if (Find (name, kv_pair))
            {
                die_offsets.swap(kv_pair.value);
                return die_offsets.size() - old_size;
            }
            return 0;
        }
        
    protected:
        const lldb_private::DataExtractor &m_data;
        const lldb_private::DataExtractor &m_string_table;
        bool m_is_apple_names; // true => .apple_names, false => .apple_types
    };
};


#endif  // SymbolFileDWARF_HashedNameToDIE_h_
