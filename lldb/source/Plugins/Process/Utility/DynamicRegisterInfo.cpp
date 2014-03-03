//===-- DynamicRegisterInfo.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "DynamicRegisterInfo.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/DataFormatters/FormatManager.h"

#ifndef LLDB_DISABLE_PYTHON
#include "lldb/Interpreter/PythonDataObjects.h"
#endif

using namespace lldb;
using namespace lldb_private;

DynamicRegisterInfo::DynamicRegisterInfo () :
    m_regs (),
    m_sets (),
    m_set_reg_nums (),
    m_set_names (),
    m_value_regs_map (),
    m_invalidate_regs_map (),
    m_reg_data_byte_size (0),
    m_finalized (false)
{
}

DynamicRegisterInfo::DynamicRegisterInfo (const lldb_private::PythonDictionary &dict, ByteOrder byte_order) :
    m_regs (),
    m_sets (),
    m_set_reg_nums (),
    m_set_names (),
    m_value_regs_map (),
    m_invalidate_regs_map (),
    m_reg_data_byte_size (0),
    m_finalized (false)
{
    SetRegisterInfo (dict, byte_order);
}

DynamicRegisterInfo::~DynamicRegisterInfo ()
{
}


size_t
DynamicRegisterInfo::SetRegisterInfo (const lldb_private::PythonDictionary &dict,
                                      ByteOrder byte_order)
{
    assert(!m_finalized);
#ifndef LLDB_DISABLE_PYTHON
    PythonList sets (dict.GetItemForKey("sets"));
    if (sets)
    {
        const uint32_t num_sets = sets.GetSize();
        for (uint32_t i=0; i<num_sets; ++i)
        {
            PythonString py_set_name(sets.GetItemAtIndex(i));
            ConstString set_name;
            if (py_set_name)
                set_name.SetCString(py_set_name.GetString());
            if (set_name)
            {
                RegisterSet new_set = { set_name.AsCString(), NULL, 0, NULL };
                m_sets.push_back (new_set);
            }
            else
            {
                Clear();
                return 0;
            }
        }
        m_set_reg_nums.resize(m_sets.size());
    }
    PythonList regs (dict.GetItemForKey("registers"));
    if (regs)
    {
        const uint32_t num_regs = regs.GetSize();
        PythonString name_pystr("name");
        PythonString altname_pystr("alt-name");
        PythonString bitsize_pystr("bitsize");
        PythonString offset_pystr("offset");
        PythonString encoding_pystr("encoding");
        PythonString format_pystr("format");
        PythonString set_pystr("set");
        PythonString gcc_pystr("gcc");
        PythonString dwarf_pystr("dwarf");
        PythonString generic_pystr("generic");
        PythonString slice_pystr("slice");
        PythonString composite_pystr("composite");
        PythonString invalidate_regs_pystr("invalidate-regs");
        
//        typedef std::map<std::string, std::vector<std::string> > InvalidateNameMap;
//        InvalidateNameMap invalidate_map;
        for (uint32_t i=0; i<num_regs; ++i)
        {
            PythonDictionary reg_info_dict(regs.GetItemAtIndex(i));
            if (reg_info_dict)
            {
                // { 'name':'rcx'       , 'bitsize' :  64, 'offset' :  16, 'encoding':'uint'  , 'format':'hex'         , 'set': 0, 'gcc' : 2, 'dwarf' : 2, 'generic':'arg4', 'alt-name':'arg4', },
                RegisterInfo reg_info;
                std::vector<uint32_t> value_regs;
                std::vector<uint32_t> invalidate_regs;
                bzero (&reg_info, sizeof(reg_info));
                
                reg_info.name = ConstString (reg_info_dict.GetItemForKeyAsString(name_pystr)).GetCString();
                if (reg_info.name == NULL)
                {
                    Clear();
                    return 0;
                }
                    
                reg_info.alt_name = ConstString (reg_info_dict.GetItemForKeyAsString(altname_pystr)).GetCString();
                
                reg_info.byte_offset = reg_info_dict.GetItemForKeyAsInteger(offset_pystr, UINT32_MAX);

                if (reg_info.byte_offset == UINT32_MAX)
                {
                    // No offset for this register, see if the register has a value expression
                    // which indicates this register is part of another register. Value expressions
                    // are things like "rax[31:0]" which state that the current register's value
                    // is in a concrete register "rax" in bits 31:0. If there is a value expression
                    // we can calculate the offset
                    bool success = false;
                    const char *slice_cstr = reg_info_dict.GetItemForKeyAsString(slice_pystr);
                    if (slice_cstr)
                    {
                        // Slices use the following format:
                        //  REGNAME[MSBIT:LSBIT]
                        // REGNAME - name of the register to grab a slice of
                        // MSBIT - the most significant bit at which the current register value starts at
                        // LSBIT - the least significant bit at which the current register value ends at
                        static RegularExpression g_bitfield_regex("([A-Za-z_][A-Za-z0-9_]*)\\[([0-9]+):([0-9]+)\\]");
                        RegularExpression::Match regex_match(3);
                        if (g_bitfield_regex.Execute(slice_cstr, &regex_match))
                        {
                            llvm::StringRef reg_name_str;
                            std::string msbit_str;
                            std::string lsbit_str;
                            if (regex_match.GetMatchAtIndex(slice_cstr, 1, reg_name_str) &&
                                regex_match.GetMatchAtIndex(slice_cstr, 2, msbit_str) &&
                                regex_match.GetMatchAtIndex(slice_cstr, 3, lsbit_str))
                            {
                                const uint32_t msbit = Args::StringToUInt32(msbit_str.c_str(), UINT32_MAX);
                                const uint32_t lsbit = Args::StringToUInt32(lsbit_str.c_str(), UINT32_MAX);
                                if (msbit != UINT32_MAX && lsbit != UINT32_MAX)
                                {
                                    if (msbit > lsbit)
                                    {
                                        const uint32_t msbyte = msbit / 8;
                                        const uint32_t lsbyte = lsbit / 8;

                                        ConstString containing_reg_name(reg_name_str);
                                        
                                        RegisterInfo *containing_reg_info = GetRegisterInfo (containing_reg_name);
                                        if (containing_reg_info)
                                        {
                                            const uint32_t max_bit = containing_reg_info->byte_size * 8;
                                            if (msbit < max_bit && lsbit < max_bit)
                                            {
                                                m_invalidate_regs_map[containing_reg_info->kinds[eRegisterKindLLDB]].push_back(i);
                                                m_value_regs_map[i].push_back(containing_reg_info->kinds[eRegisterKindLLDB]);
                                                m_invalidate_regs_map[i].push_back(containing_reg_info->kinds[eRegisterKindLLDB]);
                                                
                                                if (byte_order == eByteOrderLittle)
                                                {
                                                    success = true;
                                                    reg_info.byte_offset = containing_reg_info->byte_offset + lsbyte;
                                                }
                                                else if (byte_order == eByteOrderBig)
                                                {
                                                    success = true;
                                                    reg_info.byte_offset = containing_reg_info->byte_offset + msbyte;
                                                }
                                                else
                                                {
                                                    assert(!"Invalid byte order");
                                                }
                                            }
                                            else
                                            {
                                                if (msbit > max_bit)
                                                    printf("error: msbit (%u) must be less than the bitsize of the register (%u)\n", msbit, max_bit);
                                                else
                                                    printf("error: lsbit (%u) must be less than the bitsize of the register (%u)\n", lsbit, max_bit);
                                            }
                                        }
                                        else
                                        {
                                            printf("error: invalid concrete register \"%s\"\n", containing_reg_name.GetCString());
                                        }
                                    }
                                    else
                                    {
                                        printf("error: msbit (%u) must be greater than lsbit (%u)\n", msbit, lsbit);
                                    }
                                }
                                else
                                {
                                    printf("error: msbit (%u) and lsbit (%u) must be valid\n", msbit, lsbit);
                                }
                            }
                            else
                            {
                                // TODO: print error invalid slice string that doesn't follow the format
                                printf("error: failed to extract regex matches for parsing the register bitfield regex\n");

                            }
                        }
                        else
                        {
                            // TODO: print error invalid slice string that doesn't follow the format
                            printf("error: failed to match against register bitfield regex\n");
                        }
                    }
                    else
                    {
                        PythonList composite_reg_list (reg_info_dict.GetItemForKey(composite_pystr));
                        if (composite_reg_list)
                        {
                            const size_t num_composite_regs = composite_reg_list.GetSize();
                            if (num_composite_regs > 0)
                            {
                                uint32_t composite_offset = UINT32_MAX;
                                for (uint32_t composite_idx=0; composite_idx<num_composite_regs; ++composite_idx)
                                {
                                    PythonString composite_reg_name_pystr(composite_reg_list.GetItemAtIndex(composite_idx));
                                    if (composite_reg_name_pystr)
                                    {
                                        ConstString composite_reg_name(composite_reg_name_pystr.GetString());
                                        if (composite_reg_name)
                                        {
                                            RegisterInfo *composite_reg_info = GetRegisterInfo (composite_reg_name);
                                            if (composite_reg_info)
                                            {
                                                if (composite_offset > composite_reg_info->byte_offset)
                                                    composite_offset = composite_reg_info->byte_offset;
                                                m_value_regs_map[i].push_back(composite_reg_info->kinds[eRegisterKindLLDB]);
                                                m_invalidate_regs_map[composite_reg_info->kinds[eRegisterKindLLDB]].push_back(i);
                                                m_invalidate_regs_map[i].push_back(composite_reg_info->kinds[eRegisterKindLLDB]);
                                            }
                                            else
                                            {
                                                // TODO: print error invalid slice string that doesn't follow the format
                                                printf("error: failed to find composite register by name: \"%s\"\n", composite_reg_name.GetCString());
                                            }
                                        }
                                        else
                                        {
                                            printf("error: 'composite' key contained an empty string\n");
                                        }
                                    }
                                    else
                                    {
                                        printf("error: 'composite' list value wasn't a python string\n");
                                    }
                                }
                                if (composite_offset != UINT32_MAX)
                                {
                                    reg_info.byte_offset = composite_offset;
                                    success = m_value_regs_map.find(i) != m_value_regs_map.end();
                                }
                                else
                                {
                                    printf("error: 'composite' registers must specify at least one real register\n");
                                }
                            }
                            else
                            {
                                printf("error: 'composite' list was empty\n");
                            }
                        }
                    }
                    
                    
                    if (!success)
                    {
                        Clear();
                        return 0;
                    }
                }
                const int64_t bitsize = reg_info_dict.GetItemForKeyAsInteger(bitsize_pystr, 0);
                if (bitsize == 0)
                {
                    Clear();
                    return 0;
                }

                reg_info.byte_size =  bitsize / 8;
                
                const char *format_cstr = reg_info_dict.GetItemForKeyAsString(format_pystr);
                if (format_cstr)
                {
                    if (Args::StringToFormat(format_cstr, reg_info.format, NULL).Fail())
                    {
                        Clear();
                        return 0;
                    }
                }
                else
                {
                    reg_info.format = (Format)reg_info_dict.GetItemForKeyAsInteger (format_pystr, eFormatHex);
                }
                
                const char *encoding_cstr = reg_info_dict.GetItemForKeyAsString(encoding_pystr);
                if (encoding_cstr)
                    reg_info.encoding = Args::StringToEncoding (encoding_cstr, eEncodingUint);
                else
                    reg_info.encoding = (Encoding)reg_info_dict.GetItemForKeyAsInteger (encoding_pystr, eEncodingUint);

                const int64_t set = reg_info_dict.GetItemForKeyAsInteger(set_pystr, -1);
                if (set >= m_sets.size())
                {
                    Clear();
                    return 0;
                }

                // Fill in the register numbers
                reg_info.kinds[lldb::eRegisterKindLLDB]    = i;
                reg_info.kinds[lldb::eRegisterKindGDB]     = i;
                reg_info.kinds[lldb::eRegisterKindGCC]     = reg_info_dict.GetItemForKeyAsInteger(gcc_pystr, LLDB_INVALID_REGNUM);
                reg_info.kinds[lldb::eRegisterKindDWARF]   = reg_info_dict.GetItemForKeyAsInteger(dwarf_pystr, LLDB_INVALID_REGNUM);
                const char *generic_cstr = reg_info_dict.GetItemForKeyAsString(generic_pystr);
                if (generic_cstr)
                    reg_info.kinds[lldb::eRegisterKindGeneric] = Args::StringToGenericRegister (generic_cstr);
                else
                    reg_info.kinds[lldb::eRegisterKindGeneric] = reg_info_dict.GetItemForKeyAsInteger(generic_pystr, LLDB_INVALID_REGNUM);

                // Check if this register invalidates any other register values when it is modified
                PythonList invalidate_reg_list (reg_info_dict.GetItemForKey(invalidate_regs_pystr));
                if (invalidate_reg_list)
                {
                    const size_t num_regs = invalidate_reg_list.GetSize();
                    if (num_regs > 0)
                    {
                        for (uint32_t idx=0; idx<num_regs; ++idx)
                        {
                            PythonObject invalidate_reg_object (invalidate_reg_list.GetItemAtIndex(idx));
                            PythonString invalidate_reg_name_pystr(invalidate_reg_object);
                            if (invalidate_reg_name_pystr)
                            {
                                ConstString invalidate_reg_name(invalidate_reg_name_pystr.GetString());
                                if (invalidate_reg_name)
                                {
                                    RegisterInfo *invalidate_reg_info = GetRegisterInfo (invalidate_reg_name);
                                    if (invalidate_reg_info)
                                    {
                                        m_invalidate_regs_map[i].push_back(invalidate_reg_info->kinds[eRegisterKindLLDB]);
                                    }
                                    else
                                    {
                                        // TODO: print error invalid slice string that doesn't follow the format
                                        printf("error: failed to find a 'invalidate-regs' register for \"%s\" while parsing register \"%s\"\n", invalidate_reg_name.GetCString(), reg_info.name);
                                    }
                                }
                                else
                                {
                                    printf("error: 'invalidate-regs' list value was an empty string\n");
                                }
                            }
                            else
                            {
                                PythonInteger invalidate_reg_num(invalidate_reg_object);

                                if (invalidate_reg_num)
                                {
                                    const int64_t r = invalidate_reg_num.GetInteger();
                                    if (r != UINT64_MAX)
                                        m_invalidate_regs_map[i].push_back(r);
                                    else
                                        printf("error: 'invalidate-regs' list value wasn't a valid integer\n");
                                }
                                else
                                {
                                    printf("error: 'invalidate-regs' list value wasn't a python string or integer\n");
                                }
                            }
                        }
                    }
                    else
                    {
                        printf("error: 'invalidate-regs' contained an empty list\n");
                    }
                }

                // Calculate the register offset
                const size_t end_reg_offset = reg_info.byte_offset + reg_info.byte_size;
                if (m_reg_data_byte_size < end_reg_offset)
                    m_reg_data_byte_size = end_reg_offset;

                m_regs.push_back (reg_info);
                m_set_reg_nums[set].push_back(i);

            }
            else
            {
                Clear();
                return 0;
            }
        }
        Finalize ();
    }
#endif
    return m_regs.size();
}


void
DynamicRegisterInfo::AddRegister (RegisterInfo &reg_info,
                                  ConstString &reg_name, 
                                  ConstString &reg_alt_name, 
                                  ConstString &set_name)
{
    assert(!m_finalized);
    const uint32_t reg_num = m_regs.size();
    reg_info.name = reg_name.AsCString();
    assert (reg_info.name);
    reg_info.alt_name = reg_alt_name.AsCString(NULL);
    uint32_t i;
    if (reg_info.value_regs)
    {
        for (i=0; reg_info.value_regs[i] != LLDB_INVALID_REGNUM; ++i)
            m_value_regs_map[reg_num].push_back(reg_info.value_regs[i]);
    }
    if (reg_info.invalidate_regs)
    {
        for (i=0; reg_info.invalidate_regs[i] != LLDB_INVALID_REGNUM; ++i)
            m_invalidate_regs_map[reg_num].push_back(reg_info.invalidate_regs[i]);
    }
    m_regs.push_back (reg_info);
    uint32_t set = GetRegisterSetIndexByName (set_name, true);
    assert (set < m_sets.size());
    assert (set < m_set_reg_nums.size());
    assert (set < m_set_names.size());
    m_set_reg_nums[set].push_back(reg_num);
    size_t end_reg_offset = reg_info.byte_offset + reg_info.byte_size;
    if (m_reg_data_byte_size < end_reg_offset)
        m_reg_data_byte_size = end_reg_offset;
}

void
DynamicRegisterInfo::Finalize ()
{
    if (m_finalized)
        return;
    
    m_finalized = true;
    const size_t num_sets = m_sets.size();
    for (size_t set = 0; set < num_sets; ++set)
    {
        assert (m_sets.size() == m_set_reg_nums.size());
        m_sets[set].num_registers = m_set_reg_nums[set].size();
        m_sets[set].registers = &m_set_reg_nums[set][0];
    }
    
    // sort and unique all value registers and make sure each is terminated with
    // LLDB_INVALID_REGNUM
    
    for (reg_to_regs_map::iterator pos = m_value_regs_map.begin(), end = m_value_regs_map.end();
         pos != end;
         ++pos)
    {
        if (pos->second.size() > 1)
        {
            std::sort (pos->second.begin(), pos->second.end());
            reg_num_collection::iterator unique_end = std::unique (pos->second.begin(), pos->second.end());
            if (unique_end != pos->second.end())
                pos->second.erase(unique_end, pos->second.end());
        }
        assert (!pos->second.empty());
        if (pos->second.back() != LLDB_INVALID_REGNUM)
            pos->second.push_back(LLDB_INVALID_REGNUM);
    }
    
    // Now update all value_regs with each register info as needed
    const size_t num_regs = m_regs.size();
    for (size_t i=0; i<num_regs; ++i)
    {
        if (m_value_regs_map.find(i) != m_value_regs_map.end())
            m_regs[i].value_regs = m_value_regs_map[i].data();
        else
            m_regs[i].value_regs = NULL;
    }

    // Expand all invalidation dependencies
    for (reg_to_regs_map::iterator pos = m_invalidate_regs_map.begin(), end = m_invalidate_regs_map.end();
         pos != end;
         ++pos)
    {
        const uint32_t reg_num = pos->first;
        
        if (m_regs[reg_num].value_regs)
        {
            reg_num_collection extra_invalid_regs;
            for (const uint32_t invalidate_reg_num : pos->second)
            {
                reg_to_regs_map::iterator invalidate_pos = m_invalidate_regs_map.find(invalidate_reg_num);
                if (invalidate_pos != m_invalidate_regs_map.end())
                {
                    for (const uint32_t concrete_invalidate_reg_num : invalidate_pos->second)
                    {
                        if (concrete_invalidate_reg_num != reg_num)
                            extra_invalid_regs.push_back(concrete_invalidate_reg_num);
                    }
                }
            }
            pos->second.insert(pos->second.end(), extra_invalid_regs.begin(), extra_invalid_regs.end());
        }
    }

    // sort and unique all invalidate registers and make sure each is terminated with
    // LLDB_INVALID_REGNUM
    for (reg_to_regs_map::iterator pos = m_invalidate_regs_map.begin(), end = m_invalidate_regs_map.end();
         pos != end;
         ++pos)
    {
        if (pos->second.size() > 1)
        {
            std::sort (pos->second.begin(), pos->second.end());
            reg_num_collection::iterator unique_end = std::unique (pos->second.begin(), pos->second.end());
            if (unique_end != pos->second.end())
                pos->second.erase(unique_end, pos->second.end());
        }
        assert (!pos->second.empty());
        if (pos->second.back() != LLDB_INVALID_REGNUM)
            pos->second.push_back(LLDB_INVALID_REGNUM);
    }

    // Now update all invalidate_regs with each register info as needed
    for (size_t i=0; i<num_regs; ++i)
    {
        if (m_invalidate_regs_map.find(i) != m_invalidate_regs_map.end())
            m_regs[i].invalidate_regs = m_invalidate_regs_map[i].data();
        else
            m_regs[i].invalidate_regs = NULL;
    }
}

size_t
DynamicRegisterInfo::GetNumRegisters() const
{
    return m_regs.size();
}

size_t
DynamicRegisterInfo::GetNumRegisterSets() const
{
    return m_sets.size();
}

size_t
DynamicRegisterInfo::GetRegisterDataByteSize() const
{
    return m_reg_data_byte_size;
}

const RegisterInfo *
DynamicRegisterInfo::GetRegisterInfoAtIndex (uint32_t i) const
{
    if (i < m_regs.size())
        return &m_regs[i];
    return NULL;
}

const RegisterSet *
DynamicRegisterInfo::GetRegisterSet (uint32_t i) const
{
    if (i < m_sets.size())
        return &m_sets[i];
    return NULL;
}

uint32_t
DynamicRegisterInfo::GetRegisterSetIndexByName (ConstString &set_name, bool can_create)
{
    name_collection::iterator pos, end = m_set_names.end();
    for (pos = m_set_names.begin(); pos != end; ++pos)
    {
        if (*pos == set_name)
            return std::distance (m_set_names.begin(), pos);
    }
    
    m_set_names.push_back(set_name);
    m_set_reg_nums.resize(m_set_reg_nums.size()+1);
    RegisterSet new_set = { set_name.AsCString(), NULL, 0, NULL };
    m_sets.push_back (new_set);
    return m_sets.size() - 1;
}

uint32_t
DynamicRegisterInfo::ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num) const
{
    reg_collection::const_iterator pos, end = m_regs.end();
    for (pos = m_regs.begin(); pos != end; ++pos)
    {
        if (pos->kinds[kind] == num)
            return std::distance (m_regs.begin(), pos);
    }
    
    return LLDB_INVALID_REGNUM;
}

void
DynamicRegisterInfo::Clear()
{
    m_regs.clear();
    m_sets.clear();
    m_set_reg_nums.clear();
    m_set_names.clear();
    m_value_regs_map.clear();
    m_invalidate_regs_map.clear();
    m_reg_data_byte_size = 0;
    m_finalized = false;
}

void
DynamicRegisterInfo::Dump () const
{
    StreamFile s(stdout, false);
    const size_t num_regs = m_regs.size();
    s.Printf("%p: DynamicRegisterInfo contains %" PRIu64 " registers:\n", this, (uint64_t)num_regs);
    for (size_t i=0; i<num_regs; ++i)
    {
        s.Printf("[%3" PRIu64 "] name = %-10s", (uint64_t)i, m_regs[i].name);
        s.Printf(", size = %2u, offset = %4u, encoding = %u, format = %-10s",
                 m_regs[i].byte_size,
                 m_regs[i].byte_offset,
                 m_regs[i].encoding,
                 FormatManager::GetFormatAsCString (m_regs[i].format));
        if (m_regs[i].kinds[eRegisterKindGDB] != LLDB_INVALID_REGNUM)
            s.Printf(", gdb = %3u", m_regs[i].kinds[eRegisterKindGDB]);
        if (m_regs[i].kinds[eRegisterKindDWARF] != LLDB_INVALID_REGNUM)
            s.Printf(", dwarf = %3u", m_regs[i].kinds[eRegisterKindDWARF]);
        if (m_regs[i].kinds[eRegisterKindGCC] != LLDB_INVALID_REGNUM)
            s.Printf(", gcc = %3u", m_regs[i].kinds[eRegisterKindGCC]);
        if (m_regs[i].kinds[eRegisterKindGeneric] != LLDB_INVALID_REGNUM)
            s.Printf(", generic = %3u", m_regs[i].kinds[eRegisterKindGeneric]);
        if (m_regs[i].alt_name)
            s.Printf(", alt-name = %s", m_regs[i].alt_name);
        if (m_regs[i].value_regs)
        {
            s.Printf(", value_regs = [ ");
            for (size_t j=0; m_regs[i].value_regs[j] != LLDB_INVALID_REGNUM; ++j)
            {
                s.Printf("%s ", m_regs[m_regs[i].value_regs[j]].name);
            }
            s.Printf("]");
        }
        if (m_regs[i].invalidate_regs)
        {
            s.Printf(", invalidate_regs = [ ");
            for (size_t j=0; m_regs[i].invalidate_regs[j] != LLDB_INVALID_REGNUM; ++j)
            {
                s.Printf("%s ", m_regs[m_regs[i].invalidate_regs[j]].name);
            }
            s.Printf("]");
        }
        s.EOL();
    }
    
    const size_t num_sets = m_sets.size();
    s.Printf("%p: DynamicRegisterInfo contains %" PRIu64 " register sets:\n", this, (uint64_t)num_sets);
    for (size_t i=0; i<num_sets; ++i)
    {
        s.Printf("set[%" PRIu64 "] name = %s, regs = [", (uint64_t)i, m_sets[i].name);
        for (size_t idx=0; idx<m_sets[i].num_registers; ++idx)
        {
            s.Printf("%s ", m_regs[m_sets[i].registers[idx]].name);
        }
        s.Printf("]\n");
    }
}



lldb_private::RegisterInfo *
DynamicRegisterInfo::GetRegisterInfo (const lldb_private::ConstString &reg_name)
{
    for (auto &reg_info : m_regs)
    {
        // We can use pointer comparison since we used a ConstString to set
        // the "name" member in AddRegister()
        if (reg_info.name == reg_name.GetCString())
        {
            return &reg_info;
        }
    }
    return NULL;
}
