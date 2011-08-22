//===-- DynamicRegisterInfo.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DynamicRegisterInfo.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

DynamicRegisterInfo::DynamicRegisterInfo () :
    m_regs (),
    m_sets (),
    m_set_reg_nums (),
    m_reg_names (),
    m_reg_alt_names (),
    m_set_names (),
    m_reg_data_byte_size (0)
{
}

DynamicRegisterInfo::~DynamicRegisterInfo ()
{
}

void
DynamicRegisterInfo::AddRegister (RegisterInfo &reg_info, 
                                  ConstString &reg_name, 
                                  ConstString &reg_alt_name, 
                                  ConstString &set_name)
{
    const uint32_t reg_num = m_regs.size();
    m_reg_names.push_back (reg_name);
    m_reg_alt_names.push_back (reg_alt_name);
    reg_info.name = reg_name.AsCString();
    assert (reg_info.name);
    reg_info.alt_name = reg_alt_name.AsCString(NULL);
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
    for (uint32_t set = 0; set < m_sets.size(); ++set)
    {
        assert (m_sets.size() == m_set_reg_nums.size());
        m_sets[set].num_registers = m_set_reg_nums[set].size();
        m_sets[set].registers = &m_set_reg_nums[set][0];
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
    m_reg_names.clear();
    m_reg_alt_names.clear();
    m_set_names.clear();
}
