//===-- EmulationStateARM.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "EmulationStateARM.h"

#include "lldb/Core/Scalar.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/RegisterContext.h"

#include "Utility/ARM_DWARF_Registers.h"

using namespace lldb;
using namespace lldb_private;

EmulationStateARM::EmulationStateARM () :
    m_gpr (),
    m_vfp_regs (),
    m_memory ()
{
    ClearPseudoRegisters();
}

EmulationStateARM::~EmulationStateARM ()
{
}

bool
EmulationStateARM::LoadPseudoRegistersFromFrame (StackFrame &frame)
{
    RegisterContext *reg_context = frame.GetRegisterContext().get();
    Scalar value;
    uint64_t reg_value64;
    uint32_t reg_value32;
    
    bool success = true;
    
    for (int i = dwarf_r0; i < dwarf_r0 + 17; ++i)
    {
        uint32_t internal_reg_num = reg_context->ConvertRegisterKindToRegisterNumber (eRegisterKindDWARF, i);
        if (reg_context->ReadRegisterValue (internal_reg_num, value))
        {
            reg_value32 = (uint32_t) value.GetRawBits64 (0);
            m_gpr[i - dwarf_r0] = reg_value32;
        }
        else
            success = false;
    }
    
    for (int i = dwarf_s0; i < dwarf_s0 + 32; ++i)
    {
        uint32_t internal_reg_num = reg_context->ConvertRegisterKindToRegisterNumber (eRegisterKindDWARF, i);
        if (reg_context->ReadRegisterValue (internal_reg_num, value))
        {
            uint32_t idx = i - dwarf_s0;
            reg_value32 = (uint32_t) value.GetRawBits64 (0);
            m_vfp_regs.sd_regs[idx / 2].s_reg[idx % 2] = reg_value32;
        }
        else
            success = false;
    }
    
    for (int i = dwarf_d0; i < dwarf_d0 + 32; ++i)
    {
        uint32_t internal_reg_num = reg_context->ConvertRegisterKindToRegisterNumber (eRegisterKindDWARF, i);
        if (reg_context->ReadRegisterValue (internal_reg_num, value))
        {
            uint32_t idx = i - dwarf_d0;
            reg_value64 = value.GetRawBits64 (0);
            if (i < 16)
                m_vfp_regs.sd_regs[idx].d_reg = reg_value64;
            else
                m_vfp_regs.d_regs[idx - 16] = reg_value64;
        }
        else
            success = false;
    }
    
    return success;
}
    
bool
EmulationStateARM::StorePseudoRegisterValue (uint32_t reg_num, uint64_t value)
{
    if ((dwarf_r0 <= reg_num) && (reg_num <= dwarf_cpsr))
        m_gpr[reg_num  - dwarf_r0] = (uint32_t) value;
    else if ((dwarf_s0 <= reg_num) && (reg_num <= dwarf_s31))
    {
        uint32_t idx = reg_num - dwarf_s0;
        m_vfp_regs.sd_regs[idx / 2].s_reg[idx % 2] = (uint32_t) value;
    }
    else if ((dwarf_d0 <= reg_num) && (reg_num <= dwarf_d31))
    {
        if ((reg_num - dwarf_d0) < 16)
        {
            m_vfp_regs.sd_regs[reg_num - dwarf_d0].d_reg = value;
        }
        else
            m_vfp_regs.d_regs[reg_num - dwarf_d16] = value;
    }
    else
        return false;
        
    return true;
}
    
uint64_t
EmulationStateARM::ReadPseudoRegisterValue (uint32_t reg_num, bool &success)
{
    uint64_t value = 0;
    success = true;
    
    if ((dwarf_r0 <= reg_num) && (reg_num <= dwarf_cpsr))
        value = m_gpr[reg_num  - dwarf_r0];
    else if ((dwarf_s0 <= reg_num) && (reg_num <= dwarf_s31))
    {
        uint32_t idx = reg_num - dwarf_s0;
        value = m_vfp_regs.sd_regs[idx / 2].s_reg[idx % 2];
    }
    else if ((dwarf_d0 <= reg_num) && (reg_num <= dwarf_d31))
    {
        if ((reg_num - dwarf_d0) < 16)
            value = m_vfp_regs.sd_regs[reg_num - dwarf_d0].d_reg;
        else
            value = m_vfp_regs.d_regs[reg_num - dwarf_d16];
    }
    else
        success = false;
        
    return value;
}
    
void
EmulationStateARM::ClearPseudoRegisters ()
{
    for (int i = 0; i < 17; ++i)
        m_gpr[i] = 0;
    
    for (int i = 0; i < 16; ++i)
        m_vfp_regs.sd_regs[i].d_reg = 0;
    
    for (int i = 0; i < 16; ++i)
        m_vfp_regs.d_regs[i] = 0;
}

void
EmulationStateARM::ClearPseudoMemory ()
{
    m_memory.clear();
}
    
bool
EmulationStateARM::StoreToPseudoAddress (lldb::addr_t p_address, uint64_t value, uint32_t size)
{
    if (size > 8)
        return false;
    
    if (size <= 4)
        m_memory[p_address] = value;
    else if (size == 8)
    {
        m_memory[p_address] = (value << 32) >> 32;
        m_memory[p_address + 4] = value << 32;
    }
    return true;
}
    
uint32_t
EmulationStateARM::ReadFromPseudoAddress (lldb::addr_t p_address, uint32_t size, bool &success)
{
    std::map<lldb::addr_t,uint32_t>::iterator pos;
    uint32_t ret_val = 0;
    
    success = true;
    pos = m_memory.find(p_address);
    if (pos != m_memory.end())
        ret_val = pos->second;
    else
        success = false;
        
    return ret_val;
}

size_t
EmulationStateARM::ReadPseudoMemory (EmulateInstruction *instruction,
                                     void *baton,
                                     const EmulateInstruction::Context &context,
                                     lldb::addr_t addr,
                                     void *dst,
                                     size_t length)
{
    if (!baton)
        return 0;
        
    bool success = true;
    EmulationStateARM *pseudo_state = (EmulationStateARM *) baton;
    if (length <= 4)
    {
        uint32_t value = pseudo_state->ReadFromPseudoAddress (addr, length, success);
        if (!success)
            return 0;
            
        *((uint32_t *) dst) = value;
    }
    else if (length == 8)
    {
        uint32_t value1 = pseudo_state->ReadFromPseudoAddress (addr, 4, success);
        if (!success)
            return 0;
            
        uint32_t value2 = pseudo_state->ReadFromPseudoAddress (addr + 4, 4, success);
        if (!success)
            return 0;
            
        uint64_t value64 = value2;
        value64 = (value64 << 32) | value1;
        *((uint64_t *) dst) = value64;
    }
    else
        success = false;
    
    if (success)
        return length;
    
    return 0;
}
    
size_t
EmulationStateARM::WritePseudoMemory (EmulateInstruction *instruction,
                                      void *baton,
                                      const EmulateInstruction::Context &context,
                                      lldb::addr_t addr,
                                      const void *dst,
                                      size_t length)
{
    if (!baton)
        return 0;
        
    bool success;
    EmulationStateARM *pseudo_state = (EmulationStateARM *) baton;
    uint64_t value = *((uint64_t *) dst);
    success = pseudo_state->StoreToPseudoAddress (addr, value, length);
    if (success)
        return length;
        
    return 0;
}
    
bool
EmulationStateARM::ReadPseudoRegister (EmulateInstruction *instruction,
                                       void *baton,
                                       const RegisterInfo &reg_info,
                                       uint64_t &reg_value)
{
    if (!baton)
        return false;
        
    bool success = true;
    EmulationStateARM *pseudo_state = (EmulationStateARM *) baton;
    
    assert (reg_info.kinds[eRegisterKindDWARF] != LLDB_INVALID_REGNUM);
    reg_value = pseudo_state->ReadPseudoRegisterValue (reg_info.kinds[eRegisterKindDWARF], success);
    
    return success;
    
}
    
bool
EmulationStateARM::WritePseudoRegister (EmulateInstruction *instruction,
                                        void *baton,
                                        const EmulateInstruction::Context &context,
                                        const RegisterInfo &reg_info,
                                        uint64_t reg_value)
{
    if (!baton)
        return false;

    assert (reg_info.kinds[eRegisterKindDWARF] != LLDB_INVALID_REGNUM);
    EmulationStateARM *pseudo_state = (EmulationStateARM *) baton;
    return pseudo_state->StorePseudoRegisterValue (reg_info.kinds[eRegisterKindDWARF], reg_value);
}
                         
bool
EmulationStateARM::CompareState (EmulationStateARM &other_state)
{
    bool match = true;
   
    for (int i = 0; match && i < 17; ++i)
    {
        if (m_gpr[i] != other_state.m_gpr[i])
            match = false;
    }
    
    for (int i = 0; match && i < 16; ++i)
    {
        if (m_vfp_regs.sd_regs[i].s_reg[0] != other_state.m_vfp_regs.sd_regs[i].s_reg[0])
            match = false;

        if (m_vfp_regs.sd_regs[i].s_reg[1] != other_state.m_vfp_regs.sd_regs[i].s_reg[1])
            match = false;
    }
    
    for (int i = 0; match && i < 32; ++i)
    {
        if (i < 16)
        {
            if (m_vfp_regs.sd_regs[i].d_reg != other_state.m_vfp_regs.sd_regs[i].d_reg)
                match = false;
        }
        else
        {
            if (m_vfp_regs.d_regs[i - 16] != other_state.m_vfp_regs.d_regs[i - 16])
                match = false;
        }
    }
    
    return match;
}

bool
EmulationStateARM::LoadStateFromDictionary (OptionValueDictionary *test_data)
{
    static ConstString memory_key ("memory");
    static ConstString registers_key ("registers");
    
    if (!test_data)
        return false;
    
    OptionValueSP value_sp = test_data->GetValueForKey (memory_key);
    
    // Load memory, if present.
    
    if (value_sp.get() != NULL)
    {
        static ConstString address_key ("address");
        static ConstString data_key ("data");
        uint64_t start_address = 0;
        
        OptionValueDictionary *mem_dict = value_sp->GetAsDictionary();
        value_sp = mem_dict->GetValueForKey (address_key);
        if (value_sp.get() == NULL)
            return false;
        else
            start_address = value_sp->GetUInt64Value ();
        
        value_sp = mem_dict->GetValueForKey (data_key);
        OptionValueArray *mem_array = value_sp->GetAsArray();
        if (!mem_array)
            return false;

        uint32_t num_elts = mem_array->GetSize();
        uint32_t address = (uint32_t) start_address;
        
        for (int i = 0; i < num_elts; ++i)
        {
            value_sp = mem_array->GetValueAtIndex (i);
            if (value_sp.get() == NULL)
                return false;
            uint64_t value = value_sp->GetUInt64Value();
            StoreToPseudoAddress (address, value, 4);
            address = address + 4;
        }
    }
    
    value_sp = test_data->GetValueForKey (registers_key);
    if (value_sp.get() == NULL)
        return false;

        
    // Load General Registers
   
    OptionValueDictionary *reg_dict = value_sp->GetAsDictionary ();
   
    StreamString sstr;
    for (int i = 0; i < 16; ++i)
    {
        sstr.Clear();
        sstr.Printf ("r%d", i);
        ConstString reg_name (sstr.GetData());
        value_sp = reg_dict->GetValueForKey (reg_name);
        if (value_sp.get() == NULL)
            return false;
        uint64_t reg_value = value_sp->GetUInt64Value();
        StorePseudoRegisterValue (dwarf_r0 + i, reg_value);
    }
    
    static ConstString cpsr_name ("cpsr");
    value_sp = reg_dict->GetValueForKey (cpsr_name);
    if (value_sp.get() == NULL)
        return false;
    StorePseudoRegisterValue (dwarf_cpsr, value_sp->GetUInt64Value());
    
    // Load s/d Registers
    for (int i = 0; i < 32; ++i)
    {
        sstr.Clear();
        sstr.Printf ("s%d", i);
        ConstString reg_name (sstr.GetData());
        value_sp = reg_dict->GetValueForKey (reg_name);
        if (value_sp.get() == NULL)
            return false;
        uint64_t reg_value = value_sp->GetUInt64Value();
        StorePseudoRegisterValue (dwarf_s0 + i, reg_value);
    }

    return true;
}

