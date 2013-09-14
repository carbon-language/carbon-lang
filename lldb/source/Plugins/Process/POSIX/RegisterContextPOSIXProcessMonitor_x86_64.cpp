//===-- RegisterContextPOSIXProcessMonitor_x86_64.h ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "lldb/Target/Thread.h"
#include "lldb/Core/RegisterValue.h"

#include "ProcessPOSIX.h"
#include "RegisterContextPOSIXProcessMonitor_x86_64.h"
#include "ProcessMonitor.h"

using namespace lldb_private;
using namespace lldb;

// Support ptrace extensions even when compiled without required kernel support
#ifndef NT_X86_XSTATE
  #define NT_X86_XSTATE 0x202
#endif

#define REG_CONTEXT_SIZE (GetGPRSize() + sizeof(RegisterContextPOSIX_x86_64::FPR))

static uint32_t
size_and_rw_bits(size_t size, bool read, bool write)
{
    uint32_t rw;
    if (read)
        rw = 0x3; // READ or READ/WRITE
    else if (write)
        rw = 0x1; // WRITE
    else
        assert(0 && "read and write cannot both be false");

    switch (size)
    {
    case 1:
        return rw;
    case 2:
        return (0x1 << 2) | rw;
    case 4:
        return (0x3 << 2) | rw;
    case 8:
        return (0x2 << 2) | rw;
    default:
        assert(0 && "invalid size, must be one of 1, 2, 4, or 8");
    }
}

RegisterContextPOSIXProcessMonitor_x86_64::RegisterContextPOSIXProcessMonitor_x86_64(Thread &thread,
                                                                                     uint32_t concrete_frame_idx,
                                                                                     RegisterInfoInterface *register_info)
    : RegisterContextPOSIX_x86_64(thread, concrete_frame_idx, register_info)
{
}

ProcessMonitor &
RegisterContextPOSIXProcessMonitor_x86_64::GetMonitor()
{
    ProcessSP base = CalculateProcess();
    ProcessPOSIX *process = static_cast<ProcessPOSIX*>(base.get());
    return process->GetMonitor();
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::ReadGPR()
{
     ProcessMonitor &monitor = GetMonitor();
     return monitor.ReadGPR(m_thread.GetID(), &m_gpr, GetGPRSize());
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::ReadFPR()
{
    ProcessMonitor &monitor = GetMonitor();
    if (m_fpr_type == eFXSAVE)
        return monitor.ReadFPR(m_thread.GetID(), &m_fpr.xstate.fxsave, sizeof(m_fpr.xstate.fxsave));

    if (m_fpr_type == eXSAVE)
        return monitor.ReadRegisterSet(m_thread.GetID(), &m_iovec, sizeof(m_fpr.xstate.xsave), NT_X86_XSTATE);
    return false;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::WriteGPR()
{
    ProcessMonitor &monitor = GetMonitor();
    return monitor.WriteGPR(m_thread.GetID(), &m_gpr, GetGPRSize());
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::WriteFPR()
{
    ProcessMonitor &monitor = GetMonitor();
    if (m_fpr_type == eFXSAVE)
        return monitor.WriteFPR(m_thread.GetID(), &m_fpr.xstate.fxsave, sizeof(m_fpr.xstate.fxsave));

    if (m_fpr_type == eXSAVE)
        return monitor.WriteRegisterSet(m_thread.GetID(), &m_iovec, sizeof(m_fpr.xstate.xsave), NT_X86_XSTATE);
    return false;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::ReadRegister(const unsigned reg,
                                                        RegisterValue &value)
{
    ProcessMonitor &monitor = GetMonitor();
    return monitor.ReadRegisterValue(m_thread.GetID(),
                                     GetRegisterOffset(reg),
                                     GetRegisterName(reg),
                                     GetRegisterSize(reg),
                                     value);
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::WriteRegister(const unsigned reg,
                                                         const RegisterValue &value)
{
    ProcessMonitor &monitor = GetMonitor();
    return monitor.WriteRegisterValue(m_thread.GetID(),
                                      GetRegisterOffset(reg),
                                      GetRegisterName(reg),
                                      value);
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::ReadRegister(const RegisterInfo *reg_info, RegisterValue &value)
{
    if (!reg_info)
        return false;

    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];

    if (IsFPR(reg, GetFPRType()))
    {
        if (!ReadFPR())
            return false;
    }
    else
    {
        bool success = ReadRegister(reg, value);

        // If an i386 register should be parsed from an x86_64 register...
        if (success && reg >= k_first_i386 && reg <= k_last_i386)
            if (value.GetByteSize() > reg_info->byte_size)
                value.SetType(reg_info); // ...use the type specified by reg_info rather than the uint64_t default
        return success; 
    }

    if (reg_info->encoding == eEncodingVector)
    {
        ByteOrder byte_order = GetByteOrder();

        if (byte_order != ByteOrder::eByteOrderInvalid)
        {
            if (reg >= fpu_stmm0 && reg <= fpu_stmm7)
               value.SetBytes(m_fpr.xstate.fxsave.stmm[reg - fpu_stmm0].bytes, reg_info->byte_size, byte_order);
            if (reg >= fpu_xmm0 && reg <= fpu_xmm15)
                value.SetBytes(m_fpr.xstate.fxsave.xmm[reg - fpu_xmm0].bytes, reg_info->byte_size, byte_order);
            if (reg >= fpu_ymm0 && reg <= fpu_ymm15)
            {
                // Concatenate ymm using the register halves in xmm.bytes and ymmh.bytes
                if (GetFPRType() == eXSAVE && CopyXSTATEtoYMM(reg, byte_order))
                    value.SetBytes(m_ymm_set.ymm[reg - fpu_ymm0].bytes, reg_info->byte_size, byte_order);
                else
                    return false;
            }
            return value.GetType() == RegisterValue::eTypeBytes;
        }
        return false;
    }

    // Note that lldb uses slightly different naming conventions from sys/user.h
    switch (reg)
    {
    default:
        return false;
    case fpu_dp:
        value = m_fpr.xstate.fxsave.dp;
        break;
    case fpu_fcw:
        value = m_fpr.xstate.fxsave.fcw;
        break;
    case fpu_fsw:
        value = m_fpr.xstate.fxsave.fsw;
        break;
    case fpu_ip:
        value = m_fpr.xstate.fxsave.ip;
        break;
    case fpu_fop:
        value = m_fpr.xstate.fxsave.fop;
        break;
    case fpu_ftw:
        value = m_fpr.xstate.fxsave.ftw;
        break;
    case fpu_mxcsr:
        value = m_fpr.xstate.fxsave.mxcsr;
        break;
    case fpu_mxcsrmask:
        value = m_fpr.xstate.fxsave.mxcsrmask;
        break;
    }
    return true;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::WriteRegister(const RegisterInfo *reg_info, const RegisterValue &value)
{
    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    if (IsGPR(reg))
        return WriteRegister(reg, value);

    if (IsFPR(reg, GetFPRType()))
    {
        switch (reg)
        {
        default:
            if (reg_info->encoding != eEncodingVector)
                return false;

            if (reg >= fpu_stmm0 && reg <= fpu_stmm7)
               ::memcpy (m_fpr.xstate.fxsave.stmm[reg - fpu_stmm0].bytes, value.GetBytes(), value.GetByteSize());
            
            if (reg >= fpu_xmm0 && reg <= fpu_xmm15)
               ::memcpy (m_fpr.xstate.fxsave.xmm[reg - fpu_xmm0].bytes, value.GetBytes(), value.GetByteSize());
            
            if (reg >= fpu_ymm0 && reg <= fpu_ymm15) {
               if (GetFPRType() != eXSAVE)
                   return false; // the target processor does not support AVX

               // Store ymm register content, and split into the register halves in xmm.bytes and ymmh.bytes
               ::memcpy (m_ymm_set.ymm[reg - fpu_ymm0].bytes, value.GetBytes(), value.GetByteSize());
               if (false == CopyYMMtoXSTATE(reg, GetByteOrder()))
                   return false;
            }
            break;
        case fpu_dp:
            m_fpr.xstate.fxsave.dp = value.GetAsUInt64();
            break;
        case fpu_fcw:
            m_fpr.xstate.fxsave.fcw = value.GetAsUInt16();
            break;
        case fpu_fsw:
            m_fpr.xstate.fxsave.fsw = value.GetAsUInt16();
            break;
        case fpu_ip:
            m_fpr.xstate.fxsave.ip = value.GetAsUInt64();
            break;
        case fpu_fop:
            m_fpr.xstate.fxsave.fop = value.GetAsUInt16();
            break;
        case fpu_ftw:
            m_fpr.xstate.fxsave.ftw = value.GetAsUInt16();
            break;
        case fpu_mxcsr:
            m_fpr.xstate.fxsave.mxcsr = value.GetAsUInt32();
            break;
        case fpu_mxcsrmask:
            m_fpr.xstate.fxsave.mxcsrmask = value.GetAsUInt32();
            break;
        }
        if (WriteFPR())
        {
            if (IsAVX(reg))
                return CopyYMMtoXSTATE(reg, GetByteOrder());
            return true;
        }
    }
    return false;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::ReadAllRegisterValues(DataBufferSP &data_sp)
{
    bool success = false;
    data_sp.reset (new DataBufferHeap (REG_CONTEXT_SIZE, 0));
    if (data_sp && ReadGPR () && ReadFPR ())
    {
        uint8_t *dst = data_sp->GetBytes();
        success = dst != 0;

        if (success)
        {
            ::memcpy (dst, &m_gpr, GetGPRSize());
            dst += GetGPRSize();
        }
        if (GetFPRType() == eFXSAVE)
            ::memcpy (dst, &m_fpr.xstate.fxsave, sizeof(m_fpr.xstate.fxsave));
        
        if (GetFPRType() == eXSAVE) {
            ByteOrder byte_order = GetByteOrder();

            // Assemble the YMM register content from the register halves.
            for (uint32_t reg = fpu_ymm0; success && reg <= fpu_ymm15; ++reg)
                success = CopyXSTATEtoYMM(reg, byte_order);

            if (success) {
                // Copy the extended register state including the assembled ymm registers.
                ::memcpy (dst, &m_fpr, sizeof(m_fpr));
            }
        }
    }
    return success;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::WriteAllRegisterValues(const DataBufferSP &data_sp)
{
    bool success = false;
    if (data_sp && data_sp->GetByteSize() == REG_CONTEXT_SIZE)
    {
        uint8_t *src = data_sp->GetBytes();
        if (src)
        {
            ::memcpy (&m_gpr, src, GetGPRSize());

            if (WriteGPR())
            {
                src += GetGPRSize();
                if (GetFPRType() == eFXSAVE)
                    ::memcpy (&m_fpr.xstate.fxsave, src, sizeof(m_fpr.xstate.fxsave));
                if (GetFPRType() == eXSAVE)
                    ::memcpy (&m_fpr.xstate.xsave, src, sizeof(m_fpr.xstate.xsave));

                success = WriteFPR();
                if (success)
                {
                    success = true;

                    if (GetFPRType() == eXSAVE)
                    {
                        ByteOrder byte_order = GetByteOrder();

                        // Parse the YMM register content from the register halves.
                        for (uint32_t reg = fpu_ymm0; success && reg <= fpu_ymm15; ++reg)
                            success = CopyYMMtoXSTATE(reg, byte_order);
                    }
                }
            }
        }
    }
    return success;
}

uint32_t
RegisterContextPOSIXProcessMonitor_x86_64::SetHardwareWatchpoint(addr_t addr, size_t size,
                                              bool read, bool write)
{
    const uint32_t num_hw_watchpoints = NumSupportedHardwareWatchpoints();
    uint32_t hw_index;

    for (hw_index = 0; hw_index < num_hw_watchpoints; ++hw_index)
    {
        if (IsWatchpointVacant(hw_index))
            return SetHardwareWatchpointWithIndex(addr, size,
                                                  read, write,
                                                  hw_index);
    }

    return LLDB_INVALID_INDEX32;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::ClearHardwareWatchpoint(uint32_t hw_index)
{
    if (hw_index < NumSupportedHardwareWatchpoints())
    {
        RegisterValue current_dr7_bits;

        if (ReadRegister(dr7, current_dr7_bits))
        {
            uint64_t new_dr7_bits = current_dr7_bits.GetAsUInt64() & ~(3 << (2*hw_index));

            if (WriteRegister(dr7, RegisterValue(new_dr7_bits)))
                return true;
        }
    }

    return false;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::HardwareSingleStep(bool enable)
{
    enum { TRACE_BIT = 0x100 };
    uint64_t rflags;

    if ((rflags = ReadRegisterAsUnsigned(gpr_rflags, -1UL)) == -1UL)
        return false;
    
    if (enable)
    {
        if (rflags & TRACE_BIT)
            return true;

        rflags |= TRACE_BIT;
    }
    else
    {
        if (!(rflags & TRACE_BIT))
            return false;

        rflags &= ~TRACE_BIT;
    }

    return WriteRegisterFromUnsigned(gpr_rflags, rflags);
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::UpdateAfterBreakpoint()
{
    // PC points one byte past the int3 responsible for the breakpoint.
    lldb::addr_t pc;

    if ((pc = GetPC()) == LLDB_INVALID_ADDRESS)
        return false;

    SetPC(pc - 1);
    return true;
}

unsigned
RegisterContextPOSIXProcessMonitor_x86_64::GetRegisterIndexFromOffset(unsigned offset)
{
    unsigned reg;
    for (reg = 0; reg < k_num_registers; reg++)
    {
        if (GetRegisterInfo()[reg].byte_offset == offset)
            break;
    }
    assert(reg < k_num_registers && "Invalid register offset.");
    return reg;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::IsWatchpointHit(uint32_t hw_index)
{
    bool is_hit = false;

    if (m_watchpoints_initialized == false)
    {    
        // Reset the debug status and debug control registers
        RegisterValue zero_bits = RegisterValue(uint64_t(0));
        if (!WriteRegister(dr6, zero_bits) || !WriteRegister(dr7, zero_bits))
            assert(false && "Could not initialize watchpoint registers");
        m_watchpoints_initialized = true;
    }    

    if (hw_index < NumSupportedHardwareWatchpoints())
    {    
        RegisterValue value;

        if (ReadRegister(dr6, value))
        {    
            uint64_t val = value.GetAsUInt64();
            is_hit = val & (1 << hw_index);
        }    
    }    

    return is_hit;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::ClearWatchpointHits()
{
    return WriteRegister(dr6, RegisterValue((uint64_t)0));
}

addr_t
RegisterContextPOSIXProcessMonitor_x86_64::GetWatchpointAddress(uint32_t hw_index)
{
    addr_t wp_monitor_addr = LLDB_INVALID_ADDRESS;

    if (hw_index < NumSupportedHardwareWatchpoints())
    {
        if (!IsWatchpointVacant(hw_index))
        {
            RegisterValue value;

            if (ReadRegister(dr0 + hw_index, value))
                wp_monitor_addr = value.GetAsUInt64();
        }
    }

    return wp_monitor_addr;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::IsWatchpointVacant(uint32_t hw_index)
{
    bool is_vacant = false;
    RegisterValue value;

    assert(hw_index < NumSupportedHardwareWatchpoints());

    if (m_watchpoints_initialized == false)
    {
        // Reset the debug status and debug control registers
        RegisterValue zero_bits = RegisterValue(uint64_t(0));
        if (!WriteRegister(dr6, zero_bits) || !WriteRegister(dr7, zero_bits))
            assert(false && "Could not initialize watchpoint registers");
        m_watchpoints_initialized = true;
    }

    if (ReadRegister(dr7, value))
    {
        uint64_t val = value.GetAsUInt64();
        is_vacant = (val & (3 << 2*hw_index)) == 0;
    }

    return is_vacant;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::SetHardwareWatchpointWithIndex(addr_t addr, size_t size,
                                                       bool read, bool write,
                                                       uint32_t hw_index)
{
    const uint32_t num_hw_watchpoints = NumSupportedHardwareWatchpoints();

    if (num_hw_watchpoints == 0 || hw_index >= num_hw_watchpoints)
        return false;

    if (!(size == 1 || size == 2 || size == 4 || size == 8))
        return false;

    if (read == false && write == false)
        return false;

    if (!IsWatchpointVacant(hw_index))
        return false;

    // Set both dr7 (debug control register) and dri (debug address register).

    // dr7{7-0} encodes the local/gloabl enable bits:
    //  global enable --. .-- local enable
    //                  | |
    //                  v v
    //      dr0 -> bits{1-0}
    //      dr1 -> bits{3-2}
    //      dr2 -> bits{5-4}
    //      dr3 -> bits{7-6}
    //
    // dr7{31-16} encodes the rw/len bits:
    //  b_x+3, b_x+2, b_x+1, b_x
    //      where bits{x+1, x} => rw
    //            0b00: execute, 0b01: write, 0b11: read-or-write,
    //            0b10: io read-or-write (unused)
    //      and bits{x+3, x+2} => len
    //            0b00: 1-byte, 0b01: 2-byte, 0b11: 4-byte, 0b10: 8-byte
    //
    //      dr0 -> bits{19-16}
    //      dr1 -> bits{23-20}
    //      dr2 -> bits{27-24}
    //      dr3 -> bits{31-28}
    if (hw_index < num_hw_watchpoints)
    {
        RegisterValue current_dr7_bits;

        if (ReadRegister(dr7, current_dr7_bits))
        {
            uint64_t new_dr7_bits = current_dr7_bits.GetAsUInt64() |
                                    (1 << (2*hw_index) |
                                    size_and_rw_bits(size, read, write) <<
                                    (16+4*hw_index));

            if (WriteRegister(dr0 + hw_index, RegisterValue(addr)) &&
                WriteRegister(dr7, RegisterValue(new_dr7_bits)))
                return true;
        }
    }

    return false;
}

uint32_t
RegisterContextPOSIXProcessMonitor_x86_64::NumSupportedHardwareWatchpoints()
{
    // Available debug address registers: dr0, dr1, dr2, dr3
    return 4;
}
