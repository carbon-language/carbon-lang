//===-- NativeRegisterContextLinux_arm64.cpp --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextLinux_arm64.h"

#include "lldb/lldb-private-forward.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "Plugins/Process/Linux/NativeProcessLinux.h"

#define REG_CONTEXT_SIZE (GetGPRSize() + sizeof (m_fpr))

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

// ARM64 general purpose registers.
static const uint32_t g_gpr_regnums_arm64[] =
{
    gpr_x0_arm64,
    gpr_x1_arm64,
    gpr_x2_arm64,
    gpr_x3_arm64,
    gpr_x4_arm64,
    gpr_x5_arm64,
    gpr_x6_arm64,
    gpr_x7_arm64,
    gpr_x8_arm64,
    gpr_x9_arm64,
    gpr_x10_arm64,
    gpr_x11_arm64,
    gpr_x12_arm64,
    gpr_x13_arm64,
    gpr_x14_arm64,
    gpr_x15_arm64,
    gpr_x16_arm64,
    gpr_x17_arm64,
    gpr_x18_arm64,
    gpr_x19_arm64,
    gpr_x20_arm64,
    gpr_x21_arm64,
    gpr_x22_arm64,
    gpr_x23_arm64,
    gpr_x24_arm64,
    gpr_x25_arm64,
    gpr_x26_arm64,
    gpr_x27_arm64,
    gpr_x28_arm64,
    gpr_fp_arm64,
    gpr_lr_arm64,
    gpr_sp_arm64,
    gpr_pc_arm64,
    gpr_cpsr_arm64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert(((sizeof g_gpr_regnums_arm64 / sizeof g_gpr_regnums_arm64[0]) - 1) == k_num_gpr_registers_arm64, \
              "g_gpr_regnums_arm64 has wrong number of register infos");

// ARM64 floating point registers.
static const uint32_t g_fpu_regnums_arm64[] =
{
    fpu_v0_arm64,
    fpu_v1_arm64,
    fpu_v2_arm64,
    fpu_v3_arm64,
    fpu_v4_arm64,
    fpu_v5_arm64,
    fpu_v6_arm64,
    fpu_v7_arm64,
    fpu_v8_arm64,
    fpu_v9_arm64,
    fpu_v10_arm64,
    fpu_v11_arm64,
    fpu_v12_arm64,
    fpu_v13_arm64,
    fpu_v14_arm64,
    fpu_v15_arm64,
    fpu_v16_arm64,
    fpu_v17_arm64,
    fpu_v18_arm64,
    fpu_v19_arm64,
    fpu_v20_arm64,
    fpu_v21_arm64,
    fpu_v22_arm64,
    fpu_v23_arm64,
    fpu_v24_arm64,
    fpu_v25_arm64,
    fpu_v26_arm64,
    fpu_v27_arm64,
    fpu_v28_arm64,
    fpu_v29_arm64,
    fpu_v30_arm64,
    fpu_v31_arm64,
    fpu_fpsr_arm64,
    fpu_fpcr_arm64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert(((sizeof g_fpu_regnums_arm64 / sizeof g_fpu_regnums_arm64[0]) - 1) == k_num_fpr_registers_arm64, \
              "g_fpu_regnums_arm64 has wrong number of register infos");

namespace {
    // Number of register sets provided by this context.
    enum
    {
        k_num_register_sets = 2
    };
}

// Register sets for ARM64.
static const RegisterSet
g_reg_sets_arm64[k_num_register_sets] =
{
    { "General Purpose Registers",  "gpr", k_num_gpr_registers_arm64, g_gpr_regnums_arm64 },
    { "Floating Point Registers",   "fpu", k_num_fpr_registers_arm64, g_fpu_regnums_arm64 }
};

NativeRegisterContextLinux_arm64::NativeRegisterContextLinux_arm64 (
        NativeThreadProtocol &native_thread,
        uint32_t concrete_frame_idx,
        RegisterInfoInterface *reg_info_interface_p) :
    NativeRegisterContextRegisterInfo (native_thread, concrete_frame_idx, reg_info_interface_p)
{
    switch (reg_info_interface_p->m_target_arch.GetMachine())
    {
        case llvm::Triple::aarch64:
            m_reg_info.num_registers     = k_num_registers_arm64;
            m_reg_info.num_gpr_registers = k_num_gpr_registers_arm64;
            m_reg_info.num_fpr_registers = k_num_fpr_registers_arm64;
            m_reg_info.last_gpr          = k_last_gpr_arm64;
            m_reg_info.first_fpr         = k_first_fpr_arm64;
            m_reg_info.last_fpr          = k_last_fpr_arm64;
            m_reg_info.first_fpr_v       = fpu_v0_arm64;
            m_reg_info.last_fpr_v        = fpu_v31_arm64;
            m_reg_info.gpr_flags         = gpr_cpsr_arm64;
            break;
        default:
            assert(false && "Unhandled target architecture.");
            break;
    }

    ::memset(&m_fpr, 0, sizeof (m_fpr));
    ::memset(&m_gpr_arm64, 0, sizeof (m_gpr_arm64));
}

uint32_t
NativeRegisterContextLinux_arm64::GetRegisterSetCount () const
{
    return k_num_register_sets;
}

const RegisterSet *
NativeRegisterContextLinux_arm64::GetRegisterSet (uint32_t set_index) const
{
    if (set_index < k_num_register_sets)
        return &g_reg_sets_arm64[set_index];

    return nullptr;
}

Error
NativeRegisterContextLinux_arm64::ReadRegister (const RegisterInfo *reg_info, RegisterValue &reg_value)
{
    Error error;

    if (!reg_info)
    {
        error.SetErrorString ("reg_info NULL");
        return error;
    }

    const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

    if (IsFPR(reg))
    {
        if (!ReadFPR())
        {
            error.SetErrorString ("failed to read floating point register");
            return error;
        }
    }
    else
    {
        uint32_t full_reg = reg;
        bool is_subreg = reg_info->invalidate_regs && (reg_info->invalidate_regs[0] != LLDB_INVALID_REGNUM);

        if (is_subreg)
        {
            // Read the full aligned 64-bit register.
            full_reg = reg_info->invalidate_regs[0];
        }

        error = ReadRegisterRaw(full_reg, reg_value);

        if (error.Success ())
        {
            // If our read was not aligned (for ah,bh,ch,dh), shift our returned value one byte to the right.
            if (is_subreg && (reg_info->byte_offset & 0x1))
                reg_value.SetUInt64(reg_value.GetAsUInt64() >> 8);

            // If our return byte size was greater than the return value reg size, then
            // use the type specified by reg_info rather than the uint64_t default
            if (reg_value.GetByteSize() > reg_info->byte_size)
                reg_value.SetType(reg_info);
        }
        return error;
    }

    // Get pointer to m_fpr variable and set the data from it.
    assert (reg_info->byte_offset < sizeof m_fpr);
    uint8_t *src = (uint8_t *)&m_fpr + reg_info->byte_offset;
    switch (reg_info->byte_size)
    {
        case 2:
            reg_value.SetUInt16(*(uint16_t *)src);
            break;
        case 4:
            reg_value.SetUInt32(*(uint32_t *)src);
            break;
        case 8:
            reg_value.SetUInt64(*(uint64_t *)src);
            break;
        default:
            assert(false && "Unhandled data size.");
            error.SetErrorStringWithFormat ("unhandled byte size: %" PRIu32, reg_info->byte_size);
            break;
    }

    return error;
}

Error
NativeRegisterContextLinux_arm64::WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value)
{
    if (!reg_info)
        return Error ("reg_info NULL");

    const uint32_t reg_index = reg_info->kinds[lldb::eRegisterKindLLDB];
    if (reg_index == LLDB_INVALID_REGNUM)
        return Error ("no lldb regnum for %s", reg_info && reg_info->name ? reg_info->name : "<unknown register>");

    if (IsGPR(reg_index))
        return WriteRegisterRaw(reg_index, reg_value);

    if (IsFPR(reg_index))
    {
        // Get pointer to m_fpr variable and set the data to it.
        assert (reg_info->byte_offset < sizeof(m_fpr));
        uint8_t *dst = (uint8_t *)&m_fpr + reg_info->byte_offset;
        switch (reg_info->byte_size)
        {
            case 2:
                *(uint16_t *)dst = reg_value.GetAsUInt16();
                break;
            case 4:
                *(uint32_t *)dst = reg_value.GetAsUInt32();
                break;
            case 8:
                *(uint64_t *)dst = reg_value.GetAsUInt64();
                break;
            default:
                assert(false && "Unhandled data size.");
                return Error ("unhandled register data size %" PRIu32, reg_info->byte_size);
        }

        if (!WriteFPR())
        {
            return Error ("NativeRegisterContextLinux_arm64::WriteRegister: WriteFPR failed");
        }

        return Error ();
    }

    return Error ("failed - register wasn't recognized to be a GPR or an FPR, write strategy unknown");
}

Error
NativeRegisterContextLinux_arm64::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    Error error;

    data_sp.reset (new DataBufferHeap (REG_CONTEXT_SIZE, 0));
    if (!data_sp)
        return Error ("failed to allocate DataBufferHeap instance of size %" PRIu64, REG_CONTEXT_SIZE);

    if (!ReadGPR ())
    {
        error.SetErrorString ("ReadGPR() failed");
        return error;
    }

    if (!ReadFPR ())
    {
        error.SetErrorString ("ReadFPR() failed");
        return error;
    }

    uint8_t *dst = data_sp->GetBytes ();
    if (dst == nullptr)
    {
        error.SetErrorStringWithFormat ("DataBufferHeap instance of size %" PRIu64 " returned a null pointer", REG_CONTEXT_SIZE);
        return error;
    }

    ::memcpy (dst, &m_gpr_arm64, GetGPRSize());
    dst += GetGPRSize();
    ::memcpy (dst, &m_fpr, sizeof(m_fpr));

    return error;
}

Error
NativeRegisterContextLinux_arm64::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
{
    Error error;

    if (!data_sp)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s invalid data_sp provided", __FUNCTION__);
        return error;
    }

    if (data_sp->GetByteSize () != REG_CONTEXT_SIZE)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s data_sp contained mismatched data size, expected %" PRIu64 ", actual %" PRIu64, __FUNCTION__, REG_CONTEXT_SIZE, data_sp->GetByteSize ());
        return error;
    }


    uint8_t *src = data_sp->GetBytes ();
    if (src == nullptr)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s DataBuffer::GetBytes() returned a null pointer", __FUNCTION__);
        return error;
    }
    ::memcpy (&m_gpr_arm64, src, GetRegisterInfoInterface ().GetGPRSize ());

    if (!WriteGPR ())
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s WriteGPR() failed", __FUNCTION__);
        return error;
    }

    src += GetRegisterInfoInterface ().GetGPRSize ();
    ::memcpy (&m_fpr, src, sizeof(m_fpr));

    if (!WriteFPR ())
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s WriteFPR() failed", __FUNCTION__);
        return error;
    }

    return error;
}

Error
NativeRegisterContextLinux_arm64::WriteRegisterRaw (uint32_t reg_index, const RegisterValue &reg_value)
{
    Error error;

    uint32_t reg_to_write = reg_index;
    RegisterValue value_to_write = reg_value;

    // Check if this is a subregister of a full register.
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex(reg_index);
    if (reg_info->invalidate_regs && (reg_info->invalidate_regs[0] != LLDB_INVALID_REGNUM))
    {
        RegisterValue full_value;
        uint32_t full_reg = reg_info->invalidate_regs[0];
        const RegisterInfo *full_reg_info = GetRegisterInfoAtIndex(full_reg);

        // Read the full register.
        error = ReadRegister(full_reg_info, full_value);
        if (error.Fail ())
            return error;

        lldb::ByteOrder byte_order = GetByteOrder();
        uint8_t dst[RegisterValue::kMaxRegisterByteSize];

        // Get the bytes for the full register.
        const uint32_t dest_size = full_value.GetAsMemoryData (full_reg_info,
                                                               dst,
                                                               sizeof(dst),
                                                               byte_order,
                                                               error);
        if (error.Success() && dest_size)
        {
            uint8_t src[RegisterValue::kMaxRegisterByteSize];

            // Get the bytes for the source data.
            const uint32_t src_size = reg_value.GetAsMemoryData (reg_info, src, sizeof(src), byte_order, error);
            if (error.Success() && src_size && (src_size < dest_size))
            {
                // Copy the src bytes to the destination.
                memcpy (dst + (reg_info->byte_offset & 0x1), src, src_size);
                // Set this full register as the value to write.
                value_to_write.SetBytes(dst, full_value.GetByteSize(), byte_order);
                value_to_write.SetType(full_reg_info);
                reg_to_write = full_reg;
            }
        }
    }

    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
    {
        error.SetErrorString ("NativeProcessProtocol is NULL");
        return error;
    }

    const RegisterInfo *const register_to_write_info_p = GetRegisterInfoAtIndex (reg_to_write);
    assert (register_to_write_info_p && "register to write does not have valid RegisterInfo");
    if (!register_to_write_info_p)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_arm64::%s failed to get RegisterInfo for write register index %" PRIu32, __FUNCTION__, reg_to_write);
        return error;
    }

    NativeProcessLinux *const process_p = reinterpret_cast<NativeProcessLinux*> (process_sp.get ());
    return process_p->WriteRegisterValue(m_thread.GetID(),
                                         register_to_write_info_p->byte_offset,
                                         register_to_write_info_p->name,
                                         value_to_write);
}

Error
NativeRegisterContextLinux_arm64::ReadRegisterRaw (uint32_t reg_index, RegisterValue &reg_value)
{
    Error error;
    const RegisterInfo *const reg_info = GetRegisterInfoAtIndex (reg_index);
    if (!reg_info)
    {
        error.SetErrorStringWithFormat ("register %" PRIu32 " not found", reg_index);
        return error;
    }

    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
    {
        error.SetErrorString ("NativeProcessProtocol is NULL");
        return error;
    }

    NativeProcessLinux *const process_p = reinterpret_cast<NativeProcessLinux*> (process_sp.get ());
    return process_p->ReadRegisterValue(m_thread.GetID(),
                                        reg_info->byte_offset,
                                        reg_info->name,
                                        reg_info->byte_size,
                                        reg_value);
}

bool
NativeRegisterContextLinux_arm64::IsGPR(unsigned reg) const
{
    return reg <= m_reg_info.last_gpr;   // GPR's come first.
}

bool
NativeRegisterContextLinux_arm64::ReadGPR()
{
    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return false;
    NativeProcessLinux *const process_p = reinterpret_cast<NativeProcessLinux*> (process_sp.get ());

    return process_p->ReadGPR (m_thread.GetID (), &m_gpr_arm64, GetRegisterInfoInterface ().GetGPRSize ()).Success();
}

bool
NativeRegisterContextLinux_arm64::WriteGPR()
{
    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return false;
    NativeProcessLinux *const process_p = reinterpret_cast<NativeProcessLinux*> (process_sp.get ());

    return process_p->WriteGPR (m_thread.GetID (), &m_gpr_arm64, GetRegisterInfoInterface ().GetGPRSize ()).Success();
}

bool
NativeRegisterContextLinux_arm64::IsFPR(unsigned reg) const
{
    return (m_reg_info.first_fpr <= reg && reg <= m_reg_info.last_fpr);
}

bool
NativeRegisterContextLinux_arm64::ReadFPR ()
{
    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return false;

    NativeProcessLinux *const process_p = reinterpret_cast<NativeProcessLinux*> (process_sp.get ());
    return process_p->ReadFPR (m_thread.GetID (), &m_fpr, sizeof (m_fpr)).Success();
}

bool
NativeRegisterContextLinux_arm64::WriteFPR ()
{
    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return false;

    NativeProcessLinux *const process_p = reinterpret_cast<NativeProcessLinux*> (process_sp.get ());
    return process_p->WriteFPR (m_thread.GetID (), &m_fpr, sizeof (m_fpr)).Success();
}

lldb::ByteOrder
NativeRegisterContextLinux_arm64::GetByteOrder() const
{
    // Get the target process whose privileged thread was used for the register read.
    lldb::ByteOrder byte_order = lldb::eByteOrderInvalid;

    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return byte_order;

    if (!process_sp->GetByteOrder (byte_order))
    {
        // FIXME log here
    }

    return byte_order;
}

size_t
NativeRegisterContextLinux_arm64::GetGPRSize() const
{
    return GetRegisterInfoInterface().GetGPRSize();
}
