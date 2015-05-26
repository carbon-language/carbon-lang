//===-- NativeRegisterContextLinux_arm.cpp --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(__arm__)

#include "NativeRegisterContextLinux_arm.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/RegisterValue.h"

#include "Plugins/Process/Utility/RegisterContextLinux_arm.h"

#define REG_CONTEXT_SIZE (GetGPRSize() + sizeof (m_fpr))

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

// arm general purpose registers.
static const uint32_t g_gpr_regnums_arm[] =
{
    gpr_r0_arm,
    gpr_r1_arm,
    gpr_r2_arm,
    gpr_r3_arm,
    gpr_r4_arm,
    gpr_r5_arm,
    gpr_r6_arm,
    gpr_r7_arm,
    gpr_r8_arm,
    gpr_r9_arm,
    gpr_r10_arm,
    gpr_r11_arm,
    gpr_r12_arm,
    gpr_sp_arm,
    gpr_lr_arm,
    gpr_pc_arm,
    gpr_cpsr_arm,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert(((sizeof g_gpr_regnums_arm / sizeof g_gpr_regnums_arm[0]) - 1) == k_num_gpr_registers_arm, \
              "g_gpr_regnums_arm has wrong number of register infos");

// arm floating point registers.
static const uint32_t g_fpu_regnums_arm[] =
{
    fpu_s0_arm,
    fpu_s1_arm,
    fpu_s2_arm,
    fpu_s3_arm,
    fpu_s4_arm,
    fpu_s5_arm,
    fpu_s6_arm,
    fpu_s7_arm,
    fpu_s8_arm,
    fpu_s9_arm,
    fpu_s10_arm,
    fpu_s11_arm,
    fpu_s12_arm,
    fpu_s13_arm,
    fpu_s14_arm,
    fpu_s15_arm,
    fpu_s16_arm,
    fpu_s17_arm,
    fpu_s18_arm,
    fpu_s19_arm,
    fpu_s20_arm,
    fpu_s21_arm,
    fpu_s22_arm,
    fpu_s23_arm,
    fpu_s24_arm,
    fpu_s25_arm,
    fpu_s26_arm,
    fpu_s27_arm,
    fpu_s28_arm,
    fpu_s29_arm,
    fpu_s30_arm,
    fpu_s31_arm,
    fpu_fpscr_arm,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert(((sizeof g_fpu_regnums_arm / sizeof g_fpu_regnums_arm[0]) - 1) == k_num_fpr_registers_arm, \
              "g_fpu_regnums_arm has wrong number of register infos");

namespace {
    // Number of register sets provided by this context.
    enum
    {
        k_num_register_sets = 2
    };
}

// Register sets for arm.
static const RegisterSet
g_reg_sets_arm[k_num_register_sets] =
{
    { "General Purpose Registers",  "gpr", k_num_gpr_registers_arm, g_gpr_regnums_arm },
    { "Floating Point Registers",   "fpu", k_num_fpr_registers_arm, g_fpu_regnums_arm }
};

NativeRegisterContextLinux*
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(const ArchSpec& target_arch,
                                                                 NativeThreadProtocol &native_thread,
                                                                 uint32_t concrete_frame_idx)
{
    return new NativeRegisterContextLinux_arm(target_arch, native_thread, concrete_frame_idx);
}

NativeRegisterContextLinux_arm::NativeRegisterContextLinux_arm (const ArchSpec& target_arch,
                                                                NativeThreadProtocol &native_thread,
                                                                uint32_t concrete_frame_idx) :
    NativeRegisterContextLinux (native_thread, concrete_frame_idx, new RegisterContextLinux_arm(target_arch))
{
    switch (target_arch.GetMachine())
    {
        case llvm::Triple::arm:
            m_reg_info.num_registers     = k_num_registers_arm;
            m_reg_info.num_gpr_registers = k_num_gpr_registers_arm;
            m_reg_info.num_fpr_registers = k_num_fpr_registers_arm;
            m_reg_info.last_gpr          = k_last_gpr_arm;
            m_reg_info.first_fpr         = k_first_fpr_arm;
            m_reg_info.last_fpr          = k_last_fpr_arm;
            m_reg_info.first_fpr_v       = fpu_s0_arm;
            m_reg_info.last_fpr_v        = fpu_s31_arm;
            m_reg_info.gpr_flags         = gpr_cpsr_arm;
            break;
        default:
            assert(false && "Unhandled target architecture.");
            break;
    }

    ::memset(&m_fpr, 0, sizeof (m_fpr));
    ::memset(&m_gpr_arm, 0, sizeof (m_gpr_arm));
}

uint32_t
NativeRegisterContextLinux_arm::GetRegisterSetCount () const
{
    return k_num_register_sets;
}

uint32_t
NativeRegisterContextLinux_arm::GetUserRegisterCount() const
{
    uint32_t count = 0;
    for (uint32_t set_index = 0; set_index < k_num_register_sets; ++set_index)
        count += g_reg_sets_arm[set_index].num_registers;
    return count;
}

const RegisterSet *
NativeRegisterContextLinux_arm::GetRegisterSet (uint32_t set_index) const
{
    if (set_index < k_num_register_sets)
        return &g_reg_sets_arm[set_index];

    return nullptr;
}

Error
NativeRegisterContextLinux_arm::ReadRegister (const RegisterInfo *reg_info, RegisterValue &reg_value)
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
        error = ReadFPR();
        if (error.Fail())
            return error;
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
NativeRegisterContextLinux_arm::WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value)
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

        Error error = WriteFPR();
        if (error.Fail())
            return error;

        return Error ();
    }

    return Error ("failed - register wasn't recognized to be a GPR or an FPR, write strategy unknown");
}

Error
NativeRegisterContextLinux_arm::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    Error error;

    data_sp.reset (new DataBufferHeap (REG_CONTEXT_SIZE, 0));
    if (!data_sp)
        return Error ("failed to allocate DataBufferHeap instance of size %" PRIu64, (uint64_t)REG_CONTEXT_SIZE);

    error = ReadGPR();
    if (error.Fail())
        return error;

    error = ReadFPR();
    if (error.Fail())
        return error;

    uint8_t *dst = data_sp->GetBytes ();
    if (dst == nullptr)
    {
        error.SetErrorStringWithFormat ("DataBufferHeap instance of size %" PRIu64 " returned a null pointer", (uint64_t)REG_CONTEXT_SIZE);
        return error;
    }

    ::memcpy (dst, &m_gpr_arm, GetGPRSize());
    dst += GetGPRSize();
    ::memcpy (dst, &m_fpr, sizeof(m_fpr));

    return error;
}

Error
NativeRegisterContextLinux_arm::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
{
    Error error;

    if (!data_sp)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s invalid data_sp provided", __FUNCTION__);
        return error;
    }

    if (data_sp->GetByteSize () != REG_CONTEXT_SIZE)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s data_sp contained mismatched data size, expected %" PRIu64 ", actual %" PRIu64, __FUNCTION__, (uint64_t)REG_CONTEXT_SIZE, data_sp->GetByteSize ());
        return error;
    }


    uint8_t *src = data_sp->GetBytes ();
    if (src == nullptr)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s DataBuffer::GetBytes() returned a null pointer", __FUNCTION__);
        return error;
    }
    ::memcpy (&m_gpr_arm, src, GetRegisterInfoInterface ().GetGPRSize ());

    error = WriteGPR();
    if (error.Fail())
        return error;

    src += GetRegisterInfoInterface ().GetGPRSize ();
    ::memcpy (&m_fpr, src, sizeof(m_fpr));

    error = WriteFPR();
    if (error.Fail())
        return error;

    return error;
}

bool
NativeRegisterContextLinux_arm::IsGPR(unsigned reg) const
{
    return reg <= m_reg_info.last_gpr;   // GPR's come first.
}

bool
NativeRegisterContextLinux_arm::IsFPR(unsigned reg) const
{
    return (m_reg_info.first_fpr <= reg && reg <= m_reg_info.last_fpr);
}

#endif // defined(__arm__)
