//===-- NativeRegisterContextLinux_x86_64.cpp ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextLinux_x86_64.h"

#include "lldb/lldb-private-forward.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "Plugins/Process/Linux/NativeProcessLinux.h"

using namespace lldb_private;

// ----------------------------------------------------------------------------
// Private namespace.
// ----------------------------------------------------------------------------

namespace
{
    // x86 32-bit general purpose registers.
    const uint32_t
    g_gpr_regnums_i386[] =
    {
        lldb_eax_i386,
        lldb_ebx_i386,
        lldb_ecx_i386,
        lldb_edx_i386,
        lldb_edi_i386,
        lldb_esi_i386,
        lldb_ebp_i386,
        lldb_esp_i386,
        lldb_eip_i386,
        lldb_eflags_i386,
        lldb_cs_i386,
        lldb_fs_i386,
        lldb_gs_i386,
        lldb_ss_i386,
        lldb_ds_i386,
        lldb_es_i386,
        lldb_ax_i386,
        lldb_bx_i386,
        lldb_cx_i386,
        lldb_dx_i386,
        lldb_di_i386,
        lldb_si_i386,
        lldb_bp_i386,
        lldb_sp_i386,
        lldb_ah_i386,
        lldb_bh_i386,
        lldb_ch_i386,
        lldb_dh_i386,
        lldb_al_i386,
        lldb_bl_i386,
        lldb_cl_i386,
        lldb_dl_i386,
        LLDB_INVALID_REGNUM // register sets need to end with this flag
    };
    static_assert((sizeof(g_gpr_regnums_i386) / sizeof(g_gpr_regnums_i386[0])) - 1 == k_num_gpr_registers_i386,
                  "g_gpr_regnums_i386 has wrong number of register infos");

    // x86 32-bit floating point registers.
    const uint32_t
    g_fpu_regnums_i386[] =
    {
        lldb_fctrl_i386,
        lldb_fstat_i386,
        lldb_ftag_i386,
        lldb_fop_i386,
        lldb_fiseg_i386,
        lldb_fioff_i386,
        lldb_foseg_i386,
        lldb_fooff_i386,
        lldb_mxcsr_i386,
        lldb_mxcsrmask_i386,
        lldb_st0_i386,
        lldb_st1_i386,
        lldb_st2_i386,
        lldb_st3_i386,
        lldb_st4_i386,
        lldb_st5_i386,
        lldb_st6_i386,
        lldb_st7_i386,
        lldb_mm0_i386,
        lldb_mm1_i386,
        lldb_mm2_i386,
        lldb_mm3_i386,
        lldb_mm4_i386,
        lldb_mm5_i386,
        lldb_mm6_i386,
        lldb_mm7_i386,
        lldb_xmm0_i386,
        lldb_xmm1_i386,
        lldb_xmm2_i386,
        lldb_xmm3_i386,
        lldb_xmm4_i386,
        lldb_xmm5_i386,
        lldb_xmm6_i386,
        lldb_xmm7_i386,
        LLDB_INVALID_REGNUM // register sets need to end with this flag
    };
    static_assert((sizeof(g_fpu_regnums_i386) / sizeof(g_fpu_regnums_i386[0])) - 1 == k_num_fpr_registers_i386,
                  "g_fpu_regnums_i386 has wrong number of register infos");

    // x86 32-bit AVX registers.
    const uint32_t
    g_avx_regnums_i386[] =
    {
        lldb_ymm0_i386,
        lldb_ymm1_i386,
        lldb_ymm2_i386,
        lldb_ymm3_i386,
        lldb_ymm4_i386,
        lldb_ymm5_i386,
        lldb_ymm6_i386,
        lldb_ymm7_i386,
        LLDB_INVALID_REGNUM // register sets need to end with this flag
    };
    static_assert((sizeof(g_avx_regnums_i386) / sizeof(g_avx_regnums_i386[0])) - 1 == k_num_avx_registers_i386,
                  " g_avx_regnums_i386 has wrong number of register infos");

    // x86 64-bit general purpose registers.
    static const
    uint32_t g_gpr_regnums_x86_64[] =
    {
        lldb_rax_x86_64,
        lldb_rbx_x86_64,
        lldb_rcx_x86_64,
        lldb_rdx_x86_64,
        lldb_rdi_x86_64,
        lldb_rsi_x86_64,
        lldb_rbp_x86_64,
        lldb_rsp_x86_64,
        lldb_r8_x86_64,
        lldb_r9_x86_64,
        lldb_r10_x86_64,
        lldb_r11_x86_64,
        lldb_r12_x86_64,
        lldb_r13_x86_64,
        lldb_r14_x86_64,
        lldb_r15_x86_64,
        lldb_rip_x86_64,
        lldb_rflags_x86_64,
        lldb_cs_x86_64,
        lldb_fs_x86_64,
        lldb_gs_x86_64,
        lldb_ss_x86_64,
        lldb_ds_x86_64,
        lldb_es_x86_64,
        lldb_eax_x86_64,
        lldb_ebx_x86_64,
        lldb_ecx_x86_64,
        lldb_edx_x86_64,
        lldb_edi_x86_64,
        lldb_esi_x86_64,
        lldb_ebp_x86_64,
        lldb_esp_x86_64,
        lldb_r8d_x86_64,    // Low 32 bits or r8
        lldb_r9d_x86_64,    // Low 32 bits or r9
        lldb_r10d_x86_64,   // Low 32 bits or r10
        lldb_r11d_x86_64,   // Low 32 bits or r11
        lldb_r12d_x86_64,   // Low 32 bits or r12
        lldb_r13d_x86_64,   // Low 32 bits or r13
        lldb_r14d_x86_64,   // Low 32 bits or r14
        lldb_r15d_x86_64,   // Low 32 bits or r15
        lldb_ax_x86_64,
        lldb_bx_x86_64,
        lldb_cx_x86_64,
        lldb_dx_x86_64,
        lldb_di_x86_64,
        lldb_si_x86_64,
        lldb_bp_x86_64,
        lldb_sp_x86_64,
        lldb_r8w_x86_64,    // Low 16 bits or r8
        lldb_r9w_x86_64,    // Low 16 bits or r9
        lldb_r10w_x86_64,   // Low 16 bits or r10
        lldb_r11w_x86_64,   // Low 16 bits or r11
        lldb_r12w_x86_64,   // Low 16 bits or r12
        lldb_r13w_x86_64,   // Low 16 bits or r13
        lldb_r14w_x86_64,   // Low 16 bits or r14
        lldb_r15w_x86_64,   // Low 16 bits or r15
        lldb_ah_x86_64,
        lldb_bh_x86_64,
        lldb_ch_x86_64,
        lldb_dh_x86_64,
        lldb_al_x86_64,
        lldb_bl_x86_64,
        lldb_cl_x86_64,
        lldb_dl_x86_64,
        lldb_dil_x86_64,
        lldb_sil_x86_64,
        lldb_bpl_x86_64,
        lldb_spl_x86_64,
        lldb_r8l_x86_64,    // Low 8 bits or r8
        lldb_r9l_x86_64,    // Low 8 bits or r9
        lldb_r10l_x86_64,   // Low 8 bits or r10
        lldb_r11l_x86_64,   // Low 8 bits or r11
        lldb_r12l_x86_64,   // Low 8 bits or r12
        lldb_r13l_x86_64,   // Low 8 bits or r13
        lldb_r14l_x86_64,   // Low 8 bits or r14
        lldb_r15l_x86_64,   // Low 8 bits or r15
        LLDB_INVALID_REGNUM // register sets need to end with this flag
    };
    static_assert((sizeof(g_gpr_regnums_x86_64) / sizeof(g_gpr_regnums_x86_64[0])) - 1 == k_num_gpr_registers_x86_64,
                  "g_gpr_regnums_x86_64 has wrong number of register infos");

    // x86 64-bit floating point registers.
    static const uint32_t
    g_fpu_regnums_x86_64[] =
    {
        lldb_fctrl_x86_64,
        lldb_fstat_x86_64,
        lldb_ftag_x86_64,
        lldb_fop_x86_64,
        lldb_fiseg_x86_64,
        lldb_fioff_x86_64,
        lldb_foseg_x86_64,
        lldb_fooff_x86_64,
        lldb_mxcsr_x86_64,
        lldb_mxcsrmask_x86_64,
        lldb_st0_x86_64,
        lldb_st1_x86_64,
        lldb_st2_x86_64,
        lldb_st3_x86_64,
        lldb_st4_x86_64,
        lldb_st5_x86_64,
        lldb_st6_x86_64,
        lldb_st7_x86_64,
        lldb_mm0_x86_64,
        lldb_mm1_x86_64,
        lldb_mm2_x86_64,
        lldb_mm3_x86_64,
        lldb_mm4_x86_64,
        lldb_mm5_x86_64,
        lldb_mm6_x86_64,
        lldb_mm7_x86_64,
        lldb_xmm0_x86_64,
        lldb_xmm1_x86_64,
        lldb_xmm2_x86_64,
        lldb_xmm3_x86_64,
        lldb_xmm4_x86_64,
        lldb_xmm5_x86_64,
        lldb_xmm6_x86_64,
        lldb_xmm7_x86_64,
        lldb_xmm8_x86_64,
        lldb_xmm9_x86_64,
        lldb_xmm10_x86_64,
        lldb_xmm11_x86_64,
        lldb_xmm12_x86_64,
        lldb_xmm13_x86_64,
        lldb_xmm14_x86_64,
        lldb_xmm15_x86_64,
        LLDB_INVALID_REGNUM // register sets need to end with this flag
    };
    static_assert((sizeof(g_fpu_regnums_x86_64) / sizeof(g_fpu_regnums_x86_64[0])) - 1 == k_num_fpr_registers_x86_64,
                  "g_fpu_regnums_x86_64 has wrong number of register infos");

    // x86 64-bit AVX registers.
    static const uint32_t
    g_avx_regnums_x86_64[] =
    {
        lldb_ymm0_x86_64,
        lldb_ymm1_x86_64,
        lldb_ymm2_x86_64,
        lldb_ymm3_x86_64,
        lldb_ymm4_x86_64,
        lldb_ymm5_x86_64,
        lldb_ymm6_x86_64,
        lldb_ymm7_x86_64,
        lldb_ymm8_x86_64,
        lldb_ymm9_x86_64,
        lldb_ymm10_x86_64,
        lldb_ymm11_x86_64,
        lldb_ymm12_x86_64,
        lldb_ymm13_x86_64,
        lldb_ymm14_x86_64,
        lldb_ymm15_x86_64,
        LLDB_INVALID_REGNUM // register sets need to end with this flag
    };
    static_assert((sizeof(g_avx_regnums_x86_64) / sizeof(g_avx_regnums_x86_64[0])) - 1 == k_num_avx_registers_x86_64,
                  "g_avx_regnums_x86_64 has wrong number of register infos");

    // Number of register sets provided by this context.
    enum
    {
        k_num_extended_register_sets = 1,
        k_num_register_sets = 3
    };

    // Register sets for x86 32-bit.
    static const RegisterSet
    g_reg_sets_i386[k_num_register_sets] =
    {
        { "General Purpose Registers",  "gpr", k_num_gpr_registers_i386, g_gpr_regnums_i386 },
        { "Floating Point Registers",   "fpu", k_num_fpr_registers_i386, g_fpu_regnums_i386 },
        { "Advanced Vector Extensions", "avx", k_num_avx_registers_i386, g_avx_regnums_i386 }
    };

    // Register sets for x86 64-bit.
    static const RegisterSet
    g_reg_sets_x86_64[k_num_register_sets] =
    {
        { "General Purpose Registers",  "gpr", k_num_gpr_registers_x86_64, g_gpr_regnums_x86_64 },
        { "Floating Point Registers",   "fpu", k_num_fpr_registers_x86_64, g_fpu_regnums_x86_64 },
        { "Advanced Vector Extensions", "avx", k_num_avx_registers_x86_64, g_avx_regnums_x86_64 }
    };
}

#define REG_CONTEXT_SIZE (GetRegisterInfoInterface ().GetGPRSize () + sizeof(FPR))

// ----------------------------------------------------------------------------
// Required ptrace defines.
// ----------------------------------------------------------------------------

// Support ptrace extensions even when compiled without required kernel support
#ifndef NT_X86_XSTATE
#define NT_X86_XSTATE 0x202
#endif

// ----------------------------------------------------------------------------
// NativeRegisterContextLinux_x86_64 members.
// ----------------------------------------------------------------------------

NativeRegisterContextLinux_x86_64::NativeRegisterContextLinux_x86_64 (NativeThreadProtocol &native_thread, uint32_t concrete_frame_idx, RegisterInfoInterface *reg_info_interface_p) :
    NativeRegisterContextRegisterInfo (native_thread, concrete_frame_idx, reg_info_interface_p),
    m_fpr_type (eFPRTypeNotValid),
    m_fpr (),
    m_iovec (),
    m_ymm_set (),
    m_reg_info (),
    m_gpr_x86_64 ()
{
    // Set up data about ranges of valid registers.
    switch (reg_info_interface_p->GetTargetArchitecture ().GetMachine ())
    {
        case llvm::Triple::x86:
            m_reg_info.num_registers        = k_num_registers_i386;
            m_reg_info.num_gpr_registers    = k_num_gpr_registers_i386;
            m_reg_info.num_fpr_registers    = k_num_fpr_registers_i386;
            m_reg_info.num_avx_registers    = k_num_avx_registers_i386;
            m_reg_info.last_gpr             = k_last_gpr_i386;
            m_reg_info.first_fpr            = k_first_fpr_i386;
            m_reg_info.last_fpr             = k_last_fpr_i386;
            m_reg_info.first_st             = lldb_st0_i386;
            m_reg_info.last_st              = lldb_st7_i386;
            m_reg_info.first_mm             = lldb_mm0_i386;
            m_reg_info.last_mm              = lldb_mm7_i386;
            m_reg_info.first_xmm            = lldb_xmm0_i386;
            m_reg_info.last_xmm             = lldb_xmm7_i386;
            m_reg_info.first_ymm            = lldb_ymm0_i386;
            m_reg_info.last_ymm             = lldb_ymm7_i386;
            m_reg_info.first_dr             = lldb_dr0_i386;
            m_reg_info.gpr_flags            = lldb_eflags_i386;
            break;
        case llvm::Triple::x86_64:
            m_reg_info.num_registers        = k_num_registers_x86_64;
            m_reg_info.num_gpr_registers    = k_num_gpr_registers_x86_64;
            m_reg_info.num_fpr_registers    = k_num_fpr_registers_x86_64;
            m_reg_info.num_avx_registers    = k_num_avx_registers_x86_64;
            m_reg_info.last_gpr             = k_last_gpr_x86_64;
            m_reg_info.first_fpr            = k_first_fpr_x86_64;
            m_reg_info.last_fpr             = k_last_fpr_x86_64;
            m_reg_info.first_st             = lldb_st0_x86_64;
            m_reg_info.last_st              = lldb_st7_x86_64;
            m_reg_info.first_mm             = lldb_mm0_x86_64;
            m_reg_info.last_mm              = lldb_mm7_x86_64;
            m_reg_info.first_xmm            = lldb_xmm0_x86_64;
            m_reg_info.last_xmm             = lldb_xmm15_x86_64;
            m_reg_info.first_ymm            = lldb_ymm0_x86_64;
            m_reg_info.last_ymm             = lldb_ymm15_x86_64;
            m_reg_info.first_dr             = lldb_dr0_x86_64;
            m_reg_info.gpr_flags            = lldb_rflags_x86_64;
            break;
        default:
            assert(false && "Unhandled target architecture.");
            break;
    }

    // Initialize m_iovec to point to the buffer and buffer size
    // using the conventions of Berkeley style UIO structures, as required
    // by PTRACE extensions.
    m_iovec.iov_base = &m_fpr.xstate.xsave;
    m_iovec.iov_len = sizeof(m_fpr.xstate.xsave);

    // Clear out the FPR state.
    ::memset(&m_fpr, 0, sizeof(FPR));
}

// CONSIDER after local and llgs debugging are merged, register set support can
// be moved into a base x86-64 class with IsRegisterSetAvailable made virtual.
uint32_t
NativeRegisterContextLinux_x86_64::GetRegisterSetCount () const
{
    uint32_t sets = 0;
    for (uint32_t set_index = 0; set_index < k_num_register_sets; ++set_index)
    {
        if (IsRegisterSetAvailable (set_index))
            ++sets;
    }

    return sets;
}

const lldb_private::RegisterSet *
NativeRegisterContextLinux_x86_64::GetRegisterSet (uint32_t set_index) const
{
    if (!IsRegisterSetAvailable (set_index))
        return nullptr;

    switch (GetRegisterInfoInterface ().GetTargetArchitecture ().GetMachine ())
    {
        case llvm::Triple::x86:
            return &g_reg_sets_i386[set_index];
        case llvm::Triple::x86_64:
            return &g_reg_sets_x86_64[set_index];
        default:
            assert (false && "Unhandled target architecture.");
            return nullptr;
    }

    return nullptr;
}

lldb_private::Error
NativeRegisterContextLinux_x86_64::ReadRegisterRaw (uint32_t reg_index, RegisterValue &reg_value)
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

lldb_private::Error
NativeRegisterContextLinux_x86_64::ReadRegister (const RegisterInfo *reg_info, RegisterValue &reg_value)
{
    Error error;

    if (!reg_info)
    {
        error.SetErrorString ("reg_info NULL");
        return error;
    }

    const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
    if (reg == LLDB_INVALID_REGNUM)
    {
        // This is likely an internal register for lldb use only and should not be directly queried.
        error.SetErrorStringWithFormat ("register \"%s\" is an internal-only lldb register, cannot read directly", reg_info->name);
        return error;
    }

    if (IsFPR(reg, GetFPRType()))
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

    if (reg_info->encoding == lldb::eEncodingVector)
    {
        lldb::ByteOrder byte_order = GetByteOrder();

        if (byte_order != lldb::eByteOrderInvalid)
        {
            if (reg >= m_reg_info.first_st && reg <= m_reg_info.last_st)
                reg_value.SetBytes(m_fpr.xstate.fxsave.stmm[reg - m_reg_info.first_st].bytes, reg_info->byte_size, byte_order);
            if (reg >= m_reg_info.first_mm && reg <= m_reg_info.last_mm)
                reg_value.SetBytes(m_fpr.xstate.fxsave.stmm[reg - m_reg_info.first_mm].bytes, reg_info->byte_size, byte_order);
            if (reg >= m_reg_info.first_xmm && reg <= m_reg_info.last_xmm)
                reg_value.SetBytes(m_fpr.xstate.fxsave.xmm[reg - m_reg_info.first_xmm].bytes, reg_info->byte_size, byte_order);
            if (reg >= m_reg_info.first_ymm && reg <= m_reg_info.last_ymm)
            {
                // Concatenate ymm using the register halves in xmm.bytes and ymmh.bytes
                if (GetFPRType() == eFPRTypeXSAVE && CopyXSTATEtoYMM(reg, byte_order))
                    reg_value.SetBytes(m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes, reg_info->byte_size, byte_order);
                else
                {
                    error.SetErrorString ("failed to copy ymm register value");
                    return error;
                }
            }

            if (reg_value.GetType() != RegisterValue::eTypeBytes)
                error.SetErrorString ("write failed - type was expected to be RegisterValue::eTypeBytes");

            return error;
        }

        error.SetErrorString ("byte order is invalid");
        return error;
    }

    // Get pointer to m_fpr.xstate.fxsave variable and set the data from it.
    assert (reg_info->byte_offset < sizeof(m_fpr));
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

lldb_private::Error
NativeRegisterContextLinux_x86_64::WriteRegister(const uint32_t reg,
                                                 const RegisterValue &value)
{
    Error error;

    uint32_t reg_to_write = reg;
    RegisterValue value_to_write = value;

    // Check if this is a subregister of a full register.
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex(reg);
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
            const uint32_t src_size = value.GetAsMemoryData (reg_info, src, sizeof(src), byte_order, error);
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
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s failed to get RegisterInfo for write register index %" PRIu32, __FUNCTION__, reg_to_write);
        return error;
    }

    NativeProcessLinux *const process_p = reinterpret_cast<NativeProcessLinux*> (process_sp.get ());
    return process_p->WriteRegisterValue(m_thread.GetID(),
                                         register_to_write_info_p->byte_offset,
                                         register_to_write_info_p->name,
                                         value_to_write);
}

lldb_private::Error
NativeRegisterContextLinux_x86_64::WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value)
{
    assert (reg_info && "reg_info is null");

    const uint32_t reg_index = reg_info->kinds[lldb::eRegisterKindLLDB];
    if (reg_index == LLDB_INVALID_REGNUM)
        return Error ("no lldb regnum for %s", reg_info && reg_info->name ? reg_info->name : "<unknown register>");

    if (IsGPR(reg_index))
        return WriteRegister(reg_index, reg_value);

    if (IsFPR(reg_index, GetFPRType()))
    {
        if (reg_info->encoding == lldb::eEncodingVector)
        {
            if (reg_index >= m_reg_info.first_st && reg_index <= m_reg_info.last_st)
                ::memcpy (m_fpr.xstate.fxsave.stmm[reg_index - m_reg_info.first_st].bytes, reg_value.GetBytes(), reg_value.GetByteSize());

            if (reg_index >= m_reg_info.first_mm && reg_index <= m_reg_info.last_mm)
                ::memcpy (m_fpr.xstate.fxsave.stmm[reg_index - m_reg_info.first_mm].bytes, reg_value.GetBytes(), reg_value.GetByteSize());

            if (reg_index >= m_reg_info.first_xmm && reg_index <= m_reg_info.last_xmm)
                ::memcpy (m_fpr.xstate.fxsave.xmm[reg_index - m_reg_info.first_xmm].bytes, reg_value.GetBytes(), reg_value.GetByteSize());

            if (reg_index >= m_reg_info.first_ymm && reg_index <= m_reg_info.last_ymm)
            {
                if (GetFPRType() != eFPRTypeXSAVE)
                    return Error ("target processor does not support AVX");

                // Store ymm register content, and split into the register halves in xmm.bytes and ymmh.bytes
                ::memcpy (m_ymm_set.ymm[reg_index - m_reg_info.first_ymm].bytes, reg_value.GetBytes(), reg_value.GetByteSize());
                if (!CopyYMMtoXSTATE(reg_index, GetByteOrder()))
                    return Error ("CopyYMMtoXSTATE() failed");
            }
        }
        else
        {
            // Get pointer to m_fpr.xstate.fxsave variable and set the data to it.
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
        }

        if (WriteFPR())
        {
            if (IsAVX(reg_index))
            {
                if (!CopyYMMtoXSTATE(reg_index, GetByteOrder()))
                    return Error ("CopyYMMtoXSTATE() failed");
            }
            return Error ();
        }
    }
    return Error ("failed - register wasn't recognized to be a GPR or an FPR, write strategy unknown");
}

lldb_private::Error
NativeRegisterContextLinux_x86_64::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    Error error;

    data_sp.reset (new DataBufferHeap (REG_CONTEXT_SIZE, 0));
    if (!data_sp)
    {
        error.SetErrorStringWithFormat ("failed to allocate DataBufferHeap instance of size %" PRIu64, REG_CONTEXT_SIZE);
        return error;
    }

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

    ::memcpy (dst, &m_gpr_x86_64, GetRegisterInfoInterface ().GetGPRSize ());
    dst += GetRegisterInfoInterface ().GetGPRSize ();
    if (GetFPRType () == eFPRTypeFXSAVE)
        ::memcpy (dst, &m_fpr.xstate.fxsave, sizeof(m_fpr.xstate.fxsave));
    else if (GetFPRType () == eFPRTypeXSAVE)
    {
        lldb::ByteOrder byte_order = GetByteOrder ();

        // Assemble the YMM register content from the register halves.
        for (uint32_t reg = m_reg_info.first_ymm; reg <= m_reg_info.last_ymm; ++reg)
        {
            if (!CopyXSTATEtoYMM (reg, byte_order))
            {
                error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s CopyXSTATEtoYMM() failed for reg num %" PRIu32, __FUNCTION__, reg);
                return error;
            }
        }

        // Copy the extended register state including the assembled ymm registers.
        ::memcpy (dst, &m_fpr, sizeof (m_fpr));
    }
    else
    {
        assert (false && "how do we save the floating point registers?");
        error.SetErrorString ("unsure how to save the floating point registers");
    }

    return error;
}

lldb_private::Error
NativeRegisterContextLinux_x86_64::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
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
    ::memcpy (&m_gpr_x86_64, src, GetRegisterInfoInterface ().GetGPRSize ());

    if (!WriteGPR ())
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s WriteGPR() failed", __FUNCTION__);
        return error;
    }

    src += GetRegisterInfoInterface ().GetGPRSize ();
    if (GetFPRType () == eFPRTypeFXSAVE)
        ::memcpy (&m_fpr.xstate.fxsave, src, sizeof(m_fpr.xstate.fxsave));
    else if (GetFPRType () == eFPRTypeXSAVE)
        ::memcpy (&m_fpr.xstate.xsave, src, sizeof(m_fpr.xstate.xsave));

    if (!WriteFPR ())
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s WriteFPR() failed", __FUNCTION__);
        return error;
    }

    if (GetFPRType() == eFPRTypeXSAVE)
    {
        lldb::ByteOrder byte_order = GetByteOrder();

        // Parse the YMM register content from the register halves.
        for (uint32_t reg = m_reg_info.first_ymm; reg <= m_reg_info.last_ymm; ++reg)
        {
            if (!CopyYMMtoXSTATE (reg, byte_order))
            {
                error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s CopyYMMtoXSTATE() failed for reg num %" PRIu32, __FUNCTION__, reg);
                return error;
            }
        }
    }

    return error;
}

bool
NativeRegisterContextLinux_x86_64::IsRegisterSetAvailable (uint32_t set_index) const
{
    // Note: Extended register sets are assumed to be at the end of g_reg_sets.
    uint32_t num_sets = k_num_register_sets - k_num_extended_register_sets;

    if (GetFPRType () == eFPRTypeXSAVE)
    {
        // AVX is the first extended register set.
        ++num_sets;
    }
    return (set_index < num_sets);
}

lldb::ByteOrder
NativeRegisterContextLinux_x86_64::GetByteOrder() const
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

bool
NativeRegisterContextLinux_x86_64::IsGPR(uint32_t reg_index) const
{
    // GPRs come first.
    return reg_index <= m_reg_info.last_gpr;
}

NativeRegisterContextLinux_x86_64::FPRType
NativeRegisterContextLinux_x86_64::GetFPRType () const
{
    if (m_fpr_type == eFPRTypeNotValid)
    {
        // TODO: Use assembly to call cpuid on the inferior and query ebx or ecx.

        // Try and see if AVX register retrieval works.
        m_fpr_type = eFPRTypeXSAVE;
        if (!const_cast<NativeRegisterContextLinux_x86_64*> (this)->ReadFPR ())
        {
            // Fall back to general floating point with no AVX support.
            m_fpr_type = eFPRTypeFXSAVE;
        }
    }

    return m_fpr_type;
}

bool
NativeRegisterContextLinux_x86_64::IsFPR(uint32_t reg_index) const
{
    return (m_reg_info.first_fpr <= reg_index && reg_index <= m_reg_info.last_fpr);
}

bool
NativeRegisterContextLinux_x86_64::IsFPR(uint32_t reg_index, FPRType fpr_type) const
{
    bool generic_fpr = IsFPR(reg_index);

    if (fpr_type == eFPRTypeXSAVE)
        return generic_fpr || IsAVX(reg_index);
    return generic_fpr;
}

bool
NativeRegisterContextLinux_x86_64::WriteFPR()
{
    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return false;
    NativeProcessLinux *const process_p = reinterpret_cast<NativeProcessLinux*> (process_sp.get ());

    if (GetFPRType() == eFPRTypeFXSAVE)
        return process_p->WriteFPR (m_thread.GetID (), &m_fpr.xstate.fxsave, sizeof (m_fpr.xstate.fxsave)).Success();

    if (GetFPRType() == eFPRTypeXSAVE)
        return process_p->WriteRegisterSet (m_thread.GetID (), &m_iovec, sizeof (m_fpr.xstate.xsave), NT_X86_XSTATE).Success();
    return false;
}

bool
NativeRegisterContextLinux_x86_64::IsAVX(uint32_t reg_index) const
{
    return (m_reg_info.first_ymm <= reg_index && reg_index <= m_reg_info.last_ymm);
}

bool
NativeRegisterContextLinux_x86_64::CopyXSTATEtoYMM (uint32_t reg_index, lldb::ByteOrder byte_order)
{
    if (!IsAVX (reg_index))
        return false;

    if (byte_order == lldb::eByteOrderLittle)
    {
        ::memcpy (m_ymm_set.ymm[reg_index - m_reg_info.first_ymm].bytes,
                 m_fpr.xstate.fxsave.xmm[reg_index - m_reg_info.first_ymm].bytes,
                 sizeof (XMMReg));
        ::memcpy (m_ymm_set.ymm[reg_index - m_reg_info.first_ymm].bytes + sizeof (XMMReg),
                 m_fpr.xstate.xsave.ymmh[reg_index - m_reg_info.first_ymm].bytes,
                 sizeof (YMMHReg));
        return true;
    }

    if (byte_order == lldb::eByteOrderBig)
    {
        ::memcpy(m_ymm_set.ymm[reg_index - m_reg_info.first_ymm].bytes + sizeof (XMMReg),
                 m_fpr.xstate.fxsave.xmm[reg_index - m_reg_info.first_ymm].bytes,
                 sizeof (XMMReg));
        ::memcpy(m_ymm_set.ymm[reg_index - m_reg_info.first_ymm].bytes,
                 m_fpr.xstate.xsave.ymmh[reg_index - m_reg_info.first_ymm].bytes,
                 sizeof (YMMHReg));
        return true;
    }
    return false; // unsupported or invalid byte order

}

bool
NativeRegisterContextLinux_x86_64::CopyYMMtoXSTATE(uint32_t reg, lldb::ByteOrder byte_order)
{
    if (!IsAVX(reg))
        return false;

    if (byte_order == lldb::eByteOrderLittle)
    {
        ::memcpy(m_fpr.xstate.fxsave.xmm[reg - m_reg_info.first_ymm].bytes,
                 m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes,
                 sizeof(XMMReg));
        ::memcpy(m_fpr.xstate.xsave.ymmh[reg - m_reg_info.first_ymm].bytes,
                 m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes + sizeof(XMMReg),
                 sizeof(YMMHReg));
        return true;
    }

    if (byte_order == lldb::eByteOrderBig)
    {
        ::memcpy(m_fpr.xstate.fxsave.xmm[reg - m_reg_info.first_ymm].bytes,
                 m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes + sizeof(XMMReg),
                 sizeof(XMMReg));
        ::memcpy(m_fpr.xstate.xsave.ymmh[reg - m_reg_info.first_ymm].bytes,
                 m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes,
                 sizeof(YMMHReg));
        return true;
    }
    return false; // unsupported or invalid byte order
}

bool
NativeRegisterContextLinux_x86_64::ReadFPR ()
{
    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return false;
    NativeProcessLinux *const process_p = reinterpret_cast<NativeProcessLinux*> (process_sp.get ());

    const FPRType fpr_type = GetFPRType ();
    switch (fpr_type)
    {
    case FPRType::eFPRTypeFXSAVE:
        return process_p->ReadFPR (m_thread.GetID (), &m_fpr.xstate.fxsave, sizeof (m_fpr.xstate.fxsave)).Success();

    case FPRType::eFPRTypeXSAVE:
        return process_p->ReadRegisterSet (m_thread.GetID (), &m_iovec, sizeof (m_fpr.xstate.xsave), NT_X86_XSTATE).Success();

    default:
        return false;
    }
}

bool
NativeRegisterContextLinux_x86_64::ReadGPR()
{
    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return false;
    NativeProcessLinux *const process_p = reinterpret_cast<NativeProcessLinux*> (process_sp.get ());

    return process_p->ReadGPR (m_thread.GetID (), &m_gpr_x86_64, GetRegisterInfoInterface ().GetGPRSize ()).Success();
}

bool
NativeRegisterContextLinux_x86_64::WriteGPR()
{
    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return false;
    NativeProcessLinux *const process_p = reinterpret_cast<NativeProcessLinux*> (process_sp.get ());

    return process_p->WriteGPR (m_thread.GetID (), &m_gpr_x86_64, GetRegisterInfoInterface ().GetGPRSize ()).Success();
}

Error
NativeRegisterContextLinux_x86_64::IsWatchpointHit(uint8_t wp_index)
{
    if (wp_index >= NumSupportedHardwareWatchpoints())
        return Error ("Watchpoint index out of range");

    RegisterValue reg_value;
    Error error = ReadRegisterRaw(lldb_dr6_x86_64, reg_value);
    if (error.Fail()) return error;

    uint64_t status_bits = reg_value.GetAsUInt64();

    bool is_hit = status_bits & (1 << wp_index);

    error.SetError (!is_hit, lldb::eErrorTypeInvalid);

    return error;
}

Error
NativeRegisterContextLinux_x86_64::IsWatchpointVacant(uint32_t wp_index)
{
    if (wp_index >= NumSupportedHardwareWatchpoints())
        return Error ("Watchpoint index out of range");

    RegisterValue reg_value;
    Error error = ReadRegisterRaw(lldb_dr7_x86_64, reg_value);
    if (error.Fail()) return error;

    uint64_t control_bits = reg_value.GetAsUInt64();

    bool is_vacant = !(control_bits & (1 << (2 * wp_index)));

    error.SetError (!is_vacant, lldb::eErrorTypeInvalid);

    return error;
}

Error
NativeRegisterContextLinux_x86_64::SetHardwareWatchpointWithIndex(
        lldb::addr_t addr, size_t size, uint32_t watch_flags, uint32_t wp_index) {

    if (wp_index >= NumSupportedHardwareWatchpoints())
        return Error ("Watchpoint index out of range");

    if (watch_flags != 0x1 && watch_flags != 0x3)
        return Error ("Invalid read/write bits for watchpoint");

    if (size != 1 && size != 2 && size != 4 && size != 8)
        return Error ("Invalid size for watchpoint");

    Error error = IsWatchpointVacant (wp_index);
    if (error.Fail()) return error;

    RegisterValue reg_value;
    error = ReadRegisterRaw(lldb_dr7_x86_64, reg_value);
    if (error.Fail()) return error;

    // for watchpoints 0, 1, 2, or 3, respectively,
    // set bits 1, 3, 5, or 7
    uint64_t enable_bit = 1 << (2 * wp_index);

    // set bits 16-17, 20-21, 24-25, or 28-29
    // with 0b01 for write, and 0b11 for read/write
    uint64_t rw_bits = watch_flags << (16 + 4 * wp_index);

    // set bits 18-19, 22-23, 26-27, or 30-31
    // with 0b00, 0b01, 0b10, or 0b11
    // for 1, 2, 8 (if supported), or 4 bytes, respectively
    uint64_t size_bits = (size == 8 ? 0x2 : size - 1) << (18 + 4 * wp_index);

    uint64_t bit_mask = (0x3 << (2 * wp_index)) | (0xF << (16 + 4 * wp_index));

    uint64_t control_bits = reg_value.GetAsUInt64() & ~bit_mask;

    control_bits |= enable_bit | rw_bits | size_bits;

    error = WriteRegister(m_reg_info.first_dr + wp_index, RegisterValue(addr));
    if (error.Fail()) return error;

    error = WriteRegister(lldb_dr7_x86_64, RegisterValue(control_bits));
    if (error.Fail()) return error;

    error.Clear();
    return error;
}

bool
NativeRegisterContextLinux_x86_64::ClearHardwareWatchpoint(uint32_t wp_index)
{
    if (wp_index >= NumSupportedHardwareWatchpoints())
        return false;

    RegisterValue reg_value;

    // for watchpoints 0, 1, 2, or 3, respectively,
    // clear bits 0, 1, 2, or 3 of the debug status register (DR6)
    Error error = ReadRegisterRaw(lldb_dr6_x86_64, reg_value);
    if (error.Fail()) return false;
    uint64_t bit_mask = 1 << wp_index;
    uint64_t status_bits = reg_value.GetAsUInt64() & ~bit_mask;
    error = WriteRegister(lldb_dr6_x86_64, RegisterValue(status_bits));
    if (error.Fail()) return false;

    // for watchpoints 0, 1, 2, or 3, respectively,
    // clear bits {0-1,16-19}, {2-3,20-23}, {4-5,24-27}, or {6-7,28-31}
    // of the debug control register (DR7)
    error = ReadRegisterRaw(lldb_dr7_x86_64, reg_value);
    if (error.Fail()) return false;
    bit_mask = (0x3 << (2 * wp_index)) | (0xF << (16 + 4 * wp_index));
    uint64_t control_bits = reg_value.GetAsUInt64() & ~bit_mask;
    return WriteRegister(lldb_dr7_x86_64, RegisterValue(control_bits)).Success();
}

Error
NativeRegisterContextLinux_x86_64::ClearAllHardwareWatchpoints()
{
    RegisterValue reg_value;

    // clear bits {0-4} of the debug status register (DR6)
    Error error = ReadRegisterRaw(lldb_dr6_x86_64, reg_value);
    if (error.Fail()) return error;
    uint64_t bit_mask = 0xF;
    uint64_t status_bits = reg_value.GetAsUInt64() & ~bit_mask;
    error = WriteRegister(lldb_dr6_x86_64, RegisterValue(status_bits));
    if (error.Fail()) return error;

    // clear bits {0-7,16-31} of the debug control register (DR7)
    error = ReadRegisterRaw(lldb_dr7_x86_64, reg_value);
    if (error.Fail()) return error;
    bit_mask = 0xFF | (0xFFFF << 16);
    uint64_t control_bits = reg_value.GetAsUInt64() & ~bit_mask;
    return WriteRegister(lldb_dr7_x86_64, RegisterValue(control_bits));
}

uint32_t
NativeRegisterContextLinux_x86_64::SetHardwareWatchpoint(
        lldb::addr_t addr, size_t size, uint32_t watch_flags)
{
    const uint32_t num_hw_watchpoints = NumSupportedHardwareWatchpoints();
    for (uint32_t wp_index = 0; wp_index < num_hw_watchpoints; ++wp_index)
        if (IsWatchpointVacant(wp_index).Success())
        {
            if (SetHardwareWatchpointWithIndex(addr, size, watch_flags, wp_index).Fail())
                continue;
            return wp_index;
        }
    return LLDB_INVALID_INDEX32;
}

lldb::addr_t
NativeRegisterContextLinux_x86_64::GetWatchpointAddress(uint32_t wp_index)
{
    if (wp_index >= NumSupportedHardwareWatchpoints())
        return LLDB_INVALID_ADDRESS;
    RegisterValue reg_value;
    if (ReadRegisterRaw(m_reg_info.first_dr + wp_index, reg_value).Fail())
        return LLDB_INVALID_ADDRESS;
    return reg_value.GetAsUInt64();
}

uint32_t
NativeRegisterContextLinux_x86_64::NumSupportedHardwareWatchpoints ()
{
    // Available debug address registers: dr0, dr1, dr2, dr3
    return 4;
}
