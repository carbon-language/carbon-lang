//===-- RegisterContextPOSIX_i386.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DataExtractor.h"
#include "lldb/Target/Thread.h"
#include "lldb/Host/Endian.h"
#include "llvm/Support/Compiler.h"

#include "ProcessPOSIX.h"
#include "ProcessPOSIXLog.h"
#include "ProcessMonitor.h"
#include "RegisterContextPOSIX_i386.h"
#include "RegisterContext_x86.h"

using namespace lldb_private;
using namespace lldb;

const uint32_t
RegisterContextPOSIX_i386::g_gpr_regnums[] =
{
    gpr_eax_i386,
    gpr_ebx_i386,
    gpr_ecx_i386,
    gpr_edx_i386,
    gpr_edi_i386,
    gpr_esi_i386,
    gpr_ebp_i386,
    gpr_esp_i386,
    gpr_eip_i386,
    gpr_eflags_i386,
    gpr_cs_i386,
    gpr_fs_i386,
    gpr_gs_i386,
    gpr_ss_i386,
    gpr_ds_i386,
    gpr_es_i386,
    gpr_ax_i386,
    gpr_bx_i386,
    gpr_cx_i386,
    gpr_dx_i386,
    gpr_di_i386,
    gpr_si_i386,
    gpr_bp_i386,
    gpr_sp_i386,
    gpr_ah_i386,
    gpr_bh_i386,
    gpr_ch_i386,
    gpr_dh_i386,
    gpr_al_i386,
    gpr_bl_i386,
    gpr_cl_i386,
    gpr_dl_i386
};
static_assert((sizeof(RegisterContextPOSIX_i386::g_gpr_regnums) / sizeof(RegisterContextPOSIX_i386::g_gpr_regnums[0])) == k_num_gpr_registers_i386,
    "g_gpr_regnums has wrong number of register infos");

const uint32_t
RegisterContextPOSIX_i386::g_fpu_regnums[] =
{
    fpu_fctrl_i386,
    fpu_fstat_i386,
    fpu_ftag_i386,
    fpu_fop_i386,
    fpu_fiseg_i386,
    fpu_fioff_i386,
    fpu_foseg_i386,
    fpu_fooff_i386,
    fpu_mxcsr_i386,
    fpu_mxcsrmask_i386,
    fpu_st0_i386,
    fpu_st1_i386,
    fpu_st2_i386,
    fpu_st3_i386,
    fpu_st4_i386,
    fpu_st5_i386,
    fpu_st6_i386,
    fpu_st7_i386,
    fpu_mm0_i386,
    fpu_mm1_i386,
    fpu_mm2_i386,
    fpu_mm3_i386,
    fpu_mm4_i386,
    fpu_mm5_i386,
    fpu_mm6_i386,
    fpu_mm7_i386,
    fpu_xmm0_i386,
    fpu_xmm1_i386,
    fpu_xmm2_i386,
    fpu_xmm3_i386,
    fpu_xmm4_i386,
    fpu_xmm5_i386,
    fpu_xmm6_i386,
    fpu_xmm7_i386
};
static_assert((sizeof(RegisterContextPOSIX_i386::g_fpu_regnums) / sizeof(RegisterContextPOSIX_i386::g_fpu_regnums[0])) == k_num_fpr_registers_i386,
    "g_gpr_regnums has wrong number of register infos");

const uint32_t
RegisterContextPOSIX_i386::g_avx_regnums[] =
{
    fpu_ymm0_i386,
    fpu_ymm1_i386,
    fpu_ymm2_i386,
    fpu_ymm3_i386,
    fpu_ymm4_i386,
    fpu_ymm5_i386,
    fpu_ymm6_i386,
    fpu_ymm7_i386
};
static_assert((sizeof(RegisterContextPOSIX_i386::g_avx_regnums) / sizeof(RegisterContextPOSIX_i386::g_avx_regnums[0])) == k_num_avx_registers_i386,
    "g_gpr_regnums has wrong number of register infos");

uint32_t RegisterContextPOSIX_i386::g_contained_eax[] = { gpr_eax_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_contained_ebx[] = { gpr_ebx_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_contained_ecx[] = { gpr_ecx_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_contained_edx[] = { gpr_edx_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_contained_edi[] = { gpr_edi_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_contained_esi[] = { gpr_esi_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_contained_ebp[] = { gpr_ebp_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_contained_esp[] = { gpr_esp_i386, LLDB_INVALID_REGNUM };

uint32_t RegisterContextPOSIX_i386::g_invalidate_eax[] = { gpr_eax_i386, gpr_ax_i386, gpr_ah_i386,  gpr_al_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_invalidate_ebx[] = { gpr_ebx_i386, gpr_bx_i386, gpr_bh_i386,  gpr_bl_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_invalidate_ecx[] = { gpr_ecx_i386, gpr_cx_i386, gpr_ch_i386,  gpr_cl_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_invalidate_edx[] = { gpr_edx_i386, gpr_dx_i386, gpr_dh_i386,  gpr_dl_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_invalidate_edi[] = { gpr_edi_i386, gpr_di_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_invalidate_esi[] = { gpr_esi_i386, gpr_si_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_invalidate_ebp[] = { gpr_ebp_i386, gpr_bp_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_i386::g_invalidate_esp[] = { gpr_esp_i386, gpr_sp_i386, LLDB_INVALID_REGNUM };

// Number of register sets provided by this context.
enum
{
    k_num_extended_register_sets = 1,
    k_num_register_sets = 3
};

static const RegisterSet
g_reg_sets[k_num_register_sets] =
{
    { "General Purpose Registers",  "gpr", k_num_gpr_registers_i386, RegisterContextPOSIX_i386::g_gpr_regnums },
    { "Floating Point Registers",   "fpu", k_num_fpr_registers_i386, RegisterContextPOSIX_i386::g_fpu_regnums },
    { "Advanced Vector Extensions", "avx", k_num_avx_registers_i386, RegisterContextPOSIX_i386::g_avx_regnums }
};


RegisterContextPOSIX_i386::RegisterContextPOSIX_i386(Thread &thread,
                                                     uint32_t concrete_frame_idx,
                                                     RegisterInfoInterface *register_info)
    : RegisterContext(thread, concrete_frame_idx)
{
    m_register_info_ap.reset(register_info);
}

RegisterContextPOSIX_i386::~RegisterContextPOSIX_i386()
{
}

RegisterContextPOSIX_i386::FPRType RegisterContextPOSIX_i386::GetFPRType()
{
    if (m_fpr_type == eNotValid)
    {
        // TODO: Use assembly to call cpuid on the inferior and query ebx or ecx
        m_fpr_type = eXSAVE; // extended floating-point registers, if available
        if (false == ReadFPR())
            m_fpr_type = eFXSAVE; // assume generic floating-point registers
    }
    return m_fpr_type;
}

void
RegisterContextPOSIX_i386::Invalidate()
{
}

void
RegisterContextPOSIX_i386::InvalidateAllRegisters()
{
}

unsigned RegisterContextPOSIX_i386::GetRegisterOffset(unsigned reg)
{
    assert(reg < k_num_registers_i386 && "Invalid register number.");
    return GetRegisterInfo()[reg].byte_offset;
}

unsigned RegisterContextPOSIX_i386::GetRegisterSize(unsigned reg)
{
    assert(reg < k_num_registers_i386 && "Invalid register number.");
    return GetRegisterInfo()[reg].byte_size;
}

const RegisterInfo *
RegisterContextPOSIX_i386::GetRegisterInfo()
{
    // Commonly, this method is overridden and g_register_infos is copied and specialized.
    // So, use GetRegisterInfo() rather than g_register_infos in this scope.
    return m_register_info_ap->GetRegisterInfo ();
}

size_t
RegisterContextPOSIX_i386::GetRegisterCount()
{
    size_t num_registers = k_num_gpr_registers_i386 + k_num_fpr_registers_i386;
    if (GetFPRType() == eXSAVE)
      return num_registers + k_num_avx_registers_i386;
    return num_registers;
}

const RegisterInfo *
RegisterContextPOSIX_i386::GetRegisterInfoAtIndex(size_t reg)
{
    if (reg < k_num_registers_i386)
        return &GetRegisterInfo()[reg];
    else
        return NULL;
}

size_t
RegisterContextPOSIX_i386::GetRegisterSetCount()
{
    return k_num_register_sets;
}

const RegisterSet *
RegisterContextPOSIX_i386::GetRegisterSet(size_t set)
{
    if (set < k_num_register_sets)
        return &g_reg_sets[set];
    else
        return NULL;
}

const char *
RegisterContextPOSIX_i386::GetRegisterName(unsigned reg)
{
    assert(reg < k_num_registers_i386 && "Invalid register offset.");
    return GetRegisterInfo()[reg].name;
}

bool
RegisterContextPOSIX_i386::ReadAllRegisterValues(DataBufferSP &data_sp)
{
    return false;
}

bool
RegisterContextPOSIX_i386::WriteAllRegisterValues(const DataBufferSP &data)
{
    return false;
}

bool
RegisterContextPOSIX_i386::UpdateAfterBreakpoint()
{
    // PC points one byte past the int3 responsible for the breakpoint.
    lldb::addr_t pc;

    if ((pc = GetPC()) == LLDB_INVALID_ADDRESS)
        return false;

    SetPC(pc - 1);
    return true;
}

uint32_t
RegisterContextPOSIX_i386::ConvertRegisterKindToRegisterNumber(uint32_t kind,
                                                               uint32_t num)
{
    const uint32_t num_regs = GetRegisterCount();

    assert (kind < kNumRegisterKinds);
    for (uint32_t reg_idx = 0; reg_idx < num_regs; ++reg_idx)
    {
        const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg_idx);

        if (reg_info->kinds[kind] == num)
            return reg_idx;
    }

    return LLDB_INVALID_REGNUM;

}

bool
RegisterContextPOSIX_i386::HardwareSingleStep(bool enable)
{
    return false;
}
