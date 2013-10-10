//===-- RegisterContextPOSIX_x86.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <errno.h>
#include <stdint.h>

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Host/Endian.h"
#include "llvm/Support/Compiler.h"

#include "ProcessPOSIX.h"
#include "RegisterContext_x86.h"
#include "RegisterContextPOSIX_x86.h"
#include "Plugins/Process/elf-core/ProcessElfCore.h"

using namespace lldb_private;
using namespace lldb;

const uint32_t
g_gpr_regnums_i386[] =
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
static_assert((sizeof(g_gpr_regnums_i386) / sizeof(g_gpr_regnums_i386[0])) == k_num_gpr_registers_i386,
    "g_gpr_regnums_i386 has wrong number of register infos");

const uint32_t
g_fpu_regnums_i386[] =
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
static_assert((sizeof(g_fpu_regnums_i386) / sizeof(g_fpu_regnums_i386[0])) == k_num_fpr_registers_i386,
    "g_fpu_regnums_i386 has wrong number of register infos");

const uint32_t
g_avx_regnums_i386[] =
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
static_assert((sizeof(g_avx_regnums_i386) / sizeof(g_avx_regnums_i386[0])) == k_num_avx_registers_i386,
    " g_avx_regnums_i386 has wrong number of register infos");

static const
uint32_t g_gpr_regnums_x86_64[] =
{
    gpr_rax_x86_64,
    gpr_rbx_x86_64,
    gpr_rcx_x86_64,
    gpr_rdx_x86_64,
    gpr_rdi_x86_64,
    gpr_rsi_x86_64,
    gpr_rbp_x86_64,
    gpr_rsp_x86_64,
    gpr_r8_x86_64,
    gpr_r9_x86_64,
    gpr_r10_x86_64,
    gpr_r11_x86_64,
    gpr_r12_x86_64,
    gpr_r13_x86_64,
    gpr_r14_x86_64,
    gpr_r15_x86_64,
    gpr_rip_x86_64,
    gpr_rflags_x86_64,
    gpr_cs_x86_64,
    gpr_fs_x86_64,
    gpr_gs_x86_64,
    gpr_ss_x86_64,
    gpr_ds_x86_64,
    gpr_es_x86_64,
    gpr_eax_x86_64,
    gpr_ebx_x86_64,
    gpr_ecx_x86_64,
    gpr_edx_x86_64,
    gpr_edi_x86_64,
    gpr_esi_x86_64,
    gpr_ebp_x86_64,
    gpr_esp_x86_64,
    gpr_r8d_x86_64,    // Low 32 bits or r8
    gpr_r9d_x86_64,    // Low 32 bits or r9
    gpr_r10d_x86_64,   // Low 32 bits or r10
    gpr_r11d_x86_64,   // Low 32 bits or r11
    gpr_r12d_x86_64,   // Low 32 bits or r12
    gpr_r13d_x86_64,   // Low 32 bits or r13
    gpr_r14d_x86_64,   // Low 32 bits or r14
    gpr_r15d_x86_64,   // Low 32 bits or r15
    gpr_ax_x86_64,
    gpr_bx_x86_64,
    gpr_cx_x86_64,
    gpr_dx_x86_64,
    gpr_di_x86_64,
    gpr_si_x86_64,
    gpr_bp_x86_64,
    gpr_sp_x86_64,
    gpr_r8w_x86_64,    // Low 16 bits or r8
    gpr_r9w_x86_64,    // Low 16 bits or r9
    gpr_r10w_x86_64,   // Low 16 bits or r10
    gpr_r11w_x86_64,   // Low 16 bits or r11
    gpr_r12w_x86_64,   // Low 16 bits or r12
    gpr_r13w_x86_64,   // Low 16 bits or r13
    gpr_r14w_x86_64,   // Low 16 bits or r14
    gpr_r15w_x86_64,   // Low 16 bits or r15
    gpr_ah_x86_64,
    gpr_bh_x86_64,
    gpr_ch_x86_64,
    gpr_dh_x86_64,
    gpr_al_x86_64,
    gpr_bl_x86_64,
    gpr_cl_x86_64,
    gpr_dl_x86_64,
    gpr_dil_x86_64,
    gpr_sil_x86_64,
    gpr_bpl_x86_64,
    gpr_spl_x86_64,
    gpr_r8l_x86_64,    // Low 8 bits or r8
    gpr_r9l_x86_64,    // Low 8 bits or r9
    gpr_r10l_x86_64,   // Low 8 bits or r10
    gpr_r11l_x86_64,   // Low 8 bits or r11
    gpr_r12l_x86_64,   // Low 8 bits or r12
    gpr_r13l_x86_64,   // Low 8 bits or r13
    gpr_r14l_x86_64,   // Low 8 bits or r14
    gpr_r15l_x86_64,   // Low 8 bits or r15
};
static_assert((sizeof(g_gpr_regnums_x86_64) / sizeof(g_gpr_regnums_x86_64[0])) == k_num_gpr_registers_x86_64,
    "g_gpr_regnums_x86_64 has wrong number of register infos");

static const uint32_t
g_fpu_regnums_x86_64[] =
{
    fpu_fctrl_x86_64,
    fpu_fstat_x86_64,
    fpu_ftag_x86_64,
    fpu_fop_x86_64,
    fpu_fiseg_x86_64,
    fpu_fioff_x86_64,
    fpu_foseg_x86_64,
    fpu_fooff_x86_64,
    fpu_mxcsr_x86_64,
    fpu_mxcsrmask_x86_64,
    fpu_st0_x86_64,
    fpu_st1_x86_64,
    fpu_st2_x86_64,
    fpu_st3_x86_64,
    fpu_st4_x86_64,
    fpu_st5_x86_64,
    fpu_st6_x86_64,
    fpu_st7_x86_64,
    fpu_mm0_x86_64,
    fpu_mm1_x86_64,
    fpu_mm2_x86_64,
    fpu_mm3_x86_64,
    fpu_mm4_x86_64,
    fpu_mm5_x86_64,
    fpu_mm6_x86_64,
    fpu_mm7_x86_64,
    fpu_xmm0_x86_64,
    fpu_xmm1_x86_64,
    fpu_xmm2_x86_64,
    fpu_xmm3_x86_64,
    fpu_xmm4_x86_64,
    fpu_xmm5_x86_64,
    fpu_xmm6_x86_64,
    fpu_xmm7_x86_64,
    fpu_xmm8_x86_64,
    fpu_xmm9_x86_64,
    fpu_xmm10_x86_64,
    fpu_xmm11_x86_64,
    fpu_xmm12_x86_64,
    fpu_xmm13_x86_64,
    fpu_xmm14_x86_64,
    fpu_xmm15_x86_64
};
static_assert((sizeof(g_fpu_regnums_x86_64) / sizeof(g_fpu_regnums_x86_64[0])) == k_num_fpr_registers_x86_64,
    "g_fpu_regnums_x86_64 has wrong number of register infos");

static const uint32_t
g_avx_regnums_x86_64[] =
{
    fpu_ymm0_x86_64,
    fpu_ymm1_x86_64,
    fpu_ymm2_x86_64,
    fpu_ymm3_x86_64,
    fpu_ymm4_x86_64,
    fpu_ymm5_x86_64,
    fpu_ymm6_x86_64,
    fpu_ymm7_x86_64,
    fpu_ymm8_x86_64,
    fpu_ymm9_x86_64,
    fpu_ymm10_x86_64,
    fpu_ymm11_x86_64,
    fpu_ymm12_x86_64,
    fpu_ymm13_x86_64,
    fpu_ymm14_x86_64,
    fpu_ymm15_x86_64
};
static_assert((sizeof(g_avx_regnums_x86_64) / sizeof(g_avx_regnums_x86_64[0])) == k_num_avx_registers_x86_64,
    "g_avx_regnums_x86_64 has wrong number of register infos");

uint32_t RegisterContextPOSIX_x86::g_contained_eax[] = { gpr_eax_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_ebx[] = { gpr_ebx_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_ecx[] = { gpr_ecx_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_edx[] = { gpr_edx_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_edi[] = { gpr_edi_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_esi[] = { gpr_esi_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_ebp[] = { gpr_ebp_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_esp[] = { gpr_esp_i386, LLDB_INVALID_REGNUM };

uint32_t RegisterContextPOSIX_x86::g_invalidate_eax[] = { gpr_eax_i386, gpr_ax_i386, gpr_ah_i386,  gpr_al_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_ebx[] = { gpr_ebx_i386, gpr_bx_i386, gpr_bh_i386,  gpr_bl_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_ecx[] = { gpr_ecx_i386, gpr_cx_i386, gpr_ch_i386,  gpr_cl_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_edx[] = { gpr_edx_i386, gpr_dx_i386, gpr_dh_i386,  gpr_dl_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_edi[] = { gpr_edi_i386, gpr_di_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_esi[] = { gpr_esi_i386, gpr_si_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_ebp[] = { gpr_ebp_i386, gpr_bp_i386, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_esp[] = { gpr_esp_i386, gpr_sp_i386, LLDB_INVALID_REGNUM };

uint32_t RegisterContextPOSIX_x86::g_contained_rax[] = { gpr_rax_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_rbx[] = { gpr_rbx_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_rcx[] = { gpr_rcx_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_rdx[] = { gpr_rdx_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_rdi[] = { gpr_rdi_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_rsi[] = { gpr_rsi_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_rbp[] = { gpr_rbp_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_rsp[] = { gpr_rsp_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_r8[]  = { gpr_r8_x86_64,  LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_r9[]  = { gpr_r9_x86_64,  LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_r10[] = { gpr_r10_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_r11[] = { gpr_r11_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_r12[] = { gpr_r12_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_r13[] = { gpr_r13_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_r14[] = { gpr_r14_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_contained_r15[] = { gpr_r15_x86_64, LLDB_INVALID_REGNUM };

uint32_t RegisterContextPOSIX_x86::g_invalidate_rax[] = { gpr_rax_x86_64, gpr_eax_x86_64,  gpr_ax_x86_64,   gpr_ah_x86_64,   gpr_al_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_rbx[] = { gpr_rbx_x86_64, gpr_ebx_x86_64,  gpr_bx_x86_64,   gpr_bh_x86_64,   gpr_bl_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_rcx[] = { gpr_rcx_x86_64, gpr_ecx_x86_64,  gpr_cx_x86_64,   gpr_ch_x86_64,   gpr_cl_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_rdx[] = { gpr_rdx_x86_64, gpr_edx_x86_64,  gpr_dx_x86_64,   gpr_dh_x86_64,   gpr_dl_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_rdi[] = { gpr_rdi_x86_64, gpr_edi_x86_64,  gpr_di_x86_64,   gpr_dil_x86_64,  LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_rsi[] = { gpr_rsi_x86_64, gpr_esi_x86_64,  gpr_si_x86_64,   gpr_sil_x86_64,  LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_rbp[] = { gpr_rbp_x86_64, gpr_ebp_x86_64,  gpr_bp_x86_64,   gpr_bpl_x86_64,  LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_rsp[] = { gpr_rsp_x86_64, gpr_esp_x86_64,  gpr_sp_x86_64,   gpr_spl_x86_64,  LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_r8[]  = { gpr_r8_x86_64,  gpr_r8d_x86_64,  gpr_r8w_x86_64,  gpr_r8l_x86_64,  LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_r9[]  = { gpr_r9_x86_64,  gpr_r9d_x86_64,  gpr_r9w_x86_64,  gpr_r9l_x86_64,  LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_r10[] = { gpr_r10_x86_64, gpr_r10d_x86_64, gpr_r10w_x86_64, gpr_r10l_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_r11[] = { gpr_r11_x86_64, gpr_r11d_x86_64, gpr_r11w_x86_64, gpr_r11l_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_r12[] = { gpr_r12_x86_64, gpr_r12d_x86_64, gpr_r12w_x86_64, gpr_r12l_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_r13[] = { gpr_r13_x86_64, gpr_r13d_x86_64, gpr_r13w_x86_64, gpr_r13l_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_r14[] = { gpr_r14_x86_64, gpr_r14d_x86_64, gpr_r14w_x86_64, gpr_r14l_x86_64, LLDB_INVALID_REGNUM };
uint32_t RegisterContextPOSIX_x86::g_invalidate_r15[] = { gpr_r15_x86_64, gpr_r15d_x86_64, gpr_r15w_x86_64, gpr_r15l_x86_64, LLDB_INVALID_REGNUM };

// Number of register sets provided by this context.
enum
{
    k_num_extended_register_sets = 1,
    k_num_register_sets = 3
};

static const RegisterSet
g_reg_sets_i386[k_num_register_sets] =
{
    { "General Purpose Registers",  "gpr", k_num_gpr_registers_i386, g_gpr_regnums_i386 },
    { "Floating Point Registers",   "fpu", k_num_fpr_registers_i386, g_fpu_regnums_i386 },
    { "Advanced Vector Extensions", "avx", k_num_avx_registers_i386, g_avx_regnums_i386 }
};

static const RegisterSet
g_reg_sets_x86_64[k_num_register_sets] =
{
    { "General Purpose Registers",  "gpr", k_num_gpr_registers_x86_64, g_gpr_regnums_x86_64 },
    { "Floating Point Registers",   "fpu", k_num_fpr_registers_x86_64, g_fpu_regnums_x86_64 },
    { "Advanced Vector Extensions", "avx", k_num_avx_registers_x86_64, g_avx_regnums_x86_64 }
};

bool RegisterContextPOSIX_x86::IsGPR(unsigned reg)
{
    return reg <= m_reg_info.last_gpr;   // GPR's come first.
}

bool RegisterContextPOSIX_x86::IsFPR(unsigned reg)
{
    return (m_reg_info.first_fpr <= reg && reg <= m_reg_info.last_fpr);
}

bool RegisterContextPOSIX_x86::IsAVX(unsigned reg)
{
    return (m_reg_info.first_ymm <= reg && reg <= m_reg_info.last_ymm);
}

bool RegisterContextPOSIX_x86::IsFPR(unsigned reg, FPRType fpr_type)
{
    bool generic_fpr = IsFPR(reg);

    if (fpr_type == eXSAVE)
        return generic_fpr || IsAVX(reg);
    return generic_fpr;
}

RegisterContextPOSIX_x86::RegisterContextPOSIX_x86(Thread &thread,
                                                   uint32_t concrete_frame_idx,
                                                   RegisterInfoInterface *register_info)
    : RegisterContext(thread, concrete_frame_idx)
{
    m_register_info_ap.reset(register_info);

    switch (register_info->m_target_arch.GetCore())
    {
        case ArchSpec::eCore_x86_32_i386:
        case ArchSpec::eCore_x86_32_i486:
        case ArchSpec::eCore_x86_32_i486sx:
            m_reg_info.num_registers        = k_num_registers_i386;
            m_reg_info.num_gpr_registers    = k_num_gpr_registers_i386;
            m_reg_info.num_fpr_registers    = k_num_fpr_registers_i386;
            m_reg_info.num_avx_registers    = k_num_avx_registers_i386;
            m_reg_info.last_gpr             = k_last_gpr_i386;
            m_reg_info.first_fpr            = k_first_fpr_i386;
            m_reg_info.last_fpr             = k_last_fpr_i386;
            m_reg_info.first_st             = fpu_st0_i386;
            m_reg_info.last_st              = fpu_st7_i386;
            m_reg_info.first_mm             = fpu_mm0_i386;
            m_reg_info.last_mm              = fpu_mm7_i386;
            m_reg_info.first_xmm            = fpu_xmm0_i386;
            m_reg_info.last_xmm             = fpu_xmm7_i386;
            m_reg_info.first_ymm            = fpu_ymm0_i386;
            m_reg_info.last_ymm             = fpu_ymm7_i386;
            m_reg_info.first_dr             = dr0_i386;
            m_reg_info.gpr_flags            = gpr_eflags_i386;
            break;
        case ArchSpec::eCore_x86_64_x86_64:
            m_reg_info.num_registers        = k_num_registers_x86_64;
            m_reg_info.num_gpr_registers    = k_num_gpr_registers_x86_64;
            m_reg_info.num_fpr_registers    = k_num_fpr_registers_x86_64;
            m_reg_info.num_avx_registers    = k_num_avx_registers_x86_64;
            m_reg_info.last_gpr             = k_last_gpr_x86_64;
            m_reg_info.first_fpr            = k_first_fpr_x86_64;
            m_reg_info.last_fpr             = k_last_fpr_x86_64;
            m_reg_info.first_st             = fpu_st0_x86_64;
            m_reg_info.last_st              = fpu_st7_x86_64;
            m_reg_info.first_mm             = fpu_mm0_x86_64;
            m_reg_info.last_mm              = fpu_mm7_x86_64;
            m_reg_info.first_xmm            = fpu_xmm0_x86_64;
            m_reg_info.last_xmm             = fpu_xmm15_x86_64;
            m_reg_info.first_ymm            = fpu_ymm0_x86_64;
            m_reg_info.last_ymm             = fpu_ymm15_x86_64;
            m_reg_info.first_dr             = dr0_x86_64;
            m_reg_info.gpr_flags            = gpr_rflags_x86_64;
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

    ::memset(&m_fpr, 0, sizeof(FPR));

    // elf-core yet to support ReadFPR()
    ProcessSP base = CalculateProcess();
    if (base.get()->GetPluginName() ==  ProcessElfCore::GetPluginNameStatic())
        return;
    
    m_fpr_type = eNotValid;
}

RegisterContextPOSIX_x86::~RegisterContextPOSIX_x86()
{
}

RegisterContextPOSIX_x86::FPRType RegisterContextPOSIX_x86::GetFPRType()
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
RegisterContextPOSIX_x86::Invalidate()
{
}

void
RegisterContextPOSIX_x86::InvalidateAllRegisters()
{
}

unsigned
RegisterContextPOSIX_x86::GetRegisterOffset(unsigned reg)
{
    assert(reg < m_reg_info.num_registers && "Invalid register number.");
    return GetRegisterInfo()[reg].byte_offset;
}

unsigned
RegisterContextPOSIX_x86::GetRegisterSize(unsigned reg)
{
    assert(reg < m_reg_info.num_registers && "Invalid register number.");
    return GetRegisterInfo()[reg].byte_size;
}

size_t
RegisterContextPOSIX_x86::GetRegisterCount()
{
    size_t num_registers = m_reg_info.num_gpr_registers + m_reg_info.num_fpr_registers;
    if (GetFPRType() == eXSAVE)
      return num_registers + m_reg_info.num_avx_registers;
    return num_registers;
}

size_t
RegisterContextPOSIX_x86::GetGPRSize()
{
    return m_register_info_ap->GetGPRSize ();
}

const RegisterInfo *
RegisterContextPOSIX_x86::GetRegisterInfo()
{
    // Commonly, this method is overridden and g_register_infos is copied and specialized.
    // So, use GetRegisterInfo() rather than g_register_infos in this scope.
    return m_register_info_ap->GetRegisterInfo ();
}

const RegisterInfo *
RegisterContextPOSIX_x86::GetRegisterInfoAtIndex(size_t reg)
{
    if (reg < m_reg_info.num_registers)
        return &GetRegisterInfo()[reg];
    else
        return NULL;
}

size_t
RegisterContextPOSIX_x86::GetRegisterSetCount()
{
    size_t sets = 0;
    for (size_t set = 0; set < k_num_register_sets; ++set)
    {
        if (IsRegisterSetAvailable(set))
            ++sets;
    }

    return sets;
}

const RegisterSet *
RegisterContextPOSIX_x86::GetRegisterSet(size_t set)
{
    if (IsRegisterSetAvailable(set))
    {
        switch (m_register_info_ap->m_target_arch.GetCore())
        {
            case ArchSpec::eCore_x86_32_i386:
            case ArchSpec::eCore_x86_32_i486:
            case ArchSpec::eCore_x86_32_i486sx:
                return &g_reg_sets_i386[set];
            case ArchSpec::eCore_x86_64_x86_64:
                return &g_reg_sets_x86_64[set];
            default:
                assert(false && "Unhandled target architecture.");
                return NULL;
        }
    }
    return NULL;
}

const char *
RegisterContextPOSIX_x86::GetRegisterName(unsigned reg)
{
    assert(reg < m_reg_info.num_registers && "Invalid register offset.");
    return GetRegisterInfo()[reg].name;
}

lldb::ByteOrder
RegisterContextPOSIX_x86::GetByteOrder()
{
    // Get the target process whose privileged thread was used for the register read.
    lldb::ByteOrder byte_order = eByteOrderInvalid;
    Process *process = CalculateProcess().get();

    if (process)
        byte_order = process->GetByteOrder();
    return byte_order;
}

// Parse ymm registers and into xmm.bytes and ymmh.bytes.
bool RegisterContextPOSIX_x86::CopyYMMtoXSTATE(uint32_t reg, lldb::ByteOrder byte_order)
{
    if (!IsAVX(reg))
        return false;

    if (byte_order == eByteOrderLittle)
    {
        ::memcpy(m_fpr.xstate.fxsave.xmm[reg - m_reg_info.first_ymm].bytes,
                 m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes,
                 sizeof(XMMReg));
        ::memcpy(m_fpr.xstate.xsave.ymmh[reg - m_reg_info.first_ymm].bytes,
                 m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes + sizeof(XMMReg),
                 sizeof(YMMHReg));
        return true;
    }

    if (byte_order == eByteOrderBig)
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

// Concatenate xmm.bytes with ymmh.bytes
bool RegisterContextPOSIX_x86::CopyXSTATEtoYMM(uint32_t reg, lldb::ByteOrder byte_order)
{
    if (!IsAVX(reg))
        return false;

    if (byte_order == eByteOrderLittle)
    {
        ::memcpy(m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes,
                 m_fpr.xstate.fxsave.xmm[reg - m_reg_info.first_ymm].bytes,
                 sizeof(XMMReg));
        ::memcpy(m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes + sizeof(XMMReg),
                 m_fpr.xstate.xsave.ymmh[reg - m_reg_info.first_ymm].bytes,
                 sizeof(YMMHReg));
        return true;
    }

    if (byte_order == eByteOrderBig)
    {
        ::memcpy(m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes + sizeof(XMMReg),
                 m_fpr.xstate.fxsave.xmm[reg - m_reg_info.first_ymm].bytes,
                 sizeof(XMMReg));
        ::memcpy(m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes,
                 m_fpr.xstate.xsave.ymmh[reg - m_reg_info.first_ymm].bytes,
                 sizeof(YMMHReg));
        return true;
    }
    return false; // unsupported or invalid byte order
}

bool
RegisterContextPOSIX_x86::IsRegisterSetAvailable(size_t set_index)
{
    // Note: Extended register sets are assumed to be at the end of g_reg_sets...
    size_t num_sets = k_num_register_sets - k_num_extended_register_sets;

    if (GetFPRType() == eXSAVE) // ...and to start with AVX registers.
        ++num_sets;
    return (set_index < num_sets);
}


// Used when parsing DWARF and EH frame information and any other
// object file sections that contain register numbers in them. 
uint32_t
RegisterContextPOSIX_x86::ConvertRegisterKindToRegisterNumber(uint32_t kind,
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

