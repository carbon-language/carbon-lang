//===-- lldb-x86-register-enums.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_x86_register_enums_h
#define lldb_x86_register_enums_h

namespace lldb_private
{

    //---------------------------------------------------------------------------
    // Internal codes for all i386 registers.
    //---------------------------------------------------------------------------
    enum
    {
        k_first_gpr_i386,
        gpr_eax_i386 = k_first_gpr_i386,
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

        k_first_alias_i386,
        gpr_ax_i386 = k_first_alias_i386,
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
        gpr_dl_i386,
        k_last_alias_i386 = gpr_dl_i386,

        k_last_gpr_i386 = k_last_alias_i386,

        k_first_fpr_i386,
        fpu_fctrl_i386 = k_first_fpr_i386,
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
        fpu_xmm7_i386,
        k_last_fpr_i386 = fpu_xmm7_i386,

        k_first_avx_i386,
        fpu_ymm0_i386 = k_first_avx_i386,
        fpu_ymm1_i386,
        fpu_ymm2_i386,
        fpu_ymm3_i386,
        fpu_ymm4_i386,
        fpu_ymm5_i386,
        fpu_ymm6_i386,
        fpu_ymm7_i386,
        k_last_avx_i386 = fpu_ymm7_i386,

        dr0_i386,
        dr1_i386,
        dr2_i386,
        dr3_i386,
        dr4_i386,
        dr5_i386,
        dr6_i386,
        dr7_i386,

        k_num_registers_i386,
        k_num_gpr_registers_i386 = k_last_gpr_i386 - k_first_gpr_i386 + 1,
        k_num_fpr_registers_i386 = k_last_fpr_i386 - k_first_fpr_i386 + 1,
        k_num_avx_registers_i386 = k_last_avx_i386 - k_first_avx_i386 + 1
    };

    //---------------------------------------------------------------------------
    // Internal codes for all x86_64 registers.
    //---------------------------------------------------------------------------
    enum
    {
        k_first_gpr_x86_64,
        gpr_rax_x86_64 = k_first_gpr_x86_64,
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

        k_first_alias_x86_64,
        gpr_eax_x86_64 = k_first_alias_x86_64,
        gpr_ebx_x86_64,
        gpr_ecx_x86_64,
        gpr_edx_x86_64,
        gpr_edi_x86_64,
        gpr_esi_x86_64,
        gpr_ebp_x86_64,
        gpr_esp_x86_64,
        gpr_r8d_x86_64,    // Low 32 bits of r8
        gpr_r9d_x86_64,    // Low 32 bits of r9
        gpr_r10d_x86_64,   // Low 32 bits of r10
        gpr_r11d_x86_64,   // Low 32 bits of r11
        gpr_r12d_x86_64,   // Low 32 bits of r12
        gpr_r13d_x86_64,   // Low 32 bits of r13
        gpr_r14d_x86_64,   // Low 32 bits of r14
        gpr_r15d_x86_64,   // Low 32 bits of r15
        gpr_ax_x86_64,
        gpr_bx_x86_64,
        gpr_cx_x86_64,
        gpr_dx_x86_64,
        gpr_di_x86_64,
        gpr_si_x86_64,
        gpr_bp_x86_64,
        gpr_sp_x86_64,
        gpr_r8w_x86_64,    // Low 16 bits of r8
        gpr_r9w_x86_64,    // Low 16 bits of r9
        gpr_r10w_x86_64,   // Low 16 bits of r10
        gpr_r11w_x86_64,   // Low 16 bits of r11
        gpr_r12w_x86_64,   // Low 16 bits of r12
        gpr_r13w_x86_64,   // Low 16 bits of r13
        gpr_r14w_x86_64,   // Low 16 bits of r14
        gpr_r15w_x86_64,   // Low 16 bits of r15
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
        gpr_r8l_x86_64,    // Low 8 bits of r8
        gpr_r9l_x86_64,    // Low 8 bits of r9
        gpr_r10l_x86_64,   // Low 8 bits of r10
        gpr_r11l_x86_64,   // Low 8 bits of r11
        gpr_r12l_x86_64,   // Low 8 bits of r12
        gpr_r13l_x86_64,   // Low 8 bits of r13
        gpr_r14l_x86_64,   // Low 8 bits of r14
        gpr_r15l_x86_64,   // Low 8 bits of r15
        k_last_alias_x86_64 = gpr_r15l_x86_64,

        k_last_gpr_x86_64 = k_last_alias_x86_64,

        k_first_fpr_x86_64,
        fpu_fctrl_x86_64 = k_first_fpr_x86_64,
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
        fpu_xmm15_x86_64,
        k_last_fpr_x86_64 = fpu_xmm15_x86_64,

        k_first_avx_x86_64,
        fpu_ymm0_x86_64 = k_first_avx_x86_64,
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
        fpu_ymm15_x86_64,
        k_last_avx_x86_64 = fpu_ymm15_x86_64,

        dr0_x86_64,
        dr1_x86_64,
        dr2_x86_64,
        dr3_x86_64,
        dr4_x86_64,
        dr5_x86_64,
        dr6_x86_64,
        dr7_x86_64,

        k_num_registers_x86_64,
        k_num_gpr_registers_x86_64 = k_last_gpr_x86_64 - k_first_gpr_x86_64 + 1,
        k_num_fpr_registers_x86_64 = k_last_fpr_x86_64 - k_first_fpr_x86_64 + 1,
        k_num_avx_registers_x86_64 = k_last_avx_x86_64 - k_first_avx_x86_64 + 1
    };

}

#endif // #ifndef lldb_x86_register_enums_h
