//===-- CodeViewRegisterMapping.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeViewRegisterMapping.h"

#include "lldb/lldb-defines.h"

#include "Plugins/Process/Utility/lldb-x86-register-enums.h"

using namespace lldb_private;

static const uint32_t g_code_view_to_lldb_registers_x86[] = {
    LLDB_INVALID_REGNUM, // NONE
    lldb_al_i386,        // AL
    lldb_cl_i386,        // CL
    lldb_dl_i386,        // DL
    lldb_bl_i386,        // BL
    lldb_ah_i386,        // AH
    lldb_ch_i386,        // CH
    lldb_dh_i386,        // DH
    lldb_bh_i386,        // BH
    lldb_ax_i386,        // AX
    lldb_cx_i386,        // CX
    lldb_dx_i386,        // DX
    lldb_bx_i386,        // BX
    lldb_sp_i386,        // SP
    lldb_bp_i386,        // BP
    lldb_si_i386,        // SI
    lldb_di_i386,        // DI
    lldb_eax_i386,       // EAX
    lldb_ecx_i386,       // ECX
    lldb_edx_i386,       // EDX
    lldb_ebx_i386,       // EBX
    lldb_esp_i386,       // ESP
    lldb_ebp_i386,       // EBP
    lldb_esi_i386,       // ESI
    lldb_edi_i386,       // EDI
    lldb_es_i386,        // ES
    lldb_cs_i386,        // CS
    lldb_ss_i386,        // SS
    lldb_ds_i386,        // DS
    lldb_fs_i386,        // FS
    lldb_gs_i386,        // GS
    LLDB_INVALID_REGNUM, // IP
    LLDB_INVALID_REGNUM, // FLAGS
    lldb_eip_i386,       // EIP
    lldb_eflags_i386,    // EFLAGS
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // TEMP
    LLDB_INVALID_REGNUM, // TEMPH
    LLDB_INVALID_REGNUM, // QUOTE
    LLDB_INVALID_REGNUM, // PCDR3
    LLDB_INVALID_REGNUM, // PCDR4
    LLDB_INVALID_REGNUM, // PCDR5
    LLDB_INVALID_REGNUM, // PCDR6
    LLDB_INVALID_REGNUM, // PCDR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // CR0
    LLDB_INVALID_REGNUM, // CR1
    LLDB_INVALID_REGNUM, // CR2
    LLDB_INVALID_REGNUM, // CR3
    LLDB_INVALID_REGNUM, // CR4
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    lldb_dr0_i386, // DR0
    lldb_dr1_i386, // DR1
    lldb_dr2_i386, // DR2
    lldb_dr3_i386, // DR3
    lldb_dr4_i386, // DR4
    lldb_dr5_i386, // DR5
    lldb_dr6_i386, // DR6
    lldb_dr7_i386, // DR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // GDTR
    LLDB_INVALID_REGNUM, // GDTL
    LLDB_INVALID_REGNUM, // IDTR
    LLDB_INVALID_REGNUM, // IDTL
    LLDB_INVALID_REGNUM, // LDTR
    LLDB_INVALID_REGNUM, // TR
    LLDB_INVALID_REGNUM, // PSEUDO1
    LLDB_INVALID_REGNUM, // PSEUDO2
    LLDB_INVALID_REGNUM, // PSEUDO3
    LLDB_INVALID_REGNUM, // PSEUDO4
    LLDB_INVALID_REGNUM, // PSEUDO5
    LLDB_INVALID_REGNUM, // PSEUDO6
    LLDB_INVALID_REGNUM, // PSEUDO7
    LLDB_INVALID_REGNUM, // PSEUDO8
    LLDB_INVALID_REGNUM, // PSEUDO9
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    lldb_st0_i386,       // ST0
    lldb_st1_i386,       // ST1
    lldb_st2_i386,       // ST2
    lldb_st3_i386,       // ST3
    lldb_st4_i386,       // ST4
    lldb_st5_i386,       // ST5
    lldb_st6_i386,       // ST6
    lldb_st7_i386,       // ST7
    LLDB_INVALID_REGNUM, // CTRL
    LLDB_INVALID_REGNUM, // STAT
    LLDB_INVALID_REGNUM, // TAG
    LLDB_INVALID_REGNUM, // FPIP
    LLDB_INVALID_REGNUM, // FPCS
    LLDB_INVALID_REGNUM, // FPDO
    LLDB_INVALID_REGNUM, // FPDS
    LLDB_INVALID_REGNUM, // ISEM
    LLDB_INVALID_REGNUM, // FPEIP
    LLDB_INVALID_REGNUM, // FPEDO
    lldb_mm0_i386,       // MM0
    lldb_mm1_i386,       // MM1
    lldb_mm2_i386,       // MM2
    lldb_mm3_i386,       // MM3
    lldb_mm4_i386,       // MM4
    lldb_mm5_i386,       // MM5
    lldb_mm6_i386,       // MM6
    lldb_mm7_i386,       // MM7
    lldb_xmm0_i386,      // XMM0
    lldb_xmm1_i386,      // XMM1
    lldb_xmm2_i386,      // XMM2
    lldb_xmm3_i386,      // XMM3
    lldb_xmm4_i386,      // XMM4
    lldb_xmm5_i386,      // XMM5
    lldb_xmm6_i386,      // XMM6
    lldb_xmm7_i386       // XMM7
};

static const uint32_t g_code_view_to_lldb_registers_x86_64[] = {
    LLDB_INVALID_REGNUM, // NONE
    lldb_al_x86_64,      // AL
    lldb_cl_x86_64,      // CL
    lldb_dl_x86_64,      // DL
    lldb_bl_x86_64,      // BL
    lldb_ah_x86_64,      // AH
    lldb_ch_x86_64,      // CH
    lldb_dh_x86_64,      // DH
    lldb_bh_x86_64,      // BH
    lldb_ax_x86_64,      // AX
    lldb_cx_x86_64,      // CX
    lldb_dx_x86_64,      // DX
    lldb_bx_x86_64,      // BX
    lldb_sp_x86_64,      // SP
    lldb_bp_x86_64,      // BP
    lldb_si_x86_64,      // SI
    lldb_di_x86_64,      // DI
    lldb_eax_x86_64,     // EAX
    lldb_ecx_x86_64,     // ECX
    lldb_edx_x86_64,     // EDX
    lldb_ebx_x86_64,     // EBX
    lldb_esp_x86_64,     // ESP
    lldb_ebp_x86_64,     // EBP
    lldb_esi_x86_64,     // ESI
    lldb_edi_x86_64,     // EDI
    lldb_es_x86_64,      // ES
    lldb_cs_x86_64,      // CS
    lldb_ss_x86_64,      // SS
    lldb_ds_x86_64,      // DS
    lldb_fs_x86_64,      // FS
    lldb_gs_x86_64,      // GS
    LLDB_INVALID_REGNUM, // IP
    LLDB_INVALID_REGNUM, // FLAGS
    LLDB_INVALID_REGNUM, // EIP
    LLDB_INVALID_REGNUM, // EFLAGS
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // TEMP
    LLDB_INVALID_REGNUM, // TEMPH
    LLDB_INVALID_REGNUM, // QUOTE
    LLDB_INVALID_REGNUM, // PCDR3
    LLDB_INVALID_REGNUM, // PCDR4
    LLDB_INVALID_REGNUM, // PCDR5
    LLDB_INVALID_REGNUM, // PCDR6
    LLDB_INVALID_REGNUM, // PCDR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // CR0
    LLDB_INVALID_REGNUM, // CR1
    LLDB_INVALID_REGNUM, // CR2
    LLDB_INVALID_REGNUM, // CR3
    LLDB_INVALID_REGNUM, // CR4
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    lldb_dr0_x86_64, // DR0
    lldb_dr1_x86_64, // DR1
    lldb_dr2_x86_64, // DR2
    lldb_dr3_x86_64, // DR3
    lldb_dr4_x86_64, // DR4
    lldb_dr5_x86_64, // DR5
    lldb_dr6_x86_64, // DR6
    lldb_dr7_x86_64, // DR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // GDTR
    LLDB_INVALID_REGNUM, // GDTL
    LLDB_INVALID_REGNUM, // IDTR
    LLDB_INVALID_REGNUM, // IDTL
    LLDB_INVALID_REGNUM, // LDTR
    LLDB_INVALID_REGNUM, // TR
    LLDB_INVALID_REGNUM, // PSEUDO1
    LLDB_INVALID_REGNUM, // PSEUDO2
    LLDB_INVALID_REGNUM, // PSEUDO3
    LLDB_INVALID_REGNUM, // PSEUDO4
    LLDB_INVALID_REGNUM, // PSEUDO5
    LLDB_INVALID_REGNUM, // PSEUDO6
    LLDB_INVALID_REGNUM, // PSEUDO7
    LLDB_INVALID_REGNUM, // PSEUDO8
    LLDB_INVALID_REGNUM, // PSEUDO9
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    lldb_st0_x86_64,     // ST0
    lldb_st1_x86_64,     // ST1
    lldb_st2_x86_64,     // ST2
    lldb_st3_x86_64,     // ST3
    lldb_st4_x86_64,     // ST4
    lldb_st5_x86_64,     // ST5
    lldb_st6_x86_64,     // ST6
    lldb_st7_x86_64,     // ST7
    LLDB_INVALID_REGNUM, // CTRL
    LLDB_INVALID_REGNUM, // STAT
    LLDB_INVALID_REGNUM, // TAG
    LLDB_INVALID_REGNUM, // FPIP
    LLDB_INVALID_REGNUM, // FPCS
    LLDB_INVALID_REGNUM, // FPDO
    LLDB_INVALID_REGNUM, // FPDS
    LLDB_INVALID_REGNUM, // ISEM
    LLDB_INVALID_REGNUM, // FPEIP
    LLDB_INVALID_REGNUM, // FPEDO
    lldb_mm0_x86_64,     // MM0
    lldb_mm1_x86_64,     // MM1
    lldb_mm2_x86_64,     // MM2
    lldb_mm3_x86_64,     // MM3
    lldb_mm4_x86_64,     // MM4
    lldb_mm5_x86_64,     // MM5
    lldb_mm6_x86_64,     // MM6
    lldb_mm7_x86_64,     // MM7
    lldb_xmm0_x86_64,    // XMM0
    lldb_xmm1_x86_64,    // XMM1
    lldb_xmm2_x86_64,    // XMM2
    lldb_xmm3_x86_64,    // XMM3
    lldb_xmm4_x86_64,    // XMM4
    lldb_xmm5_x86_64,    // XMM5
    lldb_xmm6_x86_64,    // XMM6
    lldb_xmm7_x86_64,    // XMM7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM,
    lldb_mxcsr_x86_64,   // MXCSR
    LLDB_INVALID_REGNUM, // EDXEAX
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // EMM0L
    LLDB_INVALID_REGNUM, // EMM1L
    LLDB_INVALID_REGNUM, // EMM2L
    LLDB_INVALID_REGNUM, // EMM3L
    LLDB_INVALID_REGNUM, // EMM4L
    LLDB_INVALID_REGNUM, // EMM5L
    LLDB_INVALID_REGNUM, // EMM6L
    LLDB_INVALID_REGNUM, // EMM7L
    LLDB_INVALID_REGNUM, // EMM0H
    LLDB_INVALID_REGNUM, // EMM1H
    LLDB_INVALID_REGNUM, // EMM2H
    LLDB_INVALID_REGNUM, // EMM3H
    LLDB_INVALID_REGNUM, // EMM4H
    LLDB_INVALID_REGNUM, // EMM5H
    LLDB_INVALID_REGNUM, // EMM6H
    LLDB_INVALID_REGNUM, // EMM7H
    LLDB_INVALID_REGNUM, // MM00
    LLDB_INVALID_REGNUM, // MM01
    LLDB_INVALID_REGNUM, // MM10
    LLDB_INVALID_REGNUM, // MM11
    LLDB_INVALID_REGNUM, // MM20
    LLDB_INVALID_REGNUM, // MM21
    LLDB_INVALID_REGNUM, // MM30
    LLDB_INVALID_REGNUM, // MM31
    LLDB_INVALID_REGNUM, // MM40
    LLDB_INVALID_REGNUM, // MM41
    LLDB_INVALID_REGNUM, // MM50
    LLDB_INVALID_REGNUM, // MM51
    LLDB_INVALID_REGNUM, // MM60
    LLDB_INVALID_REGNUM, // MM61
    LLDB_INVALID_REGNUM, // MM70
    LLDB_INVALID_REGNUM, // MM71
    lldb_xmm8_x86_64,    // XMM8
    lldb_xmm9_x86_64,    // XMM9
    lldb_xmm10_x86_64,   // XMM10
    lldb_xmm11_x86_64,   // XMM11
    lldb_xmm12_x86_64,   // XMM12
    lldb_xmm13_x86_64,   // XMM13
    lldb_xmm14_x86_64,   // XMM14
    lldb_xmm15_x86_64,   // XMM15
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM,
    lldb_sil_x86_64,   // SIL
    lldb_dil_x86_64,   // DIL
    lldb_bpl_x86_64,   // BPL
    lldb_spl_x86_64,   // SPL
    lldb_rax_x86_64,   // RAX
    lldb_rbx_x86_64,   // RBX
    lldb_rcx_x86_64,   // RCX
    lldb_rdx_x86_64,   // RDX
    lldb_rsi_x86_64,   // RSI
    lldb_rdi_x86_64,   // RDI
    lldb_rbp_x86_64,   // RBP
    lldb_rsp_x86_64,   // RSP
    lldb_r8_x86_64,    // R8
    lldb_r9_x86_64,    // R9
    lldb_r10_x86_64,   // R10
    lldb_r11_x86_64,   // R11
    lldb_r12_x86_64,   // R12
    lldb_r13_x86_64,   // R13
    lldb_r14_x86_64,   // R14
    lldb_r15_x86_64,   // R15
    lldb_r8l_x86_64,   // R8B
    lldb_r9l_x86_64,   // R9B
    lldb_r10l_x86_64,  // R10B
    lldb_r11l_x86_64,  // R11B
    lldb_r12l_x86_64,  // R12B
    lldb_r13l_x86_64,  // R13B
    lldb_r14l_x86_64,  // R14B
    lldb_r15l_x86_64,  // R15B
    lldb_r8w_x86_64,   // R8W
    lldb_r9w_x86_64,   // R9W
    lldb_r10w_x86_64,  // R10W
    lldb_r11w_x86_64,  // R11W
    lldb_r12w_x86_64,  // R12W
    lldb_r13w_x86_64,  // R13W
    lldb_r14w_x86_64,  // R14W
    lldb_r15w_x86_64,  // R15W
    lldb_r8d_x86_64,   // R8D
    lldb_r9d_x86_64,   // R9D
    lldb_r10d_x86_64,  // R10D
    lldb_r11d_x86_64,  // R11D
    lldb_r12d_x86_64,  // R12D
    lldb_r13d_x86_64,  // R13D
    lldb_r14d_x86_64,  // R14D
    lldb_r15d_x86_64,  // R15D
    lldb_ymm0_x86_64,  // AMD64_YMM0
    lldb_ymm1_x86_64,  // AMD64_YMM1
    lldb_ymm2_x86_64,  // AMD64_YMM2
    lldb_ymm3_x86_64,  // AMD64_YMM3
    lldb_ymm4_x86_64,  // AMD64_YMM4
    lldb_ymm5_x86_64,  // AMD64_YMM5
    lldb_ymm6_x86_64,  // AMD64_YMM6
    lldb_ymm7_x86_64,  // AMD64_YMM7
    lldb_ymm8_x86_64,  // AMD64_YMM8
    lldb_ymm9_x86_64,  // AMD64_YMM9
    lldb_ymm10_x86_64, // AMD64_YMM10
    lldb_ymm11_x86_64, // AMD64_YMM11
    lldb_ymm12_x86_64, // AMD64_YMM12
    lldb_ymm13_x86_64, // AMD64_YMM13
    lldb_ymm14_x86_64, // AMD64_YMM14
    lldb_ymm15_x86_64, // AMD64_YMM15
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    lldb_bnd0_x86_64, // BND0
    lldb_bnd1_x86_64, // BND1
    lldb_bnd2_x86_64  // BND2
};

uint32_t lldb_private::npdb::GetLLDBRegisterNumber(
    llvm::Triple::ArchType arch_type, llvm::codeview::RegisterId register_id) {
  switch (arch_type) {
  case llvm::Triple::x86:
    if (static_cast<uint16_t>(register_id) <
        sizeof(g_code_view_to_lldb_registers_x86) /
            sizeof(g_code_view_to_lldb_registers_x86[0]))
      return g_code_view_to_lldb_registers_x86[static_cast<uint16_t>(
          register_id)];

    switch (register_id) {
    case llvm::codeview::RegisterId::MXCSR:
      return lldb_mxcsr_i386;
    case llvm::codeview::RegisterId::BND0:
      return lldb_bnd0_i386;
    case llvm::codeview::RegisterId::BND1:
      return lldb_bnd1_i386;
    case llvm::codeview::RegisterId::BND2:
      return lldb_bnd2_i386;
    default:
      return LLDB_INVALID_REGNUM;
    }
  case llvm::Triple::x86_64:
    if (static_cast<uint16_t>(register_id) <
        sizeof(g_code_view_to_lldb_registers_x86_64) /
            sizeof(g_code_view_to_lldb_registers_x86_64[0]))
      return g_code_view_to_lldb_registers_x86_64[static_cast<uint16_t>(
          register_id)];

    return LLDB_INVALID_REGNUM;
  default:
    return LLDB_INVALID_REGNUM;
  }
}
