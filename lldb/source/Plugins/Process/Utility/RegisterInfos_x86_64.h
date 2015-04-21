//===-- RegisterInfos_x86_64.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "llvm/Support/Compiler.h"

#include <stddef.h>

// Computes the offset of the given GPR in the user data area.
#define GPR_OFFSET(regname) \
    (LLVM_EXTENSION offsetof(GPR, regname))

// Computes the offset of the given FPR in the extended data area.
#define FPR_OFFSET(regname) \
    (LLVM_EXTENSION offsetof(UserArea, fpr) + \
     LLVM_EXTENSION offsetof(FPR, xstate) + \
     LLVM_EXTENSION offsetof(FXSAVE, regname))

// Computes the offset of the YMM register assembled from register halves.
// Based on DNBArchImplX86_64.cpp from debugserver
#define YMM_OFFSET(reg_index) \
    (LLVM_EXTENSION offsetof(UserArea, fpr) + \
     LLVM_EXTENSION offsetof(FPR, xstate) + \
     LLVM_EXTENSION offsetof(XSAVE, ymmh[0]) + \
     (32 * reg_index))

#ifdef DECLARE_REGISTER_INFOS_X86_64_STRUCT

// Number of bytes needed to represent a FPR.
#define FPR_SIZE(reg) sizeof(((FXSAVE*)NULL)->reg)

// Number of bytes needed to represent the i'th FP register.
#define FP_SIZE sizeof(((MMSReg*)NULL)->bytes)

// Number of bytes needed to represent an XMM register.
#define XMM_SIZE sizeof(XMMReg)

// Number of bytes needed to represent a YMM register.
#define YMM_SIZE sizeof(YMMReg)

#define DR_SIZE sizeof(((DBG*)NULL)->dr[0])

// RegisterKind: GCC, DWARF, Generic, GDB, LLDB

// Note that the size and offset will be updated by platform-specific classes.
#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)    \
    { #reg, alt, sizeof(((GPR*)NULL)->reg), GPR_OFFSET(reg), eEncodingUint, \
      eFormatHex, { kind1, kind2, kind3, kind4, lldb_##reg##_x86_64 }, NULL, NULL }

#define DEFINE_FPR(name, reg, kind1, kind2, kind3, kind4)    \
    { #name, NULL, FPR_SIZE(reg), FPR_OFFSET(reg), eEncodingUint,   \
      eFormatHex, { kind1, kind2, kind3, kind4, lldb_##name##_x86_64 }, NULL, NULL }

#define DEFINE_FP_ST(reg, i)                                       \
    { #reg#i, NULL, FP_SIZE, LLVM_EXTENSION FPR_OFFSET(stmm[i]),   \
      eEncodingVector, eFormatVectorOfUInt8,                       \
      { gcc_dwarf_st##i##_x86_64, gcc_dwarf_st##i##_x86_64, LLDB_INVALID_REGNUM, gdb_st##i##_x86_64, lldb_st##i##_x86_64 }, \
      NULL, NULL }

#define DEFINE_FP_MM(reg, i)                                                \
    { #reg#i, NULL, sizeof(uint64_t), LLVM_EXTENSION FPR_OFFSET(stmm[i]),   \
      eEncodingUint, eFormatHex,                                            \
      { gcc_dwarf_mm##i##_x86_64, gcc_dwarf_mm##i##_x86_64, LLDB_INVALID_REGNUM, gdb_st##i##_x86_64, lldb_mm##i##_x86_64 }, \
      NULL, NULL }

#define DEFINE_XMM(reg, i)                                         \
    { #reg#i, NULL, XMM_SIZE, LLVM_EXTENSION FPR_OFFSET(reg[i]),   \
      eEncodingVector, eFormatVectorOfUInt8,                       \
      { gcc_dwarf_##reg##i##_x86_64, gcc_dwarf_##reg##i##_x86_64, LLDB_INVALID_REGNUM, gdb_##reg##i##_x86_64, lldb_##reg##i##_x86_64}, \
      NULL, NULL }

#define DEFINE_YMM(reg, i)                                                          \
    { #reg#i, NULL, YMM_SIZE, LLVM_EXTENSION YMM_OFFSET(i),                         \
      eEncodingVector, eFormatVectorOfUInt8,                                        \
      { gcc_dwarf_##reg##i##h_x86_64, gcc_dwarf_##reg##i##h_x86_64, LLDB_INVALID_REGNUM, gdb_##reg##i##h_x86_64, lldb_##reg##i##_x86_64 }, \
      NULL, NULL }

#define DEFINE_DR(reg, i)                                               \
    { #reg#i, NULL, DR_SIZE, DR_OFFSET(i), eEncodingUint, eFormatHex,   \
      { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,  \
      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM }, NULL, NULL }

#define DEFINE_GPR_PSEUDO_32(reg32, reg64)          \
    { #reg32, NULL, 4, GPR_OFFSET(reg64), eEncodingUint,   \
      eFormatHex, { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, lldb_##reg32##_x86_64 }, RegisterContextPOSIX_x86::g_contained_##reg64, RegisterContextPOSIX_x86::g_invalidate_##reg64 }
#define DEFINE_GPR_PSEUDO_16(reg16, reg64)          \
    { #reg16, NULL, 2, GPR_OFFSET(reg64), eEncodingUint,   \
      eFormatHex, { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, lldb_##reg16##_x86_64 }, RegisterContextPOSIX_x86::g_contained_##reg64, RegisterContextPOSIX_x86::g_invalidate_##reg64 }
#define DEFINE_GPR_PSEUDO_8H(reg8, reg64)           \
    { #reg8, NULL, 1, GPR_OFFSET(reg64)+1, eEncodingUint,  \
      eFormatHex, { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, lldb_##reg8##_x86_64 }, RegisterContextPOSIX_x86::g_contained_##reg64, RegisterContextPOSIX_x86::g_invalidate_##reg64 }
#define DEFINE_GPR_PSEUDO_8L(reg8, reg64)           \
    { #reg8, NULL, 1, GPR_OFFSET(reg64), eEncodingUint,    \
      eFormatHex, { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, lldb_##reg8##_x86_64 }, RegisterContextPOSIX_x86::g_contained_##reg64, RegisterContextPOSIX_x86::g_invalidate_##reg64 }

static RegisterInfo
g_register_infos_x86_64[] =
{
    // General purpose registers.                GCC,                   DWARF,                Generic,                  GDB
    DEFINE_GPR(rax,    NULL,    gcc_dwarf_rax_x86_64,    gcc_dwarf_rax_x86_64,    LLDB_INVALID_REGNUM,       gdb_rax_x86_64),
    DEFINE_GPR(rbx,    NULL,    gcc_dwarf_rbx_x86_64,    gcc_dwarf_rbx_x86_64,    LLDB_INVALID_REGNUM,       gdb_rbx_x86_64),
    DEFINE_GPR(rcx,    "arg4",  gcc_dwarf_rcx_x86_64,    gcc_dwarf_rcx_x86_64,    LLDB_REGNUM_GENERIC_ARG4,  gdb_rcx_x86_64),
    DEFINE_GPR(rdx,    "arg3",  gcc_dwarf_rdx_x86_64,    gcc_dwarf_rdx_x86_64,    LLDB_REGNUM_GENERIC_ARG3,  gdb_rdx_x86_64),
    DEFINE_GPR(rdi,    "arg1",  gcc_dwarf_rdi_x86_64,    gcc_dwarf_rdi_x86_64,    LLDB_REGNUM_GENERIC_ARG1,  gdb_rdi_x86_64),
    DEFINE_GPR(rsi,    "arg2",  gcc_dwarf_rsi_x86_64,    gcc_dwarf_rsi_x86_64,    LLDB_REGNUM_GENERIC_ARG2,  gdb_rsi_x86_64),
    DEFINE_GPR(rbp,    "fp",    gcc_dwarf_rbp_x86_64,    gcc_dwarf_rbp_x86_64,    LLDB_REGNUM_GENERIC_FP,    gdb_rbp_x86_64),
    DEFINE_GPR(rsp,    "sp",    gcc_dwarf_rsp_x86_64,    gcc_dwarf_rsp_x86_64,    LLDB_REGNUM_GENERIC_SP,    gdb_rsp_x86_64),
    DEFINE_GPR(r8,     "arg5",  gcc_dwarf_r8_x86_64,     gcc_dwarf_r8_x86_64,     LLDB_REGNUM_GENERIC_ARG5,  gdb_r8_x86_64),
    DEFINE_GPR(r9,     "arg6",  gcc_dwarf_r9_x86_64,     gcc_dwarf_r9_x86_64,     LLDB_REGNUM_GENERIC_ARG6,  gdb_r9_x86_64),
    DEFINE_GPR(r10,    NULL,    gcc_dwarf_r10_x86_64,    gcc_dwarf_r10_x86_64,    LLDB_INVALID_REGNUM,       gdb_r10_x86_64),
    DEFINE_GPR(r11,    NULL,    gcc_dwarf_r11_x86_64,    gcc_dwarf_r11_x86_64,    LLDB_INVALID_REGNUM,       gdb_r11_x86_64),
    DEFINE_GPR(r12,    NULL,    gcc_dwarf_r12_x86_64,    gcc_dwarf_r12_x86_64,    LLDB_INVALID_REGNUM,       gdb_r12_x86_64),
    DEFINE_GPR(r13,    NULL,    gcc_dwarf_r13_x86_64,    gcc_dwarf_r13_x86_64,    LLDB_INVALID_REGNUM,       gdb_r13_x86_64),
    DEFINE_GPR(r14,    NULL,    gcc_dwarf_r14_x86_64,    gcc_dwarf_r14_x86_64,    LLDB_INVALID_REGNUM,       gdb_r14_x86_64),
    DEFINE_GPR(r15,    NULL,    gcc_dwarf_r15_x86_64,    gcc_dwarf_r15_x86_64,    LLDB_INVALID_REGNUM,       gdb_r15_x86_64),
    DEFINE_GPR(rip,    "pc",    gcc_dwarf_rip_x86_64,    gcc_dwarf_rip_x86_64,    LLDB_REGNUM_GENERIC_PC,    gdb_rip_x86_64),
    DEFINE_GPR(rflags, "flags", gcc_dwarf_rflags_x86_64, gcc_dwarf_rflags_x86_64, LLDB_REGNUM_GENERIC_FLAGS, gdb_rflags_x86_64),
    DEFINE_GPR(cs,     NULL,    gcc_dwarf_cs_x86_64,     gcc_dwarf_cs_x86_64,     LLDB_INVALID_REGNUM,       gdb_cs_x86_64),
    DEFINE_GPR(fs,     NULL,    gcc_dwarf_fs_x86_64,     gcc_dwarf_fs_x86_64,     LLDB_INVALID_REGNUM,       gdb_fs_x86_64),
    DEFINE_GPR(gs,     NULL,    gcc_dwarf_gs_x86_64,     gcc_dwarf_gs_x86_64,     LLDB_INVALID_REGNUM,       gdb_gs_x86_64),
    DEFINE_GPR(ss,     NULL,    gcc_dwarf_ss_x86_64,     gcc_dwarf_ss_x86_64,     LLDB_INVALID_REGNUM,       gdb_ss_x86_64),
    DEFINE_GPR(ds,     NULL,    gcc_dwarf_ds_x86_64,     gcc_dwarf_ds_x86_64,     LLDB_INVALID_REGNUM,       gdb_ds_x86_64),
    DEFINE_GPR(es,     NULL,    gcc_dwarf_es_x86_64,     gcc_dwarf_es_x86_64,     LLDB_INVALID_REGNUM,       gdb_es_x86_64),

    DEFINE_GPR_PSEUDO_32(eax, rax),
    DEFINE_GPR_PSEUDO_32(ebx, rbx),
    DEFINE_GPR_PSEUDO_32(ecx, rcx),
    DEFINE_GPR_PSEUDO_32(edx, rdx),
    DEFINE_GPR_PSEUDO_32(edi, rdi),
    DEFINE_GPR_PSEUDO_32(esi, rsi),
    DEFINE_GPR_PSEUDO_32(ebp, rbp),
    DEFINE_GPR_PSEUDO_32(esp, rsp),
    DEFINE_GPR_PSEUDO_32(r8d,  r8),
    DEFINE_GPR_PSEUDO_32(r9d,  r9),
    DEFINE_GPR_PSEUDO_32(r10d, r10),
    DEFINE_GPR_PSEUDO_32(r11d, r11),
    DEFINE_GPR_PSEUDO_32(r12d, r12),
    DEFINE_GPR_PSEUDO_32(r13d, r13),
    DEFINE_GPR_PSEUDO_32(r14d, r14),
    DEFINE_GPR_PSEUDO_32(r15d, r15),
    DEFINE_GPR_PSEUDO_16(ax,   rax),
    DEFINE_GPR_PSEUDO_16(bx,   rbx),
    DEFINE_GPR_PSEUDO_16(cx,   rcx),
    DEFINE_GPR_PSEUDO_16(dx,   rdx),
    DEFINE_GPR_PSEUDO_16(di,   rdi),
    DEFINE_GPR_PSEUDO_16(si,   rsi),
    DEFINE_GPR_PSEUDO_16(bp,   rbp),
    DEFINE_GPR_PSEUDO_16(sp,   rsp),
    DEFINE_GPR_PSEUDO_16(r8w,  r8),
    DEFINE_GPR_PSEUDO_16(r9w,  r9),
    DEFINE_GPR_PSEUDO_16(r10w, r10),
    DEFINE_GPR_PSEUDO_16(r11w, r11),
    DEFINE_GPR_PSEUDO_16(r12w, r12),
    DEFINE_GPR_PSEUDO_16(r13w, r13),
    DEFINE_GPR_PSEUDO_16(r14w, r14),
    DEFINE_GPR_PSEUDO_16(r15w, r15),
    DEFINE_GPR_PSEUDO_8H(ah,   rax),
    DEFINE_GPR_PSEUDO_8H(bh,   rbx),
    DEFINE_GPR_PSEUDO_8H(ch,   rcx),
    DEFINE_GPR_PSEUDO_8H(dh,   rdx),
    DEFINE_GPR_PSEUDO_8L(al,   rax),
    DEFINE_GPR_PSEUDO_8L(bl,   rbx),
    DEFINE_GPR_PSEUDO_8L(cl,   rcx),
    DEFINE_GPR_PSEUDO_8L(dl,   rdx),
    DEFINE_GPR_PSEUDO_8L(dil,  rdi),
    DEFINE_GPR_PSEUDO_8L(sil,  rsi),
    DEFINE_GPR_PSEUDO_8L(bpl,  rbp),
    DEFINE_GPR_PSEUDO_8L(spl,  rsp),
    DEFINE_GPR_PSEUDO_8L(r8l,  r8),
    DEFINE_GPR_PSEUDO_8L(r9l,  r9),
    DEFINE_GPR_PSEUDO_8L(r10l, r10),
    DEFINE_GPR_PSEUDO_8L(r11l, r11),
    DEFINE_GPR_PSEUDO_8L(r12l, r12),
    DEFINE_GPR_PSEUDO_8L(r13l, r13),
    DEFINE_GPR_PSEUDO_8L(r14l, r14),
    DEFINE_GPR_PSEUDO_8L(r15l, r15),

    // i387 Floating point registers.     GCC,                                   DWARF,               Generic,            GDB
    DEFINE_FPR(fctrl,     fctrl,          gcc_dwarf_fctrl_x86_64, gcc_dwarf_fctrl_x86_64, LLDB_INVALID_REGNUM, gdb_fctrl_x86_64),
    DEFINE_FPR(fstat,     fstat,          gcc_dwarf_fstat_x86_64, gcc_dwarf_fstat_x86_64, LLDB_INVALID_REGNUM, gdb_fstat_x86_64),
    DEFINE_FPR(ftag,      ftag,           LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM, gdb_ftag_x86_64),
    DEFINE_FPR(fop,       fop,            LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM, gdb_fop_x86_64),
    DEFINE_FPR(fiseg,     ptr.i386_.fiseg, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM, gdb_fiseg_x86_64),
    DEFINE_FPR(fioff,     ptr.i386_.fioff, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM, gdb_fioff_x86_64),
    DEFINE_FPR(foseg,     ptr.i386_.foseg, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM, gdb_foseg_x86_64),
    DEFINE_FPR(fooff,     ptr.i386_.fooff, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM, gdb_fooff_x86_64),
    DEFINE_FPR(mxcsr,     mxcsr,          gcc_dwarf_mxcsr_x86_64, gcc_dwarf_mxcsr_x86_64, LLDB_INVALID_REGNUM, gdb_mxcsr_x86_64),
    DEFINE_FPR(mxcsrmask, mxcsrmask,      LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),

    // FP registers.
    DEFINE_FP_ST(st, 0),
    DEFINE_FP_ST(st, 1),
    DEFINE_FP_ST(st, 2),
    DEFINE_FP_ST(st, 3),
    DEFINE_FP_ST(st, 4),
    DEFINE_FP_ST(st, 5),
    DEFINE_FP_ST(st, 6),
    DEFINE_FP_ST(st, 7),
    DEFINE_FP_MM(mm, 0),
    DEFINE_FP_MM(mm, 1),
    DEFINE_FP_MM(mm, 2),
    DEFINE_FP_MM(mm, 3),
    DEFINE_FP_MM(mm, 4),
    DEFINE_FP_MM(mm, 5),
    DEFINE_FP_MM(mm, 6),
    DEFINE_FP_MM(mm, 7),

    // XMM registers
    DEFINE_XMM(xmm, 0),
    DEFINE_XMM(xmm, 1),
    DEFINE_XMM(xmm, 2),
    DEFINE_XMM(xmm, 3),
    DEFINE_XMM(xmm, 4),
    DEFINE_XMM(xmm, 5),
    DEFINE_XMM(xmm, 6),
    DEFINE_XMM(xmm, 7),
    DEFINE_XMM(xmm, 8),
    DEFINE_XMM(xmm, 9),
    DEFINE_XMM(xmm, 10),
    DEFINE_XMM(xmm, 11),
    DEFINE_XMM(xmm, 12),
    DEFINE_XMM(xmm, 13),
    DEFINE_XMM(xmm, 14),
    DEFINE_XMM(xmm, 15),

    // Copy of YMM registers assembled from xmm and ymmh
    DEFINE_YMM(ymm, 0),
    DEFINE_YMM(ymm, 1),
    DEFINE_YMM(ymm, 2),
    DEFINE_YMM(ymm, 3),
    DEFINE_YMM(ymm, 4),
    DEFINE_YMM(ymm, 5),
    DEFINE_YMM(ymm, 6),
    DEFINE_YMM(ymm, 7),
    DEFINE_YMM(ymm, 8),
    DEFINE_YMM(ymm, 9),
    DEFINE_YMM(ymm, 10),
    DEFINE_YMM(ymm, 11),
    DEFINE_YMM(ymm, 12),
    DEFINE_YMM(ymm, 13),
    DEFINE_YMM(ymm, 14),
    DEFINE_YMM(ymm, 15),

    // Debug registers for lldb internal use
    DEFINE_DR(dr, 0),
    DEFINE_DR(dr, 1),
    DEFINE_DR(dr, 2),
    DEFINE_DR(dr, 3),
    DEFINE_DR(dr, 4),
    DEFINE_DR(dr, 5),
    DEFINE_DR(dr, 6),
    DEFINE_DR(dr, 7)
};
static_assert((sizeof(g_register_infos_x86_64) / sizeof(g_register_infos_x86_64[0])) == k_num_registers_x86_64,
    "g_register_infos_x86_64 has wrong number of register infos");

#undef FPR_SIZE
#undef FP_SIZE
#undef XMM_SIZE
#undef YMM_SIZE
#undef DEFINE_GPR
#undef DEFINE_FPR
#undef DEFINE_FP
#undef DEFINE_XMM
#undef DEFINE_YMM
#undef DEFINE_DR
#undef DEFINE_GPR_PSEUDO_32
#undef DEFINE_GPR_PSEUDO_16
#undef DEFINE_GPR_PSEUDO_8H
#undef DEFINE_GPR_PSEUDO_8L

#endif // DECLARE_REGISTER_INFOS_X86_64_STRUCT


#ifdef UPDATE_REGISTER_INFOS_I386_STRUCT_WITH_X86_64_OFFSETS

#define UPDATE_GPR_INFO(reg, reg64)                                             \
do {                                                                            \
    g_register_infos[lldb_##reg##_i386].byte_offset = GPR_OFFSET(reg64);         \
} while(false);

#define UPDATE_GPR_INFO_8H(reg, reg64)                                          \
do {                                                                            \
    g_register_infos[lldb_##reg##_i386].byte_offset = GPR_OFFSET(reg64) + 1;     \
} while(false);

#define UPDATE_FPR_INFO(reg, reg64)                                             \
do {                                                                            \
    g_register_infos[lldb_##reg##_i386].byte_offset = FPR_OFFSET(reg64);         \
} while(false);

#define UPDATE_FP_INFO(reg, i)                                                  \
do {                                                                            \
    g_register_infos[lldb_##reg##i##_i386].byte_offset = FPR_OFFSET(stmm[i]);    \
} while(false);

#define UPDATE_XMM_INFO(reg, i)                                                 \
do {                                                                            \
    g_register_infos[lldb_##reg##i##_i386].byte_offset = FPR_OFFSET(reg[i]);     \
} while(false);

#define UPDATE_YMM_INFO(reg, i)                                                 \
do {                                                                            \
    g_register_infos[lldb_##reg##i##_i386].byte_offset = YMM_OFFSET(i);         \
} while(false);

#define UPDATE_DR_INFO(reg_index)                                               \
do {                                                                            \
    g_register_infos[lldb_dr##reg_index##_i386].byte_offset = DR_OFFSET(reg_index);  \
} while(false);

    // Update the register offsets
    UPDATE_GPR_INFO(eax,    rax);
    UPDATE_GPR_INFO(ebx,    rbx);
    UPDATE_GPR_INFO(ecx,    rcx);
    UPDATE_GPR_INFO(edx,    rdx);
    UPDATE_GPR_INFO(edi,    rdi);
    UPDATE_GPR_INFO(esi,    rsi);
    UPDATE_GPR_INFO(ebp,    rbp);
    UPDATE_GPR_INFO(esp,    rsp);
    UPDATE_GPR_INFO(eip,    rip);
    UPDATE_GPR_INFO(eflags, rflags);
    UPDATE_GPR_INFO(cs,     cs);
    UPDATE_GPR_INFO(fs,     fs);
    UPDATE_GPR_INFO(gs,     gs);
    UPDATE_GPR_INFO(ss,     ss);
    UPDATE_GPR_INFO(ds,     ds);
    UPDATE_GPR_INFO(es,     es);

    UPDATE_GPR_INFO(ax,     rax);
    UPDATE_GPR_INFO(bx,     rbx);
    UPDATE_GPR_INFO(cx,     rcx);
    UPDATE_GPR_INFO(dx,     rdx);
    UPDATE_GPR_INFO(di,     rdi);
    UPDATE_GPR_INFO(si,     rsi);
    UPDATE_GPR_INFO(bp,     rbp);
    UPDATE_GPR_INFO(sp,     rsp);
    UPDATE_GPR_INFO_8H(ah,  rax);
    UPDATE_GPR_INFO_8H(bh,  rbx);
    UPDATE_GPR_INFO_8H(ch,  rcx);
    UPDATE_GPR_INFO_8H(dh,  rdx);
    UPDATE_GPR_INFO(al,     rax);
    UPDATE_GPR_INFO(bl,     rbx);
    UPDATE_GPR_INFO(cl,     rcx);
    UPDATE_GPR_INFO(dl,     rdx);

    UPDATE_FPR_INFO(fctrl,     fctrl);
    UPDATE_FPR_INFO(fstat,     fstat);
    UPDATE_FPR_INFO(ftag,      ftag);
    UPDATE_FPR_INFO(fop,       fop);
    UPDATE_FPR_INFO(fiseg,     ptr.i386_.fiseg);
    UPDATE_FPR_INFO(fioff,     ptr.i386_.fioff);
    UPDATE_FPR_INFO(fooff,     ptr.i386_.fooff);
    UPDATE_FPR_INFO(foseg,     ptr.i386_.foseg);
    UPDATE_FPR_INFO(mxcsr,     mxcsr);
    UPDATE_FPR_INFO(mxcsrmask, mxcsrmask);

    UPDATE_FP_INFO(st, 0);
    UPDATE_FP_INFO(st, 1);
    UPDATE_FP_INFO(st, 2);
    UPDATE_FP_INFO(st, 3);
    UPDATE_FP_INFO(st, 4);
    UPDATE_FP_INFO(st, 5);
    UPDATE_FP_INFO(st, 6);
    UPDATE_FP_INFO(st, 7);
    UPDATE_FP_INFO(mm, 0);
    UPDATE_FP_INFO(mm, 1);
    UPDATE_FP_INFO(mm, 2);
    UPDATE_FP_INFO(mm, 3);
    UPDATE_FP_INFO(mm, 4);
    UPDATE_FP_INFO(mm, 5);
    UPDATE_FP_INFO(mm, 6);
    UPDATE_FP_INFO(mm, 7);

    UPDATE_XMM_INFO(xmm, 0);
    UPDATE_XMM_INFO(xmm, 1);
    UPDATE_XMM_INFO(xmm, 2);
    UPDATE_XMM_INFO(xmm, 3);
    UPDATE_XMM_INFO(xmm, 4);
    UPDATE_XMM_INFO(xmm, 5);
    UPDATE_XMM_INFO(xmm, 6);
    UPDATE_XMM_INFO(xmm, 7);

    UPDATE_YMM_INFO(ymm, 0);
    UPDATE_YMM_INFO(ymm, 1);
    UPDATE_YMM_INFO(ymm, 2);
    UPDATE_YMM_INFO(ymm, 3);
    UPDATE_YMM_INFO(ymm, 4);
    UPDATE_YMM_INFO(ymm, 5);
    UPDATE_YMM_INFO(ymm, 6);
    UPDATE_YMM_INFO(ymm, 7);

    UPDATE_DR_INFO(0);
    UPDATE_DR_INFO(1);
    UPDATE_DR_INFO(2);
    UPDATE_DR_INFO(3);
    UPDATE_DR_INFO(4);
    UPDATE_DR_INFO(5);
    UPDATE_DR_INFO(6);
    UPDATE_DR_INFO(7);

#undef UPDATE_GPR_INFO
#undef UPDATE_GPR_INFO_8H
#undef UPDATE_FPR_INFO
#undef UPDATE_FP_INFO
#undef UPDATE_XMM_INFO
#undef UPDATE_YMM_INFO
#undef UPDATE_DR_INFO

#endif // UPDATE_REGISTER_INFOS_I386_STRUCT_WITH_X86_64_OFFSETS

#undef GPR_OFFSET
#undef FPR_OFFSET
#undef YMM_OFFSET
