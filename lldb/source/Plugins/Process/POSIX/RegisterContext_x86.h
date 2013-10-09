//===-- RegisterContext_x86.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContext_x86_H_
#define liblldb_RegisterContext_x86_H_

//---------------------------------------------------------------------------
// i386 gcc, dwarf, gdb enums
//---------------------------------------------------------------------------

// Register numbers seen in eh_frame (eRegisterKindGCC)
//
// From Jason Molenda: "gcc registers" is the register numbering used in the eh_frame
// CFI.  The only registers that are described in eh_frame CFI are those that are
// preserved across function calls aka callee-saved aka non-volatile.  And none
// of the floating point registers on x86 are preserved across function calls.
//
// The only reason there is a "gcc register" and a "dwarf register" is because of a
// mistake years and years ago with i386 where they got esp and ebp
// backwards when they emitted the eh_frame instructions.  Once there were
// binaries In The Wild using the reversed numbering, we had to stick with it
// forever.
enum
{
    // 2nd parameter in DwarfRegNum() is regnum for exception handling on x86-32.
    // See http://llvm.org/docs/WritingAnLLVMBackend.html#defining-a-register
    gcc_eax_i386 = 0,
    gcc_ecx_i386,
    gcc_edx_i386,
    gcc_ebx_i386,
    gcc_ebp_i386, // Warning: these are switched from dwarf values
    gcc_esp_i386, //
    gcc_esi_i386,
    gcc_edi_i386,
    gcc_eip_i386,
    gcc_eflags_i386,
    gcc_st0_i386 = 12,
    gcc_st1_i386,
    gcc_st2_i386,
    gcc_st3_i386,
    gcc_st4_i386,
    gcc_st5_i386,
    gcc_st6_i386,
    gcc_st7_i386,
    gcc_xmm0_i386 = 21,
    gcc_xmm1_i386,
    gcc_xmm2_i386,
    gcc_xmm3_i386,
    gcc_xmm4_i386,
    gcc_xmm5_i386,
    gcc_xmm6_i386,
    gcc_xmm7_i386,
    gcc_mm0_i386 = 29,
    gcc_mm1_i386,
    gcc_mm2_i386,
    gcc_mm3_i386,
    gcc_mm4_i386,
    gcc_mm5_i386,
    gcc_mm6_i386,
    gcc_mm7_i386,
};

// DWARF register numbers (eRegisterKindDWARF)
// Intel's x86 or IA-32
enum
{
    // General Purpose Registers.
    dwarf_eax_i386 = 0,
    dwarf_ecx_i386,
    dwarf_edx_i386,
    dwarf_ebx_i386,
    dwarf_esp_i386,
    dwarf_ebp_i386,
    dwarf_esi_i386,
    dwarf_edi_i386,
    dwarf_eip_i386,
    dwarf_eflags_i386,
    // Floating Point Registers
    dwarf_st0_i386 = 11,
    dwarf_st1_i386,
    dwarf_st2_i386,
    dwarf_st3_i386,
    dwarf_st4_i386,
    dwarf_st5_i386,
    dwarf_st6_i386,
    dwarf_st7_i386,
    // SSE Registers
    dwarf_xmm0_i386 = 21,
    dwarf_xmm1_i386,
    dwarf_xmm2_i386,
    dwarf_xmm3_i386,
    dwarf_xmm4_i386,
    dwarf_xmm5_i386,
    dwarf_xmm6_i386,
    dwarf_xmm7_i386,
    // MMX Registers
    dwarf_mm0_i386 = 29,
    dwarf_mm1_i386,
    dwarf_mm2_i386,
    dwarf_mm3_i386,
    dwarf_mm4_i386,
    dwarf_mm5_i386,
    dwarf_mm6_i386,
    dwarf_mm7_i386,
    dwarf_fctrl_i386 = 37, // x87 control word
    dwarf_fstat_i386 = 38, // x87 status word
    dwarf_mxcsr_i386 = 39,
    dwarf_es_i386 = 40,
    dwarf_cs_i386 = 41,
    dwarf_ss_i386 = 42,
    dwarf_ds_i386 = 43,
    dwarf_fs_i386 = 44,
    dwarf_gs_i386 = 45

    // I believe the ymm registers use the dwarf_xmm%_i386 register numbers and
    //  then differentiate based on size of the register.
};

// Register numbers GDB uses (eRegisterKindGDB)
//
// From Jason Molenda: The "gdb numbers" are what you would see in the stabs debug format.
enum
{
    gdb_eax_i386,
    gdb_ecx_i386,
    gdb_edx_i386,
    gdb_ebx_i386,
    gdb_esp_i386,
    gdb_ebp_i386,
    gdb_esi_i386,
    gdb_edi_i386,
    gdb_eip_i386,
    gdb_eflags_i386,
    gdb_cs_i386,
    gdb_ss_i386,
    gdb_ds_i386,
    gdb_es_i386,
    gdb_fs_i386,
    gdb_gs_i386,
    gdb_st0_i386 = 16,
    gdb_st1_i386,
    gdb_st2_i386,
    gdb_st3_i386,
    gdb_st4_i386,
    gdb_st5_i386,
    gdb_st6_i386,
    gdb_st7_i386,
    gdb_fctrl_i386, // FPU Control Word
    gdb_fstat_i386, // FPU Status Word
    gdb_ftag_i386,  // FPU Tag Word
    gdb_fiseg_i386, // FPU IP Selector 
    gdb_fioff_i386, // FPU IP Offset
    gdb_foseg_i386, // FPU Operand Pointer Selector
    gdb_fooff_i386, // FPU Operand Pointer Offset
    gdb_fop_i386,   // Last Instruction Opcode
    gdb_xmm0_i386 = 32,
    gdb_xmm1_i386,
    gdb_xmm2_i386,
    gdb_xmm3_i386,
    gdb_xmm4_i386,
    gdb_xmm5_i386,
    gdb_xmm6_i386,
    gdb_xmm7_i386,
    gdb_mxcsr_i386 = 40,
    gdb_ymm0h_i386,
    gdb_ymm1h_i386,
    gdb_ymm2h_i386,
    gdb_ymm3h_i386,
    gdb_ymm4h_i386,
    gdb_ymm5h_i386,
    gdb_ymm6h_i386,
    gdb_ymm7h_i386,
    gdb_mm0_i386,
    gdb_mm1_i386,
    gdb_mm2_i386,
    gdb_mm3_i386,
    gdb_mm4_i386,
    gdb_mm5_i386,
    gdb_mm6_i386,
    gdb_mm7_i386,
};

//---------------------------------------------------------------------------
// AMD x86_64, AMD64, Intel EM64T, or Intel 64 gcc, dwarf, gdb enums
//---------------------------------------------------------------------------

// GCC and DWARF Register numbers (eRegisterKindGCC & eRegisterKindDWARF)
//  This is the spec I used (as opposed to x86-64-abi-0.99.pdf):
//  http://software.intel.com/sites/default/files/article/402129/mpx-linux64-abi.pdf
enum
{
    // GP Registers
    gcc_dwarf_rax_x86_64 = 0,
    gcc_dwarf_rdx_x86_64,
    gcc_dwarf_rcx_x86_64,
    gcc_dwarf_rbx_x86_64,
    gcc_dwarf_rsi_x86_64,
    gcc_dwarf_rdi_x86_64,
    gcc_dwarf_rbp_x86_64,
    gcc_dwarf_rsp_x86_64,
    // Extended GP Registers
    gcc_dwarf_r8_x86_64 = 8,
    gcc_dwarf_r9_x86_64,
    gcc_dwarf_r10_x86_64,
    gcc_dwarf_r11_x86_64,
    gcc_dwarf_r12_x86_64,
    gcc_dwarf_r13_x86_64,
    gcc_dwarf_r14_x86_64,
    gcc_dwarf_r15_x86_64,
    // Return Address (RA) mapped to RIP
    gcc_dwarf_rip_x86_64 = 16,
    // SSE Vector Registers
    gcc_dwarf_xmm0_x86_64 = 17,
    gcc_dwarf_xmm1_x86_64,
    gcc_dwarf_xmm2_x86_64,
    gcc_dwarf_xmm3_x86_64,
    gcc_dwarf_xmm4_x86_64,
    gcc_dwarf_xmm5_x86_64,
    gcc_dwarf_xmm6_x86_64,
    gcc_dwarf_xmm7_x86_64,
    gcc_dwarf_xmm8_x86_64,
    gcc_dwarf_xmm9_x86_64,
    gcc_dwarf_xmm10_x86_64,
    gcc_dwarf_xmm11_x86_64,
    gcc_dwarf_xmm12_x86_64,
    gcc_dwarf_xmm13_x86_64,
    gcc_dwarf_xmm14_x86_64,
    gcc_dwarf_xmm15_x86_64,
    // Floating Point Registers
    gcc_dwarf_st0_x86_64 = 33,
    gcc_dwarf_st1_x86_64,
    gcc_dwarf_st2_x86_64,
    gcc_dwarf_st3_x86_64,
    gcc_dwarf_st4_x86_64,
    gcc_dwarf_st5_x86_64,
    gcc_dwarf_st6_x86_64,
    gcc_dwarf_st7_x86_64,
    // MMX Registers
    gcc_dwarf_mm0_x86_64 = 41,
    gcc_dwarf_mm1_x86_64,
    gcc_dwarf_mm2_x86_64,
    gcc_dwarf_mm3_x86_64,
    gcc_dwarf_mm4_x86_64,
    gcc_dwarf_mm5_x86_64,
    gcc_dwarf_mm6_x86_64,
    gcc_dwarf_mm7_x86_64,
    // Control and Status Flags Register
    gcc_dwarf_rflags_x86_64 = 49,
    //  selector registers
    gcc_dwarf_es_x86_64 = 50,
    gcc_dwarf_cs_x86_64,
    gcc_dwarf_ss_x86_64,
    gcc_dwarf_ds_x86_64,
    gcc_dwarf_fs_x86_64,
    gcc_dwarf_gs_x86_64,
    // Floating point control registers
    gcc_dwarf_mxcsr_x86_64 = 64, // Media Control and Status
    gcc_dwarf_fctrl_x86_64,      // x87 control word
    gcc_dwarf_fstat_x86_64,      // x87 status word
    // Upper Vector Registers    
    gcc_dwarf_ymm0h_x86_64 = 67,
    gcc_dwarf_ymm1h_x86_64,
    gcc_dwarf_ymm2h_x86_64,
    gcc_dwarf_ymm3h_x86_64,
    gcc_dwarf_ymm4h_x86_64,
    gcc_dwarf_ymm5h_x86_64,
    gcc_dwarf_ymm6h_x86_64,
    gcc_dwarf_ymm7h_x86_64,
    gcc_dwarf_ymm8h_x86_64,
    gcc_dwarf_ymm9h_x86_64,
    gcc_dwarf_ymm10h_x86_64,
    gcc_dwarf_ymm11h_x86_64,
    gcc_dwarf_ymm12h_x86_64,
    gcc_dwarf_ymm13h_x86_64,
    gcc_dwarf_ymm14h_x86_64,
    gcc_dwarf_ymm15h_x86_64,
    // AVX2 Vector Mask Registers
    // gcc_dwarf_k0_x86_64 = 118,
    // gcc_dwarf_k1_x86_64,
    // gcc_dwarf_k2_x86_64,
    // gcc_dwarf_k3_x86_64,
    // gcc_dwarf_k4_x86_64,
    // gcc_dwarf_k5_x86_64,
    // gcc_dwarf_k6_x86_64,
    // gcc_dwarf_k7_x86_64,
};

// GDB Register numbers (eRegisterKindGDB)
enum
{
    // GP Registers
    gdb_rax_x86_64 = 0,
    gdb_rbx_x86_64,
    gdb_rcx_x86_64,
    gdb_rdx_x86_64,
    gdb_rsi_x86_64,
    gdb_rdi_x86_64,
    gdb_rbp_x86_64,
    gdb_rsp_x86_64,
    // Extended GP Registers
    gdb_r8_x86_64,
    gdb_r9_x86_64,
    gdb_r10_x86_64,
    gdb_r11_x86_64,
    gdb_r12_x86_64,
    gdb_r13_x86_64,
    gdb_r14_x86_64,
    gdb_r15_x86_64,
    // Return Address (RA) mapped to RIP
    gdb_rip_x86_64,
    // Control and Status Flags Register
    gdb_rflags_x86_64,
    gdb_cs_x86_64,
    gdb_ss_x86_64,
    gdb_ds_x86_64,
    gdb_es_x86_64,
    gdb_fs_x86_64,
    gdb_gs_x86_64,
    // Floating Point Registers
    gdb_st0_x86_64,
    gdb_st1_x86_64,
    gdb_st2_x86_64,
    gdb_st3_x86_64,
    gdb_st4_x86_64,
    gdb_st5_x86_64,
    gdb_st6_x86_64,
    gdb_st7_x86_64,
    gdb_fctrl_x86_64,
    gdb_fstat_x86_64,
    gdb_ftag_x86_64,
    gdb_fiseg_x86_64,
    gdb_fioff_x86_64,
    gdb_foseg_x86_64,
    gdb_fooff_x86_64,
    gdb_fop_x86_64,
    // SSE Vector Registers
    gdb_xmm0_x86_64 = 40,
    gdb_xmm1_x86_64,
    gdb_xmm2_x86_64,
    gdb_xmm3_x86_64,
    gdb_xmm4_x86_64,
    gdb_xmm5_x86_64,
    gdb_xmm6_x86_64,
    gdb_xmm7_x86_64,
    gdb_xmm8_x86_64,
    gdb_xmm9_x86_64,
    gdb_xmm10_x86_64,
    gdb_xmm11_x86_64,
    gdb_xmm12_x86_64,
    gdb_xmm13_x86_64,
    gdb_xmm14_x86_64,
    gdb_xmm15_x86_64,
    // Floating point control registers
    gdb_mxcsr_x86_64 = 56,
    gdb_ymm0h_x86_64,
    gdb_ymm1h_x86_64,
    gdb_ymm2h_x86_64,
    gdb_ymm3h_x86_64,
    gdb_ymm4h_x86_64,
    gdb_ymm5h_x86_64,
    gdb_ymm6h_x86_64,
    gdb_ymm7h_x86_64,
    gdb_ymm8h_x86_64,
    gdb_ymm9h_x86_64,
    gdb_ymm10h_x86_64,
    gdb_ymm11h_x86_64,
    gdb_ymm12h_x86_64,
    gdb_ymm13h_x86_64,
    gdb_ymm14h_x86_64,
    gdb_ymm15h_x86_64
};

//---------------------------------------------------------------------------
// Generic floating-point registers
//---------------------------------------------------------------------------

struct MMSReg
{
    uint8_t bytes[10];
    uint8_t pad[6];
};

struct XMMReg
{
    uint8_t bytes[16];      // 128-bits for each XMM register
};

// i387_fxsave_struct
struct FXSAVE
{
    uint16_t fctrl;         // FPU Control Word (fcw)
    uint16_t fstat;         // FPU Status Word (fsw)
    uint16_t ftag;          // FPU Tag Word (ftw)
    uint16_t fop;           // Last Instruction Opcode (fop)
    union
    {
        struct
        {
            uint64_t fip;   // Instruction Pointer
            uint64_t fdp;   // Data Pointer
        } x86_64;
        struct
        {
            uint32_t fioff;   // FPU IP Offset (fip)
            uint32_t fiseg;   // FPU IP Selector (fcs)
            uint32_t fooff;   // FPU Operand Pointer Offset (foo)
            uint32_t foseg;   // FPU Operand Pointer Selector (fos)
        } i386;
    } ptr;
    uint32_t mxcsr;         // MXCSR Register State
    uint32_t mxcsrmask;     // MXCSR Mask 
    MMSReg   stmm[8];       // 8*16 bytes for each FP-reg = 128 bytes
    XMMReg   xmm[16];       // 16*16 bytes for each XMM-reg = 256 bytes
    uint32_t padding[24];
};

//---------------------------------------------------------------------------
// Extended floating-point registers
//---------------------------------------------------------------------------

struct YMMHReg
{
    uint8_t  bytes[16];     // 16 * 8 bits for the high bytes of each YMM register
};

struct YMMReg
{
    uint8_t  bytes[32];     // 16 * 16 bits for each YMM register
};

struct YMM
{
    YMMReg   ymm[16];       // assembled from ymmh and xmm registers
};

struct XSAVE_HDR
{
    uint64_t  xstate_bv;    // OS enabled xstate mask to determine the extended states supported by the processor
    uint64_t  reserved1[2];
    uint64_t  reserved2[5];
} __attribute__((packed));

// x86 extensions to FXSAVE (i.e. for AVX processors) 
struct XSAVE 
{
    FXSAVE    i387;         // floating point registers typical in i387_fxsave_struct
    XSAVE_HDR header;       // The xsave_hdr_struct can be used to determine if the following extensions are usable
    YMMHReg   ymmh[16];     // High 16 bytes of each of 16 YMM registers (the low bytes are in FXSAVE.xmm for compatibility with SSE)
    // Slot any extensions to the register file here
} __attribute__((packed, aligned (64)));

// Floating-point registers
struct FPR
{
    // Thread state for the floating-point unit of the processor read by ptrace.
    union XSTATE
    {
        FXSAVE   fxsave;    // Generic floating-point registers.
        XSAVE    xsave;     // x86 extended processor state.
    } xstate;
};

//---------------------------------------------------------------------------
// ptrace PTRACE_GETREGSET, PTRACE_SETREGSET structure
//---------------------------------------------------------------------------

struct IOVEC
{
    void    *iov_base;      // pointer to XSAVE
    size_t   iov_len;       // sizeof(XSAVE)
};

#endif
