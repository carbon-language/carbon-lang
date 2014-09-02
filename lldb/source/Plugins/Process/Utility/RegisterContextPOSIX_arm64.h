//===-- RegisterContextPOSIX_arm64.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextPOSIX_arm64_H_
#define liblldb_RegisterContextPOSIX_arm64_H_

#include "lldb/Core/Log.h"
#include "RegisterContextPOSIX.h"

class ProcessMonitor;

//---------------------------------------------------------------------------
// Internal codes for all ARM64 registers.
//---------------------------------------------------------------------------
enum
{
    k_first_gpr_arm64,
    gpr_x0_arm64 = k_first_gpr_arm64,
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

    k_last_gpr_arm64 = gpr_cpsr_arm64,

    k_first_fpr_arm64,
    fpu_v0_arm64 = k_first_fpr_arm64,
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
    k_last_fpr_arm64 = fpu_fpcr_arm64,

    exc_far_arm64,
    exc_esr_arm64,
    exc_exception_arm64,

    dbg_bvr0_arm64,
    dbg_bvr1_arm64,
    dbg_bvr2_arm64,
    dbg_bvr3_arm64,
    dbg_bvr4_arm64,
    dbg_bvr5_arm64,
    dbg_bvr6_arm64,
    dbg_bvr7_arm64,
    dbg_bvr8_arm64,
    dbg_bvr9_arm64,
    dbg_bvr10_arm64,
    dbg_bvr11_arm64,
    dbg_bvr12_arm64,
    dbg_bvr13_arm64,
    dbg_bvr14_arm64,
    dbg_bvr15_arm64,
    dbg_bcr0_arm64,
    dbg_bcr1_arm64,
    dbg_bcr2_arm64,
    dbg_bcr3_arm64,
    dbg_bcr4_arm64,
    dbg_bcr5_arm64,
    dbg_bcr6_arm64,
    dbg_bcr7_arm64,
    dbg_bcr8_arm64,
    dbg_bcr9_arm64,
    dbg_bcr10_arm64,
    dbg_bcr11_arm64,
    dbg_bcr12_arm64,
    dbg_bcr13_arm64,
    dbg_bcr14_arm64,
    dbg_bcr15_arm64,
    dbg_wvr0_arm64,
    dbg_wvr1_arm64,
    dbg_wvr2_arm64,
    dbg_wvr3_arm64,
    dbg_wvr4_arm64,
    dbg_wvr5_arm64,
    dbg_wvr6_arm64,
    dbg_wvr7_arm64,
    dbg_wvr8_arm64,
    dbg_wvr9_arm64,
    dbg_wvr10_arm64,
    dbg_wvr11_arm64,
    dbg_wvr12_arm64,
    dbg_wvr13_arm64,
    dbg_wvr14_arm64,
    dbg_wvr15_arm64,
    dbg_wcr0_arm64,
    dbg_wcr1_arm64,
    dbg_wcr2_arm64,
    dbg_wcr3_arm64,
    dbg_wcr4_arm64,
    dbg_wcr5_arm64,
    dbg_wcr6_arm64,
    dbg_wcr7_arm64,
    dbg_wcr8_arm64,
    dbg_wcr9_arm64,
    dbg_wcr10_arm64,
    dbg_wcr11_arm64,
    dbg_wcr12_arm64,
    dbg_wcr13_arm64,
    dbg_wcr14_arm64,
    dbg_wcr15_arm64,

    k_num_registers_arm64,
    k_num_gpr_registers_arm64 = k_last_gpr_arm64 - k_first_gpr_arm64 + 1,
    k_num_fpr_registers_arm64 = k_last_fpr_arm64 - k_first_fpr_arm64 + 1
};

class RegisterContextPOSIX_arm64
  : public lldb_private::RegisterContext
{
public:
    RegisterContextPOSIX_arm64 (lldb_private::Thread &thread,
                            uint32_t concrete_frame_idx,
                            lldb_private::RegisterInfoInterface *register_info);

    ~RegisterContextPOSIX_arm64();

    void
    Invalidate();

    void
    InvalidateAllRegisters();

    size_t
    GetRegisterCount();

    virtual size_t
    GetGPRSize();

    virtual unsigned
    GetRegisterSize(unsigned reg);

    virtual unsigned
    GetRegisterOffset(unsigned reg);

    const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex(size_t reg);

    size_t
    GetRegisterSetCount();

    const lldb_private::RegisterSet *
    GetRegisterSet(size_t set);

    const char *
    GetRegisterName(unsigned reg);

    uint32_t
    ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind, uint32_t num);

protected:
    struct RegInfo
    {
        uint32_t num_registers;
        uint32_t num_gpr_registers;
        uint32_t num_fpr_registers;

        uint32_t last_gpr;
        uint32_t first_fpr;
        uint32_t last_fpr;

        uint32_t first_fpr_v;
        uint32_t last_fpr_v;

        uint32_t gpr_flags;
    };

    // based on RegisterContextDarwin_arm64.h
    struct VReg
    {
        uint8_t bytes[16];
    };

    // based on RegisterContextDarwin_arm64.h
    struct FPU
    {
        VReg        v[32];
        uint32_t    fpsr;
        uint32_t    fpcr;
    };

    uint64_t m_gpr_arm64[k_num_gpr_registers_arm64];           // 64-bit general purpose registers.
    RegInfo  m_reg_info;
    struct RegisterContextPOSIX_arm64::FPU    m_fpr;           // floating-point registers including extended register sets.
    std::unique_ptr<lldb_private::RegisterInfoInterface> m_register_info_ap; // Register Info Interface (FreeBSD or Linux)

    // Determines if an extended register set is supported on the processor running the inferior process.
    virtual bool
    IsRegisterSetAvailable(size_t set_index);

    virtual const lldb_private::RegisterInfo *
    GetRegisterInfo();

    bool
    IsGPR(unsigned reg);

    bool
    IsFPR(unsigned reg);

    lldb::ByteOrder GetByteOrder();

    virtual bool ReadGPR() = 0;
    virtual bool ReadFPR() = 0;
    virtual bool WriteGPR() = 0;
    virtual bool WriteFPR() = 0;
};

#endif // #ifndef liblldb_RegisterContextPOSIX_arm64_H_
