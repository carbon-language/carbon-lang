//===-- RegisterContextPOSIX_powerpc.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextPOSIX_powerpc_H_
#define liblldb_RegisterContextPOSIX_powerpc_H_

#include "lldb/Core/Log.h"
#include "RegisterContextPOSIX.h"
#include "RegisterContext_powerpc.h"

class ProcessMonitor;

// ---------------------------------------------------------------------------
// Internal codes for all powerpc registers.
// ---------------------------------------------------------------------------
enum
{
    k_first_gpr_powerpc,
    gpr_r0_powerpc = k_first_gpr_powerpc,
    gpr_r1_powerpc,
    gpr_r2_powerpc,
    gpr_r3_powerpc,
    gpr_r4_powerpc,
    gpr_r5_powerpc,
    gpr_r6_powerpc,
    gpr_r7_powerpc,
    gpr_r8_powerpc,
    gpr_r9_powerpc,
    gpr_r10_powerpc,
    gpr_r11_powerpc,
    gpr_r12_powerpc,
    gpr_r13_powerpc,
    gpr_r14_powerpc,
    gpr_r15_powerpc,
    gpr_r16_powerpc,
    gpr_r17_powerpc,
    gpr_r18_powerpc,
    gpr_r19_powerpc,
    gpr_r20_powerpc,
    gpr_r21_powerpc,
    gpr_r22_powerpc,
    gpr_r23_powerpc,
    gpr_r24_powerpc,
    gpr_r25_powerpc,
    gpr_r26_powerpc,
    gpr_r27_powerpc,
    gpr_r28_powerpc,
    gpr_r29_powerpc,
    gpr_r30_powerpc,
    gpr_r31_powerpc,
    gpr_lr_powerpc,
    gpr_cr_powerpc,
    gpr_xer_powerpc,
    gpr_ctr_powerpc,
    gpr_pc_powerpc,
    k_last_gpr_powerpc = gpr_pc_powerpc,

    k_first_fpr,
    fpr_f0_powerpc = k_first_fpr,
    fpr_f1_powerpc,
    fpr_f2_powerpc,
    fpr_f3_powerpc,
    fpr_f4_powerpc,
    fpr_f5_powerpc,
    fpr_f6_powerpc,
    fpr_f7_powerpc,
    fpr_f8_powerpc,
    fpr_f9_powerpc,
    fpr_f10_powerpc,
    fpr_f11_powerpc,
    fpr_f12_powerpc,
    fpr_f13_powerpc,
    fpr_f14_powerpc,
    fpr_f15_powerpc,
    fpr_f16_powerpc,
    fpr_f17_powerpc,
    fpr_f18_powerpc,
    fpr_f19_powerpc,
    fpr_f20_powerpc,
    fpr_f21_powerpc,
    fpr_f22_powerpc,
    fpr_f23_powerpc,
    fpr_f24_powerpc,
    fpr_f25_powerpc,
    fpr_f26_powerpc,
    fpr_f27_powerpc,
    fpr_f28_powerpc,
    fpr_f29_powerpc,
    fpr_f30_powerpc,
    fpr_f31_powerpc,
    fpr_fpscr_powerpc,
    k_last_fpr = fpr_fpscr_powerpc,

    k_num_registers_powerpc,
    k_num_gpr_registers_powerpc = k_last_gpr_powerpc - k_first_gpr_powerpc + 1,
    k_num_fpr_registers_powerpc = k_last_fpr - k_first_fpr + 1,
};

class RegisterContextPOSIX_powerpc
  : public lldb_private::RegisterContext
{
public:
    RegisterContextPOSIX_powerpc (lldb_private::Thread &thread,
                            uint32_t concrete_frame_idx,
                            lldb_private::RegisterInfoInterface *register_info);

    ~RegisterContextPOSIX_powerpc();

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
    uint64_t m_gpr_powerpc[k_num_gpr_registers_powerpc];         // general purpose registers.
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

#endif // #ifndef liblldb_RegisterContextPOSIX_powerpc_H_
