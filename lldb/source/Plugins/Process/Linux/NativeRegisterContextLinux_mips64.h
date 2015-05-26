//===-- NativeRegisterContextLinux_mips64.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined (__mips__)

#ifndef lldb_NativeRegisterContextLinux_mips64_h
#define lldb_NativeRegisterContextLinux_mips64_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "Plugins/Process/Utility/RegisterContext_mips64.h"
#include "Plugins/Process/Utility/RegisterContextLinux_mips64.h"


namespace lldb_private {
namespace process_linux {

    class NativeProcessLinux;

    // ---------------------------------------------------------------------------
    // Internal codes for mips64 GP registers.
    // ---------------------------------------------------------------------------
    enum
    {
        k_first_gp_reg_mips64,
        gp_reg_r0_mips64 = k_first_gp_reg_mips64,
        gp_reg_r1_mips64,
        gp_reg_r2_mips64,
        gp_reg_r3_mips64,
        gp_reg_r4_mips64,
        gp_reg_r5_mips64,
        gp_reg_r6_mips64,
        gp_reg_r7_mips64,
        gp_reg_r8_mips64,
        gp_reg_r9_mips64,
        gp_reg_r10_mips64,
        gp_reg_r11_mips64,
        gp_reg_r12_mips64,
        gp_reg_r13_mips64,
        gp_reg_r14_mips64,
        gp_reg_r15_mips64,
        gp_reg_r16_mips64,
        gp_reg_r17_mips64,
        gp_reg_r18_mips64,
        gp_reg_r19_mips64,
        gp_reg_r20_mips64,
        gp_reg_r21_mips64,
        gp_reg_r22_mips64,
        gp_reg_r23_mips64,
        gp_reg_r24_mips64,
        gp_reg_r25_mips64,
        gp_reg_r26_mips64,
        gp_reg_r27_mips64,
        gp_reg_r28_mips64,
        gp_reg_r29_mips64,
        gp_reg_r30_mips64,
        gp_reg_r31_mips64,
        gp_reg_mullo_mips64,
        gp_reg_mulhi_mips64,
        gp_reg_pc_mips64,
        gp_reg_badvaddr_mips64,
        gp_reg_sr_mips64,
        gp_reg_cause_mips64,
        k_num_gp_reg_mips64,
    };

    class NativeRegisterContextLinux_mips64 : public NativeRegisterContextLinux
    {
    public:
        NativeRegisterContextLinux_mips64 (const ArchSpec& target_arch,
                                           NativeThreadProtocol &native_thread, 
                                           uint32_t concrete_frame_idx);

        uint32_t
        GetRegisterSetCount () const override;

        const RegisterSet *
        GetRegisterSet (uint32_t set_index) const override;

        Error
        ReadRegister (const RegisterInfo *reg_info, RegisterValue &reg_value) override;

        Error
        WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value) override;

        Error
        ReadAllRegisterValues (lldb::DataBufferSP &data_sp) override;

        Error
        WriteAllRegisterValues (const lldb::DataBufferSP &data_sp) override;

        Error
        IsWatchpointHit (uint32_t wp_index, bool &is_hit) override;

        Error
        IsWatchpointVacant (uint32_t wp_index, bool &is_vacant) override;

        bool
        ClearHardwareWatchpoint (uint32_t wp_index) override;

        Error
        ClearAllHardwareWatchpoints () override;

        Error
        SetHardwareWatchpointWithIndex (lldb::addr_t addr, size_t size,
                uint32_t watch_flags, uint32_t wp_index);

        uint32_t
        SetHardwareWatchpoint (lldb::addr_t addr, size_t size,
                uint32_t watch_flags) override;

        lldb::addr_t
        GetWatchpointAddress (uint32_t wp_index) override;

        uint32_t
        NumSupportedHardwareWatchpoints () override;

    protected:
        NativeProcessLinux::OperationUP
        GetReadRegisterValueOperation(uint32_t offset,
                                      const char* reg_name,
                                      uint32_t size,
                                      RegisterValue &value) override;

        NativeProcessLinux::OperationUP
        GetWriteRegisterValueOperation(uint32_t offset,
                                       const char* reg_name,
                                       const RegisterValue &value) override;
    };

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_mips64_h

#endif // defined (__mips__)
