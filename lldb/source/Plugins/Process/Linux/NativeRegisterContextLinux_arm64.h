//===-- NativeRegisterContextLinux_arm64.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef lldb_NativeRegisterContextLinux_arm64_h
#define lldb_NativeRegisterContextLinux_arm64_h

#include "lldb/Host/common/NativeRegisterContextRegisterInfo.h"
#include "Plugins/Process/Utility/lldb-arm64-register-enums.h"

namespace lldb_private {
namespace process_linux {

    class NativeProcessLinux;

    class NativeRegisterContextLinux_arm64 : public NativeRegisterContextRegisterInfo
    {
    public:
        NativeRegisterContextLinux_arm64 (NativeThreadProtocol &native_thread,
                                          uint32_t concrete_frame_idx,
                                          RegisterInfoInterface *reg_info_interface_p);

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

        //------------------------------------------------------------------
        // Hardware breakpoints/watchpoint mangement functions
        //------------------------------------------------------------------

        uint32_t
        SetHardwareBreakpoint (lldb::addr_t addr, size_t size) override;

        bool
        ClearHardwareBreakpoint (uint32_t hw_idx) override;

        uint32_t
        NumSupportedHardwareWatchpoints () override;

        uint32_t
        SetHardwareWatchpoint (lldb::addr_t addr, size_t size, uint32_t watch_flags) override;

        bool
        ClearHardwareWatchpoint (uint32_t hw_index) override;

        Error
        ClearAllHardwareWatchpoints () override;

        Error
        GetWatchpointHitIndex(uint32_t &wp_index, lldb::addr_t trap_addr) override;

        lldb::addr_t
        GetWatchpointAddress (uint32_t wp_index) override;

        bool
        HardwareSingleStep (bool enable) override;

        uint32_t
        GetWatchpointSize(uint32_t wp_index);

        bool
        WatchpointIsEnabled(uint32_t wp_index);

    private:
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

        uint64_t m_gpr_arm64[k_num_gpr_registers_arm64]; // 64-bit general purpose registers.
        RegInfo  m_reg_info;
        FPU m_fpr; // floating-point registers including extended register sets.

        // Debug register info for hardware breakpoints and watchpoints management.
        struct DREG
        {
            lldb::addr_t address;  // Breakpoint/watchpoint address value.
            uint32_t control;  // Breakpoint/watchpoint control value.
            uint32_t refcount;  // Serves as enable/disable and refernce counter.
        };

        struct DREG m_hbr_regs[16];  // Arm native linux hardware breakpoints
        struct DREG m_hwp_regs[16];  // Arm native linux hardware watchpoints

        uint32_t m_max_hwp_supported;
        uint32_t m_max_hbp_supported;
        bool m_refresh_hwdebug_info;

        bool
        IsGPR(unsigned reg) const;

        bool
        ReadGPR ();

        bool
        WriteGPR ();

        bool
        IsFPR(unsigned reg) const;

        bool
        ReadFPR ();

        bool
        WriteFPR ();

        Error
        ReadRegisterRaw (uint32_t reg_index, RegisterValue &reg_value);

        Error
        WriteRegisterRaw (uint32_t reg_index, const RegisterValue &reg_value);

        lldb::ByteOrder
        GetByteOrder() const;

        size_t
        GetGPRSize() const;
    };

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_arm64_h

