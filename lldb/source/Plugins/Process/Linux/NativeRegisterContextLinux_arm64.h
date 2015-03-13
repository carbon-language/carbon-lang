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

namespace lldb_private
{
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

        uint64_t m_gpr_arm64[lldb_private::k_num_gpr_registers_arm64]; // 64-bit general purpose registers.
        RegInfo  m_reg_info;
        FPU m_fpr; // floating-point registers including extended register sets.

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

        lldb_private::Error
        ReadRegisterRaw (uint32_t reg_index, RegisterValue &reg_value);

        lldb_private::Error
        WriteRegisterRaw (uint32_t reg_index, const RegisterValue &reg_value);

        lldb::ByteOrder
        GetByteOrder() const;

        size_t
        GetGPRSize() const;
    };
}

#endif // #ifndef lldb_NativeRegisterContextLinux_arm64_h

