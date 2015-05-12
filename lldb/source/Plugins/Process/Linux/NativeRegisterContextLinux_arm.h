//===-- NativeRegisterContextLinux_arm.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef lldb_NativeRegisterContextLinux_arm_h
#define lldb_NativeRegisterContextLinux_arm_h

#include "lldb/Host/common/NativeRegisterContextRegisterInfo.h"
#include "Plugins/Process/Utility/lldb-arm-register-enums.h"

namespace lldb_private {
namespace process_linux {

    class NativeProcessLinux;

    class NativeRegisterContextLinux_arm : public NativeRegisterContextRegisterInfo
    {
    public:
        NativeRegisterContextLinux_arm (NativeThreadProtocol &native_thread, uint32_t concrete_frame_idx, RegisterInfoInterface *reg_info_interface_p);

        uint32_t
        GetRegisterSetCount () const override;

        const RegisterSet *
        GetRegisterSet (uint32_t set_index) const override;

        uint32_t
        GetUserRegisterCount() const override;

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

        struct QReg
        {
            uint8_t bytes[16];
        };

        struct FPU
        {
            union {
                uint32_t s[32];
                uint64_t d[32];
                QReg     q[16];  // the 128-bit NEON registers
                } floats;
            uint32_t fpscr;
        };

        uint32_t m_gpr_arm[k_num_gpr_registers_arm];
        RegInfo  m_reg_info;
        FPU m_fpr; 

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

#endif // #ifndef lldb_NativeRegisterContextLinux_arm_h

