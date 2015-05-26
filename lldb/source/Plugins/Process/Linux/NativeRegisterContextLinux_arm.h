//===-- NativeRegisterContextLinux_arm.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(__arm__) // arm register context only needed on arm devices

#ifndef lldb_NativeRegisterContextLinux_arm_h
#define lldb_NativeRegisterContextLinux_arm_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "Plugins/Process/Utility/lldb-arm-register-enums.h"

namespace lldb_private {
namespace process_linux {

    class NativeProcessLinux;

    class NativeRegisterContextLinux_arm : public NativeRegisterContextLinux
    {
    public:
        NativeRegisterContextLinux_arm (const ArchSpec& target_arch,
                                        NativeThreadProtocol &native_thread,
                                        uint32_t concrete_frame_idx);

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

    protected:
        void*
        GetGPRBuffer() override { return &m_gpr_arm; }

        void*
        GetFPRBuffer() override { return &m_fpr; }

        size_t
        GetFPRSize() override { return sizeof(m_fpr); }

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
        IsFPR(unsigned reg) const;
    };

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_arm_h

#endif // defined(__arm__)
