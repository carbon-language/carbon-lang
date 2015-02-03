//===-- NativeRegisterContextLinux_x86_64.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef lldb_NativeRegisterContextLinux_x86_64_h
#define lldb_NativeRegisterContextLinux_x86_64_h

#include "lldb/Host/common/NativeRegisterContextRegisterInfo.h"
#include "Plugins/Process/Utility/RegisterContext_x86.h"
#include "Plugins/Process/Utility/lldb-x86-register-enums.h"

namespace lldb_private
{
    class NativeProcessLinux;

    class NativeRegisterContextLinux_x86_64 : public NativeRegisterContextRegisterInfo
    {
    public:
        NativeRegisterContextLinux_x86_64 (NativeThreadProtocol &native_thread, uint32_t concrete_frame_idx, RegisterInfoInterface *reg_info_interface_p);

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
        IsWatchpointHit(uint8_t wp_index);

        Error
        IsWatchpointVacant(uint32_t wp_index);

        bool
        ClearHardwareWatchpoint(uint32_t wp_index);

        Error
        ClearAllHardwareWatchpoints ();

        Error
        SetHardwareWatchpointWithIndex(lldb::addr_t addr, size_t size,
                uint32_t watch_flags, uint32_t wp_index);

        uint32_t
        SetHardwareWatchpoint(lldb::addr_t addr, size_t size,
                uint32_t watch_flags);

        lldb::addr_t
        GetWatchpointAddress(uint32_t wp_index);

        uint32_t
        NumSupportedHardwareWatchpoints();

    private:

        // Private member types.
        enum FPRType
        {
            eFPRTypeNotValid = 0,
            eFPRTypeFXSAVE,
            eFPRTypeXSAVE
        };

        // Info about register ranges.
        struct RegInfo
        {
            uint32_t num_registers;
            uint32_t num_gpr_registers;
            uint32_t num_fpr_registers;
            uint32_t num_avx_registers;

            uint32_t last_gpr;
            uint32_t first_fpr;
            uint32_t last_fpr;

            uint32_t first_st;
            uint32_t last_st;
            uint32_t first_mm;
            uint32_t last_mm;
            uint32_t first_xmm;
            uint32_t last_xmm;
            uint32_t first_ymm;
            uint32_t last_ymm;

            uint32_t first_dr;
            uint32_t gpr_flags;
        };

        // Private member variables.
        mutable FPRType m_fpr_type;
        FPR m_fpr;
        IOVEC m_iovec;
        YMM m_ymm_set;
        RegInfo m_reg_info;
        uint64_t m_gpr_x86_64[k_num_gpr_registers_x86_64];

        // Private member methods.
        lldb_private::Error
        WriteRegister(const uint32_t reg, const RegisterValue &value);

        bool IsRegisterSetAvailable (uint32_t set_index) const;

        lldb::ByteOrder
        GetByteOrder() const;

        bool
        IsGPR(uint32_t reg_index) const;

        FPRType
        GetFPRType () const;

        bool
        IsFPR(uint32_t reg_index) const;

        bool
        WriteFPR();

        bool IsFPR(uint32_t reg_index, FPRType fpr_type) const;

        bool
        CopyXSTATEtoYMM (uint32_t reg_index, lldb::ByteOrder byte_order);

        bool
        CopyYMMtoXSTATE(uint32_t reg, lldb::ByteOrder byte_order);

        bool
        IsAVX (uint32_t reg_index) const;

        bool
        ReadFPR ();

        lldb_private::Error
        ReadRegisterRaw (uint32_t reg_index, RegisterValue &reg_value);

        bool
        ReadGPR();

        bool
        WriteGPR();
    };
}

#endif // #ifndef lldb_NativeRegisterContextLinux_x86_64_h

