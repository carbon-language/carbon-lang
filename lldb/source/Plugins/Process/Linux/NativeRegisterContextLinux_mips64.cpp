//===-- NativeRegisterContextLinux_mips64.cpp ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined (__mips__)

#include "NativeRegisterContextLinux_mips64.h"

// C Includes
// C++ Includes

// Other libraries and framework includes
#include "lldb/Core/Error.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Host/HostInfo.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "Plugins/Process/Utility/RegisterContextLinux_mips64.h"
#include "Plugins/Process/Utility/RegisterContextLinux_mips.h"

using namespace lldb_private;
using namespace lldb_private::process_linux;

// ----------------------------------------------------------------------------
// Private namespace.
// ----------------------------------------------------------------------------

namespace
{
    // mips general purpose registers.
    const uint32_t
    g_gp_regnums_mips[] =
    {
        gpr_zero_mips,
        gpr_r1_mips,
        gpr_r2_mips,
        gpr_r3_mips,
        gpr_r4_mips,
        gpr_r5_mips,
        gpr_r6_mips,
        gpr_r7_mips,
        gpr_r8_mips,
        gpr_r9_mips,
        gpr_r10_mips,
        gpr_r11_mips,
        gpr_r12_mips,
        gpr_r13_mips,
        gpr_r14_mips,
        gpr_r15_mips,
        gpr_r16_mips,
        gpr_r17_mips,
        gpr_r18_mips,
        gpr_r19_mips,
        gpr_r20_mips,
        gpr_r21_mips,
        gpr_r22_mips,
        gpr_r23_mips,
        gpr_r24_mips,
        gpr_r25_mips,
        gpr_r26_mips,
        gpr_r27_mips,
        gpr_gp_mips,
        gpr_sp_mips,
        gpr_r30_mips,
        gpr_ra_mips,
        gpr_mullo_mips,
        gpr_mulhi_mips,
        gpr_pc_mips,
        gpr_badvaddr_mips,
        gpr_sr_mips,
        gpr_cause_mips,
        LLDB_INVALID_REGNUM     // register sets need to end with this flag
    };

    static_assert((sizeof(g_gp_regnums_mips) / sizeof(g_gp_regnums_mips[0])) - 1 == k_num_gpr_registers_mips,
                  "g_gp_regnums_mips has wrong number of register infos");

    // mips floating point registers.
    const uint32_t
    g_fp_regnums_mips[] =
    {
        fpr_f0_mips,
        fpr_f1_mips,
        fpr_f2_mips,
        fpr_f3_mips,
        fpr_f4_mips,
        fpr_f5_mips,
        fpr_f6_mips,
        fpr_f7_mips,
        fpr_f8_mips,
        fpr_f9_mips,
        fpr_f10_mips,
        fpr_f11_mips,
        fpr_f12_mips,
        fpr_f13_mips,
        fpr_f14_mips,
        fpr_f15_mips,
        fpr_f16_mips,
        fpr_f17_mips,
        fpr_f18_mips,
        fpr_f19_mips,
        fpr_f20_mips,
        fpr_f21_mips,
        fpr_f22_mips,
        fpr_f23_mips,
        fpr_f24_mips,
        fpr_f25_mips,
        fpr_f26_mips,
        fpr_f27_mips,
        fpr_f28_mips,
        fpr_f29_mips,
        fpr_f30_mips,
        fpr_f31_mips,
        fpr_fcsr_mips,
        fpr_fir_mips,
        LLDB_INVALID_REGNUM     // register sets need to end with this flag
    };

    static_assert((sizeof(g_fp_regnums_mips) / sizeof(g_fp_regnums_mips[0])) - 1 == k_num_fpr_registers_mips,
                  "g_fp_regnums_mips has wrong number of register infos");

    // mips64 general purpose registers.
    const uint32_t
    g_gp_regnums_mips64[] =
    {
        gpr_zero_mips64,
        gpr_r1_mips64,
        gpr_r2_mips64,
        gpr_r3_mips64,
        gpr_r4_mips64,
        gpr_r5_mips64,
        gpr_r6_mips64,
        gpr_r7_mips64,
        gpr_r8_mips64,
        gpr_r9_mips64,
        gpr_r10_mips64,
        gpr_r11_mips64,
        gpr_r12_mips64,
        gpr_r13_mips64,
        gpr_r14_mips64,
        gpr_r15_mips64,
        gpr_r16_mips64,
        gpr_r17_mips64,
        gpr_r18_mips64,
        gpr_r19_mips64,
        gpr_r20_mips64,
        gpr_r21_mips64,
        gpr_r22_mips64,
        gpr_r23_mips64,
        gpr_r24_mips64,
        gpr_r25_mips64,
        gpr_r26_mips64,
        gpr_r27_mips64,
        gpr_gp_mips64,
        gpr_sp_mips64,
        gpr_r30_mips64,
        gpr_ra_mips64,
        gpr_mullo_mips64,
        gpr_mulhi_mips64,
        gpr_pc_mips64,
        gpr_badvaddr_mips64,
        gpr_sr_mips64,
        gpr_cause_mips64,
        gpr_ic_mips64,
        gpr_dummy_mips64,
        LLDB_INVALID_REGNUM     // register sets need to end with this flag
    };

    static_assert((sizeof(g_gp_regnums_mips64) / sizeof(g_gp_regnums_mips64[0])) - 1 == k_num_gpr_registers_mips64,
                  "g_gp_regnums_mips64 has wrong number of register infos");

    // mips64 floating point registers.
    const uint32_t
    g_fp_regnums_mips64[] =
    {
        fpr_f0_mips64,
        fpr_f1_mips64,
        fpr_f2_mips64,
        fpr_f3_mips64,
        fpr_f4_mips64,
        fpr_f5_mips64,
        fpr_f6_mips64,
        fpr_f7_mips64,
        fpr_f8_mips64,
        fpr_f9_mips64,
        fpr_f10_mips64,
        fpr_f11_mips64,
        fpr_f12_mips64,
        fpr_f13_mips64,
        fpr_f14_mips64,
        fpr_f15_mips64,
        fpr_f16_mips64,
        fpr_f17_mips64,
        fpr_f18_mips64,
        fpr_f19_mips64,
        fpr_f20_mips64,
        fpr_f21_mips64,
        fpr_f22_mips64,
        fpr_f23_mips64,
        fpr_f24_mips64,
        fpr_f25_mips64,
        fpr_f26_mips64,
        fpr_f27_mips64,
        fpr_f28_mips64,
        fpr_f29_mips64,
        fpr_f30_mips64,
        fpr_f31_mips64,
        fpr_fcsr_mips64,
        fpr_fir_mips64,
        LLDB_INVALID_REGNUM     // register sets need to end with this flag
    };

    static_assert((sizeof(g_fp_regnums_mips64) / sizeof(g_fp_regnums_mips64[0])) - 1 == k_num_fpr_registers_mips64,
                  "g_fp_regnums_mips64 has wrong number of register infos");

    // Number of register sets provided by this context.
    enum
    {
        k_num_register_sets = 2
    };

    // Register sets for mips.
    static const RegisterSet
    g_reg_sets_mips[k_num_register_sets] =
    {
        { "General Purpose Registers",  "gpr", k_num_gpr_registers_mips, g_gp_regnums_mips },
        { "Floating Point Registers",   "fpu", k_num_fpr_registers_mips, g_fp_regnums_mips }
    };

    // Register sets for mips64.
    static const RegisterSet
    g_reg_sets_mips64[k_num_register_sets] =
    {
        { "General Purpose Registers",  "gpr", k_num_gpr_registers_mips64, g_gp_regnums_mips64 },
        { "Floating Point Registers",   "fpu", k_num_fpr_registers_mips64, g_fp_regnums_mips64 }
    };

    class ReadRegOperation : public NativeProcessLinux::Operation
    {
    public:
        ReadRegOperation(lldb::tid_t tid, uint32_t offset, RegisterValue &value) :
            m_tid(tid),
            m_offset(static_cast<uintptr_t>(offset)),
            m_value(value)
        { }

        void
        Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        uintptr_t m_offset;
        RegisterValue &m_value;
    };

    class WriteRegOperation : public NativeProcessLinux::Operation
    {
    public:
        WriteRegOperation(lldb::tid_t tid, unsigned offset, const char *reg_name, const RegisterValue &value) :
            m_tid(tid),
            m_offset(offset),
            m_reg_name(reg_name),
            m_value(value)
        { }

        void
        Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        uintptr_t m_offset;
        const char *m_reg_name;
        const RegisterValue &m_value;
    };

} // end of anonymous namespace

void
ReadRegOperation::Execute(NativeProcessLinux *monitor)
{
    elf_gregset_t regs;
    NativeProcessLinux::PtraceWrapper(PTRACE_GETREGS, m_tid, NULL, &regs, sizeof regs, m_error);
    if (m_error.Success())
    {
        lldb_private::ArchSpec arch;
        if (monitor->GetArchitecture(arch))
            m_value.SetBytes((void *)(((unsigned char *)(regs)) + m_offset), 8, arch.GetByteOrder());
        else
            m_error.SetErrorString("failed to get architecture");
    }
}

void
WriteRegOperation::Execute(NativeProcessLinux *monitor)
{
    elf_gregset_t regs;
    NativeProcessLinux::PtraceWrapper(PTRACE_GETREGS, m_tid, NULL, &regs, sizeof regs, m_error);
    if (m_error.Success())
    {
        ::memcpy((void *)(((unsigned char *)(&regs)) + m_offset), m_value.GetBytes(), 8);
        NativeProcessLinux::PtraceWrapper(PTRACE_SETREGS, m_tid, NULL, &regs, sizeof regs, m_error);
    }
}

NativeRegisterContextLinux*
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(const ArchSpec& target_arch,
                                                                 NativeThreadProtocol &native_thread,
                                                                 uint32_t concrete_frame_idx)
{
    return new NativeRegisterContextLinux_mips64(target_arch, native_thread, concrete_frame_idx);
}

#define REG_CONTEXT_SIZE (GetRegisterInfoInterface ().GetGPRSize () + sizeof(FPR_mips))

// ----------------------------------------------------------------------------
// NativeRegisterContextLinux_mips64 members.
// ----------------------------------------------------------------------------

static RegisterInfoInterface*
CreateRegisterInfoInterface(const ArchSpec& target_arch)
{
    if (HostInfo::GetArchitecture().GetAddressByteSize() == 4)
    {
        // 32-bit hosts run with a RegisterContextLinux_mips context.
        return new RegisterContextLinux_mips(target_arch);
    }
    else
    {
        assert((HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
               "Register setting path assumes this is a 64-bit host");
        // mips64 hosts know how to work with 64-bit and 32-bit EXEs using the mips64 register context.
        return new RegisterContextLinux_mips64 (target_arch);
    }
}

NativeRegisterContextLinux_mips64::NativeRegisterContextLinux_mips64 (const ArchSpec& target_arch,
                                                                      NativeThreadProtocol &native_thread, 
                                                                      uint32_t concrete_frame_idx) :
    NativeRegisterContextLinux (native_thread, concrete_frame_idx, CreateRegisterInfoInterface(target_arch))
{
    switch (target_arch.GetMachine ())
    {
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
            m_reg_info.num_registers        = k_num_registers_mips;
            m_reg_info.num_gpr_registers    = k_num_gpr_registers_mips;
            m_reg_info.num_fpr_registers    = k_num_fpr_registers_mips;
            m_reg_info.last_gpr             = k_last_gpr_mips;
            m_reg_info.first_fpr            = k_first_fpr_mips;
            m_reg_info.last_fpr             = k_last_fpr_mips;
            break;
        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
            m_reg_info.num_registers        = k_num_registers_mips64;
            m_reg_info.num_gpr_registers    = k_num_gpr_registers_mips64;
            m_reg_info.num_fpr_registers    = k_num_fpr_registers_mips64;
            m_reg_info.last_gpr             = k_last_gpr_mips64;
            m_reg_info.first_fpr            = k_first_fpr_mips64;
            m_reg_info.last_fpr             = k_last_fpr_mips64;
            break;
        default:
            assert(false && "Unhandled target architecture.");
            break;
    }

    // Clear out the FPR state.
    ::memset(&m_fpr, 0, sizeof(FPR_mips));
}

uint32_t
NativeRegisterContextLinux_mips64::GetRegisterSetCount () const
{
    return k_num_register_sets;
}

const RegisterSet *
NativeRegisterContextLinux_mips64::GetRegisterSet (uint32_t set_index) const
{
    if (set_index >= k_num_register_sets)
        return nullptr;

    switch (GetRegisterInfoInterface ().GetTargetArchitecture ().GetMachine ())
    {
        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
            return &g_reg_sets_mips64[set_index];
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
            return &g_reg_sets_mips[set_index];
        default:
            assert (false && "Unhandled target architecture.");
            return nullptr;
    }

    return nullptr;
}

lldb_private::Error
NativeRegisterContextLinux_mips64::ReadRegister (const RegisterInfo *reg_info, RegisterValue &reg_value)
{
    Error error;

    if (!reg_info)
    {
        error.SetErrorString ("reg_info NULL");
        return error;
    }

    const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
    if (reg == LLDB_INVALID_REGNUM)
    {
        // This is likely an internal register for lldb use only and should not be directly queried.
        error.SetErrorStringWithFormat ("register \"%s\" is an internal-only lldb register, cannot read directly", reg_info->name);
        return error;
    }

    if (IsFPR(reg))
    {
        error = ReadFPR();
        if (!error.Success())
        {
            error.SetErrorString ("failed to read floating point register");
            return error;
        }
        assert (reg_info->byte_offset < sizeof(FPR_mips));
        uint8_t *src = (uint8_t *)&m_fpr + reg_info->byte_offset;
        switch (reg_info->byte_size)
        {
            case 4:
                reg_value.SetUInt32(*(uint32_t *)src);
                break;
            case 8:
                reg_value.SetUInt64(*(uint64_t *)src);
                break;
            default:
                assert(false && "Unhandled data size.");
                error.SetErrorStringWithFormat ("unhandled byte size: %" PRIu32, reg_info->byte_size);
                break;
        }
    }
    else
    {
        error = ReadRegisterRaw(reg, reg_value);
        if (error.Success())
        {
            // If our return byte size was greater than the return value reg size, then
            // use the type specified by reg_info rather than the uint64_t default
            if (reg_value.GetByteSize() > reg_info->byte_size)
                reg_value.SetType(reg_info);
        }
    }

    return error;
}

lldb_private::Error
NativeRegisterContextLinux_mips64::WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value)
{
    Error error;

    assert (reg_info && "reg_info is null");

    const uint32_t reg_index = reg_info->kinds[lldb::eRegisterKindLLDB];

    if (reg_index == LLDB_INVALID_REGNUM)
        return Error ("no lldb regnum for %s", reg_info && reg_info->name ? reg_info->name : "<unknown register>");

    if (IsFPR(reg_index))
    {
        assert (reg_info->byte_offset < sizeof(FPR_mips));
        uint8_t *dst = (uint8_t *)&m_fpr + reg_info->byte_offset;
        switch (reg_info->byte_size)
        {
            case 4:
                *(uint32_t *)dst = reg_value.GetAsUInt32();
                break;
            case 8:
                *(uint64_t *)dst = reg_value.GetAsUInt64();
                break;
            default:
                assert(false && "Unhandled data size.");
                error.SetErrorStringWithFormat ("unhandled byte size: %" PRIu32, reg_info->byte_size);
                break;
        }
        error = WriteFPR();
        if (!error.Success())
        {
            error.SetErrorString ("failed to write floating point register");
            return error;
        }
    }
    else
    {
        error = WriteRegisterRaw(reg_index, reg_value);
    }

    return error;
}

Error
NativeRegisterContextLinux_mips64::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    Error error;

    data_sp.reset (new DataBufferHeap (REG_CONTEXT_SIZE, 0));
    if (!data_sp)
    {
        error.SetErrorStringWithFormat ("failed to allocate DataBufferHeap instance of size %" PRIu64, REG_CONTEXT_SIZE);
        return error;
    }

    error = ReadGPR();
    if (!error.Success())
    {
        error.SetErrorString ("ReadGPR() failed");
        return error;
    }

    error = ReadFPR();
    if (!error.Success())
    {
        error.SetErrorString ("ReadFPR() failed");
        return error;
    }

    uint8_t *dst = data_sp->GetBytes ();
    if (dst == nullptr)
    {
        error.SetErrorStringWithFormat ("DataBufferHeap instance of size %" PRIu64 " returned a null pointer", REG_CONTEXT_SIZE);
        return error;
    }

    ::memcpy (dst, &m_gpr_mips64, GetRegisterInfoInterface ().GetGPRSize ());
    dst += GetRegisterInfoInterface ().GetGPRSize ();

    ::memcpy (dst, &m_fpr, sizeof(FPR_mips));

    return error;
}

Error
NativeRegisterContextLinux_mips64::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
{
    Error error;

    if (!data_sp)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_mips64::%s invalid data_sp provided", __FUNCTION__);
        return error;
    }

    if (data_sp->GetByteSize () != REG_CONTEXT_SIZE)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_mips64::%s data_sp contained mismatched data size, expected %" PRIu64 ", actual %" PRIu64, __FUNCTION__, REG_CONTEXT_SIZE, data_sp->GetByteSize ());
        return error;
    }


    uint8_t *src = data_sp->GetBytes ();
    if (src == nullptr)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_mips64::%s DataBuffer::GetBytes() returned a null pointer", __FUNCTION__);
        return error;
    }
    ::memcpy (&m_gpr_mips64, src, GetRegisterInfoInterface ().GetGPRSize ());
    src += GetRegisterInfoInterface ().GetGPRSize ();

    ::memcpy (&m_fpr, src, sizeof(FPR_mips));

    error = WriteGPR();
    if (!error.Success())
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_mips64::%s WriteGPR() failed", __FUNCTION__);
        return error;
    }

    error = WriteFPR();
    if (!error.Success())
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_mips64::%s WriteFPR() failed", __FUNCTION__);
        return error;
    }

    return error;
}

bool
NativeRegisterContextLinux_mips64::IsFPR(uint32_t reg_index) const
{
    return (m_reg_info.first_fpr <= reg_index && reg_index <= m_reg_info.last_fpr);
}

Error
NativeRegisterContextLinux_mips64::IsWatchpointHit (uint32_t wp_index, bool &is_hit)
{
    is_hit = false;
    return Error("MIPS TODO: NativeRegisterContextLinux_mips64::IsWatchpointHit not implemented");
}

Error
NativeRegisterContextLinux_mips64::IsWatchpointVacant (uint32_t wp_index, bool &is_vacant)
{
    is_vacant = false;
    return Error("MIPS TODO: NativeRegisterContextLinux_mips64::IsWatchpointVacant not implemented");
}

bool
NativeRegisterContextLinux_mips64::ClearHardwareWatchpoint (uint32_t wp_index)
{
    return false;
}

Error
NativeRegisterContextLinux_mips64::ClearAllHardwareWatchpoints ()
{
    Error error;
    error.SetErrorString ("MIPS TODO: NativeRegisterContextLinux_mips64::ClearAllHardwareWatchpoints not implemented");
    return error;
}

Error
NativeRegisterContextLinux_mips64::SetHardwareWatchpointWithIndex (
        lldb::addr_t addr, size_t size, uint32_t watch_flags, uint32_t wp_index) 
{
    Error error;
    error.SetErrorString ("MIPS TODO: NativeRegisterContextLinux_mips64::SetHardwareWatchpointWithIndex not implemented");
    return error;
}

uint32_t
NativeRegisterContextLinux_mips64::SetHardwareWatchpoint (
        lldb::addr_t addr, size_t size, uint32_t watch_flags)
{
    return LLDB_INVALID_INDEX32;
}

lldb::addr_t
NativeRegisterContextLinux_mips64::GetWatchpointAddress (uint32_t wp_index)
{
    return LLDB_INVALID_ADDRESS;
}

uint32_t
NativeRegisterContextLinux_mips64::NumSupportedHardwareWatchpoints ()
{
    return 0;
}

NativeProcessLinux::OperationUP
NativeRegisterContextLinux_mips64::GetReadRegisterValueOperation(uint32_t offset,
                                                                 const char* reg_name,
                                                                 uint32_t size,
                                                                 RegisterValue &value)
{
    return NativeProcessLinux::OperationUP(new ReadRegOperation(m_thread.GetID(), offset, value));
}

NativeProcessLinux::OperationUP
NativeRegisterContextLinux_mips64::GetWriteRegisterValueOperation(uint32_t offset,
                                                                  const char* reg_name,
                                                                  const RegisterValue &value)
{
    return NativeProcessLinux::OperationUP(new WriteRegOperation(m_thread.GetID(), offset, reg_name, value));
}

#endif // defined (__mips__)
