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

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "Plugins/Process/Utility/RegisterContextLinux_mips64.h"

using namespace lldb_private;
using namespace lldb_private::process_linux;

// ----------------------------------------------------------------------------
// Private namespace.
// ----------------------------------------------------------------------------

namespace
{
    // mips64 general purpose registers.
    const uint32_t
    g_gp_regnums_mips64[] =
    {
        gp_reg_r0_mips64,
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
        LLDB_INVALID_REGNUM     // register sets need to end with this flag
    };

    static_assert((sizeof(g_gp_regnums_mips64) / sizeof(g_gp_regnums_mips64[0])) - 1 == k_num_gp_reg_mips64,
                  "g_gp_regnums_mips64 has wrong number of register infos");

    // Number of register sets provided by this context.
    enum
    {
        k_num_register_sets = 1
    };

    // Register sets for mips64.
    static const RegisterSet
    g_reg_sets_mips64[k_num_register_sets] =
    {
        { "General Purpose Registers",  "gpr", k_num_gp_reg_mips64, g_gp_regnums_mips64 }
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

// ----------------------------------------------------------------------------
// NativeRegisterContextLinux_mips64 members.
// ----------------------------------------------------------------------------

NativeRegisterContextLinux_mips64::NativeRegisterContextLinux_mips64 (const ArchSpec& target_arch,
                                                                      NativeThreadProtocol &native_thread, 
                                                                      uint32_t concrete_frame_idx) :
    NativeRegisterContextLinux (native_thread, concrete_frame_idx, new RegisterContextLinux_mips64 (target_arch))
{
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

    error = ReadRegisterRaw(reg, reg_value);

    if (error.Success ())
    {
        // If our return byte size was greater than the return value reg size, then
        // use the type specified by reg_info rather than the uint64_t default
        if (reg_value.GetByteSize() > reg_info->byte_size)
            reg_value.SetType(reg_info);
    }

    return error;
}

lldb_private::Error
NativeRegisterContextLinux_mips64::WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value)
{
    assert (reg_info && "reg_info is null");

    const uint32_t reg_index = reg_info->kinds[lldb::eRegisterKindLLDB];

    if (reg_index == LLDB_INVALID_REGNUM)
        return Error ("no lldb regnum for %s", reg_info && reg_info->name ? reg_info->name : "<unknown register>");

    return WriteRegisterRaw(reg_index, reg_value);
}

Error
NativeRegisterContextLinux_mips64::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    Error error;
    error.SetErrorString ("MIPS TODO: NativeRegisterContextLinux_mips64::ReadAllRegisterValues not implemented");
    return error;
}

Error
NativeRegisterContextLinux_mips64::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
{
    Error error;
    error.SetErrorString ("MIPS TODO: NativeRegisterContextLinux_mips64::WriteAllRegisterValues not implemented");
    return error;
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
