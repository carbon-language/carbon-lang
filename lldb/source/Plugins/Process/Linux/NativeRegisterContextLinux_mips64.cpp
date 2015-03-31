//===-- NativeRegisterContextLinux_mips64.cpp ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextLinux_mips64.h"

#include "lldb/lldb-private-forward.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "Plugins/Process/Linux/NativeProcessLinux.h"

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
}

// ----------------------------------------------------------------------------
// NativeRegisterContextLinux_mips64 members.
// ----------------------------------------------------------------------------

NativeRegisterContextLinux_mips64::NativeRegisterContextLinux_mips64 (NativeThreadProtocol &native_thread, 
        uint32_t concrete_frame_idx, RegisterInfoInterface *reg_info_interface_p) :
    NativeRegisterContextRegisterInfo (native_thread, concrete_frame_idx, reg_info_interface_p)
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

Error
NativeRegisterContextLinux_mips64::ReadRegister (const RegisterInfo *reg_info, RegisterValue &reg_value)
{
    Error error;
    error.SetErrorString ("MIPS TODO: NativeRegisterContextLinux_mips64::ReadRegister not implemented");
    return error;
}

Error
NativeRegisterContextLinux_mips64::WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value)
{
    Error error;
    error.SetErrorString ("MIPS TODO: NativeRegisterContextLinux_mips64::WriteRegister not implemented");
    return error;
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
