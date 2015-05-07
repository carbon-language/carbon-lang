//===-- RegisterContextWindows_x86.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private-types.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"

#include "lldb-x86-register-enums.h"
#include "ProcessWindowsLog.h"
#include "RegisterContext_x86.h"
#include "RegisterContextWindows_x86.h"
#include "TargetThreadWindows.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

#define DEFINE_GPR(reg, alt) #reg, alt, 4, 0, eEncodingUint, eFormatHexUppercase
#define DEFINE_GPR_BIN(reg, alt) #reg, alt, 4, 0, eEncodingUint, eFormatBinary

namespace
{

// This enum defines the layout of the global RegisterInfo array.  This is necessary because
// lldb register sets are defined in terms of indices into the register array.  As such, the
// order of RegisterInfos defined in global registers array must match the order defined here.
// When defining the register set layouts, these values can appear in an arbitrary order, and that
// determines the order that register values are displayed in a dump.
enum RegisterIndex
{
    eRegisterIndexEax,
    eRegisterIndexEbx,
    eRegisterIndexEcx,
    eRegisterIndexEdx,
    eRegisterIndexEdi,
    eRegisterIndexEsi,
    eRegisterIndexEbp,
    eRegisterIndexEsp,
    eRegisterIndexEip,
    eRegisterIndexEflags
};

// Array of all register information supported by Windows x86
RegisterInfo g_register_infos[] =
{
//  Macro auto defines most stuff   GCC                     DWARF                GENERIC                    GDB                   LLDB               VALUE REGS    INVALIDATE REGS
//  ==============================  ======================= ===================  =========================  ===================   =================  ==========    ===============
    { DEFINE_GPR(eax,    nullptr),  { gcc_eax_i386,         dwarf_eax_i386,      LLDB_INVALID_REGNUM,       LLDB_INVALID_REGNUM,  lldb_eax_i386   },  nullptr,      nullptr},
    { DEFINE_GPR(ebx,    nullptr),  { gcc_ebx_i386,         dwarf_ebx_i386,      LLDB_INVALID_REGNUM,       LLDB_INVALID_REGNUM,  lldb_ebx_i386   },  nullptr,      nullptr},
    { DEFINE_GPR(ecx,    nullptr),  { gcc_ecx_i386,         dwarf_ecx_i386,      LLDB_INVALID_REGNUM,       LLDB_INVALID_REGNUM,  lldb_ecx_i386   },  nullptr,      nullptr},
    { DEFINE_GPR(edx,    nullptr),  { gcc_edx_i386,         dwarf_edx_i386,      LLDB_INVALID_REGNUM,       LLDB_INVALID_REGNUM,  lldb_edx_i386   },  nullptr,      nullptr},
    { DEFINE_GPR(edi,    nullptr),  { gcc_edi_i386,         dwarf_edi_i386,      LLDB_INVALID_REGNUM,       LLDB_INVALID_REGNUM,  lldb_edi_i386   },  nullptr,      nullptr},
    { DEFINE_GPR(esi,    nullptr),  { gcc_esi_i386,         dwarf_esi_i386,      LLDB_INVALID_REGNUM,       LLDB_INVALID_REGNUM,  lldb_esi_i386   },  nullptr,      nullptr},
    { DEFINE_GPR(ebp,    "fp"),     { gcc_ebp_i386,         dwarf_ebp_i386,      LLDB_REGNUM_GENERIC_FP,    LLDB_INVALID_REGNUM,  lldb_ebp_i386   },  nullptr,      nullptr},
    { DEFINE_GPR(esp,    "sp"),     { gcc_esp_i386,         dwarf_esp_i386,      LLDB_REGNUM_GENERIC_SP,    LLDB_INVALID_REGNUM,  lldb_esp_i386   },  nullptr,      nullptr},
    { DEFINE_GPR(eip,    "pc"),     { gcc_eip_i386,         dwarf_eip_i386,      LLDB_REGNUM_GENERIC_PC,    LLDB_INVALID_REGNUM,  lldb_eip_i386   },  nullptr,      nullptr},
    { DEFINE_GPR_BIN(eflags, "flags"), { gcc_eflags_i386,   dwarf_eflags_i386,   LLDB_REGNUM_GENERIC_FLAGS, LLDB_INVALID_REGNUM,  lldb_eflags_i386},  nullptr,      nullptr},
};

// Array of lldb register numbers used to define the set of all General Purpose Registers
uint32_t g_gpr_reg_indices[] =
{
    eRegisterIndexEax,
    eRegisterIndexEbx,
    eRegisterIndexEcx,
    eRegisterIndexEdx,
    eRegisterIndexEdi,
    eRegisterIndexEsi,
    eRegisterIndexEbp,
    eRegisterIndexEsp,
    eRegisterIndexEip,
    eRegisterIndexEflags
};

RegisterSet g_register_sets[] = {
    {"General Purpose Registers", "gpr", llvm::array_lengthof(g_gpr_reg_indices), g_gpr_reg_indices},
};
}

//------------------------------------------------------------------
// Constructors and Destructors
//------------------------------------------------------------------
RegisterContextWindows_x86::RegisterContextWindows_x86(Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows(thread, concrete_frame_idx)
{
}

RegisterContextWindows_x86::~RegisterContextWindows_x86()
{
}

size_t
RegisterContextWindows_x86::GetRegisterCount()
{
    return llvm::array_lengthof(g_register_infos);
}

const RegisterInfo *
RegisterContextWindows_x86::GetRegisterInfoAtIndex(size_t reg)
{
    return &g_register_infos[reg];
}

size_t
RegisterContextWindows_x86::GetRegisterSetCount()
{
    return llvm::array_lengthof(g_register_sets);
}

const RegisterSet *
RegisterContextWindows_x86::GetRegisterSet(size_t reg_set)
{
    return &g_register_sets[reg_set];
}

bool
RegisterContextWindows_x86::ReadRegister(const RegisterInfo *reg_info, RegisterValue &reg_value)
{
    if (!CacheAllRegisterValues())
        return false;

    uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    switch (reg)
    {
        case lldb_eax_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Read value 0x%x from EAX", m_context.Eax);
            reg_value.SetUInt32(m_context.Eax);
            break;
        case lldb_ebx_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Read value 0x%x from EBX", m_context.Ebx);
            reg_value.SetUInt32(m_context.Ebx);
            break;
        case lldb_ecx_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Read value 0x%x from ECX", m_context.Ecx);
            reg_value.SetUInt32(m_context.Ecx);
            break;
        case lldb_edx_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Read value 0x%x from EDX", m_context.Edx);
            reg_value.SetUInt32(m_context.Edx);
            break;
        case lldb_edi_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Read value 0x%x from EDI", m_context.Edi);
            reg_value.SetUInt32(m_context.Edi);
            break;
        case lldb_esi_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Read value 0x%x from ESI", m_context.Esi);
            reg_value.SetUInt32(m_context.Esi);
            break;
        case lldb_ebp_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Read value 0x%x from EBP", m_context.Ebp);
            reg_value.SetUInt32(m_context.Ebp);
            break;
        case lldb_esp_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Read value 0x%x from ESP", m_context.Esp);
            reg_value.SetUInt32(m_context.Esp);
            break;
        case lldb_eip_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Read value 0x%x from EIP", m_context.Eip);
            reg_value.SetUInt32(m_context.Eip);
            break;
        case lldb_eflags_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Read value 0x%x from EFLAGS", m_context.EFlags);
            reg_value.SetUInt32(m_context.EFlags);
            break;
        default:
            WINWARN_IFALL(WINDOWS_LOG_REGISTERS, "Requested unknown register %u", reg);
            break;
    }
    return true;
}

bool
RegisterContextWindows_x86::WriteRegister(const RegisterInfo *reg_info, const RegisterValue &reg_value)
{
    // Since we cannot only write a single register value to the inferior, we need to make sure
    // our cached copy of the register values are fresh.  Otherwise when writing EAX, for example,
    // we may also overwrite some other register with a stale value.
    if (!CacheAllRegisterValues())
        return false;

    uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    switch (reg)
    {
        case lldb_eax_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EAX", reg_value.GetAsUInt32());
            m_context.Eax = reg_value.GetAsUInt32();
            break;
        case lldb_ebx_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EBX", reg_value.GetAsUInt32());
            m_context.Ebx = reg_value.GetAsUInt32();
            break;
        case lldb_ecx_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to ECX", reg_value.GetAsUInt32());
            m_context.Ecx = reg_value.GetAsUInt32();
            break;
        case lldb_edx_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EDX", reg_value.GetAsUInt32());
            m_context.Edx = reg_value.GetAsUInt32();
            break;
        case lldb_edi_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EDI", reg_value.GetAsUInt32());
            m_context.Edi = reg_value.GetAsUInt32();
            break;
        case lldb_esi_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to ESI", reg_value.GetAsUInt32());
            m_context.Esi = reg_value.GetAsUInt32();
            break;
        case lldb_ebp_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EBP", reg_value.GetAsUInt32());
            m_context.Ebp = reg_value.GetAsUInt32();
            break;
        case lldb_esp_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to ESP", reg_value.GetAsUInt32());
            m_context.Esp = reg_value.GetAsUInt32();
            break;
        case lldb_eip_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EIP", reg_value.GetAsUInt32());
            m_context.Eip = reg_value.GetAsUInt32();
            break;
        case lldb_eflags_i386:
            WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EFLAGS", reg_value.GetAsUInt32());
            m_context.EFlags = reg_value.GetAsUInt32();
            break;
        default:
            WINWARN_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to unknown register %u", reg_value.GetAsUInt32(),
                          reg);
    }

    // Physically update the registers in the target process.
    TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);
    return ::SetThreadContext(wthread.GetHostThread().GetNativeThread().GetSystemHandle(), &m_context);
}
