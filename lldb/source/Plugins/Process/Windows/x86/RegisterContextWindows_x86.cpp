//===-- RegisterContextWindows_x86.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private-types.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"

#include "lldb-x86-register-enums.h"
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

const DWORD kWinContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;

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
    : RegisterContext(thread, concrete_frame_idx)
    , m_context_stale(true)
    , m_context_ptr(nullptr)
{
}

RegisterContextWindows_x86::~RegisterContextWindows_x86()
{
}

void
RegisterContextWindows_x86::InvalidateAllRegisters()
{
    m_context_stale = true;
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

    switch (reg_info->kinds[eRegisterKindLLDB])
    {
        case lldb_eax_i386:
            reg_value.SetUInt32(m_context_ptr->Eax);
            break;
        case lldb_ebx_i386:
            reg_value.SetUInt32(m_context_ptr->Ebx);
            break;
        case lldb_ecx_i386:
            reg_value.SetUInt32(m_context_ptr->Ecx);
            break;
        case lldb_edx_i386:
            reg_value.SetUInt32(m_context_ptr->Edx);
            break;
        case lldb_edi_i386:
            reg_value.SetUInt32(m_context_ptr->Edi);
            break;
        case lldb_esi_i386:
            reg_value.SetUInt32(m_context_ptr->Esi);
            break;
        case lldb_ebp_i386:
            reg_value.SetUInt32(m_context_ptr->Ebp);
            break;
        case lldb_esp_i386:
            reg_value.SetUInt32(m_context_ptr->Esp);
            break;
        case lldb_eip_i386:
            reg_value.SetUInt32(m_context_ptr->Eip);
            break;
        case lldb_eflags_i386:
            reg_value.SetUInt32(m_context_ptr->EFlags);
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

    switch (reg_info->kinds[eRegisterKindLLDB])
    {
        case lldb_eax_i386:
            m_context_ptr->Eax = reg_value.GetAsUInt32();
            break;
        case lldb_ebx_i386:
            m_context_ptr->Ebx = reg_value.GetAsUInt32();
            break;
        case lldb_ecx_i386:
            m_context_ptr->Ecx = reg_value.GetAsUInt32();
            break;
        case lldb_edx_i386:
            m_context_ptr->Edx = reg_value.GetAsUInt32();
            break;
        case lldb_edi_i386:
            m_context_ptr->Edi = reg_value.GetAsUInt32();
            break;
        case lldb_esi_i386:
            m_context_ptr->Esi = reg_value.GetAsUInt32();
            break;
        case lldb_ebp_i386:
            m_context_ptr->Ebp = reg_value.GetAsUInt32();
            break;
        case lldb_esp_i386:
            m_context_ptr->Esp = reg_value.GetAsUInt32();
            break;
        case lldb_eip_i386:
            m_context_ptr->Eip = reg_value.GetAsUInt32();
            break;
        case lldb_eflags_i386:
            m_context_ptr->EFlags = reg_value.GetAsUInt32();
            break;
    }

    // Physically update the registers in the target process.
    TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);
    return ::SetThreadContext(wthread.GetHostThread().GetNativeThread().GetSystemHandle(), m_context_ptr);
}

bool
RegisterContextWindows_x86::ReadAllRegisterValues(lldb::DataBufferSP &data_sp)
{
    if (!CacheAllRegisterValues())
        return false;

    CONTEXT *dest_context = nullptr;
    if (!InitializeContextDataBuffer(data_sp, &dest_context))
        return false;

    // Write the OS's internal CONTEXT structure into the buffer.
    if (!CopyContext(dest_context, kWinContextFlags, m_context_ptr))
        return false;
    return true;
}

bool
RegisterContextWindows_x86::WriteAllRegisterValues(const lldb::DataBufferSP &data_sp)
{
    // data_sp could only ever have been generated by a call to ReadAllRegisterValues(), so
    // m_cached_context should already have the correct size and alignment properties.
    assert(m_cached_context->GetByteSize() == data_sp->GetByteSize());

    // As a result, we can simply memcpy the entire buffer and assume that the alignment remains
    // the same.
    memcpy(m_cached_context->GetBytes(), data_sp->GetBytes(), data_sp->GetByteSize());

    // m_context_ptr still points to the beginning of the CONTEXT structure, so use that for
    // updating the thread state.
    TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);
    if (!::SetThreadContext(wthread.GetHostThread().GetNativeThread().GetSystemHandle(), m_context_ptr))
        return false;

    return true;
}

uint32_t
RegisterContextWindows_x86::ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind, uint32_t num)
{
    const uint32_t num_regs = GetRegisterCount();

    assert(kind < kNumRegisterKinds);
    for (uint32_t reg_idx = 0; reg_idx < num_regs; ++reg_idx)
    {
        const RegisterInfo *reg_info = GetRegisterInfoAtIndex(reg_idx);

        if (reg_info->kinds[kind] == num)
            return reg_idx;
    }

    return LLDB_INVALID_REGNUM;
}

//------------------------------------------------------------------
// Subclasses can these functions if desired
//------------------------------------------------------------------
uint32_t
RegisterContextWindows_x86::NumSupportedHardwareBreakpoints()
{
    // Support for hardware breakpoints not yet implemented.
    return 0;
}

uint32_t
RegisterContextWindows_x86::SetHardwareBreakpoint(lldb::addr_t addr, size_t size)
{
    return 0;
}

bool
RegisterContextWindows_x86::ClearHardwareBreakpoint(uint32_t hw_idx)
{
    return false;
}

uint32_t
RegisterContextWindows_x86::NumSupportedHardwareWatchpoints()
{
    // Support for hardware watchpoints not yet implemented.
    return 0;
}

uint32_t
RegisterContextWindows_x86::SetHardwareWatchpoint(lldb::addr_t addr, size_t size, bool read, bool write)
{
    return 0;
}

bool
RegisterContextWindows_x86::ClearHardwareWatchpoint(uint32_t hw_index)
{
    return false;
}

bool
RegisterContextWindows_x86::HardwareSingleStep(bool enable)
{
    return false;
}

bool
RegisterContextWindows_x86::InitializeContextDataBuffer(DataBufferSP &buffer, CONTEXT **context_ptr)
{
    DWORD length = 0;
    if (!::InitializeContext(nullptr, kWinContextFlags, nullptr, &length) && GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return false;

    buffer.reset(new DataBufferHeap(length, 0));
    if (!::InitializeContext(buffer->GetBytes(), kWinContextFlags, context_ptr, &length))
    {
        buffer.reset();
        return false;
    }
    return true;
}

bool
RegisterContextWindows_x86::CacheAllRegisterValues()
{
    if (!m_context_stale)
        return true;

    if (!m_cached_context && !InitializeContextDataBuffer(m_cached_context, &m_context_ptr))
        return false;

    TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);
    if (!::GetThreadContext(wthread.GetHostThread().GetNativeThread().GetSystemHandle(), m_context_ptr))
        return false;
    m_context_stale = false;
    return true;
}
