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

#define DEFINE_GPR32(reg, offset, generic_reg)                                                                                             \
    {                                                                                                                                      \
        #reg, nullptr, 4, offset, eEncodingUint, eFormatHexUppercase,                                                                      \
            {gcc_##reg##_i386, dwarf_##reg##_i386, generic_reg, gdb_##reg##_i386, gpr_##reg##_i386 }, nullptr, nullptr                     \
    }

#define GPR_REGNUM(reg) gpr_##reg##_i386

// For now we're only supporting general purpose registers.  Unfortunately we have to maintain
// parallel arrays since that's how the RegisterContext interface expects things to be returned.
// We might be able to fix this by initializing these arrays at runtime during the construction of
// the RegisterContext by using helper functions that can update multiple arrays, register sets,
// etc all at once through a more easily understandable interface.

RegisterInfo g_register_infos[] = {DEFINE_GPR32(eax, offsetof(CONTEXT, Eax), LLDB_INVALID_REGNUM),
                                   DEFINE_GPR32(ebx, offsetof(CONTEXT, Ebx), LLDB_INVALID_REGNUM),
                                   DEFINE_GPR32(ecx, offsetof(CONTEXT, Ecx), LLDB_INVALID_REGNUM),
                                   DEFINE_GPR32(edx, offsetof(CONTEXT, Edx), LLDB_INVALID_REGNUM),
                                   DEFINE_GPR32(edi, offsetof(CONTEXT, Edi), LLDB_INVALID_REGNUM),
                                   DEFINE_GPR32(esi, offsetof(CONTEXT, Esi), LLDB_INVALID_REGNUM),
                                   DEFINE_GPR32(ebp, offsetof(CONTEXT, Ebp), LLDB_REGNUM_GENERIC_FP),
                                   DEFINE_GPR32(esp, offsetof(CONTEXT, Esp), LLDB_REGNUM_GENERIC_SP),
                                   DEFINE_GPR32(eip, offsetof(CONTEXT, Eip), LLDB_REGNUM_GENERIC_PC),
                                   DEFINE_GPR32(eflags, offsetof(CONTEXT, EFlags), LLDB_REGNUM_GENERIC_FLAGS)};

uint32_t g_gpr_regnums[] = {
    GPR_REGNUM(eax), GPR_REGNUM(ebx), GPR_REGNUM(ecx), GPR_REGNUM(edx), GPR_REGNUM(edi),
    GPR_REGNUM(esi), GPR_REGNUM(ebp), GPR_REGNUM(esp), GPR_REGNUM(eip), GPR_REGNUM(eflags),
};

RegisterSet g_register_sets[] = {{"General Purpose Registers", "gpr", llvm::array_lengthof(g_register_infos), g_gpr_regnums}};

//------------------------------------------------------------------
// Constructors and Destructors
//------------------------------------------------------------------
RegisterContextWindows_x86::RegisterContextWindows_x86(Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContext(thread, concrete_frame_idx)
    , m_context_valid(false)
    , m_cached_context(new DataBufferHeap(sizeof(CONTEXT), 0))
{
}

RegisterContextWindows_x86::~RegisterContextWindows_x86()
{
}

void
RegisterContextWindows_x86::InvalidateAllRegisters()
{
    m_context_valid = false;
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
    // For now we're reading the value of every register, and then returning the one that was
    // requested.  We should be smarter about this in the future.
    if (!CacheAllRegisterValues())
        return false;

    CONTEXT *context = GetSystemContext();
    switch (reg_info->kinds[eRegisterKindLLDB])
    {
        case gpr_eax_i386:
            reg_value.SetUInt32(context->Eax);
            break;
        case gpr_ebx_i386:
            reg_value.SetUInt32(context->Ebx);
            break;
        case gpr_ecx_i386:
            reg_value.SetUInt32(context->Ecx);
            break;
        case gpr_edx_i386:
            reg_value.SetUInt32(context->Edx);
            break;
        case gpr_edi_i386:
            reg_value.SetUInt32(context->Edi);
            break;
        case gpr_esi_i386:
            reg_value.SetUInt32(context->Esi);
            break;
        case gpr_ebp_i386:
            reg_value.SetUInt32(context->Ebp);
            break;
        case gpr_esp_i386:
            reg_value.SetUInt32(context->Esp);
            break;
        case gpr_eip_i386:
            reg_value.SetUInt32(context->Eip);
            break;
        case gpr_eflags_i386:
            reg_value.SetUInt32(context->EFlags);
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

    CONTEXT *context = GetSystemContext();
    switch (reg_info->kinds[eRegisterKindLLDB])
    {
        case gpr_eax_i386:
            context->Eax = reg_value.GetAsUInt32();
            break;
        case gpr_ebx_i386:
            context->Ebx = reg_value.GetAsUInt32();
            break;
        case gpr_ecx_i386:
            context->Ecx = reg_value.GetAsUInt32();
            break;
        case gpr_edx_i386:
            context->Edx = reg_value.GetAsUInt32();
            break;
        case gpr_edi_i386:
            context->Edi = reg_value.GetAsUInt32();
            break;
        case gpr_esi_i386:
            context->Esi = reg_value.GetAsUInt32();
            break;
        case gpr_ebp_i386:
            context->Ebp = reg_value.GetAsUInt32();
            break;
        case gpr_esp_i386:
            context->Esp = reg_value.GetAsUInt32();
            break;
        case gpr_eip_i386:
            context->Eip = reg_value.GetAsUInt32();
            break;
        case gpr_eflags_i386:
            context->EFlags = reg_value.GetAsUInt32();
            break;
    }

    // Physically update the registers in the target process.
    TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);
    return ::SetThreadContext(wthread.GetHostThread().GetNativeThread().GetSystemHandle(), context);
}

bool
RegisterContextWindows_x86::ReadAllRegisterValues(lldb::DataBufferSP &data_sp)
{
    if (!CacheAllRegisterValues())
        return false;

    if (data_sp->GetByteSize() != m_cached_context->GetByteSize())
        return false;

    // Write the OS's internal CONTEXT structure into the buffer.
    memcpy(data_sp->GetBytes(), m_cached_context->GetBytes(), data_sp->GetByteSize());
    return true;
}

bool
RegisterContextWindows_x86::WriteAllRegisterValues(const lldb::DataBufferSP &data_sp)
{
    // Since we're given every register value in the input buffer, we don't need to worry about
    // making sure our cached copy is valid and then overwriting the modified values before we
    // push the full update to the OS.  We do however need to update the cached copy with the value
    // that we're pushing to the OS.
    if (data_sp->GetByteSize() != m_cached_context->GetByteSize())
        return false;

    CONTEXT *context = GetSystemContext();
    TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);

    if (!::SetThreadContext(wthread.GetHostThread().GetNativeThread().GetSystemHandle(), context))
        return false;

    // Since the thread context was set successfully, update our cached copy.
    memcpy(m_cached_context->GetBytes(), data_sp->GetBytes(), data_sp->GetByteSize());
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

CONTEXT *
RegisterContextWindows_x86::GetSystemContext()
{
    return reinterpret_cast<CONTEXT *>(m_cached_context->GetBytes());
}

bool
RegisterContextWindows_x86::CacheAllRegisterValues()
{
    if (m_context_valid)
        return true;

    CONTEXT *context = GetSystemContext();
    // Right now we just pull every single register, regardless of what register we're ultimately
    // going to read.  We could be smarter about this, although it's not clear what the advantage
    // would be.
    context->ContextFlags = CONTEXT_FULL | CONTEXT_DEBUG_REGISTERS;
    TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);
    if (!::GetThreadContext(wthread.GetHostThread().GetNativeThread().GetSystemHandle(), context))
        return false;
    m_context_valid = true;
    return true;
}
