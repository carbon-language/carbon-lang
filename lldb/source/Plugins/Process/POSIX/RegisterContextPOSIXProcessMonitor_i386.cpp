//===-- RegisterContextPOSIXProcessMonitor_i386.h --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "lldb/Target/Thread.h"

#include "ProcessPOSIX.h"
#include "RegisterContextPOSIXProcessMonitor_i386.h"
#include "ProcessMonitor.h"

using namespace lldb_private;
using namespace lldb;

RegisterContextPOSIXProcessMonitor_i386::RegisterContextPOSIXProcessMonitor_i386(Thread &thread,
                                                                                 uint32_t concrete_frame_idx)
    : RegisterContextPOSIX_i386(thread, concrete_frame_idx)
{
}

ProcessMonitor &
RegisterContextPOSIXProcessMonitor_i386::GetMonitor()
{
    ProcessSP base = CalculateProcess();
    ProcessPOSIX *process = static_cast<ProcessPOSIX*>(base.get());
    return process->GetMonitor();
}

bool
RegisterContextPOSIXProcessMonitor_i386::ReadGPR()
{
    bool result;

    ProcessMonitor &monitor = GetMonitor();
    result = monitor.ReadGPR(m_thread.GetID(), &m_user.regs, sizeof(m_user.regs));
    LogGPR("RegisterContextPOSIXProcessMonitor_i386::ReadGPR()");
    return result;
}

bool
RegisterContextPOSIXProcessMonitor_i386::ReadFPR()
{
    ProcessMonitor &monitor = GetMonitor();
    return monitor.ReadFPR(m_thread.GetID(), &m_user.i387, sizeof(m_user.i387));
}

bool
RegisterContextPOSIXProcessMonitor_i386::WriteGPR()
{
    return false;
}

bool
RegisterContextPOSIXProcessMonitor_i386::WriteFPR()
{
    return false;
}

bool
RegisterContextPOSIXProcessMonitor_i386::ReadRegister(const RegisterInfo *reg_info,
                                                      RegisterValue &value)
{
    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    ProcessMonitor &monitor = GetMonitor();
    return monitor.ReadRegisterValue(m_thread.GetID(), GetRegOffset(reg),
                                     GetRegisterName(reg), GetRegSize(reg), value);
}

bool RegisterContextPOSIXProcessMonitor_i386::WriteRegister(const RegisterInfo *reg_info,
                                                            const RegisterValue &value)
{
    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    ProcessMonitor &monitor = GetMonitor();
    return monitor.WriteRegisterValue(m_thread.GetID(), GetRegOffset(reg),
                                      GetRegisterName(reg), value);
}
