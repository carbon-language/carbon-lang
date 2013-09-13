//===-- RegisterContextPOSIXProcessMonitor_x86_64.h ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "lldb/Target/Thread.h"

#include "ProcessPOSIX.h"
#include "RegisterContextPOSIXProcessMonitor_x86_64.h"
#include "ProcessMonitor.h"

using namespace lldb_private;
using namespace lldb;

// Support ptrace extensions even when compiled without required kernel support
#ifndef NT_X86_XSTATE
  #define NT_X86_XSTATE 0x202
#endif

RegisterContextPOSIXProcessMonitor_x86_64::RegisterContextPOSIXProcessMonitor_x86_64(Thread &thread,
                                                                                     uint32_t concrete_frame_idx,
                                                                                     RegisterInfoInterface *register_info)
    : RegisterContextPOSIX_x86_64(thread, concrete_frame_idx, register_info)
{
}

ProcessMonitor &
RegisterContextPOSIXProcessMonitor_x86_64::GetMonitor()
{
    ProcessSP base = CalculateProcess();
    ProcessPOSIX *process = static_cast<ProcessPOSIX*>(base.get());
    return process->GetMonitor();
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::ReadGPR()
{
     ProcessMonitor &monitor = GetMonitor();
     return monitor.ReadGPR(m_thread.GetID(), &m_gpr, GetGPRSize());
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::ReadFPR()
{
    ProcessMonitor &monitor = GetMonitor();
    if (m_fpr_type == eFXSAVE)
        return monitor.ReadFPR(m_thread.GetID(), &m_fpr.xstate.fxsave, sizeof(m_fpr.xstate.fxsave));

    if (m_fpr_type == eXSAVE)
        return monitor.ReadRegisterSet(m_thread.GetID(), &m_iovec, sizeof(m_fpr.xstate.xsave), NT_X86_XSTATE);
    return false;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::WriteGPR()
{
    ProcessMonitor &monitor = GetMonitor();
    return monitor.WriteGPR(m_thread.GetID(), &m_gpr, GetGPRSize());
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::WriteFPR()
{
    ProcessMonitor &monitor = GetMonitor();
    if (m_fpr_type == eFXSAVE)
        return monitor.WriteFPR(m_thread.GetID(), &m_fpr.xstate.fxsave, sizeof(m_fpr.xstate.fxsave));

    if (m_fpr_type == eXSAVE)
        return monitor.WriteRegisterSet(m_thread.GetID(), &m_iovec, sizeof(m_fpr.xstate.xsave), NT_X86_XSTATE);
    return false;
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::ReadRegister(const unsigned reg,
                                                        RegisterValue &value)
{
    ProcessMonitor &monitor = GetMonitor();
    return monitor.ReadRegisterValue(m_thread.GetID(),
                                     GetRegisterOffset(reg),
                                     GetRegisterName(reg),
                                     GetRegisterSize(reg),
                                     value);
}

bool
RegisterContextPOSIXProcessMonitor_x86_64::WriteRegister(const unsigned reg,
                                                         const RegisterValue &value)
{
    ProcessMonitor &monitor = GetMonitor();
    return monitor.WriteRegisterValue(m_thread.GetID(),
                                      GetRegisterOffset(reg),
                                      GetRegisterName(reg),
                                      value);
}
