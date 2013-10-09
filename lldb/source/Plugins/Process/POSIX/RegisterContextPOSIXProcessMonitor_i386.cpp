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
                                                                                 uint32_t concrete_frame_idx,
                                                                                 RegisterInfoInterface *register_info)
    : RegisterContextPOSIX_i386(thread, concrete_frame_idx, register_info)
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
    assert(false);
    return false;
}

bool
RegisterContextPOSIXProcessMonitor_i386::ReadFPR()
{
    assert(false);
    return false;
}

bool
RegisterContextPOSIXProcessMonitor_i386::WriteGPR()
{
    assert(false);
    return false;
}

bool
RegisterContextPOSIXProcessMonitor_i386::WriteFPR()
{
    assert(false);
    return false;
}

bool
RegisterContextPOSIXProcessMonitor_i386::ReadRegister(const RegisterInfo *reg_info,
                                                      RegisterValue &value)
{
    assert(false);
    return false;
}

bool RegisterContextPOSIXProcessMonitor_i386::WriteRegister(const RegisterInfo *reg_info,
                                                            const RegisterValue &value)
{
    assert(false);
    return false;        
}
