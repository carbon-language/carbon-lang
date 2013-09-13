//===-- RegisterContextPOSIXProcessMonitor_x86_64.h -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextPOSIXProcessMonitor_x86_64_H_
#define liblldb_RegisterContextPOSIXProcessMonitor_x86_64_H_

#include "Plugins/Process/POSIX/RegisterContextPOSIX_x86_64.h"

class RegisterContextPOSIXProcessMonitor_x86_64:
    public RegisterContextPOSIX_x86_64
{
public:
    RegisterContextPOSIXProcessMonitor_x86_64(lldb_private::Thread &thread,
                                              uint32_t concrete_frame_idx,
                                              RegisterInfoInterface *register_info);

protected:
    bool
    ReadGPR();

    bool
    ReadFPR();

    bool
    WriteGPR();

    bool
    WriteFPR();

    bool
    ReadRegister(const unsigned reg, lldb_private::RegisterValue &value);

    bool
    WriteRegister(const unsigned reg, const lldb_private::RegisterValue &value);

private:
    ProcessMonitor &
    GetMonitor();
};

#endif
