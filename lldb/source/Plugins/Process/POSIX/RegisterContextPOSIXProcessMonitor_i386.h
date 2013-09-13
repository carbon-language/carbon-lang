//===-- RegisterContextPOSIXProcessMonitor_i386.h ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextPOSIXProcessMonitor_i386_H_
#define liblldb_RegisterContextPOSIXProcessMonitor_i386_H_

#include "Plugins/Process/POSIX/RegisterContextPOSIX_i386.h"

class RegisterContextPOSIXProcessMonitor_i386:
    public RegisterContextPOSIX_i386
{
public:
    RegisterContextPOSIXProcessMonitor_i386(lldb_private::Thread &thread,
                                            uint32_t concrete_frame_idx);

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
	ReadRegister(const lldb_private::RegisterInfo *reg_info, lldb_private::RegisterValue &value);

	bool
	WriteRegister(const lldb_private::RegisterInfo *reg_info, const lldb_private::RegisterValue &value);

private:
    ProcessMonitor &
    GetMonitor();
};

#endif
