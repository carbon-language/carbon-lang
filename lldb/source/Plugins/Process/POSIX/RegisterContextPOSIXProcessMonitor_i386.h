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
    public RegisterContextPOSIX_i386,
    public POSIXBreakpointProtocol
{
public:
    RegisterContextPOSIXProcessMonitor_i386(lldb_private::Thread &thread,
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
	ReadRegister(const lldb_private::RegisterInfo *reg_info, lldb_private::RegisterValue &value);

	bool
	WriteRegister(const lldb_private::RegisterInfo *reg_info, const lldb_private::RegisterValue &value);

    // POSIXBreakpointProtocol
    virtual bool
    UpdateAfterBreakpoint() { return true; }

    virtual unsigned
    GetRegisterIndexFromOffset(unsigned offset) { return LLDB_INVALID_REGNUM; }

    virtual bool
    IsWatchpointHit (uint32_t hw_index) { return false; }

    virtual bool
    ClearWatchpointHits () { return false; }

    virtual lldb::addr_t
    GetWatchpointAddress (uint32_t hw_index) { return LLDB_INVALID_ADDRESS; }

    virtual bool
    IsWatchpointVacant (uint32_t hw_index) { return false; }

    virtual bool
    SetHardwareWatchpointWithIndex (lldb::addr_t addr, size_t size,
                                    bool read, bool write,
                                    uint32_t hw_index) { return false; }

    // From lldb_private::RegisterContext
    virtual uint32_t
    NumSupportedHardwareWatchpoints () { return 0; }

private:
    ProcessMonitor &
    GetMonitor();
};

#endif
