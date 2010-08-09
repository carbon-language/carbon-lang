/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- RemoteDebuggerDummyUnwinder.hpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Code to unwind past a debugger's dummy frame inserted when it does an
// inferior function call.
// In this case we'll need to get the saved register context from the debugger -
// it may be in the debugger's local memory or it may be saved in a nonstandard
// location in the inferior process' memory.

#ifndef __REMOTE_DEBUGGER_DUMMY_UNWINDER_HPP__
#define __REMOTE_DEBUGGER_DUMMY_UNWINDER_HPP__

#if defined (SUPPORT_REMOTE_UNWINDING)

#include "libunwind.h"
#include "Registers.hpp"
#include "AddressSpace.hpp"
#include "RemoteRegisterMap.hpp"
#include "RemoteProcInfo.hpp"

namespace lldb_private
{

template <typename A>
int stepOutOfDebuggerDummyFrame (A& addressSpace, Registers_x86_64& registers,
                                 RemoteProcInfo *procinfo, uint64_t ip, 
                                uint64_t sp, void* arg) 
{
    Registers_x86_64 newRegisters(registers);
    RemoteRegisterMap *rmap = addressSpace.getRemoteProcInfo()->getRegisterMap();
    unw_word_t regv;
    for (int i = UNW_X86_64_RAX; i <= UNW_X86_64_R15; i++) {
        int driver_regnum;
        if (!rmap->unwind_regno_to_caller_regno (i, driver_regnum))
            continue;
        if (addressSpace.accessors()->access_reg_inf_func_call (procinfo->wrap(), ip, sp, driver_regnum, &regv, 0, arg))
            newRegisters.setRegister(i, regv);
    }
    if (!addressSpace.accessors()->access_reg_inf_func_call (procinfo->wrap(), ip, sp, rmap->caller_regno_for_ip(), &regv, 0, arg))
        return UNW_EUNSPEC;
    newRegisters.setIP (regv);
    registers = newRegisters;
    return UNW_STEP_SUCCESS;
}

template <typename A>
int stepOutOfDebuggerDummyFrame (A& addressSpace, Registers_x86& registers,
                                 RemoteProcInfo *procinfo, uint64_t ip, 
                                uint64_t sp, void* arg)
{
    Registers_x86 newRegisters(registers);
    RemoteRegisterMap *rmap = addressSpace.getRemoteProcInfo()->getRegisterMap();
    unw_word_t regv;
    for (int i = UNW_X86_EAX; i <= UNW_X86_EDI; i++) {
        int driver_regnum;
        if (!rmap->unwind_regno_to_caller_regno (i, driver_regnum))
            continue;
        if (addressSpace.accessors()->access_reg_inf_func_call (procinfo->wrap(), ip, sp, driver_regnum, &regv, 0, arg))
            newRegisters.setRegister(i, regv);
    }
    if (!addressSpace.accessors()->access_reg_inf_func_call (procinfo->wrap(), ip, sp, rmap->caller_regno_for_ip(), &regv, 0, arg))
        return UNW_EUNSPEC;
    newRegisters.setIP (regv);
    registers = newRegisters;
    return UNW_STEP_SUCCESS;
}

}; // namespace lldb_private

#endif // SUPPORT_REMOTE_UNWINDING

#endif // __REMOTE_DEBUGGER_DUMMY_UNWINDER_HPP__

