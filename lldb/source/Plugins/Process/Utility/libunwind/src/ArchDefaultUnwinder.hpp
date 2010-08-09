/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- ArchDefaultUnwinder.hpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Unwind a stack frame using nothing but the default conventions on
// this architecture.

#ifndef __ARCH_DEFAULT_UNWINDER_HPP
#define __ARCH_DEFAULT_UNWINDER_HPP

#if defined (SUPPORT_REMOTE_UNWINDING)

#include "AddressSpace.hpp"
#include "Registers.hpp"
#include "RemoteRegisterMap.hpp"
#include "RemoteProcInfo.hpp"

namespace lldb_private
{

// As a last ditch attempt to unwind a stack frame, unwind by the
// architecture's typical conventions.  We try compact unwind, eh frame CFI,
// and then assembly profiling if we have function bounds -- but if we're
// looking at an address with no function bounds or unwind info, make a best
// guess at how to get out.

// In practice, this is usually hit when we try to step out of start() in a 
// stripped application binary, we've jumped to 0x0, or we're in jitted code
// in the heap.

template <typename A, typename R>
int stepByArchitectureDefault_x86 (A& addressSpace, R& registers, 
                                   uint64_t pc, int wordsize) {
    R newRegisters(registers);
    RemoteRegisterMap *rmap = addressSpace.getRemoteProcInfo()->getRegisterMap();
    int frame_reg = rmap->unwind_regno_for_frame_pointer();
    int stack_reg = rmap->unwind_regno_for_stack_pointer();
    int err;

    /* If the pc is 0x0 either we call'ed 0 (went thorugh a null function 
       pointer) or this is a thread in the middle of being created that has
       no stack at all.
       For the call-0x0 case, we know how to unwind that - the pc is at
       the stack pointer.  

       Otherwise follow the usual convention of trusting that RBP/EBP has the
       start of the stack frame and we can find the caller's pc based on
       that.  */

    uint64_t newpc, newframeptr;
    newpc = 0;
    newframeptr = -1;
    if (pc == 0) {
        uint64_t oldsp = registers.getRegister(stack_reg);
        err = 0;
        if (oldsp != 0) {
            newpc = addressSpace.getP(registers.getRegister(stack_reg), err);
            if (err != 0)
                return UNW_EUNSPEC;
            newRegisters.setIP (newpc);
            newRegisters.setRegister (stack_reg, registers.getRegister(stack_reg) +
                                                                        wordsize);
        }
    }
    else {
        newpc = addressSpace.getP(registers.getRegister(frame_reg) + 
                                           wordsize, err);
        if (err != 0)
            return UNW_EUNSPEC;

        newRegisters.setIP (newpc);
        newframeptr = addressSpace.getP(registers.getRegister(frame_reg), 
                                        err);
        if (err != 0)
            return UNW_EUNSPEC;

        newRegisters.setRegister (frame_reg, newframeptr);
        newRegisters.setRegister (stack_reg, registers.getRegister(frame_reg) + 
                                                               (wordsize * 2));
    }
    registers = newRegisters;
    if (newpc == 0 || newframeptr == 0)
        return UNW_STEP_END;
    return UNW_STEP_SUCCESS;
}

template <typename A>
int stepByArchitectureDefault (A& addressSpace, Registers_x86_64 &registers, 
                               uint64_t pc) {
    return stepByArchitectureDefault_x86 (addressSpace, registers, pc, 8);
}

template <typename A>
int stepByArchitectureDefault (A& addressSpace, Registers_x86& registers, 
                               uint64_t pc) {
    return stepByArchitectureDefault_x86 (addressSpace, registers, pc, 4);
}

}; // namespace lldb_private

#endif // SUPPORT_REMOTE_UNWINDING
#endif // __ARCH_DEFAULT_UNWINDER_HPP
