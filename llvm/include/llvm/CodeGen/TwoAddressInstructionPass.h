//===-- llvm/CodeGen/TwoAddressInstructionPass.h - Two-Address instruction pass  -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Two-Address instruction rewriter pass. In
// some architectures instructions have a combined source/destination
// operand. In those cases the instruction cannot have three operands
// as the destination is implicit (for example ADD %EAX, %EBX on the
// IA-32). After code generation this restrictions are not handled and
// instructions may have three operands. This pass remedies this and
// reduces all two-address instructions to two operands.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_TWOADDRESSINSTRUCTIONPASS_H
#define LLVM_CODEGEN_TWOADDRESSINSTRUCTIONPASS_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include <iostream>
#include <map>

namespace llvm {

    class LiveVariables;
    class MRegisterInfo;

    class TwoAddressInstructionPass : public MachineFunctionPass
    {
    private:
        MachineFunction* mf_;
        const TargetMachine* tm_;
        const MRegisterInfo* mri_;
        LiveVariables* lv_;

    public:
        virtual void getAnalysisUsage(AnalysisUsage &AU) const;

    private:
        /// runOnMachineFunction - pass entry point
        bool runOnMachineFunction(MachineFunction&);
    };

} // End llvm namespace

#endif
