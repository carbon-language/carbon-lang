//===-- TwoAddressInstructionPass.cpp - Two-Address instruction pass ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveInterval analysis pass which is used
// by the Linear Scan Register allocator. This pass linearizes the
// basic blocks of the function in DFS order and uses the
// LiveVariables pass to conservatively compute live intervals for
// each virtual and physical register.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "twoaddrinstr"
#include "llvm/CodeGen/TwoAddressInstructionPass.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegInfo.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include <iostream>

using namespace llvm;

namespace {
    RegisterAnalysis<TwoAddressInstructionPass> X(
        "twoaddressinstruction", "Two-Address instruction pass");

    Statistic<> numTwoAddressInstrs("twoaddressinstruction",
                                    "Number of two-address instructions");
    Statistic<> numInstrsAdded("twoaddressinstruction",
                               "Number of instructions added");
};

void TwoAddressInstructionPass::getAnalysisUsage(AnalysisUsage &AU) const
{
    AU.addPreserved<LiveVariables>();
    AU.addRequired<LiveVariables>();
    AU.addPreservedID(PHIEliminationID);
    AU.addRequiredID(PHIEliminationID);
    MachineFunctionPass::getAnalysisUsage(AU);
}

/// runOnMachineFunction - Reduce two-address instructions to two
/// operands
///
bool TwoAddressInstructionPass::runOnMachineFunction(MachineFunction &fn) {
    DEBUG(std::cerr << "Machine Function\n");
    mf_ = &fn;
    tm_ = &fn.getTarget();
    mri_ = tm_->getRegisterInfo();
    lv_ = &getAnalysis<LiveVariables>();

    const TargetInstrInfo& tii = tm_->getInstrInfo();

    for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
         mbbi != mbbe; ++mbbi) {
        for (MachineBasicBlock::iterator mii = mbbi->begin();
             mii != mbbi->end(); ++mii) {
            MachineInstr* mi = *mii;

            unsigned opcode = mi->getOpcode();
            // ignore if it is not a two-address instruction
            if (!tii.isTwoAddrInstr(opcode))
                continue;

            ++numTwoAddressInstrs;

            DEBUG(std::cerr << "\tinstruction: "; mi->print(std::cerr, *tm_));

            // we have nothing to do if the two operands are the same
            if (mi->getOperand(0).getAllocatedRegNum() ==
                mi->getOperand(1).getAllocatedRegNum())
                continue;

            assert(mi->getOperand(1).isRegister() &&
                   mi->getOperand(1).getAllocatedRegNum() &&
                   mi->getOperand(1).isUse() &&
                   "two address instruction invalid");

            // rewrite:
            //     a = b op c
            // to:
            //     a = b
            //     a = a op c
            unsigned regA = mi->getOperand(0).getAllocatedRegNum();
            unsigned regB = mi->getOperand(1).getAllocatedRegNum();
            bool regAisPhysical = regA < MRegisterInfo::FirstVirtualRegister;
            bool regBisPhysical = regB < MRegisterInfo::FirstVirtualRegister;

            const TargetRegisterClass* rc = regAisPhysical ?
                mri_->getRegClass(regA) :
                mf_->getSSARegMap()->getRegClass(regA);

            numInstrsAdded += mri_->copyRegToReg(*mbbi, mii, regA, regB, rc);

            MachineInstr* prevMi = *(mii - 1);
            DEBUG(std::cerr << "\t\tadded instruction: ";
                  prevMi->print(std::cerr, *tm_));

            // update live variables for regA
            if (regAisPhysical) {
                lv_->HandlePhysRegDef(regA, prevMi);
            }
            else {
                LiveVariables::VarInfo& varInfo = lv_->getVarInfo(regA);
                varInfo.DefInst = prevMi;
            }

            // update live variables for regB
            if (regBisPhysical) {
                lv_->HandlePhysRegUse(regB, prevMi);
            }
            else {
                if (lv_->removeVirtualRegisterKilled(regB, &*mbbi, mi))
                    lv_->addVirtualRegisterKilled(regB, &*mbbi, prevMi);

                if (lv_->removeVirtualRegisterDead(regB, &*mbbi, mi))
                    lv_->addVirtualRegisterDead(regB, &*mbbi, prevMi);
            }

            // replace all occurences of regB with regA
            for (unsigned i = 1; i < mi->getNumOperands(); ++i) {
                if (mi->getOperand(i).isRegister() &&
                    mi->getOperand(i).getReg() == regB)
                    mi->SetMachineOperandReg(i, regA);
            }
            DEBUG(std::cerr << "\t\tmodified original to: ";
                  mi->print(std::cerr, *tm_));
            assert(mi->getOperand(0).getAllocatedRegNum() ==
                   mi->getOperand(1).getAllocatedRegNum());
        }
    }

    return numInstrsAdded != 0;
}
