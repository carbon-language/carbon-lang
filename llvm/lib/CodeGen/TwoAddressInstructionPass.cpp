//===-- TwoAddressInstructionPass.cpp - Two-Address instruction pass ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TwoAddress instruction pass which is used
// by most register allocators. Two-Address instructions are rewritten
// from:
//
//     A = B op C
//
// to:
//
//     A = B
//     A = A op C
//
// Note that if a register allocator chooses to use this pass, that it has to
// be capable of handling the non-SSA nature of these rewritten virtual 
// registers.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "twoaddrinstr"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
    Statistic<> numTwoAddressInstrs("twoaddressinstruction",
                                    "Number of two-address instructions");
    Statistic<> numInstrsAdded("twoaddressinstruction",
                               "Number of instructions added");

    struct TwoAddressInstructionPass : public MachineFunctionPass
    {
        virtual void getAnalysisUsage(AnalysisUsage &AU) const;

        /// runOnMachineFunction - pass entry point
        bool runOnMachineFunction(MachineFunction&);
    };

    RegisterPass<TwoAddressInstructionPass> X(
        "twoaddressinstruction", "Two-Address instruction pass");
};

const PassInfo *llvm::TwoAddressInstructionPassID = X.getPassInfo();

void TwoAddressInstructionPass::getAnalysisUsage(AnalysisUsage &AU) const
{
    AU.addPreserved<LiveVariables>();
    AU.addRequired<LiveVariables>();
    AU.addPreservedID(PHIEliminationID);
    AU.addRequiredID(PHIEliminationID);
    MachineFunctionPass::getAnalysisUsage(AU);
}

/// runOnMachineFunction - Reduce two-address instructions to two
/// operands.
///
bool TwoAddressInstructionPass::runOnMachineFunction(MachineFunction &MF) {
    DEBUG(std::cerr << "Machine Function\n");
    const TargetMachine &TM = MF.getTarget();
    const MRegisterInfo &MRI = *TM.getRegisterInfo();
    const TargetInstrInfo &TII = TM.getInstrInfo();
    LiveVariables &LV = getAnalysis<LiveVariables>();

    bool MadeChange = false;

    for (MachineFunction::iterator mbbi = MF.begin(), mbbe = MF.end();
         mbbi != mbbe; ++mbbi) {
        for (MachineBasicBlock::iterator mii = mbbi->begin();
             mii != mbbi->end(); ++mii) {
            MachineInstr* mi = *mii;
            unsigned opcode = mi->getOpcode();

            // ignore if it is not a two-address instruction
            if (!TII.isTwoAddrInstr(opcode))
                continue;

            ++numTwoAddressInstrs;

            DEBUG(std::cerr << "\tinstruction: "; mi->print(std::cerr, TM));

            assert(mi->getOperand(1).isRegister() &&
                   mi->getOperand(1).getAllocatedRegNum() &&
                   mi->getOperand(1).isUse() &&
                   "two address instruction invalid");

            // we have nothing to do if the two operands are the same
            if (mi->getOperand(0).getAllocatedRegNum() ==
                mi->getOperand(1).getAllocatedRegNum())
                continue;

            MadeChange = true;

            // rewrite:
            //     a = b op c
            // to:
            //     a = b
            //     a = a op c
            unsigned regA = mi->getOperand(0).getAllocatedRegNum();
            unsigned regB = mi->getOperand(1).getAllocatedRegNum();

            assert(MRegisterInfo::isVirtualRegister(regA) &&
                   MRegisterInfo::isVirtualRegister(regB) &&
                   "cannot update physical register live information");

            // first make sure we do not have a use of a in the
            // instruction (a = b + a for example) because our
            // transformation will not work. This should never occur
            // because we are in SSA form.
            for (unsigned i = 1; i != mi->getNumOperands(); ++i)
                assert(!mi->getOperand(i).isRegister() ||
                       mi->getOperand(i).getAllocatedRegNum() != (int)regA);

            const TargetRegisterClass* rc =MF.getSSARegMap()->getRegClass(regA);
            unsigned Added = MRI.copyRegToReg(*mbbi, mii, regA, regB, rc);
            numInstrsAdded += Added;

            MachineInstr* prevMi = *(mii - 1);
            DEBUG(std::cerr << "\t\tadded instruction: ";
                  prevMi->print(std::cerr, TM));

            // update live variables for regA
            assert(Added == 1 && "Cannot handle multi-instruction copies yet!");
            LiveVariables::VarInfo& varInfo = LV.getVarInfo(regA);
            varInfo.DefInst = prevMi;

            // update live variables for regB
            if (LV.removeVirtualRegisterKilled(regB, &*mbbi, mi))
                LV.addVirtualRegisterKilled(regB, &*mbbi, prevMi);

            if (LV.removeVirtualRegisterDead(regB, &*mbbi, mi))
                LV.addVirtualRegisterDead(regB, &*mbbi, prevMi);

            // replace all occurences of regB with regA
            for (unsigned i = 1; i < mi->getNumOperands(); ++i) {
                if (mi->getOperand(i).isRegister() &&
                    mi->getOperand(i).getReg() == regB)
                    mi->SetMachineOperandReg(i, regA);
            }
            DEBUG(std::cerr << "\t\tmodified original to: ";
                  mi->print(std::cerr, TM));
            assert(mi->getOperand(0).getAllocatedRegNum() ==
                   mi->getOperand(1).getAllocatedRegNum());
        }
    }

    return MadeChange;
}
