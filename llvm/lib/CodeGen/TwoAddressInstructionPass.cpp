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
//     A op= C
//
// Note that if a register allocator chooses to use this pass, that it
// has to be capable of handling the non-SSA nature of these rewritten
// virtual registers.
//
// It is also worth noting that the duplicate operand of the two
// address instruction is removed.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "twoaddrinstr"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <iostream>
using namespace llvm;

namespace {
  static Statistic<> NumTwoAddressInstrs("twoaddressinstruction",
                                  "Number of two-address instructions");
  static Statistic<> NumCommuted("twoaddressinstruction",
                          "Number of instructions commuted to coalesce");
  static Statistic<> NumConvertedTo3Addr("twoaddressinstruction",
                                "Number of instructions promoted to 3-address");

  struct VISIBILITY_HIDDEN TwoAddressInstructionPass
   : public MachineFunctionPass {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;

    /// runOnMachineFunction - pass entry point
    bool runOnMachineFunction(MachineFunction&);
  };

  RegisterPass<TwoAddressInstructionPass>
  X("twoaddressinstruction", "Two-Address instruction pass");
}

const PassInfo *llvm::TwoAddressInstructionPassID = X.getPassInfo();

void TwoAddressInstructionPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LiveVariables>();
  AU.addPreserved<LiveVariables>();
  AU.addPreservedID(PHIEliminationID);
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// runOnMachineFunction - Reduce two-address instructions to two
/// operands.
///
bool TwoAddressInstructionPass::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(std::cerr << "Machine Function\n");
  const TargetMachine &TM = MF.getTarget();
  const MRegisterInfo &MRI = *TM.getRegisterInfo();
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  LiveVariables &LV = getAnalysis<LiveVariables>();

  bool MadeChange = false;

  DEBUG(std::cerr << "********** REWRITING TWO-ADDR INSTRS **********\n");
  DEBUG(std::cerr << "********** Function: "
                  << MF.getFunction()->getName() << '\n');

  for (MachineFunction::iterator mbbi = MF.begin(), mbbe = MF.end();
       mbbi != mbbe; ++mbbi) {
    for (MachineBasicBlock::iterator mi = mbbi->begin(), me = mbbi->end();
         mi != me; ++mi) {
      unsigned opcode = mi->getOpcode();

      // ignore if it is not a two-address instruction
      if (!TII.isTwoAddrInstr(opcode))
        continue;

      ++NumTwoAddressInstrs;
      DEBUG(std::cerr << '\t'; mi->print(std::cerr, &TM));
      assert(mi->getOperand(1).isRegister() && mi->getOperand(1).getReg() &&
             mi->getOperand(1).isUse() && "two address instruction invalid");

      // if the two operands are the same we just remove the use
      // and mark the def as def&use, otherwise we have to insert a copy.
      if (mi->getOperand(0).getReg() != mi->getOperand(1).getReg()) {
        // rewrite:
        //     a = b op c
        // to:
        //     a = b
        //     a = a op c
        unsigned regA = mi->getOperand(0).getReg();
        unsigned regB = mi->getOperand(1).getReg();

        assert(MRegisterInfo::isVirtualRegister(regA) &&
               MRegisterInfo::isVirtualRegister(regB) &&
               "cannot update physical register live information");

#ifndef NDEBUG
        // First, verify that we do not have a use of a in the instruction (a =
        // b + a for example) because our transformation will not work. This
        // should never occur because we are in SSA form.
        for (unsigned i = 1; i != mi->getNumOperands(); ++i)
          assert(!mi->getOperand(i).isRegister() ||
                 mi->getOperand(i).getReg() != regA);
#endif

        // If this instruction is not the killing user of B, see if we can
        // rearrange the code to make it so.  Making it the killing user will
        // allow us to coalesce A and B together, eliminating the copy we are
        // about to insert.
        if (!LV.KillsRegister(mi, regB)) {
          const TargetInstrDescriptor &TID = TII.get(opcode);

          // If this instruction is commutative, check to see if C dies.  If so,
          // swap the B and C operands.  This makes the live ranges of A and C
          // joinable.
          if (TID.Flags & M_COMMUTABLE) {
            assert(mi->getOperand(2).isRegister() &&
                   "Not a proper commutative instruction!");
            unsigned regC = mi->getOperand(2).getReg();
            if (LV.KillsRegister(mi, regC)) {
              DEBUG(std::cerr << "2addr: COMMUTING  : " << *mi);
              MachineInstr *NewMI = TII.commuteInstruction(mi);
              if (NewMI == 0) {
                DEBUG(std::cerr << "2addr: COMMUTING FAILED!\n");
              } else {
                DEBUG(std::cerr << "2addr: COMMUTED TO: " << *NewMI);
                // If the instruction changed to commute it, update livevar.
                if (NewMI != mi) {
                  LV.instructionChanged(mi, NewMI);  // Update live variables
                  mbbi->insert(mi, NewMI);           // Insert the new inst
                  mbbi->erase(mi);                   // Nuke the old inst.
                  mi = NewMI;
                }

                ++NumCommuted;
                regB = regC;
                goto InstructionRearranged;
              }
            }
          }
          // If this instruction is potentially convertible to a true
          // three-address instruction,
          if (TID.Flags & M_CONVERTIBLE_TO_3_ADDR)
            if (MachineInstr *New = TII.convertToThreeAddress(mi)) {
              DEBUG(std::cerr << "2addr: CONVERTING 2-ADDR: " << *mi);
              DEBUG(std::cerr << "2addr:         TO 3-ADDR: " << *New);
              LV.instructionChanged(mi, New);  // Update live variables
              mbbi->insert(mi, New);           // Insert the new inst
              mbbi->erase(mi);                 // Nuke the old inst.
              mi = New;
              ++NumConvertedTo3Addr;
              assert(!TII.isTwoAddrInstr(New->getOpcode()) &&
                     "convertToThreeAddress returned a 2-addr instruction??");
              // Done with this instruction.
              continue;
            }
        }
      InstructionRearranged:
        const TargetRegisterClass* rc = MF.getSSARegMap()->getRegClass(regA);
        MRI.copyRegToReg(*mbbi, mi, regA, regB, rc);

        MachineBasicBlock::iterator prevMi = prior(mi);
        DEBUG(std::cerr << "\t\tprepend:\t"; prevMi->print(std::cerr, &TM));

        // Update live variables for regA
        LiveVariables::VarInfo& varInfo = LV.getVarInfo(regA);
        varInfo.DefInst = prevMi;

        // update live variables for regB
        if (LV.removeVirtualRegisterKilled(regB, mbbi, mi))
          LV.addVirtualRegisterKilled(regB, prevMi);

        if (LV.removeVirtualRegisterDead(regB, mbbi, mi))
          LV.addVirtualRegisterDead(regB, prevMi);

        // replace all occurences of regB with regA
        for (unsigned i = 1, e = mi->getNumOperands(); i != e; ++i) {
          if (mi->getOperand(i).isRegister() &&
              mi->getOperand(i).getReg() == regB)
            mi->getOperand(i).setReg(regA);
        }
      }

      assert(mi->getOperand(0).isDef());
      mi->getOperand(0).setUse();
      mi->RemoveOperand(1);
      MadeChange = true;

      DEBUG(std::cerr << "\t\trewrite to:\t"; mi->print(std::cerr, &TM));
    }
  }

  return MadeChange;
}
