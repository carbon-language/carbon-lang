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
using namespace llvm;

STATISTIC(NumTwoAddressInstrs, "Number of two-address instructions");
STATISTIC(NumCommuted        , "Number of instructions commuted to coalesce");
STATISTIC(NumConvertedTo3Addr, "Number of instructions promoted to 3-address");

namespace {
  struct VISIBILITY_HIDDEN TwoAddressInstructionPass
   : public MachineFunctionPass {
    static char ID; // Pass identification, replacement for typeid
    TwoAddressInstructionPass() : MachineFunctionPass((intptr_t)&ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;

    /// runOnMachineFunction - pass entry point
    bool runOnMachineFunction(MachineFunction&);
  };

  char TwoAddressInstructionPass::ID = 0;
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
  DOUT << "Machine Function\n";
  const TargetMachine &TM = MF.getTarget();
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  const MRegisterInfo &MRI = *TM.getRegisterInfo();
  LiveVariables &LV = getAnalysis<LiveVariables>();

  bool MadeChange = false;

  DOUT << "********** REWRITING TWO-ADDR INSTRS **********\n";
  DOUT << "********** Function: " << MF.getFunction()->getName() << '\n';

  for (MachineFunction::iterator mbbi = MF.begin(), mbbe = MF.end();
       mbbi != mbbe; ++mbbi) {
    for (MachineBasicBlock::iterator mi = mbbi->begin(), me = mbbi->end();
         mi != me; ++mi) {
      const TargetInstrDescriptor *TID = mi->getInstrDescriptor();

      bool FirstTied = true;
      for (unsigned si = 1, e = TID->numOperands; si < e; ++si) {
        int ti = TID->getOperandConstraint(si, TOI::TIED_TO);
        if (ti == -1)
          continue;

        if (FirstTied) {
          ++NumTwoAddressInstrs;
          DOUT << '\t'; DEBUG(mi->print(*cerr.stream(), &TM));
        }
        FirstTied = false;

        assert(mi->getOperand(si).isRegister() && mi->getOperand(si).getReg() &&
               mi->getOperand(si).isUse() && "two address instruction invalid");

        // if the two operands are the same we just remove the use
        // and mark the def as def&use, otherwise we have to insert a copy.
        if (mi->getOperand(ti).getReg() != mi->getOperand(si).getReg()) {
          // rewrite:
          //     a = b op c
          // to:
          //     a = b
          //     a = a op c
          unsigned regA = mi->getOperand(ti).getReg();
          unsigned regB = mi->getOperand(si).getReg();

          assert(MRegisterInfo::isVirtualRegister(regA) &&
                 MRegisterInfo::isVirtualRegister(regB) &&
                 "cannot update physical register live information");

#ifndef NDEBUG
          // First, verify that we don't have a use of a in the instruction (a =
          // b + a for example) because our transformation will not work. This
          // should never occur because we are in SSA form.
          for (unsigned i = 0; i != mi->getNumOperands(); ++i)
            assert((int)i == ti ||
                   !mi->getOperand(i).isRegister() ||
                   mi->getOperand(i).getReg() != regA);
#endif

          // If this instruction is not the killing user of B, see if we can
          // rearrange the code to make it so.  Making it the killing user will
          // allow us to coalesce A and B together, eliminating the copy we are
          // about to insert.
          if (!LV.KillsRegister(mi, regB)) {
            // If this instruction is commutative, check to see if C dies.  If
            // so, swap the B and C operands.  This makes the live ranges of A
            // and C joinable.
            // FIXME: This code also works for A := B op C instructions.
            if ((TID->Flags & M_COMMUTABLE) && mi->getNumOperands() == 3) {
              assert(mi->getOperand(3-si).isRegister() &&
                     "Not a proper commutative instruction!");
              unsigned regC = mi->getOperand(3-si).getReg();
              if (LV.KillsRegister(mi, regC)) {
                DOUT << "2addr: COMMUTING  : " << *mi;
                MachineInstr *NewMI = TII.commuteInstruction(mi);
                if (NewMI == 0) {
                  DOUT << "2addr: COMMUTING FAILED!\n";
                } else {
                  DOUT << "2addr: COMMUTED TO: " << *NewMI;
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
            if (TID->Flags & M_CONVERTIBLE_TO_3_ADDR)
              // FIXME: This assumes there are no more operands which are tied
              // to another register.
#ifndef NDEBUG
              for (unsigned i = si+1, e = TID->numOperands; i < e; ++i)
                assert(TID->getOperandConstraint(i, TOI::TIED_TO) == -1);
#endif

              if (MachineInstr *New = TII.convertToThreeAddress(mbbi, mi, LV)) {
                DOUT << "2addr: CONVERTING 2-ADDR: " << *mi;
                DOUT << "2addr:         TO 3-ADDR: " << *New;
                mbbi->erase(mi);                 // Nuke the old inst.
                mi = New;
                ++NumConvertedTo3Addr;
                // Done with this instruction.
                break;
              }
          }

        InstructionRearranged:
          const TargetRegisterClass* rc = MF.getSSARegMap()->getRegClass(regA);
          MRI.copyRegToReg(*mbbi, mi, regA, regB, rc);

          MachineBasicBlock::iterator prevMi = prior(mi);
          DOUT << "\t\tprepend:\t"; DEBUG(prevMi->print(*cerr.stream(), &TM));

          // Update live variables for regA
          LiveVariables::VarInfo& varInfo = LV.getVarInfo(regA);
          varInfo.DefInst = prevMi;

          if (LV.removeVirtualRegisterKilled(regB, mbbi, mi))
            LV.addVirtualRegisterKilled(regB, prevMi);

          if (LV.removeVirtualRegisterDead(regB, mbbi, mi))
            LV.addVirtualRegisterDead(regB, prevMi);

          // replace all occurences of regB with regA
          for (unsigned i = 0, e = mi->getNumOperands(); i != e; ++i) {
            if (mi->getOperand(i).isRegister() &&
                mi->getOperand(i).getReg() == regB)
              mi->getOperand(i).setReg(regA);
          }
        }

        assert(mi->getOperand(ti).isDef() && mi->getOperand(si).isUse());
        mi->getOperand(ti).setReg(mi->getOperand(si).getReg());
        MadeChange = true;

        DOUT << "\t\trewrite to:\t"; DEBUG(mi->print(*cerr.stream(), &TM));
      }
    }
  }

  return MadeChange;
}
