//===-- TwoAddressInstructionPass.cpp - Two-Address instruction pass ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

STATISTIC(NumTwoAddressInstrs, "Number of two-address instructions");
STATISTIC(NumCommuted        , "Number of instructions commuted to coalesce");
STATISTIC(NumConvertedTo3Addr, "Number of instructions promoted to 3-address");
STATISTIC(Num3AddrSunk,        "Number of 3-address instructions sunk");

namespace {
  class VISIBILITY_HIDDEN TwoAddressInstructionPass
    : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    MachineRegisterInfo *MRI;
    LiveVariables *LV;

    bool Sink3AddrInstruction(MachineBasicBlock *MBB, MachineInstr *MI,
                              unsigned Reg,
                              MachineBasicBlock::iterator OldPos);
  public:
    static char ID; // Pass identification, replacement for typeid
    TwoAddressInstructionPass() : MachineFunctionPass((intptr_t)&ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LiveVariables>();
      AU.addPreserved<LiveVariables>();
      AU.addPreservedID(MachineLoopInfoID);
      AU.addPreservedID(MachineDominatorsID);
      AU.addPreservedID(PHIEliminationID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    /// runOnMachineFunction - Pass entry point.
    bool runOnMachineFunction(MachineFunction&);
  };
}

char TwoAddressInstructionPass::ID = 0;
static RegisterPass<TwoAddressInstructionPass>
X("twoaddressinstruction", "Two-Address instruction pass");

const PassInfo *const llvm::TwoAddressInstructionPassID = &X;

/// Sink3AddrInstruction - A two-address instruction has been converted to a
/// three-address instruction to avoid clobbering a register. Try to sink it
/// past the instruction that would kill the above mentioned register to reduce
/// register pressure.
/// 
bool TwoAddressInstructionPass::Sink3AddrInstruction(MachineBasicBlock *MBB,
                                           MachineInstr *MI, unsigned SavedReg,
                                           MachineBasicBlock::iterator OldPos) {
  // Check if it's safe to move this instruction.
  bool SeenStore = true; // Be conservative.
  if (!MI->isSafeToMove(TII, SeenStore))
    return false;

  unsigned DefReg = 0;
  SmallSet<unsigned, 4> UseRegs;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isRegister())
      continue;
    unsigned MOReg = MO.getReg();
    if (!MOReg)
      continue;
    if (MO.isUse() && MOReg != SavedReg)
      UseRegs.insert(MO.getReg());
    if (!MO.isDef())
      continue;
    if (MO.isImplicit())
      // Don't try to move it if it implicitly defines a register.
      return false;
    if (DefReg)
      // For now, don't move any instructions that define multiple registers.
      return false;
    DefReg = MO.getReg();
  }

  // Find the instruction that kills SavedReg.
  MachineInstr *KillMI = NULL;

  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(SavedReg),
         UE = MRI->use_end(); UI != UE; ++UI) {
    MachineOperand &UseMO = UI.getOperand();
    if (!UseMO.isKill())
      continue;
    KillMI = UseMO.getParent();
    break;
  }

  if (!KillMI || KillMI->getParent() != MBB)
    return false;

  // If any of the definitions are used by another instruction between the
  // position and the kill use, then it's not safe to sink it.
  // 
  // FIXME: This can be sped up if there is an easy way to query whether an
  // instruction if before or after another instruction. Then we can use
  // MachineRegisterInfo def / use instead.
  MachineOperand *KillMO = NULL;
  MachineBasicBlock::iterator KillPos = KillMI;
  ++KillPos;

  for (MachineBasicBlock::iterator I = next(OldPos); I != KillPos; ++I) {
    MachineInstr *OtherMI = I;

    for (unsigned i = 0, e = OtherMI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = OtherMI->getOperand(i);
      if (!MO.isRegister())
        continue;
      unsigned MOReg = MO.getReg();
      if (!MOReg)
        continue;
      if (DefReg == MOReg)
        return false;

      if (MO.isKill()) {
        if (OtherMI == KillMI && MOReg == SavedReg)
          // Save the operand that kills the register. We want unset the kill
          // marker is we can sink MI past it.
          KillMO = &MO;
        else if (UseRegs.count(MOReg))
          // One of the uses is killed before the destination.
          return false;
      }
    }
  }

  // Update kill and LV information.
  KillMO->setIsKill(false);
  KillMO = MI->findRegisterUseOperand(SavedReg, false, TRI);
  KillMO->setIsKill(true);
  LiveVariables::VarInfo& VarInfo = LV->getVarInfo(SavedReg);
  VarInfo.removeKill(KillMI);
  VarInfo.Kills.push_back(MI);

  // Move instruction to its destination.
  MBB->remove(MI);
  MBB->insert(KillPos, MI);

  ++Num3AddrSunk;
  return true;
}

/// runOnMachineFunction - Reduce two-address instructions to two operands.
///
bool TwoAddressInstructionPass::runOnMachineFunction(MachineFunction &MF) {
  DOUT << "Machine Function\n";
  const TargetMachine &TM = MF.getTarget();
  MRI = &MF.getRegInfo();
  TII = TM.getInstrInfo();
  TRI = TM.getRegisterInfo();
  LV = &getAnalysis<LiveVariables>();

  bool MadeChange = false;

  DOUT << "********** REWRITING TWO-ADDR INSTRS **********\n";
  DOUT << "********** Function: " << MF.getFunction()->getName() << '\n';

  SmallPtrSet<MachineInstr*, 8> ReMattedInstrs;

  for (MachineFunction::iterator mbbi = MF.begin(), mbbe = MF.end();
       mbbi != mbbe; ++mbbi) {
    for (MachineBasicBlock::iterator mi = mbbi->begin(), me = mbbi->end();
         mi != me; ) {
      MachineBasicBlock::iterator nmi = next(mi);
      const TargetInstrDesc &TID = mi->getDesc();
      bool FirstTied = true;

      for (unsigned si = 1, e = TID.getNumOperands(); si < e; ++si) {
        int ti = TID.getOperandConstraint(si, TOI::TIED_TO);
        if (ti == -1)
          continue;

        if (FirstTied) {
          ++NumTwoAddressInstrs;
          DOUT << '\t'; DEBUG(mi->print(*cerr.stream(), &TM));
        }

        FirstTied = false;

        assert(mi->getOperand(si).isRegister() && mi->getOperand(si).getReg() &&
               mi->getOperand(si).isUse() && "two address instruction invalid");

        // If the two operands are the same we just remove the use
        // and mark the def as def&use, otherwise we have to insert a copy.
        if (mi->getOperand(ti).getReg() != mi->getOperand(si).getReg()) {
          // Rewrite:
          //     a = b op c
          // to:
          //     a = b
          //     a = a op c
          unsigned regA = mi->getOperand(ti).getReg();
          unsigned regB = mi->getOperand(si).getReg();

          assert(TargetRegisterInfo::isVirtualRegister(regA) &&
                 TargetRegisterInfo::isVirtualRegister(regB) &&
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
          if (!mi->killsRegister(regB)) {
            // If this instruction is commutative, check to see if C dies.  If
            // so, swap the B and C operands.  This makes the live ranges of A
            // and C joinable.
            // FIXME: This code also works for A := B op C instructions.
            if (TID.isCommutable() && mi->getNumOperands() >= 3) {
              assert(mi->getOperand(3-si).isRegister() &&
                     "Not a proper commutative instruction!");
              unsigned regC = mi->getOperand(3-si).getReg();

              if (mi->killsRegister(regC)) {
                DOUT << "2addr: COMMUTING  : " << *mi;
                MachineInstr *NewMI = TII->commuteInstruction(mi);

                if (NewMI == 0) {
                  DOUT << "2addr: COMMUTING FAILED!\n";
                } else {
                  DOUT << "2addr: COMMUTED TO: " << *NewMI;
                  // If the instruction changed to commute it, update livevar.
                  if (NewMI != mi) {
                    LV->instructionChanged(mi, NewMI); // Update live variables
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
            if (TID.isConvertibleTo3Addr()) {
              // FIXME: This assumes there are no more operands which are tied
              // to another register.
#ifndef NDEBUG
              for (unsigned i = si + 1, e = TID.getNumOperands(); i < e; ++i)
                assert(TID.getOperandConstraint(i, TOI::TIED_TO) == -1);
#endif

              if (MachineInstr *New=TII->convertToThreeAddress(mbbi, mi, *LV)) {
                DOUT << "2addr: CONVERTING 2-ADDR: " << *mi;
                DOUT << "2addr:         TO 3-ADDR: " << *New;
                bool Sunk = false;

                if (New->findRegisterUseOperand(regB, false, TRI))
                  // FIXME: Temporary workaround. If the new instruction doesn't
                  // uses regB, convertToThreeAddress must have created more
                  // then one instruction.
                  Sunk = Sink3AddrInstruction(mbbi, New, regB, mi);

                mbbi->erase(mi); // Nuke the old inst.

                if (!Sunk) {
                  mi = New;
                  nmi = next(mi);
                }

                ++NumConvertedTo3Addr;
                break; // Done with this instruction.
              }
            }
          }

        InstructionRearranged:
          const TargetRegisterClass* rc = MF.getRegInfo().getRegClass(regA);
          MachineInstr *Orig = MRI->getVRegDef(regB);

          if (Orig && TII->isTriviallyReMaterializable(Orig)) {
            TII->reMaterialize(*mbbi, mi, regA, Orig);
            ReMattedInstrs.insert(Orig);
          } else {
            TII->copyRegToReg(*mbbi, mi, regA, regB, rc, rc);
          }

          MachineBasicBlock::iterator prevMi = prior(mi);
          DOUT << "\t\tprepend:\t"; DEBUG(prevMi->print(*cerr.stream(), &TM));

          // Update live variables for regB.
          LiveVariables::VarInfo& varInfoB = LV->getVarInfo(regB);

          // regB is used in this BB.
          varInfoB.UsedBlocks[mbbi->getNumber()] = true;

          if (LV->removeVirtualRegisterKilled(regB, mbbi, mi))
            LV->addVirtualRegisterKilled(regB, prevMi);

          if (LV->removeVirtualRegisterDead(regB, mbbi, mi))
            LV->addVirtualRegisterDead(regB, prevMi);

          // Replace all occurences of regB with regA.
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

      mi = nmi;
    }
  }

  SmallPtrSet<MachineInstr*, 8>::iterator I = ReMattedInstrs.begin();
  SmallPtrSet<MachineInstr*, 8>::iterator E = ReMattedInstrs.end();

  for (; I != E; ++I) {
    MachineInstr *MI = *I;
    bool InstrDead = true;

    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (!MO.isRegister())
        continue;
      unsigned MOReg = MO.getReg();
      if (!MOReg)
        continue;
      if (MO.isDef()) {
        if (MO.isImplicit())
          continue;

        if (MRI->use_begin(MOReg) != MRI->use_end()) {
          InstrDead = false;
          break;
        }
      }
    }

    if (InstrDead && MI->getNumOperands() > 0)
      MI->eraseFromParent();
  }

  return MadeChange;
}
