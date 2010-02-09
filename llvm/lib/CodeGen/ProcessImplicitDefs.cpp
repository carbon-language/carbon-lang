//===---------------------- ProcessImplicitDefs.cpp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "processimplicitdefs"

#include "llvm/CodeGen/ProcessImplicitDefs.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"


using namespace llvm;

char ProcessImplicitDefs::ID = 0;
static RegisterPass<ProcessImplicitDefs> X("processimpdefs",
                                           "Process Implicit Definitions.");

void ProcessImplicitDefs::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addPreserved<AliasAnalysis>();
  AU.addPreserved<LiveVariables>();
  AU.addRequired<LiveVariables>();
  AU.addPreservedID(MachineLoopInfoID);
  AU.addPreservedID(MachineDominatorsID);
  AU.addPreservedID(TwoAddressInstructionPassID);
  AU.addPreservedID(PHIEliminationID);
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool ProcessImplicitDefs::CanTurnIntoImplicitDef(MachineInstr *MI,
                                                 unsigned Reg, unsigned OpIdx,
                                                 const TargetInstrInfo *tii_) {
  unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
  if (tii_->isMoveInstr(*MI, SrcReg, DstReg, SrcSubReg, DstSubReg) &&
      Reg == SrcReg)
    return true;

  if (OpIdx == 2 && MI->isSubregToReg())
    return true;
  if (OpIdx == 1 && MI->isExtractSubreg())
    return true;
  return false;
}

/// processImplicitDefs - Process IMPLICIT_DEF instructions and make sure
/// there is one implicit_def for each use. Add isUndef marker to
/// implicit_def defs and their uses.
bool ProcessImplicitDefs::runOnMachineFunction(MachineFunction &fn) {

  DEBUG(dbgs() << "********** PROCESS IMPLICIT DEFS **********\n"
               << "********** Function: "
               << ((Value*)fn.getFunction())->getName() << '\n');

  bool Changed = false;

  const TargetInstrInfo *tii_ = fn.getTarget().getInstrInfo();
  const TargetRegisterInfo *tri_ = fn.getTarget().getRegisterInfo();
  MachineRegisterInfo *mri_ = &fn.getRegInfo();

  LiveVariables *lv_ = &getAnalysis<LiveVariables>();

  SmallSet<unsigned, 8> ImpDefRegs;
  SmallVector<MachineInstr*, 8> ImpDefMIs;
  SmallVector<MachineInstr*, 4> RUses;
  SmallPtrSet<MachineBasicBlock*,16> Visited;
  SmallPtrSet<MachineInstr*, 8> ModInsts;

  MachineBasicBlock *Entry = fn.begin();
  for (df_ext_iterator<MachineBasicBlock*, SmallPtrSet<MachineBasicBlock*,16> >
         DFI = df_ext_begin(Entry, Visited), E = df_ext_end(Entry, Visited);
       DFI != E; ++DFI) {
    MachineBasicBlock *MBB = *DFI;
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ) {
      MachineInstr *MI = &*I;
      ++I;
      if (MI->isImplicitDef()) {
        unsigned Reg = MI->getOperand(0).getReg();
        ImpDefRegs.insert(Reg);
        if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
          for (const unsigned *SS = tri_->getSubRegisters(Reg); *SS; ++SS)
            ImpDefRegs.insert(*SS);
        }
        ImpDefMIs.push_back(MI);
        continue;
      }

      if (MI->isInsertSubreg()) {
        MachineOperand &MO = MI->getOperand(2);
        if (ImpDefRegs.count(MO.getReg())) {
          // %reg1032<def> = INSERT_SUBREG %reg1032, undef, 2
          // This is an identity copy, eliminate it now.
          if (MO.isKill()) {
            LiveVariables::VarInfo& vi = lv_->getVarInfo(MO.getReg());
            vi.removeKill(MI);
          }
          MI->eraseFromParent();
          Changed = true;
          continue;
        }
      }

      bool ChangedToImpDef = false;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand& MO = MI->getOperand(i);
        if (!MO.isReg() || !MO.isUse() || MO.isUndef())
          continue;
        unsigned Reg = MO.getReg();
        if (!Reg)
          continue;
        if (!ImpDefRegs.count(Reg))
          continue;
        // Use is a copy, just turn it into an implicit_def.
        if (CanTurnIntoImplicitDef(MI, Reg, i, tii_)) {
          bool isKill = MO.isKill();
          MI->setDesc(tii_->get(TargetOpcode::IMPLICIT_DEF));
          for (int j = MI->getNumOperands() - 1, ee = 0; j > ee; --j)
            MI->RemoveOperand(j);
          if (isKill) {
            ImpDefRegs.erase(Reg);
            LiveVariables::VarInfo& vi = lv_->getVarInfo(Reg);
            vi.removeKill(MI);
          }
          ChangedToImpDef = true;
          Changed = true;
          break;
        }

        Changed = true;
        MO.setIsUndef();
        if (MO.isKill() || MI->isRegTiedToDefOperand(i)) {
          // Make sure other uses of 
          for (unsigned j = i+1; j != e; ++j) {
            MachineOperand &MOJ = MI->getOperand(j);
            if (MOJ.isReg() && MOJ.isUse() && MOJ.getReg() == Reg)
              MOJ.setIsUndef();
          }
          ImpDefRegs.erase(Reg);
        }
      }

      if (ChangedToImpDef) {
        // Backtrack to process this new implicit_def.
        --I;
      } else {
        for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
          MachineOperand& MO = MI->getOperand(i);
          if (!MO.isReg() || !MO.isDef())
            continue;
          ImpDefRegs.erase(MO.getReg());
        }
      }
    }

    // Any outstanding liveout implicit_def's?
    for (unsigned i = 0, e = ImpDefMIs.size(); i != e; ++i) {
      MachineInstr *MI = ImpDefMIs[i];
      unsigned Reg = MI->getOperand(0).getReg();
      if (TargetRegisterInfo::isPhysicalRegister(Reg) ||
          !ImpDefRegs.count(Reg)) {
        // Delete all "local" implicit_def's. That include those which define
        // physical registers since they cannot be liveout.
        MI->eraseFromParent();
        Changed = true;
        continue;
      }

      // If there are multiple defs of the same register and at least one
      // is not an implicit_def, do not insert implicit_def's before the
      // uses.
      bool Skip = false;
      SmallVector<MachineInstr*, 4> DeadImpDefs;
      for (MachineRegisterInfo::def_iterator DI = mri_->def_begin(Reg),
             DE = mri_->def_end(); DI != DE; ++DI) {
        MachineInstr *DeadImpDef = &*DI;
        if (!DeadImpDef->isImplicitDef()) {
          Skip = true;
          break;
        }
        DeadImpDefs.push_back(DeadImpDef);
      }
      if (Skip)
        continue;

      // The only implicit_def which we want to keep are those that are live
      // out of its block.
      for (unsigned j = 0, ee = DeadImpDefs.size(); j != ee; ++j)
        DeadImpDefs[j]->eraseFromParent();
      Changed = true;

      // Process each use instruction once.
      for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(Reg),
             UE = mri_->use_end(); UI != UE; ++UI) {
        MachineInstr *RMI = &*UI;
        MachineBasicBlock *RMBB = RMI->getParent();
        if (RMBB == MBB)
          continue;
        if (ModInsts.insert(RMI))
          RUses.push_back(RMI);
      }

      for (unsigned i = 0, e = RUses.size(); i != e; ++i) {
        MachineInstr *RMI = RUses[i];

        // Turn a copy use into an implicit_def.
        unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
        if (tii_->isMoveInstr(*RMI, SrcReg, DstReg, SrcSubReg, DstSubReg) &&
            Reg == SrcReg) {
          RMI->setDesc(tii_->get(TargetOpcode::IMPLICIT_DEF));

          bool isKill = false;
          SmallVector<unsigned, 4> Ops;
          for (unsigned j = 0, ee = RMI->getNumOperands(); j != ee; ++j) {
            MachineOperand &RRMO = RMI->getOperand(j);
            if (RRMO.isReg() && RRMO.getReg() == Reg) {
              Ops.push_back(j);
              if (RRMO.isKill())
                isKill = true;
            }
          }
          // Leave the other operands along.
          for (unsigned j = 0, ee = Ops.size(); j != ee; ++j) {
            unsigned OpIdx = Ops[j];
            RMI->RemoveOperand(OpIdx-j);
          }

          // Update LiveVariables varinfo if the instruction is a kill.
          if (isKill) {
            LiveVariables::VarInfo& vi = lv_->getVarInfo(Reg);
            vi.removeKill(RMI);
          }
          continue;
        }

        // Replace Reg with a new vreg that's marked implicit.
        const TargetRegisterClass* RC = mri_->getRegClass(Reg);
        unsigned NewVReg = mri_->createVirtualRegister(RC);
        bool isKill = true;
        for (unsigned j = 0, ee = RMI->getNumOperands(); j != ee; ++j) {
          MachineOperand &RRMO = RMI->getOperand(j);
          if (RRMO.isReg() && RRMO.getReg() == Reg) {
            RRMO.setReg(NewVReg);
            RRMO.setIsUndef();
            if (isKill) {
              // Only the first operand of NewVReg is marked kill.
              RRMO.setIsKill();
              isKill = false;
            }
          }
        }
      }
      RUses.clear();
      ModInsts.clear();
    }
    ImpDefRegs.clear();
    ImpDefMIs.clear();
  }

  return Changed;
}

