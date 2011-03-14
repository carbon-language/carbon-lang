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
INITIALIZE_PASS_BEGIN(ProcessImplicitDefs, "processimpdefs",
                "Process Implicit Definitions", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariables)
INITIALIZE_PASS_END(ProcessImplicitDefs, "processimpdefs",
                "Process Implicit Definitions", false, false)

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

bool
ProcessImplicitDefs::CanTurnIntoImplicitDef(MachineInstr *MI,
                                            unsigned Reg, unsigned OpIdx,
                                            SmallSet<unsigned, 8> &ImpDefRegs) {
  switch(OpIdx) {
  case 1:
    return MI->isCopy() && (MI->getOperand(0).getSubReg() == 0 ||
                            ImpDefRegs.count(MI->getOperand(0).getReg()));
  case 2:
    return MI->isSubregToReg() && (MI->getOperand(0).getSubReg() == 0 ||
                                  ImpDefRegs.count(MI->getOperand(0).getReg()));
  default: return false;
  }
}

static bool isUndefCopy(MachineInstr *MI, unsigned Reg,
                        SmallSet<unsigned, 8> &ImpDefRegs) {
  if (MI->isCopy()) {
    MachineOperand &MO0 = MI->getOperand(0);
    MachineOperand &MO1 = MI->getOperand(1);
    if (MO1.getReg() != Reg)
      return false;
    if (!MO0.getSubReg() || ImpDefRegs.count(MO0.getReg()))
      return true;
    return false;
  }
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

  TII = fn.getTarget().getInstrInfo();
  TRI = fn.getTarget().getRegisterInfo();
  MRI = &fn.getRegInfo();
  LV = &getAnalysis<LiveVariables>();

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
        if (MI->getOperand(0).getSubReg())
          continue;
        unsigned Reg = MI->getOperand(0).getReg();
        ImpDefRegs.insert(Reg);
        if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
          for (const unsigned *SS = TRI->getSubRegisters(Reg); *SS; ++SS)
            ImpDefRegs.insert(*SS);
        }
        ImpDefMIs.push_back(MI);
        continue;
      }

      // Eliminate %reg1032:sub<def> = COPY undef.
      if (MI->isCopy() && MI->getOperand(0).getSubReg()) {
        MachineOperand &MO = MI->getOperand(1);
        if (MO.isUndef() || ImpDefRegs.count(MO.getReg())) {
          if (MO.isKill()) {
            LiveVariables::VarInfo& vi = LV->getVarInfo(MO.getReg());
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
        if (!MO.isReg() || (MO.isDef() && !MO.getSubReg()) || MO.isUndef())
          continue;
        unsigned Reg = MO.getReg();
        if (!Reg)
          continue;
        if (!ImpDefRegs.count(Reg))
          continue;
        // Use is a copy, just turn it into an implicit_def.
        if (CanTurnIntoImplicitDef(MI, Reg, i, ImpDefRegs)) {
          bool isKill = MO.isKill();
          MI->setDesc(TII->get(TargetOpcode::IMPLICIT_DEF));
          for (int j = MI->getNumOperands() - 1, ee = 0; j > ee; --j)
            MI->RemoveOperand(j);
          if (isKill) {
            ImpDefRegs.erase(Reg);
            LiveVariables::VarInfo& vi = LV->getVarInfo(Reg);
            vi.removeKill(MI);
          }
          ChangedToImpDef = true;
          Changed = true;
          break;
        }

        Changed = true;
        MO.setIsUndef();
        // This is a partial register redef of an implicit def.
        // Make sure the whole register is defined by the instruction.
        if (MO.isDef()) {
          MI->addRegisterDefined(Reg);
          continue;
        }
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
      for (MachineRegisterInfo::def_iterator DI = MRI->def_begin(Reg),
             DE = MRI->def_end(); DI != DE; ++DI) {
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
      for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(Reg),
             UE = MRI->use_end(); UI != UE; ++UI) {
        if (UI.getOperand().isUndef())
          continue;
        MachineInstr *RMI = &*UI;
        if (ModInsts.insert(RMI))
          RUses.push_back(RMI);
      }

      for (unsigned i = 0, e = RUses.size(); i != e; ++i) {
        MachineInstr *RMI = RUses[i];

        // Turn a copy use into an implicit_def.
        if (isUndefCopy(RMI, Reg, ImpDefRegs)) {
          RMI->setDesc(TII->get(TargetOpcode::IMPLICIT_DEF));

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
            LiveVariables::VarInfo& vi = LV->getVarInfo(Reg);
            vi.removeKill(RMI);
          }
          continue;
        }

        // Replace Reg with a new vreg that's marked implicit.
        const TargetRegisterClass* RC = MRI->getRegClass(Reg);
        unsigned NewVReg = MRI->createVirtualRegister(RC);
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

