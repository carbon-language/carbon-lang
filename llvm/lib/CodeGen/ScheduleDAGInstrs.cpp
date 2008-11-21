//===---- ScheduleDAG.cpp - Implement the ScheduleDAG class ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the ScheduleDAG class, which is a base class used by
// scheduling implementation classes.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sched-instrs"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

ScheduleDAGInstrs::ScheduleDAGInstrs(MachineBasicBlock *bb,
                                     const TargetMachine &tm)
  : ScheduleDAG(0, bb, tm) {}

void ScheduleDAGInstrs::BuildSchedUnits() {
  SUnits.clear();
  SUnits.reserve(BB->size());

  std::vector<SUnit *> PendingLoads;
  SUnit *Terminator = 0;
  SUnit *Chain = 0;
  SUnit *Defs[TargetRegisterInfo::FirstVirtualRegister] = {};
  std::vector<SUnit *> Uses[TargetRegisterInfo::FirstVirtualRegister] = {};
  int Cost = 1; // FIXME

  for (MachineBasicBlock::iterator MII = BB->end(), MIE = BB->begin();
       MII != MIE; --MII) {
    MachineInstr *MI = prior(MII);
    SUnit *SU = NewSUnit(MI);

    for (unsigned j = 0, n = MI->getNumOperands(); j != n; ++j) {
      const MachineOperand &MO = MI->getOperand(j);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;

      assert(TRI->isPhysicalRegister(Reg) && "Virtual register encountered!");
      std::vector<SUnit *> &UseList = Uses[Reg];
      SUnit *&Def = Defs[Reg];
      // Optionally add output and anti dependences.
      if (Def && Def != SU)
        Def->addPred(SU, /*isCtrl=*/true, /*isSpecial=*/false,
                     /*PhyReg=*/Reg, Cost, /*isAntiDep=*/MO.isUse());
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        SUnit *&Def = Defs[*Alias];
        if (Def && Def != SU)
          Def->addPred(SU, /*isCtrl=*/true, /*isSpecial=*/false,
                       /*PhyReg=*/*Alias, Cost);
      }

      if (MO.isDef()) {
        // Add any data dependencies.
        for (unsigned i = 0, e = UseList.size(); i != e; ++i)
          if (UseList[i] != SU)
            UseList[i]->addPred(SU, /*isCtrl=*/false, /*isSpecial=*/false,
                                /*PhysReg=*/Reg, Cost);
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          std::vector<SUnit *> &UseList = Uses[*Alias];
          for (unsigned i = 0, e = UseList.size(); i != e; ++i)
            if (UseList[i] != SU)
              UseList[i]->addPred(SU, /*isCtrl=*/false, /*isSpecial=*/false,
                                  /*PhysReg=*/*Alias, Cost);
        }

        UseList.clear();
        Def = SU;
      } else {
        UseList.push_back(SU);
      }
    }
    bool False = false;
    bool True = true;
    if (!MI->isSafeToMove(TII, False)) {
      if (Chain)
        Chain->addPred(SU, /*isCtrl=*/false, /*isSpecial=*/false);
      for (unsigned k = 0, m = PendingLoads.size(); k != m; ++k)
        PendingLoads[k]->addPred(SU, /*isCtrl=*/false, /*isSpecial=*/false);
      PendingLoads.clear();
      Chain = SU;
    } else if (!MI->isSafeToMove(TII, True)) {
      if (Chain)
        Chain->addPred(SU, /*isCtrl=*/false, /*isSpecial=*/false);
      PendingLoads.push_back(SU);
    }
    if (Terminator && SU->Succs.empty())
      Terminator->addPred(SU, /*isCtrl=*/false, /*isSpecial=*/false);
    if (MI->getDesc().isTerminator() || MI->isLabel())
      Terminator = SU;

    // Assign the Latency field of SU using target-provided information.
    ComputeLatency(SU);
  }
}

void ScheduleDAGInstrs::ComputeLatency(SUnit *SU) {
  const InstrItineraryData &InstrItins = TM.getInstrItineraryData();

  // Compute the latency for the node.  We use the sum of the latencies for
  // all nodes flagged together into this SUnit.
  SU->Latency =
    InstrItins.getLatency(SU->getInstr()->getDesc().getSchedClass());
}

void ScheduleDAGInstrs::dumpNode(const SUnit *SU) const {
  SU->getInstr()->dump();
}

std::string ScheduleDAGInstrs::getGraphNodeLabel(const SUnit *SU) const {
  std::string s;
  raw_string_ostream oss(s);
  SU->getInstr()->print(oss);
  return oss.str();
}

// EmitSchedule - Emit the machine code in scheduled order.
MachineBasicBlock *ScheduleDAGInstrs::EmitSchedule() {
  // For MachineInstr-based scheduling, we're rescheduling the instructions in
  // the block, so start by removing them from the block.
  while (!BB->empty())
    BB->remove(BB->begin());

  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    SUnit *SU = Sequence[i];
    if (!SU) {
      // Null SUnit* is a noop.
      EmitNoop();
      continue;
    }

    BB->push_back(SU->getInstr());
  }

  return BB;
}
