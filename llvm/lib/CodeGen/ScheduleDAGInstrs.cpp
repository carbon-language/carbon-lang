//===---- ScheduleDAGInstrs.cpp - MachineInstr Rescheduling ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the ScheduleDAGInstrs class, which implements re-scheduling
// of MachineInstrs.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sched-instrs"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtarget.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallSet.h"
#include <map>
using namespace llvm;

ScheduleDAGInstrs::ScheduleDAGInstrs(MachineBasicBlock *bb,
                                     const TargetMachine &tm,
                                     const MachineLoopInfo &mli,
                                     const MachineDominatorTree &mdt)
  : ScheduleDAG(0, bb, tm), MLI(mli), MDT(mdt) {}

void ScheduleDAGInstrs::BuildSchedUnits() {
  SUnits.clear();
  SUnits.reserve(BB->size());

  // We build scheduling units by walking a block's instruction list from bottom
  // to top.

  // Remember where defs and uses of each physical register are as we procede.
  std::vector<SUnit *> Defs[TargetRegisterInfo::FirstVirtualRegister] = {};
  std::vector<SUnit *> Uses[TargetRegisterInfo::FirstVirtualRegister] = {};

  // Remember where unknown loads are after the most recent unknown store
  // as we procede.
  std::vector<SUnit *> PendingLoads;

  // Remember where a generic side-effecting instruction is as we procede. If
  // ChainMMO is null, this is assumed to have arbitrary side-effects. If
  // ChainMMO is non-null, then Chain makes only a single memory reference.
  SUnit *Chain = 0;
  MachineMemOperand *ChainMMO = 0;

  // Memory references to specific known memory locations are tracked so that
  // they can be given more precise dependencies.
  std::map<const Value *, SUnit *> MemDefs;
  std::map<const Value *, std::vector<SUnit *> > MemUses;

  // Terminators can perform control transfers, we we need to make sure that
  // all the work of the block is done before the terminator.
  SUnit *Terminator = 0;

  // Check to see if the scheduler cares about latencies.
  bool UnitLatencies = ForceUnitLatencies();

  for (MachineBasicBlock::iterator MII = BB->end(), MIE = BB->begin();
       MII != MIE; --MII) {
    MachineInstr *MI = prior(MII);
    const TargetInstrDesc &TID = MI->getDesc();
    SUnit *SU = NewSUnit(MI);

    // Assign the Latency field of SU using target-provided information.
    if (UnitLatencies)
      SU->Latency = 1;
    else
      ComputeLatency(SU);

    // Add register-based dependencies (data, anti, and output).
    for (unsigned j = 0, n = MI->getNumOperands(); j != n; ++j) {
      const MachineOperand &MO = MI->getOperand(j);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;

      assert(TRI->isPhysicalRegister(Reg) && "Virtual register encountered!");
      std::vector<SUnit *> &UseList = Uses[Reg];
      std::vector<SUnit *> &DefList = Defs[Reg];
      // Optionally add output and anti dependencies.
      // TODO: Using a latency of 1 here assumes there's no cost for
      //       reusing registers.
      SDep::Kind Kind = MO.isUse() ? SDep::Anti : SDep::Output;
      for (unsigned i = 0, e = DefList.size(); i != e; ++i) {
        SUnit *DefSU = DefList[i];
        if (DefSU != SU &&
            (Kind != SDep::Output || !MO.isDead() ||
             !DefSU->getInstr()->registerDefIsDead(Reg)))
          DefSU->addPred(SDep(SU, Kind, /*Latency=*/1, /*Reg=*/Reg));
      }
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        std::vector<SUnit *> &DefList = Defs[*Alias];
        for (unsigned i = 0, e = DefList.size(); i != e; ++i) {
          SUnit *DefSU = DefList[i];
          if (DefSU != SU &&
              (Kind != SDep::Output || !MO.isDead() ||
               !DefSU->getInstr()->registerDefIsDead(Reg)))
            DefSU->addPred(SDep(SU, Kind, /*Latency=*/1, /*Reg=*/ *Alias));
        }
      }

      if (MO.isDef()) {
        // Add any data dependencies.
        unsigned DataLatency = SU->Latency;
        for (unsigned i = 0, e = UseList.size(); i != e; ++i) {
          SUnit *UseSU = UseList[i];
          if (UseSU != SU) {
            UseSU->addPred(SDep(SU, SDep::Data, DataLatency, Reg));
          }
        }
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          std::vector<SUnit *> &UseList = Uses[*Alias];
          for (unsigned i = 0, e = UseList.size(); i != e; ++i) {
            SUnit *UseSU = UseList[i];
            if (UseSU != SU)
              UseSU->addPred(SDep(SU, SDep::Data, DataLatency, *Alias));
          }
        }

        UseList.clear();
        if (!MO.isDead())
          DefList.clear();
        DefList.push_back(SU);
      } else {
        UseList.push_back(SU);
      }
    }

    // Add chain dependencies.
    // Note that isStoreToStackSlot and isLoadFromStackSLot are not usable
    // after stack slots are lowered to actual addresses.
    // TODO: Use an AliasAnalysis and do real alias-analysis queries, and
    // produce more precise dependence information.
    if (TID.isCall() || TID.isReturn() || TID.isBranch() ||
        TID.hasUnmodeledSideEffects()) {
    new_chain:
      // This is the conservative case. Add dependencies on all memory
      // references.
      if (Chain)
        Chain->addPred(SDep(SU, SDep::Order, SU->Latency));
      Chain = SU;
      for (unsigned k = 0, m = PendingLoads.size(); k != m; ++k)
        PendingLoads[k]->addPred(SDep(SU, SDep::Order, SU->Latency));
      PendingLoads.clear();
      for (std::map<const Value *, SUnit *>::iterator I = MemDefs.begin(),
           E = MemDefs.end(); I != E; ++I) {
        I->second->addPred(SDep(SU, SDep::Order, SU->Latency));
        I->second = SU;
      }
      for (std::map<const Value *, std::vector<SUnit *> >::iterator I =
           MemUses.begin(), E = MemUses.end(); I != E; ++I) {
        for (unsigned i = 0, e = I->second.size(); i != e; ++i)
          I->second[i]->addPred(SDep(SU, SDep::Order, SU->Latency));
        I->second.clear();
      }
      // See if it is known to just have a single memory reference.
      MachineInstr *ChainMI = Chain->getInstr();
      const TargetInstrDesc &ChainTID = ChainMI->getDesc();
      if (!ChainTID.isCall() && !ChainTID.isReturn() && !ChainTID.isBranch() &&
          !ChainTID.hasUnmodeledSideEffects() &&
          ChainMI->hasOneMemOperand() &&
          !ChainMI->memoperands_begin()->isVolatile() &&
          ChainMI->memoperands_begin()->getValue())
        // We know that the Chain accesses one specific memory location.
        ChainMMO = &*ChainMI->memoperands_begin();
      else
        // Unknown memory accesses. Assume the worst.
        ChainMMO = 0;
    } else if (TID.mayStore()) {
      if (MI->hasOneMemOperand() &&
          MI->memoperands_begin()->getValue() &&
          !MI->memoperands_begin()->isVolatile() &&
          isa<PseudoSourceValue>(MI->memoperands_begin()->getValue())) {
        // A store to a specific PseudoSourceValue. Add precise dependencies.
        const Value *V = MI->memoperands_begin()->getValue();
        // Handle the def in MemDefs, if there is one.
        std::map<const Value *, SUnit *>::iterator I = MemDefs.find(V);
        if (I != MemDefs.end()) {
          I->second->addPred(SDep(SU, SDep::Order, SU->Latency, /*Reg=*/0,
                                  /*isNormalMemory=*/true));
          I->second = SU;
        } else {
          MemDefs[V] = SU;
        }
        // Handle the uses in MemUses, if there are any.
        std::map<const Value *, std::vector<SUnit *> >::iterator J =
          MemUses.find(V);
        if (J != MemUses.end()) {
          for (unsigned i = 0, e = J->second.size(); i != e; ++i)
            J->second[i]->addPred(SDep(SU, SDep::Order, SU->Latency, /*Reg=*/0,
                                       /*isNormalMemory=*/true));
          J->second.clear();
        }
        // Add a general dependence too, if needed.
        if (Chain)
          Chain->addPred(SDep(SU, SDep::Order, SU->Latency));
      } else
        // Treat all other stores conservatively.
        goto new_chain;
    } else if (TID.mayLoad()) {
      if (TII->isInvariantLoad(MI)) {
        // Invariant load, no chain dependencies needed!
      } else if (MI->hasOneMemOperand() &&
                 MI->memoperands_begin()->getValue() &&
                 !MI->memoperands_begin()->isVolatile() &&
                 isa<PseudoSourceValue>(MI->memoperands_begin()->getValue())) {
        // A load from a specific PseudoSourceValue. Add precise dependencies.
        const Value *V = MI->memoperands_begin()->getValue();
        std::map<const Value *, SUnit *>::iterator I = MemDefs.find(V);
        if (I != MemDefs.end())
          I->second->addPred(SDep(SU, SDep::Order, SU->Latency, /*Reg=*/0,
                                  /*isNormalMemory=*/true));
        MemUses[V].push_back(SU);

        // Add a general dependence too, if needed.
        if (Chain && (!ChainMMO ||
                      (ChainMMO->isStore() || ChainMMO->isVolatile())))
          Chain->addPred(SDep(SU, SDep::Order, SU->Latency));
      } else if (MI->hasVolatileMemoryRef()) {
        // Treat volatile loads conservatively. Note that this includes
        // cases where memoperand information is unavailable.
        goto new_chain;
      } else {
        // A normal load. Just depend on the general chain.
        if (Chain)
          Chain->addPred(SDep(SU, SDep::Order, SU->Latency));
        PendingLoads.push_back(SU);
      }
    }

    // Add chain edges from the terminator to ensure that all the work of the
    // block is completed before any control transfers.
    if (Terminator && SU->Succs.empty())
      Terminator->addPred(SDep(SU, SDep::Order, SU->Latency));
    if (TID.isTerminator() || MI->isLabel())
      Terminator = SU;
  }
}

void ScheduleDAGInstrs::ComputeLatency(SUnit *SU) {
  const InstrItineraryData &InstrItins = TM.getInstrItineraryData();

  // Compute the latency for the node.  We use the sum of the latencies for
  // all nodes flagged together into this SUnit.
  SU->Latency =
    InstrItins.getLatency(SU->getInstr()->getDesc().getSchedClass());

  // Simplistic target-independent heuristic: assume that loads take
  // extra time.
  if (InstrItins.isEmpty())
    if (SU->getInstr()->getDesc().mayLoad())
      SU->Latency += 2;
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
