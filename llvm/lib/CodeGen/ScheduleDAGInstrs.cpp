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
#include "ScheduleDAGInstrs.h"
#include "llvm/Operator.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtarget.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallSet.h"
using namespace llvm;

ScheduleDAGInstrs::ScheduleDAGInstrs(MachineFunction &mf,
                                     const MachineLoopInfo &mli,
                                     const MachineDominatorTree &mdt)
  : ScheduleDAG(mf), MLI(mli), MDT(mdt), LoopRegs(MLI, MDT) {}

/// Run - perform scheduling.
///
void ScheduleDAGInstrs::Run(MachineBasicBlock *bb,
                            MachineBasicBlock::iterator begin,
                            MachineBasicBlock::iterator end,
                            unsigned endcount) {
  BB = bb;
  Begin = begin;
  InsertPosIndex = endcount;

  ScheduleDAG::Run(bb, end);
}

/// getUnderlyingObjectFromInt - This is the function that does the work of
/// looking through basic ptrtoint+arithmetic+inttoptr sequences.
static const Value *getUnderlyingObjectFromInt(const Value *V) {
  do {
    if (const Operator *U = dyn_cast<Operator>(V)) {
      // If we find a ptrtoint, we can transfer control back to the
      // regular getUnderlyingObjectFromInt.
      if (U->getOpcode() == Instruction::PtrToInt)
        return U->getOperand(0);
      // If we find an add of a constant or a multiplied value, it's
      // likely that the other operand will lead us to the base
      // object. We don't have to worry about the case where the
      // object address is somehow being computed by the multiply,
      // because our callers only care when the result is an
      // identifibale object.
      if (U->getOpcode() != Instruction::Add ||
          (!isa<ConstantInt>(U->getOperand(1)) &&
           Operator::getOpcode(U->getOperand(1)) != Instruction::Mul))
        return V;
      V = U->getOperand(0);
    } else {
      return V;
    }
    assert(isa<IntegerType>(V->getType()) && "Unexpected operand type!");
  } while (1);
}

/// getUnderlyingObject - This is a wrapper around Value::getUnderlyingObject
/// and adds support for basic ptrtoint+arithmetic+inttoptr sequences.
static const Value *getUnderlyingObject(const Value *V) {
  // First just call Value::getUnderlyingObject to let it do what it does.
  do {
    V = V->getUnderlyingObject();
    // If it found an inttoptr, use special code to continue climing.
    if (Operator::getOpcode(V) != Instruction::IntToPtr)
      break;
    const Value *O = getUnderlyingObjectFromInt(cast<User>(V)->getOperand(0));
    // If that succeeded in finding a pointer, continue the search.
    if (!isa<PointerType>(O->getType()))
      break;
    V = O;
  } while (1);
  return V;
}

/// getUnderlyingObjectForInstr - If this machine instr has memory reference
/// information and it can be tracked to a normal reference to a known
/// object, return the Value for that object. Otherwise return null.
static const Value *getUnderlyingObjectForInstr(const MachineInstr *MI) {
  if (!MI->hasOneMemOperand() ||
      !MI->memoperands_begin()->getValue() ||
      MI->memoperands_begin()->isVolatile())
    return 0;

  const Value *V = MI->memoperands_begin()->getValue();
  if (!V)
    return 0;

  V = getUnderlyingObject(V);
  if (!isa<PseudoSourceValue>(V) && !isIdentifiedObject(V))
    return 0;

  return V;
}

void ScheduleDAGInstrs::StartBlock(MachineBasicBlock *BB) {
  if (MachineLoop *ML = MLI.getLoopFor(BB))
    if (BB == ML->getLoopLatch()) {
      MachineBasicBlock *Header = ML->getHeader();
      for (MachineBasicBlock::livein_iterator I = Header->livein_begin(),
           E = Header->livein_end(); I != E; ++I)
        LoopLiveInRegs.insert(*I);
      LoopRegs.VisitLoop(ML);
    }
}

void ScheduleDAGInstrs::BuildSchedGraph() {
  // We'll be allocating one SUnit for each instruction, plus one for
  // the region exit node.
  SUnits.reserve(BB->size());

  // We build scheduling units by walking a block's instruction list from bottom
  // to top.

  // Remember where a generic side-effecting instruction is as we procede. If
  // ChainMMO is null, this is assumed to have arbitrary side-effects. If
  // ChainMMO is non-null, then Chain makes only a single memory reference.
  SUnit *Chain = 0;
  MachineMemOperand *ChainMMO = 0;

  // Memory references to specific known memory locations are tracked so that
  // they can be given more precise dependencies.
  std::map<const Value *, SUnit *> MemDefs;
  std::map<const Value *, std::vector<SUnit *> > MemUses;

  // Check to see if the scheduler cares about latencies.
  bool UnitLatencies = ForceUnitLatencies();

  // Ask the target if address-backscheduling is desirable, and if so how much.
  const TargetSubtarget &ST = TM.getSubtarget<TargetSubtarget>();
  unsigned SpecialAddressLatency = ST.getSpecialAddressLatency();

  // Walk the list of instructions, from bottom moving up.
  for (MachineBasicBlock::iterator MII = InsertPos, MIE = Begin;
       MII != MIE; --MII) {
    MachineInstr *MI = prior(MII);
    const TargetInstrDesc &TID = MI->getDesc();
    assert(!TID.isTerminator() && !MI->isLabel() &&
           "Cannot schedule terminators or labels!");
    // Create the SUnit for this MI.
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
      // Optionally add output and anti dependencies. For anti
      // dependencies we use a latency of 0 because for a multi-issue
      // target we want to allow the defining instruction to issue
      // in the same cycle as the using instruction.
      // TODO: Using a latency of 1 here for output dependencies assumes
      //       there's no cost for reusing registers.
      SDep::Kind Kind = MO.isUse() ? SDep::Anti : SDep::Output;
      unsigned AOLatency = (Kind == SDep::Anti) ? 0 : 1;
      for (unsigned i = 0, e = DefList.size(); i != e; ++i) {
        SUnit *DefSU = DefList[i];
        if (DefSU != SU &&
            (Kind != SDep::Output || !MO.isDead() ||
             !DefSU->getInstr()->registerDefIsDead(Reg)))
          DefSU->addPred(SDep(SU, Kind, AOLatency, /*Reg=*/Reg));
      }
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        std::vector<SUnit *> &DefList = Defs[*Alias];
        for (unsigned i = 0, e = DefList.size(); i != e; ++i) {
          SUnit *DefSU = DefList[i];
          if (DefSU != SU &&
              (Kind != SDep::Output || !MO.isDead() ||
               !DefSU->getInstr()->registerDefIsDead(Reg)))
            DefSU->addPred(SDep(SU, Kind, AOLatency, /*Reg=*/ *Alias));
        }
      }

      if (MO.isDef()) {
        // Add any data dependencies.
        unsigned DataLatency = SU->Latency;
        for (unsigned i = 0, e = UseList.size(); i != e; ++i) {
          SUnit *UseSU = UseList[i];
          if (UseSU != SU) {
            unsigned LDataLatency = DataLatency;
            // Optionally add in a special extra latency for nodes that
            // feed addresses.
            // TODO: Do this for register aliases too.
            // TODO: Perhaps we should get rid of
            // SpecialAddressLatency and just move this into
            // adjustSchedDependency for the targets that care about
            // it.
            if (SpecialAddressLatency != 0 && !UnitLatencies) {
              MachineInstr *UseMI = UseSU->getInstr();
              const TargetInstrDesc &UseTID = UseMI->getDesc();
              int RegUseIndex = UseMI->findRegisterUseOperandIdx(Reg);
              assert(RegUseIndex >= 0 && "UseMI doesn's use register!");
              if ((UseTID.mayLoad() || UseTID.mayStore()) &&
                  (unsigned)RegUseIndex < UseTID.getNumOperands() &&
                  UseTID.OpInfo[RegUseIndex].isLookupPtrRegClass())
                LDataLatency += SpecialAddressLatency;
            }
            // Adjust the dependence latency using operand def/use
            // information (if any), and then allow the target to
            // perform its own adjustments.
            const SDep& dep = SDep(SU, SDep::Data, LDataLatency, Reg);
            if (!UnitLatencies) {
              ComputeOperandLatency(SU, UseSU, (SDep &)dep);
              ST.adjustSchedDependency(SU, UseSU, (SDep &)dep);
            }
            UseSU->addPred(dep);
          }
        }
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          std::vector<SUnit *> &UseList = Uses[*Alias];
          for (unsigned i = 0, e = UseList.size(); i != e; ++i) {
            SUnit *UseSU = UseList[i];
            if (UseSU != SU) {
              const SDep& dep = SDep(SU, SDep::Data, DataLatency, *Alias);
              if (!UnitLatencies) {
                ComputeOperandLatency(SU, UseSU, (SDep &)dep);
                ST.adjustSchedDependency(SU, UseSU, (SDep &)dep);
              }
              UseSU->addPred(dep);
            }
          }
        }

        // If a def is going to wrap back around to the top of the loop,
        // backschedule it.
        if (!UnitLatencies && DefList.empty()) {
          LoopDependencies::LoopDeps::iterator I = LoopRegs.Deps.find(Reg);
          if (I != LoopRegs.Deps.end()) {
            const MachineOperand *UseMO = I->second.first;
            unsigned Count = I->second.second;
            const MachineInstr *UseMI = UseMO->getParent();
            unsigned UseMOIdx = UseMO - &UseMI->getOperand(0);
            const TargetInstrDesc &UseTID = UseMI->getDesc();
            // TODO: If we knew the total depth of the region here, we could
            // handle the case where the whole loop is inside the region but
            // is large enough that the isScheduleHigh trick isn't needed.
            if (UseMOIdx < UseTID.getNumOperands()) {
              // Currently, we only support scheduling regions consisting of
              // single basic blocks. Check to see if the instruction is in
              // the same region by checking to see if it has the same parent.
              if (UseMI->getParent() != MI->getParent()) {
                unsigned Latency = SU->Latency;
                if (UseTID.OpInfo[UseMOIdx].isLookupPtrRegClass())
                  Latency += SpecialAddressLatency;
                // This is a wild guess as to the portion of the latency which
                // will be overlapped by work done outside the current
                // scheduling region.
                Latency -= std::min(Latency, Count);
                // Add the artifical edge.
                ExitSU.addPred(SDep(SU, SDep::Order, Latency,
                                    /*Reg=*/0, /*isNormalMemory=*/false,
                                    /*isMustAlias=*/false,
                                    /*isArtificial=*/true));
              } else if (SpecialAddressLatency > 0 &&
                         UseTID.OpInfo[UseMOIdx].isLookupPtrRegClass()) {
                // The entire loop body is within the current scheduling region
                // and the latency of this operation is assumed to be greater
                // than the latency of the loop.
                // TODO: Recursively mark data-edge predecessors as
                //       isScheduleHigh too.
                SU->isScheduleHigh = true;
              }
            }
            LoopRegs.Deps.erase(I);
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
    if (TID.isCall() || TID.hasUnmodeledSideEffects()) {
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
      if (!ChainTID.isCall() &&
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
      if (const Value *V = getUnderlyingObjectForInstr(MI)) {
        // A store to a specific PseudoSourceValue. Add precise dependencies.
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
        // Add dependencies from all the PendingLoads, since without
        // memoperands we must assume they alias anything.
        for (unsigned k = 0, m = PendingLoads.size(); k != m; ++k)
          PendingLoads[k]->addPred(SDep(SU, SDep::Order, SU->Latency));
        // Add a general dependence too, if needed.
        if (Chain)
          Chain->addPred(SDep(SU, SDep::Order, SU->Latency));
      } else
        // Treat all other stores conservatively.
        goto new_chain;
    } else if (TID.mayLoad()) {
      if (TII->isInvariantLoad(MI)) {
        // Invariant load, no chain dependencies needed!
      } else if (const Value *V = getUnderlyingObjectForInstr(MI)) {
        // A load from a specific PseudoSourceValue. Add precise dependencies.
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
        // A normal load. Depend on the general chain, as well as on
        // all stores. In the absense of MachineMemOperand information,
        // we can't even assume that the load doesn't alias well-behaved
        // memory locations.
        if (Chain)
          Chain->addPred(SDep(SU, SDep::Order, SU->Latency));
        for (std::map<const Value *, SUnit *>::iterator I = MemDefs.begin(),
             E = MemDefs.end(); I != E; ++I)
          I->second->addPred(SDep(SU, SDep::Order, SU->Latency));
        PendingLoads.push_back(SU);
      }
    }
  }

  for (int i = 0, e = TRI->getNumRegs(); i != e; ++i) {
    Defs[i].clear();
    Uses[i].clear();
  }
  PendingLoads.clear();
}

void ScheduleDAGInstrs::FinishBlock() {
  // Nothing to do.
}

void ScheduleDAGInstrs::ComputeLatency(SUnit *SU) {
  const InstrItineraryData &InstrItins = TM.getInstrItineraryData();

  // Compute the latency for the node.
  SU->Latency =
    InstrItins.getStageLatency(SU->getInstr()->getDesc().getSchedClass());

  // Simplistic target-independent heuristic: assume that loads take
  // extra time.
  if (InstrItins.isEmpty())
    if (SU->getInstr()->getDesc().mayLoad())
      SU->Latency += 2;
}

void ScheduleDAGInstrs::ComputeOperandLatency(SUnit *Def, SUnit *Use, 
                                              SDep& dep) const {
  const InstrItineraryData &InstrItins = TM.getInstrItineraryData();
  if (InstrItins.isEmpty())
    return;
  
  // For a data dependency with a known register...
  if ((dep.getKind() != SDep::Data) || (dep.getReg() == 0))
    return;

  const unsigned Reg = dep.getReg();

  // ... find the definition of the register in the defining
  // instruction
  MachineInstr *DefMI = Def->getInstr();
  int DefIdx = DefMI->findRegisterDefOperandIdx(Reg);
  if (DefIdx != -1) {
    int DefCycle = InstrItins.getOperandCycle(DefMI->getDesc().getSchedClass(), DefIdx);
    if (DefCycle >= 0) {
      MachineInstr *UseMI = Use->getInstr();
      const unsigned UseClass = UseMI->getDesc().getSchedClass();

      // For all uses of the register, calculate the maxmimum latency
      int Latency = -1;
      for (unsigned i = 0, e = UseMI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = UseMI->getOperand(i);
        if (!MO.isReg() || !MO.isUse())
          continue;
        unsigned MOReg = MO.getReg();
        if (MOReg != Reg)
          continue;

        int UseCycle = InstrItins.getOperandCycle(UseClass, i);
        if (UseCycle >= 0)
          Latency = std::max(Latency, DefCycle - UseCycle + 1);
      }

      // If we found a latency, then replace the existing dependence latency.
      if (Latency >= 0)
        dep.setLatency(Latency);
    }
  }
}

void ScheduleDAGInstrs::dumpNode(const SUnit *SU) const {
  SU->getInstr()->dump();
}

std::string ScheduleDAGInstrs::getGraphNodeLabel(const SUnit *SU) const {
  std::string s;
  raw_string_ostream oss(s);
  if (SU == &EntrySU)
    oss << "<entry>";
  else if (SU == &ExitSU)
    oss << "<exit>";
  else
    SU->getInstr()->print(oss);
  return oss.str();
}

// EmitSchedule - Emit the machine code in scheduled order.
MachineBasicBlock *ScheduleDAGInstrs::
EmitSchedule(DenseMap<MachineBasicBlock*, MachineBasicBlock*> *EM) {
  // For MachineInstr-based scheduling, we're rescheduling the instructions in
  // the block, so start by removing them from the block.
  while (Begin != InsertPos) {
    MachineBasicBlock::iterator I = Begin;
    ++Begin;
    BB->remove(I);
  }

  // Then re-insert them according to the given schedule.
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    SUnit *SU = Sequence[i];
    if (!SU) {
      // Null SUnit* is a noop.
      EmitNoop();
      continue;
    }

    BB->insert(InsertPos, SU->getInstr());
  }

  // Update the Begin iterator, as the first instruction in the block
  // may have been scheduled later.
  if (!Sequence.empty())
    Begin = Sequence[0]->getInstr();

  return BB;
}
