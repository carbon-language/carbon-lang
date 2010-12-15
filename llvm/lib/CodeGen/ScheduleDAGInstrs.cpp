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
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineMemOperand.h"
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
  : ScheduleDAG(mf), MLI(mli), MDT(mdt), MFI(mf.getFrameInfo()),
    InstrItins(mf.getTarget().getInstrItineraryData()),
    Defs(TRI->getNumRegs()), Uses(TRI->getNumRegs()), LoopRegs(MLI, MDT) {
  DbgValueVec.clear();
}

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
    assert(V->getType()->isIntegerTy() && "Unexpected operand type!");
  } while (1);
}

/// getUnderlyingObject - This is a wrapper around GetUnderlyingObject
/// and adds support for basic ptrtoint+arithmetic+inttoptr sequences.
static const Value *getUnderlyingObject(const Value *V) {
  // First just call Value::getUnderlyingObject to let it do what it does.
  do {
    V = GetUnderlyingObject(V);
    // If it found an inttoptr, use special code to continue climing.
    if (Operator::getOpcode(V) != Instruction::IntToPtr)
      break;
    const Value *O = getUnderlyingObjectFromInt(cast<User>(V)->getOperand(0));
    // If that succeeded in finding a pointer, continue the search.
    if (!O->getType()->isPointerTy())
      break;
    V = O;
  } while (1);
  return V;
}

/// getUnderlyingObjectForInstr - If this machine instr has memory reference
/// information and it can be tracked to a normal reference to a known
/// object, return the Value for that object. Otherwise return null.
static const Value *getUnderlyingObjectForInstr(const MachineInstr *MI,
                                                const MachineFrameInfo *MFI,
                                                bool &MayAlias) {
  MayAlias = true;
  if (!MI->hasOneMemOperand() ||
      !(*MI->memoperands_begin())->getValue() ||
      (*MI->memoperands_begin())->isVolatile())
    return 0;

  const Value *V = (*MI->memoperands_begin())->getValue();
  if (!V)
    return 0;

  V = getUnderlyingObject(V);
  if (const PseudoSourceValue *PSV = dyn_cast<PseudoSourceValue>(V)) {
    // For now, ignore PseudoSourceValues which may alias LLVM IR values
    // because the code that uses this function has no way to cope with
    // such aliases.
    if (PSV->isAliased(MFI))
      return 0;
    
    MayAlias = PSV->mayAlias(MFI);
    return V;
  }

  if (isIdentifiedObject(V))
    return V;

  return 0;
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

/// AddSchedBarrierDeps - Add dependencies from instructions in the current
/// list of instructions being scheduled to scheduling barrier by adding
/// the exit SU to the register defs and use list. This is because we want to
/// make sure instructions which define registers that are either used by
/// the terminator or are live-out are properly scheduled. This is
/// especially important when the definition latency of the return value(s)
/// are too high to be hidden by the branch or when the liveout registers
/// used by instructions in the fallthrough block.
void ScheduleDAGInstrs::AddSchedBarrierDeps() {
  MachineInstr *ExitMI = InsertPos != BB->end() ? &*InsertPos : 0;
  ExitSU.setInstr(ExitMI);
  bool AllDepKnown = ExitMI &&
    (ExitMI->getDesc().isCall() || ExitMI->getDesc().isBarrier());
  if (ExitMI && AllDepKnown) {
    // If it's a call or a barrier, add dependencies on the defs and uses of
    // instruction.
    for (unsigned i = 0, e = ExitMI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = ExitMI->getOperand(i);
      if (!MO.isReg() || MO.isDef()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;

      assert(TRI->isPhysicalRegister(Reg) && "Virtual register encountered!");
      Uses[Reg].push_back(&ExitSU);
    }
  } else {
    // For others, e.g. fallthrough, conditional branch, assume the exit
    // uses all the registers that are livein to the successor blocks.
    SmallSet<unsigned, 8> Seen;
    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
           SE = BB->succ_end(); SI != SE; ++SI)
      for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
             E = (*SI)->livein_end(); I != E; ++I) {    
        unsigned Reg = *I;
        if (Seen.insert(Reg))
          Uses[Reg].push_back(&ExitSU);
      }
  }
}

void ScheduleDAGInstrs::BuildSchedGraph(AliasAnalysis *AA) {
  // We'll be allocating one SUnit for each instruction, plus one for
  // the region exit node.
  SUnits.reserve(BB->size());

  // We build scheduling units by walking a block's instruction list from bottom
  // to top.

  // Remember where a generic side-effecting instruction is as we procede.
  SUnit *BarrierChain = 0, *AliasChain = 0;

  // Memory references to specific known memory locations are tracked
  // so that they can be given more precise dependencies. We track
  // separately the known memory locations that may alias and those
  // that are known not to alias
  std::map<const Value *, SUnit *> AliasMemDefs, NonAliasMemDefs;
  std::map<const Value *, std::vector<SUnit *> > AliasMemUses, NonAliasMemUses;

  // Keep track of dangling debug references to registers.
  std::vector<std::pair<MachineInstr*, unsigned> >
    DanglingDebugValue(TRI->getNumRegs(),
    std::make_pair(static_cast<MachineInstr*>(0), 0));

  // Check to see if the scheduler cares about latencies.
  bool UnitLatencies = ForceUnitLatencies();

  // Ask the target if address-backscheduling is desirable, and if so how much.
  const TargetSubtarget &ST = TM.getSubtarget<TargetSubtarget>();
  unsigned SpecialAddressLatency = ST.getSpecialAddressLatency();

  // Remove any stale debug info; sometimes BuildSchedGraph is called again
  // without emitting the info from the previous call.
  DbgValueVec.clear();

  // Model data dependencies between instructions being scheduled and the
  // ExitSU.
  AddSchedBarrierDeps();

  // Walk the list of instructions, from bottom moving up.
  for (MachineBasicBlock::iterator MII = InsertPos, MIE = Begin;
       MII != MIE; --MII) {
    MachineInstr *MI = prior(MII);
    // DBG_VALUE does not have SUnit's built, so just remember these for later
    // reinsertion.
    if (MI->isDebugValue()) {
      if (MI->getNumOperands()==3 && MI->getOperand(0).isReg() &&
          MI->getOperand(0).getReg())
        DanglingDebugValue[MI->getOperand(0).getReg()] =
             std::make_pair(MI, DbgValueVec.size());
      DbgValueVec.push_back(MI);
      continue;
    }
    const TargetInstrDesc &TID = MI->getDesc();
    assert(!TID.isTerminator() && !MI->isLabel() &&
           "Cannot schedule terminators or labels!");
    // Create the SUnit for this MI.
    SUnit *SU = NewSUnit(MI);
    SU->isCall = TID.isCall();
    SU->isCommutable = TID.isCommutable();

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

      if (MO.isDef() && DanglingDebugValue[Reg].first!=0) {
        SU->DbgInstrList.push_back(DanglingDebugValue[Reg].first);
        DbgValueVec[DanglingDebugValue[Reg].second] = 0;
        DanglingDebugValue[Reg] = std::make_pair((MachineInstr*)0, 0);
      }

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
        if (DefSU == &ExitSU)
          continue;
        if (DefSU != SU &&
            (Kind != SDep::Output || !MO.isDead() ||
             !DefSU->getInstr()->registerDefIsDead(Reg)))
          DefSU->addPred(SDep(SU, Kind, AOLatency, /*Reg=*/Reg));
      }
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        std::vector<SUnit *> &DefList = Defs[*Alias];
        for (unsigned i = 0, e = DefList.size(); i != e; ++i) {
          SUnit *DefSU = DefList[i];
          if (DefSU == &ExitSU)
            continue;
          if (DefSU != SU &&
              (Kind != SDep::Output || !MO.isDead() ||
               !DefSU->getInstr()->registerDefIsDead(*Alias)))
            DefSU->addPred(SDep(SU, Kind, AOLatency, /*Reg=*/ *Alias));
        }
      }

      if (MO.isDef()) {
        // Add any data dependencies.
        unsigned DataLatency = SU->Latency;
        for (unsigned i = 0, e = UseList.size(); i != e; ++i) {
          SUnit *UseSU = UseList[i];
          if (UseSU == SU)
            continue;
          unsigned LDataLatency = DataLatency;
          // Optionally add in a special extra latency for nodes that
          // feed addresses.
          // TODO: Do this for register aliases too.
          // TODO: Perhaps we should get rid of
          // SpecialAddressLatency and just move this into
          // adjustSchedDependency for the targets that care about it.
          if (SpecialAddressLatency != 0 && !UnitLatencies &&
              UseSU != &ExitSU) {
            MachineInstr *UseMI = UseSU->getInstr();
            const TargetInstrDesc &UseTID = UseMI->getDesc();
            int RegUseIndex = UseMI->findRegisterUseOperandIdx(Reg);
            assert(RegUseIndex >= 0 && "UseMI doesn's use register!");
            if (RegUseIndex >= 0 &&
                (UseTID.mayLoad() || UseTID.mayStore()) &&
                (unsigned)RegUseIndex < UseTID.getNumOperands() &&
                UseTID.OpInfo[RegUseIndex].isLookupPtrRegClass())
              LDataLatency += SpecialAddressLatency;
          }
          // Adjust the dependence latency using operand def/use
          // information (if any), and then allow the target to
          // perform its own adjustments.
          const SDep& dep = SDep(SU, SDep::Data, LDataLatency, Reg);
          if (!UnitLatencies) {
            ComputeOperandLatency(SU, UseSU, const_cast<SDep &>(dep));
            ST.adjustSchedDependency(SU, UseSU, const_cast<SDep &>(dep));
          }
          UseSU->addPred(dep);
        }
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          std::vector<SUnit *> &UseList = Uses[*Alias];
          for (unsigned i = 0, e = UseList.size(); i != e; ++i) {
            SUnit *UseSU = UseList[i];
            if (UseSU == SU)
              continue;
            const SDep& dep = SDep(SU, SDep::Data, DataLatency, *Alias);
            if (!UnitLatencies) {
              ComputeOperandLatency(SU, UseSU, const_cast<SDep &>(dep));
              ST.adjustSchedDependency(SU, UseSU, const_cast<SDep &>(dep));
            }
            UseSU->addPred(dep);
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
    // Chain dependencies used to enforce memory order should have
    // latency of 0 (except for true dependency of Store followed by
    // aliased Load... we estimate that with a single cycle of latency
    // assuming the hardware will bypass)
    // Note that isStoreToStackSlot and isLoadFromStackSLot are not usable
    // after stack slots are lowered to actual addresses.
    // TODO: Use an AliasAnalysis and do real alias-analysis queries, and
    // produce more precise dependence information.
#define STORE_LOAD_LATENCY 1
    unsigned TrueMemOrderLatency = 0;
    if (TID.isCall() || TID.hasUnmodeledSideEffects() ||
        (MI->hasVolatileMemoryRef() && 
         (!TID.mayLoad() || !MI->isInvariantLoad(AA)))) {
      // Be conservative with these and add dependencies on all memory
      // references, even those that are known to not alias.
      for (std::map<const Value *, SUnit *>::iterator I = 
             NonAliasMemDefs.begin(), E = NonAliasMemDefs.end(); I != E; ++I) {
        I->second->addPred(SDep(SU, SDep::Order, /*Latency=*/0));
      }
      for (std::map<const Value *, std::vector<SUnit *> >::iterator I =
             NonAliasMemUses.begin(), E = NonAliasMemUses.end(); I != E; ++I) {
        for (unsigned i = 0, e = I->second.size(); i != e; ++i)
          I->second[i]->addPred(SDep(SU, SDep::Order, TrueMemOrderLatency));
      }
      NonAliasMemDefs.clear();
      NonAliasMemUses.clear();
      // Add SU to the barrier chain.
      if (BarrierChain)
        BarrierChain->addPred(SDep(SU, SDep::Order, /*Latency=*/0));
      BarrierChain = SU;

      // fall-through
    new_alias_chain:
      // Chain all possibly aliasing memory references though SU.
      if (AliasChain)
        AliasChain->addPred(SDep(SU, SDep::Order, /*Latency=*/0));
      AliasChain = SU;
      for (unsigned k = 0, m = PendingLoads.size(); k != m; ++k)
        PendingLoads[k]->addPred(SDep(SU, SDep::Order, TrueMemOrderLatency));
      for (std::map<const Value *, SUnit *>::iterator I = AliasMemDefs.begin(),
           E = AliasMemDefs.end(); I != E; ++I) {
        I->second->addPred(SDep(SU, SDep::Order, /*Latency=*/0));
      }
      for (std::map<const Value *, std::vector<SUnit *> >::iterator I =
           AliasMemUses.begin(), E = AliasMemUses.end(); I != E; ++I) {
        for (unsigned i = 0, e = I->second.size(); i != e; ++i)
          I->second[i]->addPred(SDep(SU, SDep::Order, TrueMemOrderLatency));
      }
      PendingLoads.clear();
      AliasMemDefs.clear();
      AliasMemUses.clear();
    } else if (TID.mayStore()) {
      bool MayAlias = true;
      TrueMemOrderLatency = STORE_LOAD_LATENCY;
      if (const Value *V = getUnderlyingObjectForInstr(MI, MFI, MayAlias)) {
        // A store to a specific PseudoSourceValue. Add precise dependencies.
        // Record the def in MemDefs, first adding a dep if there is
        // an existing def.
        std::map<const Value *, SUnit *>::iterator I = 
          ((MayAlias) ? AliasMemDefs.find(V) : NonAliasMemDefs.find(V));
        std::map<const Value *, SUnit *>::iterator IE = 
          ((MayAlias) ? AliasMemDefs.end() : NonAliasMemDefs.end());
        if (I != IE) {
          I->second->addPred(SDep(SU, SDep::Order, /*Latency=*/0, /*Reg=*/0,
                                  /*isNormalMemory=*/true));
          I->second = SU;
        } else {
          if (MayAlias)
            AliasMemDefs[V] = SU;
          else
            NonAliasMemDefs[V] = SU;
        }
        // Handle the uses in MemUses, if there are any.
        std::map<const Value *, std::vector<SUnit *> >::iterator J =
          ((MayAlias) ? AliasMemUses.find(V) : NonAliasMemUses.find(V));
        std::map<const Value *, std::vector<SUnit *> >::iterator JE =
          ((MayAlias) ? AliasMemUses.end() : NonAliasMemUses.end());
        if (J != JE) {
          for (unsigned i = 0, e = J->second.size(); i != e; ++i)
            J->second[i]->addPred(SDep(SU, SDep::Order, TrueMemOrderLatency,
                                       /*Reg=*/0, /*isNormalMemory=*/true));
          J->second.clear();
        }
        if (MayAlias) {
          // Add dependencies from all the PendingLoads, i.e. loads
          // with no underlying object.
          for (unsigned k = 0, m = PendingLoads.size(); k != m; ++k)
            PendingLoads[k]->addPred(SDep(SU, SDep::Order, TrueMemOrderLatency));
          // Add dependence on alias chain, if needed.
          if (AliasChain)
            AliasChain->addPred(SDep(SU, SDep::Order, /*Latency=*/0));
        }
        // Add dependence on barrier chain, if needed.
        if (BarrierChain)
          BarrierChain->addPred(SDep(SU, SDep::Order, /*Latency=*/0));
      } else {
        // Treat all other stores conservatively.
        goto new_alias_chain;
      }

      if (!ExitSU.isPred(SU))
        // Push store's up a bit to avoid them getting in between cmp
        // and branches.
        ExitSU.addPred(SDep(SU, SDep::Order, 0,
                            /*Reg=*/0, /*isNormalMemory=*/false,
                            /*isMustAlias=*/false,
                            /*isArtificial=*/true));
    } else if (TID.mayLoad()) {
      bool MayAlias = true;
      TrueMemOrderLatency = 0;
      if (MI->isInvariantLoad(AA)) {
        // Invariant load, no chain dependencies needed!
      } else {
        if (const Value *V = 
            getUnderlyingObjectForInstr(MI, MFI, MayAlias)) {
          // A load from a specific PseudoSourceValue. Add precise dependencies.
          std::map<const Value *, SUnit *>::iterator I = 
            ((MayAlias) ? AliasMemDefs.find(V) : NonAliasMemDefs.find(V));
          std::map<const Value *, SUnit *>::iterator IE = 
            ((MayAlias) ? AliasMemDefs.end() : NonAliasMemDefs.end());
          if (I != IE)
            I->second->addPred(SDep(SU, SDep::Order, /*Latency=*/0, /*Reg=*/0,
                                    /*isNormalMemory=*/true));
          if (MayAlias)
            AliasMemUses[V].push_back(SU);
          else 
            NonAliasMemUses[V].push_back(SU);
        } else {
          // A load with no underlying object. Depend on all
          // potentially aliasing stores.
          for (std::map<const Value *, SUnit *>::iterator I = 
                 AliasMemDefs.begin(), E = AliasMemDefs.end(); I != E; ++I)
            I->second->addPred(SDep(SU, SDep::Order, /*Latency=*/0));
          
          PendingLoads.push_back(SU);
          MayAlias = true;
        }
        
        // Add dependencies on alias and barrier chains, if needed.
        if (MayAlias && AliasChain)
          AliasChain->addPred(SDep(SU, SDep::Order, /*Latency=*/0));
        if (BarrierChain)
          BarrierChain->addPred(SDep(SU, SDep::Order, /*Latency=*/0));
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
  // Compute the latency for the node.
  if (!InstrItins || InstrItins->isEmpty()) {
    SU->Latency = 1;

    // Simplistic target-independent heuristic: assume that loads take
    // extra time.
    if (SU->getInstr()->getDesc().mayLoad())
      SU->Latency += 2;
  } else {
    SU->Latency = TII->getInstrLatency(InstrItins, SU->getInstr());
  }
}

void ScheduleDAGInstrs::ComputeOperandLatency(SUnit *Def, SUnit *Use, 
                                              SDep& dep) const {
  if (!InstrItins || InstrItins->isEmpty())
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
    const MachineOperand &MO = DefMI->getOperand(DefIdx);
    if (MO.isReg() && MO.isImplicit() &&
        DefIdx >= (int)DefMI->getDesc().getNumOperands()) {
      // This is an implicit def, getOperandLatency() won't return the correct
      // latency. e.g.
      //   %D6<def>, %D7<def> = VLD1q16 %R2<kill>, 0, ..., %Q3<imp-def>
      //   %Q1<def> = VMULv8i16 %Q1<kill>, %Q3<kill>, ...
      // What we want is to compute latency between def of %D6/%D7 and use of
      // %Q3 instead.
      DefIdx = DefMI->findRegisterDefOperandIdx(Reg, false, true, TRI);
    }
    MachineInstr *UseMI = Use->getInstr();
    // For all uses of the register, calculate the maxmimum latency
    int Latency = -1;
    if (UseMI) {
      for (unsigned i = 0, e = UseMI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = UseMI->getOperand(i);
        if (!MO.isReg() || !MO.isUse())
          continue;
        unsigned MOReg = MO.getReg();
        if (MOReg != Reg)
          continue;

        int UseCycle = TII->getOperandLatency(InstrItins, DefMI, DefIdx,
                                              UseMI, i);
        Latency = std::max(Latency, UseCycle);
      }
    } else {
      // UseMI is null, then it must be a scheduling barrier.
      if (!InstrItins || InstrItins->isEmpty())
        return;
      unsigned DefClass = DefMI->getDesc().getSchedClass();
      Latency = InstrItins->getOperandCycle(DefClass, DefIdx);
    }

    // If we found a latency, then replace the existing dependence latency.
    if (Latency >= 0)
      dep.setLatency(Latency);
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
MachineBasicBlock *ScheduleDAGInstrs::EmitSchedule() {
  // For MachineInstr-based scheduling, we're rescheduling the instructions in
  // the block, so start by removing them from the block.
  while (Begin != InsertPos) {
    MachineBasicBlock::iterator I = Begin;
    ++Begin;
    BB->remove(I);
  }

  // First reinsert any remaining debug_values; these are either constants,
  // or refer to live-in registers.  The beginning of the block is the right
  // place for the latter.  The former might reasonably be placed elsewhere
  // using some kind of ordering algorithm, but right now it doesn't matter.
  for (int i = DbgValueVec.size()-1; i>=0; --i)
    if (DbgValueVec[i])
      BB->insert(InsertPos, DbgValueVec[i]);

  // Then re-insert them according to the given schedule.
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    SUnit *SU = Sequence[i];
    if (!SU) {
      // Null SUnit* is a noop.
      EmitNoop();
      continue;
    }

    BB->insert(InsertPos, SU->getInstr());
    for (unsigned i = 0, e = SU->DbgInstrList.size() ; i < e ; ++i)
      BB->insert(InsertPos, SU->DbgInstrList[i]);
  }

  // Update the Begin iterator, as the first instruction in the block
  // may have been scheduled later.
  if (!DbgValueVec.empty()) {
    for (int i = DbgValueVec.size()-1; i>=0; --i)
      if (DbgValueVec[i]!=0) {
        Begin = DbgValueVec[DbgValueVec.size()-1];
        break;
      }
  } else if (!Sequence.empty())
    Begin = Sequence[0]->getInstr();

  DbgValueVec.clear();
  return BB;
}
