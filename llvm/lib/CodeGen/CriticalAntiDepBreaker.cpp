//===----- CriticalAntiDepBreaker.cpp - Anti-dep breaker -------- ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CriticalAntiDepBreaker class, which
// implements register anti-dependence breaking along a blocks
// critical path during post-RA scheduler.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "post-RA-sched"
#include "CriticalAntiDepBreaker.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

CriticalAntiDepBreaker::
CriticalAntiDepBreaker(MachineFunction& MFi) : 
  AntiDepBreaker(), MF(MFi),
  MRI(MF.getRegInfo()),
  TRI(MF.getTarget().getRegisterInfo()),
  AllocatableSet(TRI->getAllocatableSet(MF))
{
}

CriticalAntiDepBreaker::~CriticalAntiDepBreaker() {
}

void CriticalAntiDepBreaker::StartBlock(MachineBasicBlock *BB) {
  // Clear out the register class data.
  std::fill(Classes, array_endof(Classes),
            static_cast<const TargetRegisterClass *>(0));

  // Initialize the indices to indicate that no registers are live.
  const unsigned BBSize = BB->size();
  for (unsigned i = 0; i < TRI->getNumRegs(); ++i) {
    KillIndices[i] = ~0u;
    DefIndices[i] = BBSize;
  }

  // Clear "do not change" set.
  KeepRegs.clear();

  bool IsReturnBlock = (!BB->empty() && BB->back().getDesc().isReturn());

  // Determine the live-out physregs for this block.
  if (IsReturnBlock) {
    // In a return block, examine the function live-out regs.
    for (MachineRegisterInfo::liveout_iterator I = MRI.liveout_begin(),
         E = MRI.liveout_end(); I != E; ++I) {
      unsigned Reg = *I;
      Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
      KillIndices[Reg] = BB->size();
      DefIndices[Reg] = ~0u;
      // Repeat, for all aliases.
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        unsigned AliasReg = *Alias;
        Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
        KillIndices[AliasReg] = BB->size();
        DefIndices[AliasReg] = ~0u;
      }
    }
  } else {
    // In a non-return block, examine the live-in regs of all successors.
    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
         SE = BB->succ_end(); SI != SE; ++SI)
      for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
           E = (*SI)->livein_end(); I != E; ++I) {
        unsigned Reg = *I;
        Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
        KillIndices[Reg] = BB->size();
        DefIndices[Reg] = ~0u;
        // Repeat, for all aliases.
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          unsigned AliasReg = *Alias;
          Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
          KillIndices[AliasReg] = BB->size();
          DefIndices[AliasReg] = ~0u;
        }
      }
  }

  // Mark live-out callee-saved registers. In a return block this is
  // all callee-saved registers. In non-return this is any
  // callee-saved register that is not saved in the prolog.
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  BitVector Pristine = MFI->getPristineRegs(BB);
  for (const unsigned *I = TRI->getCalleeSavedRegs(); *I; ++I) {
    unsigned Reg = *I;
    if (!IsReturnBlock && !Pristine.test(Reg)) continue;
    Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
    KillIndices[Reg] = BB->size();
    DefIndices[Reg] = ~0u;
    // Repeat, for all aliases.
    for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
      unsigned AliasReg = *Alias;
      Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
      KillIndices[AliasReg] = BB->size();
      DefIndices[AliasReg] = ~0u;
    }
  }
}

void CriticalAntiDepBreaker::FinishBlock() {
  RegRefs.clear();
  KeepRegs.clear();
}

void CriticalAntiDepBreaker::Observe(MachineInstr *MI, unsigned Count,
                                     unsigned InsertPosIndex) {
  if (MI->isDebugValue())
    return;
  assert(Count < InsertPosIndex && "Instruction index out of expected range!");

  // Any register which was defined within the previous scheduling region
  // may have been rescheduled and its lifetime may overlap with registers
  // in ways not reflected in our current liveness state. For each such
  // register, adjust the liveness state to be conservatively correct.
  for (unsigned Reg = 0; Reg != TRI->getNumRegs(); ++Reg)
    if (DefIndices[Reg] < InsertPosIndex && DefIndices[Reg] >= Count) {
      assert(KillIndices[Reg] == ~0u && "Clobbered register is live!");
      // Mark this register to be non-renamable.
      Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
      // Move the def index to the end of the previous region, to reflect
      // that the def could theoretically have been scheduled at the end.
      DefIndices[Reg] = InsertPosIndex;
    }

  PrescanInstruction(MI);
  ScanInstruction(MI, Count);
}

/// CriticalPathStep - Return the next SUnit after SU on the bottom-up
/// critical path.
static const SDep *CriticalPathStep(const SUnit *SU) {
  const SDep *Next = 0;
  unsigned NextDepth = 0;
  // Find the predecessor edge with the greatest depth.
  for (SUnit::const_pred_iterator P = SU->Preds.begin(), PE = SU->Preds.end();
       P != PE; ++P) {
    const SUnit *PredSU = P->getSUnit();
    unsigned PredLatency = P->getLatency();
    unsigned PredTotalLatency = PredSU->getDepth() + PredLatency;
    // In the case of a latency tie, prefer an anti-dependency edge over
    // other types of edges.
    if (NextDepth < PredTotalLatency ||
        (NextDepth == PredTotalLatency && P->getKind() == SDep::Anti)) {
      NextDepth = PredTotalLatency;
      Next = &*P;
    }
  }
  return Next;
}

void CriticalAntiDepBreaker::PrescanInstruction(MachineInstr *MI) {
  // Scan the register operands for this instruction and update
  // Classes and RegRefs.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    const TargetRegisterClass *NewRC = 0;
    
    if (i < MI->getDesc().getNumOperands())
      NewRC = MI->getDesc().OpInfo[i].getRegClass(TRI);

    // For now, only allow the register to be changed if its register
    // class is consistent across all uses.
    if (!Classes[Reg] && NewRC)
      Classes[Reg] = NewRC;
    else if (!NewRC || Classes[Reg] != NewRC)
      Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);

    // Now check for aliases.
    for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
      // If an alias of the reg is used during the live range, give up.
      // Note that this allows us to skip checking if AntiDepReg
      // overlaps with any of the aliases, among other things.
      unsigned AliasReg = *Alias;
      if (Classes[AliasReg]) {
        Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
        Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
      }
    }

    // If we're still willing to consider this register, note the reference.
    if (Classes[Reg] != reinterpret_cast<TargetRegisterClass *>(-1))
      RegRefs.insert(std::make_pair(Reg, &MO));

    // It's not safe to change register allocation for source operands of
    // that have special allocation requirements.
    if (MO.isUse() && MI->getDesc().hasExtraSrcRegAllocReq()) {
      if (KeepRegs.insert(Reg)) {
        for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
             *Subreg; ++Subreg)
          KeepRegs.insert(*Subreg);
      }
    }
  }
}

void CriticalAntiDepBreaker::ScanInstruction(MachineInstr *MI,
                                             unsigned Count) {
  // Update liveness.
  // Proceding upwards, registers that are defed but not used in this
  // instruction are now dead.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    if (!MO.isDef()) continue;
    // Ignore two-addr defs.
    if (MI->isRegTiedToUseOperand(i)) continue;

    DefIndices[Reg] = Count;
    KillIndices[Reg] = ~0u;
    assert(((KillIndices[Reg] == ~0u) !=
            (DefIndices[Reg] == ~0u)) &&
           "Kill and Def maps aren't consistent for Reg!");
    KeepRegs.erase(Reg);
    Classes[Reg] = 0;
    RegRefs.erase(Reg);
    // Repeat, for all subregs.
    for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
         *Subreg; ++Subreg) {
      unsigned SubregReg = *Subreg;
      DefIndices[SubregReg] = Count;
      KillIndices[SubregReg] = ~0u;
      KeepRegs.erase(SubregReg);
      Classes[SubregReg] = 0;
      RegRefs.erase(SubregReg);
    }
    // Conservatively mark super-registers as unusable.
    for (const unsigned *Super = TRI->getSuperRegisters(Reg);
         *Super; ++Super) {
      unsigned SuperReg = *Super;
      Classes[SuperReg] = reinterpret_cast<TargetRegisterClass *>(-1);
    }
  }
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    if (!MO.isUse()) continue;

    const TargetRegisterClass *NewRC = 0;
    if (i < MI->getDesc().getNumOperands())
      NewRC = MI->getDesc().OpInfo[i].getRegClass(TRI);

    // For now, only allow the register to be changed if its register
    // class is consistent across all uses.
    if (!Classes[Reg] && NewRC)
      Classes[Reg] = NewRC;
    else if (!NewRC || Classes[Reg] != NewRC)
      Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);

    RegRefs.insert(std::make_pair(Reg, &MO));

    // It wasn't previously live but now it is, this is a kill.
    if (KillIndices[Reg] == ~0u) {
      KillIndices[Reg] = Count;
      DefIndices[Reg] = ~0u;
          assert(((KillIndices[Reg] == ~0u) !=
                  (DefIndices[Reg] == ~0u)) &&
               "Kill and Def maps aren't consistent for Reg!");
    }
    // Repeat, for all aliases.
    for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
      unsigned AliasReg = *Alias;
      if (KillIndices[AliasReg] == ~0u) {
        KillIndices[AliasReg] = Count;
        DefIndices[AliasReg] = ~0u;
      }
    }
  }
}

unsigned
CriticalAntiDepBreaker::findSuitableFreeRegister(MachineInstr *MI,
                                                 unsigned AntiDepReg,
                                                 unsigned LastNewReg,
                                                 const TargetRegisterClass *RC)
{
  for (TargetRegisterClass::iterator R = RC->allocation_order_begin(MF),
       RE = RC->allocation_order_end(MF); R != RE; ++R) {
    unsigned NewReg = *R;
    // Don't replace a register with itself.
    if (NewReg == AntiDepReg) continue;
    // Don't replace a register with one that was recently used to repair
    // an anti-dependence with this AntiDepReg, because that would
    // re-introduce that anti-dependence.
    if (NewReg == LastNewReg) continue;
    // If the instruction already has a def of the NewReg, it's not suitable.
    // For example, Instruction with multiple definitions can result in this
    // condition.
    if (MI->modifiesRegister(NewReg, TRI)) continue;
    // If NewReg is dead and NewReg's most recent def is not before
    // AntiDepReg's kill, it's safe to replace AntiDepReg with NewReg.
    assert(((KillIndices[AntiDepReg] == ~0u) != (DefIndices[AntiDepReg] == ~0u))
           && "Kill and Def maps aren't consistent for AntiDepReg!");
    assert(((KillIndices[NewReg] == ~0u) != (DefIndices[NewReg] == ~0u))
           && "Kill and Def maps aren't consistent for NewReg!");
    if (KillIndices[NewReg] != ~0u ||
        Classes[NewReg] == reinterpret_cast<TargetRegisterClass *>(-1) ||
        KillIndices[AntiDepReg] > DefIndices[NewReg])
      continue;
    return NewReg;
  }

  // No registers are free and available!
  return 0;
}

unsigned CriticalAntiDepBreaker::
BreakAntiDependencies(const std::vector<SUnit>& SUnits,
                      MachineBasicBlock::iterator Begin,
                      MachineBasicBlock::iterator End,
                      unsigned InsertPosIndex) {
  // The code below assumes that there is at least one instruction,
  // so just duck out immediately if the block is empty.
  if (SUnits.empty()) return 0;

  // Find the node at the bottom of the critical path.
  const SUnit *Max = 0;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    const SUnit *SU = &SUnits[i];
    if (!Max || SU->getDepth() + SU->Latency > Max->getDepth() + Max->Latency)
      Max = SU;
  }

#ifndef NDEBUG
  {
    DEBUG(dbgs() << "Critical path has total latency "
          << (Max->getDepth() + Max->Latency) << "\n");
    DEBUG(dbgs() << "Available regs:");
    for (unsigned Reg = 0; Reg < TRI->getNumRegs(); ++Reg) {
      if (KillIndices[Reg] == ~0u)
        DEBUG(dbgs() << " " << TRI->getName(Reg));
    }
    DEBUG(dbgs() << '\n');
  }
#endif

  // Track progress along the critical path through the SUnit graph as we walk
  // the instructions.
  const SUnit *CriticalPathSU = Max;
  MachineInstr *CriticalPathMI = CriticalPathSU->getInstr();

  // Consider this pattern:
  //   A = ...
  //   ... = A
  //   A = ...
  //   ... = A
  //   A = ...
  //   ... = A
  //   A = ...
  //   ... = A
  // There are three anti-dependencies here, and without special care,
  // we'd break all of them using the same register:
  //   A = ...
  //   ... = A
  //   B = ...
  //   ... = B
  //   B = ...
  //   ... = B
  //   B = ...
  //   ... = B
  // because at each anti-dependence, B is the first register that
  // isn't A which is free.  This re-introduces anti-dependencies
  // at all but one of the original anti-dependencies that we were
  // trying to break.  To avoid this, keep track of the most recent
  // register that each register was replaced with, avoid
  // using it to repair an anti-dependence on the same register.
  // This lets us produce this:
  //   A = ...
  //   ... = A
  //   B = ...
  //   ... = B
  //   C = ...
  //   ... = C
  //   B = ...
  //   ... = B
  // This still has an anti-dependence on B, but at least it isn't on the
  // original critical path.
  //
  // TODO: If we tracked more than one register here, we could potentially
  // fix that remaining critical edge too. This is a little more involved,
  // because unlike the most recent register, less recent registers should
  // still be considered, though only if no other registers are available.
  unsigned LastNewReg[TargetRegisterInfo::FirstVirtualRegister] = {};

  // Attempt to break anti-dependence edges on the critical path. Walk the
  // instructions from the bottom up, tracking information about liveness
  // as we go to help determine which registers are available.
  unsigned Broken = 0;
  unsigned Count = InsertPosIndex - 1;
  for (MachineBasicBlock::iterator I = End, E = Begin;
       I != E; --Count) {
    MachineInstr *MI = --I;
    if (MI->isDebugValue())
      continue;

    // Check if this instruction has a dependence on the critical path that
    // is an anti-dependence that we may be able to break. If it is, set
    // AntiDepReg to the non-zero register associated with the anti-dependence.
    //
    // We limit our attention to the critical path as a heuristic to avoid
    // breaking anti-dependence edges that aren't going to significantly
    // impact the overall schedule. There are a limited number of registers
    // and we want to save them for the important edges.
    // 
    // TODO: Instructions with multiple defs could have multiple
    // anti-dependencies. The current code here only knows how to break one
    // edge per instruction. Note that we'd have to be able to break all of
    // the anti-dependencies in an instruction in order to be effective.
    unsigned AntiDepReg = 0;
    if (MI == CriticalPathMI) {
      if (const SDep *Edge = CriticalPathStep(CriticalPathSU)) {
        const SUnit *NextSU = Edge->getSUnit();

        // Only consider anti-dependence edges.
        if (Edge->getKind() == SDep::Anti) {
          AntiDepReg = Edge->getReg();
          assert(AntiDepReg != 0 && "Anti-dependence on reg0?");
          if (!AllocatableSet.test(AntiDepReg))
            // Don't break anti-dependencies on non-allocatable registers.
            AntiDepReg = 0;
          else if (KeepRegs.count(AntiDepReg))
            // Don't break anti-dependencies if an use down below requires
            // this exact register.
            AntiDepReg = 0;
          else {
            // If the SUnit has other dependencies on the SUnit that it
            // anti-depends on, don't bother breaking the anti-dependency
            // since those edges would prevent such units from being
            // scheduled past each other regardless.
            //
            // Also, if there are dependencies on other SUnits with the
            // same register as the anti-dependency, don't attempt to
            // break it.
            for (SUnit::const_pred_iterator P = CriticalPathSU->Preds.begin(),
                 PE = CriticalPathSU->Preds.end(); P != PE; ++P)
              if (P->getSUnit() == NextSU ?
                    (P->getKind() != SDep::Anti || P->getReg() != AntiDepReg) :
                    (P->getKind() == SDep::Data && P->getReg() == AntiDepReg)) {
                AntiDepReg = 0;
                break;
              }
          }
        }
        CriticalPathSU = NextSU;
        CriticalPathMI = CriticalPathSU->getInstr();
      } else {
        // We've reached the end of the critical path.
        CriticalPathSU = 0;
        CriticalPathMI = 0;
      }
    }

    PrescanInstruction(MI);

    if (MI->getDesc().hasExtraDefRegAllocReq())
      // If this instruction's defs have special allocation requirement, don't
      // break this anti-dependency.
      AntiDepReg = 0;
    else if (AntiDepReg) {
      // If this instruction has a use of AntiDepReg, breaking it
      // is invalid.
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg()) continue;
        unsigned Reg = MO.getReg();
        if (Reg == 0) continue;
        if (MO.isUse() && AntiDepReg == Reg) {
          AntiDepReg = 0;
          break;
        }
      }
    }

    // Determine AntiDepReg's register class, if it is live and is
    // consistently used within a single class.
    const TargetRegisterClass *RC = AntiDepReg != 0 ? Classes[AntiDepReg] : 0;
    assert((AntiDepReg == 0 || RC != NULL) &&
           "Register should be live if it's causing an anti-dependence!");
    if (RC == reinterpret_cast<TargetRegisterClass *>(-1))
      AntiDepReg = 0;

    // Look for a suitable register to use to break the anti-depenence.
    //
    // TODO: Instead of picking the first free register, consider which might
    // be the best.
    if (AntiDepReg != 0) {
      if (unsigned NewReg = findSuitableFreeRegister(MI, AntiDepReg,
                                                     LastNewReg[AntiDepReg],
                                                     RC)) {
        DEBUG(dbgs() << "Breaking anti-dependence edge on "
              << TRI->getName(AntiDepReg)
              << " with " << RegRefs.count(AntiDepReg) << " references"
              << " using " << TRI->getName(NewReg) << "!\n");

        // Update the references to the old register to refer to the new
        // register.
        std::pair<std::multimap<unsigned, MachineOperand *>::iterator,
                  std::multimap<unsigned, MachineOperand *>::iterator>
           Range = RegRefs.equal_range(AntiDepReg);
        for (std::multimap<unsigned, MachineOperand *>::iterator
             Q = Range.first, QE = Range.second; Q != QE; ++Q)
          Q->second->setReg(NewReg);

        // We just went back in time and modified history; the
        // liveness information for the anti-depenence reg is now
        // inconsistent. Set the state as if it were dead.
        Classes[NewReg] = Classes[AntiDepReg];
        DefIndices[NewReg] = DefIndices[AntiDepReg];
        KillIndices[NewReg] = KillIndices[AntiDepReg];
        assert(((KillIndices[NewReg] == ~0u) !=
                (DefIndices[NewReg] == ~0u)) &&
             "Kill and Def maps aren't consistent for NewReg!");

        Classes[AntiDepReg] = 0;
        DefIndices[AntiDepReg] = KillIndices[AntiDepReg];
        KillIndices[AntiDepReg] = ~0u;
        assert(((KillIndices[AntiDepReg] == ~0u) !=
                (DefIndices[AntiDepReg] == ~0u)) &&
             "Kill and Def maps aren't consistent for AntiDepReg!");

        RegRefs.erase(AntiDepReg);
        LastNewReg[AntiDepReg] = NewReg;
        ++Broken;
      }
    }

    ScanInstruction(MI, Count);
  }

  return Broken;
}
