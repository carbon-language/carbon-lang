//===----- SchedulePostRAList.cpp - list scheduler ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a top-down list scheduler, using standard algorithms.
// The basic approach uses a priority queue of available nodes to schedule.
// One at a time, nodes are taken from the priority queue (thus in priority
// order), checked for legality to schedule, and emitted if legal.
//
// Nodes may not be legal to schedule either due to structural hazards (e.g.
// pipeline or resource constraints) or because an input to the instruction has
// not completed execution.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "post-RA-sched"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/LatencyPriorityQueue.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <climits>
using namespace llvm;

STATISTIC(NumStalls, "Number of pipeline stalls");

static cl::opt<bool>
EnableAntiDepBreaking("break-anti-dependencies",
                      cl::desc("Break scheduling anti-dependencies"),
                      cl::init(false));

namespace {
  class VISIBILITY_HIDDEN PostRAScheduler : public MachineFunctionPass {
  public:
    static char ID;
    PostRAScheduler() : MachineFunctionPass(&ID) {}

    const char *getPassName() const {
      return "Post RA top-down list latency scheduler";
    }

    bool runOnMachineFunction(MachineFunction &Fn);
  };
  char PostRAScheduler::ID = 0;

  class VISIBILITY_HIDDEN SchedulePostRATDList : public ScheduleDAGInstrs {
    /// AvailableQueue - The priority queue to use for the available SUnits.
    ///
    LatencyPriorityQueue AvailableQueue;
  
    /// PendingQueue - This contains all of the instructions whose operands have
    /// been issued, but their results are not ready yet (due to the latency of
    /// the operation).  Once the operands becomes available, the instruction is
    /// added to the AvailableQueue.
    std::vector<SUnit*> PendingQueue;

    /// Topo - A topological ordering for SUnits.
    ScheduleDAGTopologicalSort Topo;

  public:
    SchedulePostRATDList(MachineBasicBlock *mbb, const TargetMachine &tm)
      : ScheduleDAGInstrs(mbb, tm), Topo(SUnits) {}

    void Schedule();

  private:
    void ReleaseSucc(SUnit *SU, SUnit *SuccSU, bool isChain);
    void ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle);
    void ListScheduleTopDown();
    bool BreakAntiDependencies();
  };
}

bool PostRAScheduler::runOnMachineFunction(MachineFunction &Fn) {
  DOUT << "PostRAScheduler\n";

  // Loop over all of the basic blocks
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {

    SchedulePostRATDList Scheduler(MBB, Fn.getTarget());

    Scheduler.Run();

    Scheduler.EmitSchedule();
  }

  return true;
}
  
/// Schedule - Schedule the DAG using list scheduling.
void SchedulePostRATDList::Schedule() {
  DOUT << "********** List Scheduling **********\n";
  
  // Build scheduling units.
  BuildSchedUnits();

  if (EnableAntiDepBreaking) {
    if (BreakAntiDependencies()) {
      // We made changes. Update the dependency graph.
      // Theoretically we could update the graph in place:
      // When a live range is changed to use a different register, remove
      // the def's anti-dependence *and* output-dependence edges due to
      // that register, and add new anti-dependence and output-dependence
      // edges based on the next live range of the register.
      SUnits.clear();
      BuildSchedUnits();
    }
  }

  AvailableQueue.initNodes(SUnits);

  ListScheduleTopDown();
  
  AvailableQueue.releaseState();
}

/// getInstrOperandRegClass - Return register class of the operand of an
/// instruction of the specified TargetInstrDesc.
static const TargetRegisterClass*
getInstrOperandRegClass(const TargetRegisterInfo *TRI,
                        const TargetInstrInfo *TII, const TargetInstrDesc &II,
                        unsigned Op) {
  if (Op >= II.getNumOperands())
    return NULL;
  if (II.OpInfo[Op].isLookupPtrRegClass())
    return TII->getPointerRegClass();
  return TRI->getRegClass(II.OpInfo[Op].RegClass);
}

/// BreakAntiDependencies - Identifiy anti-dependencies along the critical path
/// of the ScheduleDAG and break them by renaming registers.
///
bool SchedulePostRATDList::BreakAntiDependencies() {
  // The code below assumes that there is at least one instruction,
  // so just duck out immediately if the block is empty.
  if (BB->empty()) return false;

  Topo.InitDAGTopologicalSorting();

  // Compute a critical path for the DAG.
  SUnit *Max = 0;
  std::vector<SDep *> CriticalPath(SUnits.size());
  for (ScheduleDAGTopologicalSort::const_iterator I = Topo.begin(),
       E = Topo.end(); I != E; ++I) {
    SUnit *SU = &SUnits[*I];
    for (SUnit::pred_iterator P = SU->Preds.begin(), PE = SU->Preds.end();
         P != PE; ++P) {
      SUnit *PredSU = P->Dep;
      // This assumes that there's no delay for reusing registers.
      unsigned PredLatency = (P->isCtrl && P->Reg != 0) ? 1 : PredSU->Latency;
      unsigned PredTotalLatency = PredSU->CycleBound + PredLatency;
      if (SU->CycleBound < PredTotalLatency ||
          (SU->CycleBound == PredTotalLatency && !P->isAntiDep)) {
        SU->CycleBound = PredTotalLatency;
        CriticalPath[*I] = &*P;
      }
    }
    // Keep track of the node at the end of the critical path.
    if (!Max || SU->CycleBound + SU->Latency > Max->CycleBound + Max->Latency)
      Max = SU;
  }

  DOUT << "Critical path has total latency "
       << (Max ? Max->CycleBound + Max->Latency : 0) << "\n";

  // Walk the critical path from the bottom up. Collect all anti-dependence
  // edges on the critical path. Skip anti-dependencies between SUnits that
  // are connected with other edges, since such units won't be able to be
  // scheduled past each other anyway.
  //
  // The heuristic is that edges on the critical path are more important to
  // break than other edges. And since there are a limited number of
  // registers, we don't want to waste them breaking edges that aren't
  // important.
  // 
  // TODO: Instructions with multiple defs could have multiple
  // anti-dependencies. The current code here only knows how to break one
  // edge per instruction. Note that we'd have to be able to break all of
  // the anti-dependencies in an instruction in order to be effective.
  BitVector AllocatableSet = TRI->getAllocatableSet(*MF);
  DenseMap<MachineInstr *, unsigned> CriticalAntiDeps;
  for (SUnit *SU = Max; CriticalPath[SU->NodeNum];
       SU = CriticalPath[SU->NodeNum]->Dep) {
    SDep *Edge = CriticalPath[SU->NodeNum];
    SUnit *NextSU = Edge->Dep;
    unsigned AntiDepReg = Edge->Reg;
    // Only consider anti-dependence edges.
    if (!Edge->isAntiDep)
      continue;
    assert(AntiDepReg != 0 && "Anti-dependence on reg0?");
    // Don't break anti-dependencies on non-allocatable registers.
    if (!AllocatableSet.test(AntiDepReg))
      continue;
    // If the SUnit has other dependencies on the SUnit that it
    // anti-depends on, don't bother breaking the anti-dependency.
    // Also, if there are dependencies on other SUnits with the
    // same register as the anti-dependency, don't attempt to
    // break it.
    for (SUnit::pred_iterator P = SU->Preds.begin(), PE = SU->Preds.end();
         P != PE; ++P)
      if (P->Dep == NextSU ?
            (!P->isAntiDep || P->Reg != AntiDepReg) :
            (!P->isCtrl && !P->isAntiDep && P->Reg == AntiDepReg)) {
        AntiDepReg = 0;
        break;
      }
    if (AntiDepReg != 0)
      CriticalAntiDeps[SU->getInstr()] = AntiDepReg;
  }

  // For live regs that are only used in one register class in a live range,
  // the register class. If the register is not live, the corresponding value
  // is null. If the register is live but used in multiple register classes,
  // the corresponding value is -1 casted to a pointer.
  const TargetRegisterClass *
    Classes[TargetRegisterInfo::FirstVirtualRegister] = {};

  // Map registers to all their references within a live range.
  std::multimap<unsigned, MachineOperand *> RegRefs;

  // The index of the most recent kill (proceding bottom-up), or -1 if
  // the register is not live.
  unsigned KillIndices[TargetRegisterInfo::FirstVirtualRegister];
  std::fill(KillIndices, array_endof(KillIndices), -1);
  // The index of the most recent def (proceding bottom up), or -1 if
  // the register is live.
  unsigned DefIndices[TargetRegisterInfo::FirstVirtualRegister];
  std::fill(DefIndices, array_endof(DefIndices), BB->size());

  // Determine the live-out physregs for this block.
  if (!BB->empty() && BB->back().getDesc().isReturn())
    // In a return block, examine the function live-out regs.
    for (MachineRegisterInfo::liveout_iterator I = MRI.liveout_begin(),
         E = MRI.liveout_end(); I != E; ++I) {
      unsigned Reg = *I;
      Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
      KillIndices[Reg] = BB->size();
      DefIndices[Reg] = -1;
      // Repeat, for all aliases.
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        unsigned AliasReg = *Alias;
        Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
        KillIndices[AliasReg] = BB->size();
        DefIndices[AliasReg] = -1;
      }
    }
  else
    // In a non-return block, examine the live-in regs of all successors.
    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
         SE = BB->succ_end(); SI != SE; ++SI) 
      for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
           E = (*SI)->livein_end(); I != E; ++I) {
        unsigned Reg = *I;
        Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
        KillIndices[Reg] = BB->size();
        DefIndices[Reg] = -1;
        // Repeat, for all aliases.
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          unsigned AliasReg = *Alias;
          Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
          KillIndices[AliasReg] = BB->size();
          DefIndices[AliasReg] = -1;
        }
      }

  // Consider callee-saved registers as live-out, since we're running after
  // prologue/epilogue insertion so there's no way to add additional
  // saved registers.
  //
  // TODO: If the callee saves and restores these, then we can potentially
  // use them between the save and the restore. To do that, we could scan
  // the exit blocks to see which of these registers are defined.
  // Alternatively, calle-saved registers that aren't saved and restored
  // could be marked live-in in every block.
  for (const unsigned *I = TRI->getCalleeSavedRegs(); *I; ++I) {
    unsigned Reg = *I;
    Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
    KillIndices[Reg] = BB->size();
    DefIndices[Reg] = -1;
    // Repeat, for all aliases.
    for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
      unsigned AliasReg = *Alias;
      Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
      KillIndices[AliasReg] = BB->size();
      DefIndices[AliasReg] = -1;
    }
  }

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
  // register that each register was replaced with, avoid avoid
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
  bool Changed = false;
  unsigned Count = BB->size() - 1;
  for (MachineBasicBlock::reverse_iterator I = BB->rbegin(), E = BB->rend();
       I != E; ++I, --Count) {
    MachineInstr *MI = &*I;

    // Check if this instruction has an anti-dependence that we're
    // interested in.
    DenseMap<MachineInstr *, unsigned>::iterator C = CriticalAntiDeps.find(MI);
    unsigned AntiDepReg = C != CriticalAntiDeps.end() ?
      C->second : 0;

    // Scan the register operands for this instruction and update
    // Classes and RegRefs.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;
      const TargetRegisterClass *NewRC =
        getInstrOperandRegClass(TRI, TII, MI->getDesc(), i);

      // If this instruction has a use of AntiDepReg, breaking it
      // is invalid.
      if (MO.isUse() && AntiDepReg == Reg)
        AntiDepReg = 0;

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
      for (TargetRegisterClass::iterator R = RC->allocation_order_begin(*MF),
           RE = RC->allocation_order_end(*MF); R != RE; ++R) {
        unsigned NewReg = *R;
        // Don't replace a register with itself.
        if (NewReg == AntiDepReg) continue;
        // Don't replace a register with one that was recently used to repair
        // an anti-dependence with this AntiDepReg, because that would
        // re-introduce that anti-dependence.
        if (NewReg == LastNewReg[AntiDepReg]) continue;
        // If NewReg is dead and NewReg's most recent def is not before
        // AntiDepReg's kill, it's safe to replace AntiDepReg with NewReg.
        assert(((KillIndices[AntiDepReg] == -1u) != (DefIndices[AntiDepReg] == -1u)) &&
               "Kill and Def maps aren't consistent for AntiDepReg!");
        assert(((KillIndices[NewReg] == -1u) != (DefIndices[NewReg] == -1u)) &&
               "Kill and Def maps aren't consistent for NewReg!");
        if (KillIndices[NewReg] == -1u &&
            KillIndices[AntiDepReg] <= DefIndices[NewReg]) {
          DOUT << "Breaking anti-dependence edge on reg " << AntiDepReg
               << " with " << RegRefs.count(AntiDepReg) << " references"
               << " with new reg " << NewReg << "!\n";

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

          Classes[AntiDepReg] = 0;
          DefIndices[AntiDepReg] = KillIndices[AntiDepReg];
          KillIndices[AntiDepReg] = -1;

          RegRefs.erase(AntiDepReg);
          Changed = true;
          LastNewReg[AntiDepReg] = NewReg;
          break;
        }
      }
    }

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
      if (MI->isRegReDefinedByTwoAddr(Reg, i)) continue;

      DefIndices[Reg] = Count;
      KillIndices[Reg] = -1;
      Classes[Reg] = 0;
      RegRefs.erase(Reg);
      // Repeat, for all subregs.
      for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
           *Subreg; ++Subreg) {
        unsigned SubregReg = *Subreg;
        DefIndices[SubregReg] = Count;
        KillIndices[SubregReg] = -1;
        Classes[SubregReg] = 0;
        RegRefs.erase(SubregReg);
      }
    }
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;
      if (!MO.isUse()) continue;

      const TargetRegisterClass *NewRC =
        getInstrOperandRegClass(TRI, TII, MI->getDesc(), i);

      // For now, only allow the register to be changed if its register
      // class is consistent across all uses.
      if (!Classes[Reg] && NewRC)
        Classes[Reg] = NewRC;
      else if (!NewRC || Classes[Reg] != NewRC)
        Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);

      RegRefs.insert(std::make_pair(Reg, &MO));

      // It wasn't previously live but now it is, this is a kill.
      if (KillIndices[Reg] == -1u) {
        KillIndices[Reg] = Count;
        DefIndices[Reg] = -1u;
      }
      // Repeat, for all aliases.
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        unsigned AliasReg = *Alias;
        if (KillIndices[AliasReg] == -1u) {
          KillIndices[AliasReg] = Count;
          DefIndices[AliasReg] = -1u;
        }
      }
    }
  }
  assert(Count == -1u && "Count mismatch!");

  return Changed;
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the PendingQueue if the count reaches zero. Also update its cycle bound.
void SchedulePostRATDList::ReleaseSucc(SUnit *SU, SUnit *SuccSU, bool isChain) {
  --SuccSU->NumPredsLeft;
  
#ifndef NDEBUG
  if (SuccSU->NumPredsLeft < 0) {
    cerr << "*** Scheduling failed! ***\n";
    SuccSU->dump(this);
    cerr << " has been released too many times!\n";
    assert(0);
  }
#endif
  
  // Compute how many cycles it will be before this actually becomes
  // available.  This is the max of the start time of all predecessors plus
  // their latencies.
  // If this is a token edge, we don't need to wait for the latency of the
  // preceeding instruction (e.g. a long-latency load) unless there is also
  // some other data dependence.
  unsigned PredDoneCycle = SU->Cycle;
  if (!isChain)
    PredDoneCycle += SU->Latency;
  else if (SU->Latency)
    PredDoneCycle += 1;
  SuccSU->CycleBound = std::max(SuccSU->CycleBound, PredDoneCycle);
  
  if (SuccSU->NumPredsLeft == 0) {
    PendingQueue.push_back(SuccSU);
  }
}

/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void SchedulePostRATDList::ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(this));
  
  Sequence.push_back(SU);
  SU->Cycle = CurCycle;

  // Top down: release successors.
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I)
    ReleaseSucc(SU, I->Dep, I->isCtrl);

  SU->isScheduled = true;
  AvailableQueue.ScheduledNode(SU);
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void SchedulePostRATDList::ListScheduleTopDown() {
  unsigned CurCycle = 0;

  // All leaves to Available queue.
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    // It is available if it has no predecessors.
    if (SUnits[i].Preds.empty()) {
      AvailableQueue.push(&SUnits[i]);
      SUnits[i].isAvailable = true;
    }
  }
  
  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue.empty() || !PendingQueue.empty()) {
    // Check to see if any of the pending instructions are ready to issue.  If
    // so, add them to the available queue.
    for (unsigned i = 0, e = PendingQueue.size(); i != e; ++i) {
      if (PendingQueue[i]->CycleBound == CurCycle) {
        AvailableQueue.push(PendingQueue[i]);
        PendingQueue[i]->isAvailable = true;
        PendingQueue[i] = PendingQueue.back();
        PendingQueue.pop_back();
        --i; --e;
      } else {
        assert(PendingQueue[i]->CycleBound > CurCycle && "Negative latency?");
      }
    }
    
    // If there are no instructions available, don't try to issue anything.
    if (AvailableQueue.empty()) {
      ++CurCycle;
      continue;
    }

    SUnit *FoundSUnit = AvailableQueue.pop();
    
    // If we found a node to schedule, do it now.
    if (FoundSUnit) {
      ScheduleNodeTopDown(FoundSUnit, CurCycle);

      // If this is a pseudo-op node, we don't want to increment the current
      // cycle.
      if (FoundSUnit->Latency)  // Don't increment CurCycle for pseudo-ops!
        ++CurCycle;        
    } else {
      // Otherwise, we have a pipeline stall, but no other problem, just advance
      // the current cycle and try again.
      DOUT << "*** Advancing cycle, no work to do\n";
      ++NumStalls;
      ++CurCycle;
    }
  }

#ifndef NDEBUG
  VerifySchedule(/*isBottomUp=*/false);
#endif
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createPostRAScheduler() {
  return new PostRAScheduler();
}
