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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumStalls, "Number of pipeline stalls");

namespace {
  class VISIBILITY_HIDDEN PostRAScheduler : public MachineFunctionPass {
  public:
    static char ID;
    PostRAScheduler() : MachineFunctionPass(&ID) {}
  private:
    MachineFunction *MF;
    const TargetMachine *TM;
  public:
    const char *getPassName() const {
      return "Post RA top-down list latency scheduler (STUB)";
    }

    bool runOnMachineFunction(MachineFunction &Fn);
  };
  char PostRAScheduler::ID = 0;

  class VISIBILITY_HIDDEN SchedulePostRATDList : public ScheduleDAGInstrs {
  public:
    SchedulePostRATDList(MachineBasicBlock *mbb, const TargetMachine &tm)
      : ScheduleDAGInstrs(mbb, tm) {}
  private:
    MachineFunction *MF;
    const TargetMachine *TM;

    /// AvailableQueue - The priority queue to use for the available SUnits.
    ///
    LatencyPriorityQueue AvailableQueue;
  
    /// PendingQueue - This contains all of the instructions whose operands have
    /// been issued, but their results are not ready yet (due to the latency of
    /// the operation).  Once the operands becomes available, the instruction is
    /// added to the AvailableQueue.
    std::vector<SUnit*> PendingQueue;

  public:
    const char *getPassName() const {
      return "Post RA top-down list latency scheduler (STUB)";
    }

    bool runOnMachineFunction(MachineFunction &Fn);

    void Schedule();

  private:
    void ReleaseSucc(SUnit *SU, SUnit *SuccSU, bool isChain);
    void ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle);
    void ListScheduleTopDown();
  };
}

bool PostRAScheduler::runOnMachineFunction(MachineFunction &Fn) {
  DOUT << "PostRAScheduler\n";
  MF = &Fn;
  TM = &MF->getTarget();

  // Loop over all of the basic blocks
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {

    SchedulePostRATDList Scheduler(MBB, *TM);

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

  AvailableQueue.initNodes(SUnits);
  
  ListScheduleTopDown();
  
  AvailableQueue.releaseState();
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
    
    // If there are no instructions available, don't try to issue anything, and
    // don't advance the hazard recognizer.
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
