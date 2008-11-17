//===---- ScheduleDAGList.cpp - Implement a list scheduler for isel DAG ---===//
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

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/Statistic.h"
#include "LatencyPriorityQueue.h"
#include <climits>
using namespace llvm;

STATISTIC(NumNoops , "Number of noops inserted");
STATISTIC(NumStalls, "Number of pipeline stalls");

static RegisterScheduler
  tdListDAGScheduler("list-td", "Top-down list scheduler",
                     createTDListDAGScheduler);
   
namespace {
//===----------------------------------------------------------------------===//
/// ScheduleDAGList - The actual list scheduler implementation.  This supports
/// top-down scheduling.
///
class VISIBILITY_HIDDEN ScheduleDAGList : public ScheduleDAG {
private:
  /// AvailableQueue - The priority queue to use for the available SUnits.
  ///
  SchedulingPriorityQueue *AvailableQueue;
  
  /// PendingQueue - This contains all of the instructions whose operands have
  /// been issued, but their results are not ready yet (due to the latency of
  /// the operation).  Once the operands becomes available, the instruction is
  /// added to the AvailableQueue.  This keeps track of each SUnit and the
  /// number of cycles left to execute before the operation is available.
  std::vector<std::pair<unsigned, SUnit*> > PendingQueue;

  /// HazardRec - The hazard recognizer to use.
  HazardRecognizer *HazardRec;

public:
  ScheduleDAGList(SelectionDAG *dag, MachineBasicBlock *bb,
                  const TargetMachine &tm,
                  SchedulingPriorityQueue *availqueue,
                  HazardRecognizer *HR)
    : ScheduleDAG(dag, bb, tm),
      AvailableQueue(availqueue), HazardRec(HR) {
    }

  ~ScheduleDAGList() {
    delete HazardRec;
    delete AvailableQueue;
  }

  void Schedule();

private:
  void ReleaseSucc(SUnit *SuccSU, bool isChain);
  void ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle);
  void ListScheduleTopDown();
};
}  // end anonymous namespace

HazardRecognizer::~HazardRecognizer() {}


/// Schedule - Schedule the DAG using list scheduling.
void ScheduleDAGList::Schedule() {
  DOUT << "********** List Scheduling **********\n";
  
  // Build scheduling units.
  BuildSchedUnits();

  AvailableQueue->initNodes(SUnits);
  
  ListScheduleTopDown();
  
  AvailableQueue->releaseState();
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the PendingQueue if the count reaches zero.
void ScheduleDAGList::ReleaseSucc(SUnit *SuccSU, bool isChain) {
  SuccSU->NumPredsLeft--;
  
  assert(SuccSU->NumPredsLeft >= 0 &&
         "List scheduling internal error");
  
  if (SuccSU->NumPredsLeft == 0) {
    // Compute how many cycles it will be before this actually becomes
    // available.  This is the max of the start time of all predecessors plus
    // their latencies.
    unsigned AvailableCycle = 0;
    for (SUnit::pred_iterator I = SuccSU->Preds.begin(),
         E = SuccSU->Preds.end(); I != E; ++I) {
      // If this is a token edge, we don't need to wait for the latency of the
      // preceeding instruction (e.g. a long-latency load) unless there is also
      // some other data dependence.
      SUnit &Pred = *I->Dep;
      unsigned PredDoneCycle = Pred.Cycle;
      if (!I->isCtrl)
        PredDoneCycle += Pred.Latency;
      else if (Pred.Latency)
        PredDoneCycle += 1;

      AvailableCycle = std::max(AvailableCycle, PredDoneCycle);
    }
    
    PendingQueue.push_back(std::make_pair(AvailableCycle, SuccSU));
  }
}

/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGList::ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(DAG));
  
  Sequence.push_back(SU);
  SU->Cycle = CurCycle;
  
  // Top down: release successors.
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I)
    ReleaseSucc(I->Dep, I->isCtrl);
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void ScheduleDAGList::ListScheduleTopDown() {
  unsigned CurCycle = 0;

  // All leaves to Available queue.
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    // It is available if it has no predecessors.
    if (SUnits[i].Preds.empty()) {
      AvailableQueue->push(&SUnits[i]);
      SUnits[i].isAvailable = true;
    }
  }
  
  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  std::vector<SUnit*> NotReady;
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue->empty() || !PendingQueue.empty()) {
    // Check to see if any of the pending instructions are ready to issue.  If
    // so, add them to the available queue.
    for (unsigned i = 0, e = PendingQueue.size(); i != e; ++i) {
      if (PendingQueue[i].first == CurCycle) {
        AvailableQueue->push(PendingQueue[i].second);
        PendingQueue[i].second->isAvailable = true;
        PendingQueue[i] = PendingQueue.back();
        PendingQueue.pop_back();
        --i; --e;
      } else {
        assert(PendingQueue[i].first > CurCycle && "Negative latency?");
      }
    }
    
    // If there are no instructions available, don't try to issue anything, and
    // don't advance the hazard recognizer.
    if (AvailableQueue->empty()) {
      ++CurCycle;
      continue;
    }

    SUnit *FoundSUnit = 0;
    SDNode *FoundNode = 0;
    
    bool HasNoopHazards = false;
    while (!AvailableQueue->empty()) {
      SUnit *CurSUnit = AvailableQueue->pop();
      
      // Get the node represented by this SUnit.
      FoundNode = CurSUnit->getNode();
      
      // If this is a pseudo op, like copyfromreg, look to see if there is a
      // real target node flagged to it.  If so, use the target node.
      while (!FoundNode->isMachineOpcode()) {
        SDNode *N = FoundNode->getFlaggedNode();
        if (!N) break;
        FoundNode = N;
      }
      
      HazardRecognizer::HazardType HT = HazardRec->getHazardType(FoundNode);
      if (HT == HazardRecognizer::NoHazard) {
        FoundSUnit = CurSUnit;
        break;
      }
      
      // Remember if this is a noop hazard.
      HasNoopHazards |= HT == HazardRecognizer::NoopHazard;
      
      NotReady.push_back(CurSUnit);
    }
    
    // Add the nodes that aren't ready back onto the available list.
    if (!NotReady.empty()) {
      AvailableQueue->push_all(NotReady);
      NotReady.clear();
    }

    // If we found a node to schedule, do it now.
    if (FoundSUnit) {
      ScheduleNodeTopDown(FoundSUnit, CurCycle);
      HazardRec->EmitInstruction(FoundNode);
      FoundSUnit->isScheduled = true;
      AvailableQueue->ScheduledNode(FoundSUnit);

      // If this is a pseudo-op node, we don't want to increment the current
      // cycle.
      if (FoundSUnit->Latency)  // Don't increment CurCycle for pseudo-ops!
        ++CurCycle;        
    } else if (!HasNoopHazards) {
      // Otherwise, we have a pipeline stall, but no other problem, just advance
      // the current cycle and try again.
      DOUT << "*** Advancing cycle, no work to do\n";
      HazardRec->AdvanceCycle();
      ++NumStalls;
      ++CurCycle;
    } else {
      // Otherwise, we have no instructions to issue and we have instructions
      // that will fault if we don't do this right.  This is the case for
      // processors without pipeline interlocks and other cases.
      DOUT << "*** Emitting noop\n";
      HazardRec->EmitNoop();
      Sequence.push_back(0);   // NULL SUnit* -> noop
      ++NumNoops;
      ++CurCycle;
    }
  }

#ifndef NDEBUG
  // Verify that all SUnits were scheduled.
  bool AnyNotSched = false;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    if (SUnits[i].NumPredsLeft != 0) {
      if (!AnyNotSched)
        cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(DAG);
      cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
  }
  assert(!AnyNotSched);
#endif
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

/// createTDListDAGScheduler - This creates a top-down list scheduler with a
/// new hazard recognizer. This scheduler takes ownership of the hazard
/// recognizer and deletes it when done.
ScheduleDAG* llvm::createTDListDAGScheduler(SelectionDAGISel *IS,
                                            SelectionDAG *DAG,
                                            const TargetMachine *TM,
                                            MachineBasicBlock *BB, bool Fast) {
  return new ScheduleDAGList(DAG, BB, *TM,
                             new LatencyPriorityQueue(),
                             IS->CreateTargetHazardRecognizer());
}
