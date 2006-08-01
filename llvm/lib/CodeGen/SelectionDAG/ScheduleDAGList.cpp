//===---- ScheduleDAGList.cpp - Implement a list scheduler for isel DAG ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
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

#define DEBUG_TYPE "sched"
#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Visibility.h"
#include "llvm/ADT/Statistic.h"
#include <climits>
#include <iostream>
#include <queue>
using namespace llvm;

namespace {
  static Statistic<> NumNoops ("scheduler", "Number of noops inserted");
  static Statistic<> NumStalls("scheduler", "Number of pipeline stalls");
}

static RegisterScheduler
  tdListDAGScheduler("list-td", "  Top-down list scheduler",
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
  ScheduleDAGList(SelectionDAG &dag, MachineBasicBlock *bb,
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
  DEBUG(std::cerr << "********** List Scheduling **********\n");
  
  // Build scheduling units.
  BuildSchedUnits();

  AvailableQueue->initNodes(SUnits);
  
  ListScheduleTopDown();
  
  AvailableQueue->releaseState();
  
  DEBUG(std::cerr << "*** Final schedule ***\n");
  DEBUG(dumpSchedule());
  DEBUG(std::cerr << "\n");
  
  // Emit in scheduled order
  EmitSchedule();
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the PendingQueue if the count reaches zero.
void ScheduleDAGList::ReleaseSucc(SUnit *SuccSU, bool isChain) {
  if (!isChain)
    SuccSU->NumPredsLeft--;
  else
    SuccSU->NumChainPredsLeft--;
  
  assert(SuccSU->NumPredsLeft >= 0 && SuccSU->NumChainPredsLeft >= 0 &&
         "List scheduling internal error");
  
  if ((SuccSU->NumPredsLeft + SuccSU->NumChainPredsLeft) == 0) {
    // Compute how many cycles it will be before this actually becomes
    // available.  This is the max of the start time of all predecessors plus
    // their latencies.
    unsigned AvailableCycle = 0;
    for (std::set<std::pair<SUnit*, bool> >::iterator I = SuccSU->Preds.begin(),
         E = SuccSU->Preds.end(); I != E; ++I) {
      // If this is a token edge, we don't need to wait for the latency of the
      // preceeding instruction (e.g. a long-latency load) unless there is also
      // some other data dependence.
      unsigned PredDoneCycle = I->first->Cycle;
      if (!I->second)
        PredDoneCycle += I->first->Latency;
      else if (I->first->Latency)
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
  DEBUG(std::cerr << "*** Scheduling [" << CurCycle << "]: ");
  DEBUG(SU->dump(&DAG));
  
  Sequence.push_back(SU);
  SU->Cycle = CurCycle;
  
  // Bottom up: release successors.
  for (std::set<std::pair<SUnit*, bool> >::iterator I = SU->Succs.begin(),
       E = SU->Succs.end(); I != E; ++I)
    ReleaseSucc(I->first, I->second);
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void ScheduleDAGList::ListScheduleTopDown() {
  unsigned CurCycle = 0;
  SUnit *Entry = SUnitMap[DAG.getEntryNode().Val];

  // All leaves to Available queue.
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    // It is available if it has no predecessors.
    if (SUnits[i].Preds.size() == 0 && &SUnits[i] != Entry) {
      AvailableQueue->push(&SUnits[i]);
      SUnits[i].isAvailable = SUnits[i].isPending = true;
    }
  }
  
  // Emit the entry node first.
  ScheduleNodeTopDown(Entry, CurCycle);
  HazardRec->EmitInstruction(Entry->Node);
  
  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  std::vector<SUnit*> NotReady;
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
      FoundNode = CurSUnit->Node;
      
      // If this is a pseudo op, like copyfromreg, look to see if there is a
      // real target node flagged to it.  If so, use the target node.
      for (unsigned i = 0, e = CurSUnit->FlaggedNodes.size(); 
           FoundNode->getOpcode() < ISD::BUILTIN_OP_END && i != e; ++i)
        FoundNode = CurSUnit->FlaggedNodes[i];
      
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
      DEBUG(std::cerr << "*** Advancing cycle, no work to do\n");
      HazardRec->AdvanceCycle();
      ++NumStalls;
      ++CurCycle;
    } else {
      // Otherwise, we have no instructions to issue and we have instructions
      // that will fault if we don't do this right.  This is the case for
      // processors without pipeline interlocks and other cases.
      DEBUG(std::cerr << "*** Emitting noop\n");
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
    if (SUnits[i].NumPredsLeft != 0 || SUnits[i].NumChainPredsLeft != 0) {
      if (!AnyNotSched)
        std::cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(&DAG);
      std::cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
  }
  assert(!AnyNotSched);
#endif
}

//===----------------------------------------------------------------------===//
//                    LatencyPriorityQueue Implementation
//===----------------------------------------------------------------------===//
//
// This is a SchedulingPriorityQueue that schedules using latency information to
// reduce the length of the critical path through the basic block.
// 
namespace {
  class LatencyPriorityQueue;
  
  /// Sorting functions for the Available queue.
  struct latency_sort : public std::binary_function<SUnit*, SUnit*, bool> {
    LatencyPriorityQueue *PQ;
    latency_sort(LatencyPriorityQueue *pq) : PQ(pq) {}
    latency_sort(const latency_sort &RHS) : PQ(RHS.PQ) {}
    
    bool operator()(const SUnit* left, const SUnit* right) const;
  };
}  // end anonymous namespace

namespace {
  class LatencyPriorityQueue : public SchedulingPriorityQueue {
    // SUnits - The SUnits for the current graph.
    const std::vector<SUnit> *SUnits;
    
    // Latencies - The latency (max of latency from this node to the bb exit)
    // for each node.
    std::vector<int> Latencies;

    /// NumNodesSolelyBlocking - This vector contains, for every node in the
    /// Queue, the number of nodes that the node is the sole unscheduled
    /// predecessor for.  This is used as a tie-breaker heuristic for better
    /// mobility.
    std::vector<unsigned> NumNodesSolelyBlocking;

    std::priority_queue<SUnit*, std::vector<SUnit*>, latency_sort> Queue;
public:
    LatencyPriorityQueue() : Queue(latency_sort(this)) {
    }
    
    void initNodes(const std::vector<SUnit> &sunits) {
      SUnits = &sunits;
      // Calculate node priorities.
      CalculatePriorities();
    }
    void releaseState() {
      SUnits = 0;
      Latencies.clear();
    }
    
    unsigned getLatency(unsigned NodeNum) const {
      assert(NodeNum < Latencies.size());
      return Latencies[NodeNum];
    }
    
    unsigned getNumSolelyBlockNodes(unsigned NodeNum) const {
      assert(NodeNum < NumNodesSolelyBlocking.size());
      return NumNodesSolelyBlocking[NodeNum];
    }
    
    bool empty() const { return Queue.empty(); }
    
    virtual void push(SUnit *U) {
      push_impl(U);
    }
    void push_impl(SUnit *U);
    
    void push_all(const std::vector<SUnit *> &Nodes) {
      for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
        push_impl(Nodes[i]);
    }
    
    SUnit *pop() {
      if (empty()) return NULL;
      SUnit *V = Queue.top();
      Queue.pop();
      return V;
    }

    // ScheduledNode - As nodes are scheduled, we look to see if there are any
    // successor nodes that have a single unscheduled predecessor.  If so, that
    // single predecessor has a higher priority, since scheduling it will make
    // the node available.
    void ScheduledNode(SUnit *Node);

private:
    void CalculatePriorities();
    int CalcLatency(const SUnit &SU);
    void AdjustPriorityOfUnscheduledPreds(SUnit *SU);

    /// RemoveFromPriorityQueue - This is a really inefficient way to remove a
    /// node from a priority queue.  We should roll our own heap to make this
    /// better or something.
    void RemoveFromPriorityQueue(SUnit *SU) {
      std::vector<SUnit*> Temp;
      
      assert(!Queue.empty() && "Not in queue!");
      while (Queue.top() != SU) {
        Temp.push_back(Queue.top());
        Queue.pop();
        assert(!Queue.empty() && "Not in queue!");
      }

      // Remove the node from the PQ.
      Queue.pop();
      
      // Add all the other nodes back.
      for (unsigned i = 0, e = Temp.size(); i != e; ++i)
        Queue.push(Temp[i]);
    }
  };
}

bool latency_sort::operator()(const SUnit *LHS, const SUnit *RHS) const {
  unsigned LHSNum = LHS->NodeNum;
  unsigned RHSNum = RHS->NodeNum;

  // The most important heuristic is scheduling the critical path.
  unsigned LHSLatency = PQ->getLatency(LHSNum);
  unsigned RHSLatency = PQ->getLatency(RHSNum);
  if (LHSLatency < RHSLatency) return true;
  if (LHSLatency > RHSLatency) return false;
  
  // After that, if two nodes have identical latencies, look to see if one will
  // unblock more other nodes than the other.
  unsigned LHSBlocked = PQ->getNumSolelyBlockNodes(LHSNum);
  unsigned RHSBlocked = PQ->getNumSolelyBlockNodes(RHSNum);
  if (LHSBlocked < RHSBlocked) return true;
  if (LHSBlocked > RHSBlocked) return false;
  
  // Finally, just to provide a stable ordering, use the node number as a
  // deciding factor.
  return LHSNum < RHSNum;
}


/// CalcNodePriority - Calculate the maximal path from the node to the exit.
///
int LatencyPriorityQueue::CalcLatency(const SUnit &SU) {
  int &Latency = Latencies[SU.NodeNum];
  if (Latency != -1)
    return Latency;
  
  int MaxSuccLatency = 0;
  for (std::set<std::pair<SUnit*, bool> >::const_iterator I = SU.Succs.begin(),
       E = SU.Succs.end(); I != E; ++I)
    MaxSuccLatency = std::max(MaxSuccLatency, CalcLatency(*I->first));

  return Latency = MaxSuccLatency + SU.Latency;
}

/// CalculatePriorities - Calculate priorities of all scheduling units.
void LatencyPriorityQueue::CalculatePriorities() {
  Latencies.assign(SUnits->size(), -1);
  NumNodesSolelyBlocking.assign(SUnits->size(), 0);
  
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i)
    CalcLatency((*SUnits)[i]);
}

/// getSingleUnscheduledPred - If there is exactly one unscheduled predecessor
/// of SU, return it, otherwise return null.
static SUnit *getSingleUnscheduledPred(SUnit *SU) {
  SUnit *OnlyAvailablePred = 0;
  for (std::set<std::pair<SUnit*, bool> >::const_iterator I = SU->Preds.begin(),
       E = SU->Preds.end(); I != E; ++I)
    if (!I->first->isScheduled) {
      // We found an available, but not scheduled, predecessor.  If it's the
      // only one we have found, keep track of it... otherwise give up.
      if (OnlyAvailablePred && OnlyAvailablePred != I->first)
        return 0;
      OnlyAvailablePred = I->first;
    }
      
  return OnlyAvailablePred;
}

void LatencyPriorityQueue::push_impl(SUnit *SU) {
  // Look at all of the successors of this node.  Count the number of nodes that
  // this node is the sole unscheduled node for.
  unsigned NumNodesBlocking = 0;
  for (std::set<std::pair<SUnit*, bool> >::const_iterator I = SU->Succs.begin(),
       E = SU->Succs.end(); I != E; ++I)
    if (getSingleUnscheduledPred(I->first) == SU)
      ++NumNodesBlocking;
  NumNodesSolelyBlocking[SU->NodeNum] = NumNodesBlocking;
  
  Queue.push(SU);
}


// ScheduledNode - As nodes are scheduled, we look to see if there are any
// successor nodes that have a single unscheduled predecessor.  If so, that
// single predecessor has a higher priority, since scheduling it will make
// the node available.
void LatencyPriorityQueue::ScheduledNode(SUnit *SU) {
  for (std::set<std::pair<SUnit*, bool> >::const_iterator I = SU->Succs.begin(),
       E = SU->Succs.end(); I != E; ++I)
    AdjustPriorityOfUnscheduledPreds(I->first);
}

/// AdjustPriorityOfUnscheduledPreds - One of the predecessors of SU was just
/// scheduled.  If SU is not itself available, then there is at least one
/// predecessor node that has not been scheduled yet.  If SU has exactly ONE
/// unscheduled predecessor, we want to increase its priority: it getting
/// scheduled will make this node available, so it is better than some other
/// node of the same priority that will not make a node available.
void LatencyPriorityQueue::AdjustPriorityOfUnscheduledPreds(SUnit *SU) {
  if (SU->isPending) return;  // All preds scheduled.
  
  SUnit *OnlyAvailablePred = getSingleUnscheduledPred(SU);
  if (OnlyAvailablePred == 0 || !OnlyAvailablePred->isAvailable) return;
  
  // Okay, we found a single predecessor that is available, but not scheduled.
  // Since it is available, it must be in the priority queue.  First remove it.
  RemoveFromPriorityQueue(OnlyAvailablePred);

  // Reinsert the node into the priority queue, which recomputes its
  // NumNodesSolelyBlocking value.
  push(OnlyAvailablePred);
}


//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

/// createTDListDAGScheduler - This creates a top-down list scheduler with a
/// new hazard recognizer. This scheduler takes ownership of the hazard
/// recognizer and deletes it when done.
ScheduleDAG* llvm::createTDListDAGScheduler(SelectionDAGISel *IS,
                                            SelectionDAG *DAG,
                                            MachineBasicBlock *BB) {
  return new ScheduleDAGList(*DAG, BB, DAG->getTarget(),
                             new LatencyPriorityQueue(),
                             IS->CreateTargetHazardRecognizer());
}
