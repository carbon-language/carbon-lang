//===---- LatencyPriorityQueue.cpp - A latency-oriented priority queue ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LatencyPriorityQueue class, which is a
// SchedulingPriorityQueue that schedules using latency information to
// reduce the length of the critical path through the basic block.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "scheduler"
#include "LatencyPriorityQueue.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

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

  std::vector<const SUnit*> WorkList;
  WorkList.push_back(&SU);
  while (!WorkList.empty()) {
    const SUnit *Cur = WorkList.back();
    bool AllDone = true;
    int MaxSuccLatency = 0;
    for (SUnit::const_succ_iterator I = Cur->Succs.begin(),E = Cur->Succs.end();
         I != E; ++I) {
      int SuccLatency = Latencies[I->Dep->NodeNum];
      if (SuccLatency == -1) {
        AllDone = false;
        WorkList.push_back(I->Dep);
      } else {
        MaxSuccLatency = std::max(MaxSuccLatency, SuccLatency);
      }
    }
    if (AllDone) {
      Latencies[Cur->NodeNum] = MaxSuccLatency + Cur->Latency;
      WorkList.pop_back();
    }
  }

  return Latency;
}

/// CalculatePriorities - Calculate priorities of all scheduling units.
void LatencyPriorityQueue::CalculatePriorities() {
  Latencies.assign(SUnits->size(), -1);
  NumNodesSolelyBlocking.assign(SUnits->size(), 0);

  // For each node, calculate the maximal path from the node to the exit.
  std::vector<std::pair<const SUnit*, unsigned> > WorkList;
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i) {
    const SUnit *SU = &(*SUnits)[i];
    if (SU->Succs.empty())
      WorkList.push_back(std::make_pair(SU, 0U));
  }

  while (!WorkList.empty()) {
    const SUnit *SU = WorkList.back().first;
    unsigned SuccLat = WorkList.back().second;
    WorkList.pop_back();
    int &Latency = Latencies[SU->NodeNum];
    if (Latency == -1 || (SU->Latency + SuccLat) > (unsigned)Latency) {
      Latency = SU->Latency + SuccLat;
      for (SUnit::const_pred_iterator I = SU->Preds.begin(),E = SU->Preds.end();
           I != E; ++I)
        WorkList.push_back(std::make_pair(I->Dep, Latency));
    }
  }
}

/// getSingleUnscheduledPred - If there is exactly one unscheduled predecessor
/// of SU, return it, otherwise return null.
SUnit *LatencyPriorityQueue::getSingleUnscheduledPred(SUnit *SU) {
  SUnit *OnlyAvailablePred = 0;
  for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    SUnit &Pred = *I->Dep;
    if (!Pred.isScheduled) {
      // We found an available, but not scheduled, predecessor.  If it's the
      // only one we have found, keep track of it... otherwise give up.
      if (OnlyAvailablePred && OnlyAvailablePred != &Pred)
        return 0;
      OnlyAvailablePred = &Pred;
    }
  }
      
  return OnlyAvailablePred;
}

void LatencyPriorityQueue::push_impl(SUnit *SU) {
  // Look at all of the successors of this node.  Count the number of nodes that
  // this node is the sole unscheduled node for.
  unsigned NumNodesBlocking = 0;
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I)
    if (getSingleUnscheduledPred(I->Dep) == SU)
      ++NumNodesBlocking;
  NumNodesSolelyBlocking[SU->NodeNum] = NumNodesBlocking;
  
  Queue.push(SU);
}


// ScheduledNode - As nodes are scheduled, we look to see if there are any
// successor nodes that have a single unscheduled predecessor.  If so, that
// single predecessor has a higher priority, since scheduling it will make
// the node available.
void LatencyPriorityQueue::ScheduledNode(SUnit *SU) {
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I)
    AdjustPriorityOfUnscheduledPreds(I->Dep);
}

/// AdjustPriorityOfUnscheduledPreds - One of the predecessors of SU was just
/// scheduled.  If SU is not itself available, then there is at least one
/// predecessor node that has not been scheduled yet.  If SU has exactly ONE
/// unscheduled predecessor, we want to increase its priority: it getting
/// scheduled will make this node available, so it is better than some other
/// node of the same priority that will not make a node available.
void LatencyPriorityQueue::AdjustPriorityOfUnscheduledPreds(SUnit *SU) {
  if (SU->isAvailable) return;  // All preds scheduled.
  
  SUnit *OnlyAvailablePred = getSingleUnscheduledPred(SU);
  if (OnlyAvailablePred == 0 || !OnlyAvailablePred->isAvailable) return;
  
  // Okay, we found a single predecessor that is available, but not scheduled.
  // Since it is available, it must be in the priority queue.  First remove it.
  remove(OnlyAvailablePred);

  // Reinsert the node into the priority queue, which recomputes its
  // NumNodesSolelyBlocking value.
  push(OnlyAvailablePred);
}
