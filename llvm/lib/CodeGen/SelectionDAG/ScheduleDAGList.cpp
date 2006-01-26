//===---- ScheduleDAGList.cpp - Implement a list scheduler for isel DAG ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a simple two pass scheduler.  The first pass attempts to push
// backward any lengthy instructions and critical paths.  The second pass packs
// instructions into semi-optimal time slots.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include <climits>
#include <iostream>
#include <memory>
#include <queue>
using namespace llvm;

namespace {

/// SUnit - Scheduling unit. It's an wrapper around either a single SDNode or a
/// group of nodes flagged together.
struct SUnit {
  SDNode *Node;                       // Representative node.
  std::vector<SDNode*> FlaggedNodes;  // All nodes flagged to Node.
  std::vector<SUnit*> Preds;         // All real predecessors.
  std::vector<SUnit*> ChainPreds;    // All chain predecessors.
  std::vector<SUnit*> Succs;         // All real successors.
  std::vector<SUnit*> ChainSuccs;    // All chain successors.
  int NumPredsLeft;                   // # of preds not scheduled.
  int NumSuccsLeft;                   // # of succs not scheduled.
  int Priority1;                      // Scheduling priority 1.
  int Priority2;                      // Scheduling priority 2.
  unsigned Latency;                   // Node latency.
  unsigned CycleBound;                // Upper/lower cycle to be scheduled at.
  unsigned Slot;                      // Cycle node is scheduled at.
  SUnit *Next;

  SUnit(SDNode *node)
    : Node(node), NumPredsLeft(0), NumSuccsLeft(0),
      Priority1(INT_MIN), Priority2(INT_MIN), Latency(0),
      CycleBound(0), Slot(0), Next(NULL) {}

  void dump(const SelectionDAG *G, bool All=true) const;
};

void SUnit::dump(const SelectionDAG *G, bool All) const {
  std::cerr << "SU: ";
  Node->dump(G);
  std::cerr << "\n";
  if (FlaggedNodes.size() != 0) {
    for (unsigned i = 0, e = FlaggedNodes.size(); i != e; i++) {
      std::cerr << "    ";
      FlaggedNodes[i]->dump(G);
      std::cerr << "\n";
    }
  }

  if (All) {
    std::cerr << "# preds left  : " << NumPredsLeft << "\n";
    std::cerr << "# succs left  : " << NumSuccsLeft << "\n";
    std::cerr << "Latency       : " << Latency << "\n";
    std::cerr << "Priority      : " << Priority1 << " , " << Priority2 << "\n";

    if (Preds.size() != 0) {
      std::cerr << "Predecessors  :\n";
      for (unsigned i = 0, e = Preds.size(); i != e; i++) {
        std::cerr << "    ";
        Preds[i]->dump(G, false);
      }
    }
    if (ChainPreds.size() != 0) {
      std::cerr << "Chained Preds :\n";
      for (unsigned i = 0, e = ChainPreds.size(); i != e; i++) {
        std::cerr << "    ";
        ChainPreds[i]->dump(G, false);
      }
    }
    if (Succs.size() != 0) {
      std::cerr << "Successors    :\n";
      for (unsigned i = 0, e = Succs.size(); i != e; i++) {
        std::cerr << "    ";
        Succs[i]->dump(G, false);
      }
    }
    if (ChainSuccs.size() != 0) {
      std::cerr << "Chained succs :\n";
      for (unsigned i = 0, e = ChainSuccs.size(); i != e; i++) {
        std::cerr << "    ";
        ChainSuccs[i]->dump(G, false);
      }
    }
  }
}

/// Sorting functions for the Available queue.
struct ls_rr_sort : public std::binary_function<SUnit*, SUnit*, bool> {
  bool operator()(const SUnit* left, const SUnit* right) const {
    if (left->Priority1 > right->Priority1) {
      return true;
    } else if (left->Priority1 == right->Priority1) {
      unsigned lf = left->FlaggedNodes.size();
      unsigned rf = right->FlaggedNodes.size();
      if (lf > rf)
        return true;
      else if (lf == rf) {
        if (left->Priority2 > right->Priority2)
          return true;
        else if (left->Priority2 == right->Priority2) {
          if (left->CycleBound > right->CycleBound) 
            return true;
          else
            return left->Node->getNodeDepth() < right->Node->getNodeDepth();
        }
      }
    }

    return false;
  }
};

/// ScheduleDAGList - List scheduler.
class ScheduleDAGList : public ScheduleDAG {
private:
  // SDNode to SUnit mapping (many to one).
  std::map<SDNode*, SUnit*> SUnitMap;
  // Available queue.
  std::priority_queue<SUnit*, std::vector<SUnit*>, ls_rr_sort> Available;
  // The schedule.
  std::vector<SUnit*> Sequence;
  // Current scheduling cycle.
  unsigned CurrCycle;
  // First and last SUnit created.
  SUnit *HeadSUnit, *TailSUnit;

public:
  ScheduleDAGList(SelectionDAG &dag, MachineBasicBlock *bb,
                  const TargetMachine &tm)
    : ScheduleDAG(listSchedulingBURR, dag, bb, tm),
      CurrCycle(0), HeadSUnit(NULL), TailSUnit(NULL) {};

  ~ScheduleDAGList() {
    SUnit *SU = HeadSUnit;
    while (SU) {
      SUnit *NextSU = SU->Next;
      delete SU;
      SU = NextSU;
    }
  }

  void Schedule();

  void dump() const;

private:
  SUnit *NewSUnit(SDNode *N);
  void ReleasePred(SUnit *PredSU);
  void ScheduleNode(SUnit *SU);
  int  CalcNodePriority(SUnit *SU);
  void CalculatePriorities();
  void ListSchedule();
  void BuildSchedUnits();
  void EmitSchedule();
};
}  // end namespace


/// NewSUnit - Creates a new SUnit and return a ptr to it.
SUnit *ScheduleDAGList::NewSUnit(SDNode *N) {
  SUnit *CurrSUnit = new SUnit(N);

  if (HeadSUnit == NULL)
    HeadSUnit = CurrSUnit;
  if (TailSUnit != NULL)
    TailSUnit->Next = CurrSUnit;
  TailSUnit = CurrSUnit;

  return CurrSUnit;
}

/// ReleasePred - Decrement the NumSuccsLeft count of a predecessor. Add it to
/// the Available queue is the count reaches zero. Also update its cycle bound.
void ScheduleDAGList::ReleasePred(SUnit *PredSU) {
  SDNode *PredNode = PredSU->Node;

  PredSU->NumSuccsLeft--;
  if (PredSU->NumSuccsLeft == 0) {
    // EntryToken has to go last!
    if (PredNode->getOpcode() != ISD::EntryToken)
      Available.push(PredSU);
  } else if (PredSU->NumSuccsLeft < 0) {
#ifndef NDEBUG
    std::cerr << "*** List scheduling failed! ***\n";
    PredSU->dump(&DAG);
    std::cerr << " has been released too many times!\n";
    assert(0);
#endif
  }

  // FIXME: the distance between two nodes is not always == the predecessor's
  // latency. For example, the reader can very well read the register written
  // by the predecessor later than the issue cycle. It also depends on the
  // interrupt model (drain vs. freeze).
  PredSU->CycleBound = std::max(PredSU->CycleBound, CurrCycle + PredSU->Latency);
}

/// ScheduleNode - Add the node to the schedule. Decrement the pending count of
/// its predecessors. If a predecessor pending count is zero, add it to the
/// Available queue.
void ScheduleDAGList::ScheduleNode(SUnit *SU) {
  Sequence.push_back(SU);
  SU->Slot = CurrCycle;

  // Bottom up: release predecessors
  for (unsigned i = 0, e = SU->Preds.size(); i != e; i++) 
    ReleasePred(SU->Preds[i]);
  for (unsigned i = 0, e = SU->ChainPreds.size(); i != e; i++) 
    ReleasePred(SU->ChainPreds[i]);

  CurrCycle++;
}

/// isReady - True if node's lower cycle bound is less or equal to the current
/// scheduling cycle. Always true if all nodes have uniform latency 1.
static inline bool isReady(SUnit *SU, unsigned CurrCycle) {
  return SU->CycleBound <= CurrCycle;
}

/// ListSchedule - The main loop of list scheduling.
void ScheduleDAGList::ListSchedule() {
  // Add root to Available queue
  SUnit *Root = SUnitMap[DAG.getRoot().Val];
  Available.push(Root);

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back. Schedule the node.
  std::vector<SUnit*> NotReady;
  while (!Available.empty()) {
    SUnit *CurrNode = Available.top();
    Available.pop();

    NotReady.clear();
    while (!isReady(CurrNode, CurrCycle)) {
      NotReady.push_back(CurrNode);
      CurrNode = Available.top();
      Available.pop();
    }
    for (unsigned i = 0, e = NotReady.size(); i != e; ++i)
      Available.push(NotReady[i]);

    DEBUG(std::cerr << "\n*** Scheduling: ");
    DEBUG(CurrNode->dump(&DAG, false));
    DEBUG(std::cerr << "\n");
    ScheduleNode(CurrNode);
  }

  // Add entry node last
  if (DAG.getEntryNode().Val != DAG.getRoot().Val) {
    SUnit *Entry = SUnitMap[DAG.getEntryNode().Val];
    Entry->Slot = CurrCycle;
    Sequence.push_back(Entry);
  }

#ifndef NDEBUG
  bool AnyNotSched = false;
  for (SUnit *SU = HeadSUnit; SU != NULL; SU = SU->Next) {
    if (SU->NumSuccsLeft != 0) {
      if (!AnyNotSched)
        std::cerr << "*** List scheduling failed! ***\n";
      SU->dump(&DAG);
      std::cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
  }
  assert(!AnyNotSched);
#endif


  // Reverse the order if it is bottom up.
  std::reverse(Sequence.begin(), Sequence.end());

  DEBUG(std::cerr << "*** Final schedule ***\n");
  DEBUG(dump());
  DEBUG(std::cerr << "\n");
}

/// CalcNodePriority - Priority 1 is just the number of live range genned - number
/// of live range killed. Priority 2 is the Sethi Ullman number. It returns
/// priority 2 since it is calculated recursively.
/// Smaller number is the higher priority in both cases.
int ScheduleDAGList::CalcNodePriority(SUnit *SU) {
  if (SU->Priority2 != INT_MIN)
    return SU->Priority2;

  SU->Priority1 = SU->Preds.size() - SU->Succs.size();

  if (SU->Preds.size() == 0) {
    SU->Priority2 = 1;
  } else {
    int Extra = 0;
    for (unsigned i = 0, e = SU->Preds.size(); i != e; i++) {
      SUnit *PredSU = SU->Preds[i];
      int PredPriority = CalcNodePriority(PredSU);
      if (PredPriority > SU->Priority2) {
        SU->Priority2 = PredPriority;
        Extra = 0;
      } else if (PredPriority == SU->Priority2)
        Extra++;
    }

    if (SU->Node->getOpcode() != ISD::TokenFactor)
      SU->Priority2 += Extra;
    else
      SU->Priority2 = (Extra == 1) ? 0 : Extra-1;
  }

  return SU->Priority2;
}

/// CalculatePriorities - Calculate priorities of all scheduling units.
void ScheduleDAGList::CalculatePriorities() {
  for (SUnit *SU = HeadSUnit; SU != NULL; SU = SU->Next) {
    // FIXME: assumes uniform latency for now.
    SU->Latency = 1;
    (void)CalcNodePriority(SU);
    DEBUG(SU->dump(&DAG));
    DEBUG(std::cerr << "\n");
  }
}

void ScheduleDAGList::BuildSchedUnits() {
  // Pass 1: create the SUnit's.
  for (unsigned i = 0, NC = NodeCount; i < NC; i++) {
    NodeInfo *NI = &Info[i];
    SDNode *N = NI->Node;
    if (isPassiveNode(N))
      continue;

    SUnit *SU;
    if (NI->isInGroup()) {
      if (NI != NI->Group->getBottom())  // Bottom up, so only look at bottom
        continue;                        // node of the NodeGroup

      SU = NewSUnit(N);
      // Find the flagged nodes.
      SDOperand  FlagOp = N->getOperand(N->getNumOperands() - 1);
      SDNode    *Flag   = FlagOp.Val;
      unsigned   ResNo  = FlagOp.ResNo;
      while (Flag->getValueType(ResNo) == MVT::Flag) {
        NodeInfo *FNI = getNI(Flag);
        assert(FNI->Group == NI->Group);
        SU->FlaggedNodes.insert(SU->FlaggedNodes.begin(), Flag);
        SUnitMap[Flag] = SU;

        FlagOp = Flag->getOperand(Flag->getNumOperands() - 1);
        Flag   = FlagOp.Val;
        ResNo  = FlagOp.ResNo;
      }
    } else {
      SU = NewSUnit(N);
    }
    SUnitMap[N] = SU;
  }

  // Pass 2: add the preds, succs, etc.
  for (SUnit *SU = HeadSUnit; SU != NULL; SU = SU->Next) {
    SDNode   *N  = SU->Node;
    NodeInfo *NI = getNI(N);

    if (NI->isInGroup()) {
      // Find all predecessors (of the group).
      NodeGroupOpIterator NGOI(NI);
      while (!NGOI.isEnd()) {
        SDOperand Op  = NGOI.next();
        SDNode   *OpN = Op.Val;
        MVT::ValueType VT = OpN->getValueType(Op.ResNo);
        NodeInfo *OpNI = getNI(OpN);
        if (OpNI->Group != NI->Group && !isPassiveNode(OpN)) {
          assert(VT != MVT::Flag);
          SUnit *OpSU = SUnitMap[OpN];
          if (VT == MVT::Other) {
            SU  ->ChainPreds.push_back(OpSU);
            OpSU->ChainSuccs.push_back(SU);
          } else {
            SU  ->Preds.push_back(OpSU);
            OpSU->Succs.push_back(SU);
          }
          SU->NumPredsLeft++;
          OpSU->NumSuccsLeft++;
        }
      }
    } else {
      // Find node predecessors.
      for (unsigned j = 0, e = N->getNumOperands(); j != e; j++) {
        SDOperand Op  = N->getOperand(j);
        SDNode   *OpN = Op.Val;
        MVT::ValueType VT = OpN->getValueType(Op.ResNo);
        if (!isPassiveNode(OpN)) {
          assert(VT != MVT::Flag);
          SUnit *OpSU = SUnitMap[OpN];
          if (VT == MVT::Other) {
            SU  ->ChainPreds.push_back(OpSU);
            OpSU->ChainSuccs.push_back(SU);
          } else {
            SU  ->Preds.push_back(OpSU);
            OpSU->Succs.push_back(SU);
          }
          SU->NumPredsLeft++;
          OpSU->NumSuccsLeft++;
        }
      }
    }
  }
}

/// EmitSchedule - Emit the machine code in scheduled order.
void ScheduleDAGList::EmitSchedule() {
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    SDNode *N;
    SUnit *SU = Sequence[i];
    for (unsigned j = 0, ee = SU->FlaggedNodes.size(); j != ee; j++) {
      N = SU->FlaggedNodes[j];
      EmitNode(getNI(N));
    }
    EmitNode(getNI(SU->Node));
  }
}

/// dump - dump the schedule.
void ScheduleDAGList::dump() const {
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    SUnit *SU = Sequence[i];
    SU->dump(&DAG, false);
  }
}

/// Schedule - Schedule the DAG using list scheduling.
/// FIXME: Right now it only supports the burr (bottom up register reducing)
/// heuristic.
void ScheduleDAGList::Schedule() {
  DEBUG(std::cerr << "********** List Scheduling **********\n");

  // Build scheduling units.
  BuildSchedUnits();

  // Calculate node prirorities.
  CalculatePriorities();

  // Execute the actual scheduling loop.
  ListSchedule();

  // Emit in scheduled order
  EmitSchedule();
}
  
llvm::ScheduleDAG* llvm::createBURRListDAGScheduler(SelectionDAG &DAG,
                                                    MachineBasicBlock *BB) {
  return new ScheduleDAGList(DAG, BB, DAG.getTarget());
}
