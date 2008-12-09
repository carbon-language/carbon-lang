//===---- ScheduleDAG.cpp - Implement the ScheduleDAG class ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the ScheduleDAG class, which is a base class used by
// scheduling implementation classes.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include <climits>
using namespace llvm;

ScheduleDAG::ScheduleDAG(SelectionDAG *dag, MachineBasicBlock *bb,
                         const TargetMachine &tm)
  : DAG(dag), BB(bb), TM(tm), MRI(BB->getParent()->getRegInfo()) {
  TII = TM.getInstrInfo();
  MF  = BB->getParent();
  TRI = TM.getRegisterInfo();
  TLI = TM.getTargetLowering();
  ConstPool = MF->getConstantPool();
}

ScheduleDAG::~ScheduleDAG() {}

/// CalculateDepths - compute depths using algorithms for the longest
/// paths in the DAG
void ScheduleDAG::CalculateDepths() {
  unsigned DAGSize = SUnits.size();
  std::vector<SUnit*> WorkList;
  WorkList.reserve(DAGSize);

  // Initialize the data structures
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    unsigned Degree = SU->Preds.size();
    // Temporarily use the Depth field as scratch space for the degree count.
    SU->Depth = Degree;

    // Is it a node without dependencies?
    if (Degree == 0) {
      assert(SU->Preds.empty() && "SUnit should have no predecessors");
      // Collect leaf nodes
      WorkList.push_back(SU);
    }
  }

  // Process nodes in the topological order
  while (!WorkList.empty()) {
    SUnit *SU = WorkList.back();
    WorkList.pop_back();
    unsigned SUDepth = 0;

    // Use dynamic programming:
    // When current node is being processed, all of its dependencies
    // are already processed.
    // So, just iterate over all predecessors and take the longest path
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      unsigned PredDepth = I->getSUnit()->Depth;
      if (PredDepth+1 > SUDepth) {
        SUDepth = PredDepth + 1;
      }
    }

    SU->Depth = SUDepth;

    // Update degrees of all nodes depending on current SUnit
    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      SUnit *SU = I->getSUnit();
      if (!--SU->Depth)
        // If all dependencies of the node are processed already,
        // then the longest path for the node can be computed now
        WorkList.push_back(SU);
    }
  }
}

/// CalculateHeights - compute heights using algorithms for the longest
/// paths in the DAG
void ScheduleDAG::CalculateHeights() {
  unsigned DAGSize = SUnits.size();
  std::vector<SUnit*> WorkList;
  WorkList.reserve(DAGSize);

  // Initialize the data structures
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    unsigned Degree = SU->Succs.size();
    // Temporarily use the Height field as scratch space for the degree count.
    SU->Height = Degree;

    // Is it a node without dependencies?
    if (Degree == 0) {
      assert(SU->Succs.empty() && "Something wrong");
      assert(WorkList.empty() && "Should be empty");
      // Collect leaf nodes
      WorkList.push_back(SU);
    }
  }

  // Process nodes in the topological order
  while (!WorkList.empty()) {
    SUnit *SU = WorkList.back();
    WorkList.pop_back();
    unsigned SUHeight = 0;

    // Use dynamic programming:
    // When current node is being processed, all of its dependencies
    // are already processed.
    // So, just iterate over all successors and take the longest path
    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      unsigned SuccHeight = I->getSUnit()->Height;
      if (SuccHeight+1 > SUHeight) {
        SUHeight = SuccHeight + 1;
      }
    }

    SU->Height = SUHeight;

    // Update degrees of all nodes depending on current SUnit
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      SUnit *SU = I->getSUnit();
      if (!--SU->Height)
        // If all dependencies of the node are processed already,
        // then the longest path for the node can be computed now
        WorkList.push_back(SU);
    }
  }
}

/// dump - dump the schedule.
void ScheduleDAG::dumpSchedule() const {
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    if (SUnit *SU = Sequence[i])
      SU->dump(this);
    else
      cerr << "**** NOOP ****\n";
  }
}


/// Run - perform scheduling.
///
void ScheduleDAG::Run() {
  Schedule();
  
  DOUT << "*** Final schedule ***\n";
  DEBUG(dumpSchedule());
  DOUT << "\n";
}

/// SUnit - Scheduling unit. It's an wrapper around either a single SDNode or
/// a group of nodes flagged together.
void SUnit::dump(const ScheduleDAG *G) const {
  cerr << "SU(" << NodeNum << "): ";
  G->dumpNode(this);
}

void SUnit::dumpAll(const ScheduleDAG *G) const {
  dump(G);

  cerr << "  # preds left       : " << NumPredsLeft << "\n";
  cerr << "  # succs left       : " << NumSuccsLeft << "\n";
  cerr << "  Latency            : " << Latency << "\n";
  cerr << "  Depth              : " << Depth << "\n";
  cerr << "  Height             : " << Height << "\n";

  if (Preds.size() != 0) {
    cerr << "  Predecessors:\n";
    for (SUnit::const_succ_iterator I = Preds.begin(), E = Preds.end();
         I != E; ++I) {
      cerr << "   ";
      switch (I->getKind()) {
      case SDep::Data:        cerr << "val "; break;
      case SDep::Anti:        cerr << "anti"; break;
      case SDep::Output:      cerr << "out "; break;
      case SDep::Order:       cerr << "ch  "; break;
      }
      cerr << "#";
      cerr << I->getSUnit() << " - SU(" << I->getSUnit()->NodeNum << ")";
      if (I->isArtificial())
        cerr << " *";
      cerr << "\n";
    }
  }
  if (Succs.size() != 0) {
    cerr << "  Successors:\n";
    for (SUnit::const_succ_iterator I = Succs.begin(), E = Succs.end();
         I != E; ++I) {
      cerr << "   ";
      switch (I->getKind()) {
      case SDep::Data:        cerr << "val "; break;
      case SDep::Anti:        cerr << "anti"; break;
      case SDep::Output:      cerr << "out "; break;
      case SDep::Order:       cerr << "ch  "; break;
      }
      cerr << "#";
      cerr << I->getSUnit() << " - SU(" << I->getSUnit()->NodeNum << ")";
      if (I->isArtificial())
        cerr << " *";
      cerr << "\n";
    }
  }
  cerr << "\n";
}

#ifndef NDEBUG
/// VerifySchedule - Verify that all SUnits were scheduled and that
/// their state is consistent.
///
void ScheduleDAG::VerifySchedule(bool isBottomUp) {
  bool AnyNotSched = false;
  unsigned DeadNodes = 0;
  unsigned Noops = 0;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    if (!SUnits[i].isScheduled) {
      if (SUnits[i].NumPreds == 0 && SUnits[i].NumSuccs == 0) {
        ++DeadNodes;
        continue;
      }
      if (!AnyNotSched)
        cerr << "*** Scheduling failed! ***\n";
      SUnits[i].dump(this);
      cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
    if (SUnits[i].isScheduled && SUnits[i].Cycle > (unsigned)INT_MAX) {
      if (!AnyNotSched)
        cerr << "*** Scheduling failed! ***\n";
      SUnits[i].dump(this);
      cerr << "has an unexpected Cycle value!\n";
      AnyNotSched = true;
    }
    if (isBottomUp) {
      if (SUnits[i].NumSuccsLeft != 0) {
        if (!AnyNotSched)
          cerr << "*** Scheduling failed! ***\n";
        SUnits[i].dump(this);
        cerr << "has successors left!\n";
        AnyNotSched = true;
      }
    } else {
      if (SUnits[i].NumPredsLeft != 0) {
        if (!AnyNotSched)
          cerr << "*** Scheduling failed! ***\n";
        SUnits[i].dump(this);
        cerr << "has predecessors left!\n";
        AnyNotSched = true;
      }
    }
  }
  for (unsigned i = 0, e = Sequence.size(); i != e; ++i)
    if (!Sequence[i])
      ++Noops;
  assert(!AnyNotSched);
  assert(Sequence.size() + DeadNodes - Noops == SUnits.size() &&
         "The number of nodes scheduled doesn't match the expected number!");
}
#endif

/// InitDAGTopologicalSorting - create the initial topological 
/// ordering from the DAG to be scheduled.
///
/// The idea of the algorithm is taken from 
/// "Online algorithms for managing the topological order of
/// a directed acyclic graph" by David J. Pearce and Paul H.J. Kelly
/// This is the MNR algorithm, which was first introduced by 
/// A. Marchetti-Spaccamela, U. Nanni and H. Rohnert in  
/// "Maintaining a topological order under edge insertions".
///
/// Short description of the algorithm: 
///
/// Topological ordering, ord, of a DAG maps each node to a topological
/// index so that for all edges X->Y it is the case that ord(X) < ord(Y).
///
/// This means that if there is a path from the node X to the node Z, 
/// then ord(X) < ord(Z).
///
/// This property can be used to check for reachability of nodes:
/// if Z is reachable from X, then an insertion of the edge Z->X would 
/// create a cycle.
///
/// The algorithm first computes a topological ordering for the DAG by
/// initializing the Index2Node and Node2Index arrays and then tries to keep
/// the ordering up-to-date after edge insertions by reordering the DAG.
///
/// On insertion of the edge X->Y, the algorithm first marks by calling DFS
/// the nodes reachable from Y, and then shifts them using Shift to lie
/// immediately after X in Index2Node.
void ScheduleDAGTopologicalSort::InitDAGTopologicalSorting() {
  unsigned DAGSize = SUnits.size();
  std::vector<SUnit*> WorkList;
  WorkList.reserve(DAGSize);

  Index2Node.resize(DAGSize);
  Node2Index.resize(DAGSize);

  // Initialize the data structures.
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    int NodeNum = SU->NodeNum;
    unsigned Degree = SU->Succs.size();
    // Temporarily use the Node2Index array as scratch space for degree counts.
    Node2Index[NodeNum] = Degree;

    // Is it a node without dependencies?
    if (Degree == 0) {
      assert(SU->Succs.empty() && "SUnit should have no successors");
      // Collect leaf nodes.
      WorkList.push_back(SU);
    }
  }  

  int Id = DAGSize;
  while (!WorkList.empty()) {
    SUnit *SU = WorkList.back();
    WorkList.pop_back();
    Allocate(SU->NodeNum, --Id);
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      SUnit *SU = I->getSUnit();
      if (!--Node2Index[SU->NodeNum])
        // If all dependencies of the node are processed already,
        // then the node can be computed now.
        WorkList.push_back(SU);
    }
  }

  Visited.resize(DAGSize);

#ifndef NDEBUG
  // Check correctness of the ordering
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      assert(Node2Index[SU->NodeNum] > Node2Index[I->getSUnit()->NodeNum] && 
      "Wrong topological sorting");
    }
  }
#endif
}

/// AddPred - Updates the topological ordering to accomodate an edge
/// to be added from SUnit X to SUnit Y.
void ScheduleDAGTopologicalSort::AddPred(SUnit *Y, SUnit *X) {
  int UpperBound, LowerBound;
  LowerBound = Node2Index[Y->NodeNum];
  UpperBound = Node2Index[X->NodeNum];
  bool HasLoop = false;
  // Is Ord(X) < Ord(Y) ?
  if (LowerBound < UpperBound) {
    // Update the topological order.
    Visited.reset();
    DFS(Y, UpperBound, HasLoop);
    assert(!HasLoop && "Inserted edge creates a loop!");
    // Recompute topological indexes.
    Shift(Visited, LowerBound, UpperBound);
  }
}

/// RemovePred - Updates the topological ordering to accomodate an
/// an edge to be removed from the specified node N from the predecessors
/// of the current node M.
void ScheduleDAGTopologicalSort::RemovePred(SUnit *M, SUnit *N) {
  // InitDAGTopologicalSorting();
}

/// DFS - Make a DFS traversal to mark all nodes reachable from SU and mark
/// all nodes affected by the edge insertion. These nodes will later get new
/// topological indexes by means of the Shift method.
void ScheduleDAGTopologicalSort::DFS(const SUnit *SU, int UpperBound,
                                     bool& HasLoop) {
  std::vector<const SUnit*> WorkList;
  WorkList.reserve(SUnits.size()); 

  WorkList.push_back(SU);
  while (!WorkList.empty()) {
    SU = WorkList.back();
    WorkList.pop_back();
    Visited.set(SU->NodeNum);
    for (int I = SU->Succs.size()-1; I >= 0; --I) {
      int s = SU->Succs[I].getSUnit()->NodeNum;
      if (Node2Index[s] == UpperBound) {
        HasLoop = true; 
        return;
      }
      // Visit successors if not already and in affected region.
      if (!Visited.test(s) && Node2Index[s] < UpperBound) {
        WorkList.push_back(SU->Succs[I].getSUnit());
      } 
    } 
  }
}

/// Shift - Renumber the nodes so that the topological ordering is 
/// preserved.
void ScheduleDAGTopologicalSort::Shift(BitVector& Visited, int LowerBound, 
                                       int UpperBound) {
  std::vector<int> L;
  int shift = 0;
  int i;

  for (i = LowerBound; i <= UpperBound; ++i) {
    // w is node at topological index i.
    int w = Index2Node[i];
    if (Visited.test(w)) {
      // Unmark.
      Visited.reset(w);
      L.push_back(w);
      shift = shift + 1;
    } else {
      Allocate(w, i - shift);
    }
  }

  for (unsigned j = 0; j < L.size(); ++j) {
    Allocate(L[j], i - shift);
    i = i + 1;
  }
}


/// WillCreateCycle - Returns true if adding an edge from SU to TargetSU will
/// create a cycle.
bool ScheduleDAGTopologicalSort::WillCreateCycle(SUnit *SU, SUnit *TargetSU) {
  if (IsReachable(TargetSU, SU))
    return true;
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I)
    if (I->isAssignedRegDep() &&
        IsReachable(TargetSU, I->getSUnit()))
      return true;
  return false;
}

/// IsReachable - Checks if SU is reachable from TargetSU.
bool ScheduleDAGTopologicalSort::IsReachable(const SUnit *SU,
                                             const SUnit *TargetSU) {
  // If insertion of the edge SU->TargetSU would create a cycle
  // then there is a path from TargetSU to SU.
  int UpperBound, LowerBound;
  LowerBound = Node2Index[TargetSU->NodeNum];
  UpperBound = Node2Index[SU->NodeNum];
  bool HasLoop = false;
  // Is Ord(TargetSU) < Ord(SU) ?
  if (LowerBound < UpperBound) {
    Visited.reset();
    // There may be a path from TargetSU to SU. Check for it. 
    DFS(TargetSU, UpperBound, HasLoop);
  }
  return HasLoop;
}

/// Allocate - assign the topological index to the node n.
void ScheduleDAGTopologicalSort::Allocate(int n, int index) {
  Node2Index[n] = index;
  Index2Node[index] = n;
}

ScheduleDAGTopologicalSort::ScheduleDAGTopologicalSort(
                                                     std::vector<SUnit> &sunits)
 : SUnits(sunits) {}
