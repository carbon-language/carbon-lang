//===----- ScheduleDAGRRList.cpp - Reg pressure reduction list scheduler --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements bottom-up and top-down register pressure reduction list
// schedulers, using standard algorithms.  The basic approach uses a priority
// queue of available nodes to schedule.  One at a time, nodes are taken from
// the priority queue (thus in priority order), checked for legality to
// schedule, and emitted if legal.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <climits>
#include "llvm/Support/CommandLine.h"
using namespace llvm;

STATISTIC(NumBacktracks, "Number of times scheduler backtracked");
STATISTIC(NumUnfolds,    "Number of nodes unfolded");
STATISTIC(NumDups,       "Number of duplicated nodes");
STATISTIC(NumCCCopies,   "Number of cross class copies");

static RegisterScheduler
  burrListDAGScheduler("list-burr",
                       "Bottom-up register reduction list scheduling",
                       createBURRListDAGScheduler);
static RegisterScheduler
  tdrListrDAGScheduler("list-tdrr",
                       "Top-down register reduction list scheduling",
                       createTDRRListDAGScheduler);

namespace {
//===----------------------------------------------------------------------===//
/// ScheduleDAGRRList - The actual register reduction list scheduler
/// implementation.  This supports both top-down and bottom-up scheduling.
///
class VISIBILITY_HIDDEN ScheduleDAGRRList : public ScheduleDAG {
private:
  /// isBottomUp - This is true if the scheduling problem is bottom-up, false if
  /// it is top-down.
  bool isBottomUp;

  /// Fast - True if we are performing fast scheduling.
  ///
  bool Fast;
  
  /// AvailableQueue - The priority queue to use for the available SUnits.
  SchedulingPriorityQueue *AvailableQueue;

  /// LiveRegDefs - A set of physical registers and their definition
  /// that are "live". These nodes must be scheduled before any other nodes that
  /// modifies the registers can be scheduled.
  unsigned NumLiveRegs;
  std::vector<SUnit*> LiveRegDefs;
  std::vector<unsigned> LiveRegCycles;

public:
  ScheduleDAGRRList(SelectionDAG *dag, MachineBasicBlock *bb,
                    const TargetMachine &tm, bool isbottomup, bool f,
                    SchedulingPriorityQueue *availqueue)
    : ScheduleDAG(dag, bb, tm), isBottomUp(isbottomup), Fast(f),
      AvailableQueue(availqueue) {
    }

  ~ScheduleDAGRRList() {
    delete AvailableQueue;
  }

  void Schedule();

  /// IsReachable - Checks if SU is reachable from TargetSU.
  bool IsReachable(const SUnit *SU, const SUnit *TargetSU);

  /// willCreateCycle - Returns true if adding an edge from SU to TargetSU will
  /// create a cycle.
  bool WillCreateCycle(SUnit *SU, SUnit *TargetSU);

  /// AddPred - This adds the specified node X as a predecessor of 
  /// the current node Y if not already.
  /// This returns true if this is a new predecessor.
  /// Updates the topological ordering if required.
  bool AddPred(SUnit *Y, SUnit *X, bool isCtrl, bool isSpecial,
               unsigned PhyReg = 0, int Cost = 1);

  /// RemovePred - This removes the specified node N from the predecessors of 
  /// the current node M. Updates the topological ordering if required.
  bool RemovePred(SUnit *M, SUnit *N, bool isCtrl, bool isSpecial);

private:
  void ReleasePred(SUnit*, bool, unsigned);
  void ReleaseSucc(SUnit*, bool isChain, unsigned);
  void CapturePred(SUnit*, SUnit*, bool);
  void ScheduleNodeBottomUp(SUnit*, unsigned);
  void ScheduleNodeTopDown(SUnit*, unsigned);
  void UnscheduleNodeBottomUp(SUnit*);
  void BacktrackBottomUp(SUnit*, unsigned, unsigned&);
  SUnit *CopyAndMoveSuccessors(SUnit*);
  void InsertCCCopiesAndMoveSuccs(SUnit*, unsigned,
                                  const TargetRegisterClass*,
                                  const TargetRegisterClass*,
                                  SmallVector<SUnit*, 2>&);
  bool DelayForLiveRegsBottomUp(SUnit*, SmallVector<unsigned, 4>&);
  void ListScheduleTopDown();
  void ListScheduleBottomUp();
  void CommuteNodesToReducePressure();


  /// CreateNewSUnit - Creates a new SUnit and returns a pointer to it.
  /// Updates the topological ordering if required.
  SUnit *CreateNewSUnit(SDNode *N) {
    SUnit *NewNode = NewSUnit(N);
    // Update the topological ordering.
    if (NewNode->NodeNum >= Node2Index.size())
      InitDAGTopologicalSorting();
    return NewNode;
  }

  /// CreateClone - Creates a new SUnit from an existing one.
  /// Updates the topological ordering if required.
  SUnit *CreateClone(SUnit *N) {
    SUnit *NewNode = Clone(N);
    // Update the topological ordering.
    if (NewNode->NodeNum >= Node2Index.size())
      InitDAGTopologicalSorting();
    return NewNode;
  }

  /// Functions for preserving the topological ordering
  /// even after dynamic insertions of new edges.
  /// This allows a very fast implementation of IsReachable.

  /// InitDAGTopologicalSorting - create the initial topological 
  /// ordering from the DAG to be scheduled.
  void InitDAGTopologicalSorting();

  /// DFS - make a DFS traversal and mark all nodes affected by the 
  /// edge insertion. These nodes will later get new topological indexes
  /// by means of the Shift method.
  void DFS(const SUnit *SU, int UpperBound, bool& HasLoop);

  /// Shift - reassign topological indexes for the nodes in the DAG
  /// to preserve the topological ordering.
  void Shift(BitVector& Visited, int LowerBound, int UpperBound);

  /// Allocate - assign the topological index to the node n.
  void Allocate(int n, int index);

  /// Index2Node - Maps topological index to the node number.
  std::vector<int> Index2Node;
  /// Node2Index - Maps the node number to its topological index.
  std::vector<int> Node2Index;
  /// Visited - a set of nodes visited during a DFS traversal.
  BitVector Visited;
};
}  // end anonymous namespace


/// Schedule - Schedule the DAG using list scheduling.
void ScheduleDAGRRList::Schedule() {
  DOUT << "********** List Scheduling **********\n";

  NumLiveRegs = 0;
  LiveRegDefs.resize(TRI->getNumRegs(), NULL);  
  LiveRegCycles.resize(TRI->getNumRegs(), 0);

  // Build scheduling units.
  BuildSchedUnits();

  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(DAG));
  if (!Fast) {
    CalculateDepths();
    CalculateHeights();
  }
  InitDAGTopologicalSorting();

  AvailableQueue->initNodes(SUnits);
  
  // Execute the actual scheduling loop Top-Down or Bottom-Up as appropriate.
  if (isBottomUp)
    ListScheduleBottomUp();
  else
    ListScheduleTopDown();
  
  AvailableQueue->releaseState();

  if (!Fast)
    CommuteNodesToReducePressure();
}

/// CommuteNodesToReducePressure - If a node is two-address and commutable, and
/// it is not the last use of its first operand, add it to the CommuteSet if
/// possible. It will be commuted when it is translated to a MI.
void ScheduleDAGRRList::CommuteNodesToReducePressure() {
  SmallPtrSet<SUnit*, 4> OperandSeen;
  for (unsigned i = Sequence.size(); i != 0; ) {
    --i;
    SUnit *SU = Sequence[i];
    if (!SU || !SU->Node) continue;
    if (SU->isCommutable) {
      unsigned Opc = SU->Node->getMachineOpcode();
      const TargetInstrDesc &TID = TII->get(Opc);
      unsigned NumRes = TID.getNumDefs();
      unsigned NumOps = TID.getNumOperands() - NumRes;
      for (unsigned j = 0; j != NumOps; ++j) {
        if (TID.getOperandConstraint(j+NumRes, TOI::TIED_TO) == -1)
          continue;

        SDNode *OpN = SU->Node->getOperand(j).getNode();
        SUnit *OpSU = isPassiveNode(OpN) ? NULL : &SUnits[OpN->getNodeId()];
        if (OpSU && OperandSeen.count(OpSU) == 1) {
          // Ok, so SU is not the last use of OpSU, but SU is two-address so
          // it will clobber OpSU. Try to commute SU if no other source operands
          // are live below.
          bool DoCommute = true;
          for (unsigned k = 0; k < NumOps; ++k) {
            if (k != j) {
              OpN = SU->Node->getOperand(k).getNode();
              OpSU = isPassiveNode(OpN) ? NULL : &SUnits[OpN->getNodeId()];
              if (OpSU && OperandSeen.count(OpSU) == 1) {
                DoCommute = false;
                break;
              }
            }
          }
          if (DoCommute)
            CommuteSet.insert(SU->Node);
        }

        // Only look at the first use&def node for now.
        break;
      }
    }

    for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      if (!I->isCtrl)
        OperandSeen.insert(I->Dep->OrigNode);
    }
  }
}

//===----------------------------------------------------------------------===//
//  Bottom-Up Scheduling
//===----------------------------------------------------------------------===//

/// ReleasePred - Decrement the NumSuccsLeft count of a predecessor. Add it to
/// the AvailableQueue if the count reaches zero. Also update its cycle bound.
void ScheduleDAGRRList::ReleasePred(SUnit *PredSU, bool isChain, 
                                    unsigned CurCycle) {
  // FIXME: the distance between two nodes is not always == the predecessor's
  // latency. For example, the reader can very well read the register written
  // by the predecessor later than the issue cycle. It also depends on the
  // interrupt model (drain vs. freeze).
  PredSU->CycleBound = std::max(PredSU->CycleBound, CurCycle + PredSU->Latency);

  --PredSU->NumSuccsLeft;
  
#ifndef NDEBUG
  if (PredSU->NumSuccsLeft < 0) {
    cerr << "*** List scheduling failed! ***\n";
    PredSU->dump(DAG);
    cerr << " has been released too many times!\n";
    assert(0);
  }
#endif
  
  if (PredSU->NumSuccsLeft == 0) {
    PredSU->isAvailable = true;
    AvailableQueue->push(PredSU);
  }
}

/// ScheduleNodeBottomUp - Add the node to the schedule. Decrement the pending
/// count of its predecessors. If a predecessor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGRRList::ScheduleNodeBottomUp(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(DAG));
  SU->Cycle = CurCycle;

  AvailableQueue->ScheduledNode(SU);

  // Bottom up: release predecessors
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    ReleasePred(I->Dep, I->isCtrl, CurCycle);
    if (I->Cost < 0)  {
      // This is a physical register dependency and it's impossible or
      // expensive to copy the register. Make sure nothing that can 
      // clobber the register is scheduled between the predecessor and
      // this node.
      if (!LiveRegDefs[I->Reg]) {
        ++NumLiveRegs;
        LiveRegDefs[I->Reg] = I->Dep;
        LiveRegCycles[I->Reg] = CurCycle;
      }
    }
  }

  // Release all the implicit physical register defs that are live.
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->Cost < 0)  {
      if (LiveRegCycles[I->Reg] == I->Dep->Cycle) {
        assert(NumLiveRegs > 0 && "NumLiveRegs is already zero!");
        assert(LiveRegDefs[I->Reg] == SU &&
               "Physical register dependency violated?");
        --NumLiveRegs;
        LiveRegDefs[I->Reg] = NULL;
        LiveRegCycles[I->Reg] = 0;
      }
    }
  }

  SU->isScheduled = true;
}

/// CapturePred - This does the opposite of ReleasePred. Since SU is being
/// unscheduled, incrcease the succ left count of its predecessors. Remove
/// them from AvailableQueue if necessary.
void ScheduleDAGRRList::CapturePred(SUnit *PredSU, SUnit *SU, bool isChain) {  
  unsigned CycleBound = 0;
  for (SUnit::succ_iterator I = PredSU->Succs.begin(), E = PredSU->Succs.end();
       I != E; ++I) {
    if (I->Dep == SU)
      continue;
    CycleBound = std::max(CycleBound,
                          I->Dep->Cycle + PredSU->Latency);
  }

  if (PredSU->isAvailable) {
    PredSU->isAvailable = false;
    if (!PredSU->isPending)
      AvailableQueue->remove(PredSU);
  }

  PredSU->CycleBound = CycleBound;
  ++PredSU->NumSuccsLeft;
}

/// UnscheduleNodeBottomUp - Remove the node from the schedule, update its and
/// its predecessor states to reflect the change.
void ScheduleDAGRRList::UnscheduleNodeBottomUp(SUnit *SU) {
  DOUT << "*** Unscheduling [" << SU->Cycle << "]: ";
  DEBUG(SU->dump(DAG));

  AvailableQueue->UnscheduledNode(SU);

  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    CapturePred(I->Dep, SU, I->isCtrl);
    if (I->Cost < 0 && SU->Cycle == LiveRegCycles[I->Reg])  {
      assert(NumLiveRegs > 0 && "NumLiveRegs is already zero!");
      assert(LiveRegDefs[I->Reg] == I->Dep &&
             "Physical register dependency violated?");
      --NumLiveRegs;
      LiveRegDefs[I->Reg] = NULL;
      LiveRegCycles[I->Reg] = 0;
    }
  }

  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->Cost < 0)  {
      if (!LiveRegDefs[I->Reg]) {
        LiveRegDefs[I->Reg] = SU;
        ++NumLiveRegs;
      }
      if (I->Dep->Cycle < LiveRegCycles[I->Reg])
        LiveRegCycles[I->Reg] = I->Dep->Cycle;
    }
  }

  SU->Cycle = 0;
  SU->isScheduled = false;
  SU->isAvailable = true;
  AvailableQueue->push(SU);
}

/// IsReachable - Checks if SU is reachable from TargetSU.
bool ScheduleDAGRRList::IsReachable(const SUnit *SU, const SUnit *TargetSU) {
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
inline void ScheduleDAGRRList::Allocate(int n, int index) {
  Node2Index[n] = index;
  Index2Node[index] = n;
}

/// InitDAGTopologicalSorting - create the initial topological 
/// ordering from the DAG to be scheduled.

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
void ScheduleDAGRRList::InitDAGTopologicalSorting() {
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
      SUnit *SU = I->Dep;
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
       assert(Node2Index[SU->NodeNum] > Node2Index[I->Dep->NodeNum] && 
       "Wrong topological sorting");
    }
  }
#endif
}

/// AddPred - adds an edge from SUnit X to SUnit Y.
/// Updates the topological ordering if required.
bool ScheduleDAGRRList::AddPred(SUnit *Y, SUnit *X, bool isCtrl, bool isSpecial,
                 unsigned PhyReg, int Cost) {
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
  // Now really insert the edge.
  return Y->addPred(X, isCtrl, isSpecial, PhyReg, Cost);
}

/// RemovePred - This removes the specified node N from the predecessors of 
/// the current node M. Updates the topological ordering if required.
bool ScheduleDAGRRList::RemovePred(SUnit *M, SUnit *N, 
                                   bool isCtrl, bool isSpecial) {
  // InitDAGTopologicalSorting();
  return M->removePred(N, isCtrl, isSpecial);
}

/// DFS - Make a DFS traversal to mark all nodes reachable from SU and mark
/// all nodes affected by the edge insertion. These nodes will later get new
/// topological indexes by means of the Shift method.
void ScheduleDAGRRList::DFS(const SUnit *SU, int UpperBound, bool& HasLoop) {
  std::vector<const SUnit*> WorkList;
  WorkList.reserve(SUnits.size()); 

  WorkList.push_back(SU);
  while (!WorkList.empty()) {
    SU = WorkList.back();
    WorkList.pop_back();
    Visited.set(SU->NodeNum);
    for (int I = SU->Succs.size()-1; I >= 0; --I) {
      int s = SU->Succs[I].Dep->NodeNum;
      if (Node2Index[s] == UpperBound) {
        HasLoop = true; 
        return;
      }
      // Visit successors if not already and in affected region.
      if (!Visited.test(s) && Node2Index[s] < UpperBound) {
        WorkList.push_back(SU->Succs[I].Dep);
      } 
    } 
  }
}

/// Shift - Renumber the nodes so that the topological ordering is 
/// preserved.
void ScheduleDAGRRList::Shift(BitVector& Visited, int LowerBound, 
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
bool ScheduleDAGRRList::WillCreateCycle(SUnit *SU, SUnit *TargetSU) {
  if (IsReachable(TargetSU, SU))
    return true;
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I)
    if (I->Cost < 0 && IsReachable(TargetSU, I->Dep))
      return true;
  return false;
}

/// BacktrackBottomUp - Backtrack scheduling to a previous cycle specified in
/// BTCycle in order to schedule a specific node. Returns the last unscheduled
/// SUnit. Also returns if a successor is unscheduled in the process.
void ScheduleDAGRRList::BacktrackBottomUp(SUnit *SU, unsigned BtCycle,
                                          unsigned &CurCycle) {
  SUnit *OldSU = NULL;
  while (CurCycle > BtCycle) {
    OldSU = Sequence.back();
    Sequence.pop_back();
    if (SU->isSucc(OldSU))
      // Don't try to remove SU from AvailableQueue.
      SU->isAvailable = false;
    UnscheduleNodeBottomUp(OldSU);
    --CurCycle;
  }

      
  if (SU->isSucc(OldSU)) {
    assert(false && "Something is wrong!");
    abort();
  }

  ++NumBacktracks;
}

/// CopyAndMoveSuccessors - Clone the specified node and move its scheduled
/// successors to the newly created node.
SUnit *ScheduleDAGRRList::CopyAndMoveSuccessors(SUnit *SU) {
  if (SU->FlaggedNodes.size())
    return NULL;

  SDNode *N = SU->Node;
  if (!N)
    return NULL;

  SUnit *NewSU;
  bool TryUnfold = false;
  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i) {
    MVT VT = N->getValueType(i);
    if (VT == MVT::Flag)
      return NULL;
    else if (VT == MVT::Other)
      TryUnfold = true;
  }
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    const SDValue &Op = N->getOperand(i);
    MVT VT = Op.getNode()->getValueType(Op.getResNo());
    if (VT == MVT::Flag)
      return NULL;
  }

  if (TryUnfold) {
    SmallVector<SDNode*, 2> NewNodes;
    if (!TII->unfoldMemoryOperand(*DAG, N, NewNodes))
      return NULL;

    DOUT << "Unfolding SU # " << SU->NodeNum << "\n";
    assert(NewNodes.size() == 2 && "Expected a load folding node!");

    N = NewNodes[1];
    SDNode *LoadNode = NewNodes[0];
    unsigned NumVals = N->getNumValues();
    unsigned OldNumVals = SU->Node->getNumValues();
    for (unsigned i = 0; i != NumVals; ++i)
      DAG->ReplaceAllUsesOfValueWith(SDValue(SU->Node, i), SDValue(N, i));
    DAG->ReplaceAllUsesOfValueWith(SDValue(SU->Node, OldNumVals-1),
                                   SDValue(LoadNode, 1));

    // LoadNode may already exist. This can happen when there is another
    // load from the same location and producing the same type of value
    // but it has different alignment or volatileness.
    bool isNewLoad = true;
    SUnit *LoadSU;
    if (LoadNode->getNodeId() != -1) {
      LoadSU = &SUnits[LoadNode->getNodeId()];
      isNewLoad = false;
    } else {
      LoadSU = CreateNewSUnit(LoadNode);
      LoadNode->setNodeId(LoadSU->NodeNum);

      LoadSU->Depth = SU->Depth;
      LoadSU->Height = SU->Height;
      ComputeLatency(LoadSU);
    }

    SUnit *NewSU = CreateNewSUnit(N);
    assert(N->getNodeId() == -1 && "Node already inserted!");
    N->setNodeId(NewSU->NodeNum);
      
    const TargetInstrDesc &TID = TII->get(N->getMachineOpcode());
    for (unsigned i = 0; i != TID.getNumOperands(); ++i) {
      if (TID.getOperandConstraint(i, TOI::TIED_TO) != -1) {
        NewSU->isTwoAddress = true;
        break;
      }
    }
    if (TID.isCommutable())
      NewSU->isCommutable = true;
    // FIXME: Calculate height / depth and propagate the changes?
    NewSU->Depth = SU->Depth;
    NewSU->Height = SU->Height;
    ComputeLatency(NewSU);

    SUnit *ChainPred = NULL;
    SmallVector<SDep, 4> ChainSuccs;
    SmallVector<SDep, 4> LoadPreds;
    SmallVector<SDep, 4> NodePreds;
    SmallVector<SDep, 4> NodeSuccs;
    for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      if (I->isCtrl)
        ChainPred = I->Dep;
      else if (I->Dep->Node && I->Dep->Node->isOperandOf(LoadNode))
        LoadPreds.push_back(SDep(I->Dep, I->Reg, I->Cost, false, false));
      else
        NodePreds.push_back(SDep(I->Dep, I->Reg, I->Cost, false, false));
    }
    for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      if (I->isCtrl)
        ChainSuccs.push_back(SDep(I->Dep, I->Reg, I->Cost,
                                  I->isCtrl, I->isSpecial));
      else
        NodeSuccs.push_back(SDep(I->Dep, I->Reg, I->Cost,
                                 I->isCtrl, I->isSpecial));
    }

    if (ChainPred) {
      RemovePred(SU, ChainPred, true, false);
      if (isNewLoad)
        AddPred(LoadSU, ChainPred, true, false);
    }
    for (unsigned i = 0, e = LoadPreds.size(); i != e; ++i) {
      SDep *Pred = &LoadPreds[i];
      RemovePred(SU, Pred->Dep, Pred->isCtrl, Pred->isSpecial);
      if (isNewLoad) {
        AddPred(LoadSU, Pred->Dep, Pred->isCtrl, Pred->isSpecial,
                Pred->Reg, Pred->Cost);
      }
    }
    for (unsigned i = 0, e = NodePreds.size(); i != e; ++i) {
      SDep *Pred = &NodePreds[i];
      RemovePred(SU, Pred->Dep, Pred->isCtrl, Pred->isSpecial);
      AddPred(NewSU, Pred->Dep, Pred->isCtrl, Pred->isSpecial,
              Pred->Reg, Pred->Cost);
    }
    for (unsigned i = 0, e = NodeSuccs.size(); i != e; ++i) {
      SDep *Succ = &NodeSuccs[i];
      RemovePred(Succ->Dep, SU, Succ->isCtrl, Succ->isSpecial);
      AddPred(Succ->Dep, NewSU, Succ->isCtrl, Succ->isSpecial,
              Succ->Reg, Succ->Cost);
    }
    for (unsigned i = 0, e = ChainSuccs.size(); i != e; ++i) {
      SDep *Succ = &ChainSuccs[i];
      RemovePred(Succ->Dep, SU, Succ->isCtrl, Succ->isSpecial);
      if (isNewLoad) {
        AddPred(Succ->Dep, LoadSU, Succ->isCtrl, Succ->isSpecial,
                Succ->Reg, Succ->Cost);
      }
    } 
    if (isNewLoad) {
      AddPred(NewSU, LoadSU, false, false);
    }

    if (isNewLoad)
      AvailableQueue->addNode(LoadSU);
    AvailableQueue->addNode(NewSU);

    ++NumUnfolds;

    if (NewSU->NumSuccsLeft == 0) {
      NewSU->isAvailable = true;
      return NewSU;
    }
    SU = NewSU;
  }

  DOUT << "Duplicating SU # " << SU->NodeNum << "\n";
  NewSU = CreateClone(SU);

  // New SUnit has the exact same predecessors.
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I)
    if (!I->isSpecial) {
      AddPred(NewSU, I->Dep, I->isCtrl, false, I->Reg, I->Cost);
      NewSU->Depth = std::max(NewSU->Depth, I->Dep->Depth+1);
    }

  // Only copy scheduled successors. Cut them from old node's successor
  // list and move them over.
  SmallVector<std::pair<SUnit*, bool>, 4> DelDeps;
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isSpecial)
      continue;
    if (I->Dep->isScheduled) {
      NewSU->Height = std::max(NewSU->Height, I->Dep->Height+1);
      AddPred(I->Dep, NewSU, I->isCtrl, false, I->Reg, I->Cost);
      DelDeps.push_back(std::make_pair(I->Dep, I->isCtrl));
    }
  }
  for (unsigned i = 0, e = DelDeps.size(); i != e; ++i) {
    SUnit *Succ = DelDeps[i].first;
    bool isCtrl = DelDeps[i].second;
    RemovePred(Succ, SU, isCtrl, false);
  }

  AvailableQueue->updateNode(SU);
  AvailableQueue->addNode(NewSU);

  ++NumDups;
  return NewSU;
}

/// InsertCCCopiesAndMoveSuccs - Insert expensive cross register class copies
/// and move all scheduled successors of the given SUnit to the last copy.
void ScheduleDAGRRList::InsertCCCopiesAndMoveSuccs(SUnit *SU, unsigned Reg,
                                              const TargetRegisterClass *DestRC,
                                              const TargetRegisterClass *SrcRC,
                                               SmallVector<SUnit*, 2> &Copies) {
  SUnit *CopyFromSU = CreateNewSUnit(NULL);
  CopyFromSU->CopySrcRC = SrcRC;
  CopyFromSU->CopyDstRC = DestRC;
  CopyFromSU->Depth = SU->Depth;
  CopyFromSU->Height = SU->Height;

  SUnit *CopyToSU = CreateNewSUnit(NULL);
  CopyToSU->CopySrcRC = DestRC;
  CopyToSU->CopyDstRC = SrcRC;

  // Only copy scheduled successors. Cut them from old node's successor
  // list and move them over.
  SmallVector<std::pair<SUnit*, bool>, 4> DelDeps;
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isSpecial)
      continue;
    if (I->Dep->isScheduled) {
      CopyToSU->Height = std::max(CopyToSU->Height, I->Dep->Height+1);
      AddPred(I->Dep, CopyToSU, I->isCtrl, false, I->Reg, I->Cost);
      DelDeps.push_back(std::make_pair(I->Dep, I->isCtrl));
    }
  }
  for (unsigned i = 0, e = DelDeps.size(); i != e; ++i) {
    SUnit *Succ = DelDeps[i].first;
    bool isCtrl = DelDeps[i].second;
    RemovePred(Succ, SU, isCtrl, false);
  }

  AddPred(CopyFromSU, SU, false, false, Reg, -1);
  AddPred(CopyToSU, CopyFromSU, false, false, Reg, 1);

  AvailableQueue->updateNode(SU);
  AvailableQueue->addNode(CopyFromSU);
  AvailableQueue->addNode(CopyToSU);
  Copies.push_back(CopyFromSU);
  Copies.push_back(CopyToSU);

  ++NumCCCopies;
}

/// getPhysicalRegisterVT - Returns the ValueType of the physical register
/// definition of the specified node.
/// FIXME: Move to SelectionDAG?
static MVT getPhysicalRegisterVT(SDNode *N, unsigned Reg,
                                 const TargetInstrInfo *TII) {
  const TargetInstrDesc &TID = TII->get(N->getMachineOpcode());
  assert(TID.ImplicitDefs && "Physical reg def must be in implicit def list!");
  unsigned NumRes = TID.getNumDefs();
  for (const unsigned *ImpDef = TID.getImplicitDefs(); *ImpDef; ++ImpDef) {
    if (Reg == *ImpDef)
      break;
    ++NumRes;
  }
  return N->getValueType(NumRes);
}

/// DelayForLiveRegsBottomUp - Returns true if it is necessary to delay
/// scheduling of the given node to satisfy live physical register dependencies.
/// If the specific node is the last one that's available to schedule, do
/// whatever is necessary (i.e. backtracking or cloning) to make it possible.
bool ScheduleDAGRRList::DelayForLiveRegsBottomUp(SUnit *SU,
                                                 SmallVector<unsigned, 4> &LRegs){
  if (NumLiveRegs == 0)
    return false;

  SmallSet<unsigned, 4> RegAdded;
  // If this node would clobber any "live" register, then it's not ready.
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->Cost < 0)  {
      unsigned Reg = I->Reg;
      if (LiveRegDefs[Reg] && LiveRegDefs[Reg] != I->Dep) {
        if (RegAdded.insert(Reg))
          LRegs.push_back(Reg);
      }
      for (const unsigned *Alias = TRI->getAliasSet(Reg);
           *Alias; ++Alias)
        if (LiveRegDefs[*Alias] && LiveRegDefs[*Alias] != I->Dep) {
          if (RegAdded.insert(*Alias))
            LRegs.push_back(*Alias);
        }
    }
  }

  for (unsigned i = 0, e = SU->FlaggedNodes.size()+1; i != e; ++i) {
    SDNode *Node = (i == 0) ? SU->Node : SU->FlaggedNodes[i-1];
    if (!Node || !Node->isMachineOpcode())
      continue;
    const TargetInstrDesc &TID = TII->get(Node->getMachineOpcode());
    if (!TID.ImplicitDefs)
      continue;
    for (const unsigned *Reg = TID.ImplicitDefs; *Reg; ++Reg) {
      if (LiveRegDefs[*Reg] && LiveRegDefs[*Reg] != SU) {
        if (RegAdded.insert(*Reg))
          LRegs.push_back(*Reg);
      }
      for (const unsigned *Alias = TRI->getAliasSet(*Reg);
           *Alias; ++Alias)
        if (LiveRegDefs[*Alias] && LiveRegDefs[*Alias] != SU) {
          if (RegAdded.insert(*Alias))
            LRegs.push_back(*Alias);
        }
    }
  }
  return !LRegs.empty();
}


/// ListScheduleBottomUp - The main loop of list scheduling for bottom-up
/// schedulers.
void ScheduleDAGRRList::ListScheduleBottomUp() {
  unsigned CurCycle = 0;
  // Add root to Available queue.
  if (!SUnits.empty()) {
    SUnit *RootSU = &SUnits[DAG->getRoot().getNode()->getNodeId()];
    assert(RootSU->Succs.empty() && "Graph root shouldn't have successors!");
    RootSU->isAvailable = true;
    AvailableQueue->push(RootSU);
  }

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  SmallVector<SUnit*, 4> NotReady;
  DenseMap<SUnit*, SmallVector<unsigned, 4> > LRegsMap;
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue->empty()) {
    bool Delayed = false;
    LRegsMap.clear();
    SUnit *CurSU = AvailableQueue->pop();
    while (CurSU) {
      if (CurSU->CycleBound <= CurCycle) {
        SmallVector<unsigned, 4> LRegs;
        if (!DelayForLiveRegsBottomUp(CurSU, LRegs))
          break;
        Delayed = true;
        LRegsMap.insert(std::make_pair(CurSU, LRegs));
      }

      CurSU->isPending = true;  // This SU is not in AvailableQueue right now.
      NotReady.push_back(CurSU);
      CurSU = AvailableQueue->pop();
    }

    // All candidates are delayed due to live physical reg dependencies.
    // Try backtracking, code duplication, or inserting cross class copies
    // to resolve it.
    if (Delayed && !CurSU) {
      for (unsigned i = 0, e = NotReady.size(); i != e; ++i) {
        SUnit *TrySU = NotReady[i];
        SmallVector<unsigned, 4> &LRegs = LRegsMap[TrySU];

        // Try unscheduling up to the point where it's safe to schedule
        // this node.
        unsigned LiveCycle = CurCycle;
        for (unsigned j = 0, ee = LRegs.size(); j != ee; ++j) {
          unsigned Reg = LRegs[j];
          unsigned LCycle = LiveRegCycles[Reg];
          LiveCycle = std::min(LiveCycle, LCycle);
        }
        SUnit *OldSU = Sequence[LiveCycle];
        if (!WillCreateCycle(TrySU, OldSU))  {
          BacktrackBottomUp(TrySU, LiveCycle, CurCycle);
          // Force the current node to be scheduled before the node that
          // requires the physical reg dep.
          if (OldSU->isAvailable) {
            OldSU->isAvailable = false;
            AvailableQueue->remove(OldSU);
          }
          AddPred(TrySU, OldSU, true, true);
          // If one or more successors has been unscheduled, then the current
          // node is no longer avaialable. Schedule a successor that's now
          // available instead.
          if (!TrySU->isAvailable)
            CurSU = AvailableQueue->pop();
          else {
            CurSU = TrySU;
            TrySU->isPending = false;
            NotReady.erase(NotReady.begin()+i);
          }
          break;
        }
      }

      if (!CurSU) {
        // Can't backtrack. Try duplicating the nodes that produces these
        // "expensive to copy" values to break the dependency. In case even
        // that doesn't work, insert cross class copies.
        SUnit *TrySU = NotReady[0];
        SmallVector<unsigned, 4> &LRegs = LRegsMap[TrySU];
        assert(LRegs.size() == 1 && "Can't handle this yet!");
        unsigned Reg = LRegs[0];
        SUnit *LRDef = LiveRegDefs[Reg];
        SUnit *NewDef = CopyAndMoveSuccessors(LRDef);
        if (!NewDef) {
          // Issue expensive cross register class copies.
          MVT VT = getPhysicalRegisterVT(LRDef->Node, Reg, TII);
          const TargetRegisterClass *RC =
            TRI->getPhysicalRegisterRegClass(Reg, VT);
          const TargetRegisterClass *DestRC = TRI->getCrossCopyRegClass(RC);
          if (!DestRC) {
            assert(false && "Don't know how to copy this physical register!");
            abort();
          }
          SmallVector<SUnit*, 2> Copies;
          InsertCCCopiesAndMoveSuccs(LRDef, Reg, DestRC, RC, Copies);
          DOUT << "Adding an edge from SU # " << TrySU->NodeNum
               << " to SU #" << Copies.front()->NodeNum << "\n";
          AddPred(TrySU, Copies.front(), true, true);
          NewDef = Copies.back();
        }

        DOUT << "Adding an edge from SU # " << NewDef->NodeNum
             << " to SU #" << TrySU->NodeNum << "\n";
        LiveRegDefs[Reg] = NewDef;
        AddPred(NewDef, TrySU, true, true);
        TrySU->isAvailable = false;
        CurSU = NewDef;
      }

      if (!CurSU) {
        assert(false && "Unable to resolve live physical register dependencies!");
        abort();
      }
    }

    // Add the nodes that aren't ready back onto the available list.
    for (unsigned i = 0, e = NotReady.size(); i != e; ++i) {
      NotReady[i]->isPending = false;
      // May no longer be available due to backtracking.
      if (NotReady[i]->isAvailable)
        AvailableQueue->push(NotReady[i]);
    }
    NotReady.clear();

    if (!CurSU)
      Sequence.push_back(0);
    else {
      ScheduleNodeBottomUp(CurSU, CurCycle);
      Sequence.push_back(CurSU);
    }
    ++CurCycle;
  }

  // Reverse the order if it is bottom up.
  std::reverse(Sequence.begin(), Sequence.end());
  
  
#ifndef NDEBUG
  // Verify that all SUnits were scheduled.
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
        cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(DAG);
      cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
    if (SUnits[i].NumSuccsLeft != 0) {
      if (!AnyNotSched)
        cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(DAG);
      cerr << "has successors left!\n";
      AnyNotSched = true;
    }
  }
  for (unsigned i = 0, e = Sequence.size(); i != e; ++i)
    if (!Sequence[i])
      ++Noops;
  assert(!AnyNotSched);
  assert(Sequence.size() + DeadNodes - Noops == SUnits.size() &&
         "The number of nodes scheduled doesn't match the expected number!");
#endif
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the AvailableQueue if the count reaches zero. Also update its cycle bound.
void ScheduleDAGRRList::ReleaseSucc(SUnit *SuccSU, bool isChain, 
                                    unsigned CurCycle) {
  // FIXME: the distance between two nodes is not always == the predecessor's
  // latency. For example, the reader can very well read the register written
  // by the predecessor later than the issue cycle. It also depends on the
  // interrupt model (drain vs. freeze).
  SuccSU->CycleBound = std::max(SuccSU->CycleBound, CurCycle + SuccSU->Latency);

  --SuccSU->NumPredsLeft;
  
#ifndef NDEBUG
  if (SuccSU->NumPredsLeft < 0) {
    cerr << "*** List scheduling failed! ***\n";
    SuccSU->dump(DAG);
    cerr << " has been released too many times!\n";
    assert(0);
  }
#endif
  
  if (SuccSU->NumPredsLeft == 0) {
    SuccSU->isAvailable = true;
    AvailableQueue->push(SuccSU);
  }
}


/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGRRList::ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(DAG));
  SU->Cycle = CurCycle;

  AvailableQueue->ScheduledNode(SU);

  // Top down: release successors
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I)
    ReleaseSucc(I->Dep, I->isCtrl, CurCycle);
  SU->isScheduled = true;
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void ScheduleDAGRRList::ListScheduleTopDown() {
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
  while (!AvailableQueue->empty()) {
    SUnit *CurSU = AvailableQueue->pop();
    while (CurSU && CurSU->CycleBound > CurCycle) {
      NotReady.push_back(CurSU);
      CurSU = AvailableQueue->pop();
    }
    
    // Add the nodes that aren't ready back onto the available list.
    AvailableQueue->push_all(NotReady);
    NotReady.clear();

    if (!CurSU)
      Sequence.push_back(0);
    else {
      ScheduleNodeTopDown(CurSU, CurCycle);
      Sequence.push_back(CurSU);
    }
    ++CurCycle;
  }
  
  
#ifndef NDEBUG
  // Verify that all SUnits were scheduled.
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
        cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(DAG);
      cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
    if (SUnits[i].NumPredsLeft != 0) {
      if (!AnyNotSched)
        cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(DAG);
      cerr << "has predecessors left!\n";
      AnyNotSched = true;
    }
  }
  for (unsigned i = 0, e = Sequence.size(); i != e; ++i)
    if (!Sequence[i])
      ++Noops;
  assert(!AnyNotSched);
  assert(Sequence.size() + DeadNodes - Noops == SUnits.size() &&
         "The number of nodes scheduled doesn't match the expected number!");
#endif
}



//===----------------------------------------------------------------------===//
//                RegReductionPriorityQueue Implementation
//===----------------------------------------------------------------------===//
//
// This is a SchedulingPriorityQueue that schedules using Sethi Ullman numbers
// to reduce register pressure.
// 
namespace {
  template<class SF>
  class RegReductionPriorityQueue;
  
  /// Sorting functions for the Available queue.
  struct bu_ls_rr_sort : public std::binary_function<SUnit*, SUnit*, bool> {
    RegReductionPriorityQueue<bu_ls_rr_sort> *SPQ;
    bu_ls_rr_sort(RegReductionPriorityQueue<bu_ls_rr_sort> *spq) : SPQ(spq) {}
    bu_ls_rr_sort(const bu_ls_rr_sort &RHS) : SPQ(RHS.SPQ) {}
    
    bool operator()(const SUnit* left, const SUnit* right) const;
  };

  struct bu_ls_rr_fast_sort : public std::binary_function<SUnit*, SUnit*, bool>{
    RegReductionPriorityQueue<bu_ls_rr_fast_sort> *SPQ;
    bu_ls_rr_fast_sort(RegReductionPriorityQueue<bu_ls_rr_fast_sort> *spq)
      : SPQ(spq) {}
    bu_ls_rr_fast_sort(const bu_ls_rr_fast_sort &RHS) : SPQ(RHS.SPQ) {}
    
    bool operator()(const SUnit* left, const SUnit* right) const;
  };

  struct td_ls_rr_sort : public std::binary_function<SUnit*, SUnit*, bool> {
    RegReductionPriorityQueue<td_ls_rr_sort> *SPQ;
    td_ls_rr_sort(RegReductionPriorityQueue<td_ls_rr_sort> *spq) : SPQ(spq) {}
    td_ls_rr_sort(const td_ls_rr_sort &RHS) : SPQ(RHS.SPQ) {}
    
    bool operator()(const SUnit* left, const SUnit* right) const;
  };
}  // end anonymous namespace

static inline bool isCopyFromLiveIn(const SUnit *SU) {
  SDNode *N = SU->Node;
  return N && N->getOpcode() == ISD::CopyFromReg &&
    N->getOperand(N->getNumOperands()-1).getValueType() != MVT::Flag;
}

/// CalcNodeBUSethiUllmanNumber - Compute Sethi Ullman number for bottom up
/// scheduling. Smaller number is the higher priority.
static unsigned
CalcNodeBUSethiUllmanNumber(const SUnit *SU, std::vector<unsigned> &SUNumbers) {
  unsigned &SethiUllmanNumber = SUNumbers[SU->NodeNum];
  if (SethiUllmanNumber != 0)
    return SethiUllmanNumber;

  unsigned Extra = 0;
  for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl) continue;  // ignore chain preds
    SUnit *PredSU = I->Dep;
    unsigned PredSethiUllman = CalcNodeBUSethiUllmanNumber(PredSU, SUNumbers);
    if (PredSethiUllman > SethiUllmanNumber) {
      SethiUllmanNumber = PredSethiUllman;
      Extra = 0;
    } else if (PredSethiUllman == SethiUllmanNumber && !I->isCtrl)
      ++Extra;
  }

  SethiUllmanNumber += Extra;

  if (SethiUllmanNumber == 0)
    SethiUllmanNumber = 1;
  
  return SethiUllmanNumber;
}

/// CalcNodeTDSethiUllmanNumber - Compute Sethi Ullman number for top down
/// scheduling. Smaller number is the higher priority.
static unsigned
CalcNodeTDSethiUllmanNumber(const SUnit *SU, std::vector<unsigned> &SUNumbers) {
  unsigned &SethiUllmanNumber = SUNumbers[SU->NodeNum];
  if (SethiUllmanNumber != 0)
    return SethiUllmanNumber;

  unsigned Opc = SU->Node ? SU->Node->getOpcode() : 0;
  if (Opc == ISD::TokenFactor || Opc == ISD::CopyToReg)
    SethiUllmanNumber = 0xffff;
  else if (SU->NumSuccsLeft == 0)
    // If SU does not have a use, i.e. it doesn't produce a value that would
    // be consumed (e.g. store), then it terminates a chain of computation.
    // Give it a small SethiUllman number so it will be scheduled right before
    // its predecessors that it doesn't lengthen their live ranges.
    SethiUllmanNumber = 0;
  else if (SU->NumPredsLeft == 0 &&
           (Opc != ISD::CopyFromReg || isCopyFromLiveIn(SU)))
    SethiUllmanNumber = 0xffff;
  else {
    int Extra = 0;
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      if (I->isCtrl) continue;  // ignore chain preds
      SUnit *PredSU = I->Dep;
      unsigned PredSethiUllman = CalcNodeTDSethiUllmanNumber(PredSU, SUNumbers);
      if (PredSethiUllman > SethiUllmanNumber) {
        SethiUllmanNumber = PredSethiUllman;
        Extra = 0;
      } else if (PredSethiUllman == SethiUllmanNumber && !I->isCtrl)
        ++Extra;
    }

    SethiUllmanNumber += Extra;
  }
  
  return SethiUllmanNumber;
}


namespace {
  template<class SF>
  class VISIBILITY_HIDDEN RegReductionPriorityQueue
   : public SchedulingPriorityQueue {
    PriorityQueue<SUnit*, std::vector<SUnit*>, SF> Queue;
    unsigned currentQueueId;

  public:
    RegReductionPriorityQueue() :
    Queue(SF(this)), currentQueueId(0) {}
    
    virtual void initNodes(std::vector<SUnit> &sunits) = 0;

    virtual void addNode(const SUnit *SU) = 0;

    virtual void updateNode(const SUnit *SU) = 0;

    virtual void releaseState() = 0;
    
    virtual unsigned getNodePriority(const SUnit *SU) const = 0;
    
    unsigned size() const { return Queue.size(); }

    bool empty() const { return Queue.empty(); }
    
    void push(SUnit *U) {
      assert(!U->NodeQueueId && "Node in the queue already");
      U->NodeQueueId = ++currentQueueId;
      Queue.push(U);
    }

    void push_all(const std::vector<SUnit *> &Nodes) {
      for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
        push(Nodes[i]);
    }
    
    SUnit *pop() {
      if (empty()) return NULL;
      SUnit *V = Queue.top();
      Queue.pop();
      V->NodeQueueId = 0;
      return V;
    }

    void remove(SUnit *SU) {
      assert(!Queue.empty() && "Queue is empty!");
      assert(SU->NodeQueueId != 0 && "Not in queue!");
      Queue.erase_one(SU);
      SU->NodeQueueId = 0;
    }
  };

  class VISIBILITY_HIDDEN BURegReductionPriorityQueue
   : public RegReductionPriorityQueue<bu_ls_rr_sort> {
    // SUnits - The SUnits for the current graph.
    std::vector<SUnit> *SUnits;
    
    // SethiUllmanNumbers - The SethiUllman number for each node.
    std::vector<unsigned> SethiUllmanNumbers;

    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    ScheduleDAGRRList *scheduleDAG;

  public:
    explicit BURegReductionPriorityQueue(const TargetInstrInfo *tii,
                                         const TargetRegisterInfo *tri)
      : TII(tii), TRI(tri), scheduleDAG(NULL) {}

    void initNodes(std::vector<SUnit> &sunits) {
      SUnits = &sunits;
      // Add pseudo dependency edges for two-address nodes.
      AddPseudoTwoAddrDeps();
      // Calculate node priorities.
      CalculateSethiUllmanNumbers();
    }

    void addNode(const SUnit *SU) {
      unsigned SUSize = SethiUllmanNumbers.size();
      if (SUnits->size() > SUSize)
        SethiUllmanNumbers.resize(SUSize*2, 0);
      CalcNodeBUSethiUllmanNumber(SU, SethiUllmanNumbers);
    }

    void updateNode(const SUnit *SU) {
      SethiUllmanNumbers[SU->NodeNum] = 0;
      CalcNodeBUSethiUllmanNumber(SU, SethiUllmanNumbers);
    }

    void releaseState() {
      SUnits = 0;
      SethiUllmanNumbers.clear();
    }

    unsigned getNodePriority(const SUnit *SU) const {
      assert(SU->NodeNum < SethiUllmanNumbers.size());
      unsigned Opc = SU->Node ? SU->Node->getOpcode() : 0;
      if (Opc == ISD::CopyFromReg && !isCopyFromLiveIn(SU))
        // CopyFromReg should be close to its def because it restricts
        // allocation choices. But if it is a livein then perhaps we want it
        // closer to its uses so it can be coalesced.
        return 0xffff;
      else if (Opc == ISD::TokenFactor || Opc == ISD::CopyToReg)
        // CopyToReg should be close to its uses to facilitate coalescing and
        // avoid spilling.
        return 0;
      else if (Opc == TargetInstrInfo::EXTRACT_SUBREG ||
               Opc == TargetInstrInfo::INSERT_SUBREG)
        // EXTRACT_SUBREG / INSERT_SUBREG should be close to its use to
        // facilitate coalescing.
        return 0;
      else if (SU->NumSuccs == 0)
        // If SU does not have a use, i.e. it doesn't produce a value that would
        // be consumed (e.g. store), then it terminates a chain of computation.
        // Give it a large SethiUllman number so it will be scheduled right
        // before its predecessors that it doesn't lengthen their live ranges.
        return 0xffff;
      else if (SU->NumPreds == 0)
        // If SU does not have a def, schedule it close to its uses because it
        // does not lengthen any live ranges.
        return 0;
      else
        return SethiUllmanNumbers[SU->NodeNum];
    }

    void setScheduleDAG(ScheduleDAGRRList *scheduleDag) { 
      scheduleDAG = scheduleDag; 
    }

  private:
    bool canClobber(const SUnit *SU, const SUnit *Op);
    void AddPseudoTwoAddrDeps();
    void CalculateSethiUllmanNumbers();
  };


  class VISIBILITY_HIDDEN BURegReductionFastPriorityQueue
   : public RegReductionPriorityQueue<bu_ls_rr_fast_sort> {
    // SUnits - The SUnits for the current graph.
    const std::vector<SUnit> *SUnits;
    
    // SethiUllmanNumbers - The SethiUllman number for each node.
    std::vector<unsigned> SethiUllmanNumbers;
  public:
    explicit BURegReductionFastPriorityQueue() {}

    void initNodes(std::vector<SUnit> &sunits) {
      SUnits = &sunits;
      // Calculate node priorities.
      CalculateSethiUllmanNumbers();
    }

    void addNode(const SUnit *SU) {
      unsigned SUSize = SethiUllmanNumbers.size();
      if (SUnits->size() > SUSize)
        SethiUllmanNumbers.resize(SUSize*2, 0);
      CalcNodeBUSethiUllmanNumber(SU, SethiUllmanNumbers);
    }

    void updateNode(const SUnit *SU) {
      SethiUllmanNumbers[SU->NodeNum] = 0;
      CalcNodeBUSethiUllmanNumber(SU, SethiUllmanNumbers);
    }

    void releaseState() {
      SUnits = 0;
      SethiUllmanNumbers.clear();
    }

    unsigned getNodePriority(const SUnit *SU) const {
      return SethiUllmanNumbers[SU->NodeNum];
    }

  private:
    void CalculateSethiUllmanNumbers();
  };


  class VISIBILITY_HIDDEN TDRegReductionPriorityQueue
   : public RegReductionPriorityQueue<td_ls_rr_sort> {
    // SUnits - The SUnits for the current graph.
    const std::vector<SUnit> *SUnits;
    
    // SethiUllmanNumbers - The SethiUllman number for each node.
    std::vector<unsigned> SethiUllmanNumbers;

  public:
    TDRegReductionPriorityQueue() {}

    void initNodes(std::vector<SUnit> &sunits) {
      SUnits = &sunits;
      // Calculate node priorities.
      CalculateSethiUllmanNumbers();
    }

    void addNode(const SUnit *SU) {
      unsigned SUSize = SethiUllmanNumbers.size();
      if (SUnits->size() > SUSize)
        SethiUllmanNumbers.resize(SUSize*2, 0);
      CalcNodeTDSethiUllmanNumber(SU, SethiUllmanNumbers);
    }

    void updateNode(const SUnit *SU) {
      SethiUllmanNumbers[SU->NodeNum] = 0;
      CalcNodeTDSethiUllmanNumber(SU, SethiUllmanNumbers);
    }

    void releaseState() {
      SUnits = 0;
      SethiUllmanNumbers.clear();
    }

    unsigned getNodePriority(const SUnit *SU) const {
      assert(SU->NodeNum < SethiUllmanNumbers.size());
      return SethiUllmanNumbers[SU->NodeNum];
    }

  private:
    void CalculateSethiUllmanNumbers();
  };
}

/// closestSucc - Returns the scheduled cycle of the successor which is
/// closet to the current cycle.
static unsigned closestSucc(const SUnit *SU) {
  unsigned MaxCycle = 0;
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    unsigned Cycle = I->Dep->Cycle;
    // If there are bunch of CopyToRegs stacked up, they should be considered
    // to be at the same position.
    if (I->Dep->Node && I->Dep->Node->getOpcode() == ISD::CopyToReg)
      Cycle = closestSucc(I->Dep)+1;
    if (Cycle > MaxCycle)
      MaxCycle = Cycle;
  }
  return MaxCycle;
}

/// calcMaxScratches - Returns an cost estimate of the worse case requirement
/// for scratch registers. Live-in operands and live-out results don't count
/// since they are "fixed".
static unsigned calcMaxScratches(const SUnit *SU) {
  unsigned Scratches = 0;
  for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl) continue;  // ignore chain preds
    if (!I->Dep->Node || I->Dep->Node->getOpcode() != ISD::CopyFromReg)
      Scratches++;
  }
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isCtrl) continue;  // ignore chain succs
    if (!I->Dep->Node || I->Dep->Node->getOpcode() != ISD::CopyToReg)
      Scratches += 10;
  }
  return Scratches;
}

// Bottom up
bool bu_ls_rr_sort::operator()(const SUnit *left, const SUnit *right) const {
  unsigned LPriority = SPQ->getNodePriority(left);
  unsigned RPriority = SPQ->getNodePriority(right);
  if (LPriority != RPriority)
    return LPriority > RPriority;

  // Try schedule def + use closer when Sethi-Ullman numbers are the same.
  // e.g.
  // t1 = op t2, c1
  // t3 = op t4, c2
  //
  // and the following instructions are both ready.
  // t2 = op c3
  // t4 = op c4
  //
  // Then schedule t2 = op first.
  // i.e.
  // t4 = op c4
  // t2 = op c3
  // t1 = op t2, c1
  // t3 = op t4, c2
  //
  // This creates more short live intervals.
  unsigned LDist = closestSucc(left);
  unsigned RDist = closestSucc(right);
  if (LDist != RDist)
    return LDist < RDist;

  // Intuitively, it's good to push down instructions whose results are
  // liveout so their long live ranges won't conflict with other values
  // which are needed inside the BB. Further prioritize liveout instructions
  // by the number of operands which are calculated within the BB.
  unsigned LScratch = calcMaxScratches(left);
  unsigned RScratch = calcMaxScratches(right);
  if (LScratch != RScratch)
    return LScratch > RScratch;

  if (left->Height != right->Height)
    return left->Height > right->Height;
  
  if (left->Depth != right->Depth)
    return left->Depth < right->Depth;

  if (left->CycleBound != right->CycleBound)
    return left->CycleBound > right->CycleBound;

  assert(left->NodeQueueId && right->NodeQueueId && 
         "NodeQueueId cannot be zero");
  return (left->NodeQueueId > right->NodeQueueId);
}

bool
bu_ls_rr_fast_sort::operator()(const SUnit *left, const SUnit *right) const {
  unsigned LPriority = SPQ->getNodePriority(left);
  unsigned RPriority = SPQ->getNodePriority(right);
  if (LPriority != RPriority)
    return LPriority > RPriority;
  assert(left->NodeQueueId && right->NodeQueueId && 
         "NodeQueueId cannot be zero");
  return (left->NodeQueueId > right->NodeQueueId);
}

bool
BURegReductionPriorityQueue::canClobber(const SUnit *SU, const SUnit *Op) {
  if (SU->isTwoAddress) {
    unsigned Opc = SU->Node->getMachineOpcode();
    const TargetInstrDesc &TID = TII->get(Opc);
    unsigned NumRes = TID.getNumDefs();
    unsigned NumOps = TID.getNumOperands() - NumRes;
    for (unsigned i = 0; i != NumOps; ++i) {
      if (TID.getOperandConstraint(i+NumRes, TOI::TIED_TO) != -1) {
        SDNode *DU = SU->Node->getOperand(i).getNode();
        if (DU->getNodeId() != -1 &&
            Op->OrigNode == &(*SUnits)[DU->getNodeId()])
          return true;
      }
    }
  }
  return false;
}


/// hasCopyToRegUse - Return true if SU has a value successor that is a
/// CopyToReg node.
static bool hasCopyToRegUse(const SUnit *SU) {
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isCtrl) continue;
    const SUnit *SuccSU = I->Dep;
    if (SuccSU->Node && SuccSU->Node->getOpcode() == ISD::CopyToReg)
      return true;
  }
  return false;
}

/// canClobberPhysRegDefs - True if SU would clobber one of SuccSU's
/// physical register defs.
static bool canClobberPhysRegDefs(const SUnit *SuccSU, const SUnit *SU,
                                  const TargetInstrInfo *TII,
                                  const TargetRegisterInfo *TRI) {
  SDNode *N = SuccSU->Node;
  unsigned NumDefs = TII->get(N->getMachineOpcode()).getNumDefs();
  const unsigned *ImpDefs = TII->get(N->getMachineOpcode()).getImplicitDefs();
  assert(ImpDefs && "Caller should check hasPhysRegDefs");
  const unsigned *SUImpDefs =
    TII->get(SU->Node->getMachineOpcode()).getImplicitDefs();
  if (!SUImpDefs)
    return false;
  for (unsigned i = NumDefs, e = N->getNumValues(); i != e; ++i) {
    MVT VT = N->getValueType(i);
    if (VT == MVT::Flag || VT == MVT::Other)
      continue;
    if (!N->hasAnyUseOfValue(i))
      continue;
    unsigned Reg = ImpDefs[i - NumDefs];
    for (;*SUImpDefs; ++SUImpDefs) {
      unsigned SUReg = *SUImpDefs;
      if (TRI->regsOverlap(Reg, SUReg))
        return true;
    }
  }
  return false;
}

/// AddPseudoTwoAddrDeps - If two nodes share an operand and one of them uses
/// it as a def&use operand. Add a pseudo control edge from it to the other
/// node (if it won't create a cycle) so the two-address one will be scheduled
/// first (lower in the schedule). If both nodes are two-address, favor the
/// one that has a CopyToReg use (more likely to be a loop induction update).
/// If both are two-address, but one is commutable while the other is not
/// commutable, favor the one that's not commutable.
void BURegReductionPriorityQueue::AddPseudoTwoAddrDeps() {
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i) {
    SUnit *SU = &(*SUnits)[i];
    if (!SU->isTwoAddress)
      continue;

    SDNode *Node = SU->Node;
    if (!Node || !Node->isMachineOpcode() || SU->FlaggedNodes.size() > 0)
      continue;

    unsigned Opc = Node->getMachineOpcode();
    const TargetInstrDesc &TID = TII->get(Opc);
    unsigned NumRes = TID.getNumDefs();
    unsigned NumOps = TID.getNumOperands() - NumRes;
    for (unsigned j = 0; j != NumOps; ++j) {
      if (TID.getOperandConstraint(j+NumRes, TOI::TIED_TO) != -1) {
        SDNode *DU = SU->Node->getOperand(j).getNode();
        if (DU->getNodeId() == -1)
          continue;
        const SUnit *DUSU = &(*SUnits)[DU->getNodeId()];
        if (!DUSU) continue;
        for (SUnit::const_succ_iterator I = DUSU->Succs.begin(),
             E = DUSU->Succs.end(); I != E; ++I) {
          if (I->isCtrl) continue;
          SUnit *SuccSU = I->Dep;
          if (SuccSU == SU)
            continue;
          // Be conservative. Ignore if nodes aren't at roughly the same
          // depth and height.
          if (SuccSU->Height < SU->Height && (SU->Height - SuccSU->Height) > 1)
            continue;
          if (!SuccSU->Node || !SuccSU->Node->isMachineOpcode())
            continue;
          // Don't constrain nodes with physical register defs if the
          // predecessor can clobber them.
          if (SuccSU->hasPhysRegDefs) {
            if (canClobberPhysRegDefs(SuccSU, SU, TII, TRI))
              continue;
          }
          // Don't constraint extract_subreg / insert_subreg these may be
          // coalesced away. We don't them close to their uses.
          unsigned SuccOpc = SuccSU->Node->getMachineOpcode();
          if (SuccOpc == TargetInstrInfo::EXTRACT_SUBREG ||
              SuccOpc == TargetInstrInfo::INSERT_SUBREG)
            continue;
          if ((!canClobber(SuccSU, DUSU) ||
               (hasCopyToRegUse(SU) && !hasCopyToRegUse(SuccSU)) ||
               (!SU->isCommutable && SuccSU->isCommutable)) &&
              !scheduleDAG->IsReachable(SuccSU, SU)) {
            DOUT << "Adding an edge from SU # " << SU->NodeNum
                 << " to SU #" << SuccSU->NodeNum << "\n";
            scheduleDAG->AddPred(SU, SuccSU, true, true);
          }
        }
      }
    }
  }
}

/// CalculateSethiUllmanNumbers - Calculate Sethi-Ullman numbers of all
/// scheduling units.
void BURegReductionPriorityQueue::CalculateSethiUllmanNumbers() {
  SethiUllmanNumbers.assign(SUnits->size(), 0);
  
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i)
    CalcNodeBUSethiUllmanNumber(&(*SUnits)[i], SethiUllmanNumbers);
}
void BURegReductionFastPriorityQueue::CalculateSethiUllmanNumbers() {
  SethiUllmanNumbers.assign(SUnits->size(), 0);
  
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i)
    CalcNodeBUSethiUllmanNumber(&(*SUnits)[i], SethiUllmanNumbers);
}

/// LimitedSumOfUnscheduledPredsOfSuccs - Compute the sum of the unscheduled
/// predecessors of the successors of the SUnit SU. Stop when the provided
/// limit is exceeded.
static unsigned LimitedSumOfUnscheduledPredsOfSuccs(const SUnit *SU, 
                                                    unsigned Limit) {
  unsigned Sum = 0;
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    const SUnit *SuccSU = I->Dep;
    for (SUnit::const_pred_iterator II = SuccSU->Preds.begin(),
         EE = SuccSU->Preds.end(); II != EE; ++II) {
      SUnit *PredSU = II->Dep;
      if (!PredSU->isScheduled)
        if (++Sum > Limit)
          return Sum;
    }
  }
  return Sum;
}


// Top down
bool td_ls_rr_sort::operator()(const SUnit *left, const SUnit *right) const {
  unsigned LPriority = SPQ->getNodePriority(left);
  unsigned RPriority = SPQ->getNodePriority(right);
  bool LIsTarget = left->Node && left->Node->isMachineOpcode();
  bool RIsTarget = right->Node && right->Node->isMachineOpcode();
  bool LIsFloater = LIsTarget && left->NumPreds == 0;
  bool RIsFloater = RIsTarget && right->NumPreds == 0;
  unsigned LBonus = (LimitedSumOfUnscheduledPredsOfSuccs(left,1) == 1) ? 2 : 0;
  unsigned RBonus = (LimitedSumOfUnscheduledPredsOfSuccs(right,1) == 1) ? 2 : 0;

  if (left->NumSuccs == 0 && right->NumSuccs != 0)
    return false;
  else if (left->NumSuccs != 0 && right->NumSuccs == 0)
    return true;

  if (LIsFloater)
    LBonus -= 2;
  if (RIsFloater)
    RBonus -= 2;
  if (left->NumSuccs == 1)
    LBonus += 2;
  if (right->NumSuccs == 1)
    RBonus += 2;

  if (LPriority+LBonus != RPriority+RBonus)
    return LPriority+LBonus < RPriority+RBonus;

  if (left->Depth != right->Depth)
    return left->Depth < right->Depth;

  if (left->NumSuccsLeft != right->NumSuccsLeft)
    return left->NumSuccsLeft > right->NumSuccsLeft;

  if (left->CycleBound != right->CycleBound)
    return left->CycleBound > right->CycleBound;

  assert(left->NodeQueueId && right->NodeQueueId && 
         "NodeQueueId cannot be zero");
  return (left->NodeQueueId > right->NodeQueueId);
}

/// CalculateSethiUllmanNumbers - Calculate Sethi-Ullman numbers of all
/// scheduling units.
void TDRegReductionPriorityQueue::CalculateSethiUllmanNumbers() {
  SethiUllmanNumbers.assign(SUnits->size(), 0);
  
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i)
    CalcNodeTDSethiUllmanNumber(&(*SUnits)[i], SethiUllmanNumbers);
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

llvm::ScheduleDAG* llvm::createBURRListDAGScheduler(SelectionDAGISel *IS,
                                                    SelectionDAG *DAG,
                                                    const TargetMachine *TM,
                                                    MachineBasicBlock *BB,
                                                    bool Fast) {
  if (Fast)
    return new ScheduleDAGRRList(DAG, BB, *TM, true, true,
                                 new BURegReductionFastPriorityQueue());

  const TargetInstrInfo *TII = TM->getInstrInfo();
  const TargetRegisterInfo *TRI = TM->getRegisterInfo();
  
  BURegReductionPriorityQueue *PQ = new BURegReductionPriorityQueue(TII, TRI);

  ScheduleDAGRRList *SD =
    new ScheduleDAGRRList(DAG, BB, *TM, true, false, PQ);
  PQ->setScheduleDAG(SD);
  return SD;  
}

llvm::ScheduleDAG* llvm::createTDRRListDAGScheduler(SelectionDAGISel *IS,
                                                    SelectionDAG *DAG,
                                                    const TargetMachine *TM,
                                                    MachineBasicBlock *BB,
                                                    bool Fast) {
  return new ScheduleDAGRRList(DAG, BB, *TM, false, Fast,
                               new TDRegReductionPriorityQueue());
}
