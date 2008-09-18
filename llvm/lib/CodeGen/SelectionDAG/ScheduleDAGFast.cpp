//===----- ScheduleDAGFast.cpp - Fast poor list scheduler -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a fast scheduler.
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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

STATISTIC(NumUnfolds,    "Number of nodes unfolded");
STATISTIC(NumDups,       "Number of duplicated nodes");
STATISTIC(NumCCCopies,   "Number of cross class copies");

static RegisterScheduler
  fastDAGScheduler("fast", "  Fast suboptimal list scheduling",
                   createFastDAGScheduler);

namespace {
  /// FastPriorityQueue - A degenerate priority queue that considers
  /// all nodes to have the same priority.
  ///
  struct VISIBILITY_HIDDEN FastPriorityQueue {
    std::vector<SUnit *> Queue;

    bool empty() const { return Queue.empty(); }
    
    void push(SUnit *U) {
      Queue.push_back(U);
    }

    SUnit *pop() {
      if (empty()) return NULL;
      SUnit *V = Queue.back();
      Queue.pop_back();
      return V;
    }
  };

//===----------------------------------------------------------------------===//
/// ScheduleDAGFast - The actual "fast" list scheduler implementation.
///
class VISIBILITY_HIDDEN ScheduleDAGFast : public ScheduleDAG {
private:
  /// AvailableQueue - The priority queue to use for the available SUnits.
  FastPriorityQueue AvailableQueue;

  /// LiveRegs / LiveRegDefs - A set of physical registers and their definition
  /// that are "live". These nodes must be scheduled before any other nodes that
  /// modifies the registers can be scheduled.
  SmallSet<unsigned, 4> LiveRegs;
  std::vector<SUnit*> LiveRegDefs;
  std::vector<unsigned> LiveRegCycles;

public:
  ScheduleDAGFast(SelectionDAG &dag, MachineBasicBlock *bb,
                  const TargetMachine &tm)
    : ScheduleDAG(dag, bb, tm) {}

  void Schedule();

  /// AddPred - This adds the specified node X as a predecessor of 
  /// the current node Y if not already.
  /// This returns true if this is a new predecessor.
  bool AddPred(SUnit *Y, SUnit *X, bool isCtrl, bool isSpecial,
               unsigned PhyReg = 0, int Cost = 1);

  /// RemovePred - This removes the specified node N from the predecessors of 
  /// the current node M.
  bool RemovePred(SUnit *M, SUnit *N, bool isCtrl, bool isSpecial);

private:
  void ReleasePred(SUnit*, bool, unsigned);
  void ScheduleNodeBottomUp(SUnit*, unsigned);
  SUnit *CopyAndMoveSuccessors(SUnit*);
  void InsertCCCopiesAndMoveSuccs(SUnit*, unsigned,
                                  const TargetRegisterClass*,
                                  const TargetRegisterClass*,
                                  SmallVector<SUnit*, 2>&);
  bool DelayForLiveRegsBottomUp(SUnit*, SmallVector<unsigned, 4>&);
  void ListScheduleBottomUp();

  /// CreateNewSUnit - Creates a new SUnit and returns a pointer to it.
  SUnit *CreateNewSUnit(SDNode *N) {
    SUnit *NewNode = NewSUnit(N);
    return NewNode;
  }

  /// CreateClone - Creates a new SUnit from an existing one.
  SUnit *CreateClone(SUnit *N) {
    SUnit *NewNode = Clone(N);
    return NewNode;
  }
};
}  // end anonymous namespace


/// Schedule - Schedule the DAG using list scheduling.
void ScheduleDAGFast::Schedule() {
  DOUT << "********** List Scheduling **********\n";

  LiveRegDefs.resize(TRI->getNumRegs(), NULL);  
  LiveRegCycles.resize(TRI->getNumRegs(), 0);

  // Build scheduling units.
  BuildSchedUnits();

  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(&DAG));

  // Execute the actual scheduling loop.
  ListScheduleBottomUp();
}

//===----------------------------------------------------------------------===//
//  Bottom-Up Scheduling
//===----------------------------------------------------------------------===//

/// ReleasePred - Decrement the NumSuccsLeft count of a predecessor. Add it to
/// the AvailableQueue if the count reaches zero. Also update its cycle bound.
void ScheduleDAGFast::ReleasePred(SUnit *PredSU, bool isChain, 
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
    PredSU->dump(&DAG);
    cerr << " has been released too many times!\n";
    assert(0);
  }
#endif
  
  if (PredSU->NumSuccsLeft == 0) {
    PredSU->isAvailable = true;
    AvailableQueue.push(PredSU);
  }
}

/// ScheduleNodeBottomUp - Add the node to the schedule. Decrement the pending
/// count of its predecessors. If a predecessor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGFast::ScheduleNodeBottomUp(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(&DAG));
  SU->Cycle = CurCycle;

  // Bottom up: release predecessors
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    ReleasePred(I->Dep, I->isCtrl, CurCycle);
    if (I->Cost < 0)  {
      // This is a physical register dependency and it's impossible or
      // expensive to copy the register. Make sure nothing that can 
      // clobber the register is scheduled between the predecessor and
      // this node.
      if (LiveRegs.insert(I->Reg)) {
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
        LiveRegs.erase(I->Reg);
        assert(LiveRegDefs[I->Reg] == SU &&
               "Physical register dependency violated?");
        LiveRegDefs[I->Reg] = NULL;
        LiveRegCycles[I->Reg] = 0;
      }
    }
  }

  SU->isScheduled = true;
}

/// AddPred - adds an edge from SUnit X to SUnit Y.
bool ScheduleDAGFast::AddPred(SUnit *Y, SUnit *X, bool isCtrl, bool isSpecial,
                              unsigned PhyReg, int Cost) {
  return Y->addPred(X, isCtrl, isSpecial, PhyReg, Cost);
}

/// RemovePred - This removes the specified node N from the predecessors of 
/// the current node M.
bool ScheduleDAGFast::RemovePred(SUnit *M, SUnit *N, 
                                 bool isCtrl, bool isSpecial) {
  return M->removePred(N, isCtrl, isSpecial);
}

/// CopyAndMoveSuccessors - Clone the specified node and move its scheduled
/// successors to the newly created node.
SUnit *ScheduleDAGFast::CopyAndMoveSuccessors(SUnit *SU) {
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
    if (!TII->unfoldMemoryOperand(DAG, N, NewNodes))
      return NULL;

    DOUT << "Unfolding SU # " << SU->NodeNum << "\n";
    assert(NewNodes.size() == 2 && "Expected a load folding node!");

    N = NewNodes[1];
    SDNode *LoadNode = NewNodes[0];
    unsigned NumVals = N->getNumValues();
    unsigned OldNumVals = SU->Node->getNumValues();
    for (unsigned i = 0; i != NumVals; ++i)
      DAG.ReplaceAllUsesOfValueWith(SDValue(SU->Node, i), SDValue(N, i));
    DAG.ReplaceAllUsesOfValueWith(SDValue(SU->Node, OldNumVals-1),
                                  SDValue(LoadNode, 1));

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

  ++NumDups;
  return NewSU;
}

/// InsertCCCopiesAndMoveSuccs - Insert expensive cross register class copies
/// and move all scheduled successors of the given SUnit to the last copy.
void ScheduleDAGFast::InsertCCCopiesAndMoveSuccs(SUnit *SU, unsigned Reg,
                                              const TargetRegisterClass *DestRC,
                                              const TargetRegisterClass *SrcRC,
                                               SmallVector<SUnit*, 2> &Copies) {
  SUnit *CopyFromSU = CreateNewSUnit(NULL);
  CopyFromSU->CopySrcRC = SrcRC;
  CopyFromSU->CopyDstRC = DestRC;

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
bool ScheduleDAGFast::DelayForLiveRegsBottomUp(SUnit *SU,
                                               SmallVector<unsigned, 4> &LRegs){
  if (LiveRegs.empty())
    return false;

  SmallSet<unsigned, 4> RegAdded;
  // If this node would clobber any "live" register, then it's not ready.
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->Cost < 0)  {
      unsigned Reg = I->Reg;
      if (LiveRegs.count(Reg) && LiveRegDefs[Reg] != I->Dep) {
        if (RegAdded.insert(Reg))
          LRegs.push_back(Reg);
      }
      for (const unsigned *Alias = TRI->getAliasSet(Reg);
           *Alias; ++Alias)
        if (LiveRegs.count(*Alias) && LiveRegDefs[*Alias] != I->Dep) {
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
      if (LiveRegs.count(*Reg) && LiveRegDefs[*Reg] != SU) {
        if (RegAdded.insert(*Reg))
          LRegs.push_back(*Reg);
      }
      for (const unsigned *Alias = TRI->getAliasSet(*Reg);
           *Alias; ++Alias)
        if (LiveRegs.count(*Alias) && LiveRegDefs[*Alias] != SU) {
          if (RegAdded.insert(*Alias))
            LRegs.push_back(*Alias);
        }
    }
  }
  return !LRegs.empty();
}


/// ListScheduleBottomUp - The main loop of list scheduling for bottom-up
/// schedulers.
void ScheduleDAGFast::ListScheduleBottomUp() {
  unsigned CurCycle = 0;
  // Add root to Available queue.
  if (!SUnits.empty()) {
    SUnit *RootSU = &SUnits[DAG.getRoot().getNode()->getNodeId()];
    assert(RootSU->Succs.empty() && "Graph root shouldn't have successors!");
    RootSU->isAvailable = true;
    AvailableQueue.push(RootSU);
  }

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  SmallVector<SUnit*, 4> NotReady;
  DenseMap<SUnit*, SmallVector<unsigned, 4> > LRegsMap;
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue.empty()) {
    bool Delayed = false;
    LRegsMap.clear();
    SUnit *CurSU = AvailableQueue.pop();
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
      CurSU = AvailableQueue.pop();
    }

    // All candidates are delayed due to live physical reg dependencies.
    // Try code duplication or inserting cross class copies
    // to resolve it.
    if (Delayed && !CurSU) {
      if (!CurSU) {
        // Try duplicating the nodes that produces these
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
        AvailableQueue.push(NotReady[i]);
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
      SUnits[i].dump(&DAG);
      cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
    if (SUnits[i].NumSuccsLeft != 0) {
      if (!AnyNotSched)
        cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(&DAG);
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
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

llvm::ScheduleDAG* llvm::createFastDAGScheduler(SelectionDAGISel *IS,
                                                SelectionDAG *DAG,
                                                MachineBasicBlock *BB, bool) {
  return new ScheduleDAGFast(*DAG, BB, DAG->getTarget());
}
