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
#include "ScheduleDAGSDNodes.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <climits>
using namespace llvm;

STATISTIC(NumBacktracks, "Number of times scheduler backtracked");
STATISTIC(NumUnfolds,    "Number of nodes unfolded");
STATISTIC(NumDups,       "Number of duplicated nodes");
STATISTIC(NumPRCopies,   "Number of physical register copies");

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
class VISIBILITY_HIDDEN ScheduleDAGRRList : public ScheduleDAGSDNodes {
private:
  /// isBottomUp - This is true if the scheduling problem is bottom-up, false if
  /// it is top-down.
  bool isBottomUp;

  /// AvailableQueue - The priority queue to use for the available SUnits.
  SchedulingPriorityQueue *AvailableQueue;

  /// LiveRegDefs - A set of physical registers and their definition
  /// that are "live". These nodes must be scheduled before any other nodes that
  /// modifies the registers can be scheduled.
  unsigned NumLiveRegs;
  std::vector<SUnit*> LiveRegDefs;
  std::vector<unsigned> LiveRegCycles;

  /// Topo - A topological ordering for SUnits which permits fast IsReachable
  /// and similar queries.
  ScheduleDAGTopologicalSort Topo;

public:
  ScheduleDAGRRList(MachineFunction &mf,
                    bool isbottomup,
                    SchedulingPriorityQueue *availqueue)
    : ScheduleDAGSDNodes(mf), isBottomUp(isbottomup),
      AvailableQueue(availqueue), Topo(SUnits) {
    }

  ~ScheduleDAGRRList() {
    delete AvailableQueue;
  }

  void Schedule();

  /// IsReachable - Checks if SU is reachable from TargetSU.
  bool IsReachable(const SUnit *SU, const SUnit *TargetSU) {
    return Topo.IsReachable(SU, TargetSU);
  }

  /// WillCreateCycle - Returns true if adding an edge from SU to TargetSU will
  /// create a cycle.
  bool WillCreateCycle(SUnit *SU, SUnit *TargetSU) {
    return Topo.WillCreateCycle(SU, TargetSU);
  }

  /// AddPred - adds a predecessor edge to SUnit SU.
  /// This returns true if this is a new predecessor.
  /// Updates the topological ordering if required.
  void AddPred(SUnit *SU, const SDep &D) {
    Topo.AddPred(SU, D.getSUnit());
    SU->addPred(D);
  }

  /// RemovePred - removes a predecessor edge from SUnit SU.
  /// This returns true if an edge was removed.
  /// Updates the topological ordering if required.
  void RemovePred(SUnit *SU, const SDep &D) {
    Topo.RemovePred(SU, D.getSUnit());
    SU->removePred(D);
  }

private:
  void ReleasePred(SUnit *SU, const SDep *PredEdge);
  void ReleasePredecessors(SUnit *SU, unsigned CurCycle);
  void ReleaseSucc(SUnit *SU, const SDep *SuccEdge);
  void ReleaseSuccessors(SUnit *SU);
  void CapturePred(SDep *PredEdge);
  void ScheduleNodeBottomUp(SUnit*, unsigned);
  void ScheduleNodeTopDown(SUnit*, unsigned);
  void UnscheduleNodeBottomUp(SUnit*);
  void BacktrackBottomUp(SUnit*, unsigned, unsigned&);
  SUnit *CopyAndMoveSuccessors(SUnit*);
  void InsertCopiesAndMoveSuccs(SUnit*, unsigned,
                                const TargetRegisterClass*,
                                const TargetRegisterClass*,
                                SmallVector<SUnit*, 2>&);
  bool DelayForLiveRegsBottomUp(SUnit*, SmallVector<unsigned, 4>&);
  void ListScheduleTopDown();
  void ListScheduleBottomUp();


  /// CreateNewSUnit - Creates a new SUnit and returns a pointer to it.
  /// Updates the topological ordering if required.
  SUnit *CreateNewSUnit(SDNode *N) {
    unsigned NumSUnits = SUnits.size();
    SUnit *NewNode = NewSUnit(N);
    // Update the topological ordering.
    if (NewNode->NodeNum >= NumSUnits)
      Topo.InitDAGTopologicalSorting();
    return NewNode;
  }

  /// CreateClone - Creates a new SUnit from an existing one.
  /// Updates the topological ordering if required.
  SUnit *CreateClone(SUnit *N) {
    unsigned NumSUnits = SUnits.size();
    SUnit *NewNode = Clone(N);
    // Update the topological ordering.
    if (NewNode->NodeNum >= NumSUnits)
      Topo.InitDAGTopologicalSorting();
    return NewNode;
  }

  /// ForceUnitLatencies - Return true, since register-pressure-reducing
  /// scheduling doesn't need actual latency information.
  bool ForceUnitLatencies() const { return true; }
};
}  // end anonymous namespace


/// Schedule - Schedule the DAG using list scheduling.
void ScheduleDAGRRList::Schedule() {
  DOUT << "********** List Scheduling **********\n";

  NumLiveRegs = 0;
  LiveRegDefs.resize(TRI->getNumRegs(), NULL);  
  LiveRegCycles.resize(TRI->getNumRegs(), 0);

  // Build the scheduling graph.
  BuildSchedGraph();

  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(this));
  Topo.InitDAGTopologicalSorting();

  AvailableQueue->initNodes(SUnits);
  
  // Execute the actual scheduling loop Top-Down or Bottom-Up as appropriate.
  if (isBottomUp)
    ListScheduleBottomUp();
  else
    ListScheduleTopDown();
  
  AvailableQueue->releaseState();
}

//===----------------------------------------------------------------------===//
//  Bottom-Up Scheduling
//===----------------------------------------------------------------------===//

/// ReleasePred - Decrement the NumSuccsLeft count of a predecessor. Add it to
/// the AvailableQueue if the count reaches zero. Also update its cycle bound.
void ScheduleDAGRRList::ReleasePred(SUnit *SU, const SDep *PredEdge) {
  SUnit *PredSU = PredEdge->getSUnit();
  --PredSU->NumSuccsLeft;
  
#ifndef NDEBUG
  if (PredSU->NumSuccsLeft < 0) {
    cerr << "*** Scheduling failed! ***\n";
    PredSU->dump(this);
    cerr << " has been released too many times!\n";
    llvm_unreachable(0);
  }
#endif
  
  // If all the node's successors are scheduled, this node is ready
  // to be scheduled. Ignore the special EntrySU node.
  if (PredSU->NumSuccsLeft == 0 && PredSU != &EntrySU) {
    PredSU->isAvailable = true;
    AvailableQueue->push(PredSU);
  }
}

void ScheduleDAGRRList::ReleasePredecessors(SUnit *SU, unsigned CurCycle) {
  // Bottom up: release predecessors
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    ReleasePred(SU, &*I);
    if (I->isAssignedRegDep()) {
      // This is a physical register dependency and it's impossible or
      // expensive to copy the register. Make sure nothing that can 
      // clobber the register is scheduled between the predecessor and
      // this node.
      if (!LiveRegDefs[I->getReg()]) {
        ++NumLiveRegs;
        LiveRegDefs[I->getReg()] = I->getSUnit();
        LiveRegCycles[I->getReg()] = CurCycle;
      }
    }
  }
}

/// ScheduleNodeBottomUp - Add the node to the schedule. Decrement the pending
/// count of its predecessors. If a predecessor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGRRList::ScheduleNodeBottomUp(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(this));

  assert(CurCycle >= SU->getHeight() && "Node scheduled below its height!");
  SU->setHeightToAtLeast(CurCycle);
  Sequence.push_back(SU);

  ReleasePredecessors(SU, CurCycle);

  // Release all the implicit physical register defs that are live.
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isAssignedRegDep()) {
      if (LiveRegCycles[I->getReg()] == I->getSUnit()->getHeight()) {
        assert(NumLiveRegs > 0 && "NumLiveRegs is already zero!");
        assert(LiveRegDefs[I->getReg()] == SU &&
               "Physical register dependency violated?");
        --NumLiveRegs;
        LiveRegDefs[I->getReg()] = NULL;
        LiveRegCycles[I->getReg()] = 0;
      }
    }
  }

  SU->isScheduled = true;
  AvailableQueue->ScheduledNode(SU);
}

/// CapturePred - This does the opposite of ReleasePred. Since SU is being
/// unscheduled, incrcease the succ left count of its predecessors. Remove
/// them from AvailableQueue if necessary.
void ScheduleDAGRRList::CapturePred(SDep *PredEdge) {  
  SUnit *PredSU = PredEdge->getSUnit();
  if (PredSU->isAvailable) {
    PredSU->isAvailable = false;
    if (!PredSU->isPending)
      AvailableQueue->remove(PredSU);
  }

  ++PredSU->NumSuccsLeft;
}

/// UnscheduleNodeBottomUp - Remove the node from the schedule, update its and
/// its predecessor states to reflect the change.
void ScheduleDAGRRList::UnscheduleNodeBottomUp(SUnit *SU) {
  DOUT << "*** Unscheduling [" << SU->getHeight() << "]: ";
  DEBUG(SU->dump(this));

  AvailableQueue->UnscheduledNode(SU);

  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    CapturePred(&*I);
    if (I->isAssignedRegDep() && SU->getHeight() == LiveRegCycles[I->getReg()]) {
      assert(NumLiveRegs > 0 && "NumLiveRegs is already zero!");
      assert(LiveRegDefs[I->getReg()] == I->getSUnit() &&
             "Physical register dependency violated?");
      --NumLiveRegs;
      LiveRegDefs[I->getReg()] = NULL;
      LiveRegCycles[I->getReg()] = 0;
    }
  }

  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isAssignedRegDep()) {
      if (!LiveRegDefs[I->getReg()]) {
        LiveRegDefs[I->getReg()] = SU;
        ++NumLiveRegs;
      }
      if (I->getSUnit()->getHeight() < LiveRegCycles[I->getReg()])
        LiveRegCycles[I->getReg()] = I->getSUnit()->getHeight();
    }
  }

  SU->setHeightDirty();
  SU->isScheduled = false;
  SU->isAvailable = true;
  AvailableQueue->push(SU);
}

/// BacktrackBottomUp - Backtrack scheduling to a previous cycle specified in
/// BTCycle in order to schedule a specific node.
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

  assert(!SU->isSucc(OldSU) && "Something is wrong!");

  ++NumBacktracks;
}

/// CopyAndMoveSuccessors - Clone the specified node and move its scheduled
/// successors to the newly created node.
SUnit *ScheduleDAGRRList::CopyAndMoveSuccessors(SUnit *SU) {
  if (SU->getNode()->getFlaggedNode())
    return NULL;

  SDNode *N = SU->getNode();
  if (!N)
    return NULL;

  SUnit *NewSU;
  bool TryUnfold = false;
  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i) {
    EVT VT = N->getValueType(i);
    if (VT == EVT::Flag)
      return NULL;
    else if (VT == EVT::Other)
      TryUnfold = true;
  }
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    const SDValue &Op = N->getOperand(i);
    EVT VT = Op.getNode()->getValueType(Op.getResNo());
    if (VT == EVT::Flag)
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
    unsigned OldNumVals = SU->getNode()->getNumValues();
    for (unsigned i = 0; i != NumVals; ++i)
      DAG->ReplaceAllUsesOfValueWith(SDValue(SU->getNode(), i), SDValue(N, i));
    DAG->ReplaceAllUsesOfValueWith(SDValue(SU->getNode(), OldNumVals-1),
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
    ComputeLatency(NewSU);

    // Record all the edges to and from the old SU, by category.
    SmallVector<SDep, 4> ChainPreds;
    SmallVector<SDep, 4> ChainSuccs;
    SmallVector<SDep, 4> LoadPreds;
    SmallVector<SDep, 4> NodePreds;
    SmallVector<SDep, 4> NodeSuccs;
    for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      if (I->isCtrl())
        ChainPreds.push_back(*I);
      else if (I->getSUnit()->getNode() &&
               I->getSUnit()->getNode()->isOperandOf(LoadNode))
        LoadPreds.push_back(*I);
      else
        NodePreds.push_back(*I);
    }
    for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      if (I->isCtrl())
        ChainSuccs.push_back(*I);
      else
        NodeSuccs.push_back(*I);
    }

    // Now assign edges to the newly-created nodes.
    for (unsigned i = 0, e = ChainPreds.size(); i != e; ++i) {
      const SDep &Pred = ChainPreds[i];
      RemovePred(SU, Pred);
      if (isNewLoad)
        AddPred(LoadSU, Pred);
    }
    for (unsigned i = 0, e = LoadPreds.size(); i != e; ++i) {
      const SDep &Pred = LoadPreds[i];
      RemovePred(SU, Pred);
      if (isNewLoad)
        AddPred(LoadSU, Pred);
    }
    for (unsigned i = 0, e = NodePreds.size(); i != e; ++i) {
      const SDep &Pred = NodePreds[i];
      RemovePred(SU, Pred);
      AddPred(NewSU, Pred);
    }
    for (unsigned i = 0, e = NodeSuccs.size(); i != e; ++i) {
      SDep D = NodeSuccs[i];
      SUnit *SuccDep = D.getSUnit();
      D.setSUnit(SU);
      RemovePred(SuccDep, D);
      D.setSUnit(NewSU);
      AddPred(SuccDep, D);
    }
    for (unsigned i = 0, e = ChainSuccs.size(); i != e; ++i) {
      SDep D = ChainSuccs[i];
      SUnit *SuccDep = D.getSUnit();
      D.setSUnit(SU);
      RemovePred(SuccDep, D);
      if (isNewLoad) {
        D.setSUnit(LoadSU);
        AddPred(SuccDep, D);
      }
    } 

    // Add a data dependency to reflect that NewSU reads the value defined
    // by LoadSU.
    AddPred(NewSU, SDep(LoadSU, SDep::Data, LoadSU->Latency));

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
    if (!I->isArtificial())
      AddPred(NewSU, *I);

  // Only copy scheduled successors. Cut them from old node's successor
  // list and move them over.
  SmallVector<std::pair<SUnit *, SDep>, 4> DelDeps;
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isArtificial())
      continue;
    SUnit *SuccSU = I->getSUnit();
    if (SuccSU->isScheduled) {
      SDep D = *I;
      D.setSUnit(NewSU);
      AddPred(SuccSU, D);
      D.setSUnit(SU);
      DelDeps.push_back(std::make_pair(SuccSU, D));
    }
  }
  for (unsigned i = 0, e = DelDeps.size(); i != e; ++i)
    RemovePred(DelDeps[i].first, DelDeps[i].second);

  AvailableQueue->updateNode(SU);
  AvailableQueue->addNode(NewSU);

  ++NumDups;
  return NewSU;
}

/// InsertCopiesAndMoveSuccs - Insert register copies and move all
/// scheduled successors of the given SUnit to the last copy.
void ScheduleDAGRRList::InsertCopiesAndMoveSuccs(SUnit *SU, unsigned Reg,
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
  SmallVector<std::pair<SUnit *, SDep>, 4> DelDeps;
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isArtificial())
      continue;
    SUnit *SuccSU = I->getSUnit();
    if (SuccSU->isScheduled) {
      SDep D = *I;
      D.setSUnit(CopyToSU);
      AddPred(SuccSU, D);
      DelDeps.push_back(std::make_pair(SuccSU, *I));
    }
  }
  for (unsigned i = 0, e = DelDeps.size(); i != e; ++i)
    RemovePred(DelDeps[i].first, DelDeps[i].second);

  AddPred(CopyFromSU, SDep(SU, SDep::Data, SU->Latency, Reg));
  AddPred(CopyToSU, SDep(CopyFromSU, SDep::Data, CopyFromSU->Latency, 0));

  AvailableQueue->updateNode(SU);
  AvailableQueue->addNode(CopyFromSU);
  AvailableQueue->addNode(CopyToSU);
  Copies.push_back(CopyFromSU);
  Copies.push_back(CopyToSU);

  ++NumPRCopies;
}

/// getPhysicalRegisterVT - Returns the ValueType of the physical register
/// definition of the specified node.
/// FIXME: Move to SelectionDAG?
static EVT getPhysicalRegisterVT(SDNode *N, unsigned Reg,
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

/// CheckForLiveRegDef - Return true and update live register vector if the
/// specified register def of the specified SUnit clobbers any "live" registers.
static bool CheckForLiveRegDef(SUnit *SU, unsigned Reg,
                               std::vector<SUnit*> &LiveRegDefs,
                               SmallSet<unsigned, 4> &RegAdded,
                               SmallVector<unsigned, 4> &LRegs,
                               const TargetRegisterInfo *TRI) {
  bool Added = false;
  if (LiveRegDefs[Reg] && LiveRegDefs[Reg] != SU) {
    if (RegAdded.insert(Reg)) {
      LRegs.push_back(Reg);
      Added = true;
    }
  }
  for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias)
    if (LiveRegDefs[*Alias] && LiveRegDefs[*Alias] != SU) {
      if (RegAdded.insert(*Alias)) {
        LRegs.push_back(*Alias);
        Added = true;
      }
    }
  return Added;
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
    if (I->isAssignedRegDep())
      CheckForLiveRegDef(I->getSUnit(), I->getReg(), LiveRegDefs,
                         RegAdded, LRegs, TRI);
  }

  for (SDNode *Node = SU->getNode(); Node; Node = Node->getFlaggedNode()) {
    if (Node->getOpcode() == ISD::INLINEASM) {
      // Inline asm can clobber physical defs.
      unsigned NumOps = Node->getNumOperands();
      if (Node->getOperand(NumOps-1).getValueType() == EVT::Flag)
        --NumOps;  // Ignore the flag operand.

      for (unsigned i = 2; i != NumOps;) {
        unsigned Flags =
          cast<ConstantSDNode>(Node->getOperand(i))->getZExtValue();
        unsigned NumVals = (Flags & 0xffff) >> 3;

        ++i; // Skip the ID value.
        if ((Flags & 7) == 2 || (Flags & 7) == 6) {
          // Check for def of register or earlyclobber register.
          for (; NumVals; --NumVals, ++i) {
            unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
            if (TargetRegisterInfo::isPhysicalRegister(Reg))
              CheckForLiveRegDef(SU, Reg, LiveRegDefs, RegAdded, LRegs, TRI);
          }
        } else
          i += NumVals;
      }
      continue;
    }

    if (!Node->isMachineOpcode())
      continue;
    const TargetInstrDesc &TID = TII->get(Node->getMachineOpcode());
    if (!TID.ImplicitDefs)
      continue;
    for (const unsigned *Reg = TID.ImplicitDefs; *Reg; ++Reg)
      CheckForLiveRegDef(SU, *Reg, LiveRegDefs, RegAdded, LRegs, TRI);
  }
  return !LRegs.empty();
}


/// ListScheduleBottomUp - The main loop of list scheduling for bottom-up
/// schedulers.
void ScheduleDAGRRList::ListScheduleBottomUp() {
  unsigned CurCycle = 0;

  // Release any predecessors of the special Exit node.
  ReleasePredecessors(&ExitSU, CurCycle);

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
      SmallVector<unsigned, 4> LRegs;
      if (!DelayForLiveRegsBottomUp(CurSU, LRegs))
        break;
      Delayed = true;
      LRegsMap.insert(std::make_pair(CurSU, LRegs));

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
          AddPred(TrySU, SDep(OldSU, SDep::Order, /*Latency=*/1,
                              /*Reg=*/0, /*isNormalMemory=*/false,
                              /*isMustAlias=*/false, /*isArtificial=*/true));
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
        // Can't backtrack. If it's too expensive to copy the value, then try
        // duplicate the nodes that produces these "too expensive to copy"
        // values to break the dependency. In case even that doesn't work,
        // insert cross class copies.
        // If it's not too expensive, i.e. cost != -1, issue copies.
        SUnit *TrySU = NotReady[0];
        SmallVector<unsigned, 4> &LRegs = LRegsMap[TrySU];
        assert(LRegs.size() == 1 && "Can't handle this yet!");
        unsigned Reg = LRegs[0];
        SUnit *LRDef = LiveRegDefs[Reg];
        EVT VT = getPhysicalRegisterVT(LRDef->getNode(), Reg, TII);
        const TargetRegisterClass *RC =
          TRI->getPhysicalRegisterRegClass(Reg, VT);
        const TargetRegisterClass *DestRC = TRI->getCrossCopyRegClass(RC);

        // If cross copy register class is null, then it must be possible copy
        // the value directly. Do not try duplicate the def.
        SUnit *NewDef = 0;
        if (DestRC)
          NewDef = CopyAndMoveSuccessors(LRDef);
        else
          DestRC = RC;
        if (!NewDef) {
          // Issue copies, these can be expensive cross register class copies.
          SmallVector<SUnit*, 2> Copies;
          InsertCopiesAndMoveSuccs(LRDef, Reg, DestRC, RC, Copies);
          DOUT << "Adding an edge from SU #" << TrySU->NodeNum
               << " to SU #" << Copies.front()->NodeNum << "\n";
          AddPred(TrySU, SDep(Copies.front(), SDep::Order, /*Latency=*/1,
                              /*Reg=*/0, /*isNormalMemory=*/false,
                              /*isMustAlias=*/false,
                              /*isArtificial=*/true));
          NewDef = Copies.back();
        }

        DOUT << "Adding an edge from SU #" << NewDef->NodeNum
             << " to SU #" << TrySU->NodeNum << "\n";
        LiveRegDefs[Reg] = NewDef;
        AddPred(NewDef, SDep(TrySU, SDep::Order, /*Latency=*/1,
                             /*Reg=*/0, /*isNormalMemory=*/false,
                             /*isMustAlias=*/false,
                             /*isArtificial=*/true));
        TrySU->isAvailable = false;
        CurSU = NewDef;
      }

      assert(CurSU && "Unable to resolve live physical register dependencies!");
    }

    // Add the nodes that aren't ready back onto the available list.
    for (unsigned i = 0, e = NotReady.size(); i != e; ++i) {
      NotReady[i]->isPending = false;
      // May no longer be available due to backtracking.
      if (NotReady[i]->isAvailable)
        AvailableQueue->push(NotReady[i]);
    }
    NotReady.clear();

    if (CurSU)
      ScheduleNodeBottomUp(CurSU, CurCycle);
    ++CurCycle;
  }

  // Reverse the order if it is bottom up.
  std::reverse(Sequence.begin(), Sequence.end());
  
#ifndef NDEBUG
  VerifySchedule(isBottomUp);
#endif
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the AvailableQueue if the count reaches zero. Also update its cycle bound.
void ScheduleDAGRRList::ReleaseSucc(SUnit *SU, const SDep *SuccEdge) {
  SUnit *SuccSU = SuccEdge->getSUnit();
  --SuccSU->NumPredsLeft;
  
#ifndef NDEBUG
  if (SuccSU->NumPredsLeft < 0) {
    cerr << "*** Scheduling failed! ***\n";
    SuccSU->dump(this);
    cerr << " has been released too many times!\n";
    llvm_unreachable(0);
  }
#endif
  
  // If all the node's predecessors are scheduled, this node is ready
  // to be scheduled. Ignore the special ExitSU node.
  if (SuccSU->NumPredsLeft == 0 && SuccSU != &ExitSU) {
    SuccSU->isAvailable = true;
    AvailableQueue->push(SuccSU);
  }
}

void ScheduleDAGRRList::ReleaseSuccessors(SUnit *SU) {
  // Top down: release successors
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    assert(!I->isAssignedRegDep() &&
           "The list-tdrr scheduler doesn't yet support physreg dependencies!");

    ReleaseSucc(SU, &*I);
  }
}

/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGRRList::ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(this));

  assert(CurCycle >= SU->getDepth() && "Node scheduled above its depth!");
  SU->setDepthToAtLeast(CurCycle);
  Sequence.push_back(SU);

  ReleaseSuccessors(SU);
  SU->isScheduled = true;
  AvailableQueue->ScheduledNode(SU);
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void ScheduleDAGRRList::ListScheduleTopDown() {
  unsigned CurCycle = 0;

  // Release any successors of the special Entry node.
  ReleaseSuccessors(&EntrySU);

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
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue->empty()) {
    SUnit *CurSU = AvailableQueue->pop();
    
    if (CurSU)
      ScheduleNodeTopDown(CurSU, CurCycle);
    ++CurCycle;
  }
  
#ifndef NDEBUG
  VerifySchedule(isBottomUp);
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

  struct td_ls_rr_sort : public std::binary_function<SUnit*, SUnit*, bool> {
    RegReductionPriorityQueue<td_ls_rr_sort> *SPQ;
    td_ls_rr_sort(RegReductionPriorityQueue<td_ls_rr_sort> *spq) : SPQ(spq) {}
    td_ls_rr_sort(const td_ls_rr_sort &RHS) : SPQ(RHS.SPQ) {}
    
    bool operator()(const SUnit* left, const SUnit* right) const;
  };
}  // end anonymous namespace

/// CalcNodeSethiUllmanNumber - Compute Sethi Ullman number.
/// Smaller number is the higher priority.
static unsigned
CalcNodeSethiUllmanNumber(const SUnit *SU, std::vector<unsigned> &SUNumbers) {
  unsigned &SethiUllmanNumber = SUNumbers[SU->NodeNum];
  if (SethiUllmanNumber != 0)
    return SethiUllmanNumber;

  unsigned Extra = 0;
  for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl()) continue;  // ignore chain preds
    SUnit *PredSU = I->getSUnit();
    unsigned PredSethiUllman = CalcNodeSethiUllmanNumber(PredSU, SUNumbers);
    if (PredSethiUllman > SethiUllmanNumber) {
      SethiUllmanNumber = PredSethiUllman;
      Extra = 0;
    } else if (PredSethiUllman == SethiUllmanNumber)
      ++Extra;
  }

  SethiUllmanNumber += Extra;

  if (SethiUllmanNumber == 0)
    SethiUllmanNumber = 1;
  
  return SethiUllmanNumber;
}

namespace {
  template<class SF>
  class VISIBILITY_HIDDEN RegReductionPriorityQueue
   : public SchedulingPriorityQueue {
    PriorityQueue<SUnit*, std::vector<SUnit*>, SF> Queue;
    unsigned currentQueueId;

  protected:
    // SUnits - The SUnits for the current graph.
    std::vector<SUnit> *SUnits;
    
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    ScheduleDAGRRList *scheduleDAG;

    // SethiUllmanNumbers - The SethiUllman number for each node.
    std::vector<unsigned> SethiUllmanNumbers;

  public:
    RegReductionPriorityQueue(const TargetInstrInfo *tii,
                              const TargetRegisterInfo *tri) :
    Queue(SF(this)), currentQueueId(0),
    TII(tii), TRI(tri), scheduleDAG(NULL) {}
    
    void initNodes(std::vector<SUnit> &sunits) {
      SUnits = &sunits;
      // Add pseudo dependency edges for two-address nodes.
      AddPseudoTwoAddrDeps();
      // Reroute edges to nodes with multiple uses.
      PrescheduleNodesWithMultipleUses();
      // Calculate node priorities.
      CalculateSethiUllmanNumbers();
    }

    void addNode(const SUnit *SU) {
      unsigned SUSize = SethiUllmanNumbers.size();
      if (SUnits->size() > SUSize)
        SethiUllmanNumbers.resize(SUSize*2, 0);
      CalcNodeSethiUllmanNumber(SU, SethiUllmanNumbers);
    }

    void updateNode(const SUnit *SU) {
      SethiUllmanNumbers[SU->NodeNum] = 0;
      CalcNodeSethiUllmanNumber(SU, SethiUllmanNumbers);
    }

    void releaseState() {
      SUnits = 0;
      SethiUllmanNumbers.clear();
    }

    unsigned getNodePriority(const SUnit *SU) const {
      assert(SU->NodeNum < SethiUllmanNumbers.size());
      unsigned Opc = SU->getNode() ? SU->getNode()->getOpcode() : 0;
      if (Opc == ISD::TokenFactor || Opc == ISD::CopyToReg)
        // CopyToReg should be close to its uses to facilitate coalescing and
        // avoid spilling.
        return 0;
      if (Opc == TargetInstrInfo::EXTRACT_SUBREG ||
          Opc == TargetInstrInfo::SUBREG_TO_REG ||
          Opc == TargetInstrInfo::INSERT_SUBREG)
        // EXTRACT_SUBREG, INSERT_SUBREG, and SUBREG_TO_REG nodes should be
        // close to their uses to facilitate coalescing.
        return 0;
      if (SU->NumSuccs == 0 && SU->NumPreds != 0)
        // If SU does not have a register use, i.e. it doesn't produce a value
        // that would be consumed (e.g. store), then it terminates a chain of
        // computation.  Give it a large SethiUllman number so it will be
        // scheduled right before its predecessors that it doesn't lengthen
        // their live ranges.
        return 0xffff;
      if (SU->NumPreds == 0 && SU->NumSuccs != 0)
        // If SU does not have a register def, schedule it close to its uses
        // because it does not lengthen any live ranges.
        return 0;
      return SethiUllmanNumbers[SU->NodeNum];
    }
    
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

    void setScheduleDAG(ScheduleDAGRRList *scheduleDag) { 
      scheduleDAG = scheduleDag; 
    }

  protected:
    bool canClobber(const SUnit *SU, const SUnit *Op);
    void AddPseudoTwoAddrDeps();
    void PrescheduleNodesWithMultipleUses();
    void CalculateSethiUllmanNumbers();
  };

  typedef RegReductionPriorityQueue<bu_ls_rr_sort>
    BURegReductionPriorityQueue;

  typedef RegReductionPriorityQueue<td_ls_rr_sort>
    TDRegReductionPriorityQueue;
}

/// closestSucc - Returns the scheduled cycle of the successor which is
/// closest to the current cycle.
static unsigned closestSucc(const SUnit *SU) {
  unsigned MaxHeight = 0;
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isCtrl()) continue;  // ignore chain succs
    unsigned Height = I->getSUnit()->getHeight();
    // If there are bunch of CopyToRegs stacked up, they should be considered
    // to be at the same position.
    if (I->getSUnit()->getNode() &&
        I->getSUnit()->getNode()->getOpcode() == ISD::CopyToReg)
      Height = closestSucc(I->getSUnit())+1;
    if (Height > MaxHeight)
      MaxHeight = Height;
  }
  return MaxHeight;
}

/// calcMaxScratches - Returns an cost estimate of the worse case requirement
/// for scratch registers, i.e. number of data dependencies.
static unsigned calcMaxScratches(const SUnit *SU) {
  unsigned Scratches = 0;
  for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl()) continue;  // ignore chain preds
    Scratches++;
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

  // How many registers becomes live when the node is scheduled.
  unsigned LScratch = calcMaxScratches(left);
  unsigned RScratch = calcMaxScratches(right);
  if (LScratch != RScratch)
    return LScratch > RScratch;

  if (left->getHeight() != right->getHeight())
    return left->getHeight() > right->getHeight();
  
  if (left->getDepth() != right->getDepth())
    return left->getDepth() < right->getDepth();

  assert(left->NodeQueueId && right->NodeQueueId && 
         "NodeQueueId cannot be zero");
  return (left->NodeQueueId > right->NodeQueueId);
}

template<class SF>
bool
RegReductionPriorityQueue<SF>::canClobber(const SUnit *SU, const SUnit *Op) {
  if (SU->isTwoAddress) {
    unsigned Opc = SU->getNode()->getMachineOpcode();
    const TargetInstrDesc &TID = TII->get(Opc);
    unsigned NumRes = TID.getNumDefs();
    unsigned NumOps = TID.getNumOperands() - NumRes;
    for (unsigned i = 0; i != NumOps; ++i) {
      if (TID.getOperandConstraint(i+NumRes, TOI::TIED_TO) != -1) {
        SDNode *DU = SU->getNode()->getOperand(i).getNode();
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
    if (I->isCtrl()) continue;
    const SUnit *SuccSU = I->getSUnit();
    if (SuccSU->getNode() && SuccSU->getNode()->getOpcode() == ISD::CopyToReg)
      return true;
  }
  return false;
}

/// canClobberPhysRegDefs - True if SU would clobber one of SuccSU's
/// physical register defs.
static bool canClobberPhysRegDefs(const SUnit *SuccSU, const SUnit *SU,
                                  const TargetInstrInfo *TII,
                                  const TargetRegisterInfo *TRI) {
  SDNode *N = SuccSU->getNode();
  unsigned NumDefs = TII->get(N->getMachineOpcode()).getNumDefs();
  const unsigned *ImpDefs = TII->get(N->getMachineOpcode()).getImplicitDefs();
  assert(ImpDefs && "Caller should check hasPhysRegDefs");
  for (const SDNode *SUNode = SU->getNode(); SUNode;
       SUNode = SUNode->getFlaggedNode()) {
    if (!SUNode->isMachineOpcode())
      continue;
    const unsigned *SUImpDefs =
      TII->get(SUNode->getMachineOpcode()).getImplicitDefs();
    if (!SUImpDefs)
      return false;
    for (unsigned i = NumDefs, e = N->getNumValues(); i != e; ++i) {
      EVT VT = N->getValueType(i);
      if (VT == EVT::Flag || VT == EVT::Other)
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
  }
  return false;
}

/// PrescheduleNodesWithMultipleUses - Nodes with multiple uses
/// are not handled well by the general register pressure reduction
/// heuristics. When presented with code like this:
///
///      N
///    / |
///   /  |
///  U  store
///  |
/// ...
///
/// the heuristics tend to push the store up, but since the
/// operand of the store has another use (U), this would increase
/// the length of that other use (the U->N edge).
///
/// This function transforms code like the above to route U's
/// dependence through the store when possible, like this:
///
///      N
///      ||
///      ||
///     store
///       |
///       U
///       |
///      ...
///
/// This results in the store being scheduled immediately
/// after N, which shortens the U->N live range, reducing
/// register pressure.
///
template<class SF>
void RegReductionPriorityQueue<SF>::PrescheduleNodesWithMultipleUses() {
  // Visit all the nodes in topological order, working top-down.
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i) {
    SUnit *SU = &(*SUnits)[i];
    // For now, only look at nodes with no data successors, such as stores.
    // These are especially important, due to the heuristics in
    // getNodePriority for nodes with no data successors.
    if (SU->NumSuccs != 0)
      continue;
    // For now, only look at nodes with exactly one data predecessor.
    if (SU->NumPreds != 1)
      continue;
    // Avoid prescheduling copies to virtual registers, which don't behave
    // like other nodes from the perspective of scheduling heuristics.
    if (SDNode *N = SU->getNode())
      if (N->getOpcode() == ISD::CopyToReg &&
          TargetRegisterInfo::isVirtualRegister
            (cast<RegisterSDNode>(N->getOperand(1))->getReg()))
        continue;

    // Locate the single data predecessor.
    SUnit *PredSU = 0;
    for (SUnit::const_pred_iterator II = SU->Preds.begin(),
         EE = SU->Preds.end(); II != EE; ++II)
      if (!II->isCtrl()) {
        PredSU = II->getSUnit();
        break;
      }
    assert(PredSU);

    // Don't rewrite edges that carry physregs, because that requires additional
    // support infrastructure.
    if (PredSU->hasPhysRegDefs)
      continue;
    // Short-circuit the case where SU is PredSU's only data successor.
    if (PredSU->NumSuccs == 1)
      continue;
    // Avoid prescheduling to copies from virtual registers, which don't behave
    // like other nodes from the perspective of scheduling // heuristics.
    if (SDNode *N = SU->getNode())
      if (N->getOpcode() == ISD::CopyFromReg &&
          TargetRegisterInfo::isVirtualRegister
            (cast<RegisterSDNode>(N->getOperand(1))->getReg()))
        continue;

    // Perform checks on the successors of PredSU.
    for (SUnit::const_succ_iterator II = PredSU->Succs.begin(),
         EE = PredSU->Succs.end(); II != EE; ++II) {
      SUnit *PredSuccSU = II->getSUnit();
      if (PredSuccSU == SU) continue;
      // If PredSU has another successor with no data successors, for
      // now don't attempt to choose either over the other.
      if (PredSuccSU->NumSuccs == 0)
        goto outer_loop_continue;
      // Don't break physical register dependencies.
      if (SU->hasPhysRegClobbers && PredSuccSU->hasPhysRegDefs)
        if (canClobberPhysRegDefs(PredSuccSU, SU, TII, TRI))
          goto outer_loop_continue;
      // Don't introduce graph cycles.
      if (scheduleDAG->IsReachable(SU, PredSuccSU))
        goto outer_loop_continue;
    }

    // Ok, the transformation is safe and the heuristics suggest it is
    // profitable. Update the graph.
    DOUT << "Prescheduling SU # " << SU->NodeNum
         << " next to PredSU # " << PredSU->NodeNum
         << " to guide scheduling in the presence of multiple uses\n";
    for (unsigned i = 0; i != PredSU->Succs.size(); ++i) {
      SDep Edge = PredSU->Succs[i];
      assert(!Edge.isAssignedRegDep());
      SUnit *SuccSU = Edge.getSUnit();
      if (SuccSU != SU) {
        Edge.setSUnit(PredSU);
        scheduleDAG->RemovePred(SuccSU, Edge);
        scheduleDAG->AddPred(SU, Edge);
        Edge.setSUnit(SU);
        scheduleDAG->AddPred(SuccSU, Edge);
        --i;
      }
    }
  outer_loop_continue:;
  }
}

/// AddPseudoTwoAddrDeps - If two nodes share an operand and one of them uses
/// it as a def&use operand. Add a pseudo control edge from it to the other
/// node (if it won't create a cycle) so the two-address one will be scheduled
/// first (lower in the schedule). If both nodes are two-address, favor the
/// one that has a CopyToReg use (more likely to be a loop induction update).
/// If both are two-address, but one is commutable while the other is not
/// commutable, favor the one that's not commutable.
template<class SF>
void RegReductionPriorityQueue<SF>::AddPseudoTwoAddrDeps() {
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i) {
    SUnit *SU = &(*SUnits)[i];
    if (!SU->isTwoAddress)
      continue;

    SDNode *Node = SU->getNode();
    if (!Node || !Node->isMachineOpcode() || SU->getNode()->getFlaggedNode())
      continue;

    unsigned Opc = Node->getMachineOpcode();
    const TargetInstrDesc &TID = TII->get(Opc);
    unsigned NumRes = TID.getNumDefs();
    unsigned NumOps = TID.getNumOperands() - NumRes;
    for (unsigned j = 0; j != NumOps; ++j) {
      if (TID.getOperandConstraint(j+NumRes, TOI::TIED_TO) == -1)
        continue;
      SDNode *DU = SU->getNode()->getOperand(j).getNode();
      if (DU->getNodeId() == -1)
        continue;
      const SUnit *DUSU = &(*SUnits)[DU->getNodeId()];
      if (!DUSU) continue;
      for (SUnit::const_succ_iterator I = DUSU->Succs.begin(),
           E = DUSU->Succs.end(); I != E; ++I) {
        if (I->isCtrl()) continue;
        SUnit *SuccSU = I->getSUnit();
        if (SuccSU == SU)
          continue;
        // Be conservative. Ignore if nodes aren't at roughly the same
        // depth and height.
        if (SuccSU->getHeight() < SU->getHeight() &&
            (SU->getHeight() - SuccSU->getHeight()) > 1)
          continue;
        // Skip past COPY_TO_REGCLASS nodes, so that the pseudo edge
        // constrains whatever is using the copy, instead of the copy
        // itself. In the case that the copy is coalesced, this
        // preserves the intent of the pseudo two-address heurietics.
        while (SuccSU->Succs.size() == 1 &&
               SuccSU->getNode()->isMachineOpcode() &&
               SuccSU->getNode()->getMachineOpcode() ==
                 TargetInstrInfo::COPY_TO_REGCLASS)
          SuccSU = SuccSU->Succs.front().getSUnit();
        // Don't constrain non-instruction nodes.
        if (!SuccSU->getNode() || !SuccSU->getNode()->isMachineOpcode())
          continue;
        // Don't constrain nodes with physical register defs if the
        // predecessor can clobber them.
        if (SuccSU->hasPhysRegDefs && SU->hasPhysRegClobbers) {
          if (canClobberPhysRegDefs(SuccSU, SU, TII, TRI))
            continue;
        }
        // Don't constrain EXTRACT_SUBREG, INSERT_SUBREG, and SUBREG_TO_REG;
        // these may be coalesced away. We want them close to their uses.
        unsigned SuccOpc = SuccSU->getNode()->getMachineOpcode();
        if (SuccOpc == TargetInstrInfo::EXTRACT_SUBREG ||
            SuccOpc == TargetInstrInfo::INSERT_SUBREG ||
            SuccOpc == TargetInstrInfo::SUBREG_TO_REG)
          continue;
        if ((!canClobber(SuccSU, DUSU) ||
             (hasCopyToRegUse(SU) && !hasCopyToRegUse(SuccSU)) ||
             (!SU->isCommutable && SuccSU->isCommutable)) &&
            !scheduleDAG->IsReachable(SuccSU, SU)) {
          DOUT << "Adding a pseudo-two-addr edge from SU # " << SU->NodeNum
               << " to SU #" << SuccSU->NodeNum << "\n";
          scheduleDAG->AddPred(SU, SDep(SuccSU, SDep::Order, /*Latency=*/0,
                                        /*Reg=*/0, /*isNormalMemory=*/false,
                                        /*isMustAlias=*/false,
                                        /*isArtificial=*/true));
        }
      }
    }
  }
}

/// CalculateSethiUllmanNumbers - Calculate Sethi-Ullman numbers of all
/// scheduling units.
template<class SF>
void RegReductionPriorityQueue<SF>::CalculateSethiUllmanNumbers() {
  SethiUllmanNumbers.assign(SUnits->size(), 0);
  
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i)
    CalcNodeSethiUllmanNumber(&(*SUnits)[i], SethiUllmanNumbers);
}

/// LimitedSumOfUnscheduledPredsOfSuccs - Compute the sum of the unscheduled
/// predecessors of the successors of the SUnit SU. Stop when the provided
/// limit is exceeded.
static unsigned LimitedSumOfUnscheduledPredsOfSuccs(const SUnit *SU, 
                                                    unsigned Limit) {
  unsigned Sum = 0;
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    const SUnit *SuccSU = I->getSUnit();
    for (SUnit::const_pred_iterator II = SuccSU->Preds.begin(),
         EE = SuccSU->Preds.end(); II != EE; ++II) {
      SUnit *PredSU = II->getSUnit();
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
  bool LIsTarget = left->getNode() && left->getNode()->isMachineOpcode();
  bool RIsTarget = right->getNode() && right->getNode()->isMachineOpcode();
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

  if (left->getDepth() != right->getDepth())
    return left->getDepth() < right->getDepth();

  if (left->NumSuccsLeft != right->NumSuccsLeft)
    return left->NumSuccsLeft > right->NumSuccsLeft;

  assert(left->NodeQueueId && right->NodeQueueId && 
         "NodeQueueId cannot be zero");
  return (left->NodeQueueId > right->NodeQueueId);
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

llvm::ScheduleDAGSDNodes *
llvm::createBURRListDAGScheduler(SelectionDAGISel *IS, CodeGenOpt::Level) {
  const TargetMachine &TM = IS->TM;
  const TargetInstrInfo *TII = TM.getInstrInfo();
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();
  
  BURegReductionPriorityQueue *PQ = new BURegReductionPriorityQueue(TII, TRI);

  ScheduleDAGRRList *SD =
    new ScheduleDAGRRList(*IS->MF, true, PQ);
  PQ->setScheduleDAG(SD);
  return SD;  
}

llvm::ScheduleDAGSDNodes *
llvm::createTDRRListDAGScheduler(SelectionDAGISel *IS, CodeGenOpt::Level) {
  const TargetMachine &TM = IS->TM;
  const TargetInstrInfo *TII = TM.getInstrInfo();
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();
  
  TDRegReductionPriorityQueue *PQ = new TDRegReductionPriorityQueue(TII, TRI);

  ScheduleDAGRRList *SD =
    new ScheduleDAGRRList(*IS->MF, false, PQ);
  PQ->setScheduleDAG(SD);
  return SD;
}
