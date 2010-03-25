//===--- ScheduleDAGSDNodes.cpp - Implement the ScheduleDAGSDNodes class --===//
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
#include "SDNodeDbgValue.h"
#include "ScheduleDAGSDNodes.h"
#include "InstrEmitter.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(LoadsClustered, "Number of loads clustered together");

ScheduleDAGSDNodes::ScheduleDAGSDNodes(MachineFunction &mf)
  : ScheduleDAG(mf) {
}

/// Run - perform scheduling.
///
void ScheduleDAGSDNodes::Run(SelectionDAG *dag, MachineBasicBlock *bb,
                             MachineBasicBlock::iterator insertPos) {
  DAG = dag;
  ScheduleDAG::Run(bb, insertPos);
}

SUnit *ScheduleDAGSDNodes::Clone(SUnit *Old) {
  SUnit *SU = NewSUnit(Old->getNode());
  SU->OrigNode = Old->OrigNode;
  SU->Latency = Old->Latency;
  SU->isTwoAddress = Old->isTwoAddress;
  SU->isCommutable = Old->isCommutable;
  SU->hasPhysRegDefs = Old->hasPhysRegDefs;
  SU->hasPhysRegClobbers = Old->hasPhysRegClobbers;
  Old->isCloned = true;
  return SU;
}

/// CheckForPhysRegDependency - Check if the dependency between def and use of
/// a specified operand is a physical register dependency. If so, returns the
/// register and the cost of copying the register.
static void CheckForPhysRegDependency(SDNode *Def, SDNode *User, unsigned Op,
                                      const TargetRegisterInfo *TRI, 
                                      const TargetInstrInfo *TII,
                                      unsigned &PhysReg, int &Cost) {
  if (Op != 2 || User->getOpcode() != ISD::CopyToReg)
    return;

  unsigned Reg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
  if (TargetRegisterInfo::isVirtualRegister(Reg))
    return;

  unsigned ResNo = User->getOperand(2).getResNo();
  if (Def->isMachineOpcode()) {
    const TargetInstrDesc &II = TII->get(Def->getMachineOpcode());
    if (ResNo >= II.getNumDefs() &&
        II.ImplicitDefs[ResNo - II.getNumDefs()] == Reg) {
      PhysReg = Reg;
      const TargetRegisterClass *RC =
        TRI->getPhysicalRegisterRegClass(Reg, Def->getValueType(ResNo));
      Cost = RC->getCopyCost();
    }
  }
}

static void AddFlags(SDNode *N, SDValue Flag, bool AddFlag,
                     SelectionDAG *DAG) {
  SmallVector<EVT, 4> VTs;
  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i)
    VTs.push_back(N->getValueType(i));
  if (AddFlag)
    VTs.push_back(MVT::Flag);
  SmallVector<SDValue, 4> Ops;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    Ops.push_back(N->getOperand(i));
  if (Flag.getNode())
    Ops.push_back(Flag);
  SDVTList VTList = DAG->getVTList(&VTs[0], VTs.size());
  DAG->MorphNodeTo(N, N->getOpcode(), VTList, &Ops[0], Ops.size());
}

/// ClusterNeighboringLoads - Force nearby loads together by "flagging" them.
/// This function finds loads of the same base and different offsets. If the
/// offsets are not far apart (target specific), it add MVT::Flag inputs and
/// outputs to ensure they are scheduled together and in order. This
/// optimization may benefit some targets by improving cache locality.
void ScheduleDAGSDNodes::ClusterNeighboringLoads() {
  SmallPtrSet<SDNode*, 16> Visited;
  SmallVector<int64_t, 4> Offsets;
  DenseMap<long long, SDNode*> O2SMap;  // Map from offset to SDNode.
  for (SelectionDAG::allnodes_iterator NI = DAG->allnodes_begin(),
       E = DAG->allnodes_end(); NI != E; ++NI) {
    SDNode *Node = &*NI;
    if (!Node || !Node->isMachineOpcode())
      continue;

    unsigned Opc = Node->getMachineOpcode();
    const TargetInstrDesc &TID = TII->get(Opc);
    if (!TID.mayLoad())
      continue;

    SDNode *Chain = 0;
    unsigned NumOps = Node->getNumOperands();
    if (Node->getOperand(NumOps-1).getValueType() == MVT::Other)
      Chain = Node->getOperand(NumOps-1).getNode();
    if (!Chain)
      continue;

    // Look for other loads of the same chain. Find loads that are loading from
    // the same base pointer and different offsets.
    Visited.clear();
    Offsets.clear();
    O2SMap.clear();
    bool Cluster = false;
    SDNode *Base = Node;
    int64_t BaseOffset;
    for (SDNode::use_iterator I = Chain->use_begin(), E = Chain->use_end();
         I != E; ++I) {
      SDNode *User = *I;
      if (User == Node || !Visited.insert(User))
        continue;
      int64_t Offset1, Offset2;
      if (!TII->areLoadsFromSameBasePtr(Base, User, Offset1, Offset2) ||
          Offset1 == Offset2)
        // FIXME: Should be ok if they addresses are identical. But earlier
        // optimizations really should have eliminated one of the loads.
        continue;
      if (O2SMap.insert(std::make_pair(Offset1, Base)).second)
        Offsets.push_back(Offset1);
      O2SMap.insert(std::make_pair(Offset2, User));
      Offsets.push_back(Offset2);
      if (Offset2 < Offset1) {
        Base = User;
        BaseOffset = Offset2;
      } else {
        BaseOffset = Offset1;
      }
      Cluster = true;
    }

    if (!Cluster)
      continue;

    // Sort them in increasing order.
    std::sort(Offsets.begin(), Offsets.end());

    // Check if the loads are close enough.
    SmallVector<SDNode*, 4> Loads;
    unsigned NumLoads = 0;
    int64_t BaseOff = Offsets[0];
    SDNode *BaseLoad = O2SMap[BaseOff];
    Loads.push_back(BaseLoad);
    for (unsigned i = 1, e = Offsets.size(); i != e; ++i) {
      int64_t Offset = Offsets[i];
      SDNode *Load = O2SMap[Offset];
      if (!TII->shouldScheduleLoadsNear(BaseLoad, Load, BaseOff, Offset,
                                        NumLoads))
        break; // Stop right here. Ignore loads that are further away.
      Loads.push_back(Load);
      ++NumLoads;
    }

    if (NumLoads == 0)
      continue;

    // Cluster loads by adding MVT::Flag outputs and inputs. This also
    // ensure they are scheduled in order of increasing addresses.
    SDNode *Lead = Loads[0];
    AddFlags(Lead, SDValue(0,0), true, DAG);
    SDValue InFlag = SDValue(Lead, Lead->getNumValues()-1);
    for (unsigned i = 1, e = Loads.size(); i != e; ++i) {
      bool OutFlag = i < e-1;
      SDNode *Load = Loads[i];
      AddFlags(Load, InFlag, OutFlag, DAG);
      if (OutFlag)
        InFlag = SDValue(Load, Load->getNumValues()-1);
      ++LoadsClustered;
    }
  }
}

void ScheduleDAGSDNodes::BuildSchedUnits() {
  // During scheduling, the NodeId field of SDNode is used to map SDNodes
  // to their associated SUnits by holding SUnits table indices. A value
  // of -1 means the SDNode does not yet have an associated SUnit.
  unsigned NumNodes = 0;
  for (SelectionDAG::allnodes_iterator NI = DAG->allnodes_begin(),
       E = DAG->allnodes_end(); NI != E; ++NI) {
    NI->setNodeId(-1);
    ++NumNodes;
  }

  // Reserve entries in the vector for each of the SUnits we are creating.  This
  // ensure that reallocation of the vector won't happen, so SUnit*'s won't get
  // invalidated.
  // FIXME: Multiply by 2 because we may clone nodes during scheduling.
  // This is a temporary workaround.
  SUnits.reserve(NumNodes * 2);
  
  // Check to see if the scheduler cares about latencies.
  bool UnitLatencies = ForceUnitLatencies();

  // Add all nodes in depth first order.
  SmallVector<SDNode*, 64> Worklist;
  SmallPtrSet<SDNode*, 64> Visited;
  Worklist.push_back(DAG->getRoot().getNode());
  Visited.insert(DAG->getRoot().getNode());
  
  while (!Worklist.empty()) {
    SDNode *NI = Worklist.pop_back_val();
    
    // Add all operands to the worklist unless they've already been added.
    for (unsigned i = 0, e = NI->getNumOperands(); i != e; ++i)
      if (Visited.insert(NI->getOperand(i).getNode()))
        Worklist.push_back(NI->getOperand(i).getNode());
  
    if (isPassiveNode(NI))  // Leaf node, e.g. a TargetImmediate.
      continue;
    
    // If this node has already been processed, stop now.
    if (NI->getNodeId() != -1) continue;
    
    SUnit *NodeSUnit = NewSUnit(NI);
    
    // See if anything is flagged to this node, if so, add them to flagged
    // nodes.  Nodes can have at most one flag input and one flag output.  Flags
    // are required to be the last operand and result of a node.
    
    // Scan up to find flagged preds.
    SDNode *N = NI;
    while (N->getNumOperands() &&
           N->getOperand(N->getNumOperands()-1).getValueType() == MVT::Flag) {
      N = N->getOperand(N->getNumOperands()-1).getNode();
      assert(N->getNodeId() == -1 && "Node already inserted!");
      N->setNodeId(NodeSUnit->NodeNum);
    }
    
    // Scan down to find any flagged succs.
    N = NI;
    while (N->getValueType(N->getNumValues()-1) == MVT::Flag) {
      SDValue FlagVal(N, N->getNumValues()-1);
      
      // There are either zero or one users of the Flag result.
      bool HasFlagUse = false;
      for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end(); 
           UI != E; ++UI)
        if (FlagVal.isOperandOf(*UI)) {
          HasFlagUse = true;
          assert(N->getNodeId() == -1 && "Node already inserted!");
          N->setNodeId(NodeSUnit->NodeNum);
          N = *UI;
          break;
        }
      if (!HasFlagUse) break;
    }
    
    // If there are flag operands involved, N is now the bottom-most node
    // of the sequence of nodes that are flagged together.
    // Update the SUnit.
    NodeSUnit->setNode(N);
    assert(N->getNodeId() == -1 && "Node already inserted!");
    N->setNodeId(NodeSUnit->NodeNum);

    // Assign the Latency field of NodeSUnit using target-provided information.
    if (UnitLatencies)
      NodeSUnit->Latency = 1;
    else
      ComputeLatency(NodeSUnit);
  }
}

void ScheduleDAGSDNodes::AddSchedEdges() {
  const TargetSubtarget &ST = TM.getSubtarget<TargetSubtarget>();

  // Check to see if the scheduler cares about latencies.
  bool UnitLatencies = ForceUnitLatencies();

  // Pass 2: add the preds, succs, etc.
  for (unsigned su = 0, e = SUnits.size(); su != e; ++su) {
    SUnit *SU = &SUnits[su];
    SDNode *MainNode = SU->getNode();
    
    if (MainNode->isMachineOpcode()) {
      unsigned Opc = MainNode->getMachineOpcode();
      const TargetInstrDesc &TID = TII->get(Opc);
      for (unsigned i = 0; i != TID.getNumOperands(); ++i) {
        if (TID.getOperandConstraint(i, TOI::TIED_TO) != -1) {
          SU->isTwoAddress = true;
          break;
        }
      }
      if (TID.isCommutable())
        SU->isCommutable = true;
    }
    
    // Find all predecessors and successors of the group.
    for (SDNode *N = SU->getNode(); N; N = N->getFlaggedNode()) {
      if (N->isMachineOpcode() &&
          TII->get(N->getMachineOpcode()).getImplicitDefs()) {
        SU->hasPhysRegClobbers = true;
        unsigned NumUsed = InstrEmitter::CountResults(N);
        while (NumUsed != 0 && !N->hasAnyUseOfValue(NumUsed - 1))
          --NumUsed;    // Skip over unused values at the end.
        if (NumUsed > TII->get(N->getMachineOpcode()).getNumDefs())
          SU->hasPhysRegDefs = true;
      }
      
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
        SDNode *OpN = N->getOperand(i).getNode();
        if (isPassiveNode(OpN)) continue;   // Not scheduled.
        SUnit *OpSU = &SUnits[OpN->getNodeId()];
        assert(OpSU && "Node has no SUnit!");
        if (OpSU == SU) continue;           // In the same group.

        EVT OpVT = N->getOperand(i).getValueType();
        assert(OpVT != MVT::Flag && "Flagged nodes should be in same sunit!");
        bool isChain = OpVT == MVT::Other;

        unsigned PhysReg = 0;
        int Cost = 1;
        // Determine if this is a physical register dependency.
        CheckForPhysRegDependency(OpN, N, i, TRI, TII, PhysReg, Cost);
        assert((PhysReg == 0 || !isChain) &&
               "Chain dependence via physreg data?");
        // FIXME: See ScheduleDAGSDNodes::EmitCopyFromReg. For now, scheduler
        // emits a copy from the physical register to a virtual register unless
        // it requires a cross class copy (cost < 0). That means we are only
        // treating "expensive to copy" register dependency as physical register
        // dependency. This may change in the future though.
        if (Cost >= 0)
          PhysReg = 0;

        const SDep& dep = SDep(OpSU, isChain ? SDep::Order : SDep::Data,
                               OpSU->Latency, PhysReg);
        if (!isChain && !UnitLatencies) {
          ComputeOperandLatency(OpSU, SU, (SDep &)dep);
          ST.adjustSchedDependency(OpSU, SU, (SDep &)dep);
        }

        SU->addPred(dep);
      }
    }
  }
}

/// BuildSchedGraph - Build the SUnit graph from the selection dag that we
/// are input.  This SUnit graph is similar to the SelectionDAG, but
/// excludes nodes that aren't interesting to scheduling, and represents
/// flagged together nodes with a single SUnit.
void ScheduleDAGSDNodes::BuildSchedGraph(AliasAnalysis *AA) {
  // Cluster loads from "near" addresses into combined SUnits.
  ClusterNeighboringLoads();
  // Populate the SUnits array.
  BuildSchedUnits();
  // Compute all the scheduling dependencies between nodes.
  AddSchedEdges();
}

void ScheduleDAGSDNodes::ComputeLatency(SUnit *SU) {
  const InstrItineraryData &InstrItins = TM.getInstrItineraryData();
  
  // Compute the latency for the node.  We use the sum of the latencies for
  // all nodes flagged together into this SUnit.
  SU->Latency = 0;
  for (SDNode *N = SU->getNode(); N; N = N->getFlaggedNode())
    if (N->isMachineOpcode()) {
      SU->Latency += InstrItins.
        getStageLatency(TII->get(N->getMachineOpcode()).getSchedClass());
    }
}

void ScheduleDAGSDNodes::dumpNode(const SUnit *SU) const {
  if (!SU->getNode()) {
    dbgs() << "PHYS REG COPY\n";
    return;
  }

  SU->getNode()->dump(DAG);
  dbgs() << "\n";
  SmallVector<SDNode *, 4> FlaggedNodes;
  for (SDNode *N = SU->getNode()->getFlaggedNode(); N; N = N->getFlaggedNode())
    FlaggedNodes.push_back(N);
  while (!FlaggedNodes.empty()) {
    dbgs() << "    ";
    FlaggedNodes.back()->dump(DAG);
    dbgs() << "\n";
    FlaggedNodes.pop_back();
  }
}

namespace {
  struct OrderSorter {
    bool operator()(const std::pair<unsigned, MachineInstr*> &A,
                    const std::pair<unsigned, MachineInstr*> &B) {
      return A.first < B.first;
    }
  };
}

// ProcessSourceNode - Process nodes with source order numbers. These are added
// to a vector which EmitSchedule use to determine how to insert dbg_value
// instructions in the right order.
static void ProcessSourceNode(SDNode *N, SelectionDAG *DAG,
                           InstrEmitter &Emitter,
                           DenseMap<MachineBasicBlock*, MachineBasicBlock*> *EM,
                           DenseMap<SDValue, unsigned> &VRBaseMap,
                    SmallVector<std::pair<unsigned, MachineInstr*>, 32> &Orders,
                           SmallSet<unsigned, 8> &Seen) {
  unsigned Order = DAG->GetOrdering(N);
  if (!Order || !Seen.insert(Order))
    return;

  MachineBasicBlock *BB = Emitter.getBlock();
  if (BB->empty() || BB->back().isPHI()) {
    // Did not insert any instruction.
    Orders.push_back(std::make_pair(Order, (MachineInstr*)0));
    return;
  }

  Orders.push_back(std::make_pair(Order, &BB->back()));
  if (!N->getHasDebugValue())
    return;
  // Opportunistically insert immediate dbg_value uses, i.e. those with source
  // order number right after the N.
  MachineBasicBlock::iterator InsertPos = Emitter.getInsertPos();
  SmallVector<SDDbgValue*,2> &DVs = DAG->GetDbgValues(N);
  for (unsigned i = 0, e = DVs.size(); i != e; ++i) {
    if (DVs[i]->isInvalidated())
      continue;
    unsigned DVOrder = DVs[i]->getOrder();
    if (DVOrder == ++Order) {
      MachineInstr *DbgMI = Emitter.EmitDbgValue(DVs[i], BB, VRBaseMap, EM);
      Orders.push_back(std::make_pair(DVOrder, DbgMI));
      BB->insert(InsertPos, DbgMI);
      DVs[i]->setIsInvalidated();
    }
  }
}


/// EmitSchedule - Emit the machine code in scheduled order.
MachineBasicBlock *ScheduleDAGSDNodes::
EmitSchedule(DenseMap<MachineBasicBlock*, MachineBasicBlock*> *EM) {
  InstrEmitter Emitter(BB, InsertPos);
  DenseMap<SDValue, unsigned> VRBaseMap;
  DenseMap<SUnit*, unsigned> CopyVRBaseMap;
  SmallVector<std::pair<unsigned, MachineInstr*>, 32> Orders;
  SmallSet<unsigned, 8> Seen;
  bool HasDbg = DAG->hasDebugValues();

  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    SUnit *SU = Sequence[i];
    if (!SU) {
      // Null SUnit* is a noop.
      EmitNoop();
      continue;
    }

    // For pre-regalloc scheduling, create instructions corresponding to the
    // SDNode and any flagged SDNodes and append them to the block.
    if (!SU->getNode()) {
      // Emit a copy.
      EmitPhysRegCopy(SU, CopyVRBaseMap);
      continue;
    }

    SmallVector<SDNode *, 4> FlaggedNodes;
    for (SDNode *N = SU->getNode()->getFlaggedNode(); N;
         N = N->getFlaggedNode())
      FlaggedNodes.push_back(N);
    while (!FlaggedNodes.empty()) {
      SDNode *N = FlaggedNodes.back();
      Emitter.EmitNode(FlaggedNodes.back(), SU->OrigNode != SU, SU->isCloned,
                       VRBaseMap, EM);
      // Remember the the source order of the inserted instruction.
      if (HasDbg)
        ProcessSourceNode(N, DAG, Emitter, EM, VRBaseMap, Orders, Seen);
      FlaggedNodes.pop_back();
    }
    Emitter.EmitNode(SU->getNode(), SU->OrigNode != SU, SU->isCloned,
                     VRBaseMap, EM);
    // Remember the the source order of the inserted instruction.
    if (HasDbg)
      ProcessSourceNode(SU->getNode(), DAG, Emitter, EM, VRBaseMap, Orders,
                        Seen);
  }

  // Insert all the dbg_value which have not already been inserted in source
  // order sequence.
  if (HasDbg) {
    MachineBasicBlock::iterator BBBegin = BB->empty() ? BB->end() : BB->begin();
    while (BBBegin != BB->end() && BBBegin->isPHI())
      ++BBBegin;

    // Sort the source order instructions and use the order to insert debug
    // values.
    std::sort(Orders.begin(), Orders.end(), OrderSorter());

    SDDbgInfo::DbgIterator DI = DAG->DbgBegin();
    SDDbgInfo::DbgIterator DE = DAG->DbgEnd();
    // Now emit the rest according to source order.
    unsigned LastOrder = 0;
    MachineInstr *LastMI = 0;
    for (unsigned i = 0, e = Orders.size(); i != e && DI != DE; ++i) {
      unsigned Order = Orders[i].first;
      MachineInstr *MI = Orders[i].second;
      // Insert all SDDbgValue's whose order(s) are before "Order".
      if (!MI)
        continue;
      MachineBasicBlock *MIBB = MI->getParent();
#ifndef NDEBUG
      unsigned LastDIOrder = 0;
#endif
      for (; DI != DE &&
             (*DI)->getOrder() >= LastOrder && (*DI)->getOrder() < Order; ++DI) {
#ifndef NDEBUG
        assert((*DI)->getOrder() >= LastDIOrder &&
               "SDDbgValue nodes must be in source order!");
        LastDIOrder = (*DI)->getOrder();
#endif
        if ((*DI)->isInvalidated())
          continue;
        MachineInstr *DbgMI = Emitter.EmitDbgValue(*DI, MIBB, VRBaseMap, EM);
        if (!LastOrder)
          // Insert to start of the BB (after PHIs).
          BB->insert(BBBegin, DbgMI);
        else {
          MachineBasicBlock::iterator Pos = MI;
          MIBB->insert(llvm::next(Pos), DbgMI);
        }
      }
      LastOrder = Order;
      LastMI = MI;
    }
    // Add trailing DbgValue's before the terminator. FIXME: May want to add
    // some of them before one or more conditional branches?
    while (DI != DE) {
      MachineBasicBlock *InsertBB = Emitter.getBlock();
      MachineBasicBlock::iterator Pos= Emitter.getBlock()->getFirstTerminator();
      if (!(*DI)->isInvalidated()) {
        MachineInstr *DbgMI= Emitter.EmitDbgValue(*DI, InsertBB, VRBaseMap, EM);
        InsertBB->insert(Pos, DbgMI);
      }
      ++DI;
    }
  }

  BB = Emitter.getBlock();
  InsertPos = Emitter.getInsertPos();
  return BB;
}
