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
using namespace llvm;

ScheduleDAG::ScheduleDAG(SelectionDAG &dag, MachineBasicBlock *bb,
                         const TargetMachine &tm)
  : DAG(dag), BB(bb), TM(tm), MRI(BB->getParent()->getRegInfo()) {
  TII = TM.getInstrInfo();
  MF  = &DAG.getMachineFunction();
  TRI = TM.getRegisterInfo();
  TLI = &DAG.getTargetLoweringInfo();
  ConstPool = BB->getParent()->getConstantPool();
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

SUnit *ScheduleDAG::Clone(SUnit *Old) {
  SUnit *SU = NewSUnit(Old->Node);
  SU->OrigNode = Old->OrigNode;
  SU->FlaggedNodes = Old->FlaggedNodes;
  SU->Latency = Old->Latency;
  SU->isTwoAddress = Old->isTwoAddress;
  SU->isCommutable = Old->isCommutable;
  SU->hasPhysRegDefs = Old->hasPhysRegDefs;
  return SU;
}


/// BuildSchedUnits - Build SUnits from the selection dag that we are input.
/// This SUnit graph is similar to the SelectionDAG, but represents flagged
/// together nodes with a single SUnit.
void ScheduleDAG::BuildSchedUnits() {
  // Reserve entries in the vector for each of the SUnits we are creating.  This
  // ensure that reallocation of the vector won't happen, so SUnit*'s won't get
  // invalidated.
  SUnits.reserve(DAG.allnodes_size());
  
  // During scheduling, the NodeId field of SDNode is used to map SDNodes
  // to their associated SUnits by holding SUnits table indices. A value
  // of -1 means the SDNode does not yet have an associated SUnit.
  for (SelectionDAG::allnodes_iterator NI = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); NI != E; ++NI)
    NI->setNodeId(-1);

  for (SelectionDAG::allnodes_iterator NI = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); NI != E; ++NI) {
    if (isPassiveNode(NI))  // Leaf node, e.g. a TargetImmediate.
      continue;
    
    // If this node has already been processed, stop now.
    if (NI->getNodeId() != -1) continue;
    
    SUnit *NodeSUnit = NewSUnit(NI);
    
    // See if anything is flagged to this node, if so, add them to flagged
    // nodes.  Nodes can have at most one flag input and one flag output.  Flags
    // are required the be the last operand and result of a node.
    
    // Scan up, adding flagged preds to FlaggedNodes.
    SDNode *N = NI;
    if (N->getNumOperands() &&
        N->getOperand(N->getNumOperands()-1).getValueType() == MVT::Flag) {
      do {
        N = N->getOperand(N->getNumOperands()-1).getNode();
        NodeSUnit->FlaggedNodes.push_back(N);
        assert(N->getNodeId() == -1 && "Node already inserted!");
        N->setNodeId(NodeSUnit->NodeNum);
      } while (N->getNumOperands() &&
               N->getOperand(N->getNumOperands()-1).getValueType()== MVT::Flag);
      std::reverse(NodeSUnit->FlaggedNodes.begin(),
                   NodeSUnit->FlaggedNodes.end());
    }
    
    // Scan down, adding this node and any flagged succs to FlaggedNodes if they
    // have a user of the flag operand.
    N = NI;
    while (N->getValueType(N->getNumValues()-1) == MVT::Flag) {
      SDValue FlagVal(N, N->getNumValues()-1);
      
      // There are either zero or one users of the Flag result.
      bool HasFlagUse = false;
      for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end(); 
           UI != E; ++UI)
        if (FlagVal.isOperandOf(*UI)) {
          HasFlagUse = true;
          NodeSUnit->FlaggedNodes.push_back(N);
          assert(N->getNodeId() == -1 && "Node already inserted!");
          N->setNodeId(NodeSUnit->NodeNum);
          N = *UI;
          break;
        }
      if (!HasFlagUse) break;
    }
    
    // Now all flagged nodes are in FlaggedNodes and N is the bottom-most node.
    // Update the SUnit
    NodeSUnit->Node = N;
    assert(N->getNodeId() == -1 && "Node already inserted!");
    N->setNodeId(NodeSUnit->NodeNum);

    ComputeLatency(NodeSUnit);
  }
  
  // Pass 2: add the preds, succs, etc.
  for (unsigned su = 0, e = SUnits.size(); su != e; ++su) {
    SUnit *SU = &SUnits[su];
    SDNode *MainNode = SU->Node;
    
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
    // Temporarily add N to make code simpler.
    SU->FlaggedNodes.push_back(MainNode);
    
    for (unsigned n = 0, e = SU->FlaggedNodes.size(); n != e; ++n) {
      SDNode *N = SU->FlaggedNodes[n];
      if (N->isMachineOpcode() &&
          TII->get(N->getMachineOpcode()).getImplicitDefs() &&
          CountResults(N) > TII->get(N->getMachineOpcode()).getNumDefs())
        SU->hasPhysRegDefs = true;
      
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
        SDNode *OpN = N->getOperand(i).getNode();
        if (isPassiveNode(OpN)) continue;   // Not scheduled.
        SUnit *OpSU = &SUnits[OpN->getNodeId()];
        assert(OpSU && "Node has no SUnit!");
        if (OpSU == SU) continue;           // In the same group.

        MVT OpVT = N->getOperand(i).getValueType();
        assert(OpVT != MVT::Flag && "Flagged nodes should be in same sunit!");
        bool isChain = OpVT == MVT::Other;

        unsigned PhysReg = 0;
        int Cost = 1;
        // Determine if this is a physical register dependency.
        CheckForPhysRegDependency(OpN, N, i, TRI, TII, PhysReg, Cost);
        SU->addPred(OpSU, isChain, false, PhysReg, Cost);
      }
    }
    
    // Remove MainNode from FlaggedNodes again.
    SU->FlaggedNodes.pop_back();
  }
}

void ScheduleDAG::ComputeLatency(SUnit *SU) {
  const InstrItineraryData &InstrItins = TM.getInstrItineraryData();
  
  // Compute the latency for the node.  We use the sum of the latencies for
  // all nodes flagged together into this SUnit.
  if (InstrItins.isEmpty()) {
    // No latency information.
    SU->Latency = 1;
    return;
  }

  SU->Latency = 0;
  if (SU->Node->isMachineOpcode()) {
    unsigned SchedClass = TII->get(SU->Node->getMachineOpcode()).getSchedClass();
    const InstrStage *S = InstrItins.begin(SchedClass);
    const InstrStage *E = InstrItins.end(SchedClass);
    for (; S != E; ++S)
      SU->Latency += S->Cycles;
  }
  for (unsigned i = 0, e = SU->FlaggedNodes.size(); i != e; ++i) {
    SDNode *FNode = SU->FlaggedNodes[i];
    if (FNode->isMachineOpcode()) {
      unsigned SchedClass = TII->get(FNode->getMachineOpcode()).getSchedClass();
      const InstrStage *S = InstrItins.begin(SchedClass);
      const InstrStage *E = InstrItins.end(SchedClass);
      for (; S != E; ++S)
        SU->Latency += S->Cycles;
    }
  }
}

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
      unsigned PredDepth = I->Dep->Depth;
      if (PredDepth+1 > SUDepth) {
          SUDepth = PredDepth + 1;
      }
    }

    SU->Depth = SUDepth;

    // Update degrees of all nodes depending on current SUnit
    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      SUnit *SU = I->Dep;
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
      unsigned SuccHeight = I->Dep->Height;
      if (SuccHeight+1 > SUHeight) {
          SUHeight = SuccHeight + 1;
      }
    }

    SU->Height = SUHeight;

    // Update degrees of all nodes depending on current SUnit
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      SUnit *SU = I->Dep;
      if (!--SU->Height)
        // If all dependencies of the node are processed already,
        // then the longest path for the node can be computed now
        WorkList.push_back(SU);
    }
  }
}

/// CountResults - The results of target nodes have register or immediate
/// operands first, then an optional chain, and optional flag operands (which do
/// not go into the resulting MachineInstr).
unsigned ScheduleDAG::CountResults(SDNode *Node) {
  unsigned N = Node->getNumValues();
  while (N && Node->getValueType(N - 1) == MVT::Flag)
    --N;
  if (N && Node->getValueType(N - 1) == MVT::Other)
    --N;    // Skip over chain result.
  return N;
}

/// CountOperands - The inputs to target nodes have any actual inputs first,
/// followed by special operands that describe memory references, then an
/// optional chain operand, then flag operands.  Compute the number of
/// actual operands that will go into the resulting MachineInstr.
unsigned ScheduleDAG::CountOperands(SDNode *Node) {
  unsigned N = ComputeMemOperandsEnd(Node);
  while (N && isa<MemOperandSDNode>(Node->getOperand(N - 1).getNode()))
    --N; // Ignore MEMOPERAND nodes
  return N;
}

/// ComputeMemOperandsEnd - Find the index one past the last MemOperandSDNode
/// operand
unsigned ScheduleDAG::ComputeMemOperandsEnd(SDNode *Node) {
  unsigned N = Node->getNumOperands();
  while (N && Node->getOperand(N - 1).getValueType() == MVT::Flag)
    --N;
  if (N && Node->getOperand(N - 1).getValueType() == MVT::Other)
    --N; // Ignore chain if it exists.
  return N;
}


/// dump - dump the schedule.
void ScheduleDAG::dumpSchedule() const {
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    if (SUnit *SU = Sequence[i])
      SU->dump(&DAG);
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
void SUnit::dump(const SelectionDAG *G) const {
  cerr << "SU(" << NodeNum << "): ";
  if (Node)
    Node->dump(G);
  else
    cerr << "CROSS RC COPY ";
  cerr << "\n";
  if (FlaggedNodes.size() != 0) {
    for (unsigned i = 0, e = FlaggedNodes.size(); i != e; i++) {
      cerr << "    ";
      FlaggedNodes[i]->dump(G);
      cerr << "\n";
    }
  }
}

void SUnit::dumpAll(const SelectionDAG *G) const {
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
      if (I->isCtrl)
        cerr << "   ch  #";
      else
        cerr << "   val #";
      cerr << I->Dep << " - SU(" << I->Dep->NodeNum << ")";
      if (I->isSpecial)
        cerr << " *";
      cerr << "\n";
    }
  }
  if (Succs.size() != 0) {
    cerr << "  Successors:\n";
    for (SUnit::const_succ_iterator I = Succs.begin(), E = Succs.end();
         I != E; ++I) {
      if (I->isCtrl)
        cerr << "   ch  #";
      else
        cerr << "   val #";
      cerr << I->Dep << " - SU(" << I->Dep->NodeNum << ")";
      if (I->isSpecial)
        cerr << " *";
      cerr << "\n";
    }
  }
  cerr << "\n";
}
