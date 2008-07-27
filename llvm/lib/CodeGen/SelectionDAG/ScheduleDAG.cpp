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
#include "llvm/Type.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

STATISTIC(NumCommutes,   "Number of instructions commuted");

namespace {
  static cl::opt<bool>
  SchedLiveInCopies("schedule-livein-copies",
                    cl::desc("Schedule copies of livein registers"),
                    cl::init(false));
}

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

  unsigned ResNo = User->getOperand(2).ResNo;
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
        N = N->getOperand(N->getNumOperands()-1).Val;
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
        SDNode *OpN = N->getOperand(i).Val;
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
  std::vector<unsigned> InDegree(DAGSize);
  std::vector<SUnit*> WorkList;
  WorkList.reserve(DAGSize);

  // Initialize the data structures
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    int NodeNum = SU->NodeNum;
    unsigned Degree = SU->Preds.size();
    InDegree[NodeNum] = Degree;
    SU->Depth = 0;

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
    unsigned &SUDepth  = SU->Depth;

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

    // Update InDegrees of all nodes depending on current SUnit
    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      SUnit *SU = I->Dep;
      if (!--InDegree[SU->NodeNum])
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
  std::vector<unsigned> InDegree(DAGSize);
  std::vector<SUnit*> WorkList;
  WorkList.reserve(DAGSize);

  // Initialize the data structures
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    int NodeNum = SU->NodeNum;
    unsigned Degree = SU->Succs.size();
    InDegree[NodeNum] = Degree;
    SU->Height = 0;

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
    unsigned &SUHeight  = SU->Height;

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

    // Update InDegrees of all nodes depending on current SUnit
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      SUnit *SU = I->Dep;
      if (!--InDegree[SU->NodeNum])
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
  while (N && isa<MemOperandSDNode>(Node->getOperand(N - 1).Val))
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

/// getInstrOperandRegClass - Return register class of the operand of an
/// instruction of the specified TargetInstrDesc.
static const TargetRegisterClass*
getInstrOperandRegClass(const TargetRegisterInfo *TRI, 
                        const TargetInstrInfo *TII, const TargetInstrDesc &II,
                        unsigned Op) {
  if (Op >= II.getNumOperands()) {
    assert(II.isVariadic() && "Invalid operand # of instruction");
    return NULL;
  }
  if (II.OpInfo[Op].isLookupPtrRegClass())
    return TII->getPointerRegClass();
  return TRI->getRegClass(II.OpInfo[Op].RegClass);
}

/// EmitCopyFromReg - Generate machine code for an CopyFromReg node or an
/// implicit physical register output.
void ScheduleDAG::EmitCopyFromReg(SDNode *Node, unsigned ResNo,
                                  bool IsClone, unsigned SrcReg,
                                  DenseMap<SDValue, unsigned> &VRBaseMap) {
  unsigned VRBase = 0;
  if (TargetRegisterInfo::isVirtualRegister(SrcReg)) {
    // Just use the input register directly!
    SDValue Op(Node, ResNo);
    if (IsClone)
      VRBaseMap.erase(Op);
    bool isNew = VRBaseMap.insert(std::make_pair(Op, SrcReg)).second;
    isNew = isNew; // Silence compiler warning.
    assert(isNew && "Node emitted out of order - early");
    return;
  }

  // If the node is only used by a CopyToReg and the dest reg is a vreg, use
  // the CopyToReg'd destination register instead of creating a new vreg.
  bool MatchReg = true;
  for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
       UI != E; ++UI) {
    SDNode *User = *UI;
    bool Match = true;
    if (User->getOpcode() == ISD::CopyToReg && 
        User->getOperand(2).Val == Node &&
        User->getOperand(2).ResNo == ResNo) {
      unsigned DestReg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
      if (TargetRegisterInfo::isVirtualRegister(DestReg)) {
        VRBase = DestReg;
        Match = false;
      } else if (DestReg != SrcReg)
        Match = false;
    } else {
      for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i) {
        SDValue Op = User->getOperand(i);
        if (Op.Val != Node || Op.ResNo != ResNo)
          continue;
        MVT VT = Node->getValueType(Op.ResNo);
        if (VT != MVT::Other && VT != MVT::Flag)
          Match = false;
      }
    }
    MatchReg &= Match;
    if (VRBase)
      break;
  }

  const TargetRegisterClass *SrcRC = 0, *DstRC = 0;
  SrcRC = TRI->getPhysicalRegisterRegClass(SrcReg, Node->getValueType(ResNo));
  
  // Figure out the register class to create for the destreg.
  if (VRBase) {
    DstRC = MRI.getRegClass(VRBase);
  } else {
    DstRC = TLI->getRegClassFor(Node->getValueType(ResNo));
  }
    
  // If all uses are reading from the src physical register and copying the
  // register is either impossible or very expensive, then don't create a copy.
  if (MatchReg && SrcRC->getCopyCost() < 0) {
    VRBase = SrcReg;
  } else {
    // Create the reg, emit the copy.
    VRBase = MRI.createVirtualRegister(DstRC);
    TII->copyRegToReg(*BB, BB->end(), VRBase, SrcReg, DstRC, SrcRC);
  }

  SDValue Op(Node, ResNo);
  if (IsClone)
    VRBaseMap.erase(Op);
  bool isNew = VRBaseMap.insert(std::make_pair(Op, VRBase)).second;
  isNew = isNew; // Silence compiler warning.
  assert(isNew && "Node emitted out of order - early");
}

/// getDstOfCopyToRegUse - If the only use of the specified result number of
/// node is a CopyToReg, return its destination register. Return 0 otherwise.
unsigned ScheduleDAG::getDstOfOnlyCopyToRegUse(SDNode *Node,
                                               unsigned ResNo) const {
  if (!Node->hasOneUse())
    return 0;

  SDNode *User = *Node->use_begin();
  if (User->getOpcode() == ISD::CopyToReg && 
      User->getOperand(2).Val == Node &&
      User->getOperand(2).ResNo == ResNo) {
    unsigned Reg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      return Reg;
  }
  return 0;
}

void ScheduleDAG::CreateVirtualRegisters(SDNode *Node, MachineInstr *MI,
                                 const TargetInstrDesc &II,
                                 DenseMap<SDValue, unsigned> &VRBaseMap) {
  assert(Node->getMachineOpcode() != TargetInstrInfo::IMPLICIT_DEF &&
         "IMPLICIT_DEF should have been handled as a special case elsewhere!");

  for (unsigned i = 0; i < II.getNumDefs(); ++i) {
    // If the specific node value is only used by a CopyToReg and the dest reg
    // is a vreg, use the CopyToReg'd destination register instead of creating
    // a new vreg.
    unsigned VRBase = 0;
    for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
         UI != E; ++UI) {
      SDNode *User = *UI;
      if (User->getOpcode() == ISD::CopyToReg && 
          User->getOperand(2).Val == Node &&
          User->getOperand(2).ResNo == i) {
        unsigned Reg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
        if (TargetRegisterInfo::isVirtualRegister(Reg)) {
          VRBase = Reg;
          MI->addOperand(MachineOperand::CreateReg(Reg, true));
          break;
        }
      }
    }

    // Create the result registers for this node and add the result regs to
    // the machine instruction.
    if (VRBase == 0) {
      const TargetRegisterClass *RC = getInstrOperandRegClass(TRI, TII, II, i);
      assert(RC && "Isn't a register operand!");
      VRBase = MRI.createVirtualRegister(RC);
      MI->addOperand(MachineOperand::CreateReg(VRBase, true));
    }

    SDValue Op(Node, i);
    bool isNew = VRBaseMap.insert(std::make_pair(Op, VRBase)).second;
    isNew = isNew; // Silence compiler warning.
    assert(isNew && "Node emitted out of order - early");
  }
}

/// getVR - Return the virtual register corresponding to the specified result
/// of the specified node.
unsigned ScheduleDAG::getVR(SDValue Op,
                            DenseMap<SDValue, unsigned> &VRBaseMap) {
  if (Op.isMachineOpcode() &&
      Op.getMachineOpcode() == TargetInstrInfo::IMPLICIT_DEF) {
    // Add an IMPLICIT_DEF instruction before every use.
    unsigned VReg = getDstOfOnlyCopyToRegUse(Op.Val, Op.ResNo);
    // IMPLICIT_DEF can produce any type of result so its TargetInstrDesc
    // does not include operand register class info.
    if (!VReg) {
      const TargetRegisterClass *RC = TLI->getRegClassFor(Op.getValueType());
      VReg = MRI.createVirtualRegister(RC);
    }
    BuildMI(BB, TII->get(TargetInstrInfo::IMPLICIT_DEF), VReg);
    return VReg;
  }

  DenseMap<SDValue, unsigned>::iterator I = VRBaseMap.find(Op);
  assert(I != VRBaseMap.end() && "Node emitted out of order - late");
  return I->second;
}


/// AddOperand - Add the specified operand to the specified machine instr.  II
/// specifies the instruction information for the node, and IIOpNum is the
/// operand number (in the II) that we are adding. IIOpNum and II are used for 
/// assertions only.
void ScheduleDAG::AddOperand(MachineInstr *MI, SDValue Op,
                             unsigned IIOpNum,
                             const TargetInstrDesc *II,
                             DenseMap<SDValue, unsigned> &VRBaseMap) {
  if (Op.isMachineOpcode()) {
    // Note that this case is redundant with the final else block, but we
    // include it because it is the most common and it makes the logic
    // simpler here.
    assert(Op.getValueType() != MVT::Other &&
           Op.getValueType() != MVT::Flag &&
           "Chain and flag operands should occur at end of operand list!");
    // Get/emit the operand.
    unsigned VReg = getVR(Op, VRBaseMap);
    const TargetInstrDesc &TID = MI->getDesc();
    bool isOptDef = IIOpNum < TID.getNumOperands() &&
      TID.OpInfo[IIOpNum].isOptionalDef();
    MI->addOperand(MachineOperand::CreateReg(VReg, isOptDef));
    
    // Verify that it is right.
    assert(TargetRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");
#ifndef NDEBUG
    if (II) {
      // There may be no register class for this operand if it is a variadic
      // argument (RC will be NULL in this case).  In this case, we just assume
      // the regclass is ok.
      const TargetRegisterClass *RC =
                          getInstrOperandRegClass(TRI, TII, *II, IIOpNum);
      assert((RC || II->isVariadic()) && "Expected reg class info!");
      const TargetRegisterClass *VRC = MRI.getRegClass(VReg);
      if (RC && VRC != RC) {
        cerr << "Register class of operand and regclass of use don't agree!\n";
        cerr << "Operand = " << IIOpNum << "\n";
        cerr << "Op->Val = "; Op.Val->dump(&DAG); cerr << "\n";
        cerr << "MI = "; MI->print(cerr);
        cerr << "VReg = " << VReg << "\n";
        cerr << "VReg RegClass     size = " << VRC->getSize()
             << ", align = " << VRC->getAlignment() << "\n";
        cerr << "Expected RegClass size = " << RC->getSize()
             << ", align = " << RC->getAlignment() << "\n";
        cerr << "Fatal error, aborting.\n";
        abort();
      }
    }
#endif
  } else if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateImm(C->getValue()));
  } else if (ConstantFPSDNode *F = dyn_cast<ConstantFPSDNode>(Op)) {
    ConstantFP *CFP = ConstantFP::get(F->getValueAPF());
    MI->addOperand(MachineOperand::CreateFPImm(CFP));
  } else if (RegisterSDNode *R = dyn_cast<RegisterSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateReg(R->getReg(), false));
  } else if (GlobalAddressSDNode *TGA = dyn_cast<GlobalAddressSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateGA(TGA->getGlobal(),TGA->getOffset()));
  } else if (BasicBlockSDNode *BB = dyn_cast<BasicBlockSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateMBB(BB->getBasicBlock()));
  } else if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateFI(FI->getIndex()));
  } else if (JumpTableSDNode *JT = dyn_cast<JumpTableSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateJTI(JT->getIndex()));
  } else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Op)) {
    int Offset = CP->getOffset();
    unsigned Align = CP->getAlignment();
    const Type *Type = CP->getType();
    // MachineConstantPool wants an explicit alignment.
    if (Align == 0) {
      Align = TM.getTargetData()->getPreferredTypeAlignmentShift(Type);
      if (Align == 0) {
        // Alignment of vector types.  FIXME!
        Align = TM.getTargetData()->getABITypeSize(Type);
        Align = Log2_64(Align);
      }
    }
    
    unsigned Idx;
    if (CP->isMachineConstantPoolEntry())
      Idx = ConstPool->getConstantPoolIndex(CP->getMachineCPVal(), Align);
    else
      Idx = ConstPool->getConstantPoolIndex(CP->getConstVal(), Align);
    MI->addOperand(MachineOperand::CreateCPI(Idx, Offset));
  } else if (ExternalSymbolSDNode *ES = dyn_cast<ExternalSymbolSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateES(ES->getSymbol()));
  } else {
    assert(Op.getValueType() != MVT::Other &&
           Op.getValueType() != MVT::Flag &&
           "Chain and flag operands should occur at end of operand list!");
    unsigned VReg = getVR(Op, VRBaseMap);
    MI->addOperand(MachineOperand::CreateReg(VReg, false));
    
    // Verify that it is right.  Note that the reg class of the physreg and the
    // vreg don't necessarily need to match, but the target copy insertion has
    // to be able to handle it.  This handles things like copies from ST(0) to
    // an FP vreg on x86.
    assert(TargetRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");
    if (II && !II->isVariadic()) {
      assert(getInstrOperandRegClass(TRI, TII, *II, IIOpNum) &&
             "Don't have operand info for this instruction!");
    }
  }  
}

void ScheduleDAG::AddMemOperand(MachineInstr *MI, const MachineMemOperand &MO) {
  MI->addMemOperand(*MF, MO);
}

/// getSubRegisterRegClass - Returns the register class of specified register
/// class' "SubIdx"'th sub-register class.
static const TargetRegisterClass*
getSubRegisterRegClass(const TargetRegisterClass *TRC, unsigned SubIdx) {
  // Pick the register class of the subregister
  TargetRegisterInfo::regclass_iterator I =
    TRC->subregclasses_begin() + SubIdx-1;
  assert(I < TRC->subregclasses_end() && 
         "Invalid subregister index for register class");
  return *I;
}

/// getSuperRegisterRegClass - Returns the register class of a superreg A whose
/// "SubIdx"'th sub-register class is the specified register class and whose
/// type matches the specified type.
static const TargetRegisterClass*
getSuperRegisterRegClass(const TargetRegisterClass *TRC,
                         unsigned SubIdx, MVT VT) {
  // Pick the register class of the superegister for this type
  for (TargetRegisterInfo::regclass_iterator I = TRC->superregclasses_begin(),
         E = TRC->superregclasses_end(); I != E; ++I)
    if ((*I)->hasType(VT) && getSubRegisterRegClass(*I, SubIdx) == TRC)
      return *I;
  assert(false && "Couldn't find the register class");
  return 0;
}

/// EmitSubregNode - Generate machine code for subreg nodes.
///
void ScheduleDAG::EmitSubregNode(SDNode *Node, 
                           DenseMap<SDValue, unsigned> &VRBaseMap) {
  unsigned VRBase = 0;
  unsigned Opc = Node->getMachineOpcode();
  
  // If the node is only used by a CopyToReg and the dest reg is a vreg, use
  // the CopyToReg'd destination register instead of creating a new vreg.
  for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
       UI != E; ++UI) {
    SDNode *User = *UI;
    if (User->getOpcode() == ISD::CopyToReg && 
        User->getOperand(2).Val == Node) {
      unsigned DestReg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
      if (TargetRegisterInfo::isVirtualRegister(DestReg)) {
        VRBase = DestReg;
        break;
      }
    }
  }
  
  if (Opc == TargetInstrInfo::EXTRACT_SUBREG) {
    unsigned SubIdx = cast<ConstantSDNode>(Node->getOperand(1))->getValue();

    // Create the extract_subreg machine instruction.
    MachineInstr *MI = BuildMI(*MF, TII->get(TargetInstrInfo::EXTRACT_SUBREG));

    // Figure out the register class to create for the destreg.
    unsigned VReg = getVR(Node->getOperand(0), VRBaseMap);
    const TargetRegisterClass *TRC = MRI.getRegClass(VReg);
    const TargetRegisterClass *SRC = getSubRegisterRegClass(TRC, SubIdx);

    if (VRBase) {
      // Grab the destination register
#ifndef NDEBUG
      const TargetRegisterClass *DRC = MRI.getRegClass(VRBase);
      assert(SRC && DRC && SRC == DRC && 
             "Source subregister and destination must have the same class");
#endif
    } else {
      // Create the reg
      assert(SRC && "Couldn't find source register class");
      VRBase = MRI.createVirtualRegister(SRC);
    }
    
    // Add def, source, and subreg index
    MI->addOperand(MachineOperand::CreateReg(VRBase, true));
    AddOperand(MI, Node->getOperand(0), 0, 0, VRBaseMap);
    MI->addOperand(MachineOperand::CreateImm(SubIdx));
    BB->push_back(MI);    
  } else if (Opc == TargetInstrInfo::INSERT_SUBREG ||
             Opc == TargetInstrInfo::SUBREG_TO_REG) {
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);
    SDValue N2 = Node->getOperand(2);
    unsigned SubReg = getVR(N1, VRBaseMap);
    unsigned SubIdx = cast<ConstantSDNode>(N2)->getValue();
    
      
    // Figure out the register class to create for the destreg.
    const TargetRegisterClass *TRC = 0;
    if (VRBase) {
      TRC = MRI.getRegClass(VRBase);
    } else {
      TRC = getSuperRegisterRegClass(MRI.getRegClass(SubReg), SubIdx, 
                                     Node->getValueType(0));
      assert(TRC && "Couldn't determine register class for insert_subreg");
      VRBase = MRI.createVirtualRegister(TRC); // Create the reg
    }
    
    // Create the insert_subreg or subreg_to_reg machine instruction.
    MachineInstr *MI = BuildMI(*MF, TII->get(Opc));
    MI->addOperand(MachineOperand::CreateReg(VRBase, true));
    
    // If creating a subreg_to_reg, then the first input operand
    // is an implicit value immediate, otherwise it's a register
    if (Opc == TargetInstrInfo::SUBREG_TO_REG) {
      const ConstantSDNode *SD = cast<ConstantSDNode>(N0);
      MI->addOperand(MachineOperand::CreateImm(SD->getValue()));
    } else
      AddOperand(MI, N0, 0, 0, VRBaseMap);
    // Add the subregster being inserted
    AddOperand(MI, N1, 0, 0, VRBaseMap);
    MI->addOperand(MachineOperand::CreateImm(SubIdx));
    BB->push_back(MI);
  } else
    assert(0 && "Node is not insert_subreg, extract_subreg, or subreg_to_reg");
     
  SDValue Op(Node, 0);
  bool isNew = VRBaseMap.insert(std::make_pair(Op, VRBase)).second;
  isNew = isNew; // Silence compiler warning.
  assert(isNew && "Node emitted out of order - early");
}

/// EmitNode - Generate machine code for an node and needed dependencies.
///
void ScheduleDAG::EmitNode(SDNode *Node, bool IsClone,
                           DenseMap<SDValue, unsigned> &VRBaseMap) {
  // If machine instruction
  if (Node->isMachineOpcode()) {
    unsigned Opc = Node->getMachineOpcode();
    
    // Handle subreg insert/extract specially
    if (Opc == TargetInstrInfo::EXTRACT_SUBREG || 
        Opc == TargetInstrInfo::INSERT_SUBREG ||
        Opc == TargetInstrInfo::SUBREG_TO_REG) {
      EmitSubregNode(Node, VRBaseMap);
      return;
    }

    if (Opc == TargetInstrInfo::IMPLICIT_DEF)
      // We want a unique VR for each IMPLICIT_DEF use.
      return;
    
    const TargetInstrDesc &II = TII->get(Opc);
    unsigned NumResults = CountResults(Node);
    unsigned NodeOperands = CountOperands(Node);
    unsigned MemOperandsEnd = ComputeMemOperandsEnd(Node);
    bool HasPhysRegOuts = (NumResults > II.getNumDefs()) &&
                          II.getImplicitDefs() != 0;
#ifndef NDEBUG
    unsigned NumMIOperands = NodeOperands + NumResults;
    assert((II.getNumOperands() == NumMIOperands ||
            HasPhysRegOuts || II.isVariadic()) &&
           "#operands for dag node doesn't match .td file!"); 
#endif

    // Create the new machine instruction.
    MachineInstr *MI = BuildMI(*MF, II);
    
    // Add result register values for things that are defined by this
    // instruction.
    if (NumResults)
      CreateVirtualRegisters(Node, MI, II, VRBaseMap);
    
    // Emit all of the actual operands of this instruction, adding them to the
    // instruction as appropriate.
    for (unsigned i = 0; i != NodeOperands; ++i)
      AddOperand(MI, Node->getOperand(i), i+II.getNumDefs(), &II, VRBaseMap);

    // Emit all of the memory operands of this instruction
    for (unsigned i = NodeOperands; i != MemOperandsEnd; ++i)
      AddMemOperand(MI, cast<MemOperandSDNode>(Node->getOperand(i))->MO);

    // Commute node if it has been determined to be profitable.
    if (CommuteSet.count(Node)) {
      MachineInstr *NewMI = TII->commuteInstruction(MI);
      if (NewMI == 0)
        DOUT << "Sched: COMMUTING FAILED!\n";
      else {
        DOUT << "Sched: COMMUTED TO: " << *NewMI;
        if (MI != NewMI) {
          MF->DeleteMachineInstr(MI);
          MI = NewMI;
        }
        ++NumCommutes;
      }
    }

    if (II.usesCustomDAGSchedInsertionHook())
      // Insert this instruction into the basic block using a target
      // specific inserter which may returns a new basic block.
      BB = TLI->EmitInstrWithCustomInserter(MI, BB);
    else
      BB->push_back(MI);

    // Additional results must be an physical register def.
    if (HasPhysRegOuts) {
      for (unsigned i = II.getNumDefs(); i < NumResults; ++i) {
        unsigned Reg = II.getImplicitDefs()[i - II.getNumDefs()];
        if (Node->hasAnyUseOfValue(i))
          EmitCopyFromReg(Node, i, IsClone, Reg, VRBaseMap);
      }
    }
    return;
  }

  switch (Node->getOpcode()) {
  default:
#ifndef NDEBUG
    Node->dump(&DAG);
#endif
    assert(0 && "This target-independent node should have been selected!");
    break;
  case ISD::EntryToken:
    assert(0 && "EntryToken should have been excluded from the schedule!");
    break;
  case ISD::TokenFactor: // fall thru
    break;
  case ISD::CopyToReg: {
    unsigned SrcReg;
    SDValue SrcVal = Node->getOperand(2);
    if (RegisterSDNode *R = dyn_cast<RegisterSDNode>(SrcVal))
      SrcReg = R->getReg();
    else
      SrcReg = getVR(SrcVal, VRBaseMap);
      
    unsigned DestReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
    if (SrcReg == DestReg) // Coalesced away the copy? Ignore.
      break;
      
    const TargetRegisterClass *SrcTRC = 0, *DstTRC = 0;
    // Get the register classes of the src/dst.
    if (TargetRegisterInfo::isVirtualRegister(SrcReg))
      SrcTRC = MRI.getRegClass(SrcReg);
    else
      SrcTRC = TRI->getPhysicalRegisterRegClass(SrcReg,SrcVal.getValueType());

    if (TargetRegisterInfo::isVirtualRegister(DestReg))
      DstTRC = MRI.getRegClass(DestReg);
    else
      DstTRC = TRI->getPhysicalRegisterRegClass(DestReg,
                                            Node->getOperand(1).getValueType());
    TII->copyRegToReg(*BB, BB->end(), DestReg, SrcReg, DstTRC, SrcTRC);
    break;
  }
  case ISD::CopyFromReg: {
    unsigned SrcReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
    EmitCopyFromReg(Node, 0, IsClone, SrcReg, VRBaseMap);
    break;
  }
  case ISD::INLINEASM: {
    unsigned NumOps = Node->getNumOperands();
    if (Node->getOperand(NumOps-1).getValueType() == MVT::Flag)
      --NumOps;  // Ignore the flag operand.
      
    // Create the inline asm machine instruction.
    MachineInstr *MI = BuildMI(*MF, TII->get(TargetInstrInfo::INLINEASM));

    // Add the asm string as an external symbol operand.
    const char *AsmStr =
      cast<ExternalSymbolSDNode>(Node->getOperand(1))->getSymbol();
    MI->addOperand(MachineOperand::CreateES(AsmStr));
      
    // Add all of the operand registers to the instruction.
    for (unsigned i = 2; i != NumOps;) {
      unsigned Flags = cast<ConstantSDNode>(Node->getOperand(i))->getValue();
      unsigned NumVals = Flags >> 3;
        
      MI->addOperand(MachineOperand::CreateImm(Flags));
      ++i;  // Skip the ID value.
        
      switch (Flags & 7) {
      default: assert(0 && "Bad flags!");
      case 2:   // Def of register.
        for (; NumVals; --NumVals, ++i) {
          unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
          MI->addOperand(MachineOperand::CreateReg(Reg, true));
        }
        break;
      case 1:  // Use of register.
      case 3:  // Immediate.
      case 4:  // Addressing mode.
        // The addressing mode has been selected, just add all of the
        // operands to the machine instruction.
        for (; NumVals; --NumVals, ++i)
          AddOperand(MI, Node->getOperand(i), 0, 0, VRBaseMap);
        break;
      }
    }
    BB->push_back(MI);
    break;
  }
  }
}

void ScheduleDAG::EmitNoop() {
  TII->insertNoop(*BB, BB->end());
}

void ScheduleDAG::EmitCrossRCCopy(SUnit *SU,
                                  DenseMap<SUnit*, unsigned> &VRBaseMap) {
  for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl) continue;  // ignore chain preds
    if (!I->Dep->Node) {
      // Copy to physical register.
      DenseMap<SUnit*, unsigned>::iterator VRI = VRBaseMap.find(I->Dep);
      assert(VRI != VRBaseMap.end() && "Node emitted out of order - late");
      // Find the destination physical register.
      unsigned Reg = 0;
      for (SUnit::const_succ_iterator II = SU->Succs.begin(),
             EE = SU->Succs.end(); II != EE; ++II) {
        if (I->Reg) {
          Reg = I->Reg;
          break;
        }
      }
      assert(I->Reg && "Unknown physical register!");
      TII->copyRegToReg(*BB, BB->end(), Reg, VRI->second,
                        SU->CopyDstRC, SU->CopySrcRC);
    } else {
      // Copy from physical register.
      assert(I->Reg && "Unknown physical register!");
      unsigned VRBase = MRI.createVirtualRegister(SU->CopyDstRC);
      bool isNew = VRBaseMap.insert(std::make_pair(SU, VRBase)).second;
      isNew = isNew; // Silence compiler warning.
      assert(isNew && "Node emitted out of order - early");
      TII->copyRegToReg(*BB, BB->end(), VRBase, I->Reg,
                        SU->CopyDstRC, SU->CopySrcRC);
    }
    break;
  }
}

/// EmitLiveInCopy - Emit a copy for a live in physical register. If the
/// physical register has only a single copy use, then coalesced the copy
/// if possible.
void ScheduleDAG::EmitLiveInCopy(MachineBasicBlock *MBB,
                                 MachineBasicBlock::iterator &InsertPos,
                                 unsigned VirtReg, unsigned PhysReg,
                                 const TargetRegisterClass *RC,
                                 DenseMap<MachineInstr*, unsigned> &CopyRegMap){
  unsigned NumUses = 0;
  MachineInstr *UseMI = NULL;
  for (MachineRegisterInfo::use_iterator UI = MRI.use_begin(VirtReg),
         UE = MRI.use_end(); UI != UE; ++UI) {
    UseMI = &*UI;
    if (++NumUses > 1)
      break;
  }

  // If the number of uses is not one, or the use is not a move instruction,
  // don't coalesce. Also, only coalesce away a virtual register to virtual
  // register copy.
  bool Coalesced = false;
  unsigned SrcReg, DstReg;
  if (NumUses == 1 &&
      TII->isMoveInstr(*UseMI, SrcReg, DstReg) &&
      TargetRegisterInfo::isVirtualRegister(DstReg)) {
    VirtReg = DstReg;
    Coalesced = true;
  }

  // Now find an ideal location to insert the copy.
  MachineBasicBlock::iterator Pos = InsertPos;
  while (Pos != MBB->begin()) {
    MachineInstr *PrevMI = prior(Pos);
    DenseMap<MachineInstr*, unsigned>::iterator RI = CopyRegMap.find(PrevMI);
    // copyRegToReg might emit multiple instructions to do a copy.
    unsigned CopyDstReg = (RI == CopyRegMap.end()) ? 0 : RI->second;
    if (CopyDstReg && !TRI->regsOverlap(CopyDstReg, PhysReg))
      // This is what the BB looks like right now:
      // r1024 = mov r0
      // ...
      // r1    = mov r1024
      //
      // We want to insert "r1025 = mov r1". Inserting this copy below the
      // move to r1024 makes it impossible for that move to be coalesced.
      //
      // r1025 = mov r1
      // r1024 = mov r0
      // ...
      // r1    = mov 1024
      // r2    = mov 1025
      break; // Woot! Found a good location.
    --Pos;
  }

  TII->copyRegToReg(*MBB, Pos, VirtReg, PhysReg, RC, RC);
  CopyRegMap.insert(std::make_pair(prior(Pos), VirtReg));
  if (Coalesced) {
    if (&*InsertPos == UseMI) ++InsertPos;
    MBB->erase(UseMI);
  }
}

/// EmitLiveInCopies - If this is the first basic block in the function,
/// and if it has live ins that need to be copied into vregs, emit the
/// copies into the top of the block.
void ScheduleDAG::EmitLiveInCopies(MachineBasicBlock *MBB) {
  DenseMap<MachineInstr*, unsigned> CopyRegMap;
  MachineBasicBlock::iterator InsertPos = MBB->begin();
  for (MachineRegisterInfo::livein_iterator LI = MRI.livein_begin(),
         E = MRI.livein_end(); LI != E; ++LI)
    if (LI->second) {
      const TargetRegisterClass *RC = MRI.getRegClass(LI->second);
      EmitLiveInCopy(MBB, InsertPos, LI->second, LI->first, RC, CopyRegMap);
    }
}

/// EmitSchedule - Emit the machine code in scheduled order.
MachineBasicBlock *ScheduleDAG::EmitSchedule() {
  bool isEntryBB = &MF->front() == BB;

  if (isEntryBB && !SchedLiveInCopies) {
    // If this is the first basic block in the function, and if it has live ins
    // that need to be copied into vregs, emit the copies into the top of the
    // block before emitting the code for the block.
    for (MachineRegisterInfo::livein_iterator LI = MRI.livein_begin(),
           E = MRI.livein_end(); LI != E; ++LI)
      if (LI->second) {
        const TargetRegisterClass *RC = MRI.getRegClass(LI->second);
        TII->copyRegToReg(*MF->begin(), MF->begin()->end(), LI->second,
                          LI->first, RC, RC);
      }
  }

  // Finally, emit the code for all of the scheduled instructions.
  DenseMap<SDValue, unsigned> VRBaseMap;
  DenseMap<SUnit*, unsigned> CopyVRBaseMap;
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    SUnit *SU = Sequence[i];
    if (!SU) {
      // Null SUnit* is a noop.
      EmitNoop();
      continue;
    }
    for (unsigned j = 0, ee = SU->FlaggedNodes.size(); j != ee; ++j)
      EmitNode(SU->FlaggedNodes[j], SU->OrigNode != SU, VRBaseMap);
    if (!SU->Node)
      EmitCrossRCCopy(SU, CopyVRBaseMap);
    else
      EmitNode(SU->Node, SU->OrigNode != SU, VRBaseMap);
  }

  if (isEntryBB && SchedLiveInCopies)
    EmitLiveInCopies(MF->begin());

  return BB;
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
