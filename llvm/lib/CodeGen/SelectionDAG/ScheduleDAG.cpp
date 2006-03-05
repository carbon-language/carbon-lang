//===---- ScheduleDAG.cpp - Implement the ScheduleDAG class ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
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
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetInstrItineraries.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/Debug.h"
#include "llvm/Constant.h"
#include <iostream>
using namespace llvm;


/// CountResults - The results of target nodes have register or immediate
/// operands first, then an optional chain, and optional flag operands (which do
/// not go into the machine instrs.)
static unsigned CountResults(SDNode *Node) {
  unsigned N = Node->getNumValues();
  while (N && Node->getValueType(N - 1) == MVT::Flag)
    --N;
  if (N && Node->getValueType(N - 1) == MVT::Other)
    --N;    // Skip over chain result.
  return N;
}

/// CountOperands  The inputs to target nodes have any actual inputs first,
/// followed by an optional chain operand, then flag operands.  Compute the
/// number of actual operands that  will go into the machine instr.
static unsigned CountOperands(SDNode *Node) {
  unsigned N = Node->getNumOperands();
  while (N && Node->getOperand(N - 1).getValueType() == MVT::Flag)
    --N;
  if (N && Node->getOperand(N - 1).getValueType() == MVT::Other)
    --N; // Ignore chain if it exists.
  return N;
}

/// PrepareNodeInfo - Set up the basic minimum node info for scheduling.
/// 
void ScheduleDAG::PrepareNodeInfo() {
  // Allocate node information
  Info = new NodeInfo[NodeCount];

  unsigned i = 0;
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I, ++i) {
    // Fast reference to node schedule info
    NodeInfo* NI = &Info[i];
    // Set up map
    Map[I] = NI;
    // Set node
    NI->Node = I;
    // Set pending visit count
    NI->setPending(I->use_size());
  }
}

/// IdentifyGroups - Put flagged nodes into groups.
///
void ScheduleDAG::IdentifyGroups() {
  for (unsigned i = 0, N = NodeCount; i < N; i++) {
    NodeInfo* NI = &Info[i];
    SDNode *Node = NI->Node;

    // For each operand (in reverse to only look at flags)
    for (unsigned N = Node->getNumOperands(); 0 < N--;) {
      // Get operand
      SDOperand Op = Node->getOperand(N);
      // No more flags to walk
      if (Op.getValueType() != MVT::Flag) break;
      // Add to node group
      AddToGroup(getNI(Op.Val), NI);
      // Let everyone else know
      HasGroups = true;
    }
  }
}

static unsigned CreateVirtualRegisters(MachineInstr *MI,
                                       unsigned NumResults,
                                       SSARegMap *RegMap,
                                       const TargetInstrDescriptor &II) {
  // Create the result registers for this node and add the result regs to
  // the machine instruction.
  const TargetOperandInfo *OpInfo = II.OpInfo;
  unsigned ResultReg = RegMap->createVirtualRegister(OpInfo[0].RegClass);
  MI->addRegOperand(ResultReg, MachineOperand::Def);
  for (unsigned i = 1; i != NumResults; ++i) {
    assert(OpInfo[i].RegClass && "Isn't a register operand!");
    MI->addRegOperand(RegMap->createVirtualRegister(OpInfo[i].RegClass),
                      MachineOperand::Def);
  }
  return ResultReg;
}

/// AddOperand - Add the specified operand to the specified machine instr.  II
/// specifies the instruction information for the node, and IIOpNum is the
/// operand number (in the II) that we are adding. IIOpNum and II are used for 
/// assertions only.
void ScheduleDAG::AddOperand(MachineInstr *MI, SDOperand Op,
                             unsigned IIOpNum,
                             const TargetInstrDescriptor *II) {
  if (Op.isTargetOpcode()) {
    // Note that this case is redundant with the final else block, but we
    // include it because it is the most common and it makes the logic
    // simpler here.
    assert(Op.getValueType() != MVT::Other &&
           Op.getValueType() != MVT::Flag &&
           "Chain and flag operands should occur at end of operand list!");
    
    // Get/emit the operand.
    unsigned VReg = getVR(Op);
    MI->addRegOperand(VReg, MachineOperand::Use);
    
    // Verify that it is right.
    assert(MRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");
    if (II) {
      assert(II->OpInfo[IIOpNum].RegClass &&
             "Don't have operand info for this instruction!");
      assert(RegMap->getRegClass(VReg) == II->OpInfo[IIOpNum].RegClass &&
             "Register class of operand and regclass of use don't agree!");
    }
  } else if (ConstantSDNode *C =
             dyn_cast<ConstantSDNode>(Op)) {
    MI->addZeroExtImm64Operand(C->getValue());
  } else if (RegisterSDNode*R =
             dyn_cast<RegisterSDNode>(Op)) {
    MI->addRegOperand(R->getReg(), MachineOperand::Use);
  } else if (GlobalAddressSDNode *TGA =
             dyn_cast<GlobalAddressSDNode>(Op)) {
    MI->addGlobalAddressOperand(TGA->getGlobal(), false, TGA->getOffset());
  } else if (BasicBlockSDNode *BB =
             dyn_cast<BasicBlockSDNode>(Op)) {
    MI->addMachineBasicBlockOperand(BB->getBasicBlock());
  } else if (FrameIndexSDNode *FI =
             dyn_cast<FrameIndexSDNode>(Op)) {
    MI->addFrameIndexOperand(FI->getIndex());
  } else if (ConstantPoolSDNode *CP = 
             dyn_cast<ConstantPoolSDNode>(Op)) {
    int Offset = CP->getOffset();
    unsigned Align = CP->getAlignment();
    // MachineConstantPool wants an explicit alignment.
    if (Align == 0) {
      if (CP->get()->getType() == Type::DoubleTy)
        Align = 3;  // always 8-byte align doubles.
      else
        Align = TM.getTargetData()
          .getTypeAlignmentShift(CP->get()->getType());
    }
    
    unsigned Idx = ConstPool->getConstantPoolIndex(CP->get(), Align);
    MI->addConstantPoolIndexOperand(Idx, Offset);
  } else if (ExternalSymbolSDNode *ES = 
             dyn_cast<ExternalSymbolSDNode>(Op)) {
    MI->addExternalSymbolOperand(ES->getSymbol(), false);
  } else {
    assert(Op.getValueType() != MVT::Other &&
           Op.getValueType() != MVT::Flag &&
           "Chain and flag operands should occur at end of operand list!");
    unsigned VReg = getVR(Op);
    MI->addRegOperand(VReg, MachineOperand::Use);
    
    // Verify that it is right.
    assert(MRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");
    if (II) {
      assert(II->OpInfo[IIOpNum].RegClass &&
             "Don't have operand info for this instruction!");
      assert(RegMap->getRegClass(VReg) == II->OpInfo[IIOpNum].RegClass &&
             "Register class of operand and regclass of use don't agree!");
    }
  }
  
}


/// EmitNode - Generate machine code for an node and needed dependencies.
///
void ScheduleDAG::EmitNode(NodeInfo *NI) {
  unsigned VRBase = 0;                 // First virtual register for node
  SDNode *Node = NI->Node;
  
  // If machine instruction
  if (Node->isTargetOpcode()) {
    unsigned Opc = Node->getTargetOpcode();
    const TargetInstrDescriptor &II = TII->get(Opc);

    unsigned NumResults = CountResults(Node);
    unsigned NodeOperands = CountOperands(Node);
    unsigned NumMIOperands = NodeOperands + NumResults;
#ifndef NDEBUG
    assert((unsigned(II.numOperands) == NumMIOperands || II.numOperands == -1)&&
           "#operands for dag node doesn't match .td file!"); 
#endif

    // Create the new machine instruction.
    MachineInstr *MI = new MachineInstr(Opc, NumMIOperands, true, true);
    
    // Add result register values for things that are defined by this
    // instruction.
    
    // If the node is only used by a CopyToReg and the dest reg is a vreg, use
    // the CopyToReg'd destination register instead of creating a new vreg.
    if (NumResults == 1) {
      for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
           UI != E; ++UI) {
        SDNode *Use = *UI;
        if (Use->getOpcode() == ISD::CopyToReg && 
            Use->getOperand(2).Val == Node) {
          unsigned Reg = cast<RegisterSDNode>(Use->getOperand(1))->getReg();
          if (MRegisterInfo::isVirtualRegister(Reg)) {
            VRBase = Reg;
            MI->addRegOperand(Reg, MachineOperand::Def);
            break;
          }
        }
      }
    }
    
    // Otherwise, create new virtual registers.
    if (NumResults && VRBase == 0)
      VRBase = CreateVirtualRegisters(MI, NumResults, RegMap, II);
    
    // Emit all of the actual operands of this instruction, adding them to the
    // instruction as appropriate.
    for (unsigned i = 0; i != NodeOperands; ++i)
      AddOperand(MI, Node->getOperand(i), i+NumResults, &II);
    
    // Now that we have emitted all operands, emit this instruction itself.
    if ((II.Flags & M_USES_CUSTOM_DAG_SCHED_INSERTION) == 0) {
      BB->insert(BB->end(), MI);
    } else {
      // Insert this instruction into the end of the basic block, potentially
      // taking some custom action.
      BB = DAG.getTargetLoweringInfo().InsertAtEndOfBasicBlock(MI, BB);
    }
  } else {
    switch (Node->getOpcode()) {
    default:
      Node->dump(); 
      assert(0 && "This target-independent node should have been selected!");
    case ISD::EntryToken: // fall thru
    case ISD::TokenFactor:
      break;
    case ISD::CopyToReg: {
      unsigned InReg = getVR(Node->getOperand(2));
      unsigned DestReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
      if (InReg != DestReg)   // Coallesced away the copy?
        MRI->copyRegToReg(*BB, BB->end(), DestReg, InReg,
                          RegMap->getRegClass(InReg));
      break;
    }
    case ISD::CopyFromReg: {
      unsigned SrcReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
      if (MRegisterInfo::isVirtualRegister(SrcReg)) {
        VRBase = SrcReg;  // Just use the input register directly!
        break;
      }

      // If the node is only used by a CopyToReg and the dest reg is a vreg, use
      // the CopyToReg'd destination register instead of creating a new vreg.
      for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
           UI != E; ++UI) {
        SDNode *Use = *UI;
        if (Use->getOpcode() == ISD::CopyToReg && 
            Use->getOperand(2).Val == Node) {
          unsigned DestReg = cast<RegisterSDNode>(Use->getOperand(1))->getReg();
          if (MRegisterInfo::isVirtualRegister(DestReg)) {
            VRBase = DestReg;
            break;
          }
        }
      }

      // Figure out the register class to create for the destreg.
      const TargetRegisterClass *TRC = 0;
      if (VRBase) {
        TRC = RegMap->getRegClass(VRBase);
      } else {

        // Pick the register class of the right type that contains this physreg.
        for (MRegisterInfo::regclass_iterator I = MRI->regclass_begin(),
             E = MRI->regclass_end(); I != E; ++I)
          if ((*I)->hasType(Node->getValueType(0)) &&
              (*I)->contains(SrcReg)) {
            TRC = *I;
            break;
          }
        assert(TRC && "Couldn't find register class for reg copy!");
      
        // Create the reg, emit the copy.
        VRBase = RegMap->createVirtualRegister(TRC);
      }
      MRI->copyRegToReg(*BB, BB->end(), VRBase, SrcReg, TRC);
      break;
    }
    case ISD::INLINEASM: {
      unsigned NumOps = Node->getNumOperands();
      if (Node->getOperand(NumOps-1).getValueType() == MVT::Flag)
        --NumOps;  // Ignore the flag operand.
      
      // Create the inline asm machine instruction.
      MachineInstr *MI =
        new MachineInstr(BB, TargetInstrInfo::INLINEASM, (NumOps-2)/2+1);

      // Add the asm string as an external symbol operand.
      const char *AsmStr =
        cast<ExternalSymbolSDNode>(Node->getOperand(1))->getSymbol();
      MI->addExternalSymbolOperand(AsmStr, false);
      
      // Add all of the operand registers to the instruction.
      for (unsigned i = 2; i != NumOps;) {
        unsigned Flags = cast<ConstantSDNode>(Node->getOperand(i))->getValue();
        unsigned NumVals = Flags >> 3;
        
        MI->addZeroExtImm64Operand(Flags);
        ++i;  // Skip the ID value.
        
        switch (Flags & 7) {
        default: assert(0 && "Bad flags!");
        case 1:  // Use of register.
          for (; NumVals; --NumVals, ++i) {
            unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
            MI->addMachineRegOperand(Reg, MachineOperand::Use);
          }
          break;
        case 2:   // Def of register.
          for (; NumVals; --NumVals, ++i) {
            unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
            MI->addMachineRegOperand(Reg, MachineOperand::Def);
          }
          break;
        case 3: { // Immediate.
          assert(NumVals == 1 && "Unknown immediate value!");
          uint64_t Val = cast<ConstantSDNode>(Node->getOperand(i))->getValue();
          MI->addZeroExtImm64Operand(Val);
          ++i;
          break;
        }
        case 4:  // Addressing mode.
          // The addressing mode has been selected, just add all of the
          // operands to the machine instruction.
          for (; NumVals; --NumVals, ++i)
            AddOperand(MI, Node->getOperand(i), 0, 0);
          break;
        }
      }
      break;
    }
    }
  }

  assert(NI->VRBase == 0 && "Node emitted out of order - early");
  NI->VRBase = VRBase;
}

void ScheduleDAG::EmitNoop() {
  TII->insertNoop(*BB, BB->end());
}

/// EmitAll - Emit all nodes in schedule sorted order.
///
void ScheduleDAG::EmitAll() {
  // For each node in the ordering
  for (unsigned i = 0, N = Ordering.size(); i < N; i++) {
    // Get the scheduling info
    NodeInfo *NI = Ordering[i];
    if (NI->isInGroup()) {
      NodeGroupIterator NGI(Ordering[i]);
      while (NodeInfo *NI = NGI.next()) EmitNode(NI);
    } else {
      EmitNode(NI);
    }
  }
}

/// isFlagDefiner - Returns true if the node defines a flag result.
static bool isFlagDefiner(SDNode *A) {
  unsigned N = A->getNumValues();
  return N && A->getValueType(N - 1) == MVT::Flag;
}

/// isFlagUser - Returns true if the node uses a flag result.
///
static bool isFlagUser(SDNode *A) {
  unsigned N = A->getNumOperands();
  return N && A->getOperand(N - 1).getValueType() == MVT::Flag;
}

/// printNI - Print node info.
///
void ScheduleDAG::printNI(std::ostream &O, NodeInfo *NI) const {
#ifndef NDEBUG
  SDNode *Node = NI->Node;
  O << " "
    << std::hex << Node << std::dec
    << ", Lat=" << NI->Latency
    << ", Slot=" << NI->Slot
    << ", ARITY=(" << Node->getNumOperands() << ","
                   << Node->getNumValues() << ")"
    << " " << Node->getOperationName(&DAG);
  if (isFlagDefiner(Node)) O << "<#";
  if (isFlagUser(Node)) O << ">#";
#endif
}

/// printChanges - Hilight changes in order caused by scheduling.
///
void ScheduleDAG::printChanges(unsigned Index) const {
#ifndef NDEBUG
  // Get the ordered node count
  unsigned N = Ordering.size();
  // Determine if any changes
  unsigned i = 0;
  for (; i < N; i++) {
    NodeInfo *NI = Ordering[i];
    if (NI->Preorder != i) break;
  }
  
  if (i < N) {
    std::cerr << Index << ". New Ordering\n";
    
    for (i = 0; i < N; i++) {
      NodeInfo *NI = Ordering[i];
      std::cerr << "  " << NI->Preorder << ". ";
      printNI(std::cerr, NI);
      std::cerr << "\n";
      if (NI->isGroupDominator()) {
        NodeGroup *Group = NI->Group;
        for (NIIterator NII = Group->group_begin(), E = Group->group_end();
             NII != E; NII++) {
          std::cerr << "          ";
          printNI(std::cerr, *NII);
          std::cerr << "\n";
        }
      }
    }
  } else {
    std::cerr << Index << ". No Changes\n";
  }
#endif
}

/// print - Print ordering to specified output stream.
///
void ScheduleDAG::print(std::ostream &O) const {
#ifndef NDEBUG
  using namespace std;
  O << "Ordering\n";
  for (unsigned i = 0, N = Ordering.size(); i < N; i++) {
    NodeInfo *NI = Ordering[i];
    printNI(O, NI);
    O << "\n";
    if (NI->isGroupDominator()) {
      NodeGroup *Group = NI->Group;
      for (NIIterator NII = Group->group_begin(), E = Group->group_end();
           NII != E; NII++) {
        O << "    ";
        printNI(O, *NII);
        O << "\n";
      }
    }
  }
#endif
}

void ScheduleDAG::dump(const char *tag) const {
  std::cerr << tag; dump();
}

void ScheduleDAG::dump() const {
  print(std::cerr);
}

/// Run - perform scheduling.
///
MachineBasicBlock *ScheduleDAG::Run() {
  TII = TM.getInstrInfo();
  MRI = TM.getRegisterInfo();
  RegMap = BB->getParent()->getSSARegMap();
  ConstPool = BB->getParent()->getConstantPool();

  // Number the nodes
  NodeCount = std::distance(DAG.allnodes_begin(), DAG.allnodes_end());
  // Set up minimum info for scheduling
  PrepareNodeInfo();
  // Construct node groups for flagged nodes
  IdentifyGroups();

  Schedule();
  return BB;
}


/// CountInternalUses - Returns the number of edges between the two nodes.
///
static unsigned CountInternalUses(NodeInfo *D, NodeInfo *U) {
  unsigned N = 0;
  for (unsigned M = U->Node->getNumOperands(); 0 < M--;) {
    SDOperand Op = U->Node->getOperand(M);
    if (Op.Val == D->Node) N++;
  }

  return N;
}

//===----------------------------------------------------------------------===//
/// Add - Adds a definer and user pair to a node group.
///
void ScheduleDAG::AddToGroup(NodeInfo *D, NodeInfo *U) {
  // Get current groups
  NodeGroup *DGroup = D->Group;
  NodeGroup *UGroup = U->Group;
  // If both are members of groups
  if (DGroup && UGroup) {
    // There may have been another edge connecting 
    if (DGroup == UGroup) return;
    // Add the pending users count
    DGroup->addPending(UGroup->getPending());
    // For each member of the users group
    NodeGroupIterator UNGI(U);
    while (NodeInfo *UNI = UNGI.next() ) {
      // Change the group
      UNI->Group = DGroup;
      // For each member of the definers group
      NodeGroupIterator DNGI(D);
      while (NodeInfo *DNI = DNGI.next() ) {
        // Remove internal edges
        DGroup->addPending(-CountInternalUses(DNI, UNI));
      }
    }
    // Merge the two lists
    DGroup->group_insert(DGroup->group_end(),
                         UGroup->group_begin(), UGroup->group_end());
  } else if (DGroup) {
    // Make user member of definers group
    U->Group = DGroup;
    // Add users uses to definers group pending
    DGroup->addPending(U->Node->use_size());
    // For each member of the definers group
    NodeGroupIterator DNGI(D);
    while (NodeInfo *DNI = DNGI.next() ) {
      // Remove internal edges
      DGroup->addPending(-CountInternalUses(DNI, U));
    }
    DGroup->group_push_back(U);
  } else if (UGroup) {
    // Make definer member of users group
    D->Group = UGroup;
    // Add definers uses to users group pending
    UGroup->addPending(D->Node->use_size());
    // For each member of the users group
    NodeGroupIterator UNGI(U);
    while (NodeInfo *UNI = UNGI.next() ) {
      // Remove internal edges
      UGroup->addPending(-CountInternalUses(D, UNI));
    }
    UGroup->group_insert(UGroup->group_begin(), D);
  } else {
    D->Group = U->Group = DGroup = new NodeGroup();
    DGroup->addPending(D->Node->use_size() + U->Node->use_size() -
                       CountInternalUses(D, U));
    DGroup->group_push_back(D);
    DGroup->group_push_back(U);

    if (HeadNG == NULL)
      HeadNG = DGroup;
    if (TailNG != NULL)
      TailNG->Next = DGroup;
    TailNG = DGroup;
  }
}
