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

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/Type.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

/// BuildSchedUnits - Build SUnits from the selection dag that we are input.
/// This SUnit graph is similar to the SelectionDAG, but represents flagged
/// together nodes with a single SUnit.
void ScheduleDAG::BuildSchedUnits() {
  // Reserve entries in the vector for each of the SUnits we are creating.  This
  // ensure that reallocation of the vector won't happen, so SUnit*'s won't get
  // invalidated.
  SUnits.reserve(std::distance(DAG.allnodes_begin(), DAG.allnodes_end()));
  
  const InstrItineraryData &InstrItins = TM.getInstrItineraryData();
  
  for (SelectionDAG::allnodes_iterator NI = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); NI != E; ++NI) {
    if (isPassiveNode(NI))  // Leaf node, e.g. a TargetImmediate.
      continue;
    
    // If this node has already been processed, stop now.
    if (SUnitMap[NI]) continue;
    
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
        SUnitMap[N] = NodeSUnit;
      } while (N->getNumOperands() &&
               N->getOperand(N->getNumOperands()-1).getValueType()== MVT::Flag);
      std::reverse(NodeSUnit->FlaggedNodes.begin(),
                   NodeSUnit->FlaggedNodes.end());
    }
    
    // Scan down, adding this node and any flagged succs to FlaggedNodes if they
    // have a user of the flag operand.
    N = NI;
    while (N->getValueType(N->getNumValues()-1) == MVT::Flag) {
      SDOperand FlagVal(N, N->getNumValues()-1);
      
      // There are either zero or one users of the Flag result.
      bool HasFlagUse = false;
      for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end(); 
           UI != E; ++UI)
        if (FlagVal.isOperand(*UI)) {
          HasFlagUse = true;
          NodeSUnit->FlaggedNodes.push_back(N);
          SUnitMap[N] = NodeSUnit;
          N = *UI;
          break;
        }
      if (!HasFlagUse) break;
    }
    
    // Now all flagged nodes are in FlaggedNodes and N is the bottom-most node.
    // Update the SUnit
    NodeSUnit->Node = N;
    SUnitMap[N] = NodeSUnit;
    
    // Compute the latency for the node.  We use the sum of the latencies for
    // all nodes flagged together into this SUnit.
    if (InstrItins.isEmpty()) {
      // No latency information.
      NodeSUnit->Latency = 1;
    } else {
      NodeSUnit->Latency = 0;
      if (N->isTargetOpcode()) {
        unsigned SchedClass = TII->getSchedClass(N->getTargetOpcode());
        InstrStage *S = InstrItins.begin(SchedClass);
        InstrStage *E = InstrItins.end(SchedClass);
        for (; S != E; ++S)
          NodeSUnit->Latency += S->Cycles;
      }
      for (unsigned i = 0, e = NodeSUnit->FlaggedNodes.size(); i != e; ++i) {
        SDNode *FNode = NodeSUnit->FlaggedNodes[i];
        if (FNode->isTargetOpcode()) {
          unsigned SchedClass = TII->getSchedClass(FNode->getTargetOpcode());
          InstrStage *S = InstrItins.begin(SchedClass);
          InstrStage *E = InstrItins.end(SchedClass);
          for (; S != E; ++S)
            NodeSUnit->Latency += S->Cycles;
        }
      }
    }
  }
  
  // Pass 2: add the preds, succs, etc.
  for (unsigned su = 0, e = SUnits.size(); su != e; ++su) {
    SUnit *SU = &SUnits[su];
    SDNode *MainNode = SU->Node;
    
    if (MainNode->isTargetOpcode()) {
      unsigned Opc = MainNode->getTargetOpcode();
      for (unsigned i = 0, ee = TII->getNumOperands(Opc); i != ee; ++i) {
        if (TII->getOperandConstraint(Opc, i, TOI::TIED_TO) != -1) {
          SU->isTwoAddress = true;
          break;
        }
      }
      if (TII->isCommutableInstr(Opc))
        SU->isCommutable = true;
    }
    
    // Find all predecessors and successors of the group.
    // Temporarily add N to make code simpler.
    SU->FlaggedNodes.push_back(MainNode);
    
    for (unsigned n = 0, e = SU->FlaggedNodes.size(); n != e; ++n) {
      SDNode *N = SU->FlaggedNodes[n];
      
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
        SDNode *OpN = N->getOperand(i).Val;
        if (isPassiveNode(OpN)) continue;   // Not scheduled.
        SUnit *OpSU = SUnitMap[OpN];
        assert(OpSU && "Node has no SUnit!");
        if (OpSU == SU) continue;           // In the same group.

        MVT::ValueType OpVT = N->getOperand(i).getValueType();
        assert(OpVT != MVT::Flag && "Flagged nodes should be in same sunit!");
        bool isChain = OpVT == MVT::Other;
        
        if (SU->addPred(OpSU, isChain)) {
          if (!isChain) {
            SU->NumPreds++;
            SU->NumPredsLeft++;
          } else {
            SU->NumChainPredsLeft++;
          }
        }
        if (OpSU->addSucc(SU, isChain)) {
          if (!isChain) {
            OpSU->NumSuccs++;
            OpSU->NumSuccsLeft++;
          } else {
            OpSU->NumChainSuccsLeft++;
          }
        }
      }
    }
    
    // Remove MainNode from FlaggedNodes again.
    SU->FlaggedNodes.pop_back();
  }
  
  return;
}

void ScheduleDAG::CalculateDepths() {
  std::vector<std::pair<SUnit*, unsigned> > WorkList;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i)
    if (SUnits[i].Preds.size() == 0/* && &SUnits[i] != Entry*/)
      WorkList.push_back(std::make_pair(&SUnits[i], 0U));

  while (!WorkList.empty()) {
    SUnit *SU = WorkList.back().first;
    unsigned Depth = WorkList.back().second;
    WorkList.pop_back();
    if (SU->Depth == 0 || Depth > SU->Depth) {
      SU->Depth = Depth;
      for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
           I != E; ++I)
        WorkList.push_back(std::make_pair(I->first, Depth+1));
    }
  }
}

void ScheduleDAG::CalculateHeights() {
  std::vector<std::pair<SUnit*, unsigned> > WorkList;
  SUnit *Root = SUnitMap[DAG.getRoot().Val];
  WorkList.push_back(std::make_pair(Root, 0U));

  while (!WorkList.empty()) {
    SUnit *SU = WorkList.back().first;
    unsigned Height = WorkList.back().second;
    WorkList.pop_back();
    if (SU->Height == 0 || Height > SU->Height) {
      SU->Height = Height;
      for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
           I != E; ++I)
        WorkList.push_back(std::make_pair(I->first, Height+1));
    }
  }
}

/// CountResults - The results of target nodes have register or immediate
/// operands first, then an optional chain, and optional flag operands (which do
/// not go into the machine instrs.)
unsigned ScheduleDAG::CountResults(SDNode *Node) {
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
unsigned ScheduleDAG::CountOperands(SDNode *Node) {
  unsigned N = Node->getNumOperands();
  while (N && Node->getOperand(N - 1).getValueType() == MVT::Flag)
    --N;
  if (N && Node->getOperand(N - 1).getValueType() == MVT::Other)
    --N; // Ignore chain if it exists.
  return N;
}

static const TargetRegisterClass *getInstrOperandRegClass(
        const MRegisterInfo *MRI, 
        const TargetInstrInfo *TII,
        const TargetInstrDescriptor *II,
        unsigned Op) {
  if (Op >= II->numOperands) {
    assert((II->Flags & M_VARIABLE_OPS)&& "Invalid operand # of instruction");
    return NULL;
  }
  const TargetOperandInfo &toi = II->OpInfo[Op];
  return (toi.Flags & M_LOOK_UP_PTR_REG_CLASS)
         ? TII->getPointerRegClass() : MRI->getRegClass(toi.RegClass);
}

static void CreateVirtualRegisters(SDNode *Node,
                                   unsigned NumResults, 
                                   const MRegisterInfo *MRI,
                                   MachineInstr *MI,
                                   SSARegMap *RegMap,
                                   const TargetInstrInfo *TII,
                                   const TargetInstrDescriptor &II,
                                   DenseMap<SDOperand, unsigned> &VRBaseMap) {
  for (unsigned i = 0; i < NumResults; ++i) {
    // If the specific node value is only used by a CopyToReg and the dest reg
    // is a vreg, use the CopyToReg'd destination register instead of creating
    // a new vreg.
    unsigned VRBase = 0;
    for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
         UI != E; ++UI) {
      SDNode *Use = *UI;
      if (Use->getOpcode() == ISD::CopyToReg && 
          Use->getOperand(2).Val == Node &&
          Use->getOperand(2).ResNo == i) {
        unsigned Reg = cast<RegisterSDNode>(Use->getOperand(1))->getReg();
        if (MRegisterInfo::isVirtualRegister(Reg)) {
          VRBase = Reg;
          MI->addRegOperand(Reg, true);
          break;
        }
      }
    }

    if (VRBase == 0) {
      // Create the result registers for this node and add the result regs to
      // the machine instruction.
      const TargetRegisterClass *RC = getInstrOperandRegClass(MRI, TII, &II, i);
      assert(RC && "Isn't a register operand!");
      VRBase = RegMap->createVirtualRegister(RC);
      MI->addRegOperand(VRBase, true);
    }

    bool isNew = VRBaseMap.insert(std::make_pair(SDOperand(Node,i), VRBase));
    assert(isNew && "Node emitted out of order - early");
  }
}

/// getVR - Return the virtual register corresponding to the specified result
/// of the specified node.
static unsigned getVR(SDOperand Op, DenseMap<SDOperand, unsigned> &VRBaseMap) {
  DenseMap<SDOperand, unsigned>::iterator I = VRBaseMap.find(Op);
  assert(I != VRBaseMap.end() && "Node emitted out of order - late");
  return I->second;
}


/// AddOperand - Add the specified operand to the specified machine instr.  II
/// specifies the instruction information for the node, and IIOpNum is the
/// operand number (in the II) that we are adding. IIOpNum and II are used for 
/// assertions only.
void ScheduleDAG::AddOperand(MachineInstr *MI, SDOperand Op,
                             unsigned IIOpNum,
                             const TargetInstrDescriptor *II,
                             DenseMap<SDOperand, unsigned> &VRBaseMap) {
  if (Op.isTargetOpcode()) {
    // Note that this case is redundant with the final else block, but we
    // include it because it is the most common and it makes the logic
    // simpler here.
    assert(Op.getValueType() != MVT::Other &&
           Op.getValueType() != MVT::Flag &&
           "Chain and flag operands should occur at end of operand list!");
    
    // Get/emit the operand.
    unsigned VReg = getVR(Op, VRBaseMap);
    const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
    bool isOptDef = (IIOpNum < TID->numOperands)
      ? (TID->OpInfo[IIOpNum].Flags & M_OPTIONAL_DEF_OPERAND) : false;
    MI->addRegOperand(VReg, isOptDef);
    
    // Verify that it is right.
    assert(MRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");
    if (II) {
      const TargetRegisterClass *RC =
                          getInstrOperandRegClass(MRI, TII, II, IIOpNum);
      assert(RC && "Don't have operand info for this instruction!");
      const TargetRegisterClass *VRC = RegMap->getRegClass(VReg);
      if (VRC != RC) {
        cerr << "Register class of operand and regclass of use don't agree!\n";
#ifndef NDEBUG
        cerr << "Operand = " << IIOpNum << "\n";
        cerr << "Op->Val = "; Op.Val->dump(&DAG); cerr << "\n";
        cerr << "MI = "; MI->print(cerr);
        cerr << "VReg = " << VReg << "\n";
        cerr << "VReg RegClass     size = " << VRC->getSize()
             << ", align = " << VRC->getAlignment() << "\n";
        cerr << "Expected RegClass size = " << RC->getSize()
             << ", align = " << RC->getAlignment() << "\n";
#endif
        cerr << "Fatal error, aborting.\n";
        abort();
      }
    }
  } else if (ConstantSDNode *C =
             dyn_cast<ConstantSDNode>(Op)) {
    MI->addImmOperand(C->getValue());
  } else if (RegisterSDNode *R =
             dyn_cast<RegisterSDNode>(Op)) {
    MI->addRegOperand(R->getReg(), false);
  } else if (GlobalAddressSDNode *TGA =
             dyn_cast<GlobalAddressSDNode>(Op)) {
    MI->addGlobalAddressOperand(TGA->getGlobal(), TGA->getOffset());
  } else if (BasicBlockSDNode *BB =
             dyn_cast<BasicBlockSDNode>(Op)) {
    MI->addMachineBasicBlockOperand(BB->getBasicBlock());
  } else if (FrameIndexSDNode *FI =
             dyn_cast<FrameIndexSDNode>(Op)) {
    MI->addFrameIndexOperand(FI->getIndex());
  } else if (JumpTableSDNode *JT =
             dyn_cast<JumpTableSDNode>(Op)) {
    MI->addJumpTableIndexOperand(JT->getIndex());
  } else if (ConstantPoolSDNode *CP = 
             dyn_cast<ConstantPoolSDNode>(Op)) {
    int Offset = CP->getOffset();
    unsigned Align = CP->getAlignment();
    const Type *Type = CP->getType();
    // MachineConstantPool wants an explicit alignment.
    if (Align == 0) {
      Align = TM.getTargetData()->getPreferredTypeAlignmentShift(Type);
      if (Align == 0) {
        // Alignment of vector types.  FIXME!
        Align = TM.getTargetData()->getTypeSize(Type);
        Align = Log2_64(Align);
      }
    }
    
    unsigned Idx;
    if (CP->isMachineConstantPoolEntry())
      Idx = ConstPool->getConstantPoolIndex(CP->getMachineCPVal(), Align);
    else
      Idx = ConstPool->getConstantPoolIndex(CP->getConstVal(), Align);
    MI->addConstantPoolIndexOperand(Idx, Offset);
  } else if (ExternalSymbolSDNode *ES = 
             dyn_cast<ExternalSymbolSDNode>(Op)) {
    MI->addExternalSymbolOperand(ES->getSymbol());
  } else {
    assert(Op.getValueType() != MVT::Other &&
           Op.getValueType() != MVT::Flag &&
           "Chain and flag operands should occur at end of operand list!");
    unsigned VReg = getVR(Op, VRBaseMap);
    MI->addRegOperand(VReg, false);
    
    // Verify that it is right.
    assert(MRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");
    if (II) {
      const TargetRegisterClass *RC =
                            getInstrOperandRegClass(MRI, TII, II, IIOpNum);
      assert(RC && "Don't have operand info for this instruction!");
      assert(RegMap->getRegClass(VReg) == RC &&
             "Register class of operand and regclass of use don't agree!");
    }
  }
  
}

// Returns the Register Class of a physical register
static const TargetRegisterClass *getPhysicalRegisterRegClass(
        const MRegisterInfo *MRI,
        MVT::ValueType VT,
        unsigned reg) {
  assert(MRegisterInfo::isPhysicalRegister(reg) &&
         "reg must be a physical register");
  // Pick the register class of the right type that contains this physreg.
  for (MRegisterInfo::regclass_iterator I = MRI->regclass_begin(),
         E = MRI->regclass_end(); I != E; ++I)
    if ((*I)->hasType(VT) && (*I)->contains(reg))
      return *I;
  assert(false && "Couldn't find the register class");
  return 0;
}

/// EmitNode - Generate machine code for an node and needed dependencies.
///
void ScheduleDAG::EmitNode(SDNode *Node, 
                           DenseMap<SDOperand, unsigned> &VRBaseMap) {
  // If machine instruction
  if (Node->isTargetOpcode()) {
    unsigned Opc = Node->getTargetOpcode();
    const TargetInstrDescriptor &II = TII->get(Opc);

    unsigned NumResults = CountResults(Node);
    unsigned NodeOperands = CountOperands(Node);
    unsigned NumMIOperands = NodeOperands + NumResults;
#ifndef NDEBUG
    assert((unsigned(II.numOperands) == NumMIOperands ||
            (II.Flags & M_VARIABLE_OPS)) &&
           "#operands for dag node doesn't match .td file!"); 
#endif

    // Create the new machine instruction.
    MachineInstr *MI = new MachineInstr(II);
    
    // Add result register values for things that are defined by this
    // instruction.
    if (NumResults)
      CreateVirtualRegisters(Node, NumResults, MRI, MI, RegMap,
                             TII, II, VRBaseMap);
    
    // Emit all of the actual operands of this instruction, adding them to the
    // instruction as appropriate.
    for (unsigned i = 0; i != NodeOperands; ++i)
      AddOperand(MI, Node->getOperand(i), i+NumResults, &II, VRBaseMap);

    // Commute node if it has been determined to be profitable.
    if (CommuteSet.count(Node)) {
      MachineInstr *NewMI = TII->commuteInstruction(MI);
      if (NewMI == 0)
        DOUT << "Sched: COMMUTING FAILED!\n";
      else {
        DOUT << "Sched: COMMUTED TO: " << *NewMI;
        if (MI != NewMI) {
          delete MI;
          MI = NewMI;
        }
      }
    }

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
#ifndef NDEBUG
      Node->dump(&DAG);
#endif
      assert(0 && "This target-independent node should have been selected!");
    case ISD::EntryToken: // fall thru
    case ISD::TokenFactor:
    case ISD::LABEL:
      break;
    case ISD::CopyToReg: {
      unsigned InReg;
      if (RegisterSDNode *R = dyn_cast<RegisterSDNode>(Node->getOperand(2)))
        InReg = R->getReg();
      else
        InReg = getVR(Node->getOperand(2), VRBaseMap);
      unsigned DestReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
      if (InReg != DestReg)  {// Coalesced away the copy?
        const TargetRegisterClass *TRC = 0;
        // Get the target register class
        if (MRegisterInfo::isVirtualRegister(InReg))
          TRC = RegMap->getRegClass(InReg);
        else
          TRC = getPhysicalRegisterRegClass(MRI,
                                            Node->getOperand(2).getValueType(),
                                            InReg);
        MRI->copyRegToReg(*BB, BB->end(), DestReg, InReg, TRC);
      }
      break;
    }
    case ISD::CopyFromReg: {
      unsigned VRBase = 0;
      unsigned SrcReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
      if (MRegisterInfo::isVirtualRegister(SrcReg)) {
        // Just use the input register directly!
        bool isNew = VRBaseMap.insert(std::make_pair(SDOperand(Node,0),SrcReg));
        assert(isNew && "Node emitted out of order - early");
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
        TRC = getPhysicalRegisterRegClass(MRI, Node->getValueType(0), SrcReg);

        // Create the reg, emit the copy.
        VRBase = RegMap->createVirtualRegister(TRC);
      }
      MRI->copyRegToReg(*BB, BB->end(), VRBase, SrcReg, TRC);

      bool isNew = VRBaseMap.insert(std::make_pair(SDOperand(Node,0), VRBase));
      assert(isNew && "Node emitted out of order - early");
      break;
    }
    case ISD::INLINEASM: {
      unsigned NumOps = Node->getNumOperands();
      if (Node->getOperand(NumOps-1).getValueType() == MVT::Flag)
        --NumOps;  // Ignore the flag operand.
      
      // Create the inline asm machine instruction.
      MachineInstr *MI =
        new MachineInstr(BB, TII->get(TargetInstrInfo::INLINEASM));

      // Add the asm string as an external symbol operand.
      const char *AsmStr =
        cast<ExternalSymbolSDNode>(Node->getOperand(1))->getSymbol();
      MI->addExternalSymbolOperand(AsmStr);
      
      // Add all of the operand registers to the instruction.
      for (unsigned i = 2; i != NumOps;) {
        unsigned Flags = cast<ConstantSDNode>(Node->getOperand(i))->getValue();
        unsigned NumVals = Flags >> 3;
        
        MI->addImmOperand(Flags);
        ++i;  // Skip the ID value.
        
        switch (Flags & 7) {
        default: assert(0 && "Bad flags!");
        case 1:  // Use of register.
          for (; NumVals; --NumVals, ++i) {
            unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
            MI->addRegOperand(Reg, false);
          }
          break;
        case 2:   // Def of register.
          for (; NumVals; --NumVals, ++i) {
            unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
            MI->addRegOperand(Reg, true);
          }
          break;
        case 3: { // Immediate.
          assert(NumVals == 1 && "Unknown immediate value!");
          if (ConstantSDNode *CS=dyn_cast<ConstantSDNode>(Node->getOperand(i))){
            MI->addImmOperand(CS->getValue());
          } else {
            GlobalAddressSDNode *GA = 
              cast<GlobalAddressSDNode>(Node->getOperand(i));
            MI->addGlobalAddressOperand(GA->getGlobal(), GA->getOffset());
          }
          ++i;
          break;
        }
        case 4:  // Addressing mode.
          // The addressing mode has been selected, just add all of the
          // operands to the machine instruction.
          for (; NumVals; --NumVals, ++i)
            AddOperand(MI, Node->getOperand(i), 0, 0, VRBaseMap);
          break;
        }
      }
      break;
    }
    }
  }
}

void ScheduleDAG::EmitNoop() {
  TII->insertNoop(*BB, BB->end());
}

/// EmitSchedule - Emit the machine code in scheduled order.
void ScheduleDAG::EmitSchedule() {
  // If this is the first basic block in the function, and if it has live ins
  // that need to be copied into vregs, emit the copies into the top of the
  // block before emitting the code for the block.
  MachineFunction &MF = DAG.getMachineFunction();
  if (&MF.front() == BB && MF.livein_begin() != MF.livein_end()) {
    for (MachineFunction::livein_iterator LI = MF.livein_begin(),
         E = MF.livein_end(); LI != E; ++LI)
      if (LI->second)
        MRI->copyRegToReg(*MF.begin(), MF.begin()->end(), LI->second,
                          LI->first, RegMap->getRegClass(LI->second));
  }
  
  
  // Finally, emit the code for all of the scheduled instructions.
  DenseMap<SDOperand, unsigned> VRBaseMap;
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    if (SUnit *SU = Sequence[i]) {
      for (unsigned j = 0, ee = SU->FlaggedNodes.size(); j != ee; j++)
        EmitNode(SU->FlaggedNodes[j], VRBaseMap);
      EmitNode(SU->Node, VRBaseMap);
    } else {
      // Null SUnit* is a noop.
      EmitNoop();
    }
  }
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
MachineBasicBlock *ScheduleDAG::Run() {
  TII = TM.getInstrInfo();
  MRI = TM.getRegisterInfo();
  RegMap = BB->getParent()->getSSARegMap();
  ConstPool = BB->getParent()->getConstantPool();

  Schedule();
  return BB;
}

/// SUnit - Scheduling unit. It's an wrapper around either a single SDNode or
/// a group of nodes flagged together.
void SUnit::dump(const SelectionDAG *G) const {
  cerr << "SU(" << NodeNum << "): ";
  Node->dump(G);
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
  cerr << "  # chain preds left : " << NumChainPredsLeft << "\n";
  cerr << "  # chain succs left : " << NumChainSuccsLeft << "\n";
  cerr << "  Latency            : " << Latency << "\n";
  cerr << "  Depth              : " << Depth << "\n";
  cerr << "  Height             : " << Height << "\n";

  if (Preds.size() != 0) {
    cerr << "  Predecessors:\n";
    for (SUnit::const_succ_iterator I = Preds.begin(), E = Preds.end();
         I != E; ++I) {
      if (I->second)
        cerr << "   ch  #";
      else
        cerr << "   val #";
      cerr << I->first << " - SU(" << I->first->NodeNum << ")\n";
    }
  }
  if (Succs.size() != 0) {
    cerr << "  Successors:\n";
    for (SUnit::const_succ_iterator I = Succs.begin(), E = Succs.end();
         I != E; ++I) {
      if (I->second)
        cerr << "   ch  #";
      else
        cerr << "   val #";
      cerr << I->first << " - SU(" << I->first->NodeNum << ")\n";
    }
  }
  cerr << "\n";
}
