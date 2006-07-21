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
#include <iostream>
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
    while (N->getNumOperands() &&
           N->getOperand(N->getNumOperands()-1).getValueType() == MVT::Flag) {
      N = N->getOperand(N->getNumOperands()-1).Val;
      NodeSUnit->FlaggedNodes.push_back(N);
      SUnitMap[N] = NodeSUnit;
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
      if (TII->isTwoAddrInstr(Opc))
        SU->isTwoAddress = true;
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
        
        if (SU->Preds.insert(std::make_pair(OpSU, isChain)).second) {
          if (!isChain) {
            SU->NumPreds++;
            SU->NumPredsLeft++;
          } else {
            SU->NumChainPredsLeft++;
          }
        }
        if (OpSU->Succs.insert(std::make_pair(SU, isChain)).second) {
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

static void CalculateDepths(SUnit *SU, unsigned Depth) {
  if (SU->Depth == 0 || Depth > SU->Depth) {
    SU->Depth = Depth;
    for (std::set<std::pair<SUnit*, bool> >::iterator I = SU->Succs.begin(),
           E = SU->Succs.end(); I != E; ++I)
      CalculateDepths(I->first, Depth+1);
  }
}

void ScheduleDAG::CalculateDepths() {
  SUnit *Entry = SUnitMap[DAG.getEntryNode().Val];
  ::CalculateDepths(Entry, 0U);
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i)
    if (SUnits[i].Preds.size() == 0 && &SUnits[i] != Entry) {
      ::CalculateDepths(&SUnits[i], 0U);
    }
}

static void CalculateHeights(SUnit *SU, unsigned Height) {
  if (SU->Height == 0 || Height > SU->Height) {
    SU->Height = Height;
    for (std::set<std::pair<SUnit*, bool> >::iterator I = SU->Preds.begin(),
           E = SU->Preds.end(); I != E; ++I)
      CalculateHeights(I->first, Height+1);
  }
}
void ScheduleDAG::CalculateHeights() {
  SUnit *Root = SUnitMap[DAG.getRoot().Val];
  ::CalculateHeights(Root, 0U);
}

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

static unsigned CreateVirtualRegisters(const MRegisterInfo *MRI,
                                       MachineInstr *MI,
                                       unsigned NumResults,
                                       SSARegMap *RegMap,
                                       const TargetInstrInfo *TII,
                                       const TargetInstrDescriptor &II) {
  // Create the result registers for this node and add the result regs to
  // the machine instruction.
  unsigned ResultReg =
    RegMap->createVirtualRegister(getInstrOperandRegClass(MRI, TII, &II, 0));
  MI->addRegOperand(ResultReg, MachineOperand::Def);
  for (unsigned i = 1; i != NumResults; ++i) {
    const TargetRegisterClass *RC = getInstrOperandRegClass(MRI, TII, &II, i);
    assert(RC && "Isn't a register operand!");
    MI->addRegOperand(RegMap->createVirtualRegister(RC), MachineOperand::Def);
  }
  return ResultReg;
}

/// getVR - Return the virtual register corresponding to the specified result
/// of the specified node.
static unsigned getVR(SDOperand Op, std::map<SDNode*, unsigned> &VRBaseMap) {
  std::map<SDNode*, unsigned>::iterator I = VRBaseMap.find(Op.Val);
  assert(I != VRBaseMap.end() && "Node emitted out of order - late");
  return I->second + Op.ResNo;
}


/// AddOperand - Add the specified operand to the specified machine instr.  II
/// specifies the instruction information for the node, and IIOpNum is the
/// operand number (in the II) that we are adding. IIOpNum and II are used for 
/// assertions only.
void ScheduleDAG::AddOperand(MachineInstr *MI, SDOperand Op,
                             unsigned IIOpNum,
                             const TargetInstrDescriptor *II,
                             std::map<SDNode*, unsigned> &VRBaseMap) {
  if (Op.isTargetOpcode()) {
    // Note that this case is redundant with the final else block, but we
    // include it because it is the most common and it makes the logic
    // simpler here.
    assert(Op.getValueType() != MVT::Other &&
           Op.getValueType() != MVT::Flag &&
           "Chain and flag operands should occur at end of operand list!");
    
    // Get/emit the operand.
    unsigned VReg = getVR(Op, VRBaseMap);
    MI->addRegOperand(VReg, MachineOperand::Use);
    
    // Verify that it is right.
    assert(MRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");
    if (II) {
      const TargetRegisterClass *RC =
                          getInstrOperandRegClass(MRI, TII, II, IIOpNum);
      assert(RC && "Don't have operand info for this instruction!");
      assert(RegMap->getRegClass(VReg) == RC &&
             "Register class of operand and regclass of use don't agree!");
    }
  } else if (ConstantSDNode *C =
             dyn_cast<ConstantSDNode>(Op)) {
    MI->addImmOperand(C->getValue());
  } else if (RegisterSDNode*R =
             dyn_cast<RegisterSDNode>(Op)) {
    MI->addRegOperand(R->getReg(), MachineOperand::Use);
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
    // MachineConstantPool wants an explicit alignment.
    if (Align == 0) {
      if (CP->get()->getType() == Type::DoubleTy)
        Align = 3;  // always 8-byte align doubles.
      else {
        Align = TM.getTargetData()
          ->getTypeAlignmentShift(CP->get()->getType());
        if (Align == 0) {
          // Alignment of packed types.  FIXME!
          Align = TM.getTargetData()->getTypeSize(CP->get()->getType());
          Align = Log2_64(Align);
        }
      }
    }
    
    unsigned Idx = ConstPool->getConstantPoolIndex(CP->get(), Align);
    MI->addConstantPoolIndexOperand(Idx, Offset);
  } else if (ExternalSymbolSDNode *ES = 
             dyn_cast<ExternalSymbolSDNode>(Op)) {
    MI->addExternalSymbolOperand(ES->getSymbol());
  } else {
    assert(Op.getValueType() != MVT::Other &&
           Op.getValueType() != MVT::Flag &&
           "Chain and flag operands should occur at end of operand list!");
    unsigned VReg = getVR(Op, VRBaseMap);
    MI->addRegOperand(VReg, MachineOperand::Use);
    
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


/// EmitNode - Generate machine code for an node and needed dependencies.
///
void ScheduleDAG::EmitNode(SDNode *Node, 
                           std::map<SDNode*, unsigned> &VRBaseMap) {
  unsigned VRBase = 0;                 // First virtual register for node
  
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
    MachineInstr *MI = new MachineInstr(Opc, NumMIOperands);
    
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
      VRBase = CreateVirtualRegisters(MRI, MI, NumResults, RegMap, TII, II);
    
    // Emit all of the actual operands of this instruction, adding them to the
    // instruction as appropriate.
    for (unsigned i = 0; i != NodeOperands; ++i)
      AddOperand(MI, Node->getOperand(i), i+NumResults, &II, VRBaseMap);

    // Commute node if it has been determined to be profitable.
    if (CommuteSet.count(Node)) {
      MachineInstr *NewMI = TII->commuteInstruction(MI);
      if (NewMI == 0)
        DEBUG(std::cerr << "Sched: COMMUTING FAILED!\n");
      else {
        DEBUG(std::cerr << "Sched: COMMUTED TO: " << *NewMI);
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
      Node->dump();
#endif
      assert(0 && "This target-independent node should have been selected!");
    case ISD::EntryToken: // fall thru
    case ISD::TokenFactor:
      break;
    case ISD::CopyToReg: {
      unsigned InReg = getVR(Node->getOperand(2), VRBaseMap);
      unsigned DestReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
      if (InReg != DestReg)   // Coalesced away the copy?
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
            MI->addRegOperand(Reg, MachineOperand::Use);
          }
          break;
        case 2:   // Def of register.
          for (; NumVals; --NumVals, ++i) {
            unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
            MI->addRegOperand(Reg, MachineOperand::Def);
          }
          break;
        case 3: { // Immediate.
          assert(NumVals == 1 && "Unknown immediate value!");
          uint64_t Val = cast<ConstantSDNode>(Node->getOperand(i))->getValue();
          MI->addImmOperand(Val);
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

  assert(!VRBaseMap.count(Node) && "Node emitted out of order - early");
  VRBaseMap[Node] = VRBase;
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
  std::map<SDNode*, unsigned> VRBaseMap;
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
      std::cerr << "**** NOOP ****\n";
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
  std::cerr << "SU(" << NodeNum << "): ";
  Node->dump(G);
  std::cerr << "\n";
  if (FlaggedNodes.size() != 0) {
    for (unsigned i = 0, e = FlaggedNodes.size(); i != e; i++) {
      std::cerr << "    ";
      FlaggedNodes[i]->dump(G);
      std::cerr << "\n";
    }
  }
}

void SUnit::dumpAll(const SelectionDAG *G) const {
  dump(G);

  std::cerr << "  # preds left       : " << NumPredsLeft << "\n";
  std::cerr << "  # succs left       : " << NumSuccsLeft << "\n";
  std::cerr << "  # chain preds left : " << NumChainPredsLeft << "\n";
  std::cerr << "  # chain succs left : " << NumChainSuccsLeft << "\n";
  std::cerr << "  Latency            : " << Latency << "\n";
  std::cerr << "  Depth              : " << Depth << "\n";
  std::cerr << "  Height             : " << Height << "\n";

  if (Preds.size() != 0) {
    std::cerr << "  Predecessors:\n";
    for (std::set<std::pair<SUnit*,bool> >::const_iterator I = Preds.begin(),
           E = Preds.end(); I != E; ++I) {
      if (I->second)
        std::cerr << "   ch  ";
      else
        std::cerr << "   val ";
      I->first->dump(G);
    }
  }
  if (Succs.size() != 0) {
    std::cerr << "  Successors:\n";
    for (std::set<std::pair<SUnit*, bool> >::const_iterator I = Succs.begin(),
           E = Succs.end(); I != E; ++I) {
      if (I->second)
        std::cerr << "   ch  ";
      else
        std::cerr << "   val ";
      I->first->dump(G);
    }
  }
  std::cerr << "\n";
}
