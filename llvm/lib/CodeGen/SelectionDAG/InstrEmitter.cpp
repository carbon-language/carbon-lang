//==--- InstrEmitter.cpp - Emit MachineInstrs for the SelectionDAG class ---==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the Emit routines for the SelectionDAG class, which creates
// MachineInstrs based on the decisions of the SelectionDAG instruction
// selection.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instr-emitter"
#include "InstrEmitter.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

/// CountResults - The results of target nodes have register or immediate
/// operands first, then an optional chain, and optional flag operands (which do
/// not go into the resulting MachineInstr).
unsigned InstrEmitter::CountResults(SDNode *Node) {
  unsigned N = Node->getNumValues();
  while (N && Node->getValueType(N - 1) == MVT::Flag)
    --N;
  if (N && Node->getValueType(N - 1) == MVT::Other)
    --N;    // Skip over chain result.
  return N;
}

/// CountOperands - The inputs to target nodes have any actual inputs first,
/// followed by an optional chain operand, then an optional flag operand.
/// Compute the number of actual operands that will go into the resulting
/// MachineInstr.
unsigned InstrEmitter::CountOperands(SDNode *Node) {
  unsigned N = Node->getNumOperands();
  while (N && Node->getOperand(N - 1).getValueType() == MVT::Flag)
    --N;
  if (N && Node->getOperand(N - 1).getValueType() == MVT::Other)
    --N; // Ignore chain if it exists.
  return N;
}

/// EmitCopyFromReg - Generate machine code for an CopyFromReg node or an
/// implicit physical register output.
void InstrEmitter::
EmitCopyFromReg(SDNode *Node, unsigned ResNo, bool IsClone, bool IsCloned,
                unsigned SrcReg, DenseMap<SDValue, unsigned> &VRBaseMap) {
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
  const TargetRegisterClass *UseRC = NULL;
  if (!IsClone && !IsCloned)
    for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
         UI != E; ++UI) {
      SDNode *User = *UI;
      bool Match = true;
      if (User->getOpcode() == ISD::CopyToReg && 
          User->getOperand(2).getNode() == Node &&
          User->getOperand(2).getResNo() == ResNo) {
        unsigned DestReg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
        if (TargetRegisterInfo::isVirtualRegister(DestReg)) {
          VRBase = DestReg;
          Match = false;
        } else if (DestReg != SrcReg)
          Match = false;
      } else {
        for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i) {
          SDValue Op = User->getOperand(i);
          if (Op.getNode() != Node || Op.getResNo() != ResNo)
            continue;
          EVT VT = Node->getValueType(Op.getResNo());
          if (VT == MVT::Other || VT == MVT::Flag)
            continue;
          Match = false;
          if (User->isMachineOpcode()) {
            const TargetInstrDesc &II = TII->get(User->getMachineOpcode());
            const TargetRegisterClass *RC = 0;
            if (i+II.getNumDefs() < II.getNumOperands())
              RC = II.OpInfo[i+II.getNumDefs()].getRegClass(TRI);
            if (!UseRC)
              UseRC = RC;
            else if (RC) {
              const TargetRegisterClass *ComRC = getCommonSubClass(UseRC, RC);
              // If multiple uses expect disjoint register classes, we emit
              // copies in AddRegisterOperand.
              if (ComRC)
                UseRC = ComRC;
            }
          }
        }
      }
      MatchReg &= Match;
      if (VRBase)
        break;
    }

  EVT VT = Node->getValueType(ResNo);
  const TargetRegisterClass *SrcRC = 0, *DstRC = 0;
  SrcRC = TRI->getPhysicalRegisterRegClass(SrcReg, VT);
  
  // Figure out the register class to create for the destreg.
  if (VRBase) {
    DstRC = MRI->getRegClass(VRBase);
  } else if (UseRC) {
    assert(UseRC->hasType(VT) && "Incompatible phys register def and uses!");
    DstRC = UseRC;
  } else {
    DstRC = TLI->getRegClassFor(VT);
  }
    
  // If all uses are reading from the src physical register and copying the
  // register is either impossible or very expensive, then don't create a copy.
  if (MatchReg && SrcRC->getCopyCost() < 0) {
    VRBase = SrcReg;
  } else {
    // Create the reg, emit the copy.
    VRBase = MRI->createVirtualRegister(DstRC);
    bool Emitted = TII->copyRegToReg(*MBB, InsertPos, VRBase, SrcReg,
                                     DstRC, SrcRC);

    assert(Emitted && "Unable to issue a copy instruction!\n");
    (void) Emitted;
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
unsigned InstrEmitter::getDstOfOnlyCopyToRegUse(SDNode *Node,
                                                unsigned ResNo) const {
  if (!Node->hasOneUse())
    return 0;

  SDNode *User = *Node->use_begin();
  if (User->getOpcode() == ISD::CopyToReg && 
      User->getOperand(2).getNode() == Node &&
      User->getOperand(2).getResNo() == ResNo) {
    unsigned Reg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      return Reg;
  }
  return 0;
}

void InstrEmitter::CreateVirtualRegisters(SDNode *Node, MachineInstr *MI,
                                       const TargetInstrDesc &II,
                                       bool IsClone, bool IsCloned,
                                       DenseMap<SDValue, unsigned> &VRBaseMap) {
  assert(Node->getMachineOpcode() != TargetInstrInfo::IMPLICIT_DEF &&
         "IMPLICIT_DEF should have been handled as a special case elsewhere!");

  for (unsigned i = 0; i < II.getNumDefs(); ++i) {
    // If the specific node value is only used by a CopyToReg and the dest reg
    // is a vreg in the same register class, use the CopyToReg'd destination
    // register instead of creating a new vreg.
    unsigned VRBase = 0;
    const TargetRegisterClass *RC = II.OpInfo[i].getRegClass(TRI);
    if (II.OpInfo[i].isOptionalDef()) {
      // Optional def must be a physical register.
      unsigned NumResults = CountResults(Node);
      VRBase = cast<RegisterSDNode>(Node->getOperand(i-NumResults))->getReg();
      assert(TargetRegisterInfo::isPhysicalRegister(VRBase));
      MI->addOperand(MachineOperand::CreateReg(VRBase, true));
    }

    if (!VRBase && !IsClone && !IsCloned)
      for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
           UI != E; ++UI) {
        SDNode *User = *UI;
        if (User->getOpcode() == ISD::CopyToReg && 
            User->getOperand(2).getNode() == Node &&
            User->getOperand(2).getResNo() == i) {
          unsigned Reg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
          if (TargetRegisterInfo::isVirtualRegister(Reg)) {
            const TargetRegisterClass *RegRC = MRI->getRegClass(Reg);
            if (RegRC == RC) {
              VRBase = Reg;
              MI->addOperand(MachineOperand::CreateReg(Reg, true));
              break;
            }
          }
        }
      }

    // Create the result registers for this node and add the result regs to
    // the machine instruction.
    if (VRBase == 0) {
      assert(RC && "Isn't a register operand!");
      VRBase = MRI->createVirtualRegister(RC);
      MI->addOperand(MachineOperand::CreateReg(VRBase, true));
    }

    SDValue Op(Node, i);
    if (IsClone)
      VRBaseMap.erase(Op);
    bool isNew = VRBaseMap.insert(std::make_pair(Op, VRBase)).second;
    isNew = isNew; // Silence compiler warning.
    assert(isNew && "Node emitted out of order - early");
  }
}

/// getVR - Return the virtual register corresponding to the specified result
/// of the specified node.
unsigned InstrEmitter::getVR(SDValue Op,
                             DenseMap<SDValue, unsigned> &VRBaseMap) {
  if (Op.isMachineOpcode() &&
      Op.getMachineOpcode() == TargetInstrInfo::IMPLICIT_DEF) {
    // Add an IMPLICIT_DEF instruction before every use.
    unsigned VReg = getDstOfOnlyCopyToRegUse(Op.getNode(), Op.getResNo());
    // IMPLICIT_DEF can produce any type of result so its TargetInstrDesc
    // does not include operand register class info.
    if (!VReg) {
      const TargetRegisterClass *RC = TLI->getRegClassFor(Op.getValueType());
      VReg = MRI->createVirtualRegister(RC);
    }
    BuildMI(MBB, Op.getDebugLoc(),
            TII->get(TargetInstrInfo::IMPLICIT_DEF), VReg);
    return VReg;
  }

  DenseMap<SDValue, unsigned>::iterator I = VRBaseMap.find(Op);
  assert(I != VRBaseMap.end() && "Node emitted out of order - late");
  return I->second;
}


/// AddRegisterOperand - Add the specified register as an operand to the
/// specified machine instr. Insert register copies if the register is
/// not in the required register class.
void
InstrEmitter::AddRegisterOperand(MachineInstr *MI, SDValue Op,
                                 unsigned IIOpNum,
                                 const TargetInstrDesc *II,
                                 DenseMap<SDValue, unsigned> &VRBaseMap) {
  assert(Op.getValueType() != MVT::Other &&
         Op.getValueType() != MVT::Flag &&
         "Chain and flag operands should occur at end of operand list!");
  // Get/emit the operand.
  unsigned VReg = getVR(Op, VRBaseMap);
  assert(TargetRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");

  const TargetInstrDesc &TID = MI->getDesc();
  bool isOptDef = IIOpNum < TID.getNumOperands() &&
    TID.OpInfo[IIOpNum].isOptionalDef();

  // If the instruction requires a register in a different class, create
  // a new virtual register and copy the value into it.
  if (II) {
    const TargetRegisterClass *SrcRC = MRI->getRegClass(VReg);
    const TargetRegisterClass *DstRC = 0;
    if (IIOpNum < II->getNumOperands())
      DstRC = II->OpInfo[IIOpNum].getRegClass(TRI);
    assert((DstRC || (TID.isVariadic() && IIOpNum >= TID.getNumOperands())) &&
           "Don't have operand info for this instruction!");
    if (DstRC && SrcRC != DstRC && !SrcRC->hasSuperClass(DstRC)) {
      unsigned NewVReg = MRI->createVirtualRegister(DstRC);
      bool Emitted = TII->copyRegToReg(*MBB, InsertPos, NewVReg, VReg,
                                       DstRC, SrcRC);
      assert(Emitted && "Unable to issue a copy instruction!\n");
      (void) Emitted;
      VReg = NewVReg;
    }
  }

  MI->addOperand(MachineOperand::CreateReg(VReg, isOptDef));
}

/// AddOperand - Add the specified operand to the specified machine instr.  II
/// specifies the instruction information for the node, and IIOpNum is the
/// operand number (in the II) that we are adding. IIOpNum and II are used for 
/// assertions only.
void InstrEmitter::AddOperand(MachineInstr *MI, SDValue Op,
                              unsigned IIOpNum,
                              const TargetInstrDesc *II,
                              DenseMap<SDValue, unsigned> &VRBaseMap) {
  if (Op.isMachineOpcode()) {
    AddRegisterOperand(MI, Op, IIOpNum, II, VRBaseMap);
  } else if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateImm(C->getSExtValue()));
  } else if (ConstantFPSDNode *F = dyn_cast<ConstantFPSDNode>(Op)) {
    const ConstantFP *CFP = F->getConstantFPValue();
    MI->addOperand(MachineOperand::CreateFPImm(CFP));
  } else if (RegisterSDNode *R = dyn_cast<RegisterSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateReg(R->getReg(), false));
  } else if (GlobalAddressSDNode *TGA = dyn_cast<GlobalAddressSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateGA(TGA->getGlobal(), TGA->getOffset(),
                                            TGA->getTargetFlags()));
  } else if (BasicBlockSDNode *BBNode = dyn_cast<BasicBlockSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateMBB(BBNode->getBasicBlock()));
  } else if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateFI(FI->getIndex()));
  } else if (JumpTableSDNode *JT = dyn_cast<JumpTableSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateJTI(JT->getIndex(),
                                             JT->getTargetFlags()));
  } else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Op)) {
    int Offset = CP->getOffset();
    unsigned Align = CP->getAlignment();
    const Type *Type = CP->getType();
    // MachineConstantPool wants an explicit alignment.
    if (Align == 0) {
      Align = TM->getTargetData()->getPrefTypeAlignment(Type);
      if (Align == 0) {
        // Alignment of vector types.  FIXME!
        Align = TM->getTargetData()->getTypeAllocSize(Type);
      }
    }
    
    unsigned Idx;
    MachineConstantPool *MCP = MF->getConstantPool();
    if (CP->isMachineConstantPoolEntry())
      Idx = MCP->getConstantPoolIndex(CP->getMachineCPVal(), Align);
    else
      Idx = MCP->getConstantPoolIndex(CP->getConstVal(), Align);
    MI->addOperand(MachineOperand::CreateCPI(Idx, Offset,
                                             CP->getTargetFlags()));
  } else if (ExternalSymbolSDNode *ES = dyn_cast<ExternalSymbolSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateES(ES->getSymbol(),
                                            ES->getTargetFlags()));
  } else if (BlockAddressSDNode *BA = dyn_cast<BlockAddressSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateBA(BA->getBlockAddress()));
  } else {
    assert(Op.getValueType() != MVT::Other &&
           Op.getValueType() != MVT::Flag &&
           "Chain and flag operands should occur at end of operand list!");
    AddRegisterOperand(MI, Op, IIOpNum, II, VRBaseMap);
  }
}

/// getSuperRegisterRegClass - Returns the register class of a superreg A whose
/// "SubIdx"'th sub-register class is the specified register class and whose
/// type matches the specified type.
static const TargetRegisterClass*
getSuperRegisterRegClass(const TargetRegisterClass *TRC,
                         unsigned SubIdx, EVT VT) {
  // Pick the register class of the superegister for this type
  for (TargetRegisterInfo::regclass_iterator I = TRC->superregclasses_begin(),
         E = TRC->superregclasses_end(); I != E; ++I)
    if ((*I)->hasType(VT) && (*I)->getSubRegisterRegClass(SubIdx) == TRC)
      return *I;
  assert(false && "Couldn't find the register class");
  return 0;
}

/// EmitSubregNode - Generate machine code for subreg nodes.
///
void InstrEmitter::EmitSubregNode(SDNode *Node, 
                                  DenseMap<SDValue, unsigned> &VRBaseMap){
  unsigned VRBase = 0;
  unsigned Opc = Node->getMachineOpcode();
  
  // If the node is only used by a CopyToReg and the dest reg is a vreg, use
  // the CopyToReg'd destination register instead of creating a new vreg.
  for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
       UI != E; ++UI) {
    SDNode *User = *UI;
    if (User->getOpcode() == ISD::CopyToReg && 
        User->getOperand(2).getNode() == Node) {
      unsigned DestReg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
      if (TargetRegisterInfo::isVirtualRegister(DestReg)) {
        VRBase = DestReg;
        break;
      }
    }
  }
  
  if (Opc == TargetInstrInfo::EXTRACT_SUBREG) {
    unsigned SubIdx = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();

    // Create the extract_subreg machine instruction.
    MachineInstr *MI = BuildMI(*MF, Node->getDebugLoc(),
                               TII->get(TargetInstrInfo::EXTRACT_SUBREG));

    // Figure out the register class to create for the destreg.
    unsigned VReg = getVR(Node->getOperand(0), VRBaseMap);
    const TargetRegisterClass *TRC = MRI->getRegClass(VReg);
    const TargetRegisterClass *SRC = TRC->getSubRegisterRegClass(SubIdx);
    assert(SRC && "Invalid subregister index in EXTRACT_SUBREG");

    // Figure out the register class to create for the destreg.
    // Note that if we're going to directly use an existing register,
    // it must be precisely the required class, and not a subclass
    // thereof.
    if (VRBase == 0 || SRC != MRI->getRegClass(VRBase)) {
      // Create the reg
      assert(SRC && "Couldn't find source register class");
      VRBase = MRI->createVirtualRegister(SRC);
    }

    // Add def, source, and subreg index
    MI->addOperand(MachineOperand::CreateReg(VRBase, true));
    AddOperand(MI, Node->getOperand(0), 0, 0, VRBaseMap);
    MI->addOperand(MachineOperand::CreateImm(SubIdx));
    MBB->insert(InsertPos, MI);
  } else if (Opc == TargetInstrInfo::INSERT_SUBREG ||
             Opc == TargetInstrInfo::SUBREG_TO_REG) {
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);
    SDValue N2 = Node->getOperand(2);
    unsigned SubReg = getVR(N1, VRBaseMap);
    unsigned SubIdx = cast<ConstantSDNode>(N2)->getZExtValue();
    const TargetRegisterClass *TRC = MRI->getRegClass(SubReg);
    const TargetRegisterClass *SRC =
      getSuperRegisterRegClass(TRC, SubIdx,
                               Node->getValueType(0));

    // Figure out the register class to create for the destreg.
    // Note that if we're going to directly use an existing register,
    // it must be precisely the required class, and not a subclass
    // thereof.
    if (VRBase == 0 || SRC != MRI->getRegClass(VRBase)) {
      // Create the reg
      assert(SRC && "Couldn't find source register class");
      VRBase = MRI->createVirtualRegister(SRC);
    }

    // Create the insert_subreg or subreg_to_reg machine instruction.
    MachineInstr *MI = BuildMI(*MF, Node->getDebugLoc(), TII->get(Opc));
    MI->addOperand(MachineOperand::CreateReg(VRBase, true));
    
    // If creating a subreg_to_reg, then the first input operand
    // is an implicit value immediate, otherwise it's a register
    if (Opc == TargetInstrInfo::SUBREG_TO_REG) {
      const ConstantSDNode *SD = cast<ConstantSDNode>(N0);
      MI->addOperand(MachineOperand::CreateImm(SD->getZExtValue()));
    } else
      AddOperand(MI, N0, 0, 0, VRBaseMap);
    // Add the subregster being inserted
    AddOperand(MI, N1, 0, 0, VRBaseMap);
    MI->addOperand(MachineOperand::CreateImm(SubIdx));
    MBB->insert(InsertPos, MI);
  } else
    llvm_unreachable("Node is not insert_subreg, extract_subreg, or subreg_to_reg");
     
  SDValue Op(Node, 0);
  bool isNew = VRBaseMap.insert(std::make_pair(Op, VRBase)).second;
  isNew = isNew; // Silence compiler warning.
  assert(isNew && "Node emitted out of order - early");
}

/// EmitCopyToRegClassNode - Generate machine code for COPY_TO_REGCLASS nodes.
/// COPY_TO_REGCLASS is just a normal copy, except that the destination
/// register is constrained to be in a particular register class.
///
void
InstrEmitter::EmitCopyToRegClassNode(SDNode *Node,
                                     DenseMap<SDValue, unsigned> &VRBaseMap) {
  unsigned VReg = getVR(Node->getOperand(0), VRBaseMap);
  const TargetRegisterClass *SrcRC = MRI->getRegClass(VReg);

  unsigned DstRCIdx = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
  const TargetRegisterClass *DstRC = TRI->getRegClass(DstRCIdx);

  // Create the new VReg in the destination class and emit a copy.
  unsigned NewVReg = MRI->createVirtualRegister(DstRC);
  bool Emitted = TII->copyRegToReg(*MBB, InsertPos, NewVReg, VReg,
                                   DstRC, SrcRC);
  assert(Emitted &&
         "Unable to issue a copy instruction for a COPY_TO_REGCLASS node!\n");
  (void) Emitted;

  SDValue Op(Node, 0);
  bool isNew = VRBaseMap.insert(std::make_pair(Op, NewVReg)).second;
  isNew = isNew; // Silence compiler warning.
  assert(isNew && "Node emitted out of order - early");
}

/// EmitNode - Generate machine code for an node and needed dependencies.
///
void InstrEmitter::EmitNode(SDNode *Node, bool IsClone, bool IsCloned,
                            DenseMap<SDValue, unsigned> &VRBaseMap,
                         DenseMap<MachineBasicBlock*, MachineBasicBlock*> *EM) {
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

    // Handle COPY_TO_REGCLASS specially.
    if (Opc == TargetInstrInfo::COPY_TO_REGCLASS) {
      EmitCopyToRegClassNode(Node, VRBaseMap);
      return;
    }

    if (Opc == TargetInstrInfo::IMPLICIT_DEF)
      // We want a unique VR for each IMPLICIT_DEF use.
      return;
    
    const TargetInstrDesc &II = TII->get(Opc);
    unsigned NumResults = CountResults(Node);
    unsigned NodeOperands = CountOperands(Node);
    bool HasPhysRegOuts = (NumResults > II.getNumDefs()) &&
                          II.getImplicitDefs() != 0;
#ifndef NDEBUG
    unsigned NumMIOperands = NodeOperands + NumResults;
    assert((II.getNumOperands() == NumMIOperands ||
            HasPhysRegOuts || II.isVariadic()) &&
           "#operands for dag node doesn't match .td file!"); 
#endif

    // Create the new machine instruction.
    MachineInstr *MI = BuildMI(*MF, Node->getDebugLoc(), II);
    
    // Add result register values for things that are defined by this
    // instruction.
    if (NumResults)
      CreateVirtualRegisters(Node, MI, II, IsClone, IsCloned, VRBaseMap);
    
    // Emit all of the actual operands of this instruction, adding them to the
    // instruction as appropriate.
    bool HasOptPRefs = II.getNumDefs() > NumResults;
    assert((!HasOptPRefs || !HasPhysRegOuts) &&
           "Unable to cope with optional defs and phys regs defs!");
    unsigned NumSkip = HasOptPRefs ? II.getNumDefs() - NumResults : 0;
    for (unsigned i = NumSkip; i != NodeOperands; ++i)
      AddOperand(MI, Node->getOperand(i), i-NumSkip+II.getNumDefs(), &II,
                 VRBaseMap);

    // Transfer all of the memory reference descriptions of this instruction.
    MI->setMemRefs(cast<MachineSDNode>(Node)->memoperands_begin(),
                   cast<MachineSDNode>(Node)->memoperands_end());

    if (II.usesCustomInsertionHook()) {
      // Insert this instruction into the basic block using a target
      // specific inserter which may returns a new basic block.
      MBB = TLI->EmitInstrWithCustomInserter(MI, MBB, EM);
      InsertPos = MBB->end();
    } else {
      MBB->insert(InsertPos, MI);
    }

    // Additional results must be an physical register def.
    if (HasPhysRegOuts) {
      for (unsigned i = II.getNumDefs(); i < NumResults; ++i) {
        unsigned Reg = II.getImplicitDefs()[i - II.getNumDefs()];
        if (Node->hasAnyUseOfValue(i))
          EmitCopyFromReg(Node, i, IsClone, IsCloned, Reg, VRBaseMap);
        // If there are no uses, mark the register as dead now, so that
        // MachineLICM/Sink can see that it's dead. Don't do this if the
        // node has a Flag value, for the benefit of targets still using
        // Flag for values in physregs.
        else if (Node->getValueType(Node->getNumValues()-1) != MVT::Flag)
          MI->addRegisterDead(Reg, TRI);
      }
    }
    return;
  }

  switch (Node->getOpcode()) {
  default:
#ifndef NDEBUG
    Node->dump();
#endif
    llvm_unreachable("This target-independent node should have been selected!");
    break;
  case ISD::EntryToken:
    llvm_unreachable("EntryToken should have been excluded from the schedule!");
    break;
  case ISD::MERGE_VALUES:
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
      SrcTRC = MRI->getRegClass(SrcReg);
    else
      SrcTRC = TRI->getPhysicalRegisterRegClass(SrcReg,SrcVal.getValueType());

    if (TargetRegisterInfo::isVirtualRegister(DestReg))
      DstTRC = MRI->getRegClass(DestReg);
    else
      DstTRC = TRI->getPhysicalRegisterRegClass(DestReg,
                                            Node->getOperand(1).getValueType());

    bool Emitted = TII->copyRegToReg(*MBB, InsertPos, DestReg, SrcReg,
                                     DstTRC, SrcTRC);
    assert(Emitted && "Unable to issue a copy instruction!\n");
    (void) Emitted;
    break;
  }
  case ISD::CopyFromReg: {
    unsigned SrcReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
    EmitCopyFromReg(Node, 0, IsClone, IsCloned, SrcReg, VRBaseMap);
    break;
  }
  case ISD::INLINEASM: {
    unsigned NumOps = Node->getNumOperands();
    if (Node->getOperand(NumOps-1).getValueType() == MVT::Flag)
      --NumOps;  // Ignore the flag operand.
      
    // Create the inline asm machine instruction.
    MachineInstr *MI = BuildMI(*MF, Node->getDebugLoc(),
                               TII->get(TargetInstrInfo::INLINEASM));

    // Add the asm string as an external symbol operand.
    const char *AsmStr =
      cast<ExternalSymbolSDNode>(Node->getOperand(1))->getSymbol();
    MI->addOperand(MachineOperand::CreateES(AsmStr));
      
    // Add all of the operand registers to the instruction.
    for (unsigned i = 2; i != NumOps;) {
      unsigned Flags =
        cast<ConstantSDNode>(Node->getOperand(i))->getZExtValue();
      unsigned NumVals = InlineAsm::getNumOperandRegisters(Flags);
        
      MI->addOperand(MachineOperand::CreateImm(Flags));
      ++i;  // Skip the ID value.
        
      switch (Flags & 7) {
      default: llvm_unreachable("Bad flags!");
      case 2:   // Def of register.
        for (; NumVals; --NumVals, ++i) {
          unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
          MI->addOperand(MachineOperand::CreateReg(Reg, true));
        }
        break;
      case 6:   // Def of earlyclobber register.
        for (; NumVals; --NumVals, ++i) {
          unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
          MI->addOperand(MachineOperand::CreateReg(Reg, true, false, false, 
                                                   false, false, true));
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
    MBB->insert(InsertPos, MI);
    break;
  }
  }
}

/// InstrEmitter - Construct an InstrEmitter and set it to start inserting
/// at the given position in the given block.
InstrEmitter::InstrEmitter(MachineBasicBlock *mbb,
                           MachineBasicBlock::iterator insertpos)
  : MF(mbb->getParent()),
    MRI(&MF->getRegInfo()),
    TM(&MF->getTarget()),
    TII(TM->getInstrInfo()),
    TRI(TM->getRegisterInfo()),
    TLI(TM->getTargetLowering()),
    MBB(mbb), InsertPos(insertpos) {
}
