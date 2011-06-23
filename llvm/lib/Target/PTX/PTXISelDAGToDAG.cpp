//===-- PTXISelDAGToDAG.cpp - A dag to dag inst selector for PTX ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the PTX target.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXTargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
// PTXDAGToDAGISel - PTX specific code to select PTX machine
// instructions for SelectionDAG operations.
class PTXDAGToDAGISel : public SelectionDAGISel {
  public:
    PTXDAGToDAGISel(PTXTargetMachine &TM, CodeGenOpt::Level OptLevel);

    virtual const char *getPassName() const {
      return "PTX DAG->DAG Pattern Instruction Selection";
    }

    SDNode *Select(SDNode *Node);

    // Complex Pattern Selectors.
    bool SelectADDRrr(SDValue &Addr, SDValue &R1, SDValue &R2);
    bool SelectADDRri(SDValue &Addr, SDValue &Base, SDValue &Offset);
    bool SelectADDRii(SDValue &Addr, SDValue &Base, SDValue &Offset);

    // Include the pieces auto'gened from the target description
#include "PTXGenDAGISel.inc"

  private:
    SDNode *SelectREAD_PARAM(SDNode *Node);
    //SDNode *SelectSTORE_PARAM(SDNode *Node);
    
    // We need this only because we can't match intruction BRAdp
    // pattern (PTXbrcond bb:$d, ...) in PTXInstrInfo.td
    SDNode *SelectBRCOND(SDNode *Node);

    bool isImm(const SDValue &operand);
    bool SelectImm(const SDValue &operand, SDValue &imm);

    const PTXSubtarget& getSubtarget() const;
}; // class PTXDAGToDAGISel
} // namespace

// createPTXISelDag - This pass converts a legalized DAG into a
// PTX-specific DAG, ready for instruction scheduling
FunctionPass *llvm::createPTXISelDag(PTXTargetMachine &TM,
                                     CodeGenOpt::Level OptLevel) {
  return new PTXDAGToDAGISel(TM, OptLevel);
}

PTXDAGToDAGISel::PTXDAGToDAGISel(PTXTargetMachine &TM,
                                 CodeGenOpt::Level OptLevel)
  : SelectionDAGISel(TM, OptLevel) {}

SDNode *PTXDAGToDAGISel::Select(SDNode *Node) {
  switch (Node->getOpcode()) {
    case PTXISD::READ_PARAM:
      return SelectREAD_PARAM(Node);
    // case PTXISD::STORE_PARAM:
    //   return SelectSTORE_PARAM(Node);
    case ISD::BRCOND:
      return SelectBRCOND(Node);
    default:
      return SelectCode(Node);
  }
}

SDNode *PTXDAGToDAGISel::SelectREAD_PARAM(SDNode *Node) {
  SDValue  index = Node->getOperand(1);
  DebugLoc dl    = Node->getDebugLoc();
  unsigned opcode;

  if (index.getOpcode() != ISD::TargetConstant)
    llvm_unreachable("READ_PARAM: index is not ISD::TargetConstant");

  if (Node->getValueType(0) == MVT::i16) {
    opcode = PTX::LDpiU16;
  } else if (Node->getValueType(0) == MVT::i32) {
    opcode = PTX::LDpiU32;
  } else if (Node->getValueType(0) == MVT::i64) {
    opcode = PTX::LDpiU64;
  } else if (Node->getValueType(0) == MVT::f32) {
    opcode = PTX::LDpiF32;
  } else if (Node->getValueType(0) == MVT::f64) {
    opcode = PTX::LDpiF64;
  } else {
    llvm_unreachable("Unknown parameter type for ld.param");
  }

  return PTXInstrInfo::
    GetPTXMachineNode(CurDAG, opcode, dl, Node->getValueType(0), index);
}

// SDNode *PTXDAGToDAGISel::SelectSTORE_PARAM(SDNode *Node) {
//   SDValue  Chain = Node->getOperand(0);
//   SDValue  index = Node->getOperand(1);
//   SDValue  value = Node->getOperand(2);
//   DebugLoc dl    = Node->getDebugLoc();
//   unsigned opcode;

//   if (index.getOpcode() != ISD::TargetConstant)
//     llvm_unreachable("STORE_PARAM: index is not ISD::TargetConstant");

//   if (value->getValueType(0) == MVT::i16) {
//     opcode = PTX::STpiU16;
//   } else if (value->getValueType(0) == MVT::i32) {
//     opcode = PTX::STpiU32;
//   } else if (value->getValueType(0) == MVT::i64) {
//     opcode = PTX::STpiU64;
//   } else if (value->getValueType(0) == MVT::f32) {
//     opcode = PTX::STpiF32;
//   } else if (value->getValueType(0) == MVT::f64) {
//     opcode = PTX::STpiF64;
//   } else {
//     llvm_unreachable("Unknown parameter type for st.param");
//   }

//   SDVTList VTs = CurDAG->getVTList(MVT::Other, MVT::Glue);
//   SDValue PredReg = CurDAG->getRegister(PTX::NoRegister, MVT::i1);
//   SDValue PredOp = CurDAG->getTargetConstant(PTX::PRED_NORMAL, MVT::i32);
//   SDValue Ops[] = { Chain, index, value, PredReg, PredOp };
//   //SDNode *RetNode = PTXInstrInfo::
//   //  GetPTXMachineNode(CurDAG, opcode, dl, VTs, index, value);
//   SDNode *RetNode = CurDAG->getMachineNode(opcode, dl, VTs, Ops, array_lengthof(Ops));
//   DEBUG(dbgs() << "SelectSTORE_PARAM: Selected: ");
//   RetNode->dumpr(CurDAG);
//   return RetNode;
// }

SDNode *PTXDAGToDAGISel::SelectBRCOND(SDNode *Node) {
  assert(Node->getNumOperands() >= 3);

  SDValue Chain  = Node->getOperand(0);
  SDValue Pred   = Node->getOperand(1);
  SDValue Target = Node->getOperand(2); // branch target
  SDValue PredOp = CurDAG->getTargetConstant(PTX::PRED_NORMAL, MVT::i32);
  DebugLoc dl = Node->getDebugLoc();

  assert(Target.getOpcode()  == ISD::BasicBlock);
  assert(Pred.getValueType() == MVT::i1);

  // Emit BRAdp
  SDValue Ops[] = { Target, Pred, PredOp, Chain };
  return CurDAG->getMachineNode(PTX::BRAdp, dl, MVT::Other, Ops, 4);
}

// Match memory operand of the form [reg+reg]
bool PTXDAGToDAGISel::SelectADDRrr(SDValue &Addr, SDValue &R1, SDValue &R2) {
  if (Addr.getOpcode() != ISD::ADD || Addr.getNumOperands() < 2 ||
      isImm(Addr.getOperand(0)) || isImm(Addr.getOperand(1)))
    return false;

  assert(Addr.getValueType().isSimple() && "Type must be simple");

  R1 = Addr;
  R2 = CurDAG->getTargetConstant(0, Addr.getValueType().getSimpleVT());

  return true;
}

// Match memory operand of the form [reg], [imm+reg], and [reg+imm]
bool PTXDAGToDAGISel::SelectADDRri(SDValue &Addr, SDValue &Base,
                                   SDValue &Offset) {
  if (Addr.getOpcode() != ISD::ADD) {
    // let SelectADDRii handle the [imm] case
    if (isImm(Addr))
      return false;
    // it is [reg]

    assert(Addr.getValueType().isSimple() && "Type must be simple");

    Base = Addr;
    Offset = CurDAG->getTargetConstant(0, Addr.getValueType().getSimpleVT());

    return true;
  }

  if (Addr.getNumOperands() < 2)
    return false;

  // let SelectADDRii handle the [imm+imm] case
  if (isImm(Addr.getOperand(0)) && isImm(Addr.getOperand(1)))
    return false;

  // try [reg+imm] and [imm+reg]
  for (int i = 0; i < 2; i ++)
    if (SelectImm(Addr.getOperand(1-i), Offset)) {
      Base = Addr.getOperand(i);
      return true;
    }

  // neither [reg+imm] nor [imm+reg]
  return false;
}

// Match memory operand of the form [imm+imm] and [imm]
bool PTXDAGToDAGISel::SelectADDRii(SDValue &Addr, SDValue &Base,
                                   SDValue &Offset) {
  // is [imm+imm]?
  if (Addr.getOpcode() == ISD::ADD) {
    return SelectImm(Addr.getOperand(0), Base) &&
           SelectImm(Addr.getOperand(1), Offset);
  }

  // is [imm]?
  if (SelectImm(Addr, Base)) {
    assert(Addr.getValueType().isSimple() && "Type must be simple");

    Offset = CurDAG->getTargetConstant(0, Addr.getValueType().getSimpleVT());

    return true;
  }

  return false;
}

bool PTXDAGToDAGISel::isImm(const SDValue &operand) {
  return ConstantSDNode::classof(operand.getNode());
}

bool PTXDAGToDAGISel::SelectImm(const SDValue &operand, SDValue &imm) {
  SDNode *node = operand.getNode();
  if (!ConstantSDNode::classof(node))
    return false;

  ConstantSDNode *CN = cast<ConstantSDNode>(node);
  imm = CurDAG->getTargetConstant(*CN->getConstantIntValue(),
                                  operand.getValueType());
  return true;
}

const PTXSubtarget& PTXDAGToDAGISel::getSubtarget() const
{
  return TM.getSubtarget<PTXSubtarget>();
}

