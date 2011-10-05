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
#include "PTXMachineFunctionInfo.h"
#include "PTXTargetMachine.h"
#include "llvm/ADT/StringExtras.h"
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
    bool SelectADDRlocal(SDValue &Addr, SDValue &Base, SDValue &Offset);

    // Include the pieces auto'gened from the target description
#include "PTXGenDAGISel.inc"

  private:
    // We need this only because we can't match intruction BRAdp
    // pattern (PTXbrcond bb:$d, ...) in PTXInstrInfo.td
    SDNode *SelectBRCOND(SDNode *Node);

    SDNode *SelectREADPARAM(SDNode *Node);
    SDNode *SelectWRITEPARAM(SDNode *Node);
    SDNode *SelectFrameIndex(SDNode *Node);

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
    case ISD::BRCOND:
      return SelectBRCOND(Node);
    case PTXISD::READ_PARAM:
      return SelectREADPARAM(Node);
    case PTXISD::WRITE_PARAM:
      return SelectWRITEPARAM(Node);
    case ISD::FrameIndex:
      return SelectFrameIndex(Node);
    default:
      return SelectCode(Node);
  }
}

SDNode *PTXDAGToDAGISel::SelectBRCOND(SDNode *Node) {
  assert(Node->getNumOperands() >= 3);

  SDValue Chain  = Node->getOperand(0);
  SDValue Pred   = Node->getOperand(1);
  SDValue Target = Node->getOperand(2); // branch target
  SDValue PredOp = CurDAG->getTargetConstant(PTXPredicate::Normal, MVT::i32);
  DebugLoc dl = Node->getDebugLoc();

  assert(Target.getOpcode()  == ISD::BasicBlock);
  assert(Pred.getValueType() == MVT::i1);

  // Emit BRAdp
  SDValue Ops[] = { Target, Pred, PredOp, Chain };
  return CurDAG->getMachineNode(PTX::BRAdp, dl, MVT::Other, Ops, 4);
}

SDNode *PTXDAGToDAGISel::SelectREADPARAM(SDNode *Node) {
  SDValue Chain = Node->getOperand(0);
  SDValue Index = Node->getOperand(1);

  int OpCode;

  // Get the type of parameter we are reading
  EVT VT = Node->getValueType(0);
  assert(VT.isSimple() && "READ_PARAM only implemented for MVT types");

  MVT Type = VT.getSimpleVT();

  if (Type == MVT::i1)
    OpCode = PTX::READPARAMPRED;
  else if (Type == MVT::i16)
    OpCode = PTX::READPARAMI16;
  else if (Type == MVT::i32)
    OpCode = PTX::READPARAMI32;
  else if (Type == MVT::i64)
    OpCode = PTX::READPARAMI64;
  else if (Type == MVT::f32)
    OpCode = PTX::READPARAMF32;
  else {
    assert(Type == MVT::f64 && "Unexpected type!");
    OpCode = PTX::READPARAMF64;
  }

  SDValue Pred = CurDAG->getRegister(PTX::NoRegister, MVT::i1);
  SDValue PredOp = CurDAG->getTargetConstant(PTXPredicate::None, MVT::i32);
  DebugLoc dl = Node->getDebugLoc();

  SDValue Ops[] = { Index, Pred, PredOp, Chain };
  return CurDAG->getMachineNode(OpCode, dl, VT, Ops, 4);
}

SDNode *PTXDAGToDAGISel::SelectWRITEPARAM(SDNode *Node) {

  SDValue Chain = Node->getOperand(0);
  SDValue Value = Node->getOperand(1);

  int OpCode;

  //Node->dumpr(CurDAG);

  // Get the type of parameter we are writing
  EVT VT = Value->getValueType(0);
  assert(VT.isSimple() && "WRITE_PARAM only implemented for MVT types");

  MVT Type = VT.getSimpleVT();

  if (Type == MVT::i1)
    OpCode = PTX::WRITEPARAMPRED;
  else if (Type == MVT::i16)
    OpCode = PTX::WRITEPARAMI16;
  else if (Type == MVT::i32)
    OpCode = PTX::WRITEPARAMI32;
  else if (Type == MVT::i64)
    OpCode = PTX::WRITEPARAMI64;
  else if (Type == MVT::f32)
    OpCode = PTX::WRITEPARAMF32;
  else if (Type == MVT::f64)
    OpCode = PTX::WRITEPARAMF64;
  else
    llvm_unreachable("Invalid type in SelectWRITEPARAM");

  SDValue Pred = CurDAG->getRegister(PTX::NoRegister, MVT::i1);
  SDValue PredOp = CurDAG->getTargetConstant(PTXPredicate::None, MVT::i32);
  DebugLoc dl = Node->getDebugLoc();

  SDValue Ops[] = { Value, Pred, PredOp, Chain };
  SDNode* Ret = CurDAG->getMachineNode(OpCode, dl, MVT::Other, Ops, 4);

  //dbgs() << "SelectWRITEPARAM produced:\n\t";
  //Ret->dumpr(CurDAG);

  return Ret;
}

SDNode *PTXDAGToDAGISel::SelectFrameIndex(SDNode *Node) {
  int FI = cast<FrameIndexSDNode>(Node)->getIndex();
  //dbgs() << "Selecting FrameIndex at index " << FI << "\n";
  //SDValue TFI = CurDAG->getTargetFrameIndex(FI, Node->getValueType(0));

  PTXMachineFunctionInfo *MFI = MF->getInfo<PTXMachineFunctionInfo>();

  SDValue FrameSymbol = CurDAG->getTargetExternalSymbol(MFI->getFrameSymbol(FI),
                                                        Node->getValueType(0));

  return FrameSymbol.getNode();
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
  // FrameIndex addresses are handled separately
  //errs() << "SelectADDRri: ";
  //Addr.getNode()->dumpr();
  if (isa<FrameIndexSDNode>(Addr)) {
    //errs() << "Failure\n";
    return false;
  }

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    Base = Addr.getOperand(0);
    if (isa<FrameIndexSDNode>(Base)) {
      //errs() << "Failure\n";
      return false;
    }
    ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1));
    Offset = CurDAG->getTargetConstant(CN->getZExtValue(), MVT::i32);
    //errs() << "Success\n";
    return true;
  }

  /*if (Addr.getNumOperands() == 1) {
    Base = Addr;
    Offset = CurDAG->getTargetConstant(0, Addr.getValueType().getSimpleVT());
    errs() << "Success\n";
    return true;
  }*/

  //errs() << "SelectADDRri fails on: ";
  //Addr.getNode()->dumpr();

  if (isImm(Addr)) {
    //errs() << "Failure\n";
    return false;
  }

  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, Addr.getValueType().getSimpleVT());

  //errs() << "Success\n";
  return true;

  /*if (Addr.getOpcode() != ISD::ADD) {
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
  return false;*/
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

// Match memory operand of the form [reg], [imm+reg], and [reg+imm]
bool PTXDAGToDAGISel::SelectADDRlocal(SDValue &Addr, SDValue &Base,
                                      SDValue &Offset) {
  //errs() << "SelectADDRlocal: ";
  //Addr.getNode()->dumpr();
  if (isa<FrameIndexSDNode>(Addr)) {
    Base = Addr;
    Offset = CurDAG->getTargetConstant(0, Addr.getValueType().getSimpleVT());
    //errs() << "Success\n";
    return true;
  }

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    Base = Addr.getOperand(0);
    if (!isa<FrameIndexSDNode>(Base)) {
      //errs() << "Failure\n";
      return false;
    }
    ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1));
    Offset = CurDAG->getTargetConstant(CN->getZExtValue(), MVT::i32);
    //errs() << "Offset: ";
    //Offset.getNode()->dumpr();
    //errs() << "Success\n";
    return true;
  }

  //errs() << "Failure\n";
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

