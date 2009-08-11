//===-- LegalizeVectorOps.cpp - Implement SelectionDAG::LegalizeVectors ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAG::LegalizeVectors method.
//
// The vector legalizer looks for vector operations which might need to be
// scalarized and legalizes them. This is a separate step from Legalize because
// scalarizing can introduce illegal types.  For example, suppose we have an
// ISD::SDIV of type v2i64 on x86-32.  The type is legal (for example, addition
// on a v2i64 is legal), but ISD::SDIV isn't legal, so we have to unroll the
// operation, which introduces nodes with the illegal type i64 which must be
// expanded.  Similarly, suppose we have an ISD::SRA of type v16i8 on PowerPC;
// the operation must be unrolled, which introduces nodes with the illegal
// type i8 which must be promoted.
//
// This does not legalize vector manipulations like ISD::BUILD_VECTOR,
// or operations that happen to take a vector which are custom-lowered;
// the legalization for such operations never produces nodes
// with illegal types, so it's okay to put off legalizing them until
// SelectionDAG::Legalize runs.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

namespace {
class VectorLegalizer {
  SelectionDAG& DAG;
  TargetLowering& TLI;
  bool Changed; // Keep track of whether anything changed

  /// LegalizedNodes - For nodes that are of legal width, and that have more
  /// than one use, this map indicates what regularized operand to use.  This
  /// allows us to avoid legalizing the same thing more than once.
  DenseMap<SDValue, SDValue> LegalizedNodes;

  // Adds a node to the translation cache
  void AddLegalizedOperand(SDValue From, SDValue To) {
    LegalizedNodes.insert(std::make_pair(From, To));
    // If someone requests legalization of the new node, return itself.
    if (From != To)
      LegalizedNodes.insert(std::make_pair(To, To));
  }

  // Legalizes the given node
  SDValue LegalizeOp(SDValue Op);
  // Assuming the node is legal, "legalize" the results
  SDValue TranslateLegalizeResults(SDValue Op, SDValue Result);
  // Implements unrolling a generic vector operation, i.e. turning it into
  // scalar operations.
  SDValue UnrollVectorOp(SDValue Op);
  // Implements unrolling a VSETCC.
  SDValue UnrollVSETCC(SDValue Op);
  // Implements expansion for FNEG; falls back to UnrollVectorOp if FSUB
  // isn't legal.
  SDValue ExpandFNEG(SDValue Op);
  // Implements vector promotion; this is essentially just bitcasting the
  // operands to a different type and bitcasting the result back to the
  // original type.
  SDValue PromoteVectorOp(SDValue Op);

  public:
  bool Run();
  VectorLegalizer(SelectionDAG& dag) :
      DAG(dag), TLI(dag.getTargetLoweringInfo()), Changed(false) {}
};

bool VectorLegalizer::Run() {
  // The legalize process is inherently a bottom-up recursive process (users
  // legalize their uses before themselves).  Given infinite stack space, we
  // could just start legalizing on the root and traverse the whole graph.  In
  // practice however, this causes us to run out of stack space on large basic
  // blocks.  To avoid this problem, compute an ordering of the nodes where each
  // node is only legalized after all of its operands are legalized.
  DAG.AssignTopologicalOrder();
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = prior(DAG.allnodes_end()); I != next(E); ++I)
    LegalizeOp(SDValue(I, 0));

  // Finally, it's possible the root changed.  Get the new root.
  SDValue OldRoot = DAG.getRoot();
  assert(LegalizedNodes.count(OldRoot) && "Root didn't get legalized?");
  DAG.setRoot(LegalizedNodes[OldRoot]);

  LegalizedNodes.clear();

  // Remove dead nodes now.
  DAG.RemoveDeadNodes();

  return Changed;
}

SDValue VectorLegalizer::TranslateLegalizeResults(SDValue Op, SDValue Result) {
  // Generic legalization: just pass the operand through.
  for (unsigned i = 0, e = Op.getNode()->getNumValues(); i != e; ++i)
    AddLegalizedOperand(Op.getValue(i), Result.getValue(i));
  return Result.getValue(Op.getResNo());
}

SDValue VectorLegalizer::LegalizeOp(SDValue Op) {
  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  DenseMap<SDValue, SDValue>::iterator I = LegalizedNodes.find(Op);
  if (I != LegalizedNodes.end()) return I->second;

  SDNode* Node = Op.getNode();

  // Legalize the operands
  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
    Ops.push_back(LegalizeOp(Node->getOperand(i)));

  SDValue Result =
      DAG.UpdateNodeOperands(Op.getValue(0), Ops.data(), Ops.size());

  bool HasVectorValue = false;
  for (SDNode::value_iterator J = Node->value_begin(), E = Node->value_end();
       J != E;
       ++J)
    HasVectorValue |= J->isVector();
  if (!HasVectorValue)
    return TranslateLegalizeResults(Op, Result);

  EVT QueryType;
  switch (Op.getOpcode()) {
  default:
    return TranslateLegalizeResults(Op, Result);
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SREM:
  case ISD::UREM:
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FREM:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTL:
  case ISD::ROTR:
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP:
  case ISD::SELECT:
  case ISD::SELECT_CC:
  case ISD::VSETCC:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
  case ISD::TRUNCATE:
  case ISD::SIGN_EXTEND:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::FNEG:
  case ISD::FABS:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::FPOWI:
  case ISD::FPOW:
  case ISD::FLOG:
  case ISD::FLOG2:
  case ISD::FLOG10:
  case ISD::FEXP:
  case ISD::FEXP2:
  case ISD::FCEIL:
  case ISD::FTRUNC:
  case ISD::FRINT:
  case ISD::FNEARBYINT:
  case ISD::FFLOOR:
    QueryType = Node->getValueType(0);
    break;
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    QueryType = Node->getOperand(0).getValueType();
    break;
  }

  switch (TLI.getOperationAction(Node->getOpcode(), QueryType)) {
  case TargetLowering::Promote:
    // "Promote" the operation by bitcasting
    Result = PromoteVectorOp(Op);
    Changed = true;
    break;
  case TargetLowering::Legal: break;
  case TargetLowering::Custom: {
    SDValue Tmp1 = TLI.LowerOperation(Op, DAG);
    if (Tmp1.getNode()) {
      Result = Tmp1;
      break;
    }
    // FALL THROUGH
  }
  case TargetLowering::Expand:
    if (Node->getOpcode() == ISD::FNEG)
      Result = ExpandFNEG(Op);
    else if (Node->getOpcode() == ISD::VSETCC)
      Result = UnrollVSETCC(Op);
    else
      Result = UnrollVectorOp(Op);
    break;
  }

  // Make sure that the generated code is itself legal.
  if (Result != Op) {
    Result = LegalizeOp(Result);
    Changed = true;
  }

  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  AddLegalizedOperand(Op, Result);
  return Result;
}

SDValue VectorLegalizer::PromoteVectorOp(SDValue Op) {
  // Vector "promotion" is basically just bitcasting and doing the operation
  // in a different type.  For example, x86 promotes ISD::AND on v2i32 to
  // v1i64.
  EVT VT = Op.getValueType();
  assert(Op.getNode()->getNumValues() == 1 &&
         "Can't promote a vector with multiple results!");
  EVT NVT = TLI.getTypeToPromoteTo(Op.getOpcode(), VT);
  DebugLoc dl = Op.getDebugLoc();
  SmallVector<SDValue, 4> Operands(Op.getNumOperands());

  for (unsigned j = 0; j != Op.getNumOperands(); ++j) {
    if (Op.getOperand(j).getValueType().isVector())
      Operands[j] = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Op.getOperand(j));
    else
      Operands[j] = Op.getOperand(j);
  }

  Op = DAG.getNode(Op.getOpcode(), dl, NVT, &Operands[0], Operands.size());

  return DAG.getNode(ISD::BIT_CONVERT, dl, VT, Op);
}

SDValue VectorLegalizer::ExpandFNEG(SDValue Op) {
  if (TLI.isOperationLegalOrCustom(ISD::FSUB, Op.getValueType())) {
    SDValue Zero = DAG.getConstantFP(-0.0, Op.getValueType());
    return DAG.getNode(ISD::FSUB, Op.getDebugLoc(), Op.getValueType(),
                       Zero, Op.getOperand(0));
  }
  return UnrollVectorOp(Op);
}

SDValue VectorLegalizer::UnrollVSETCC(SDValue Op) {
  EVT VT = Op.getValueType();
  unsigned NumElems = VT.getVectorNumElements();
  EVT EltVT = VT.getVectorElementType();
  SDValue LHS = Op.getOperand(0), RHS = Op.getOperand(1), CC = Op.getOperand(2);
  EVT TmpEltVT = LHS.getValueType().getVectorElementType();
  DebugLoc dl = Op.getDebugLoc();
  SmallVector<SDValue, 8> Ops(NumElems);
  for (unsigned i = 0; i < NumElems; ++i) {
    SDValue LHSElem = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, TmpEltVT, LHS,
                                  DAG.getIntPtrConstant(i));
    SDValue RHSElem = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, TmpEltVT, RHS,
                                  DAG.getIntPtrConstant(i));
    Ops[i] = DAG.getNode(ISD::SETCC, dl, TLI.getSetCCResultType(TmpEltVT),
                         LHSElem, RHSElem, CC);
    Ops[i] = DAG.getNode(ISD::SELECT, dl, EltVT, Ops[i],
                         DAG.getConstant(APInt::getAllOnesValue
                                         (EltVT.getSizeInBits()), EltVT),
                         DAG.getConstant(0, EltVT));
  }
  return DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &Ops[0], NumElems);
}

/// UnrollVectorOp - We know that the given vector has a legal type, however
/// the operation it performs is not legal, and the target has requested that
/// the operation be expanded.  "Unroll" the vector, splitting out the scalars
/// and operating on each element individually.
SDValue VectorLegalizer::UnrollVectorOp(SDValue Op) {
  EVT VT = Op.getValueType();
  assert(Op.getNode()->getNumValues() == 1 &&
         "Can't unroll a vector with multiple results!");
  unsigned NE = VT.getVectorNumElements();
  EVT EltVT = VT.getVectorElementType();
  DebugLoc dl = Op.getDebugLoc();

  SmallVector<SDValue, 8> Scalars;
  SmallVector<SDValue, 4> Operands(Op.getNumOperands());
  for (unsigned i = 0; i != NE; ++i) {
    for (unsigned j = 0; j != Op.getNumOperands(); ++j) {
      SDValue Operand = Op.getOperand(j);
      EVT OperandVT = Operand.getValueType();
      if (OperandVT.isVector()) {
        // A vector operand; extract a single element.
        EVT OperandEltVT = OperandVT.getVectorElementType();
        Operands[j] = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl,
                                  OperandEltVT,
                                  Operand,
                                  DAG.getConstant(i, MVT::i32));
      } else {
        // A scalar operand; just use it as is.
        Operands[j] = Operand;
      }
    }

    switch (Op.getOpcode()) {
    default:
      Scalars.push_back(DAG.getNode(Op.getOpcode(), dl, EltVT,
                                    &Operands[0], Operands.size()));
      break;
    case ISD::SHL:
    case ISD::SRA:
    case ISD::SRL:
    case ISD::ROTL:
    case ISD::ROTR:
      Scalars.push_back(DAG.getNode(Op.getOpcode(), dl, EltVT, Operands[0],
                                    DAG.getShiftAmountOperand(Operands[1])));
      break;
    }
  }

  return DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &Scalars[0], Scalars.size());
}

}

bool SelectionDAG::LegalizeVectors() {
  return VectorLegalizer(*this).Run();
}
