//===-- LegalizeTypesFloatToInt.cpp - LegalizeTypes float to int support --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements float to integer conversion for LegalizeTypes.  This
// is the act of turning a computation in an invalid floating point type into
// a computation in an integer type of the same size.  For example, turning
// f32 arithmetic into operations using i32.  Also known as "soft float".
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  Result Float to Integer Conversion.
//===----------------------------------------------------------------------===//

void DAGTypeLegalizer::FloatToIntResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "FloatToInt node result " << ResNo << ": "; N->dump(&DAG);
        cerr << "\n");
  SDOperand R = SDOperand();

  // FIXME: Custom lowering for float-to-int?
#if 0
  // See if the target wants to custom convert this node to an integer.
  if (TLI.getOperationAction(N->getOpcode(), N->getValueType(0)) ==
      TargetLowering::Custom) {
    // If the target wants to, allow it to lower this itself.
    if (SDNode *P = TLI.FloatToIntOperationResult(N, DAG)) {
      // Everything that once used N now uses P.  We are guaranteed that the
      // result value types of N and the result value types of P match.
      ReplaceNodeWith(N, P);
      return;
    }
  }
#endif

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "FloatToIntResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to convert the result of this operator!");
    abort();

    case ISD::BIT_CONVERT: R = FloatToIntRes_BIT_CONVERT(N); break;
    case ISD::BUILD_PAIR:  R = FloatToIntRes_BUILD_PAIR(N); break;
    case ISD::FCOPYSIGN:   R = FloatToIntRes_FCOPYSIGN(N); break;
  }

  // If R is null, the sub-method took care of registering the result.
  if (R.Val)
    SetIntegerOp(SDOperand(N, ResNo), R);
}

SDOperand DAGTypeLegalizer::FloatToIntRes_BIT_CONVERT(SDNode *N) {
  return BitConvertToInteger(N->getOperand(0));
}

SDOperand DAGTypeLegalizer::FloatToIntRes_BUILD_PAIR(SDNode *N) {
  // Convert the inputs to integers, and build a new pair out of them.
  return DAG.getNode(ISD::BUILD_PAIR,
                     TLI.getTypeToTransformTo(N->getValueType(0)),
                     BitConvertToInteger(N->getOperand(0)),
                     BitConvertToInteger(N->getOperand(1)));
}

SDOperand DAGTypeLegalizer::FloatToIntRes_FCOPYSIGN(SDNode *N) {
  SDOperand LHS = GetIntegerOp(N->getOperand(0));
  SDOperand RHS = BitConvertToInteger(N->getOperand(1));

  MVT::ValueType LVT = LHS.getValueType();
  MVT::ValueType RVT = RHS.getValueType();

  unsigned LSize = MVT::getSizeInBits(LVT);
  unsigned RSize = MVT::getSizeInBits(RVT);

  // First get the sign bit of second operand.
  SDOperand SignBit = DAG.getNode(ISD::SHL, RVT, DAG.getConstant(1, RVT),
                                  DAG.getConstant(RSize - 1,
                                                  TLI.getShiftAmountTy()));
  SignBit = DAG.getNode(ISD::AND, RVT, RHS, SignBit);

  // Shift right or sign-extend it if the two operands have different types.
  int SizeDiff = MVT::getSizeInBits(RVT) - MVT::getSizeInBits(LVT);
  if (SizeDiff > 0) {
    SignBit = DAG.getNode(ISD::SRL, RVT, SignBit,
                          DAG.getConstant(SizeDiff, TLI.getShiftAmountTy()));
    SignBit = DAG.getNode(ISD::TRUNCATE, LVT, SignBit);
  } else if (SizeDiff < 0) {
    SignBit = DAG.getNode(ISD::ANY_EXTEND, LVT, SignBit);
    SignBit = DAG.getNode(ISD::SHL, LVT, SignBit,
                          DAG.getConstant(-SizeDiff, TLI.getShiftAmountTy()));
  }

  // Clear the sign bit of the first operand.
  SDOperand Mask = DAG.getNode(ISD::SHL, LVT, DAG.getConstant(1, LVT),
                               DAG.getConstant(LSize - 1,
                                               TLI.getShiftAmountTy()));
  Mask = DAG.getNode(ISD::SUB, LVT, Mask, DAG.getConstant(1, LVT));
  LHS = DAG.getNode(ISD::AND, LVT, LHS, Mask);

  // Or the value with the sign bit.
  return DAG.getNode(ISD::OR, LVT, LHS, SignBit);
}


//===----------------------------------------------------------------------===//
//  Operand Float to Integer Conversion..
//===----------------------------------------------------------------------===//

bool DAGTypeLegalizer::FloatToIntOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "FloatToInt node operand " << OpNo << ": "; N->dump(&DAG);
        cerr << "\n");
  SDOperand Res(0, 0);

  // FIXME: Custom lowering for float-to-int?
#if 0
  if (TLI.getOperationAction(N->getOpcode(), N->getOperand(OpNo).getValueType())
      == TargetLowering::Custom)
    Res = TLI.LowerOperation(SDOperand(N, 0), DAG);
#endif

  if (Res.Val == 0) {
    switch (N->getOpcode()) {
    default:
#ifndef NDEBUG
      cerr << "FloatToIntOperand Op #" << OpNo << ": ";
      N->dump(&DAG); cerr << "\n";
#endif
      assert(0 && "Do not know how to convert this operator's operand!");
      abort();

      case ISD::BIT_CONVERT: Res = FloatToIntOp_BIT_CONVERT(N); break;
    }
  }

  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.Val) return false;

  // If the result is N, the sub-method updated N in place.  Check to see if any
  // operands are new, and if so, mark them.
  if (Res.Val == N) {
    // Mark N as new and remark N and its operands.  This allows us to correctly
    // revisit N if it needs another step of promotion and allows us to visit
    // any new operands to N.
    ReanalyzeNode(N);
    return true;
  }

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");

  ReplaceValueWith(SDOperand(N, 0), Res);
  return false;
}

SDOperand DAGTypeLegalizer::FloatToIntOp_BIT_CONVERT(SDNode *N) {
  return DAG.getNode(ISD::BIT_CONVERT, N->getValueType(0),
                     GetIntegerOp(N->getOperand(0)));
}
