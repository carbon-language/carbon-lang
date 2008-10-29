//===-------- LegalizeFloatTypes.cpp - Legalization of float types --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements float type expansion and softening for LegalizeTypes.
// Softening is the act of turning a computation in an illegal floating point
// type into a computation in an integer type of the same size; also known as
// "soft float".  For example, turning f32 arithmetic into operations using i32.
// The resulting integer value is the same as what you would get by performing
// the floating point operation and bitcasting the result to the integer type.
// Expansion is the act of changing a computation in an illegal type to be a
// computation in two identical registers of a smaller type.  For example,
// implementing ppcf128 arithmetic in two f64 registers.
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
using namespace llvm;

/// GetFPLibCall - Return the right libcall for the given floating point type.
static RTLIB::Libcall GetFPLibCall(MVT VT,
                                   RTLIB::Libcall Call_F32,
                                   RTLIB::Libcall Call_F64,
                                   RTLIB::Libcall Call_F80,
                                   RTLIB::Libcall Call_PPCF128) {
  return
    VT == MVT::f32 ? Call_F32 :
    VT == MVT::f64 ? Call_F64 :
    VT == MVT::f80 ? Call_F80 :
    VT == MVT::ppcf128 ? Call_PPCF128 :
    RTLIB::UNKNOWN_LIBCALL;
}

//===----------------------------------------------------------------------===//
//  Result Float to Integer Conversion.
//===----------------------------------------------------------------------===//

void DAGTypeLegalizer::SoftenFloatResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Soften float result " << ResNo << ": "; N->dump(&DAG);
        cerr << "\n");
  SDValue R = SDValue();

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "SoftenFloatResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to soften the result of this operator!");
    abort();

    case ISD::BIT_CONVERT: R = SoftenFloatRes_BIT_CONVERT(N); break;
    case ISD::BUILD_PAIR:  R = SoftenFloatRes_BUILD_PAIR(N); break;
    case ISD::ConstantFP:
      R = SoftenFloatRes_ConstantFP(cast<ConstantFPSDNode>(N));
      break;
    case ISD::FABS:        R = SoftenFloatRes_FABS(N); break;
    case ISD::FADD:        R = SoftenFloatRes_FADD(N); break;
    case ISD::FCOPYSIGN:   R = SoftenFloatRes_FCOPYSIGN(N); break;
    case ISD::FDIV:        R = SoftenFloatRes_FDIV(N); break;
    case ISD::FMUL:        R = SoftenFloatRes_FMUL(N); break;
    case ISD::FP_EXTEND:   R = SoftenFloatRes_FP_EXTEND(N); break;
    case ISD::FP_ROUND:    R = SoftenFloatRes_FP_ROUND(N); break;
    case ISD::FPOW:        R = SoftenFloatRes_FPOW(N); break;
    case ISD::FPOWI:       R = SoftenFloatRes_FPOWI(N); break;
    case ISD::FSUB:        R = SoftenFloatRes_FSUB(N); break;
    case ISD::LOAD:        R = SoftenFloatRes_LOAD(N); break;
    case ISD::SELECT:      R = SoftenFloatRes_SELECT(N); break;
    case ISD::SELECT_CC:   R = SoftenFloatRes_SELECT_CC(N); break;
    case ISD::SINT_TO_FP:  R = SoftenFloatRes_SINT_TO_FP(N); break;
    case ISD::UINT_TO_FP:  R = SoftenFloatRes_UINT_TO_FP(N); break;
  }

  // If R is null, the sub-method took care of registering the result.
  if (R.getNode())
    SetSoftenedFloat(SDValue(N, ResNo), R);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_BIT_CONVERT(SDNode *N) {
  return BitConvertToInteger(N->getOperand(0));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_BUILD_PAIR(SDNode *N) {
  // Convert the inputs to integers, and build a new pair out of them.
  return DAG.getNode(ISD::BUILD_PAIR,
                     TLI.getTypeToTransformTo(N->getValueType(0)),
                     BitConvertToInteger(N->getOperand(0)),
                     BitConvertToInteger(N->getOperand(1)));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_ConstantFP(ConstantFPSDNode *N) {
  return DAG.getConstant(N->getValueAPF().bitcastToAPInt(),
                         TLI.getTypeToTransformTo(N->getValueType(0)));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FABS(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  unsigned Size = NVT.getSizeInBits();

  // Mask = ~(1 << (Size-1))
  SDValue Mask = DAG.getConstant(APInt::getAllOnesValue(Size).clear(Size-1), 
                                 NVT);
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return DAG.getNode(ISD::AND, NVT, Op, Mask);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FADD(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                       GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::ADD_F32,
                                  RTLIB::ADD_F64,
                                  RTLIB::ADD_F80,
                                  RTLIB::ADD_PPCF128),
                     NVT, Ops, 2, false);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FCOPYSIGN(SDNode *N) {
  SDValue LHS = GetSoftenedFloat(N->getOperand(0));
  SDValue RHS = BitConvertToInteger(N->getOperand(1));

  MVT LVT = LHS.getValueType();
  MVT RVT = RHS.getValueType();

  unsigned LSize = LVT.getSizeInBits();
  unsigned RSize = RVT.getSizeInBits();

  // First get the sign bit of second operand.
  SDValue SignBit = DAG.getNode(ISD::SHL, RVT, DAG.getConstant(1, RVT),
                                  DAG.getConstant(RSize - 1,
                                                  TLI.getShiftAmountTy()));
  SignBit = DAG.getNode(ISD::AND, RVT, RHS, SignBit);

  // Shift right or sign-extend it if the two operands have different types.
  int SizeDiff = RVT.getSizeInBits() - LVT.getSizeInBits();
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
  SDValue Mask = DAG.getNode(ISD::SHL, LVT, DAG.getConstant(1, LVT),
                               DAG.getConstant(LSize - 1,
                                               TLI.getShiftAmountTy()));
  Mask = DAG.getNode(ISD::SUB, LVT, Mask, DAG.getConstant(1, LVT));
  LHS = DAG.getNode(ISD::AND, LVT, LHS, Mask);

  // Or the value with the sign bit.
  return DAG.getNode(ISD::OR, LVT, LHS, SignBit);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FDIV(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                     GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::DIV_F32,
                                  RTLIB::DIV_F64,
                                  RTLIB::DIV_F80,
                                  RTLIB::DIV_PPCF128),
                     NVT, Ops, 2, false);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FMUL(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                       GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::MUL_F32,
                                  RTLIB::MUL_F64,
                                  RTLIB::MUL_F80,
                                  RTLIB::MUL_PPCF128),
                     NVT, Ops, 2, false);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FP_EXTEND(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Op = N->getOperand(0);
  RTLIB::Libcall LC = RTLIB::getFPEXT(Op.getValueType(), N->getValueType(0));
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_EXTEND!");
  return MakeLibCall(LC, NVT, &Op, 1, false);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FP_ROUND(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Op = N->getOperand(0);
  RTLIB::Libcall LC = RTLIB::getFPROUND(Op.getValueType(), N->getValueType(0));
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_ROUND!");
  return MakeLibCall(LC, NVT, &Op, 1, false);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FPOW(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                     GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::POW_F32,
                                  RTLIB::POW_F64,
                                  RTLIB::POW_F80,
                                  RTLIB::POW_PPCF128),
                     NVT, Ops, 2, false);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FPOWI(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)), N->getOperand(1) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::POWI_F32,
                                  RTLIB::POWI_F64,
                                  RTLIB::POWI_F80,
                                  RTLIB::POWI_PPCF128),
                     NVT, Ops, 2, false);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FSUB(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                       GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::SUB_F32,
                                  RTLIB::SUB_F64,
                                  RTLIB::SUB_F80,
                                  RTLIB::SUB_PPCF128),
                     NVT, Ops, 2, false);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_LOAD(SDNode *N) {
  LoadSDNode *L = cast<LoadSDNode>(N);
  MVT VT = N->getValueType(0);
  MVT NVT = TLI.getTypeToTransformTo(VT);

  SDValue NewL;
  if (L->getExtensionType() == ISD::NON_EXTLOAD) {
    NewL = DAG.getLoad(L->getAddressingMode(), L->getExtensionType(),
                       NVT, L->getChain(), L->getBasePtr(), L->getOffset(),
                       L->getSrcValue(), L->getSrcValueOffset(), NVT,
                       L->isVolatile(), L->getAlignment());
    // Legalized the chain result - switch anything that used the old chain to
    // use the new one.
    ReplaceValueWith(SDValue(N, 1), NewL.getValue(1));
    return NewL;
  }

  // Do a non-extending load followed by FP_EXTEND.
  NewL = DAG.getLoad(L->getAddressingMode(), ISD::NON_EXTLOAD,
                     L->getMemoryVT(), L->getChain(),
                     L->getBasePtr(), L->getOffset(),
                     L->getSrcValue(), L->getSrcValueOffset(),
                     L->getMemoryVT(),
                     L->isVolatile(), L->getAlignment());
  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDValue(N, 1), NewL.getValue(1));
  return BitConvertToInteger(DAG.getNode(ISD::FP_EXTEND, VT, NewL));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_SELECT(SDNode *N) {
  SDValue LHS = GetSoftenedFloat(N->getOperand(1));
  SDValue RHS = GetSoftenedFloat(N->getOperand(2));
  return DAG.getNode(ISD::SELECT, LHS.getValueType(), N->getOperand(0),LHS,RHS);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_SELECT_CC(SDNode *N) {
  SDValue LHS = GetSoftenedFloat(N->getOperand(2));
  SDValue RHS = GetSoftenedFloat(N->getOperand(3));
  return DAG.getNode(ISD::SELECT_CC, LHS.getValueType(), N->getOperand(0),
                     N->getOperand(1), LHS, RHS, N->getOperand(4));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_SINT_TO_FP(SDNode *N) {
  SDValue Op = N->getOperand(0);
  MVT RVT = N->getValueType(0);
  RTLIB::Libcall LC = RTLIB::getSINTTOFP(Op.getValueType(), RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported SINT_TO_FP!");
  return MakeLibCall(LC, TLI.getTypeToTransformTo(RVT), &Op, 1, false);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_UINT_TO_FP(SDNode *N) {
  SDValue Op = N->getOperand(0);
  MVT RVT = N->getValueType(0);
  RTLIB::Libcall LC = RTLIB::getUINTTOFP(Op.getValueType(), RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported UINT_TO_FP!");
  return MakeLibCall(LC, TLI.getTypeToTransformTo(RVT), &Op, 1, false);
}


//===----------------------------------------------------------------------===//
//  Operand Float to Integer Conversion..
//===----------------------------------------------------------------------===//

bool DAGTypeLegalizer::SoftenFloatOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Soften float operand " << OpNo << ": "; N->dump(&DAG);
        cerr << "\n");
  SDValue Res = SDValue();

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "SoftenFloatOperand Op #" << OpNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to soften this operator's operand!");
    abort();

  case ISD::BIT_CONVERT: Res = SoftenFloatOp_BIT_CONVERT(N); break;
  case ISD::BR_CC:       Res = SoftenFloatOp_BR_CC(N); break;
  case ISD::FP_ROUND:    Res = SoftenFloatOp_FP_ROUND(N); break;
  case ISD::FP_TO_SINT:  Res = SoftenFloatOp_FP_TO_SINT(N); break;
  case ISD::FP_TO_UINT:  Res = SoftenFloatOp_FP_TO_UINT(N); break;
  case ISD::SELECT_CC:   Res = SoftenFloatOp_SELECT_CC(N); break;
  case ISD::SETCC:       Res = SoftenFloatOp_SETCC(N); break;
  case ISD::STORE:       Res = SoftenFloatOp_STORE(N, OpNo); break;
  }

  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.getNode()) return false;

  // If the result is N, the sub-method updated N in place.  Check to see if any
  // operands are new, and if so, mark them.
  if (Res.getNode() == N) {
    // Mark N as new and remark N and its operands.  This allows us to correctly
    // revisit N if it needs another step of promotion and allows us to visit
    // any new operands to N.
    ReanalyzeNode(N);
    return true;
  }

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");

  ReplaceValueWith(SDValue(N, 0), Res);
  return false;
}

/// SoftenSetCCOperands - Soften the operands of a comparison.  This code is
/// shared among BR_CC, SELECT_CC, and SETCC handlers.
void DAGTypeLegalizer::SoftenSetCCOperands(SDValue &NewLHS, SDValue &NewRHS,
                                           ISD::CondCode &CCCode) {
  SDValue LHSInt = GetSoftenedFloat(NewLHS);
  SDValue RHSInt = GetSoftenedFloat(NewRHS);
  MVT VT = NewLHS.getValueType();

  assert((VT == MVT::f32 || VT == MVT::f64) && "Unsupported setcc type!");

  // Expand into one or more soft-fp libcall(s).
  RTLIB::Libcall LC1 = RTLIB::UNKNOWN_LIBCALL, LC2 = RTLIB::UNKNOWN_LIBCALL;
  switch (CCCode) {
  case ISD::SETEQ:
  case ISD::SETOEQ:
    LC1 = (VT == MVT::f32) ? RTLIB::OEQ_F32 : RTLIB::OEQ_F64;
    break;
  case ISD::SETNE:
  case ISD::SETUNE:
    LC1 = (VT == MVT::f32) ? RTLIB::UNE_F32 : RTLIB::UNE_F64;
    break;
  case ISD::SETGE:
  case ISD::SETOGE:
    LC1 = (VT == MVT::f32) ? RTLIB::OGE_F32 : RTLIB::OGE_F64;
    break;
  case ISD::SETLT:
  case ISD::SETOLT:
    LC1 = (VT == MVT::f32) ? RTLIB::OLT_F32 : RTLIB::OLT_F64;
    break;
  case ISD::SETLE:
  case ISD::SETOLE:
    LC1 = (VT == MVT::f32) ? RTLIB::OLE_F32 : RTLIB::OLE_F64;
    break;
  case ISD::SETGT:
  case ISD::SETOGT:
    LC1 = (VT == MVT::f32) ? RTLIB::OGT_F32 : RTLIB::OGT_F64;
    break;
  case ISD::SETUO:
    LC1 = (VT == MVT::f32) ? RTLIB::UO_F32 : RTLIB::UO_F64;
    break;
  case ISD::SETO:
    LC1 = (VT == MVT::f32) ? RTLIB::O_F32 : RTLIB::O_F64;
    break;
  default:
    LC1 = (VT == MVT::f32) ? RTLIB::UO_F32 : RTLIB::UO_F64;
    switch (CCCode) {
    case ISD::SETONE:
      // SETONE = SETOLT | SETOGT
      LC1 = (VT == MVT::f32) ? RTLIB::OLT_F32 : RTLIB::OLT_F64;
      // Fallthrough
    case ISD::SETUGT:
      LC2 = (VT == MVT::f32) ? RTLIB::OGT_F32 : RTLIB::OGT_F64;
      break;
    case ISD::SETUGE:
      LC2 = (VT == MVT::f32) ? RTLIB::OGE_F32 : RTLIB::OGE_F64;
      break;
    case ISD::SETULT:
      LC2 = (VT == MVT::f32) ? RTLIB::OLT_F32 : RTLIB::OLT_F64;
      break;
    case ISD::SETULE:
      LC2 = (VT == MVT::f32) ? RTLIB::OLE_F32 : RTLIB::OLE_F64;
      break;
    case ISD::SETUEQ:
      LC2 = (VT == MVT::f32) ? RTLIB::OEQ_F32 : RTLIB::OEQ_F64;
      break;
    default: assert(false && "Do not know how to soften this setcc!");
    }
  }

  MVT RetVT = MVT::i32; // FIXME: is this the correct return type?
  SDValue Ops[2] = { LHSInt, RHSInt };
  NewLHS = MakeLibCall(LC1, RetVT, Ops, 2, false/*sign irrelevant*/);
  NewRHS = DAG.getConstant(0, RetVT);
  CCCode = TLI.getCmpLibcallCC(LC1);
  if (LC2 != RTLIB::UNKNOWN_LIBCALL) {
    SDValue Tmp = DAG.getNode(ISD::SETCC, TLI.getSetCCResultType(NewLHS),
                                NewLHS, NewRHS, DAG.getCondCode(CCCode));
    NewLHS = MakeLibCall(LC2, RetVT, Ops, 2, false/*sign irrelevant*/);
    NewLHS = DAG.getNode(ISD::SETCC, TLI.getSetCCResultType(NewLHS), NewLHS,
                         NewRHS, DAG.getCondCode(TLI.getCmpLibcallCC(LC2)));
    NewLHS = DAG.getNode(ISD::OR, Tmp.getValueType(), Tmp, NewLHS);
    NewRHS = SDValue();
  }
}

SDValue DAGTypeLegalizer::SoftenFloatOp_BIT_CONVERT(SDNode *N) {
  return DAG.getNode(ISD::BIT_CONVERT, N->getValueType(0),
                     GetSoftenedFloat(N->getOperand(0)));
}

SDValue DAGTypeLegalizer::SoftenFloatOp_FP_ROUND(SDNode *N) {
  MVT SVT = N->getOperand(0).getValueType();
  MVT RVT = N->getValueType(0);

  RTLIB::Libcall LC = RTLIB::getFPROUND(SVT, RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_ROUND libcall");

  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(LC, RVT, &Op, 1, false);
}

SDValue DAGTypeLegalizer::SoftenFloatOp_BR_CC(SDNode *N) {
  SDValue NewLHS = N->getOperand(2), NewRHS = N->getOperand(3);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(1))->get();
  SoftenSetCCOperands(NewLHS, NewRHS, CCCode);

  // If SoftenSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.getNode() == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDValue(N, 0), N->getOperand(0),
                                DAG.getCondCode(CCCode), NewLHS, NewRHS,
                                N->getOperand(4));
}

SDValue DAGTypeLegalizer::SoftenFloatOp_FP_TO_SINT(SDNode *N) {
  MVT RVT = N->getValueType(0);
  RTLIB::Libcall LC = RTLIB::getFPTOSINT(N->getOperand(0).getValueType(), RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_TO_SINT!");
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(LC, RVT, &Op, 1, false);
}

SDValue DAGTypeLegalizer::SoftenFloatOp_FP_TO_UINT(SDNode *N) {
  MVT RVT = N->getValueType(0);
  RTLIB::Libcall LC = RTLIB::getFPTOUINT(N->getOperand(0).getValueType(), RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_TO_UINT!");
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(LC, RVT, &Op, 1, false);
}

SDValue DAGTypeLegalizer::SoftenFloatOp_SELECT_CC(SDNode *N) {
  SDValue NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(4))->get();
  SoftenSetCCOperands(NewLHS, NewRHS, CCCode);

  // If SoftenSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.getNode() == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDValue(N, 0), NewLHS, NewRHS,
                                N->getOperand(2), N->getOperand(3),
                                DAG.getCondCode(CCCode));
}

SDValue DAGTypeLegalizer::SoftenFloatOp_SETCC(SDNode *N) {
  SDValue NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(2))->get();
  SoftenSetCCOperands(NewLHS, NewRHS, CCCode);

  // If SoftenSetCCOperands returned a scalar, use it.
  if (NewRHS.getNode() == 0) {
    assert(NewLHS.getValueType() == N->getValueType(0) &&
           "Unexpected setcc expansion!");
    return NewLHS;
  }

  // Otherwise, update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDValue(N, 0), NewLHS, NewRHS,
                                DAG.getCondCode(CCCode));
}

SDValue DAGTypeLegalizer::SoftenFloatOp_STORE(SDNode *N, unsigned OpNo) {
  assert(ISD::isUNINDEXEDStore(N) && "Indexed store during type legalization!");
  assert(OpNo == 1 && "Can only soften the stored value!");
  StoreSDNode *ST = cast<StoreSDNode>(N);
  SDValue Val = ST->getValue();

  if (ST->isTruncatingStore())
    // Do an FP_ROUND followed by a non-truncating store.
    Val = BitConvertToInteger(DAG.getNode(ISD::FP_ROUND, ST->getMemoryVT(),
                                          Val, DAG.getIntPtrConstant(0)));
  else
    Val = GetSoftenedFloat(Val);

  return DAG.getStore(ST->getChain(), Val, ST->getBasePtr(),
                      ST->getSrcValue(), ST->getSrcValueOffset(),
                      ST->isVolatile(), ST->getAlignment());
}


//===----------------------------------------------------------------------===//
//  Float Result Expansion
//===----------------------------------------------------------------------===//

/// ExpandFloatResult - This method is called when the specified result of the
/// specified node is found to need expansion.  At this point, the node may also
/// have invalid operands or may have other results that need promotion, we just
/// know that (at least) one result needs expansion.
void DAGTypeLegalizer::ExpandFloatResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Expand float result: "; N->dump(&DAG); cerr << "\n");
  SDValue Lo, Hi;
  Lo = Hi = SDValue();

  // See if the target wants to custom expand this node.
  if (TLI.getOperationAction(N->getOpcode(), N->getValueType(ResNo)) ==
      TargetLowering::Custom) {
    // If the target wants to, allow it to lower this itself.
    if (SDNode *P = TLI.ReplaceNodeResults(N, DAG)) {
      // Everything that once used N now uses P.  We are guaranteed that the
      // result value types of N and the result value types of P match.
      ReplaceNodeWith(N, P);
      return;
    }
  }

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "ExpandFloatResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to expand the result of this operator!");
    abort();

  case ISD::MERGE_VALUES: SplitRes_MERGE_VALUES(N, Lo, Hi); break;
  case ISD::UNDEF:        SplitRes_UNDEF(N, Lo, Hi); break;
  case ISD::SELECT:       SplitRes_SELECT(N, Lo, Hi); break;
  case ISD::SELECT_CC:    SplitRes_SELECT_CC(N, Lo, Hi); break;

  case ISD::BIT_CONVERT:        ExpandRes_BIT_CONVERT(N, Lo, Hi); break;
  case ISD::BUILD_PAIR:         ExpandRes_BUILD_PAIR(N, Lo, Hi); break;
  case ISD::EXTRACT_ELEMENT:    ExpandRes_EXTRACT_ELEMENT(N, Lo, Hi); break;
  case ISD::EXTRACT_VECTOR_ELT: ExpandRes_EXTRACT_VECTOR_ELT(N, Lo, Hi); break;

  case ISD::ConstantFP: ExpandFloatRes_ConstantFP(N, Lo, Hi); break;
  case ISD::FABS:       ExpandFloatRes_FABS(N, Lo, Hi); break;
  case ISD::FADD:       ExpandFloatRes_FADD(N, Lo, Hi); break;
  case ISD::FDIV:       ExpandFloatRes_FDIV(N, Lo, Hi); break;
  case ISD::FMUL:       ExpandFloatRes_FMUL(N, Lo, Hi); break;
  case ISD::FNEG:       ExpandFloatRes_FNEG(N, Lo, Hi); break;
  case ISD::FP_EXTEND:  ExpandFloatRes_FP_EXTEND(N, Lo, Hi); break;
  case ISD::FSUB:       ExpandFloatRes_FSUB(N, Lo, Hi); break;
  case ISD::LOAD:       ExpandFloatRes_LOAD(N, Lo, Hi); break;
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: ExpandFloatRes_XINT_TO_FP(N, Lo, Hi); break;
  }

  // If Lo/Hi is null, the sub-method took care of registering results etc.
  if (Lo.getNode())
    SetExpandedFloat(SDValue(N, ResNo), Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_ConstantFP(SDNode *N, SDValue &Lo,
                                                 SDValue &Hi) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  assert(NVT.getSizeInBits() == integerPartWidth &&
         "Do not know how to expand this float constant!");
  APInt C = cast<ConstantFPSDNode>(N)->getValueAPF().bitcastToAPInt();
  Lo = DAG.getConstantFP(APFloat(APInt(integerPartWidth, 1,
                                       &C.getRawData()[1])), NVT);
  Hi = DAG.getConstantFP(APFloat(APInt(integerPartWidth, 1,
                                       &C.getRawData()[0])), NVT);
}

void DAGTypeLegalizer::ExpandFloatRes_FADD(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SDValue Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                            RTLIB::ADD_F32,
                                            RTLIB::ADD_F64,
                                            RTLIB::ADD_F80,
                                            RTLIB::ADD_PPCF128),
                               N->getValueType(0), Ops, 2,
                               false);
  assert(Call.getNode()->getOpcode() == ISD::BUILD_PAIR &&
         "Call lowered wrongly!");
  Lo = Call.getOperand(0); Hi = Call.getOperand(1);
}

void DAGTypeLegalizer::ExpandFloatRes_FABS(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  assert(N->getValueType(0) == MVT::ppcf128 &&
         "Logic only correct for ppcf128!");
  SDValue Tmp;
  GetExpandedFloat(N->getOperand(0), Lo, Tmp);
  Hi = DAG.getNode(ISD::FABS, Tmp.getValueType(), Tmp);
  // Lo = Hi==fabs(Hi) ? Lo : -Lo;
  Lo = DAG.getNode(ISD::SELECT_CC, Lo.getValueType(), Tmp, Hi, Lo,
                   DAG.getNode(ISD::FNEG, Lo.getValueType(), Lo),
                   DAG.getCondCode(ISD::SETEQ));
}

void DAGTypeLegalizer::ExpandFloatRes_FDIV(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SDValue Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                            RTLIB::DIV_F32,
                                            RTLIB::DIV_F64,
                                            RTLIB::DIV_F80,
                                            RTLIB::DIV_PPCF128),
                               N->getValueType(0), Ops, 2,
                               false);
  assert(Call.getNode()->getOpcode() == ISD::BUILD_PAIR &&
         "Call lowered wrongly!");
  Lo = Call.getOperand(0); Hi = Call.getOperand(1);
}

void DAGTypeLegalizer::ExpandFloatRes_FMUL(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SDValue Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                            RTLIB::MUL_F32,
                                            RTLIB::MUL_F64,
                                            RTLIB::MUL_F80,
                                            RTLIB::MUL_PPCF128),
                               N->getValueType(0), Ops, 2,
                               false);
  assert(Call.getNode()->getOpcode() == ISD::BUILD_PAIR &&
         "Call lowered wrongly!");
  Lo = Call.getOperand(0); Hi = Call.getOperand(1);
}

void DAGTypeLegalizer::ExpandFloatRes_FNEG(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  GetExpandedFloat(N->getOperand(0), Lo, Hi);
  Lo = DAG.getNode(ISD::FNEG, Lo.getValueType(), Lo);
  Hi = DAG.getNode(ISD::FNEG, Hi.getValueType(), Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FP_EXTEND(SDNode *N, SDValue &Lo,
                                                SDValue &Hi) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  Hi = DAG.getNode(ISD::FP_EXTEND, NVT, N->getOperand(0));
  Lo = DAG.getConstantFP(APFloat(APInt(NVT.getSizeInBits(), 0)), NVT);
}

void DAGTypeLegalizer::ExpandFloatRes_FSUB(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SDValue Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                            RTLIB::SUB_F32,
                                            RTLIB::SUB_F64,
                                            RTLIB::SUB_F80,
                                            RTLIB::SUB_PPCF128),
                               N->getValueType(0), Ops, 2,
                               false);
  assert(Call.getNode()->getOpcode() == ISD::BUILD_PAIR &&
         "Call lowered wrongly!");
  Lo = Call.getOperand(0); Hi = Call.getOperand(1);
}

void DAGTypeLegalizer::ExpandFloatRes_LOAD(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  if (ISD::isNormalLoad(N)) {
    ExpandRes_NormalLoad(N, Lo, Hi);
    return;
  }

  assert(ISD::isUNINDEXEDLoad(N) && "Indexed load during type legalization!");
  LoadSDNode *LD = cast<LoadSDNode>(N);
  SDValue Chain = LD->getChain();
  SDValue Ptr = LD->getBasePtr();

  MVT NVT = TLI.getTypeToTransformTo(LD->getValueType(0));
  assert(NVT.isByteSized() && "Expanded type not byte sized!");
  assert(LD->getMemoryVT().bitsLE(NVT) && "Float type not round?");

  Lo = DAG.getExtLoad(LD->getExtensionType(), NVT, Chain, Ptr,
                      LD->getSrcValue(), LD->getSrcValueOffset(),
                      LD->getMemoryVT(),
                      LD->isVolatile(), LD->getAlignment());

  // Remember the chain.
  Chain = Lo.getValue(1);

  // The high part is undefined.
  Hi = DAG.getNode(ISD::UNDEF, NVT);

  // Modified the chain - switch anything that used the old chain to use the
  // new one.
  ReplaceValueWith(SDValue(LD, 1), Chain);
}

void DAGTypeLegalizer::ExpandFloatRes_XINT_TO_FP(SDNode *N, SDValue &Lo,
                                                 SDValue &Hi) {
  assert(N->getValueType(0) == MVT::ppcf128 && "Unsupported XINT_TO_FP!");
  MVT VT = N->getValueType(0);
  MVT NVT = TLI.getTypeToTransformTo(VT);
  SDValue Src = N->getOperand(0);
  MVT SrcVT = Src.getValueType();

  // First do an SINT_TO_FP, whether the original was signed or unsigned.
  if (SrcVT.bitsLE(MVT::i32)) {
    // The integer can be represented exactly in an f64.
    Src = DAG.getNode(ISD::SIGN_EXTEND, MVT::i32, Src);
    Lo = DAG.getConstantFP(APFloat(APInt(NVT.getSizeInBits(), 0)), NVT);
    Hi = DAG.getNode(ISD::SINT_TO_FP, NVT, Src);
  } else {
    RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
    if (SrcVT.bitsLE(MVT::i64)) {
      Src = DAG.getNode(ISD::SIGN_EXTEND, MVT::i64, Src);
      LC = RTLIB::SINTTOFP_I64_PPCF128;
    } else if (SrcVT.bitsLE(MVT::i128)) {
      Src = DAG.getNode(ISD::SIGN_EXTEND, MVT::i128, Src);
      LC = RTLIB::SINTTOFP_I128_PPCF128;
    }
    assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported XINT_TO_FP!");

    Hi = MakeLibCall(LC, VT, &Src, 1, true);
    assert(Hi.getNode()->getOpcode() == ISD::BUILD_PAIR &&
           "Call lowered wrongly!");
    Lo = Hi.getOperand(0); Hi = Hi.getOperand(1);
  }

  if (N->getOpcode() == ISD::SINT_TO_FP)
    return;

  // Unsigned - fix up the SINT_TO_FP value just calculated.
  Hi = DAG.getNode(ISD::BUILD_PAIR, VT, Lo, Hi);
  SrcVT = Src.getValueType();

  // x>=0 ? (ppcf128)(iN)x : (ppcf128)(iN)x + 2^N; N=32,64,128.
  static const uint64_t TwoE32[]  = { 0x41f0000000000000LL, 0 };
  static const uint64_t TwoE64[]  = { 0x43f0000000000000LL, 0 };
  static const uint64_t TwoE128[] = { 0x47f0000000000000LL, 0 };
  const uint64_t *Parts = 0;

  switch (SrcVT.getSimpleVT()) {
  default:
    assert(false && "Unsupported UINT_TO_FP!");
  case MVT::i32:
    Parts = TwoE32;
  case MVT::i64:
    Parts = TwoE64;
  case MVT::i128:
    Parts = TwoE128;
  }

  Lo = DAG.getNode(ISD::FADD, VT, Hi,
                   DAG.getConstantFP(APFloat(APInt(128, 2, Parts)),
                                     MVT::ppcf128));
  Lo = DAG.getNode(ISD::SELECT_CC, VT, Src, DAG.getConstant(0, SrcVT), Lo, Hi,
                   DAG.getCondCode(ISD::SETLT));
  Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, NVT, Lo, DAG.getIntPtrConstant(1));
  Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, NVT, Lo, DAG.getIntPtrConstant(0));
}


//===----------------------------------------------------------------------===//
//  Float Operand Expansion
//===----------------------------------------------------------------------===//

/// ExpandFloatOperand - This method is called when the specified operand of the
/// specified node is found to need expansion.  At this point, all of the result
/// types of the node are known to be legal, but other operands of the node may
/// need promotion or expansion as well as the specified one.
bool DAGTypeLegalizer::ExpandFloatOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Expand float operand: "; N->dump(&DAG); cerr << "\n");
  SDValue Res = SDValue();

  if (TLI.getOperationAction(N->getOpcode(), N->getOperand(OpNo).getValueType())
      == TargetLowering::Custom)
    Res = TLI.LowerOperation(SDValue(N, 0), DAG);

  if (Res.getNode() == 0) {
    switch (N->getOpcode()) {
    default:
  #ifndef NDEBUG
      cerr << "ExpandFloatOperand Op #" << OpNo << ": ";
      N->dump(&DAG); cerr << "\n";
  #endif
      assert(0 && "Do not know how to expand this operator's operand!");
      abort();

    case ISD::BIT_CONVERT:     Res = ExpandOp_BIT_CONVERT(N); break;
    case ISD::BUILD_VECTOR:    Res = ExpandOp_BUILD_VECTOR(N); break;
    case ISD::EXTRACT_ELEMENT: Res = ExpandOp_EXTRACT_ELEMENT(N); break;

    case ISD::BR_CC:      Res = ExpandFloatOp_BR_CC(N); break;
    case ISD::FP_ROUND:   Res = ExpandFloatOp_FP_ROUND(N); break;
    case ISD::FP_TO_SINT: Res = ExpandFloatOp_FP_TO_SINT(N); break;
    case ISD::FP_TO_UINT: Res = ExpandFloatOp_FP_TO_UINT(N); break;
    case ISD::SELECT_CC:  Res = ExpandFloatOp_SELECT_CC(N); break;
    case ISD::SETCC:      Res = ExpandFloatOp_SETCC(N); break;
    case ISD::STORE:      Res = ExpandFloatOp_STORE(cast<StoreSDNode>(N),
                                                    OpNo); break;
    }
  }

  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.getNode()) return false;
  // If the result is N, the sub-method updated N in place.  Check to see if any
  // operands are new, and if so, mark them.
  if (Res.getNode() == N) {
    // Mark N as new and remark N and its operands.  This allows us to correctly
    // revisit N if it needs another step of expansion and allows us to visit
    // any new operands to N.
    ReanalyzeNode(N);
    return true;
  }

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");

  ReplaceValueWith(SDValue(N, 0), Res);
  return false;
}

/// FloatExpandSetCCOperands - Expand the operands of a comparison.  This code
/// is shared among BR_CC, SELECT_CC, and SETCC handlers.
void DAGTypeLegalizer::FloatExpandSetCCOperands(SDValue &NewLHS,
                                                SDValue &NewRHS,
                                                ISD::CondCode &CCCode) {
  SDValue LHSLo, LHSHi, RHSLo, RHSHi;
  GetExpandedFloat(NewLHS, LHSLo, LHSHi);
  GetExpandedFloat(NewRHS, RHSLo, RHSHi);

  MVT VT = NewLHS.getValueType();
  assert(VT == MVT::ppcf128 && "Unsupported setcc type!");

  // FIXME:  This generated code sucks.  We want to generate
  //         FCMP crN, hi1, hi2
  //         BNE crN, L:
  //         FCMP crN, lo1, lo2
  // The following can be improved, but not that much.
  SDValue Tmp1, Tmp2, Tmp3;
  Tmp1 = DAG.getSetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi, ISD::SETEQ);
  Tmp2 = DAG.getSetCC(TLI.getSetCCResultType(LHSLo), LHSLo, RHSLo, CCCode);
  Tmp3 = DAG.getNode(ISD::AND, Tmp1.getValueType(), Tmp1, Tmp2);
  Tmp1 = DAG.getSetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi, ISD::SETNE);
  Tmp2 = DAG.getSetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi, CCCode);
  Tmp1 = DAG.getNode(ISD::AND, Tmp1.getValueType(), Tmp1, Tmp2);
  NewLHS = DAG.getNode(ISD::OR, Tmp1.getValueType(), Tmp1, Tmp3);
  NewRHS = SDValue();   // LHS is the result, not a compare.
}

SDValue DAGTypeLegalizer::ExpandFloatOp_BR_CC(SDNode *N) {
  SDValue NewLHS = N->getOperand(2), NewRHS = N->getOperand(3);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(1))->get();
  FloatExpandSetCCOperands(NewLHS, NewRHS, CCCode);

  // If ExpandSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.getNode() == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDValue(N, 0), N->getOperand(0),
                                DAG.getCondCode(CCCode), NewLHS, NewRHS,
                                N->getOperand(4));
}

SDValue DAGTypeLegalizer::ExpandFloatOp_FP_ROUND(SDNode *N) {
  assert(N->getOperand(0).getValueType() == MVT::ppcf128 &&
         "Logic only correct for ppcf128!");
  SDValue Lo, Hi;
  GetExpandedFloat(N->getOperand(0), Lo, Hi);
  // Round it the rest of the way (e.g. to f32) if needed.
  return DAG.getNode(ISD::FP_ROUND, N->getValueType(0), Hi, N->getOperand(1));
}

SDValue DAGTypeLegalizer::ExpandFloatOp_FP_TO_SINT(SDNode *N) {
  MVT RVT = N->getValueType(0);

  // Expand ppcf128 to i32 by hand for the benefit of llvm-gcc bootstrap on
  // PPC (the libcall is not available).  FIXME: Do this in a less hacky way.
  if (RVT == MVT::i32) {
    assert(N->getOperand(0).getValueType() == MVT::ppcf128 &&
           "Logic only correct for ppcf128!");
    SDValue Res = DAG.getNode(ISD::FP_ROUND_INREG, MVT::ppcf128,
                              N->getOperand(0), DAG.getValueType(MVT::f64));
    Res = DAG.getNode(ISD::FP_ROUND, MVT::f64, Res, DAG.getIntPtrConstant(1));
    return DAG.getNode(ISD::FP_TO_SINT, MVT::i32, Res);
  }

  RTLIB::Libcall LC = RTLIB::getFPTOSINT(N->getOperand(0).getValueType(), RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_TO_SINT!");
  return MakeLibCall(LC, RVT, &N->getOperand(0), 1, false);
}

SDValue DAGTypeLegalizer::ExpandFloatOp_FP_TO_UINT(SDNode *N) {
  MVT RVT = N->getValueType(0);

  // Expand ppcf128 to i32 by hand for the benefit of llvm-gcc bootstrap on
  // PPC (the libcall is not available).  FIXME: Do this in a less hacky way.
  if (RVT == MVT::i32) {
    assert(N->getOperand(0).getValueType() == MVT::ppcf128 &&
           "Logic only correct for ppcf128!");
    const uint64_t TwoE31[] = {0x41e0000000000000LL, 0};
    APFloat APF = APFloat(APInt(128, 2, TwoE31));
    SDValue Tmp = DAG.getConstantFP(APF, MVT::ppcf128);
    //  X>=2^31 ? (int)(X-2^31)+0x80000000 : (int)X
    // FIXME: generated code sucks.
    return DAG.getNode(ISD::SELECT_CC, MVT::i32, N->getOperand(0), Tmp,
                       DAG.getNode(ISD::ADD, MVT::i32,
                                   DAG.getNode(ISD::FP_TO_SINT, MVT::i32,
                                               DAG.getNode(ISD::FSUB,
                                                           MVT::ppcf128,
                                                           N->getOperand(0),
                                                           Tmp)),
                                   DAG.getConstant(0x80000000, MVT::i32)),
                       DAG.getNode(ISD::FP_TO_SINT, MVT::i32, N->getOperand(0)),
                       DAG.getCondCode(ISD::SETGE));
  }

  RTLIB::Libcall LC = RTLIB::getFPTOUINT(N->getOperand(0).getValueType(), RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_TO_UINT!");
  return MakeLibCall(LC, N->getValueType(0), &N->getOperand(0), 1, false);
}

SDValue DAGTypeLegalizer::ExpandFloatOp_SELECT_CC(SDNode *N) {
  SDValue NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(4))->get();
  FloatExpandSetCCOperands(NewLHS, NewRHS, CCCode);

  // If ExpandSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.getNode() == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDValue(N, 0), NewLHS, NewRHS,
                                N->getOperand(2), N->getOperand(3),
                                DAG.getCondCode(CCCode));
}

SDValue DAGTypeLegalizer::ExpandFloatOp_SETCC(SDNode *N) {
  SDValue NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(2))->get();
  FloatExpandSetCCOperands(NewLHS, NewRHS, CCCode);

  // If ExpandSetCCOperands returned a scalar, use it.
  if (NewRHS.getNode() == 0) {
    assert(NewLHS.getValueType() == N->getValueType(0) &&
           "Unexpected setcc expansion!");
    return NewLHS;
  }

  // Otherwise, update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDValue(N, 0), NewLHS, NewRHS,
                                DAG.getCondCode(CCCode));
}

SDValue DAGTypeLegalizer::ExpandFloatOp_STORE(SDNode *N, unsigned OpNo) {
  if (ISD::isNormalStore(N))
    return ExpandOp_NormalStore(N, OpNo);

  assert(ISD::isUNINDEXEDStore(N) && "Indexed store during type legalization!");
  assert(OpNo == 1 && "Can only expand the stored value so far");
  StoreSDNode *ST = cast<StoreSDNode>(N);

  SDValue Chain = ST->getChain();
  SDValue Ptr = ST->getBasePtr();

  MVT NVT = TLI.getTypeToTransformTo(ST->getValue().getValueType());
  assert(NVT.isByteSized() && "Expanded type not byte sized!");
  assert(ST->getMemoryVT().bitsLE(NVT) && "Float type not round?");

  SDValue Lo, Hi;
  GetExpandedOp(ST->getValue(), Lo, Hi);

  return DAG.getTruncStore(Chain, Lo, Ptr,
                           ST->getSrcValue(), ST->getSrcValueOffset(),
                           ST->getMemoryVT(),
                           ST->isVolatile(), ST->getAlignment());
}
