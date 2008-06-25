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
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
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
    cerr << "SoftenFloatResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to convert the result of this operator!");
    abort();

    case ISD::BIT_CONVERT: R = SoftenFloatRes_BIT_CONVERT(N); break;
    case ISD::BUILD_PAIR:  R = SoftenFloatRes_BUILD_PAIR(N); break;
    case ISD::ConstantFP:
      R = SoftenFloatRes_ConstantFP(cast<ConstantFPSDNode>(N));
      break;
    case ISD::FCOPYSIGN:   R = SoftenFloatRes_FCOPYSIGN(N); break;
    case ISD::LOAD:        R = SoftenFloatRes_LOAD(N); break;
    case ISD::SINT_TO_FP:
    case ISD::UINT_TO_FP:  R = SoftenFloatRes_XINT_TO_FP(N); break;

    case ISD::FADD: R = SoftenFloatRes_FADD(N); break;
    case ISD::FMUL: R = SoftenFloatRes_FMUL(N); break;
    case ISD::FSUB: R = SoftenFloatRes_FSUB(N); break;
  }

  // If R is null, the sub-method took care of registering the result.
  if (R.Val)
    SetSoftenedFloat(SDOperand(N, ResNo), R);
}

SDOperand DAGTypeLegalizer::SoftenFloatRes_BIT_CONVERT(SDNode *N) {
  return BitConvertToInteger(N->getOperand(0));
}

SDOperand DAGTypeLegalizer::SoftenFloatRes_BUILD_PAIR(SDNode *N) {
  // Convert the inputs to integers, and build a new pair out of them.
  return DAG.getNode(ISD::BUILD_PAIR,
                     TLI.getTypeToTransformTo(N->getValueType(0)),
                     BitConvertToInteger(N->getOperand(0)),
                     BitConvertToInteger(N->getOperand(1)));
}

SDOperand DAGTypeLegalizer::SoftenFloatRes_ConstantFP(ConstantFPSDNode *N) {
  return DAG.getConstant(N->getValueAPF().convertToAPInt(),
                         TLI.getTypeToTransformTo(N->getValueType(0)));
}

SDOperand DAGTypeLegalizer::SoftenFloatRes_FADD(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDOperand Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                       GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::ADD_F32,
                                  RTLIB::ADD_F64,
                                  RTLIB::ADD_F80,
                                  RTLIB::ADD_PPCF128),
                     NVT, Ops, 2, false);
}

SDOperand DAGTypeLegalizer::SoftenFloatRes_FCOPYSIGN(SDNode *N) {
  SDOperand LHS = GetSoftenedFloat(N->getOperand(0));
  SDOperand RHS = BitConvertToInteger(N->getOperand(1));

  MVT LVT = LHS.getValueType();
  MVT RVT = RHS.getValueType();

  unsigned LSize = LVT.getSizeInBits();
  unsigned RSize = RVT.getSizeInBits();

  // First get the sign bit of second operand.
  SDOperand SignBit = DAG.getNode(ISD::SHL, RVT, DAG.getConstant(1, RVT),
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
  SDOperand Mask = DAG.getNode(ISD::SHL, LVT, DAG.getConstant(1, LVT),
                               DAG.getConstant(LSize - 1,
                                               TLI.getShiftAmountTy()));
  Mask = DAG.getNode(ISD::SUB, LVT, Mask, DAG.getConstant(1, LVT));
  LHS = DAG.getNode(ISD::AND, LVT, LHS, Mask);

  // Or the value with the sign bit.
  return DAG.getNode(ISD::OR, LVT, LHS, SignBit);
}

SDOperand DAGTypeLegalizer::SoftenFloatRes_FMUL(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDOperand Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                       GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::MUL_F32,
                                  RTLIB::MUL_F64,
                                  RTLIB::MUL_F80,
                                  RTLIB::MUL_PPCF128),
                     NVT, Ops, 2, false);
}

SDOperand DAGTypeLegalizer::SoftenFloatRes_FSUB(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDOperand Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                       GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::SUB_F32,
                                  RTLIB::SUB_F64,
                                  RTLIB::SUB_F80,
                                  RTLIB::SUB_PPCF128),
                     NVT, Ops, 2, false);
}

SDOperand DAGTypeLegalizer::SoftenFloatRes_LOAD(SDNode *N) {
  LoadSDNode *L = cast<LoadSDNode>(N);
  MVT VT = N->getValueType(0);
  MVT NVT = TLI.getTypeToTransformTo(VT);

  if (L->getExtensionType() == ISD::NON_EXTLOAD)
     return DAG.getLoad(L->getAddressingMode(), L->getExtensionType(),
                        NVT, L->getChain(), L->getBasePtr(), L->getOffset(),
                        L->getSrcValue(), L->getSrcValueOffset(), NVT,
                        L->isVolatile(), L->getAlignment());

  // Do a non-extending load followed by FP_EXTEND.
  SDOperand NL = DAG.getLoad(L->getAddressingMode(), ISD::NON_EXTLOAD,
                             L->getMemoryVT(), L->getChain(),
                             L->getBasePtr(), L->getOffset(),
                             L->getSrcValue(), L->getSrcValueOffset(),
                             L->getMemoryVT(),
                             L->isVolatile(), L->getAlignment());
  return BitConvertToInteger(DAG.getNode(ISD::FP_EXTEND, VT, NL));
}

SDOperand DAGTypeLegalizer::SoftenFloatRes_XINT_TO_FP(SDNode *N) {
  bool isSigned = N->getOpcode() == ISD::SINT_TO_FP;
  MVT DestVT = N->getValueType(0);
  SDOperand Op = N->getOperand(0);

  if (Op.getValueType() == MVT::i32) {
    // simple 32-bit [signed|unsigned] integer to float/double expansion

    // Get the stack frame index of a 8 byte buffer.
    SDOperand StackSlot = DAG.CreateStackTemporary(MVT::f64);

    // word offset constant for Hi/Lo address computation
    SDOperand Offset =
      DAG.getConstant(MVT(MVT::i32).getSizeInBits() / 8,
                      TLI.getPointerTy());
    // set up Hi and Lo (into buffer) address based on endian
    SDOperand Hi = StackSlot;
    SDOperand Lo = DAG.getNode(ISD::ADD, TLI.getPointerTy(), StackSlot, Offset);
    if (TLI.isLittleEndian())
      std::swap(Hi, Lo);

    // if signed map to unsigned space
    SDOperand OpMapped;
    if (isSigned) {
      // constant used to invert sign bit (signed to unsigned mapping)
      SDOperand SignBit = DAG.getConstant(0x80000000u, MVT::i32);
      OpMapped = DAG.getNode(ISD::XOR, MVT::i32, Op, SignBit);
    } else {
      OpMapped = Op;
    }
    // store the lo of the constructed double - based on integer input
    SDOperand Store1 = DAG.getStore(DAG.getEntryNode(),
                                    OpMapped, Lo, NULL, 0);
    // initial hi portion of constructed double
    SDOperand InitialHi = DAG.getConstant(0x43300000u, MVT::i32);
    // store the hi of the constructed double - biased exponent
    SDOperand Store2=DAG.getStore(Store1, InitialHi, Hi, NULL, 0);
    // load the constructed double
    SDOperand Load = DAG.getLoad(MVT::f64, Store2, StackSlot, NULL, 0);
    // FP constant to bias correct the final result
    SDOperand Bias = DAG.getConstantFP(isSigned ?
                                            BitsToDouble(0x4330000080000000ULL)
                                          : BitsToDouble(0x4330000000000000ULL),
                                     MVT::f64);
    // subtract the bias
    SDOperand Sub = DAG.getNode(ISD::FSUB, MVT::f64, Load, Bias);
    // final result
    SDOperand Result;
    // handle final rounding
    if (DestVT == MVT::f64) {
      // do nothing
      Result = Sub;
    } else if (DestVT.bitsLT(MVT::f64)) {
      Result = DAG.getNode(ISD::FP_ROUND, DestVT, Sub,
                           DAG.getIntPtrConstant(0));
    } else if (DestVT.bitsGT(MVT::f64)) {
      Result = DAG.getNode(ISD::FP_EXTEND, DestVT, Sub);
    }
    return BitConvertToInteger(Result);
  }
  assert(!isSigned && "Legalize cannot Expand SINT_TO_FP for i64 yet");
  SDOperand Tmp1 = DAG.getNode(ISD::SINT_TO_FP, DestVT, Op);

  SDOperand SignSet = DAG.getSetCC(TLI.getSetCCResultType(Op), Op,
                                   DAG.getConstant(0, Op.getValueType()),
                                   ISD::SETLT);
  SDOperand Zero = DAG.getIntPtrConstant(0), Four = DAG.getIntPtrConstant(4);
  SDOperand CstOffset = DAG.getNode(ISD::SELECT, Zero.getValueType(),
                                    SignSet, Four, Zero);

  // If the sign bit of the integer is set, the large number will be treated
  // as a negative number.  To counteract this, the dynamic code adds an
  // offset depending on the data type.
  uint64_t FF;
  switch (Op.getValueType().getSimpleVT()) {
  default: assert(0 && "Unsupported integer type!");
  case MVT::i8 : FF = 0x43800000ULL; break;  // 2^8  (as a float)
  case MVT::i16: FF = 0x47800000ULL; break;  // 2^16 (as a float)
  case MVT::i32: FF = 0x4F800000ULL; break;  // 2^32 (as a float)
  case MVT::i64: FF = 0x5F800000ULL; break;  // 2^64 (as a float)
  }
  if (TLI.isLittleEndian()) FF <<= 32;
  static Constant *FudgeFactor = ConstantInt::get(Type::Int64Ty, FF);

  SDOperand CPIdx = DAG.getConstantPool(FudgeFactor, TLI.getPointerTy());
  CPIdx = DAG.getNode(ISD::ADD, TLI.getPointerTy(), CPIdx, CstOffset);
  SDOperand FudgeInReg;
  if (DestVT == MVT::f32)
    FudgeInReg = DAG.getLoad(MVT::f32, DAG.getEntryNode(), CPIdx,
                             PseudoSourceValue::getConstantPool(), 0);
  else {
    FudgeInReg = DAG.getExtLoad(ISD::EXTLOAD, DestVT,
                                DAG.getEntryNode(), CPIdx,
                                PseudoSourceValue::getConstantPool(), 0,
                                MVT::f32);
  }

  return BitConvertToInteger(DAG.getNode(ISD::FADD, DestVT, Tmp1, FudgeInReg));
}


//===----------------------------------------------------------------------===//
//  Operand Float to Integer Conversion..
//===----------------------------------------------------------------------===//

bool DAGTypeLegalizer::SoftenFloatOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Soften float operand " << OpNo << ": "; N->dump(&DAG);
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
      cerr << "SoftenFloatOperand Op #" << OpNo << ": ";
      N->dump(&DAG); cerr << "\n";
#endif
      assert(0 && "Do not know how to convert this operator's operand!");
      abort();

    case ISD::BIT_CONVERT: Res = SoftenFloatOp_BIT_CONVERT(N); break;

    case ISD::BR_CC:     Res = SoftenFloatOp_BR_CC(N); break;
    case ISD::SELECT_CC: Res = SoftenFloatOp_BR_CC(N); break;
    case ISD::SETCC:     Res = SoftenFloatOp_BR_CC(N); break;
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

/// SoftenSetCCOperands - Soften the operands of a comparison.  This code is
/// shared among BR_CC, SELECT_CC, and SETCC handlers.
void DAGTypeLegalizer::SoftenSetCCOperands(SDOperand &NewLHS, SDOperand &NewRHS,
                                           ISD::CondCode &CCCode) {
  SDOperand LHSInt = GetSoftenedFloat(NewLHS);
  SDOperand RHSInt = GetSoftenedFloat(NewRHS);
  MVT VT  = NewLHS.getValueType();
  MVT NVT = LHSInt.getValueType();

  assert((VT == MVT::f32 || VT == MVT::f64) && "Unsupported setcc type!");

  // Expand into one or more soft-fp libcall(s).
  RTLIB::Libcall LC1, LC2 = RTLIB::UNKNOWN_LIBCALL;
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

  SDOperand Ops[2] = { LHSInt, RHSInt };
  NewLHS = MakeLibCall(LC1, NVT, Ops, 2, false/*sign irrelevant*/);
  NewRHS = DAG.getConstant(0, NVT);
  if (LC2 != RTLIB::UNKNOWN_LIBCALL) {
    SDOperand Tmp = DAG.getNode(ISD::SETCC, TLI.getSetCCResultType(NewLHS),
                                NewLHS, NewRHS,
                                DAG.getCondCode(TLI.getCmpLibcallCC(LC1)));
    NewLHS = MakeLibCall(LC2, NVT, Ops, 2, false/*sign irrelevant*/);
    NewLHS = DAG.getNode(ISD::SETCC, TLI.getSetCCResultType(NewLHS), NewLHS,
                         NewRHS, DAG.getCondCode(TLI.getCmpLibcallCC(LC2)));
    NewLHS = DAG.getNode(ISD::OR, Tmp.getValueType(), Tmp, NewLHS);
    NewRHS = SDOperand();
  }
}

SDOperand DAGTypeLegalizer::SoftenFloatOp_BIT_CONVERT(SDNode *N) {
  return DAG.getNode(ISD::BIT_CONVERT, N->getValueType(0),
                     GetSoftenedFloat(N->getOperand(0)));
}

SDOperand DAGTypeLegalizer::SoftenFloatOp_BR_CC(SDNode *N) {
  SDOperand NewLHS = N->getOperand(2), NewRHS = N->getOperand(3);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(1))->get();
  SoftenSetCCOperands(NewLHS, NewRHS, CCCode);

  // If SoftenSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.Val == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), N->getOperand(0),
                                DAG.getCondCode(CCCode), NewLHS, NewRHS,
                                N->getOperand(4));
}

SDOperand DAGTypeLegalizer::SoftenFloatOp_SELECT_CC(SDNode *N) {
  SDOperand NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(4))->get();
  SoftenSetCCOperands(NewLHS, NewRHS, CCCode);

  // If SoftenSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.Val == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), NewLHS, NewRHS,
                                N->getOperand(2), N->getOperand(3),
                                DAG.getCondCode(CCCode));
}

SDOperand DAGTypeLegalizer::SoftenFloatOp_SETCC(SDNode *N) {
  SDOperand NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(2))->get();
  SoftenSetCCOperands(NewLHS, NewRHS, CCCode);

  // If SoftenSetCCOperands returned a scalar, use it.
  if (NewRHS.Val == 0) {
    assert(NewLHS.getValueType() == N->getValueType(0) &&
           "Unexpected setcc expansion!");
    return NewLHS;
  }

  // Otherwise, update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), NewLHS, NewRHS,
                                DAG.getCondCode(CCCode));
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
  SDOperand Lo, Hi;
  Lo = Hi = SDOperand();

  // See if the target wants to custom expand this node.
  if (TLI.getOperationAction(N->getOpcode(), N->getValueType(0)) ==
          TargetLowering::Custom) {
    // If the target wants to, allow it to lower this itself.
    if (SDNode *P = TLI.ExpandOperationResult(N, DAG)) {
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
  case ISD::FADD:       ExpandFloatRes_FADD(N, Lo, Hi); break;
  case ISD::FDIV:       ExpandFloatRes_FDIV(N, Lo, Hi); break;
  case ISD::FMUL:       ExpandFloatRes_FMUL(N, Lo, Hi); break;
  case ISD::FSUB:       ExpandFloatRes_FSUB(N, Lo, Hi); break;
  case ISD::LOAD:       ExpandFloatRes_LOAD(N, Lo, Hi); break;
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: ExpandFloatRes_XINT_TO_FP(N, Lo, Hi); break;
  }

  // If Lo/Hi is null, the sub-method took care of registering results etc.
  if (Lo.Val)
    SetExpandedFloat(SDOperand(N, ResNo), Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_ConstantFP(SDNode *N, SDOperand &Lo,
                                                 SDOperand &Hi) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  assert(NVT.getSizeInBits() == integerPartWidth &&
         "Do not know how to expand this float constant!");
  APInt C = cast<ConstantFPSDNode>(N)->getValueAPF().convertToAPInt();
  Lo = DAG.getConstantFP(APFloat(APInt(integerPartWidth, 1,
                                       &C.getRawData()[1])), NVT);
  Hi = DAG.getConstantFP(APFloat(APInt(integerPartWidth, 1,
                                       &C.getRawData()[0])), NVT);
}

void DAGTypeLegalizer::ExpandFloatRes_FADD(SDNode *N, SDOperand &Lo,
                                           SDOperand &Hi) {
  SDOperand Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SDOperand Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                            RTLIB::ADD_F32,
                                            RTLIB::ADD_F64,
                                            RTLIB::ADD_F80,
                                            RTLIB::ADD_PPCF128),
                               N->getValueType(0), Ops, 2,
                               false);
  assert(Call.Val->getOpcode() == ISD::BUILD_PAIR && "Call lowered wrongly!");
  Lo = Call.getOperand(0); Hi = Call.getOperand(1);
}

void DAGTypeLegalizer::ExpandFloatRes_FDIV(SDNode *N, SDOperand &Lo,
                                           SDOperand &Hi) {
  SDOperand Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SDOperand Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                            RTLIB::DIV_F32,
                                            RTLIB::DIV_F64,
                                            RTLIB::DIV_F80,
                                            RTLIB::DIV_PPCF128),
                               N->getValueType(0), Ops, 2,
                               false);
  assert(Call.Val->getOpcode() == ISD::BUILD_PAIR && "Call lowered wrongly!");
  Lo = Call.getOperand(0); Hi = Call.getOperand(1);
}

void DAGTypeLegalizer::ExpandFloatRes_FMUL(SDNode *N, SDOperand &Lo,
                                           SDOperand &Hi) {
  SDOperand Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SDOperand Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                            RTLIB::MUL_F32,
                                            RTLIB::MUL_F64,
                                            RTLIB::MUL_F80,
                                            RTLIB::MUL_PPCF128),
                               N->getValueType(0), Ops, 2,
                               false);
  assert(Call.Val->getOpcode() == ISD::BUILD_PAIR && "Call lowered wrongly!");
  Lo = Call.getOperand(0); Hi = Call.getOperand(1);
}

void DAGTypeLegalizer::ExpandFloatRes_FSUB(SDNode *N, SDOperand &Lo,
                                           SDOperand &Hi) {
  SDOperand Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SDOperand Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                            RTLIB::SUB_F32,
                                            RTLIB::SUB_F64,
                                            RTLIB::SUB_F80,
                                            RTLIB::SUB_PPCF128),
                               N->getValueType(0), Ops, 2,
                               false);
  assert(Call.Val->getOpcode() == ISD::BUILD_PAIR && "Call lowered wrongly!");
  Lo = Call.getOperand(0); Hi = Call.getOperand(1);
}

void DAGTypeLegalizer::ExpandFloatRes_LOAD(SDNode *N, SDOperand &Lo,
                                           SDOperand &Hi) {
  if (ISD::isNormalLoad(N)) {
    ExpandRes_NormalLoad(N, Lo, Hi);
    return;
  }

  assert(ISD::isUNINDEXEDLoad(N) && "Indexed load during type legalization!");
  LoadSDNode *LD = cast<LoadSDNode>(N);
  SDOperand Chain = LD->getChain();
  SDOperand Ptr = LD->getBasePtr();

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
  ReplaceValueWith(SDOperand(LD, 1), Chain);
}

void DAGTypeLegalizer::ExpandFloatRes_XINT_TO_FP(SDNode *N, SDOperand &Lo,
                                                 SDOperand &Hi) {
  assert(N->getValueType(0) == MVT::ppcf128 && "Unsupported XINT_TO_FP!");
  MVT VT = N->getValueType(0);
  MVT NVT = TLI.getTypeToTransformTo(VT);
  SDOperand Src = N->getOperand(0);
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
    assert(Hi.Val->getOpcode() == ISD::BUILD_PAIR && "Call lowered wrongly!");
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
  Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, NVT, Lo,
                   DAG.getConstant(1, TLI.getPointerTy()));
  Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, NVT, Lo,
                   DAG.getConstant(0, TLI.getPointerTy()));
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
  SDOperand Res(0, 0);

  if (TLI.getOperationAction(N->getOpcode(), N->getOperand(OpNo).getValueType())
      == TargetLowering::Custom)
    Res = TLI.LowerOperation(SDOperand(N, 0), DAG);

  if (Res.Val == 0) {
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

    case ISD::BR_CC:     Res = ExpandFloatOp_BR_CC(N); break;
    case ISD::SELECT_CC: Res = ExpandFloatOp_BR_CC(N); break;
    case ISD::SETCC:     Res = ExpandFloatOp_BR_CC(N); break;

    case ISD::FP_ROUND:   Res = ExpandFloatOp_FP_ROUND(N); break;
    case ISD::FP_TO_SINT: Res = ExpandFloatOp_FP_TO_SINT(N); break;
    case ISD::FP_TO_UINT: Res = ExpandFloatOp_FP_TO_UINT(N); break;

    case ISD::STORE:
      Res = ExpandFloatOp_STORE(cast<StoreSDNode>(N), OpNo);
      break;
    }
  }

  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.Val) return false;
  // If the result is N, the sub-method updated N in place.  Check to see if any
  // operands are new, and if so, mark them.
  if (Res.Val == N) {
    // Mark N as new and remark N and its operands.  This allows us to correctly
    // revisit N if it needs another step of expansion and allows us to visit
    // any new operands to N.
    ReanalyzeNode(N);
    return true;
  }

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");

  ReplaceValueWith(SDOperand(N, 0), Res);
  return false;
}

/// FloatExpandSetCCOperands - Expand the operands of a comparison.  This code
/// is shared among BR_CC, SELECT_CC, and SETCC handlers.
void DAGTypeLegalizer::FloatExpandSetCCOperands(SDOperand &NewLHS,
                                                SDOperand &NewRHS,
                                                ISD::CondCode &CCCode) {
  SDOperand LHSLo, LHSHi, RHSLo, RHSHi;
  GetExpandedFloat(NewLHS, LHSLo, LHSHi);
  GetExpandedFloat(NewRHS, RHSLo, RHSHi);

  MVT VT = NewLHS.getValueType();
  assert(VT == MVT::ppcf128 && "Unsupported setcc type!");

  // FIXME:  This generated code sucks.  We want to generate
  //         FCMP crN, hi1, hi2
  //         BNE crN, L:
  //         FCMP crN, lo1, lo2
  // The following can be improved, but not that much.
  SDOperand Tmp1, Tmp2, Tmp3;
  Tmp1 = DAG.getSetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi, ISD::SETEQ);
  Tmp2 = DAG.getSetCC(TLI.getSetCCResultType(LHSLo), LHSLo, RHSLo, CCCode);
  Tmp3 = DAG.getNode(ISD::AND, Tmp1.getValueType(), Tmp1, Tmp2);
  Tmp1 = DAG.getSetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi, ISD::SETNE);
  Tmp2 = DAG.getSetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi, CCCode);
  Tmp1 = DAG.getNode(ISD::AND, Tmp1.getValueType(), Tmp1, Tmp2);
  NewLHS = DAG.getNode(ISD::OR, Tmp1.getValueType(), Tmp1, Tmp3);
  NewRHS = SDOperand();   // LHS is the result, not a compare.
}

SDOperand DAGTypeLegalizer::ExpandFloatOp_BR_CC(SDNode *N) {
  SDOperand NewLHS = N->getOperand(2), NewRHS = N->getOperand(3);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(1))->get();
  FloatExpandSetCCOperands(NewLHS, NewRHS, CCCode);

  // If ExpandSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.Val == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), N->getOperand(0),
                                DAG.getCondCode(CCCode), NewLHS, NewRHS,
                                N->getOperand(4));
}

SDOperand DAGTypeLegalizer::ExpandFloatOp_SELECT_CC(SDNode *N) {
  SDOperand NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(4))->get();
  FloatExpandSetCCOperands(NewLHS, NewRHS, CCCode);

  // If ExpandSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.Val == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), NewLHS, NewRHS,
                                N->getOperand(2), N->getOperand(3),
                                DAG.getCondCode(CCCode));
}

SDOperand DAGTypeLegalizer::ExpandFloatOp_SETCC(SDNode *N) {
  SDOperand NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(2))->get();
  FloatExpandSetCCOperands(NewLHS, NewRHS, CCCode);

  // If ExpandSetCCOperands returned a scalar, use it.
  if (NewRHS.Val == 0) {
    assert(NewLHS.getValueType() == N->getValueType(0) &&
           "Unexpected setcc expansion!");
    return NewLHS;
  }

  // Otherwise, update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), NewLHS, NewRHS,
                                DAG.getCondCode(CCCode));
}

SDOperand DAGTypeLegalizer::ExpandFloatOp_FP_TO_UINT(SDNode *N) {
  assert(N->getOperand(0).getValueType() == MVT::ppcf128 &&
         "Unsupported FP_TO_UINT!");

  RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
  switch (N->getValueType(0).getSimpleVT()) {
  default:
    assert(false && "Unsupported FP_TO_UINT!");
  case MVT::i32:
    LC = RTLIB::FPTOUINT_PPCF128_I32;
    break;
  case MVT::i64:
    LC = RTLIB::FPTOUINT_PPCF128_I64;
    break;
  case MVT::i128:
    LC = RTLIB::FPTOUINT_PPCF128_I128;
    break;
  }

  return MakeLibCall(LC, N->getValueType(0), &N->getOperand(0), 1, false);
}

SDOperand DAGTypeLegalizer::ExpandFloatOp_FP_TO_SINT(SDNode *N) {
  assert(N->getOperand(0).getValueType() == MVT::ppcf128 &&
         "Unsupported FP_TO_SINT!");

  RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
  switch (N->getValueType(0).getSimpleVT()) {
  default:
    assert(false && "Unsupported FP_TO_SINT!");
  case MVT::i32:
    LC = RTLIB::FPTOSINT_PPCF128_I32;
  case MVT::i64:
    LC = RTLIB::FPTOSINT_PPCF128_I64;
    break;
  case MVT::i128:
    LC = RTLIB::FPTOSINT_PPCF128_I64;
    break;
  }

  return MakeLibCall(LC, N->getValueType(0), &N->getOperand(0), 1, false);
}

SDOperand DAGTypeLegalizer::ExpandFloatOp_FP_ROUND(SDNode *N) {
  assert(N->getOperand(0).getValueType() == MVT::ppcf128 &&
         "Logic only correct for ppcf128!");
  SDOperand Lo, Hi;
  GetExpandedFloat(N->getOperand(0), Lo, Hi);
  // Round it the rest of the way (e.g. to f32) if needed.
  return DAG.getNode(ISD::FP_ROUND, N->getValueType(0), Hi, N->getOperand(1));
}

SDOperand DAGTypeLegalizer::ExpandFloatOp_STORE(SDNode *N, unsigned OpNo) {
  if (ISD::isNormalStore(N))
    return ExpandOp_NormalStore(N, OpNo);

  assert(ISD::isUNINDEXEDStore(N) && "Indexed store during type legalization!");
  assert(OpNo == 1 && "Can only expand the stored value so far");
  StoreSDNode *ST = cast<StoreSDNode>(N);

  SDOperand Chain = ST->getChain();
  SDOperand Ptr = ST->getBasePtr();

  MVT NVT = TLI.getTypeToTransformTo(ST->getValue().getValueType());
  assert(NVT.isByteSized() && "Expanded type not byte sized!");
  assert(ST->getMemoryVT().bitsLE(NVT) && "Float type not round?");

  SDOperand Lo, Hi;
  GetExpandedOp(ST->getValue(), Lo, Hi);

  return DAG.getTruncStore(Chain, Lo, Ptr,
                           ST->getSrcValue(), ST->getSrcValueOffset(),
                           ST->getMemoryVT(),
                           ST->isVolatile(), ST->getAlignment());
}
