//===-------- LegalizeFloatTypes.cpp - Legalization of float types --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements float type expansion and conversion of float types to
// integer types on behalf of LegalizeTypes.
// Converting to integer is the act of turning a computation in an illegal
// floating point type into a computation in an integer type of the same size.
// For example, turning f32 arithmetic into operations using i32.  Also known as
// "soft float".  The result is equivalent to bitcasting the float value to the
// integer type.
// Expansion is the act of changing a computation in an illegal type to be a
// computation in multiple registers of a smaller type.  For example,
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

void DAGTypeLegalizer::PromoteFloatResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Promote float result " << ResNo << ": "; N->dump(&DAG);
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
    cerr << "PromoteFloatResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to convert the result of this operator!");
    abort();

    case ISD::BIT_CONVERT: R = PromoteFloatRes_BIT_CONVERT(N); break;
    case ISD::BUILD_PAIR:  R = PromoteFloatRes_BUILD_PAIR(N); break;
    case ISD::ConstantFP:
      R = PromoteFloatRes_ConstantFP(cast<ConstantFPSDNode>(N));
      break;
    case ISD::FCOPYSIGN:   R = PromoteFloatRes_FCOPYSIGN(N); break;
    case ISD::LOAD:        R = PromoteFloatRes_LOAD(N); break;
    case ISD::SINT_TO_FP:
    case ISD::UINT_TO_FP:  R = PromoteFloatRes_XINT_TO_FP(N); break;

    case ISD::FADD: R = PromoteFloatRes_FADD(N); break;
    case ISD::FMUL: R = PromoteFloatRes_FMUL(N); break;
    case ISD::FSUB: R = PromoteFloatRes_FSUB(N); break;
  }

  // If R is null, the sub-method took care of registering the result.
  if (R.Val)
    SetPromotedFloat(SDOperand(N, ResNo), R);
}

SDOperand DAGTypeLegalizer::PromoteFloatRes_BIT_CONVERT(SDNode *N) {
  return BitConvertToInteger(N->getOperand(0));
}

SDOperand DAGTypeLegalizer::PromoteFloatRes_BUILD_PAIR(SDNode *N) {
  // Convert the inputs to integers, and build a new pair out of them.
  return DAG.getNode(ISD::BUILD_PAIR,
                     TLI.getTypeToTransformTo(N->getValueType(0)),
                     BitConvertToInteger(N->getOperand(0)),
                     BitConvertToInteger(N->getOperand(1)));
}

SDOperand DAGTypeLegalizer::PromoteFloatRes_ConstantFP(ConstantFPSDNode *N) {
  return DAG.getConstant(N->getValueAPF().convertToAPInt(),
                         TLI.getTypeToTransformTo(N->getValueType(0)));
}

SDOperand DAGTypeLegalizer::PromoteFloatRes_FADD(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDOperand Ops[2] = { GetPromotedFloat(N->getOperand(0)),
                       GetPromotedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::ADD_F32,
                                  RTLIB::ADD_F64,
                                  RTLIB::ADD_F80,
                                  RTLIB::ADD_PPCF128),
                     NVT, Ops, 2, false/*sign irrelevant*/);
}

SDOperand DAGTypeLegalizer::PromoteFloatRes_FCOPYSIGN(SDNode *N) {
  SDOperand LHS = GetPromotedFloat(N->getOperand(0));
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

SDOperand DAGTypeLegalizer::PromoteFloatRes_FMUL(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDOperand Ops[2] = { GetPromotedFloat(N->getOperand(0)),
                       GetPromotedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::MUL_F32,
                                  RTLIB::MUL_F64,
                                  RTLIB::MUL_F80,
                                  RTLIB::MUL_PPCF128),
                     NVT, Ops, 2, false/*sign irrelevant*/);
}

SDOperand DAGTypeLegalizer::PromoteFloatRes_FSUB(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDOperand Ops[2] = { GetPromotedFloat(N->getOperand(0)),
                       GetPromotedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::SUB_F32,
                                  RTLIB::SUB_F64,
                                  RTLIB::SUB_F80,
                                  RTLIB::SUB_PPCF128),
                     NVT, Ops, 2, false/*sign irrelevant*/);
}

SDOperand DAGTypeLegalizer::PromoteFloatRes_LOAD(SDNode *N) {
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

SDOperand DAGTypeLegalizer::PromoteFloatRes_XINT_TO_FP(SDNode *N) {
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

bool DAGTypeLegalizer::PromoteFloatOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Promote float operand " << OpNo << ": "; N->dump(&DAG);
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
      cerr << "PromoteFloatOperand Op #" << OpNo << ": ";
      N->dump(&DAG); cerr << "\n";
#endif
      assert(0 && "Do not know how to convert this operator's operand!");
      abort();

      case ISD::BIT_CONVERT: Res = PromoteFloatOp_BIT_CONVERT(N); break;
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

SDOperand DAGTypeLegalizer::PromoteFloatOp_BIT_CONVERT(SDNode *N) {
  return DAG.getNode(ISD::BIT_CONVERT, N->getValueType(0),
                     GetPromotedFloat(N->getOperand(0)));
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
  }

  // If Lo/Hi is null, the sub-method took care of registering results etc.
  if (Lo.Val)
    SetExpandedFloat(SDOperand(N, ResNo), Lo, Hi);
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
