//===-- LegalizeTypesPromote.cpp - Promotion for LegalizeTypes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements promotion support for LegalizeTypes.  Promotion is the
// act of changing a computation in an invalid type to be a computation in a 
// larger type.  For example, implementing i8 arithmetic in an i32 register (as
// is often needed on powerpc for example).
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  Result Promotion
//===----------------------------------------------------------------------===//

/// PromoteResult - This method is called when a result of a node is found to be
/// in need of promotion to a larger type.  At this point, the node may also
/// have invalid operands or may have other results that need expansion, we just
/// know that (at least) one result needs promotion.
void DAGTypeLegalizer::PromoteResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Promote node result: "; N->dump(&DAG); cerr << "\n");
  SDOperand Result = SDOperand();
  
  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "PromoteResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to promote this operator!");
    abort();
  case ISD::UNDEF:    Result = PromoteResult_UNDEF(N); break;
  case ISD::Constant: Result = PromoteResult_Constant(N); break;

  case ISD::TRUNCATE:    Result = PromoteResult_TRUNCATE(N); break;
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:  Result = PromoteResult_INT_EXTEND(N); break;
  case ISD::FP_ROUND:    Result = PromoteResult_FP_ROUND(N); break;
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:  Result = PromoteResult_FP_TO_XINT(N); break;
  case ISD::SETCC:    Result = PromoteResult_SETCC(N); break;
  case ISD::LOAD:     Result = PromoteResult_LOAD(cast<LoadSDNode>(N)); break;
  case ISD::BUILD_PAIR:  Result = PromoteResult_BUILD_PAIR(N); break;
  case ISD::BIT_CONVERT: Result = PromoteResult_BIT_CONVERT(N); break;

  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:      Result = PromoteResult_SimpleIntBinOp(N); break;

  case ISD::SDIV:
  case ISD::SREM:     Result = PromoteResult_SDIV(N); break;

  case ISD::UDIV:
  case ISD::UREM:     Result = PromoteResult_UDIV(N); break;

  case ISD::SHL:      Result = PromoteResult_SHL(N); break;
  case ISD::SRA:      Result = PromoteResult_SRA(N); break;
  case ISD::SRL:      Result = PromoteResult_SRL(N); break;

  case ISD::SELECT:    Result = PromoteResult_SELECT(N); break;
  case ISD::SELECT_CC: Result = PromoteResult_SELECT_CC(N); break;

  case ISD::CTLZ:     Result = PromoteResult_CTLZ(N); break;
  case ISD::CTPOP:    Result = PromoteResult_CTPOP(N); break;
  case ISD::CTTZ:     Result = PromoteResult_CTTZ(N); break;

  case ISD::EXTRACT_VECTOR_ELT:
    Result = PromoteResult_EXTRACT_VECTOR_ELT(N);
    break;
  }      

  // If Result is null, the sub-method took care of registering the result.
  if (Result.Val)
    SetPromotedOp(SDOperand(N, ResNo), Result);
}

SDOperand DAGTypeLegalizer::PromoteResult_UNDEF(SDNode *N) {
  return DAG.getNode(ISD::UNDEF, TLI.getTypeToTransformTo(N->getValueType(0)));
}

SDOperand DAGTypeLegalizer::PromoteResult_Constant(SDNode *N) {
  MVT VT = N->getValueType(0);
  // Zero extend things like i1, sign extend everything else.  It shouldn't
  // matter in theory which one we pick, but this tends to give better code?
  unsigned Opc = VT != MVT::i1 ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
  SDOperand Result = DAG.getNode(Opc, TLI.getTypeToTransformTo(VT),
                                 SDOperand(N, 0));
  assert(isa<ConstantSDNode>(Result) && "Didn't constant fold ext?");
  return Result;
}

SDOperand DAGTypeLegalizer::PromoteResult_TRUNCATE(SDNode *N) {
  SDOperand Res;

  switch (getTypeAction(N->getOperand(0).getValueType())) {
  default: assert(0 && "Unknown type action!");
  case Legal:
  case Expand:
    Res = N->getOperand(0);
    break;
  case Promote:
    Res = GetPromotedOp(N->getOperand(0));
    break;
  }

  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  assert(Res.getValueType().getSizeInBits() >= NVT.getSizeInBits() &&
         "Truncation doesn't make sense!");
  if (Res.getValueType() == NVT)
    return Res;

  // Truncate to NVT instead of VT
  return DAG.getNode(ISD::TRUNCATE, NVT, Res);
}

SDOperand DAGTypeLegalizer::PromoteResult_INT_EXTEND(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));

  if (getTypeAction(N->getOperand(0).getValueType()) == Promote) {
    SDOperand Res = GetPromotedOp(N->getOperand(0));
    assert(Res.getValueType().getSizeInBits() <= NVT.getSizeInBits() &&
           "Extension doesn't make sense!");

    // If the result and operand types are the same after promotion, simplify
    // to an in-register extension.
    if (NVT == Res.getValueType()) {
      // The high bits are not guaranteed to be anything.  Insert an extend.
      if (N->getOpcode() == ISD::SIGN_EXTEND)
        return DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Res,
                           DAG.getValueType(N->getOperand(0).getValueType()));
      if (N->getOpcode() == ISD::ZERO_EXTEND)
        return DAG.getZeroExtendInReg(Res, N->getOperand(0).getValueType());
      assert(N->getOpcode() == ISD::ANY_EXTEND && "Unknown integer extension!");
      return Res;
    }
  }

  // Otherwise, just extend the original operand all the way to the larger type.
  return DAG.getNode(N->getOpcode(), NVT, N->getOperand(0));
}

SDOperand DAGTypeLegalizer::PromoteResult_FP_ROUND(SDNode *N) {
  // NOTE: Assumes input is legal.
  if (N->getConstantOperandVal(1) == 0) 
    return DAG.getNode(ISD::FP_ROUND_INREG, N->getOperand(0).getValueType(),
                       N->getOperand(0), DAG.getValueType(N->getValueType(0)));
  // If the precision discard isn't needed, just return the operand unrounded.
  return N->getOperand(0);
}

SDOperand DAGTypeLegalizer::PromoteResult_FP_TO_XINT(SDNode *N) {
  SDOperand Op = N->getOperand(0);
  // If the operand needed to be promoted, do so now.
  if (getTypeAction(Op.getValueType()) == Promote)
    // The input result is prerounded, so we don't have to do anything special.
    Op = GetPromotedOp(Op);
  
  unsigned NewOpc = N->getOpcode();
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  
  // If we're promoting a UINT to a larger size, check to see if the new node
  // will be legal.  If it isn't, check to see if FP_TO_SINT is legal, since
  // we can use that instead.  This allows us to generate better code for
  // FP_TO_UINT for small destination sizes on targets where FP_TO_UINT is not
  // legal, such as PowerPC.
  if (N->getOpcode() == ISD::FP_TO_UINT) {
    if (!TLI.isOperationLegal(ISD::FP_TO_UINT, NVT) &&
        (TLI.isOperationLegal(ISD::FP_TO_SINT, NVT) ||
         TLI.getOperationAction(ISD::FP_TO_SINT, NVT)==TargetLowering::Custom))
      NewOpc = ISD::FP_TO_SINT;
  }

  return DAG.getNode(NewOpc, NVT, Op);
}

SDOperand DAGTypeLegalizer::PromoteResult_SETCC(SDNode *N) {
  assert(isTypeLegal(TLI.getSetCCResultType(N->getOperand(0)))
         && "SetCC type is not legal??");
  return DAG.getNode(ISD::SETCC, TLI.getSetCCResultType(N->getOperand(0)),
                     N->getOperand(0), N->getOperand(1), N->getOperand(2));
}

SDOperand DAGTypeLegalizer::PromoteResult_LOAD(LoadSDNode *N) {
  // FIXME: Add support for indexed loads.
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  ISD::LoadExtType ExtType =
    ISD::isNON_EXTLoad(N) ? ISD::EXTLOAD : N->getExtensionType();
  SDOperand Res = DAG.getExtLoad(ExtType, NVT, N->getChain(), N->getBasePtr(),
                                 N->getSrcValue(), N->getSrcValueOffset(),
                                 N->getMemoryVT(), N->isVolatile(),
                                 N->getAlignment());

  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDOperand(N, 1), Res.getValue(1));
  return Res;
}

SDOperand DAGTypeLegalizer::PromoteResult_BUILD_PAIR(SDNode *N) {
  // The pair element type may be legal, or may not promote to the same type as
  // the result, for example i14 = BUILD_PAIR (i7, i7).  Handle all cases.
  return DAG.getNode(ISD::ANY_EXTEND,
                     TLI.getTypeToTransformTo(N->getValueType(0)),
                     JoinIntegers(N->getOperand(0), N->getOperand(1)));
}

SDOperand DAGTypeLegalizer::PromoteResult_BIT_CONVERT(SDNode *N) {
  SDOperand InOp = N->getOperand(0);
  MVT InVT = InOp.getValueType();
  MVT NInVT = TLI.getTypeToTransformTo(InVT);
  MVT OutVT = TLI.getTypeToTransformTo(N->getValueType(0));

  switch (getTypeAction(InVT)) {
  default:
    assert(false && "Unknown type action!");
    break;
  case Legal:
    break;
  case Promote:
    if (OutVT.getSizeInBits() == NInVT.getSizeInBits())
      // The input promotes to the same size.  Convert the promoted value.
      return DAG.getNode(ISD::BIT_CONVERT, OutVT, GetPromotedOp(InOp));
    break;
  case Expand:
    break;
  case FloatToInt:
    // Promote the integer operand by hand.
    return DAG.getNode(ISD::ANY_EXTEND, OutVT, GetIntegerOp(InOp));
  case Scalarize:
    // Convert the element to an integer and promote it by hand.
    return DAG.getNode(ISD::ANY_EXTEND, OutVT,
                       BitConvertToInteger(GetScalarizedOp(InOp)));
  case Split:
    // For example, i32 = BIT_CONVERT v2i16 on alpha.  Convert the split
    // pieces of the input into integers and reassemble in the final type.
    SDOperand Lo, Hi;
    GetSplitOp(N->getOperand(0), Lo, Hi);
    Lo = BitConvertToInteger(Lo);
    Hi = BitConvertToInteger(Hi);

    if (TLI.isBigEndian())
      std::swap(Lo, Hi);

    InOp = DAG.getNode(ISD::ANY_EXTEND,
                       MVT::getIntegerVT(OutVT.getSizeInBits()),
                       JoinIntegers(Lo, Hi));
    return DAG.getNode(ISD::BIT_CONVERT, OutVT, InOp);
  }

  // Otherwise, lower the bit-convert to a store/load from the stack, then
  // promote the load.
  SDOperand Op = CreateStackStoreLoad(InOp, N->getValueType(0));
  return PromoteResult_LOAD(cast<LoadSDNode>(Op.Val));
}

SDOperand DAGTypeLegalizer::PromoteResult_SimpleIntBinOp(SDNode *N) {
  // The input may have strange things in the top bits of the registers, but
  // these operations don't care.  They may have weird bits going out, but
  // that too is okay if they are integer operations.
  SDOperand LHS = GetPromotedOp(N->getOperand(0));
  SDOperand RHS = GetPromotedOp(N->getOperand(1));
  return DAG.getNode(N->getOpcode(), LHS.getValueType(), LHS, RHS);
}

SDOperand DAGTypeLegalizer::PromoteResult_SDIV(SDNode *N) {
  // Sign extend the input.
  SDOperand LHS = GetPromotedOp(N->getOperand(0));
  SDOperand RHS = GetPromotedOp(N->getOperand(1));
  MVT VT = N->getValueType(0);
  LHS = DAG.getNode(ISD::SIGN_EXTEND_INREG, LHS.getValueType(), LHS,
                    DAG.getValueType(VT));
  RHS = DAG.getNode(ISD::SIGN_EXTEND_INREG, RHS.getValueType(), RHS,
                    DAG.getValueType(VT));

  return DAG.getNode(N->getOpcode(), LHS.getValueType(), LHS, RHS);
}

SDOperand DAGTypeLegalizer::PromoteResult_UDIV(SDNode *N) {
  // Zero extend the input.
  SDOperand LHS = GetPromotedOp(N->getOperand(0));
  SDOperand RHS = GetPromotedOp(N->getOperand(1));
  MVT VT = N->getValueType(0);
  LHS = DAG.getZeroExtendInReg(LHS, VT);
  RHS = DAG.getZeroExtendInReg(RHS, VT);

  return DAG.getNode(N->getOpcode(), LHS.getValueType(), LHS, RHS);
}

SDOperand DAGTypeLegalizer::PromoteResult_SHL(SDNode *N) {
  return DAG.getNode(ISD::SHL, TLI.getTypeToTransformTo(N->getValueType(0)),
                     GetPromotedOp(N->getOperand(0)), N->getOperand(1));
}

SDOperand DAGTypeLegalizer::PromoteResult_SRA(SDNode *N) {
  // The input value must be properly sign extended.
  MVT VT = N->getValueType(0);
  MVT NVT = TLI.getTypeToTransformTo(VT);
  SDOperand Res = GetPromotedOp(N->getOperand(0));
  Res = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Res, DAG.getValueType(VT));
  return DAG.getNode(ISD::SRA, NVT, Res, N->getOperand(1));
}

SDOperand DAGTypeLegalizer::PromoteResult_SRL(SDNode *N) {
  // The input value must be properly zero extended.
  MVT VT = N->getValueType(0);
  MVT NVT = TLI.getTypeToTransformTo(VT);
  SDOperand Res = GetPromotedZExtOp(N->getOperand(0));
  return DAG.getNode(ISD::SRL, NVT, Res, N->getOperand(1));
}

SDOperand DAGTypeLegalizer::PromoteResult_SELECT(SDNode *N) {
  SDOperand LHS = GetPromotedOp(N->getOperand(1));
  SDOperand RHS = GetPromotedOp(N->getOperand(2));
  return DAG.getNode(ISD::SELECT, LHS.getValueType(), N->getOperand(0),LHS,RHS);
}

SDOperand DAGTypeLegalizer::PromoteResult_SELECT_CC(SDNode *N) {
  SDOperand LHS = GetPromotedOp(N->getOperand(2));
  SDOperand RHS = GetPromotedOp(N->getOperand(3));
  return DAG.getNode(ISD::SELECT_CC, LHS.getValueType(), N->getOperand(0),
                     N->getOperand(1), LHS, RHS, N->getOperand(4));
}

SDOperand DAGTypeLegalizer::PromoteResult_CTLZ(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  MVT OVT = N->getValueType(0);
  MVT NVT = Op.getValueType();
  // Zero extend to the promoted type and do the count there.
  Op = DAG.getNode(ISD::CTLZ, NVT, DAG.getZeroExtendInReg(Op, OVT));
  // Subtract off the extra leading bits in the bigger type.
  return DAG.getNode(ISD::SUB, NVT, Op,
                     DAG.getConstant(NVT.getSizeInBits() -
                                     OVT.getSizeInBits(), NVT));
}

SDOperand DAGTypeLegalizer::PromoteResult_CTPOP(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  MVT OVT = N->getValueType(0);
  MVT NVT = Op.getValueType();
  // Zero extend to the promoted type and do the count there.
  return DAG.getNode(ISD::CTPOP, NVT, DAG.getZeroExtendInReg(Op, OVT));
}

SDOperand DAGTypeLegalizer::PromoteResult_CTTZ(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  MVT OVT = N->getValueType(0);
  MVT NVT = Op.getValueType();
  // The count is the same in the promoted type except if the original
  // value was zero.  This can be handled by setting the bit just off
  // the top of the original type.
  Op = DAG.getNode(ISD::OR, NVT, Op,
                   // FIXME: Do this using an APINT constant.
                   DAG.getConstant(1UL << OVT.getSizeInBits(), NVT));
  return DAG.getNode(ISD::CTTZ, NVT, Op);
}

SDOperand DAGTypeLegalizer::PromoteResult_EXTRACT_VECTOR_ELT(SDNode *N) {
  MVT OldVT = N->getValueType(0);
  SDOperand OldVec = N->getOperand(0);
  unsigned OldElts = OldVec.getValueType().getVectorNumElements();

  if (OldElts == 1) {
    assert(!isTypeLegal(OldVec.getValueType()) &&
           "Legal one-element vector of a type needing promotion!");
    // It is tempting to follow GetScalarizedOp by a call to GetPromotedOp,
    // but this would be wrong because the scalarized value may not yet have
    // been processed.
    return DAG.getNode(ISD::ANY_EXTEND, TLI.getTypeToTransformTo(OldVT),
                       GetScalarizedOp(OldVec));
  }

  // Convert to a vector half as long with an element type of twice the width,
  // for example <4 x i16> -> <2 x i32>.
  assert(!(OldElts & 1) && "Odd length vectors not supported!");
  MVT NewVT = MVT::getIntegerVT(2 * OldVT.getSizeInBits());
  assert(OldVT.isSimple() && NewVT.isSimple());

  SDOperand NewVec = DAG.getNode(ISD::BIT_CONVERT,
                                 MVT::getVectorVT(NewVT, OldElts / 2),
                                 OldVec);

  // Extract the element at OldIdx / 2 from the new vector.
  SDOperand OldIdx = N->getOperand(1);
  SDOperand NewIdx = DAG.getNode(ISD::SRL, OldIdx.getValueType(), OldIdx,
                                 DAG.getConstant(1, TLI.getShiftAmountTy()));
  SDOperand Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, NewVT, NewVec, NewIdx);

  // Select the appropriate half of the element: Lo if OldIdx was even,
  // Hi if it was odd.
  SDOperand Lo = Elt;
  SDOperand Hi = DAG.getNode(ISD::SRL, NewVT, Elt,
                             DAG.getConstant(OldVT.getSizeInBits(),
                                             TLI.getShiftAmountTy()));
  if (TLI.isBigEndian())
    std::swap(Lo, Hi);

  SDOperand Odd = DAG.getNode(ISD::AND, OldIdx.getValueType(), OldIdx,
                              DAG.getConstant(1, TLI.getShiftAmountTy()));
  return DAG.getNode(ISD::SELECT, NewVT, Odd, Hi, Lo);
}

//===----------------------------------------------------------------------===//
//  Operand Promotion
//===----------------------------------------------------------------------===//

/// PromoteOperand - This method is called when the specified operand of the
/// specified node is found to need promotion.  At this point, all of the result
/// types of the node are known to be legal, but other operands of the node may
/// need promotion or expansion as well as the specified one.
bool DAGTypeLegalizer::PromoteOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Promote node operand: "; N->dump(&DAG); cerr << "\n");
  SDOperand Res;
  switch (N->getOpcode()) {
    default:
#ifndef NDEBUG
    cerr << "PromoteOperand Op #" << OpNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to promote this operator's operand!");
    abort();
    
  case ISD::ANY_EXTEND:  Res = PromoteOperand_ANY_EXTEND(N); break;
  case ISD::ZERO_EXTEND: Res = PromoteOperand_ZERO_EXTEND(N); break;
  case ISD::SIGN_EXTEND: Res = PromoteOperand_SIGN_EXTEND(N); break;
  case ISD::TRUNCATE:    Res = PromoteOperand_TRUNCATE(N); break;
  case ISD::FP_EXTEND:   Res = PromoteOperand_FP_EXTEND(N); break;
  case ISD::FP_ROUND:    Res = PromoteOperand_FP_ROUND(N); break;
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:  Res = PromoteOperand_INT_TO_FP(N); break;
  case ISD::BUILD_PAIR:  Res = PromoteOperand_BUILD_PAIR(N); break;

  case ISD::SELECT:      Res = PromoteOperand_SELECT(N, OpNo); break;
  case ISD::BRCOND:      Res = PromoteOperand_BRCOND(N, OpNo); break;
  case ISD::BR_CC:       Res = PromoteOperand_BR_CC(N, OpNo); break;
  case ISD::SETCC:       Res = PromoteOperand_SETCC(N, OpNo); break;

  case ISD::STORE:       Res = PromoteOperand_STORE(cast<StoreSDNode>(N),
                                                    OpNo); break;

  case ISD::BUILD_VECTOR: Res = PromoteOperand_BUILD_VECTOR(N); break;
  case ISD::INSERT_VECTOR_ELT:
    Res = PromoteOperand_INSERT_VECTOR_ELT(N, OpNo);
    break;

  case ISD::RET:         Res = PromoteOperand_RET(N, OpNo); break;

  case ISD::MEMBARRIER:  Res = PromoteOperand_MEMBARRIER(N); break;
  }

  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.Val) return false;
  // If the result is N, the sub-method updated N in place.
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

SDOperand DAGTypeLegalizer::PromoteOperand_ANY_EXTEND(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  return DAG.getNode(ISD::ANY_EXTEND, N->getValueType(0), Op);
}

SDOperand DAGTypeLegalizer::PromoteOperand_ZERO_EXTEND(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  Op = DAG.getNode(ISD::ANY_EXTEND, N->getValueType(0), Op);
  return DAG.getZeroExtendInReg(Op, N->getOperand(0).getValueType());
}

SDOperand DAGTypeLegalizer::PromoteOperand_SIGN_EXTEND(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  Op = DAG.getNode(ISD::ANY_EXTEND, N->getValueType(0), Op);
  return DAG.getNode(ISD::SIGN_EXTEND_INREG, Op.getValueType(),
                     Op, DAG.getValueType(N->getOperand(0).getValueType()));
}

SDOperand DAGTypeLegalizer::PromoteOperand_TRUNCATE(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  return DAG.getNode(ISD::TRUNCATE, N->getValueType(0), Op);
}

SDOperand DAGTypeLegalizer::PromoteOperand_FP_EXTEND(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  return DAG.getNode(ISD::FP_EXTEND, N->getValueType(0), Op);
}

SDOperand DAGTypeLegalizer::PromoteOperand_FP_ROUND(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  return DAG.getNode(ISD::FP_ROUND, N->getValueType(0), Op,
                     DAG.getIntPtrConstant(0));
}

SDOperand DAGTypeLegalizer::PromoteOperand_INT_TO_FP(SDNode *N) {
  SDOperand In = GetPromotedOp(N->getOperand(0));
  MVT OpVT = N->getOperand(0).getValueType();
  if (N->getOpcode() == ISD::UINT_TO_FP)
    In = DAG.getZeroExtendInReg(In, OpVT);
  else
    In = DAG.getNode(ISD::SIGN_EXTEND_INREG, In.getValueType(),
                     In, DAG.getValueType(OpVT));
  
  return DAG.UpdateNodeOperands(SDOperand(N, 0), In);
}

SDOperand DAGTypeLegalizer::PromoteOperand_BUILD_PAIR(SDNode *N) {
  // Since the result type is legal, the operands must promote to it.
  MVT OVT = N->getOperand(0).getValueType();
  SDOperand Lo = GetPromotedOp(N->getOperand(0));
  SDOperand Hi = GetPromotedOp(N->getOperand(1));
  assert(Lo.getValueType() == N->getValueType(0) && "Operand over promoted?");

  Lo = DAG.getZeroExtendInReg(Lo, OVT);
  Hi = DAG.getNode(ISD::SHL, N->getValueType(0), Hi,
                   DAG.getConstant(OVT.getSizeInBits(),
                                   TLI.getShiftAmountTy()));
  return DAG.getNode(ISD::OR, N->getValueType(0), Lo, Hi);
}

SDOperand DAGTypeLegalizer::PromoteOperand_SELECT(SDNode *N, unsigned OpNo) {
  assert(OpNo == 0 && "Only know how to promote condition");
  SDOperand Cond = GetPromotedOp(N->getOperand(0));  // Promote the condition.

  // The top bits of the promoted condition are not necessarily zero, ensure
  // that the value is properly zero extended.
  unsigned BitWidth = Cond.getValueSizeInBits();
  if (!DAG.MaskedValueIsZero(Cond, 
                             APInt::getHighBitsSet(BitWidth, BitWidth-1)))
    Cond = DAG.getZeroExtendInReg(Cond, MVT::i1);

  // The chain (Op#0) and basic block destination (Op#2) are always legal types.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), Cond, N->getOperand(1),
                                N->getOperand(2));
}

SDOperand DAGTypeLegalizer::PromoteOperand_BRCOND(SDNode *N, unsigned OpNo) {
  assert(OpNo == 1 && "only know how to promote condition");
  SDOperand Cond = GetPromotedOp(N->getOperand(1));  // Promote the condition.
  
  // The top bits of the promoted condition are not necessarily zero, ensure
  // that the value is properly zero extended.
  unsigned BitWidth = Cond.getValueSizeInBits();
  if (!DAG.MaskedValueIsZero(Cond, 
                             APInt::getHighBitsSet(BitWidth, BitWidth-1)))
    Cond = DAG.getZeroExtendInReg(Cond, MVT::i1);

  // The chain (Op#0) and basic block destination (Op#2) are always legal types.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), N->getOperand(0), Cond,
                                N->getOperand(2));
}

SDOperand DAGTypeLegalizer::PromoteOperand_BR_CC(SDNode *N, unsigned OpNo) {
  assert(OpNo == 2 && "Don't know how to promote this operand");
  
  SDOperand LHS = N->getOperand(2);
  SDOperand RHS = N->getOperand(3);
  PromoteSetCCOperands(LHS, RHS, cast<CondCodeSDNode>(N->getOperand(1))->get());
  
  // The chain (Op#0), CC (#1) and basic block destination (Op#4) are always
  // legal types.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), N->getOperand(0),
                                N->getOperand(1), LHS, RHS, N->getOperand(4));
}

SDOperand DAGTypeLegalizer::PromoteOperand_SETCC(SDNode *N, unsigned OpNo) {
  assert(OpNo == 0 && "Don't know how to promote this operand");

  SDOperand LHS = N->getOperand(0);
  SDOperand RHS = N->getOperand(1);
  PromoteSetCCOperands(LHS, RHS, cast<CondCodeSDNode>(N->getOperand(2))->get());

  // The CC (#2) is always legal.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), LHS, RHS, N->getOperand(2));
}

/// PromoteSetCCOperands - Promote the operands of a comparison.  This code is
/// shared among BR_CC, SELECT_CC, and SETCC handlers.
void DAGTypeLegalizer::PromoteSetCCOperands(SDOperand &NewLHS,SDOperand &NewRHS,
                                            ISD::CondCode CCCode) {
  MVT VT = NewLHS.getValueType();
  
  // Get the promoted values.
  NewLHS = GetPromotedOp(NewLHS);
  NewRHS = GetPromotedOp(NewRHS);
  
  // If this is an FP compare, the operands have already been extended.
  if (!NewLHS.getValueType().isInteger())
    return;
  
  // Otherwise, we have to insert explicit sign or zero extends.  Note
  // that we could insert sign extends for ALL conditions, but zero extend
  // is cheaper on many machines (an AND instead of two shifts), so prefer
  // it.
  switch (CCCode) {
  default: assert(0 && "Unknown integer comparison!");
  case ISD::SETEQ:
  case ISD::SETNE:
  case ISD::SETUGE:
  case ISD::SETUGT:
  case ISD::SETULE:
  case ISD::SETULT:
    // ALL of these operations will work if we either sign or zero extend
    // the operands (including the unsigned comparisons!).  Zero extend is
    // usually a simpler/cheaper operation, so prefer it.
    NewLHS = DAG.getZeroExtendInReg(NewLHS, VT);
    NewRHS = DAG.getZeroExtendInReg(NewRHS, VT);
    return;
  case ISD::SETGE:
  case ISD::SETGT:
  case ISD::SETLT:
  case ISD::SETLE:
    NewLHS = DAG.getNode(ISD::SIGN_EXTEND_INREG, NewLHS.getValueType(), NewLHS,
                         DAG.getValueType(VT));
    NewRHS = DAG.getNode(ISD::SIGN_EXTEND_INREG, NewRHS.getValueType(), NewRHS,
                         DAG.getValueType(VT));
    return;
  }
}

SDOperand DAGTypeLegalizer::PromoteOperand_STORE(StoreSDNode *N, unsigned OpNo){
  // FIXME: Add support for indexed stores.
  SDOperand Ch = N->getChain(), Ptr = N->getBasePtr();
  int SVOffset = N->getSrcValueOffset();
  unsigned Alignment = N->getAlignment();
  bool isVolatile = N->isVolatile();
  
  SDOperand Val = GetPromotedOp(N->getValue());  // Get promoted value.

  assert(!N->isTruncatingStore() && "Cannot promote this store operand!");
  
  // Truncate the value and store the result.
  return DAG.getTruncStore(Ch, Val, Ptr, N->getSrcValue(),
                           SVOffset, N->getMemoryVT(),
                           isVolatile, Alignment);
}

SDOperand DAGTypeLegalizer::PromoteOperand_BUILD_VECTOR(SDNode *N) {
  // The vector type is legal but the element type is not.  This implies
  // that the vector is a power-of-two in length and that the element
  // type does not have a strange size (eg: it is not i1).
  MVT VecVT = N->getValueType(0);
  unsigned NumElts = VecVT.getVectorNumElements();
  assert(!(NumElts & 1) && "Legal vector of one illegal element?");

  // Build a vector of half the length out of elements of twice the bitwidth.
  // For example <4 x i16> -> <2 x i32>.
  MVT OldVT = N->getOperand(0).getValueType();
  MVT NewVT = MVT::getIntegerVT(2 * OldVT.getSizeInBits());
  assert(OldVT.isSimple() && NewVT.isSimple());

  std::vector<SDOperand> NewElts;
  NewElts.reserve(NumElts/2);

  for (unsigned i = 0; i < NumElts; i += 2) {
    // Combine two successive elements into one promoted element.
    SDOperand Lo = N->getOperand(i);
    SDOperand Hi = N->getOperand(i+1);
    if (TLI.isBigEndian())
      std::swap(Lo, Hi);
    NewElts.push_back(JoinIntegers(Lo, Hi));
  }

  SDOperand NewVec = DAG.getNode(ISD::BUILD_VECTOR,
                                 MVT::getVectorVT(NewVT, NewElts.size()),
                                 &NewElts[0], NewElts.size());

  // Convert the new vector to the old vector type.
  return DAG.getNode(ISD::BIT_CONVERT, VecVT, NewVec);
}

SDOperand DAGTypeLegalizer::PromoteOperand_INSERT_VECTOR_ELT(SDNode *N,
                                                             unsigned OpNo) {
  if (OpNo == 1) {
    // Promote the inserted value.  This is valid because the type does not
    // have to match the vector element type.

    // Check that any extra bits introduced will be truncated away.
    assert(N->getOperand(1).getValueType().getSizeInBits() >=
           N->getValueType(0).getVectorElementType().getSizeInBits() &&
           "Type of inserted value narrower than vector element type!");
    return DAG.UpdateNodeOperands(SDOperand(N, 0), N->getOperand(0),
                                  GetPromotedOp(N->getOperand(1)),
                                  N->getOperand(2));
  }

  assert(OpNo == 2 && "Different operand and result vector types?");

  // Promote the index.
  SDOperand Idx = N->getOperand(2);
  Idx = DAG.getZeroExtendInReg(GetPromotedOp(Idx), Idx.getValueType());
  return DAG.UpdateNodeOperands(SDOperand(N, 0), N->getOperand(0),
                                N->getOperand(1), Idx);
}

SDOperand DAGTypeLegalizer::PromoteOperand_RET(SDNode *N, unsigned OpNo) {
  assert(!(OpNo & 1) && "Return values should be legally typed!");
  assert((N->getNumOperands() & 1) && "Wrong number of operands!");

  // It's a flag.  Promote all the flags in one hit, as an optimization.
  SmallVector<SDOperand, 8> NewValues(N->getNumOperands());
  NewValues[0] = N->getOperand(0); // The chain
  for (unsigned i = 1, e = N->getNumOperands(); i < e; i += 2) {
    // The return value.
    NewValues[i] = N->getOperand(i);

    // The flag.
    SDOperand Flag = N->getOperand(i + 1);
    if (getTypeAction(Flag.getValueType()) == Promote)
      // The promoted value may have rubbish in the new bits, but that
      // doesn't matter because those bits aren't queried anyway.
      Flag = GetPromotedOp(Flag);
    NewValues[i + 1] = Flag;
  }

  return DAG.UpdateNodeOperands(SDOperand (N, 0),
                                &NewValues[0], NewValues.size());
}

SDOperand DAGTypeLegalizer::PromoteOperand_MEMBARRIER(SDNode *N) {
  SDOperand NewOps[6];
  NewOps[0] = N->getOperand(0);
  for (unsigned i = 1; i < array_lengthof(NewOps); ++i) {
    SDOperand Flag = GetPromotedOp(N->getOperand(i));
    NewOps[i] = DAG.getZeroExtendInReg(Flag, MVT::i1);
  }
  return DAG.UpdateNodeOperands(SDOperand (N, 0), NewOps,
                                array_lengthof(NewOps));
}
