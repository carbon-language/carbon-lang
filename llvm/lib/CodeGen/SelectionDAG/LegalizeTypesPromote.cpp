//===-- LegalizeTypesPromote.cpp - Promotion for LegalizeTypes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

  }      
  
  // If Result is null, the sub-method took care of registering the result.
  if (Result.Val)
    SetPromotedOp(SDOperand(N, ResNo), Result);
}

SDOperand DAGTypeLegalizer::PromoteResult_UNDEF(SDNode *N) {
  return DAG.getNode(ISD::UNDEF, TLI.getTypeToTransformTo(N->getValueType(0)));
}

SDOperand DAGTypeLegalizer::PromoteResult_Constant(SDNode *N) {
  MVT::ValueType VT = N->getValueType(0);
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

  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  assert(MVT::getSizeInBits(Res.getValueType()) >= MVT::getSizeInBits(NVT) &&
         "Truncation doesn't make sense!");
  if (Res.getValueType() == NVT)
    return Res;

  // Truncate to NVT instead of VT
  return DAG.getNode(ISD::TRUNCATE, NVT, Res);
}

SDOperand DAGTypeLegalizer::PromoteResult_INT_EXTEND(SDNode *N) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));

  if (getTypeAction(N->getOperand(0).getValueType()) == Promote) {
    SDOperand Res = GetPromotedOp(N->getOperand(0));
    assert(MVT::getSizeInBits(Res.getValueType()) <= MVT::getSizeInBits(NVT) &&
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
  return DAG.getNode(ISD::FP_ROUND_INREG, N->getOperand(0).getValueType(),
                     N->getOperand(0), DAG.getValueType(N->getValueType(0)));
}

SDOperand DAGTypeLegalizer::PromoteResult_FP_TO_XINT(SDNode *N) {
  SDOperand Op = N->getOperand(0);
  // If the operand needed to be promoted, do so now.
  if (getTypeAction(Op.getValueType()) == Promote)
    // The input result is prerounded, so we don't have to do anything special.
    Op = GetPromotedOp(Op);
  
  unsigned NewOpc = N->getOpcode();
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  
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
  assert(isTypeLegal(TLI.getSetCCResultTy()) && "SetCC type is not legal??");
  return DAG.getNode(ISD::SETCC, TLI.getSetCCResultTy(), N->getOperand(0),
                     N->getOperand(1), N->getOperand(2));
}

SDOperand DAGTypeLegalizer::PromoteResult_LOAD(LoadSDNode *N) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  ISD::LoadExtType ExtType =
    ISD::isNON_EXTLoad(N) ? ISD::EXTLOAD : N->getExtensionType();
  SDOperand Res = DAG.getExtLoad(ExtType, NVT, N->getChain(), N->getBasePtr(),
                                 N->getSrcValue(), N->getSrcValueOffset(),
                                 N->getLoadedVT(), N->isVolatile(),
                                 N->getAlignment());

  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDOperand(N, 1), Res.getValue(1));
  return Res;
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
  MVT::ValueType VT = N->getValueType(0);
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
  MVT::ValueType VT = N->getValueType(0);
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
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  SDOperand Res = GetPromotedOp(N->getOperand(0));
  Res = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Res, DAG.getValueType(VT));
  return DAG.getNode(ISD::SRA, NVT, Res, N->getOperand(1));
}

SDOperand DAGTypeLegalizer::PromoteResult_SRL(SDNode *N) {
  // The input value must be properly zero extended.
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
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
    
  case ISD::SELECT:      Res = PromoteOperand_SELECT(N, OpNo); break;
  case ISD::BRCOND:      Res = PromoteOperand_BRCOND(N, OpNo); break;
  case ISD::BR_CC:       Res = PromoteOperand_BR_CC(N, OpNo); break;
  case ISD::SETCC:       Res = PromoteOperand_SETCC(N, OpNo); break;

  case ISD::STORE:       Res = PromoteOperand_STORE(cast<StoreSDNode>(N),
                                                    OpNo); break;
  case ISD::MEMSET:
  case ISD::MEMCPY:
  case ISD::MEMMOVE:     Res = HandleMemIntrinsic(N); break;
  }
  
  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.Val) return false;
  // If the result is N, the sub-method updated N in place.
  if (Res.Val == N) {
    // Mark N as new and remark N and its operands.  This allows us to correctly
    // revisit N if it needs another step of promotion and allows us to visit
    // any new operands to N.
    N->setNodeId(NewNode);
    MarkNewNodes(N);
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
  return DAG.getNode(ISD::FP_ROUND, N->getValueType(0), Op);
}

SDOperand DAGTypeLegalizer::PromoteOperand_INT_TO_FP(SDNode *N) {
  SDOperand In = GetPromotedOp(N->getOperand(0));
  MVT::ValueType OpVT = N->getOperand(0).getValueType();
  if (N->getOpcode() == ISD::UINT_TO_FP)
    In = DAG.getZeroExtendInReg(In, OpVT);
  else
    In = DAG.getNode(ISD::SIGN_EXTEND_INREG, In.getValueType(),
                     In, DAG.getValueType(OpVT));
  
  return DAG.UpdateNodeOperands(SDOperand(N, 0), In);
}

SDOperand DAGTypeLegalizer::PromoteOperand_SELECT(SDNode *N, unsigned OpNo) {
  assert(OpNo == 0 && "Only know how to promote condition");
  SDOperand Cond = GetPromotedOp(N->getOperand(0));  // Promote the condition.

  // The top bits of the promoted condition are not necessarily zero, ensure
  // that the value is properly zero extended.
  if (!DAG.MaskedValueIsZero(Cond, 
                             MVT::getIntVTBitMask(Cond.getValueType())^1)) {
    Cond = DAG.getZeroExtendInReg(Cond, MVT::i1);
    MarkNewNodes(Cond.Val); 
  }

  // The chain (Op#0) and basic block destination (Op#2) are always legal types.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), Cond, N->getOperand(1),
                                N->getOperand(2));
}

SDOperand DAGTypeLegalizer::PromoteOperand_BRCOND(SDNode *N, unsigned OpNo) {
  assert(OpNo == 1 && "only know how to promote condition");
  SDOperand Cond = GetPromotedOp(N->getOperand(1));  // Promote the condition.
  
  // The top bits of the promoted condition are not necessarily zero, ensure
  // that the value is properly zero extended.
  if (!DAG.MaskedValueIsZero(Cond, 
                             MVT::getIntVTBitMask(Cond.getValueType())^1)) {
    Cond = DAG.getZeroExtendInReg(Cond, MVT::i1);
    MarkNewNodes(Cond.Val); 
  }
  
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
  MVT::ValueType VT = NewLHS.getValueType();
  
  // Get the promoted values.
  NewLHS = GetPromotedOp(NewLHS);
  NewRHS = GetPromotedOp(NewRHS);
  
  // If this is an FP compare, the operands have already been extended.
  if (!MVT::isInteger(NewLHS.getValueType()))
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
  SDOperand Ch = N->getChain(), Ptr = N->getBasePtr();
  int SVOffset = N->getSrcValueOffset();
  unsigned Alignment = N->getAlignment();
  bool isVolatile = N->isVolatile();
  
  SDOperand Val = GetPromotedOp(N->getValue());  // Get promoted value.

  assert(!N->isTruncatingStore() && "Cannot promote this store operand!");
  
  // Truncate the value and store the result.
  return DAG.getTruncStore(Ch, Val, Ptr, N->getSrcValue(),
                           SVOffset, N->getStoredVT(),
                           isVolatile, Alignment);
}
