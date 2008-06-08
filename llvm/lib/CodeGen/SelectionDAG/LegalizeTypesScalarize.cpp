//===-- LegalizeTypesScalarize.cpp - Scalarization for LegalizeTypes ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements scalarization support for LegalizeTypes.  Scalarization
// is the act of changing a computation in an invalid single-element vector type
// to be a computation in its scalar element type.  For example, implementing
// <1 x f32> arithmetic in a scalar f32 register.  This is needed as a base case
// when scalarizing vector arithmetic like <4 x f32>, which eventually
// decomposes to scalars if the target doesn't support v4f32 or v2f32 types.
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  Result Vector Scalarization: <1 x ty> -> ty.
//===----------------------------------------------------------------------===//

void DAGTypeLegalizer::ScalarizeResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Scalarize node result " << ResNo << ": "; N->dump(&DAG); 
        cerr << "\n");
  SDOperand R = SDOperand();
  
  // FIXME: Custom lowering for scalarization?
#if 0
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
#endif
  
  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "ScalarizeResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to scalarize the result of this operator!");
    abort();
    
  case ISD::UNDEF:       R = ScalarizeRes_UNDEF(N); break;
  case ISD::LOAD:        R = ScalarizeRes_LOAD(cast<LoadSDNode>(N)); break;
  case ISD::ADD:
  case ISD::FADD:
  case ISD::SUB:
  case ISD::FSUB:
  case ISD::MUL:
  case ISD::FMUL:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::FDIV:
  case ISD::SREM:
  case ISD::UREM:
  case ISD::FREM:
  case ISD::FPOW:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:         R = ScalarizeRes_BinOp(N); break;
  case ISD::FNEG:
  case ISD::FABS:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:              R = ScalarizeRes_UnaryOp(N); break;
  case ISD::FPOWI:             R = ScalarizeRes_FPOWI(N); break;
  case ISD::BUILD_VECTOR:      R = N->getOperand(0); break;
  case ISD::INSERT_VECTOR_ELT: R = ScalarizeRes_INSERT_VECTOR_ELT(N); break;
  case ISD::VECTOR_SHUFFLE:    R = ScalarizeRes_VECTOR_SHUFFLE(N); break;
  case ISD::BIT_CONVERT:       R = ScalarizeRes_BIT_CONVERT(N); break;
  case ISD::SELECT:            R = ScalarizeRes_SELECT(N); break;
  }
  
  // If R is null, the sub-method took care of registering the result.
  if (R.Val)
    SetScalarizedOp(SDOperand(N, ResNo), R);
}

SDOperand DAGTypeLegalizer::ScalarizeRes_UNDEF(SDNode *N) {
  return DAG.getNode(ISD::UNDEF, N->getValueType(0).getVectorElementType());
}

SDOperand DAGTypeLegalizer::ScalarizeRes_LOAD(LoadSDNode *N) {
  // FIXME: Add support for indexed loads.
  SDOperand Result = DAG.getLoad(N->getValueType(0).getVectorElementType(),
                                 N->getChain(), N->getBasePtr(), 
                                 N->getSrcValue(), N->getSrcValueOffset(),
                                 N->isVolatile(), N->getAlignment());
  
  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDOperand(N, 1), Result.getValue(1));
  return Result;
}

SDOperand DAGTypeLegalizer::ScalarizeRes_BinOp(SDNode *N) {
  SDOperand LHS = GetScalarizedOp(N->getOperand(0));
  SDOperand RHS = GetScalarizedOp(N->getOperand(1));
  return DAG.getNode(N->getOpcode(), LHS.getValueType(), LHS, RHS);
}

SDOperand DAGTypeLegalizer::ScalarizeRes_UnaryOp(SDNode *N) {
  SDOperand Op = GetScalarizedOp(N->getOperand(0));
  return DAG.getNode(N->getOpcode(), Op.getValueType(), Op);
}

SDOperand DAGTypeLegalizer::ScalarizeRes_FPOWI(SDNode *N) {
  SDOperand Op = GetScalarizedOp(N->getOperand(0));
  return DAG.getNode(ISD::FPOWI, Op.getValueType(), Op, N->getOperand(1));
}

SDOperand DAGTypeLegalizer::ScalarizeRes_INSERT_VECTOR_ELT(SDNode *N) {
  // The value to insert may have a wider type than the vector element type,
  // so be sure to truncate it to the element type if necessary.
  SDOperand Op = N->getOperand(1);
  MVT EltVT = N->getValueType(0).getVectorElementType();
  if (Op.getValueType().bitsGT(EltVT))
    Op = DAG.getNode(ISD::TRUNCATE, EltVT, Op);
  assert(Op.getValueType() == EltVT && "Invalid type for inserted value!");
  return Op;
}

SDOperand DAGTypeLegalizer::ScalarizeRes_VECTOR_SHUFFLE(SDNode *N) {
  // Figure out if the scalar is the LHS or RHS and return it.
  SDOperand EltNum = N->getOperand(2).getOperand(0);
  unsigned Op = cast<ConstantSDNode>(EltNum)->getValue() != 0;
  return GetScalarizedOp(N->getOperand(Op));
}

SDOperand DAGTypeLegalizer::ScalarizeRes_BIT_CONVERT(SDNode *N) {
  MVT NewVT = N->getValueType(0).getVectorElementType();
  return DAG.getNode(ISD::BIT_CONVERT, NewVT, N->getOperand(0));
}

SDOperand DAGTypeLegalizer::ScalarizeRes_SELECT(SDNode *N) {
  SDOperand LHS = GetScalarizedOp(N->getOperand(1));
  return DAG.getNode(ISD::SELECT, LHS.getValueType(), N->getOperand(0), LHS,
                     GetScalarizedOp(N->getOperand(2)));
}


//===----------------------------------------------------------------------===//
//  Operand Vector Scalarization <1 x ty> -> ty.
//===----------------------------------------------------------------------===//

bool DAGTypeLegalizer::ScalarizeOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Scalarize node operand " << OpNo << ": "; N->dump(&DAG); 
        cerr << "\n");
  SDOperand Res(0, 0);
  
  // FIXME: Should we support custom lowering for scalarization?
#if 0
  if (TLI.getOperationAction(N->getOpcode(), N->getValueType(0)) == 
      TargetLowering::Custom)
    Res = TLI.LowerOperation(SDOperand(N, 0), DAG);
#endif
  
  if (Res.Val == 0) {
    switch (N->getOpcode()) {
    default:
#ifndef NDEBUG
      cerr << "ScalarizeOperand Op #" << OpNo << ": ";
      N->dump(&DAG); cerr << "\n";
#endif
      assert(0 && "Do not know how to scalarize this operator's operand!");
      abort();
      
    case ISD::BIT_CONVERT:
      Res = ScalarizeOp_BIT_CONVERT(N); break;

    case ISD::EXTRACT_VECTOR_ELT:
      Res = ScalarizeOp_EXTRACT_VECTOR_ELT(N); break;

    case ISD::STORE:
      Res = ScalarizeOp_STORE(cast<StoreSDNode>(N), OpNo); break;
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

/// ScalarizeOp_BIT_CONVERT - If the value to convert is a vector that needs
/// to be scalarized, it must be <1 x ty>.  Convert the element instead.
SDOperand DAGTypeLegalizer::ScalarizeOp_BIT_CONVERT(SDNode *N) {
  SDOperand Elt = GetScalarizedOp(N->getOperand(0));
  return DAG.getNode(ISD::BIT_CONVERT, N->getValueType(0), Elt);
}

/// ScalarizeOp_EXTRACT_VECTOR_ELT - If the input is a vector that needs to be
/// scalarized, it must be <1 x ty>, so just return the element, ignoring the
/// index.
SDOperand DAGTypeLegalizer::ScalarizeOp_EXTRACT_VECTOR_ELT(SDNode *N) {
  return GetScalarizedOp(N->getOperand(0));
}

/// ScalarizeOp_STORE - If the value to store is a vector that needs to be
/// scalarized, it must be <1 x ty>.  Just store the element.
SDOperand DAGTypeLegalizer::ScalarizeOp_STORE(StoreSDNode *N, unsigned OpNo) {
  // FIXME: Add support for indexed stores.
  assert(OpNo == 1 && "Do not know how to scalarize this operand!");
  return DAG.getStore(N->getChain(), GetScalarizedOp(N->getOperand(1)),
                      N->getBasePtr(), N->getSrcValue(), N->getSrcValueOffset(),
                      N->isVolatile(), N->getAlignment());
}
