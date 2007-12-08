//===-- LegalizeTypesSplit.cpp - Vector Splitting for LegalizeTypes -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements vector splitting support for LegalizeTypes.  Vector
// splitting is the act of changing a computation in an invalid vector type to
// be a computation in multiple vectors of a smaller type.  For example,
// implementing <128 x f32> operations in terms of two <64 x f32> operations.
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
using namespace llvm;

/// GetSplitDestVTs - Compute the VTs needed for the low/hi parts of a vector
/// type that needs to be split.  This handles non-power of two vectors.
static void GetSplitDestVTs(MVT::ValueType InVT,
                            MVT::ValueType &Lo, MVT::ValueType &Hi) {
  MVT::ValueType NewEltVT = MVT::getVectorElementType(InVT);
  unsigned NumElements = MVT::getVectorNumElements(InVT);
  if ((NumElements & (NumElements-1)) == 0) {  // Simple power of two vector.
    NumElements >>= 1;
    Lo = Hi =  MVT::getVectorType(NewEltVT, NumElements);
  } else {                                     // Non-power-of-two vectors.
    unsigned NewNumElts_Lo = 1 << Log2_32(NumElements-1);
    unsigned NewNumElts_Hi = NumElements - NewNumElts_Lo;
    Lo = MVT::getVectorType(NewEltVT, NewNumElts_Lo);
    Hi = MVT::getVectorType(NewEltVT, NewNumElts_Hi);
  }
}


//===----------------------------------------------------------------------===//
//  Result Vector Splitting
//===----------------------------------------------------------------------===//

/// SplitResult - This method is called when the specified result of the
/// specified node is found to need vector splitting.  At this point, the node
/// may also have invalid operands or may have other results that need
/// legalization, we just know that (at least) one result needs vector
/// splitting.
void DAGTypeLegalizer::SplitResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Expand node result: "; N->dump(&DAG); cerr << "\n");
  SDOperand Lo, Hi;
  
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
    cerr << "SplitResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to split the result of this operator!");
    abort();
    
  case ISD::UNDEF:       SplitRes_UNDEF(N, Lo, Hi); break;
  case ISD::LOAD:        SplitRes_LOAD(cast<LoadSDNode>(N), Lo, Hi); break;
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::FDIV:
  case ISD::FPOW:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::UREM:
  case ISD::SREM:
  case ISD::FREM:        SplitRes_BinOp(N, Lo, Hi); break;
  }
  
  // If Lo/Hi is null, the sub-method took care of registering results etc.
  if (Lo.Val)
    SetSplitOp(SDOperand(N, ResNo), Lo, Hi);
}

void DAGTypeLegalizer::SplitRes_UNDEF(SDNode *N, SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType LoVT, HiVT;
  GetSplitDestVTs(N->getValueType(0), LoVT, HiVT);

  Lo = DAG.getNode(ISD::UNDEF, LoVT);
  Hi = DAG.getNode(ISD::UNDEF, HiVT);
}

void DAGTypeLegalizer::SplitRes_LOAD(LoadSDNode *LD, 
                                     SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType LoVT, HiVT;
  GetSplitDestVTs(LD->getValueType(0), LoVT, HiVT);
  
  SDOperand Ch = LD->getChain();
  SDOperand Ptr = LD->getBasePtr();
  const Value *SV = LD->getSrcValue();
  int SVOffset = LD->getSrcValueOffset();
  unsigned Alignment = LD->getAlignment();
  bool isVolatile = LD->isVolatile();
  
  Lo = DAG.getLoad(LoVT, Ch, Ptr, SV, SVOffset, isVolatile, Alignment);
  unsigned IncrementSize = MVT::getSizeInBits(LoVT)/8;
  Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                    getIntPtrConstant(IncrementSize));
  SVOffset += IncrementSize;
  Alignment = MinAlign(Alignment, IncrementSize);
  Hi = DAG.getLoad(HiVT, Ch, Ptr, SV, SVOffset, isVolatile, Alignment);
  
  // Build a factor node to remember that this load is independent of the
  // other one.
  SDOperand TF = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                             Hi.getValue(1));
  
  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDOperand(LD, 1), TF);
}

void DAGTypeLegalizer::SplitRes_BinOp(SDNode *N, SDOperand &Lo, SDOperand &Hi) {
  SDOperand LHSLo, LHSHi;
  GetSplitOp(N->getOperand(0), LHSLo, LHSHi);
  SDOperand RHSLo, RHSHi;
  GetSplitOp(N->getOperand(1), RHSLo, RHSHi);
  
  Lo = DAG.getNode(N->getOpcode(), LHSLo.getValueType(), LHSLo, RHSLo);
  Hi = DAG.getNode(N->getOpcode(), LHSHi.getValueType(), LHSHi, RHSHi);
}


//===----------------------------------------------------------------------===//
//  Operand Vector Splitting
//===----------------------------------------------------------------------===//

/// SplitOperand - This method is called when the specified operand of the
/// specified node is found to need vector splitting.  At this point, all of the
/// result types of the node are known to be legal, but other operands of the
/// node may need legalization as well as the specified one.
bool DAGTypeLegalizer::SplitOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Split node operand: "; N->dump(&DAG); cerr << "\n");
  SDOperand Res(0, 0);
  
#if 0
  if (TLI.getOperationAction(N->getOpcode(), N->getValueType(0)) == 
      TargetLowering::Custom)
    Res = TLI.LowerOperation(SDOperand(N, 0), DAG);
#endif
  
  if (Res.Val == 0) {
    switch (N->getOpcode()) {
    default:
#ifndef NDEBUG
      cerr << "SplitOperand Op #" << OpNo << ": ";
      N->dump(&DAG); cerr << "\n";
#endif
      assert(0 && "Do not know how to split this operator's operand!");
      abort();
#if 0
    case ISD::STORE:
      Res = ExpandOperand_STORE(cast<StoreSDNode>(N), OpNo);
      break;
#endif
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
    N->setNodeId(NewNode);
    MarkNewNodes(N);
    return true;
  }
  
  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");
  
  ReplaceValueWith(SDOperand(N, 0), Res);
  return false;
}
