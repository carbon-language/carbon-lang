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
  Lo = Hi = SDOperand();
  
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
    
#if 0
  case ISD::UNDEF:       SplitResult_UNDEF(N, Lo, Hi); break;
#endif
  }
  
  // If Lo/Hi is null, the sub-method took care of registering results etc.
  if (Lo.Val)
    SetSplitOp(SDOperand(N, ResNo), Lo, Hi);
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
