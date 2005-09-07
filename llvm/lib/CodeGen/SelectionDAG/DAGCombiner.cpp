//===-- DAGCombiner.cpp - Implement a DAG node combiner -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass combines dag nodes to form fewer, simpler DAG nodes.  It can be run
// both before and after the DAG is legalized.
//
// FIXME: Missing folds
// sdiv, udiv, srem, urem (X, const) where X is an integer can be expanded into
//  a sequence of multiplies, shifts, and adds.  This should be controlled by
//  some kind of hint from the target that int div is expensive.
// various folds of mulh[s,u] by constants such as -1, powers of 2, etc.
//
// FIXME: Should add a corresponding version of fold AND with
// ZERO_EXTEND/SIGN_EXTEND by converting them to an ANY_EXTEND node which
// we don't have yet.
//
// FIXME: mul (x, const) -> shifts + adds
// FIXME: undef values
// FIXME: zero extend when top bits are 0 -> drop it ?
// FIXME: make truncate see through SIGN_EXTEND and AND
// FIXME: sext_in_reg(setcc) on targets that return zero or one, and where 
//        EVT != MVT::i1 can drop the sext.
// FIXME: (sra (sra x, c1), c2) -> (sra x, c1+c2)
// FIXME: verify that getNode can't return extends with an operand whose type
//        is >= to that of the extend.
// FIXME: divide by zero is currently left unfolded.  do we want to turn this
//        into an undef?
// 
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dagcombine"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetLowering.h"
#include <cmath>
using namespace llvm;

namespace {
  Statistic<> NodesCombined ("dagcombiner", "Number of dag nodes combined");

  class DAGCombiner {
    SelectionDAG &DAG;
    TargetLowering &TLI;
    bool AfterLegalize;

    // Worklist of all of the nodes that need to be simplified.
    std::vector<SDNode*> WorkList;

    /// AddUsersToWorkList - When an instruction is simplified, add all users of
    /// the instruction to the work lists because they might get more simplified
    /// now.
    ///
    void AddUsersToWorkList(SDNode *N) {
      for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end();
           UI != UE; ++UI)
        WorkList.push_back(*UI);
    }

    /// removeFromWorkList - remove all instances of N from the worklist.
    void removeFromWorkList(SDNode *N) {
      WorkList.erase(std::remove(WorkList.begin(), WorkList.end(), N),
                     WorkList.end());
    }
    
    /// visit - call the node-specific routine that knows how to fold each
    /// particular type of node.
    SDOperand visit(SDNode *N);

    // Visitation implementation - Implement dag node combining for different
    // node types.  The semantics are as follows:
    // Return Value:
    //   SDOperand.Val == 0   - No change was made
    //   otherwise            - N should be replaced by the returned Operand.
    //
    SDOperand visitTokenFactor(SDNode *N);
    SDOperand visitADD(SDNode *N);
    SDOperand visitSUB(SDNode *N);
    SDOperand visitMUL(SDNode *N);
    SDOperand visitSDIV(SDNode *N);
    SDOperand visitUDIV(SDNode *N);
    SDOperand visitSREM(SDNode *N);
    SDOperand visitUREM(SDNode *N);
    SDOperand visitMULHU(SDNode *N);
    SDOperand visitMULHS(SDNode *N);
    SDOperand visitAND(SDNode *N);
    SDOperand visitOR(SDNode *N);
    SDOperand visitXOR(SDNode *N);
    SDOperand visitSHL(SDNode *N);
    SDOperand visitSRA(SDNode *N);
    SDOperand visitSRL(SDNode *N);
    SDOperand visitCTLZ(SDNode *N);
    SDOperand visitCTTZ(SDNode *N);
    SDOperand visitCTPOP(SDNode *N);
    // select
    // select_cc
    // setcc
    SDOperand visitSIGN_EXTEND(SDNode *N);
    SDOperand visitZERO_EXTEND(SDNode *N);
    SDOperand visitSIGN_EXTEND_INREG(SDNode *N);
    SDOperand visitTRUNCATE(SDNode *N);
    SDOperand visitSINT_TO_FP(SDNode *N);
    SDOperand visitUINT_TO_FP(SDNode *N);
    SDOperand visitFP_TO_SINT(SDNode *N);
    SDOperand visitFP_TO_UINT(SDNode *N);
    SDOperand visitFP_ROUND(SDNode *N);
    SDOperand visitFP_ROUND_INREG(SDNode *N);
    SDOperand visitFP_EXTEND(SDNode *N);
    SDOperand visitFNEG(SDNode *N);
    SDOperand visitFABS(SDNode *N);
    // brcond
    // brcondtwoway
    // br_cc
    // brtwoway_cc
public:
    DAGCombiner(SelectionDAG &D)
      : DAG(D), TLI(D.getTargetLoweringInfo()), AfterLegalize(false) {}
    
    /// Run - runs the dag combiner on all nodes in the work list
    void Run(bool RunningAfterLegalize); 
  };
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  V and Mask are known to
/// be the same type.
static bool MaskedValueIsZero(const SDOperand &Op, uint64_t Mask,
                              const TargetLowering &TLI) {
  unsigned SrcBits;
  if (Mask == 0) return true;
  
  // If we know the result of a setcc has the top bits zero, use this info.
  switch (Op.getOpcode()) {
  case ISD::Constant:
    return (cast<ConstantSDNode>(Op)->getValue() & Mask) == 0;
  case ISD::SETCC:
    // FIXME: teach this about non ZeroOrOne values, such as 0 or -1
    return ((Mask & 1) == 0) &&
    TLI.getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult;
  case ISD::ZEXTLOAD:
    SrcBits = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(3))->getVT());
    return (Mask & ((1ULL << SrcBits)-1)) == 0; // Returning only the zext bits.
  case ISD::ZERO_EXTEND:
    SrcBits = MVT::getSizeInBits(Op.getOperand(0).getValueType());
    return MaskedValueIsZero(Op.getOperand(0),Mask & ((1ULL << SrcBits)-1),TLI);
  case ISD::AssertZext:
    SrcBits = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(1))->getVT());
    return (Mask & ((1ULL << SrcBits)-1)) == 0; // Returning only the zext bits.
  case ISD::AND:
    // (X & C1) & C2 == 0   iff   C1 & C2 == 0.
    if (ConstantSDNode *AndRHS = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      return MaskedValueIsZero(Op.getOperand(0),AndRHS->getValue() & Mask, TLI);
    // FALL THROUGH
  case ISD::OR:
  case ISD::XOR:
    return MaskedValueIsZero(Op.getOperand(0), Mask, TLI) &&
    MaskedValueIsZero(Op.getOperand(1), Mask, TLI);
  case ISD::SELECT:
    return MaskedValueIsZero(Op.getOperand(1), Mask, TLI) &&
    MaskedValueIsZero(Op.getOperand(2), Mask, TLI);
  case ISD::SELECT_CC:
    return MaskedValueIsZero(Op.getOperand(2), Mask, TLI) &&
    MaskedValueIsZero(Op.getOperand(3), Mask, TLI);
  case ISD::SRL:
    // (ushr X, C1) & C2 == 0   iff  X & (C2 << C1) == 0
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t NewVal = Mask << ShAmt->getValue();
      SrcBits = MVT::getSizeInBits(Op.getValueType());
      if (SrcBits != 64) NewVal &= (1ULL << SrcBits)-1;
      return MaskedValueIsZero(Op.getOperand(0), NewVal, TLI);
    }
    return false;
  case ISD::SHL:
    // (ushl X, C1) & C2 == 0   iff  X & (C2 >> C1) == 0
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t NewVal = Mask >> ShAmt->getValue();
      return MaskedValueIsZero(Op.getOperand(0), NewVal, TLI);
    }
    return false;
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP:
    // Bit counting instructions can not set the high bits of the result
    // register.  The max number of bits sets depends on the input.
    return (Mask & (MVT::getSizeInBits(Op.getValueType())*2-1)) == 0;

    // TODO we could handle some SRA cases here.
  default: break;
  }
  return false;
}

// isSetCCEquivalent - Return true if this node is a setcc, or is a select_cc
// that selects between the values 1 and 0, making it equivalent to a setcc.
// Also, set the incoming LHS, RHS, and CC references to the appropriate 
// nodes based on the type of node we are checking.  This simplifies life a
// bit for the callers.
static bool isSetCCEquivalent(SDOperand N, SDOperand &LHS, SDOperand &RHS,
                              SDOperand &CC) {
  if (N.getOpcode() == ISD::SETCC) {
    LHS = N.getOperand(0);
    RHS = N.getOperand(1);
    CC  = N.getOperand(2);
    return true;
  }
  if (N.getOpcode() == ISD::SELECT_CC && 
      N.getOperand(2).getOpcode() == ISD::Constant &&
      N.getOperand(3).getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(N.getOperand(2))->getValue() == 1 &&
      cast<ConstantSDNode>(N.getOperand(3))->isNullValue()) {
    LHS = N.getOperand(0);
    RHS = N.getOperand(1);
    CC  = N.getOperand(4);
    return true;
  }
  return false;
}

// isInvertibleForFree - Return true if there is no cost to emitting the logical
// inverse of this node.
static bool isInvertibleForFree(SDOperand N) {
  SDOperand N0, N1, N2;
  if (isa<ConstantSDNode>(N.Val)) return true;
  if (isSetCCEquivalent(N, N0, N1, N2) && N.Val->hasOneUse())
    return true;
  return false;
}

void DAGCombiner::Run(bool RunningAfterLegalize) {
  // set the instance variable, so that the various visit routines may use it.
  AfterLegalize = RunningAfterLegalize;

  // Add all the dag nodes to the worklist.
  WorkList.insert(WorkList.end(), DAG.allnodes_begin(), DAG.allnodes_end());
  
  // while the worklist isn't empty, inspect the node on the end of it and
  // try and combine it.
  while (!WorkList.empty()) {
    SDNode *N = WorkList.back();
    WorkList.pop_back();
    
    // If N has no uses, it is dead.  Make sure to revisit all N's operands once
    // N is deleted from the DAG, since they too may now be dead.
    // FIXME: is there a better way to keep from deleting the dag root because
    // we think it has no uses?  This works for now...
    if (N->use_empty() && N != DAG.getRoot().Val) {
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
        WorkList.push_back(N->getOperand(i).Val);
      
      DAG.DeleteNode(N);
      removeFromWorkList(N);
      continue;
    }
    
    SDOperand RV = visit(N);
    if (RV.Val) {
      ++NodesCombined;
      // If we get back the same node we passed in, rather than a new node or
      // zero, we know that the node must have defined multiple values and
      // CombineTo was used.  Since CombineTo takes care of the worklist 
      // mechanics for us, we have no work to do in this case.
      if (RV.Val != N) {
        DEBUG(std::cerr << "\nReplacing "; N->dump();
              std::cerr << "\nWith: "; RV.Val->dump();
              std::cerr << '\n');
        DAG.ReplaceAllUsesWith(SDOperand(N, 0), RV);
          
        // Push the new node and any users onto the worklist
        WorkList.push_back(RV.Val);
        AddUsersToWorkList(RV.Val);
          
        // Nodes can end up on the worklist more than once.  Make sure we do
        // not process a node that has been replaced.
        removeFromWorkList(N);
      }
    }
  }
}

SDOperand DAGCombiner::visit(SDNode *N) {
  switch(N->getOpcode()) {
  default: break;
  case ISD::TokenFactor:        return visitTokenFactor(N);
  case ISD::ADD:                return visitADD(N);
  case ISD::SUB:                return visitSUB(N);
  case ISD::MUL:                return visitMUL(N);
  case ISD::SDIV:               return visitSDIV(N);
  case ISD::UDIV:               return visitUDIV(N);
  case ISD::SREM:               return visitSREM(N);
  case ISD::UREM:               return visitUREM(N);
  case ISD::MULHU:              return visitMULHU(N);
  case ISD::MULHS:              return visitMULHS(N);
  case ISD::AND:                return visitAND(N);
  case ISD::OR:                 return visitOR(N);
  case ISD::XOR:                return visitXOR(N);
  case ISD::SHL:                return visitSHL(N);
  case ISD::SRA:                return visitSRA(N);
  case ISD::SRL:                return visitSRL(N);
  case ISD::CTLZ:               return visitCTLZ(N);
  case ISD::CTTZ:               return visitCTTZ(N);
  case ISD::CTPOP:              return visitCTPOP(N);
  case ISD::SIGN_EXTEND:        return visitSIGN_EXTEND(N);
  case ISD::ZERO_EXTEND:        return visitZERO_EXTEND(N);
  case ISD::SIGN_EXTEND_INREG:  return visitSIGN_EXTEND_INREG(N);
  case ISD::TRUNCATE:           return visitTRUNCATE(N);
  case ISD::SINT_TO_FP:         return visitSINT_TO_FP(N);
  case ISD::UINT_TO_FP:         return visitUINT_TO_FP(N);
  case ISD::FP_TO_SINT:         return visitFP_TO_SINT(N);
  case ISD::FP_TO_UINT:         return visitFP_TO_UINT(N);
  case ISD::FP_ROUND:           return visitFP_ROUND(N);
  case ISD::FP_ROUND_INREG:     return visitFP_ROUND_INREG(N);
  case ISD::FP_EXTEND:          return visitFP_EXTEND(N);
  case ISD::FNEG:               return visitFNEG(N);
  case ISD::FABS:               return visitFABS(N);
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitTokenFactor(SDNode *N) {
  // If the token factor has two operands and one is the entry token, replace
  // the token factor with the other operand.
  if (N->getNumOperands() == 2) {
    if (N->getOperand(0).getOpcode() == ISD::EntryToken)
      return N->getOperand(1);
    if (N->getOperand(1).getOpcode() == ISD::EntryToken)
      return N->getOperand(0);
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitADD(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  
  // fold (add c1, c2) -> c1+c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() + N1C->getValue(), VT);
  // fold (add x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (add (add x, c1), c2) -> (add x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::ADD && 
      N0.getOperand(1).getOpcode() == ISD::Constant)
    return DAG.getNode(ISD::ADD, VT, N0.getOperand(0), 
                       DAG.getConstant(N1C->getValue() +
                             cast<ConstantSDNode>(N0.getOperand(1))->getValue(),
                             VT));
  // fold floating point (add c1, c2) -> c1+c2
  if (N0CFP && N1CFP)
    return DAG.getConstantFP(N0CFP->getValue() + N1CFP->getValue(), VT);
  // fold (A + (-B)) -> A-B
  if (N1.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::SUB, VT, N0, N1.getOperand(0));
  // fold ((-A) + B) -> B-A
  if (N0.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::SUB, VT, N1, N0.getOperand(0));
  // fold ((0-A) + B) -> B-A
  if (N0.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N0.getOperand(0)) &&
      cast<ConstantSDNode>(N0.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, VT, N1, N0.getOperand(1));
  // fold (A + (0-B)) -> A-B
  if (N1.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N1.getOperand(0)) &&
      cast<ConstantSDNode>(N1.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, VT, N0, N1.getOperand(1));
  // fold (A+(B-A)) -> B for non-fp types
  if (N1.getOpcode() == ISD::SUB && N0 == N1.getOperand(1) &&
      !MVT::isFloatingPoint(N1.getValueType()))
    return N1.getOperand(0);
  return SDOperand();
}

SDOperand DAGCombiner::visitSUB(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0.Val);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1.Val);
  
  // fold (sub c1, c2) -> c1-c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() - N1C->getValue(),
                           N->getValueType(0));
  // fold (sub x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold floating point (sub c1, c2) -> c1-c2
  if (N0CFP && N1CFP)
    return DAG.getConstantFP(N0CFP->getValue() - N1CFP->getValue(),
                             N->getValueType(0));
  // fold (A+B)-A -> B
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(0) == N1 &&
      !MVT::isFloatingPoint(N1.getValueType()))
    return N0.getOperand(1);
  // fold (A+B)-B -> A
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(1) == N1 &&
      !MVT::isFloatingPoint(N1.getValueType()))
    return N0.getOperand(0);
  // fold (A-(-B)) -> A+B
  if (N1.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::ADD, N0.getValueType(), N0, N1.getOperand(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitMUL(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  
  // fold (mul c1, c2) -> c1*c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() * N1C->getValue(),
                           N->getValueType(0));
  // fold (mul x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mul x, -1) -> 0-x
  if (N1C && N1C->isAllOnesValue())
    return DAG.getNode(ISD::SUB, N->getValueType(0), 
                       DAG.getConstant(0, N->getValueType(0)), N0);
  // fold (mul x, (1 << c)) -> x << c
  if (N1C && isPowerOf2_64(N1C->getValue()))
    return DAG.getNode(ISD::SHL, N->getValueType(0), N0,
                       DAG.getConstant(Log2_64(N1C->getValue()),
                                       TLI.getShiftAmountTy()));
  // fold floating point (mul c1, c2) -> c1*c2
  if (N0CFP && N1CFP)
    return DAG.getConstantFP(N0CFP->getValue() * N1CFP->getValue(),
                             N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitSDIV(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0.Val);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1.Val);

  // fold (sdiv c1, c2) -> c1/c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getConstant(N0C->getSignExtended() / N1C->getSignExtended(),
                           N->getValueType(0));
  // fold floating point (sdiv c1, c2) -> c1/c2
  if (N0CFP && N1CFP)
    return DAG.getConstantFP(N0CFP->getValue() / N1CFP->getValue(),
                             N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitUDIV(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  
  // fold (udiv c1, c2) -> c1/c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getConstant(N0C->getValue() / N1C->getValue(),
                           N->getValueType(0));
  // fold (udiv x, (1 << c)) -> x >>u c
  if (N1C && isPowerOf2_64(N1C->getValue()))
    return DAG.getNode(ISD::SRL, N->getValueType(0), N0,
                       DAG.getConstant(Log2_64(N1C->getValue()),
                                       TLI.getShiftAmountTy()));
  return SDOperand();
}

SDOperand DAGCombiner::visitSREM(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  
  // fold (srem c1, c2) -> c1%c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getConstant(N0C->getSignExtended() % N1C->getSignExtended(),
                           N->getValueType(0));
  // fold floating point (srem c1, c2) -> fmod(c1, c2)
  if (N0CFP && N1CFP)
    return DAG.getConstantFP(fmod(N0CFP->getValue(),N1CFP->getValue()),
                             N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitUREM(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // fold (urem c1, c2) -> c1%c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getConstant(N0C->getValue() % N1C->getValue(),
                           N->getValueType(0));
  // FIXME: c2 power of 2 -> mask?
  return SDOperand();
}

SDOperand DAGCombiner::visitMULHS(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // fold (mulhs x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mulhs x, 1) -> (sra x, size(x)-1)
  if (N1C && N1C->getValue() == 1)
    return DAG.getNode(ISD::SRA, N0.getValueType(), N0, 
                       DAG.getConstant(MVT::getSizeInBits(N0.getValueType())-1,
                                       TLI.getShiftAmountTy()));
  return SDOperand();
}

SDOperand DAGCombiner::visitMULHU(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // fold (mulhu x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mulhu x, 1) -> 0
  if (N1C && N1C->getValue() == 1)
    return DAG.getConstant(0, N0.getValueType());
  return SDOperand();
}

SDOperand DAGCombiner::visitAND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N1.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (and c1, c2) -> c1&c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() & N1C->getValue(), VT);
  // fold (and x, -1) -> x
  if (N1C && N1C->isAllOnesValue())
    return N0;
  // if (and x, c) is known to be zero, return 0
  if (N1C && MaskedValueIsZero(SDOperand(N, 0), ~0ULL >> (64-OpSizeInBits),TLI))
    return DAG.getConstant(0, VT);
  // fold (and x, c) -> x iff (x & ~c) == 0
  if (N1C && MaskedValueIsZero(N0,~N1C->getValue() & (~0ULL>>(64-OpSizeInBits)),
                               TLI))
    return N0;
  // fold (and (sign_extend_inreg x, i16 to i32), 1) -> (and x, 1)
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    unsigned ExtendBits =
    MVT::getSizeInBits(cast<VTSDNode>(N0.getOperand(1))->getVT());
    if ((N1C->getValue() & (~0ULL << ExtendBits)) == 0)
      return DAG.getNode(ISD::AND, VT, N0.getOperand(0), N1);
  }
  // fold (and (or x, 0xFFFF), 0xFF) -> 0xFF
  if (N0.getOpcode() == ISD::OR)
    if (ConstantSDNode *ORI = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
      if ((ORI->getValue() & N1C->getValue()) == N1C->getValue())
        return N1;
  return SDOperand();
}

SDOperand DAGCombiner::visitOR(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N1.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (or c1, c2) -> c1|c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() | N1C->getValue(),
                           N->getValueType(0));
  // fold (or x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (or x, -1) -> -1
  if (N1C && N1C->isAllOnesValue())
    return N1;
  // fold (or x, c) -> c iff (x & ~c) == 0
  if (N1C && MaskedValueIsZero(N0,~N1C->getValue() & (~0ULL>>(64-OpSizeInBits)),
                               TLI))
    return N1;
  return SDOperand();
}

SDOperand DAGCombiner::visitXOR(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand LHS, RHS, CC;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  
  // fold (xor c1, c2) -> c1^c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() ^ N1C->getValue(), VT);
  // fold (xor x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold !(x cc y) -> (x !cc y)
  if (N1C && N1C->getValue() == 1 && isSetCCEquivalent(N0, LHS, RHS, CC)) {
    bool isInt = MVT::isInteger(LHS.getValueType());
    ISD::CondCode NotCC = ISD::getSetCCInverse(cast<CondCodeSDNode>(CC)->get(),
                                               isInt);
    if (N0.getOpcode() == ISD::SETCC)
      return DAG.getSetCC(VT, LHS, RHS, NotCC);
    if (N0.getOpcode() == ISD::SELECT_CC)
      return DAG.getSelectCC(LHS, RHS, N0.getOperand(2),N0.getOperand(3),NotCC);
    assert(0 && "Unhandled SetCC Equivalent!");
    abort();
  }
  // fold !(x or y) -> (!x and !y) iff x or y are freely invertible
  if (N1C && N1C->isAllOnesValue() && N0.getOpcode() == ISD::OR) {
    SDOperand LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isInvertibleForFree(RHS) || isInvertibleForFree(LHS)) {
      LHS = DAG.getNode(ISD::XOR, VT, LHS, N1);  // RHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, VT, RHS, N1);  // RHS = ~RHS
      return DAG.getNode(ISD::AND, VT, LHS, RHS);
    }
  }
  // fold !(x and y) -> (!x or !y) iff x or y are freely invertible
  if (N1C && N1C->isAllOnesValue() && N0.getOpcode() == ISD::AND) {
    SDOperand LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isInvertibleForFree(RHS) || isInvertibleForFree(LHS)) {
      LHS = DAG.getNode(ISD::XOR, VT, LHS, N1);  // RHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, VT, RHS, N1);  // RHS = ~RHS
      return DAG.getNode(ISD::OR, VT, LHS, RHS);
    }
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitSHL(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (shl c1, c2) -> c1<<c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() << N1C->getValue(), VT);
  // fold (shl 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (shl x, c >= size(x)) -> undef
  if (N1C && N1C->getValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (shl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // if (shl x, c) is known to be zero, return 0
  if (N1C && MaskedValueIsZero(SDOperand(N, 0), ~0ULL >> (64-OpSizeInBits),TLI))
    return DAG.getConstant(0, VT);
  // fold (shl (shl x, c1), c2) -> 0 or (shl x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::SHL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N1C->getValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SHL, VT, N0.getOperand(0), 
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }
  // fold (shl (srl x, c1), c2) -> (shl (and x, -1 << c1), c2-c1) or
  //                               (srl (and x, -1 << c1), c1-c2)
  if (N1C && N0.getOpcode() == ISD::SRL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N1C->getValue();
    SDOperand Mask = DAG.getNode(ISD::AND, VT, N0.getOperand(0),
                                 DAG.getConstant(~0ULL << c1, VT));
    if (c2 > c1)
      return DAG.getNode(ISD::SHL, VT, Mask, 
                         DAG.getConstant(c2-c1, N1.getValueType()));
    else
      return DAG.getNode(ISD::SRL, VT, Mask, 
                         DAG.getConstant(c1-c2, N1.getValueType()));
  }
  // fold (shl (sra x, c1), c1) -> (and x, -1 << c1)
  if (N1C && N0.getOpcode() == ISD::SRA && N1 == N0.getOperand(1))
    return DAG.getNode(ISD::AND, VT, N0.getOperand(0),
                       DAG.getConstant(~0ULL << N1C->getValue(), VT));
  return SDOperand();
}

SDOperand DAGCombiner::visitSRA(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (sra c1, c2) -> c1>>c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getSignExtended() >> N1C->getValue(), VT);
  // fold (sra 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (sra -1, x) -> -1
  if (N0C && N0C->isAllOnesValue())
    return N0;
  // fold (sra x, c >= size(x)) -> undef
  if (N1C && N1C->getValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (sra x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // If the sign bit is known to be zero, switch this to a SRL.
  if (N1C && MaskedValueIsZero(N0, (1ULL << (OpSizeInBits-1)), TLI))
    return DAG.getNode(ISD::SRL, VT, N0, N1);
  return SDOperand();
}

SDOperand DAGCombiner::visitSRL(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (srl c1, c2) -> c1 >>u c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() >> N1C->getValue(), VT);
  // fold (srl 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (srl x, c >= size(x)) -> undef
  if (N1C && N1C->getValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (srl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // if (srl x, c) is known to be zero, return 0
  if (N1C && MaskedValueIsZero(SDOperand(N, 0), ~0ULL >> (64-OpSizeInBits),TLI))
    return DAG.getConstant(0, VT);
  // fold (srl (srl x, c1), c2) -> 0 or (srl x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::SRL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N1C->getValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SRL, VT, N0.getOperand(0), 
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitCTLZ(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);

  // fold (ctlz c1) -> c2
  if (N0C)
    return DAG.getConstant(CountLeadingZeros_64(N0C->getValue()),
                           N0.getValueType());
  return SDOperand();
}

SDOperand DAGCombiner::visitCTTZ(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  
  // fold (cttz c1) -> c2
  if (N0C)
    return DAG.getConstant(CountTrailingZeros_64(N0C->getValue()),
                           N0.getValueType());
  return SDOperand();
}

SDOperand DAGCombiner::visitCTPOP(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  
  // fold (ctpop c1) -> c2
  if (N0C)
    return DAG.getConstant(CountPopulation_64(N0C->getValue()),
                           N0.getValueType());
  return SDOperand();
}

SDOperand DAGCombiner::visitSIGN_EXTEND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);

  // fold (sext c1) -> c1
  if (N0C)
    return DAG.getConstant(N0C->getSignExtended(), VT);
  // fold (sext (sext x)) -> (sext x)
  if (N0.getOpcode() == ISD::SIGN_EXTEND)
    return DAG.getNode(ISD::SIGN_EXTEND, VT, N0.getOperand(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitZERO_EXTEND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);

  // fold (zext c1) -> c1
  if (N0C)
    return DAG.getConstant(N0C->getValue(), VT);
  // fold (zext (zext x)) -> (zext x)
  if (N0.getOpcode() == ISD::ZERO_EXTEND)
    return DAG.getNode(ISD::ZERO_EXTEND, VT, N0.getOperand(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitSIGN_EXTEND_INREG(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand LHS, RHS, CC;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N1)->getVT();
  
  // fold (sext_in_reg c1) -> c1
  if (N0C) {
    SDOperand Truncate = DAG.getConstant(N0C->getValue(), EVT);
    return DAG.getNode(ISD::SIGN_EXTEND, VT, Truncate);
  }
  // fold (sext_in_reg (sext_in_reg x, VT2), VT1) -> (sext_in_reg x, minVT) pt1
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG && 
      cast<VTSDNode>(N0.getOperand(1))->getVT() < EVT) {
    return N0;
  }
  // fold (sext_in_reg (sext_in_reg x, VT2), VT1) -> (sext_in_reg x, minVT) pt2
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      EVT < cast<VTSDNode>(N0.getOperand(1))->getVT()) {
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, VT, N0.getOperand(0), N1);
  }
  // fold (sext_in_reg (assert_sext x)) -> (assert_sext x)
  if (N0.getOpcode() == ISD::AssertSext && 
      cast<VTSDNode>(N0.getOperand(1))->getVT() <= EVT) {
    return N0;
  }
  // fold (sext_in_reg (sextload x)) -> (sextload x)
  if (N0.getOpcode() == ISD::SEXTLOAD && 
      cast<VTSDNode>(N0.getOperand(3))->getVT() <= EVT) {
    return N0;
  }
  // fold (sext_in_reg (setcc x)) -> setcc x iff (setcc x) == 0 or -1
  // FIXME: teach isSetCCEquivalent about 0, -1 and then use it here
  if (N0.getOpcode() == ISD::SETCC &&
      TLI.getSetCCResultContents() == 
        TargetLowering::ZeroOrNegativeOneSetCCResult)
    return N0;
  // FIXME: this code is currently just ported over from SelectionDAG.cpp
  // we probably actually want to handle this in two pieces.  Rather than
  // checking all the top bits for zero, just check the sign bit here and turn
  // it into a zero extend inreg (AND with constant).
  // then, let the code for AND figure out if the mask is superfluous rather
  // than doing so here.
  if (N0.getOpcode() == ISD::AND && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t Mask = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    unsigned NumBits = MVT::getSizeInBits(EVT);
    if ((Mask & (~0ULL << (NumBits-1))) == 0)
      return N0;
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitTRUNCATE(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);

  // noop truncate
  if (N0.getValueType() == N->getValueType(0))
    return N0;
  // fold (truncate c1) -> c1
  if (N0C)
    return DAG.getConstant(N0C->getValue(), VT);
  // fold (truncate (truncate x)) -> (truncate x)
  if (N0.getOpcode() == ISD::TRUNCATE)
    return DAG.getNode(ISD::TRUNCATE, VT, N0.getOperand(0));
  // fold (truncate (ext x)) -> (ext x) or (truncate x) or x
  if (N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::SIGN_EXTEND){
    if (N0.getValueType() < VT)
      // if the source is smaller than the dest, we still need an extend
      return DAG.getNode(N0.getOpcode(), VT, N0.getOperand(0));
    else if (N0.getValueType() > VT)
      // if the source is larger than the dest, than we just need the truncate
      return DAG.getNode(ISD::TRUNCATE, VT, N0.getOperand(0));
    else
      // if the source and dest are the same type, we can drop both the extend
      // and the truncate
      return N0.getOperand(0);
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitSINT_TO_FP(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  
  // fold (sint_to_fp c1) -> c1fp
  if (N0C)
    return DAG.getConstantFP(N0C->getSignExtended(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitUINT_TO_FP(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  
  // fold (uint_to_fp c1) -> c1fp
  if (N0C)
    return DAG.getConstantFP(N0C->getValue(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_TO_SINT(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_to_sint c1fp) -> c1
  if (N0CFP)
    return DAG.getConstant((int64_t)N0CFP->getValue(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_TO_UINT(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_to_uint c1fp) -> c1
  if (N0CFP)
    return DAG.getConstant((uint64_t)N0CFP->getValue(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_ROUND(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_round c1fp) -> c1fp
  if (N0CFP)
    return DAG.getConstantFP(N0CFP->getValue(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_ROUND_INREG(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  
  // noop fp_round_inreg
  if (EVT == VT)
    return N0;
  // fold (fp_round_inreg c1fp) -> c1fp
  if (N0CFP) {
    SDOperand Round = DAG.getConstantFP(N0CFP->getValue(), EVT);
    return DAG.getNode(ISD::FP_EXTEND, VT, Round);
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_EXTEND(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_extend c1fp) -> c1fp
  if (N0CFP)
    return DAG.getConstantFP(N0CFP->getValue(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFNEG(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  // fold (neg c1) -> -c1
  if (N0CFP)
    return DAG.getConstantFP(-N0CFP->getValue(), N->getValueType(0));
  // fold (neg (sub x, y)) -> (sub y, x)
  if (N->getOperand(0).getOpcode() == ISD::SUB)
    return DAG.getNode(ISD::SUB, N->getValueType(0), N->getOperand(1), 
                       N->getOperand(0));
  // fold (neg (neg x)) -> x
  if (N->getOperand(0).getOpcode() == ISD::FNEG)
    return N->getOperand(0).getOperand(0);
  return SDOperand();
}

SDOperand DAGCombiner::visitFABS(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  // fold (fabs c1) -> fabs(c1)
  if (N0CFP)
    return DAG.getConstantFP(fabs(N0CFP->getValue()), N->getValueType(0));
  // fold (fabs (fabs x)) -> (fabs x)
  if (N->getOperand(0).getOpcode() == ISD::FABS)
    return N->getOperand(0);
  // fold (fabs (fneg x)) -> (fabs x)
  if (N->getOperand(0).getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::FABS, N->getValueType(0), 
                       N->getOperand(0).getOperand(0));
  return SDOperand();
}

// SelectionDAG::Combine - This is the entry point for the file.
//
void SelectionDAG::Combine(bool RunningAfterLegalize) {
  /// run - This is the main entry point to this class.
  ///
  DAGCombiner(*this).Run(RunningAfterLegalize);
}
