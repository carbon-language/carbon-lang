//===-- DAGCombiner.cpp - Implement a trivial DAG combiner ----------------===//
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
//
// FIXME: undef values
//
// FIXME: zero extend when top bits are 0 -> drop it ?
// 
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dagcombine"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetLowering.h"
#include <cmath>
using namespace llvm;

namespace {
  Statistic<> NodesCombined ("dagcombiner", "Number of dag nodes combined");

  class DAGCombiner {
    SelectionDAG &DAG;
    TargetLowering &TLI;

    // Worklist of all of the nodes that need to be simplified.
    std::vector<SDNode*> WorkList;

    /// AddUsersToWorkList - When an instruction is simplified, add all users of
    /// the instruction to the work lists because they might get more simplified
    /// now.
    ///
    void AddUsersToWorkList(SDNode *N) {
      for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end();
           UI != UE; ++UI) {
        SDNode *U = *UI;
        for (unsigned i = 0, e = U->getNumOperands(); i != e; ++i)
          WorkList.push_back(U->getOperand(i).Val);
      }
    }

    /// removeFromWorkList - remove all instances of N from the worklist.
    void removeFromWorkList(SDNode *N) {
      WorkList.erase(std::remove(WorkList.begin(), WorkList.end(), N),
                     WorkList.end());
    }
    
    /// visit - call the node-specific routine that knows how to fold each
    /// particular type of node.
    SDNode *visit(SDNode *N);

    // Visitation implementation - Implement dag node combining for different
    // node types.  The semantics are as follows:
    // Return Value:
    //    null        - No change was made
    //   otherwise    - Node N should be replaced by the returned node.
    //
    SDNode *visitTokenFactor(SDNode *N);
    SDNode *visitAdd(SDNode *N);
    SDNode *visitSub(SDNode *N);
    SDNode *visitMul(SDNode *N);
    SDNode *visitSdiv(SDNode *N);
    SDNode *visitUdiv(SDNode *N);
    SDNode *visitSrem(SDNode *N);
    SDNode *visitUrem(SDNode *N);
    SDNode *visitMulHiU(SDNode *N);
    SDNode *visitMulHiS(SDNode *N);
    SDNode *visitAnd(SDNode *N);
    SDNode *visitOr(SDNode *N);
    SDNode *visitXor(SDNode *N);
    SDNode *visitShl(SDNode *N);
    SDNode *visitSra(SDNode *N);
    SDNode *visitSrl(SDNode *N);
    SDNode *visitCtlz(SDNode *N);
    SDNode *visitCttz(SDNode *N);
    SDNode *visitCtpop(SDNode *N);
    // select
    // select_cc
    // setcc
    SDNode *visitSignExtend(SDNode *N);
    SDNode *visitZeroExtend(SDNode *N);
    SDNode *visitSignExtendInReg(SDNode *N);
    SDNode *visitTruncate(SDNode *N);
    SDNode *visitSintToFP(SDNode *N);
    SDNode *visitUintToFP(SDNode *N);
    SDNode *visitFPToSint(SDNode *N);
    SDNode *visitFPToUint(SDNode *N);
    SDNode *visitFPRound(SDNode *N);
    SDNode *visitFPRoundInReg(SDNode *N);
    SDNode *visitFPExtend(SDNode *N);
    SDNode *visitFneg(SDNode *N);
    SDNode *visitFabs(SDNode *N);
    SDNode *visitExtLoad(SDNode *N);
    SDNode *visitSextLoad(SDNode *N);
    SDNode *visitZextLoad(SDNode *N);
    SDNode *visitTruncStore(SDNode *N);
    // brcond
    // brcondtwoway
    // br_cc
    // brtwoway_cc
public:
    DAGCombiner(SelectionDAG &D)
      : DAG(D), TLI(D.getTargetLoweringInfo()) {
      // Add all the dag nodes to the worklist.
      WorkList.insert(WorkList.end(), D.allnodes_begin(), D.allnodes_end());
    }
    
    /// Run - runs the dag combiner on all nodes in the work list
    void Run(bool AfterLegalize); 
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
      return ((Mask & 1) == 0) &&
      TLI.getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult;
      
    case ISD::ZEXTLOAD:
      SrcBits = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(3))->getVT());
      return (Mask & ((1ULL << SrcBits)-1)) == 0; // Returning only the zext bits.
    case ISD::ZERO_EXTEND:
    case ISD::AssertZext:
      SrcBits = MVT::getSizeInBits(Op.getOperand(0).getValueType());
      return MaskedValueIsZero(Op.getOperand(0),Mask & ((1ULL << SrcBits)-1),TLI);
      
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

// isInvertibleForFree - Return true if there is no cost to emitting the logical
// inverse of this node.
static bool isInvertibleForFree(SDOperand N) {
  if (isa<ConstantSDNode>(N.Val)) return true;
  if (N.Val->getOpcode() == ISD::SETCC && N.Val->hasOneUse())
    return true;
  return false;
}

// isSetCCEquivalent - Return true if this node is a select_cc that selects
// between the values 1 and 0, making it equivalent to a setcc.
static bool isSetCCEquivalent(SDOperand N) {
  if (N.getOpcode() == ISD::SELECT_CC && 
      N.getOperand(2).getOpcode() == ISD::Constant &&
      N.getOperand(3).getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(N.getOperand(2))->getValue() == 1 &&
      cast<ConstantSDNode>(N.getOperand(3))->isNullValue()) 
    return true;
  return false;
}

void DAGCombiner::Run(bool AfterLegalize) {
  // while the worklist isn't empty, inspect the node on the end of it and
  // try and combine it.
  while (!WorkList.empty()) {
    SDNode *N = WorkList.back();
    WorkList.pop_back();
    
    // If N has no uses, it is dead.  Make sure to revisit all N's operands once
    // N is deleted from the DAG, since they too may now be dead.
    if (N->use_empty()) {
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
        WorkList.push_back(N->getOperand(i).Val);
      
      DAG.DeleteNode(N);
      removeFromWorkList(N);
      continue;
    }
    
    if (SDNode *Result = visit(N)) {
      ++NodesCombined;
      assert(Result != N && "Modifying DAG nodes in place is illegal!");

      std::cerr << "DC: Old = "; N->dump();
      std::cerr << "    New = "; Result->dump();
      std::cerr << '\n';
      DAG.ReplaceAllUsesWith(N, Result);
        
      // Push the new node and any users onto the worklist
      WorkList.push_back(Result);
      AddUsersToWorkList(Result);
        
      // Nodes can end up on the worklist more than once.  Make sure we do
      // not process a node that has been replaced.
      removeFromWorkList(N);
    }
  }
}

SDNode *DAGCombiner::visit(SDNode *N) {
  switch(N->getOpcode()) {
  default: break;
  case ISD::TokenFactor:        return visitTokenFactor(N);
  case ISD::ADD:                return visitAdd(N);
  case ISD::SUB:                return visitSub(N);
  case ISD::MUL:                return visitMul(N);
  case ISD::SDIV:               return visitSdiv(N);
  case ISD::UDIV:               return visitUdiv(N);
  case ISD::SREM:               return visitSrem(N);
  case ISD::UREM:               return visitUrem(N);
  case ISD::MULHU:              return visitMulHiU(N);
  case ISD::MULHS:              return visitMulHiS(N);
  case ISD::AND:                return visitAnd(N);
  case ISD::OR:                 return visitOr(N);
  case ISD::XOR:                return visitXor(N);
  case ISD::SHL:                return visitShl(N);
  case ISD::SRA:                return visitSra(N);
  case ISD::SRL:                return visitSrl(N);
  case ISD::CTLZ:               return visitCtlz(N);
  case ISD::CTTZ:               return visitCttz(N);
  case ISD::CTPOP:              return visitCtpop(N);
  case ISD::SIGN_EXTEND:        return visitSignExtend(N);
  case ISD::ZERO_EXTEND:        return visitZeroExtend(N);
  case ISD::SIGN_EXTEND_INREG:  return visitSignExtendInReg(N);
  case ISD::TRUNCATE:           return visitTruncate(N);
  case ISD::SINT_TO_FP:         return visitSintToFP(N);
  case ISD::UINT_TO_FP:         return visitUintToFP(N);
  case ISD::FP_TO_SINT:         return visitFPToSint(N);
  case ISD::FP_TO_UINT:         return visitFPToUint(N);
  case ISD::FP_ROUND:           return visitFPRound(N);
  case ISD::FP_ROUND_INREG:     return visitFPRoundInReg(N);
  case ISD::FP_EXTEND:          return visitFPExtend(N);
  case ISD::FNEG:               return visitFneg(N);
  case ISD::FABS:               return visitFabs(N);
  case ISD::EXTLOAD:            return visitExtLoad(N);
  case ISD::SEXTLOAD:           return visitSextLoad(N);
  case ISD::ZEXTLOAD:           return visitZextLoad(N);
  case ISD::TRUNCSTORE:         return visitTruncStore(N);
  }
  return 0;
}

SDNode *DAGCombiner::visitTokenFactor(SDNode *N) {
  // If the token factor only has one operand, fold TF(x) -> x
  if (N->getNumOperands() == 1)
    return N->getOperand(0).Val;
  
  // If the token factor has two operands and one is the entry token, replace
  // the token factor with the other operand.
  if (N->getNumOperands() == 2) {
    if (N->getOperand(0).getOpcode() == ISD::EntryToken)
      return N->getOperand(1).Val;
    if (N->getOperand(1).getOpcode() == ISD::EntryToken)
      return N->getOperand(0).Val;
  }
  return 0;
}

SDNode *DAGCombiner::visitAdd(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N0.Val);
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N1.Val);
  
  // fold (add c1, c2) -> c1+c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getValue() + N2C->getValue(),
                           N->getValueType(0)).Val;
  // fold (add x, 0) -> x
  if (N2C && N2C->isNullValue())
    return N0.Val;
  // fold floating point (add c1, c2) -> c1+c2
  if (N1CFP && N2CFP)
    return DAG.getConstantFP(N1CFP->getValue() + N2CFP->getValue(),
                             N->getValueType(0)).Val;
  // fold (A + (-B)) -> A-B
  if (N1.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::SUB, N->getValueType(0), N0, N1.getOperand(0)).Val;
  // fold ((-A) + B) -> B-A
  if (N0.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::SUB, N->getValueType(0), N1, N0.getOperand(0)).Val;
  // fold ((0-A) + B) -> B-A
  if (N0.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N0.getOperand(0)) &&
      cast<ConstantSDNode>(N0.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, N->getValueType(0), N1, N0.getOperand(1)).Val;
  // fold (A + (0-B)) -> A-B
  if (N1.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N1.getOperand(0)) &&
      cast<ConstantSDNode>(N1.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, N->getValueType(0), N0, N1.getOperand(1)).Val;
  // fold (A+(B-A)) -> B for non-fp types
  if (N1.getOpcode() == ISD::SUB && N0 == N1.getOperand(1) &&
      !MVT::isFloatingPoint(N1.getValueType()))
    return N1.getOperand(0).Val;
  return 0;
}

SDNode *DAGCombiner::visitSub(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N0.Val);
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N1.Val);
  
  // fold (sub c1, c2) -> c1-c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getValue() - N2C->getValue(),
                           N->getValueType(0)).Val;
  // fold (sub x, 0) -> x
  if (N2C && N2C->isNullValue())
    return N0.Val;
  // fold floating point (sub c1, c2) -> c1-c2
  if (N1CFP && N2CFP)
    return DAG.getConstantFP(N1CFP->getValue() - N2CFP->getValue(),
                             N->getValueType(0)).Val;
  // fold (A+B)-A -> B
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(0) == N1 &&
      !MVT::isFloatingPoint(N1.getValueType()))
    return N0.getOperand(1).Val;
  // fold (A+B)-B -> A
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(1) == N1 &&
      !MVT::isFloatingPoint(N1.getValueType()))
    return N0.getOperand(0).Val;
  // fold (A-(-B)) -> A+B
  if (N1.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::ADD, N0.getValueType(), N0, N1.getOperand(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitMul(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N0.Val);
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N1.Val);
  
  // fold (mul c1, c2) -> c1*c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getValue() * N2C->getValue(),
                           N->getValueType(0)).Val;
  // fold (mul x, 0) -> 0
  if (N2C && N2C->isNullValue())
    return N1.Val;
  // fold (mul x, -1) -> 0-x
  if (N2C && N2C->isAllOnesValue())
    return DAG.getNode(ISD::SUB, N->getValueType(0), 
                       DAG.getConstant(0, N->getValueType(0)), N0).Val;
  // fold (mul x, (1 << c)) -> x << c
  if (N2C && isPowerOf2_64(N2C->getValue()))
    return DAG.getNode(ISD::SHL, N->getValueType(0), N0,
                       DAG.getConstant(Log2_64(N2C->getValue()),
                                       TLI.getShiftAmountTy())).Val;
  // fold floating point (mul c1, c2) -> c1*c2
  if (N1CFP && N2CFP)
    return DAG.getConstantFP(N1CFP->getValue() * N2CFP->getValue(),
                             N->getValueType(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitSdiv(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N0.Val);
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N1.Val);

  // fold (sdiv c1, c2) -> c1/c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getSignExtended() / N2C->getSignExtended(),
                           N->getValueType(0)).Val;
  // fold floating point (sdiv c1, c2) -> c1/c2
  if (N1CFP && N2CFP)
    return DAG.getConstantFP(N1CFP->getValue() / N2CFP->getValue(),
                             N->getValueType(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitUdiv(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  
  // fold (udiv c1, c2) -> c1/c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getValue() / N2C->getValue(),
                           N->getValueType(0)).Val;
  // fold (udiv x, (1 << c)) -> x >>u c
  if (N2C && isPowerOf2_64(N2C->getValue()))
    return DAG.getNode(ISD::SRL, N->getValueType(0), N0,
                       DAG.getConstant(Log2_64(N2C->getValue()),
                                       TLI.getShiftAmountTy())).Val;
  return 0;
}

SDNode *DAGCombiner::visitSrem(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N0.Val);
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N1.Val);
  
  // fold (srem c1, c2) -> c1%c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getSignExtended() % N2C->getSignExtended(),
                           N->getValueType(0)).Val;
  // fold floating point (srem c1, c2) -> fmod(c1, c2)
  if (N1CFP && N2CFP)
    return DAG.getConstantFP(fmod(N1CFP->getValue(),N2CFP->getValue()),
                             N->getValueType(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitUrem(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  
  // fold (urem c1, c2) -> c1%c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getValue() % N2C->getValue(),
                           N->getValueType(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitMulHiS(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  
  // fold (mulhs x, 0) -> 0
  if (N2C && N2C->isNullValue())
    return N1.Val;
  
  // fold (mulhs x, 1) -> (sra x, size(x)-1)
  if (N2C && N2C->getValue() == 1)
    return DAG.getNode(ISD::SRA, N0.getValueType(), N0, 
                       DAG.getConstant(MVT::getSizeInBits(N0.getValueType())-1,
                                       TLI.getShiftAmountTy())).Val;
  return 0;
}

SDNode *DAGCombiner::visitMulHiU(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  
  // fold (mulhu x, 0) -> 0
  if (N2C && N2C->isNullValue())
    return N1.Val;
  
  // fold (mulhu x, 1) -> 0
  if (N2C && N2C->getValue() == 1)
    return DAG.getConstant(0, N0.getValueType()).Val;
  return 0;
}

SDNode *DAGCombiner::visitAnd(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  MVT::ValueType VT = N1.getValueType();
  
  // fold (and c1, c2) -> c1&c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getValue() & N2C->getValue(), VT).Val;
  // fold (and x, 0) -> 0
  if (N2C && N2C->isNullValue())
    return N1.Val;
  // fold (and x, -1) -> x
  if (N2C && N2C->isAllOnesValue())
    return N0.Val;
  // fold (and x, 0) -> 0
  if (MaskedValueIsZero(N0, N2C->getValue(), TLI))
    return DAG.getConstant(0, VT).Val;
  // fold (and x, mask containing x) -> x
  uint64_t NotC2 = ~N2C->getValue();
  if (MVT::i64 != VT) NotC2 &= (1ULL << MVT::getSizeInBits(VT))-1;
  if (MaskedValueIsZero(N0, NotC2, TLI))
    return N0.Val;
  // fold (and (sign_extend_inreg x, i16 to i32), 1) -> (and x, 1)
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    unsigned ExtendBits =
    MVT::getSizeInBits(cast<VTSDNode>(N0.getOperand(1))->getVT());
    if ((N2C->getValue() & (~0ULL << ExtendBits)) == 0)
      return DAG.getNode(ISD::AND, VT, N0.getOperand(0), N1).Val;
  }
  // fold (and (or x, 0xFFFF), 0xFF) -> 0xFF
  if (N0.getOpcode() == ISD::OR)
    if (ConstantSDNode *ORI = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
      if ((ORI->getValue() & N2C->getValue()) == N2C->getValue())
        return N1.Val;
  // fold (and (assert_zext x, i16), 0xFFFF) -> (assert_zext x, i16)
  if (N0.getOpcode() == ISD::AssertZext) {
    unsigned ExtendBits =
    MVT::getSizeInBits(cast<VTSDNode>(N0.getOperand(1))->getVT());
    if (N2C->getValue() == (1ULL << ExtendBits)-1)
      return N0.Val;
  }
  return 0;
}

SDNode *DAGCombiner::visitOr(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  
  // fold (or c1, c2) -> c1|c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getValue() | N2C->getValue(),
                           N->getValueType(0)).Val;
  // fold (or x, 0) -> x
  if (N2C && N2C->isNullValue())
    return N0.Val;
  // fold (or x, -1) -> -1
  if (N2C && N2C->isAllOnesValue())
    return N1.Val;
  return 0;
}

SDNode *DAGCombiner::visitXor(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  MVT::ValueType VT = N0.getValueType();
  
  // fold (xor c1, c2) -> c1^c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getValue() ^ N2C->getValue(), VT).Val;
  // fold (xor x, 0) -> x
  if (N2C && N2C->isNullValue())
    return N0.Val;
  // fold !(x cc y) -> (x !cc y)
  if (N2C && N2C->isAllOnesValue() && N0.getOpcode() == ISD::SETCC) {
    bool isInt = MVT::isInteger(N0.getOperand(0).getValueType());
    ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();
    return DAG.getSetCC(VT, N0.getOperand(0), N0.getOperand(1), 
                        ISD::getSetCCInverse(CC, isInt)).Val;
  }
  // fold !(x cc y) -> (x !cc y)
  if (N2C && N2C->isAllOnesValue() && isSetCCEquivalent(N0)) {
    bool isInt = MVT::isInteger(N0.getOperand(0).getValueType());
    ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(4))->get();
    return DAG.getSelectCC(N0.getOperand(0), N0.getOperand(1), 
                           N0.getOperand(2), N0.getOperand(3),
                           ISD::getSetCCInverse(CC, isInt)).Val;
  }
  // fold !(x or y) -> (!x and !y) iff x or y are freely invertible
  if (N2C && N2C->isAllOnesValue() && N0.getOpcode() == ISD::OR) {
    SDOperand LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isInvertibleForFree(RHS) || isInvertibleForFree(LHS)) {
      LHS = DAG.getNode(ISD::XOR, VT, LHS, N1);  // RHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, VT, RHS, N1);  // RHS = ~RHS
      return DAG.getNode(ISD::AND, VT, LHS, RHS).Val;
    }
  }
  // fold !(x and y) -> (!x or !y) iff x or y are freely invertible
  if (N2C && N2C->isAllOnesValue() && N0.getOpcode() == ISD::AND) {
    SDOperand LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isInvertibleForFree(RHS) || isInvertibleForFree(LHS)) {
      LHS = DAG.getNode(ISD::XOR, VT, LHS, N1);  // RHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, VT, RHS, N1);  // RHS = ~RHS
      return DAG.getNode(ISD::OR, VT, LHS, RHS).Val;
    }
  }
  return 0;
}

SDNode *DAGCombiner::visitShl(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  MVT::ValueType VT = N0.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (shl c1, c2) -> c1<<c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getValue() << N2C->getValue(), VT).Val;
  // fold (shl 0, x) -> 0
  if (N1C && N1C->isNullValue())
    return N0.Val;
  // fold (shl x, c >= size(x)) -> undef
  if (N2C && N2C->getValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT).Val;
  // fold (shl x, 0) -> x
  if (N2C && N2C->isNullValue())
    return N0.Val;
  // if (shl x, c) is known to be zero, return 0
  if (N2C && MaskedValueIsZero(N0,(~0ULL >> (64-OpSizeInBits))>>N2C->getValue(),
                               TLI))
    return DAG.getConstant(0, VT).Val;
  // fold (shl (shl x, c1), c2) -> 0 or (shl x, c1+c2)
  if (N2C && N0.getOpcode() == ISD::SHL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N2C->getValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT).Val;
    return DAG.getNode(ISD::SHL, VT, N0.getOperand(0), 
                       DAG.getConstant(c1 + c2, N1.getValueType())).Val;
  }
  // fold (shl (srl x, c1), c2) -> (shl (and x, -1 << c1), c2-c1) or
  //                               (srl (and x, -1 << c1), c1-c2)
  if (N2C && N0.getOpcode() == ISD::SRL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N2C->getValue();
    SDOperand Mask = DAG.getNode(ISD::AND, VT, N0.getOperand(0),
                                 DAG.getConstant(~0ULL << c1, VT));
    if (c2 > c1)
      return DAG.getNode(ISD::SHL, VT, Mask, 
                         DAG.getConstant(c2-c1, N1.getValueType())).Val;
    else
      return DAG.getNode(ISD::SRL, VT, Mask,
                         DAG.getConstant(c1-c2, N1.getValueType())).Val;
  }
  // fold (shl (sra x, c1), c1) -> (and x, -1 << c1)
  if (N2C && N0.getOpcode() == ISD::SRA &&
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N2C->getValue();
    if (c1 == c2)
      return DAG.getNode(ISD::AND, VT, N0.getOperand(0),
                         DAG.getConstant(~0ULL << c1, VT)).Val;
  }
  return 0;
}

SDNode *DAGCombiner::visitSra(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  MVT::ValueType VT = N0.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (sra c1, c2) -> c1>>c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getSignExtended() >> N2C->getValue(), VT).Val;
  // fold (sra 0, x) -> 0
  if (N1C && N1C->isNullValue())
    return N0.Val;
  // fold (sra -1, x) -> -1
  if (N1C && N1C->isAllOnesValue())
    return N0.Val;
  // fold (sra x, c >= size(x)) -> undef
  if (N2C && N2C->getValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT).Val;
  // fold (sra x, 0) -> x
  if (N2C && N2C->isNullValue())
    return N0.Val;
  // If the sign bit is known to be zero, switch this to a SRL.
  if (N2C && MaskedValueIsZero(N0, (1ULL << (OpSizeInBits-1)), TLI))
    return DAG.getNode(ISD::SRL, VT, N0, N1).Val;
  return 0;
}

SDNode *DAGCombiner::visitSrl(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N1.Val);
  MVT::ValueType VT = N0.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (srl c1, c2) -> c1 >>u c2
  if (N1C && N2C)
    return DAG.getConstant(N1C->getValue() >> N2C->getValue(), VT).Val;
  // fold (srl 0, x) -> 0
  if (N1C && N1C->isNullValue())
    return N0.Val;
  // fold (srl x, c >= size(x)) -> undef
  if (N2C && N2C->getValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT).Val;
  // fold (srl x, 0) -> x
  if (N2C && N2C->isNullValue())
    return N0.Val;
  // if (srl x, c) is known to be zero, return 0
  if (N2C && MaskedValueIsZero(N0,(~0ULL >> (64-OpSizeInBits))<<N2C->getValue(),
                               TLI))
    return DAG.getConstant(0, VT).Val;
  // fold (srl (srl x, c1), c2) -> 0 or (srl x, c1+c2)
  if (N2C && N0.getOpcode() == ISD::SRL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N2C->getValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT).Val;
    return DAG.getNode(ISD::SRL, VT, N0.getOperand(0), 
                       DAG.getConstant(c1 + c2, N1.getValueType())).Val;
  }
  return 0;
}

SDNode *DAGCombiner::visitCtlz(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);

  // fold (ctlz c1) -> c2
  if (N1C)
    return DAG.getConstant(CountLeadingZeros_64(N1C->getValue()),
                           N0.getValueType()).Val;
  return 0;
}

SDNode *DAGCombiner::visitCttz(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  
  // fold (cttz c1) -> c2
  if (N1C)
    return DAG.getConstant(CountTrailingZeros_64(N1C->getValue()),
                           N0.getValueType()).Val;
  return 0;
}

SDNode *DAGCombiner::visitCtpop(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  
  // fold (ctpop c1) -> c2
  if (N1C)
    return DAG.getConstant(CountPopulation_64(N1C->getValue()),
                           N0.getValueType()).Val;
  return 0;
}

SDNode *DAGCombiner::visitSignExtend(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  MVT::ValueType VT = N->getValueType(0);

  // noop sext
  if (N0.getValueType() == N->getValueType(0))
    return N0.Val;
  // fold (sext c1) -> c1
  if (N1C)
    return DAG.getConstant(N1C->getSignExtended(), VT).Val;
  // fold (sext (sext x)) -> (sext x)
  if (N0.getOpcode() == ISD::SIGN_EXTEND)
    return DAG.getNode(ISD::SIGN_EXTEND, VT, N0.getOperand(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitZeroExtend(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  MVT::ValueType VT = N->getValueType(0);

  // noop zext
  if (N0.getValueType() == N->getValueType(0))
    return N0.Val;
  // fold (zext c1) -> c1
  if (N1C)
    return DAG.getConstant(N1C->getValue(), VT).Val;
  // fold (zext (zext x)) -> (zext x)
  if (N0.getOpcode() == ISD::ZERO_EXTEND)
    return DAG.getNode(ISD::ZERO_EXTEND, VT, N0.getOperand(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitSignExtendInReg(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  
  // noop sext_in_reg
  if (EVT == VT)
    return N0.Val;
  // fold (sext_in_reg c1) -> c1
  if (N1C) {
    SDOperand Truncate = DAG.getConstant(N1C->getValue(), EVT);
    return DAG.getNode(ISD::SIGN_EXTEND, VT, Truncate).Val;
  }
  // fold (sext_in_reg (sext_in_reg x)) -> (sext_in_reg x)
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG && 
      cast<VTSDNode>(N0.getOperand(1))->getVT() <= EVT) {
    return N0.Val;
  }
  // fold (sext_in_reg (assert_sext x)) -> (assert_sext x)
  if (N0.getOpcode() == ISD::AssertSext && 
      cast<VTSDNode>(N0.getOperand(1))->getVT() <= EVT) {
    return N0.Val;
  }
  // fold (sext_in_reg (sextload x)) -> (sextload x)
  if (N0.getOpcode() == ISD::SEXTLOAD && 
      cast<VTSDNode>(N0.getOperand(3))->getVT() <= EVT) {
    return N0.Val;
  }
  // fold (sext_in_reg (setcc x)) -> setcc x iff (setcc x) == 0 or 1
  if (N0.getOpcode() == ISD::SETCC &&
      TLI.getSetCCResultContents() == 
        TargetLowering::ZeroOrNegativeOneSetCCResult)
    return N0.Val;
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
      return N0.Val;
  }
  return 0;
}

SDNode *DAGCombiner::visitTruncate(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  MVT::ValueType VT = N->getValueType(0);

  // noop truncate
  if (N0.getValueType() == N->getValueType(0))
    return N0.Val;
  // fold (truncate c1) -> c1
  if (N1C)
    return DAG.getConstant(N1C->getValue(), VT).Val;
  // fold (truncate (truncate x)) -> (truncate x)
  if (N0.getOpcode() == ISD::TRUNCATE)
    return DAG.getNode(ISD::TRUNCATE, VT, N0.getOperand(0)).Val;
  // fold (truncate (ext x)) -> (ext x) or (truncate x) or x
  if (N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::SIGN_EXTEND){
    if (N0.getValueType() < VT)
      // if the source is smaller than the dest, we still need an extend
      return DAG.getNode(N0.getOpcode(), VT, N0.getOperand(0)).Val;
    else if (N0.getValueType() > VT)
      // if the source is larger than the dest, than we just need the truncate
      return DAG.getNode(ISD::TRUNCATE, VT, N0.getOperand(0)).Val;
    else
      // if the source and dest are the same type, we can drop both the extend
      // and the truncate
      return N0.getOperand(0).Val;
  }
  return 0;
}

SDNode *DAGCombiner::visitSintToFP(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (sint_to_fp c1) -> c1fp
  if (N1C)
    return DAG.getConstantFP(N1C->getSignExtended(), VT).Val;
  return 0;
}

SDNode *DAGCombiner::visitUintToFP(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N0.Val);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (uint_to_fp c1) -> c1fp
  if (N1C)
    return DAG.getConstantFP(N1C->getValue(), VT).Val;
  return 0;
}

SDNode *DAGCombiner::visitFPToSint(SDNode *N) {
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_to_sint c1fp) -> c1
  if (N1CFP)
    return DAG.getConstant((int64_t)N1CFP->getValue(), N->getValueType(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitFPToUint(SDNode *N) {
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_to_uint c1fp) -> c1
  if (N1CFP)
    return DAG.getConstant((uint64_t)N1CFP->getValue(), N->getValueType(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitFPRound(SDNode *N) {
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_round c1fp) -> c1fp
  if (N1CFP)
    return DAG.getConstantFP(N1CFP->getValue(), N->getValueType(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitFPRoundInReg(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N0);
  
  // noop fp_round_inreg
  if (EVT == VT)
    return N0.Val;
  // fold (fp_round_inreg c1fp) -> c1fp
  if (N1CFP) {
    SDOperand Round = DAG.getConstantFP(N1CFP->getValue(), EVT);
    return DAG.getNode(ISD::FP_EXTEND, VT, Round).Val;
  }
  return 0;
}

SDNode *DAGCombiner::visitFPExtend(SDNode *N) {
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_extend c1fp) -> c1fp
  if (N1CFP)
    return DAG.getConstantFP(N1CFP->getValue(), N->getValueType(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitFneg(SDNode *N) {
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  // fold (neg c1) -> -c1
  if (N1CFP)
    return DAG.getConstantFP(-N1CFP->getValue(), N->getValueType(0)).Val;
  // fold (neg (sub x, y)) -> (sub y, x)
  if (N->getOperand(0).getOpcode() == ISD::SUB)
    return DAG.getNode(ISD::SUB, N->getValueType(0), N->getOperand(1), 
                       N->getOperand(0)).Val;
  // fold (neg (neg x)) -> x
  if (N->getOperand(0).getOpcode() == ISD::FNEG)
    return N->getOperand(0).getOperand(0).Val;
  return 0;
}

SDNode *DAGCombiner::visitFabs(SDNode *N) {
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  // fold (fabs c1) -> fabs(c1)
  if (N1CFP)
    return DAG.getConstantFP(fabs(N1CFP->getValue()), N->getValueType(0)).Val;
  // fold (fabs (fabs x)) -> (fabs x)
  if (N->getOperand(0).getOpcode() == ISD::FABS)
    return N->getOperand(0).Val;
  // fold (fabs (fneg x)) -> (fabs x)
  if (N->getOperand(0).getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::FABS, N->getValueType(0), 
                       N->getOperand(0).getOperand(0)).Val;
  return 0;
}

SDNode *DAGCombiner::visitExtLoad(SDNode *N) {
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N->getOperand(3))->getVT();
  
  // fold (extload vt, x) -> (load x)
  if (EVT == VT)
    return DAG.getLoad(VT, N->getOperand(0), N->getOperand(1), 
                       N->getOperand(2)).Val;
  return 0;
}

SDNode *DAGCombiner::visitSextLoad(SDNode *N) {
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N->getOperand(3))->getVT();
  
  // fold (sextload vt, x) -> (load x)
  if (EVT == VT)
    return DAG.getLoad(VT, N->getOperand(0), N->getOperand(1), 
                       N->getOperand(2)).Val;
  return 0;
}

SDNode *DAGCombiner::visitZextLoad(SDNode *N) {
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N->getOperand(3))->getVT();
  
  // fold (zextload vt, x) -> (load x)
  if (EVT == VT)
    return DAG.getLoad(VT, N->getOperand(0), N->getOperand(1), 
                       N->getOperand(2)).Val;
  return 0;
}

SDNode *DAGCombiner::visitTruncStore(SDNode *N) {
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N->getOperand(4))->getVT();
  
  // fold (truncstore x, vt) -> (store x)
  if (N->getOperand(0).getValueType() == EVT)
    return DAG.getNode(ISD::STORE, VT, N->getOperand(0), N->getOperand(1), 
                       N->getOperand(2), N->getOperand(3)).Val;
  return 0;
}

// SelectionDAG::Combine - This is the entry point for the file.
//
void SelectionDAG::Combine(bool AfterLegalize) {
  /// run - This is the main entry point to this class.
  ///
  DAGCombiner(*this).Run(AfterLegalize);
}
