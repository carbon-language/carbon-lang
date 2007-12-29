//===-- LegalizeTypes.h - Definition of the DAG Type Legalizer class ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DAGTypeLegalizer class.  This is a private interface
// shared between the code that implements the SelectionDAG::LegalizeTypes
// method.
//
//===----------------------------------------------------------------------===//

#ifndef SELECTIONDAG_LEGALIZETYPES_H
#define SELECTIONDAG_LEGALIZETYPES_H

#define DEBUG_TYPE "legalize-types"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"

namespace llvm {

//===----------------------------------------------------------------------===//
/// DAGTypeLegalizer - This takes an arbitrary SelectionDAG as input and
/// hacks on it until the target machine can handle it.  This involves
/// eliminating value sizes the machine cannot handle (promoting small sizes to
/// large sizes or splitting up large values into small values) as well as
/// eliminating operations the machine cannot handle.
///
/// This code also does a small amount of optimization and recognition of idioms
/// as part of its processing.  For example, if a target does not support a
/// 'setcc' instruction efficiently, but does support 'brcc' instruction, this
/// will attempt merge setcc and brc instructions into brcc's.
///
class VISIBILITY_HIDDEN DAGTypeLegalizer {
  TargetLowering &TLI;
  SelectionDAG &DAG;
  
  // NodeIDFlags - This pass uses the NodeID on the SDNodes to hold information
  // about the state of the node.  The enum has all the values.
  enum NodeIDFlags {
    /// ReadyToProcess - All operands have been processed, so this node is ready
    /// to be handled.
    ReadyToProcess = 0,
    
    /// NewNode - This is a new node that was created in the process of
    /// legalizing some other node.
    NewNode = -1,
    
    /// Processed - This is a node that has already been processed.
    Processed = -2
    
    // 1+ - This is a node which has this many unlegalized operands.
  };
  
  enum LegalizeAction {
    Legal,      // The target natively supports this type.
    Promote,    // This type should be executed in a larger type.
    Expand      // This type should be split into two types of half the size.
  };
  
  /// ValueTypeActions - This is a bitvector that contains two bits for each
  /// simple value type, where the two bits correspond to the LegalizeAction
  /// enum.  This can be queried with "getTypeAction(VT)".
  TargetLowering::ValueTypeActionImpl ValueTypeActions;
  
  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal or we need to expand it into multiple registers of
  /// smaller integer type, or we need to promote it to a larger type.
  LegalizeAction getTypeAction(MVT::ValueType VT) const {
    return (LegalizeAction)ValueTypeActions.getTypeAction(VT);
  }
  
  /// isTypeLegal - Return true if this type is legal on this target.
  ///
  bool isTypeLegal(MVT::ValueType VT) const {
    return getTypeAction(VT) == Legal;
  }
  
  SDOperand getIntPtrConstant(uint64_t Val) {
    return DAG.getConstant(Val, TLI.getPointerTy());
  }
  
  /// PromotedNodes - For nodes that are below legal width, this map indicates
  /// what promoted value to use.
  DenseMap<SDOperand, SDOperand> PromotedNodes;
  
  /// ExpandedNodes - For nodes that need to be expanded this map indicates
  /// which operands are the expanded version of the input.
  DenseMap<SDOperand, std::pair<SDOperand, SDOperand> > ExpandedNodes;

  /// ScalarizedNodes - For nodes that are <1 x ty>, this map indicates the
  /// scalar value of type 'ty' to use.
  DenseMap<SDOperand, SDOperand> ScalarizedNodes;

  /// SplitNodes - For nodes that need to be split this map indicates
  /// which operands are the expanded version of the input.
  DenseMap<SDOperand, std::pair<SDOperand, SDOperand> > SplitNodes;
  
  /// ReplacedNodes - For nodes that have been replaced with another,
  /// indicates the replacement node to use.
  DenseMap<SDOperand, SDOperand> ReplacedNodes;

  /// Worklist - This defines a worklist of nodes to process.  In order to be
  /// pushed onto this worklist, all operands of a node must have already been
  /// processed.
  SmallVector<SDNode*, 128> Worklist;
  
public:
  explicit DAGTypeLegalizer(SelectionDAG &dag)
    : TLI(dag.getTargetLoweringInfo()), DAG(dag),
    ValueTypeActions(TLI.getValueTypeActions()) {
    assert(MVT::LAST_VALUETYPE <= 32 &&
           "Too many value types for ValueTypeActions to hold!");
  }      
  
  void run();
  
private:
  void MarkNewNodes(SDNode *N);
  
  void ReplaceValueWith(SDOperand From, SDOperand To);
  void ReplaceNodeWith(SDNode *From, SDNode *To);

  void RemapNode(SDOperand &N);

  // Common routines.
  SDOperand CreateStackStoreLoad(SDOperand Op, MVT::ValueType DestVT);
  SDOperand HandleMemIntrinsic(SDNode *N);
  void SplitOp(SDOperand Op, SDOperand &Lo, SDOperand &Hi);

  //===--------------------------------------------------------------------===//
  // Promotion Support: LegalizeTypesPromote.cpp
  //===--------------------------------------------------------------------===//
  
  SDOperand GetPromotedOp(SDOperand Op) {
    SDOperand &PromotedOp = PromotedNodes[Op];
    RemapNode(PromotedOp);
    assert(PromotedOp.Val && "Operand wasn't promoted?");
    return PromotedOp;
  }
  void SetPromotedOp(SDOperand Op, SDOperand Result);
  
  /// GetPromotedZExtOp - Get a promoted operand and zero extend it to the final
  /// size.
  SDOperand GetPromotedZExtOp(SDOperand Op) {
    MVT::ValueType OldVT = Op.getValueType();
    Op = GetPromotedOp(Op);
    return DAG.getZeroExtendInReg(Op, OldVT);
  }    
    
  // Result Promotion.
  void PromoteResult(SDNode *N, unsigned ResNo);
  SDOperand PromoteResult_UNDEF(SDNode *N);
  SDOperand PromoteResult_Constant(SDNode *N);
  SDOperand PromoteResult_TRUNCATE(SDNode *N);
  SDOperand PromoteResult_INT_EXTEND(SDNode *N);
  SDOperand PromoteResult_FP_ROUND(SDNode *N);
  SDOperand PromoteResult_FP_TO_XINT(SDNode *N);
  SDOperand PromoteResult_SETCC(SDNode *N);
  SDOperand PromoteResult_LOAD(LoadSDNode *N);
  SDOperand PromoteResult_SimpleIntBinOp(SDNode *N);
  SDOperand PromoteResult_SDIV(SDNode *N);
  SDOperand PromoteResult_UDIV(SDNode *N);
  SDOperand PromoteResult_SHL(SDNode *N);
  SDOperand PromoteResult_SRA(SDNode *N);
  SDOperand PromoteResult_SRL(SDNode *N);
  SDOperand PromoteResult_SELECT   (SDNode *N);
  SDOperand PromoteResult_SELECT_CC(SDNode *N);
  
  // Operand Promotion.
  bool PromoteOperand(SDNode *N, unsigned OperandNo);
  SDOperand PromoteOperand_ANY_EXTEND(SDNode *N);
  SDOperand PromoteOperand_ZERO_EXTEND(SDNode *N);
  SDOperand PromoteOperand_SIGN_EXTEND(SDNode *N);
  SDOperand PromoteOperand_TRUNCATE(SDNode *N);
  SDOperand PromoteOperand_FP_EXTEND(SDNode *N);
  SDOperand PromoteOperand_FP_ROUND(SDNode *N);
  SDOperand PromoteOperand_INT_TO_FP(SDNode *N);
  SDOperand PromoteOperand_SELECT(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_BRCOND(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_BR_CC(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_SETCC(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_STORE(StoreSDNode *N, unsigned OpNo);
  
  void PromoteSetCCOperands(SDOperand &LHS,SDOperand &RHS, ISD::CondCode Code);

  //===--------------------------------------------------------------------===//
  // Expansion Support: LegalizeTypesExpand.cpp
  //===--------------------------------------------------------------------===//
  
  void GetExpandedOp(SDOperand Op, SDOperand &Lo, SDOperand &Hi);
  void SetExpandedOp(SDOperand Op, SDOperand Lo, SDOperand Hi);
    
  // Result Expansion.
  void ExpandResult(SDNode *N, unsigned ResNo);
  void ExpandResult_UNDEF      (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_Constant   (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_BUILD_PAIR (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_MERGE_VALUES(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ANY_EXTEND (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ZERO_EXTEND(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SIGN_EXTEND(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_BIT_CONVERT(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SIGN_EXTEND_INREG(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_LOAD       (LoadSDNode *N, SDOperand &Lo, SDOperand &Hi);

  void ExpandResult_Logical    (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_BSWAP      (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ADDSUB     (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ADDSUBC    (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ADDSUBE    (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SELECT     (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SELECT_CC  (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_MUL        (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_Shift      (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  
  void ExpandShiftByConstant(SDNode *N, unsigned Amt, 
                             SDOperand &Lo, SDOperand &Hi);
  bool ExpandShiftWithKnownAmountBit(SDNode *N, SDOperand &Lo, SDOperand &Hi);

  // Operand Expansion.
  bool ExpandOperand(SDNode *N, unsigned OperandNo);
  SDOperand ExpandOperand_TRUNCATE(SDNode *N);
  SDOperand ExpandOperand_BIT_CONVERT(SDNode *N);
  SDOperand ExpandOperand_UINT_TO_FP(SDOperand Source, MVT::ValueType DestTy);
  SDOperand ExpandOperand_SINT_TO_FP(SDOperand Source, MVT::ValueType DestTy);
  SDOperand ExpandOperand_EXTRACT_ELEMENT(SDNode *N);
  SDOperand ExpandOperand_SETCC(SDNode *N);
  SDOperand ExpandOperand_STORE(StoreSDNode *N, unsigned OpNo);
  
  void ExpandSetCCOperands(SDOperand &NewLHS, SDOperand &NewRHS,
                           ISD::CondCode &CCCode);
  
  //===--------------------------------------------------------------------===//
  // Scalarization Support: LegalizeTypesScalarize.cpp
  //===--------------------------------------------------------------------===//
  
  SDOperand GetScalarizedOp(SDOperand Op) {
    SDOperand &ScalarOp = ScalarizedNodes[Op];
    RemapNode(ScalarOp);
    assert(ScalarOp.Val && "Operand wasn't scalarized?");
    return ScalarOp;
  }
  void SetScalarizedOp(SDOperand Op, SDOperand Result);
    
  // Result Vector Scalarization: <1 x ty> -> ty.
  void ScalarizeResult(SDNode *N, unsigned OpNo);
  SDOperand ScalarizeRes_UNDEF(SDNode *N);
  SDOperand ScalarizeRes_LOAD(LoadSDNode *N);
  SDOperand ScalarizeRes_BinOp(SDNode *N);
  SDOperand ScalarizeRes_UnaryOp(SDNode *N);
  SDOperand ScalarizeRes_FPOWI(SDNode *N);
  SDOperand ScalarizeRes_VECTOR_SHUFFLE(SDNode *N);
  SDOperand ScalarizeRes_BIT_CONVERT(SDNode *N);
  SDOperand ScalarizeRes_SELECT(SDNode *N);
  
  // Operand Vector Scalarization: <1 x ty> -> ty.
  bool ScalarizeOperand(SDNode *N, unsigned OpNo);
  SDOperand ScalarizeOp_EXTRACT_VECTOR_ELT(SDNode *N, unsigned OpNo);

  //===--------------------------------------------------------------------===//
  // Vector Splitting Support: LegalizeTypesSplit.cpp
  //===--------------------------------------------------------------------===//
  
  void GetSplitOp(SDOperand Op, SDOperand &Lo, SDOperand &Hi);
  void SetSplitOp(SDOperand Op, SDOperand Lo, SDOperand Hi);
  
  // Result Vector Splitting: <128 x ty> -> 2 x <64 x ty>.
  void SplitResult(SDNode *N, unsigned OpNo);

  void SplitRes_UNDEF(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void SplitRes_LOAD(LoadSDNode *N, SDOperand &Lo, SDOperand &Hi);
  void SplitRes_BUILD_PAIR(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void SplitRes_INSERT_VECTOR_ELT(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void SplitRes_VECTOR_SHUFFLE(SDNode *N, SDOperand &Lo, SDOperand &Hi);

  void SplitRes_BUILD_VECTOR(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void SplitRes_CONCAT_VECTORS(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void SplitRes_BIT_CONVERT(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void SplitRes_UnOp(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void SplitRes_BinOp(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void SplitRes_FPOWI(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void SplitRes_SELECT(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  
  // Operand Vector Scalarization: <128 x ty> -> 2 x <64 x ty>.
  bool SplitOperand(SDNode *N, unsigned OpNo);
  
  SDOperand SplitOp_STORE(StoreSDNode *N, unsigned OpNo);
  SDOperand SplitOp_RET(SDNode *N, unsigned OpNo);
};

} // end namespace llvm.

#endif
