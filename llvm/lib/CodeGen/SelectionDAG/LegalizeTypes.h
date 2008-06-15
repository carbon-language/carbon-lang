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
public:
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
private:
  enum LegalizeAction {
    Legal,      // The target natively supports this type.
    Promote,    // This type should be executed in a larger type.
    Expand,     // This type should be split into two types of half the size.
    FloatToInt, // Convert a floating point type to an integer of the same size.
    Scalarize,  // Replace this one-element vector type with its element type.
    Split       // This vector type should be split into smaller vectors.
  };

  /// ValueTypeActions - This is a bitvector that contains two bits for each
  /// simple value type, where the two bits correspond to the LegalizeAction
  /// enum from TargetLowering.  This can be queried with "getTypeAction(VT)".
  TargetLowering::ValueTypeActionImpl ValueTypeActions;
  
  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal, or we need to promote it to a larger integer type, or
  /// we need to expand it into multiple registers of a smaller integer type, or
  /// we need to scalarize a one-element vector type into the element type, or
  /// we need to split a vector type into smaller vector types.
  LegalizeAction getTypeAction(MVT VT) const {
    switch (ValueTypeActions.getTypeAction(VT)) {
    default:
      assert(false && "Unknown legalize action!");
    case TargetLowering::Legal:
      return Legal;
    case TargetLowering::Promote:
      return Promote;
    case TargetLowering::Expand:
      // Expand can mean
      // 1) split scalar in half, 2) convert a float to an integer,
      // 3) scalarize a single-element vector, 4) split a vector in two.
      if (!VT.isVector()) {
        if (VT.getSizeInBits() == TLI.getTypeToTransformTo(VT).getSizeInBits())
          return FloatToInt;
        else
          return Expand;
      } else if (VT.getVectorNumElements() == 1) {
        return Scalarize;
      } else {
        return Split;
      }
    }
  }

  /// isTypeLegal - Return true if this type is legal on this target.
  bool isTypeLegal(MVT VT) const {
    return ValueTypeActions.getTypeAction(VT) == TargetLowering::Legal;
  }

  /// PromotedNodes - For nodes that are below legal width, this map indicates
  /// what promoted value to use.
  DenseMap<SDOperand, SDOperand> PromotedNodes;
  
  /// ExpandedNodes - For nodes that need to be expanded this map indicates
  /// which operands are the expanded version of the input.
  DenseMap<SDOperand, std::pair<SDOperand, SDOperand> > ExpandedNodes;

  /// FloatToIntedNodes - For floating point nodes converted to integers of
  /// the same size, this map indicates the converted value to use.
  DenseMap<SDOperand, SDOperand> FloatToIntedNodes;

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
  
  /// ReanalyzeNode - Recompute the NodeID and correct processed operands
  /// for the specified node, adding it to the worklist if ready.
  void ReanalyzeNode(SDNode *N) {
    N->setNodeId(NewNode);
    AnalyzeNewNode(N);
  }

  void NoteReplacement(SDOperand From, SDOperand To) {
    ExpungeNode(From);
    ExpungeNode(To);
    ReplacedNodes[From] = To;
  }

private:
  void AnalyzeNewNode(SDNode *&N);

  void ReplaceValueWith(SDOperand From, SDOperand To);
  void ReplaceNodeWith(SDNode *From, SDNode *To);

  void RemapNode(SDOperand &N);
  void ExpungeNode(SDOperand N);

  // Common routines.
  SDOperand CreateStackStoreLoad(SDOperand Op, MVT DestVT);
  SDOperand MakeLibCall(RTLIB::Libcall LC, MVT RetVT,
                        const SDOperand *Ops, unsigned NumOps, bool isSigned);

  SDOperand BitConvertToInteger(SDOperand Op);
  SDOperand JoinIntegers(SDOperand Lo, SDOperand Hi);
  void SplitInteger(SDOperand Op, SDOperand &Lo, SDOperand &Hi);
  void SplitInteger(SDOperand Op, MVT LoVT, MVT HiVT,
                    SDOperand &Lo, SDOperand &Hi);

  SDOperand GetVectorElementPointer(SDOperand VecPtr, MVT EltVT,
                                    SDOperand Index);

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
    MVT OldVT = Op.getValueType();
    Op = GetPromotedOp(Op);
    return DAG.getZeroExtendInReg(Op, OldVT);
  }    
    
  // Result Promotion.
  void PromoteResult(SDNode *N, unsigned ResNo);
  SDOperand PromoteResult_BIT_CONVERT(SDNode *N);
  SDOperand PromoteResult_BUILD_PAIR(SDNode *N);
  SDOperand PromoteResult_Constant(SDNode *N);
  SDOperand PromoteResult_CTLZ(SDNode *N);
  SDOperand PromoteResult_CTPOP(SDNode *N);
  SDOperand PromoteResult_CTTZ(SDNode *N);
  SDOperand PromoteResult_EXTRACT_VECTOR_ELT(SDNode *N);
  SDOperand PromoteResult_FP_ROUND(SDNode *N);
  SDOperand PromoteResult_FP_TO_XINT(SDNode *N);
  SDOperand PromoteResult_INT_EXTEND(SDNode *N);
  SDOperand PromoteResult_LOAD(LoadSDNode *N);
  SDOperand PromoteResult_SDIV(SDNode *N);
  SDOperand PromoteResult_SELECT   (SDNode *N);
  SDOperand PromoteResult_SELECT_CC(SDNode *N);
  SDOperand PromoteResult_SETCC(SDNode *N);
  SDOperand PromoteResult_SHL(SDNode *N);
  SDOperand PromoteResult_SimpleIntBinOp(SDNode *N);
  SDOperand PromoteResult_SRA(SDNode *N);
  SDOperand PromoteResult_SRL(SDNode *N);
  SDOperand PromoteResult_TRUNCATE(SDNode *N);
  SDOperand PromoteResult_UDIV(SDNode *N);
  SDOperand PromoteResult_UNDEF(SDNode *N);

  // Operand Promotion.
  bool PromoteOperand(SDNode *N, unsigned OperandNo);
  SDOperand PromoteOperand_ANY_EXTEND(SDNode *N);
  SDOperand PromoteOperand_BUILD_PAIR(SDNode *N);
  SDOperand PromoteOperand_BR_CC(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_BRCOND(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_BUILD_VECTOR(SDNode *N);
  SDOperand PromoteOperand_FP_EXTEND(SDNode *N);
  SDOperand PromoteOperand_FP_ROUND(SDNode *N);
  SDOperand PromoteOperand_INT_TO_FP(SDNode *N);
  SDOperand PromoteOperand_INSERT_VECTOR_ELT(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_MEMBARRIER(SDNode *N);
  SDOperand PromoteOperand_RET(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_SELECT(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_SETCC(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_SIGN_EXTEND(SDNode *N);
  SDOperand PromoteOperand_STORE(StoreSDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_TRUNCATE(SDNode *N);
  SDOperand PromoteOperand_ZERO_EXTEND(SDNode *N);

  void PromoteSetCCOperands(SDOperand &LHS,SDOperand &RHS, ISD::CondCode Code);

  //===--------------------------------------------------------------------===//
  // Expansion Support: LegalizeTypesExpand.cpp
  //===--------------------------------------------------------------------===//
  
  void GetExpandedOp(SDOperand Op, SDOperand &Lo, SDOperand &Hi);
  void SetExpandedOp(SDOperand Op, SDOperand Lo, SDOperand Hi);
    
  // Result Expansion.
  void ExpandResult(SDNode *N, unsigned ResNo);
  void ExpandResult_ANY_EXTEND (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_AssertZext (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_BIT_CONVERT(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_BUILD_PAIR (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_Constant   (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_CTLZ       (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_CTPOP      (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_CTTZ       (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_EXTRACT_VECTOR_ELT(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_LOAD       (LoadSDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_MERGE_VALUES(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SIGN_EXTEND(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SIGN_EXTEND_INREG(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_TRUNCATE   (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_UNDEF      (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ZERO_EXTEND(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_FP_TO_SINT (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_FP_TO_UINT (SDNode *N, SDOperand &Lo, SDOperand &Hi);

  void ExpandResult_Logical    (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_BSWAP      (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ADDSUB     (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ADDSUBC    (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ADDSUBE    (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SELECT     (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SELECT_CC  (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_MUL        (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SDIV       (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SREM       (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_UDIV       (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_UREM       (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_Shift      (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  
  void ExpandShiftByConstant(SDNode *N, unsigned Amt, 
                             SDOperand &Lo, SDOperand &Hi);
  bool ExpandShiftWithKnownAmountBit(SDNode *N, SDOperand &Lo, SDOperand &Hi);

  // Operand Expansion.
  bool ExpandOperand(SDNode *N, unsigned OperandNo);
  SDOperand ExpandOperand_BIT_CONVERT(SDNode *N);
  SDOperand ExpandOperand_BR_CC(SDNode *N);
  SDOperand ExpandOperand_BUILD_VECTOR(SDNode *N);
  SDOperand ExpandOperand_EXTRACT_ELEMENT(SDNode *N);
  SDOperand ExpandOperand_SETCC(SDNode *N);
  SDOperand ExpandOperand_SINT_TO_FP(SDOperand Source, MVT DestTy);
  SDOperand ExpandOperand_STORE(StoreSDNode *N, unsigned OpNo);
  SDOperand ExpandOperand_TRUNCATE(SDNode *N);
  SDOperand ExpandOperand_UINT_TO_FP(SDOperand Source, MVT DestTy);

  void ExpandSetCCOperands(SDOperand &NewLHS, SDOperand &NewRHS,
                           ISD::CondCode &CCCode);
  
  //===--------------------------------------------------------------------===//
  // Float to Integer Conversion Support: LegalizeTypesFloatToInt.cpp
  //===--------------------------------------------------------------------===//

  SDOperand GetIntegerOp(SDOperand Op) {
    SDOperand &IntegerOp = FloatToIntedNodes[Op];
    RemapNode(IntegerOp);
    assert(IntegerOp.Val && "Operand wasn't converted to integer?");
    return IntegerOp;
  }
  void SetIntegerOp(SDOperand Op, SDOperand Result);

  // Result Float to Integer Conversion.
  void FloatToIntResult(SDNode *N, unsigned OpNo);
  SDOperand FloatToIntRes_BIT_CONVERT(SDNode *N);
  SDOperand FloatToIntRes_BUILD_PAIR(SDNode *N);
  SDOperand FloatToIntRes_ConstantFP(ConstantFPSDNode *N);
  SDOperand FloatToIntRes_FADD(SDNode *N);
  SDOperand FloatToIntRes_FCOPYSIGN(SDNode *N);
  SDOperand FloatToIntRes_FMUL(SDNode *N);
  SDOperand FloatToIntRes_FSUB(SDNode *N);
  SDOperand FloatToIntRes_LOAD(SDNode *N);
  SDOperand FloatToIntRes_XINT_TO_FP(SDNode *N);

  // Operand Float to Integer Conversion.
  bool FloatToIntOperand(SDNode *N, unsigned OpNo);
  SDOperand FloatToIntOp_BIT_CONVERT(SDNode *N);

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
  SDOperand ScalarizeRes_BinOp(SDNode *N);
  SDOperand ScalarizeRes_UnaryOp(SDNode *N);

  SDOperand ScalarizeRes_BIT_CONVERT(SDNode *N);
  SDOperand ScalarizeRes_FPOWI(SDNode *N);
  SDOperand ScalarizeRes_INSERT_VECTOR_ELT(SDNode *N);
  SDOperand ScalarizeRes_LOAD(LoadSDNode *N);
  SDOperand ScalarizeRes_SELECT(SDNode *N);
  SDOperand ScalarizeRes_UNDEF(SDNode *N);
  SDOperand ScalarizeRes_VECTOR_SHUFFLE(SDNode *N);

  // Operand Vector Scalarization: <1 x ty> -> ty.
  bool ScalarizeOperand(SDNode *N, unsigned OpNo);
  SDOperand ScalarizeOp_BIT_CONVERT(SDNode *N);
  SDOperand ScalarizeOp_EXTRACT_VECTOR_ELT(SDNode *N);
  SDOperand ScalarizeOp_STORE(StoreSDNode *N, unsigned OpNo);

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
  
  // Operand Vector Splitting: <128 x ty> -> 2 x <64 x ty>.
  bool SplitOperand(SDNode *N, unsigned OpNo);

  SDOperand SplitOp_BIT_CONVERT(SDNode *N);
  SDOperand SplitOp_EXTRACT_SUBVECTOR(SDNode *N);
  SDOperand SplitOp_EXTRACT_VECTOR_ELT(SDNode *N);
  SDOperand SplitOp_RET(SDNode *N, unsigned OpNo);
  SDOperand SplitOp_STORE(StoreSDNode *N, unsigned OpNo);
  SDOperand SplitOp_VECTOR_SHUFFLE(SDNode *N, unsigned OpNo);
};

} // end namespace llvm.

#endif
