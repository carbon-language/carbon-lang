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
/// DAGTypeLegalizer - This takes an arbitrary SelectionDAG as input and hacks
/// on it until only value types the target machine can handle are left.  This
/// involves promoting small sizes to large sizes or splitting up large values
/// into small values.
///
class VISIBILITY_HIDDEN DAGTypeLegalizer {
  TargetLowering &TLI;
  SelectionDAG &DAG;
public:
  // NodeIdFlags - This pass uses the NodeId on the SDNodes to hold information
  // about the state of the node.  The enum has all the values.
  enum NodeIdFlags {
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
    Legal,           // The target natively supports this type.
    PromoteInteger,  // Replace this integer type with a larger one.
    ExpandInteger,   // Split this integer type into two of half the size.
    SoftenFloat,     // Convert this float type to a same size integer type.
    ExpandFloat,     // Split this float type into two of half the size.
    ScalarizeVector, // Replace this one-element vector with its element type.
    SplitVector      // This vector type should be split into smaller vectors.
  };

  /// ValueTypeActions - This is a bitvector that contains two bits for each
  /// simple value type, where the two bits correspond to the LegalizeAction
  /// enum from TargetLowering.  This can be queried with "getTypeAction(VT)".
  TargetLowering::ValueTypeActionImpl ValueTypeActions;

  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal, or we need to promote it to a larger integer type, or
  /// we need to expand it into multiple registers of a smaller integer type, or
  /// we need to split a vector type into smaller vector types, or we need to
  /// convert it to a different type of the same size.
  LegalizeAction getTypeAction(MVT VT) const {
    switch (ValueTypeActions.getTypeAction(VT)) {
    default:
      assert(false && "Unknown legalize action!");
    case TargetLowering::Legal:
      return Legal;
    case TargetLowering::Promote:
      // Promote can mean
      //   1) On integers, it means to promote type (e.g., i8 to i32)
      //   2) For vectors, it means try to widen (e.g., v3i32 to v4i32)
      if (!VT.isVector()) {
        return PromoteInteger;
      }
      else {
        // TODO: move widen code to LegalizeType
        if (VT.getVectorNumElements() == 1) {
          return ScalarizeVector;
        } else {
          return SplitVector;
        }        
      }
    case TargetLowering::Expand:
      // Expand can mean
      // 1) split scalar in half, 2) convert a float to an integer,
      // 3) scalarize a single-element vector, 4) split a vector in two.
      if (!VT.isVector()) {
        if (VT.isInteger())
          return ExpandInteger;
        else if (VT.getSizeInBits() ==
                 TLI.getTypeToTransformTo(VT).getSizeInBits())
          return SoftenFloat;
        else
          return ExpandFloat;
      } else if (VT.getVectorNumElements() == 1) {
        return ScalarizeVector;
      } else {
        return SplitVector;
      }
    }
  }

  /// isTypeLegal - Return true if this type is legal on this target.
  bool isTypeLegal(MVT VT) const {
    return ValueTypeActions.getTypeAction(VT) == TargetLowering::Legal;
  }

  /// IgnoreNodeResults - Pretend all of this node's results are legal.
  bool IgnoreNodeResults(SDNode *N) const {
    return N->getOpcode() == ISD::TargetConstant;
  }

  /// PromotedIntegers - For integer nodes that are below legal width, this map
  /// indicates what promoted value to use.
  DenseMap<SDValue, SDValue> PromotedIntegers;

  /// ExpandedIntegers - For integer nodes that need to be expanded this map
  /// indicates which operands are the expanded version of the input.
  DenseMap<SDValue, std::pair<SDValue, SDValue> > ExpandedIntegers;

  /// SoftenedFloats - For floating point nodes converted to integers of
  /// the same size, this map indicates the converted value to use.
  DenseMap<SDValue, SDValue> SoftenedFloats;

  /// ExpandedFloats - For float nodes that need to be expanded this map
  /// indicates which operands are the expanded version of the input.
  DenseMap<SDValue, std::pair<SDValue, SDValue> > ExpandedFloats;

  /// ScalarizedVectors - For nodes that are <1 x ty>, this map indicates the
  /// scalar value of type 'ty' to use.
  DenseMap<SDValue, SDValue> ScalarizedVectors;

  /// SplitVectors - For nodes that need to be split this map indicates
  /// which operands are the expanded version of the input.
  DenseMap<SDValue, std::pair<SDValue, SDValue> > SplitVectors;

  /// ReplacedValues - For values that have been replaced with another,
  /// indicates the replacement value to use.
  DenseMap<SDValue, SDValue> ReplacedValues;

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

  /// ReanalyzeNode - Recompute the NodeId and correct processed operands
  /// for the specified node, adding it to the worklist if ready.
  void ReanalyzeNode(SDNode *N) {
    N->setNodeId(NewNode);
    AnalyzeNewNode(N);
    // The node may have changed but we don't care.
  }

  void NoteDeletion(SDNode *Old, SDNode *New) {
    ExpungeNode(Old);
    ExpungeNode(New);
    for (unsigned i = 0, e = Old->getNumValues(); i != e; ++i)
      ReplacedValues[SDValue(Old, i)] = SDValue(New, i);
  }

private:
  SDNode *AnalyzeNewNode(SDNode *N);
  void AnalyzeNewValue(SDValue &Val);

  void ReplaceValueWith(SDValue From, SDValue To);
  void ReplaceNodeWith(SDNode *From, SDNode *To);

  void RemapValue(SDValue &N);
  void ExpungeNode(SDNode *N);

  // Common routines.
  SDValue CreateStackStoreLoad(SDValue Op, MVT DestVT);
  SDValue MakeLibCall(RTLIB::Libcall LC, MVT RetVT,
                      const SDValue *Ops, unsigned NumOps, bool isSigned);
  SDValue LibCallify(RTLIB::Libcall LC, SDNode *N, bool isSigned);

  SDValue BitConvertToInteger(SDValue Op);
  SDValue JoinIntegers(SDValue Lo, SDValue Hi);
  void SplitInteger(SDValue Op, SDValue &Lo, SDValue &Hi);
  void SplitInteger(SDValue Op, MVT LoVT, MVT HiVT,
                    SDValue &Lo, SDValue &Hi);

  SDValue GetVectorElementPointer(SDValue VecPtr, MVT EltVT, SDValue Index);

  //===--------------------------------------------------------------------===//
  // Integer Promotion Support: LegalizeIntegerTypes.cpp
  //===--------------------------------------------------------------------===//

  SDValue GetPromotedInteger(SDValue Op) {
    SDValue &PromotedOp = PromotedIntegers[Op];
    RemapValue(PromotedOp);
    assert(PromotedOp.getNode() && "Operand wasn't promoted?");
    return PromotedOp;
  }
  void SetPromotedInteger(SDValue Op, SDValue Result);

  /// ZExtPromotedInteger - Get a promoted operand and zero extend it to the
  /// final size.
  SDValue ZExtPromotedInteger(SDValue Op) {
    MVT OldVT = Op.getValueType();
    Op = GetPromotedInteger(Op);
    return DAG.getZeroExtendInReg(Op, OldVT);
  }

  // Integer Result Promotion.
  void PromoteIntegerResult(SDNode *N, unsigned ResNo);
  SDValue PromoteIntRes_AssertSext(SDNode *N);
  SDValue PromoteIntRes_AssertZext(SDNode *N);
  SDValue PromoteIntRes_Atomic1(AtomicSDNode *N);
  SDValue PromoteIntRes_Atomic2(AtomicSDNode *N);
  SDValue PromoteIntRes_BIT_CONVERT(SDNode *N);
  SDValue PromoteIntRes_BSWAP(SDNode *N);
  SDValue PromoteIntRes_BUILD_PAIR(SDNode *N);
  SDValue PromoteIntRes_Constant(SDNode *N);
  SDValue PromoteIntRes_CTLZ(SDNode *N);
  SDValue PromoteIntRes_CTPOP(SDNode *N);
  SDValue PromoteIntRes_CTTZ(SDNode *N);
  SDValue PromoteIntRes_EXTRACT_VECTOR_ELT(SDNode *N);
  SDValue PromoteIntRes_FP_TO_XINT(SDNode *N);
  SDValue PromoteIntRes_INT_EXTEND(SDNode *N);
  SDValue PromoteIntRes_LOAD(LoadSDNode *N);
  SDValue PromoteIntRes_SDIV(SDNode *N);
  SDValue PromoteIntRes_SELECT   (SDNode *N);
  SDValue PromoteIntRes_SELECT_CC(SDNode *N);
  SDValue PromoteIntRes_SETCC(SDNode *N);
  SDValue PromoteIntRes_SHL(SDNode *N);
  SDValue PromoteIntRes_SimpleIntBinOp(SDNode *N);
  SDValue PromoteIntRes_SIGN_EXTEND_INREG(SDNode *N);
  SDValue PromoteIntRes_SRA(SDNode *N);
  SDValue PromoteIntRes_SRL(SDNode *N);
  SDValue PromoteIntRes_TRUNCATE(SDNode *N);
  SDValue PromoteIntRes_UDIV(SDNode *N);
  SDValue PromoteIntRes_UNDEF(SDNode *N);
  SDValue PromoteIntRes_VAARG(SDNode *N);

  // Integer Operand Promotion.
  bool PromoteIntegerOperand(SDNode *N, unsigned OperandNo);
  SDValue PromoteIntOp_ANY_EXTEND(SDNode *N);
  SDValue PromoteIntOp_BUILD_PAIR(SDNode *N);
  SDValue PromoteIntOp_BR_CC(SDNode *N, unsigned OpNo);
  SDValue PromoteIntOp_BRCOND(SDNode *N, unsigned OpNo);
  SDValue PromoteIntOp_BUILD_VECTOR(SDNode *N);
  SDValue PromoteIntOp_FP_EXTEND(SDNode *N);
  SDValue PromoteIntOp_FP_ROUND(SDNode *N);
  SDValue PromoteIntOp_INT_TO_FP(SDNode *N);
  SDValue PromoteIntOp_INSERT_VECTOR_ELT(SDNode *N, unsigned OpNo);
  SDValue PromoteIntOp_MEMBARRIER(SDNode *N);
  SDValue PromoteIntOp_SELECT(SDNode *N, unsigned OpNo);
  SDValue PromoteIntOp_SELECT_CC(SDNode *N, unsigned OpNo);
  SDValue PromoteIntOp_SETCC(SDNode *N, unsigned OpNo);
  SDValue PromoteIntOp_SIGN_EXTEND(SDNode *N);
  SDValue PromoteIntOp_STORE(StoreSDNode *N, unsigned OpNo);
  SDValue PromoteIntOp_TRUNCATE(SDNode *N);
  SDValue PromoteIntOp_ZERO_EXTEND(SDNode *N);

  void PromoteSetCCOperands(SDValue &LHS,SDValue &RHS, ISD::CondCode Code);

  //===--------------------------------------------------------------------===//
  // Integer Expansion Support: LegalizeIntegerTypes.cpp
  //===--------------------------------------------------------------------===//

  void GetExpandedInteger(SDValue Op, SDValue &Lo, SDValue &Hi);
  void SetExpandedInteger(SDValue Op, SDValue Lo, SDValue Hi);

  // Integer Result Expansion.
  void ExpandIntegerResult(SDNode *N, unsigned ResNo);
  void ExpandIntRes_ANY_EXTEND        (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_AssertSext        (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_AssertZext        (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_Constant          (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_CTLZ              (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_CTPOP             (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_CTTZ              (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_LOAD          (LoadSDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_SIGN_EXTEND       (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_SIGN_EXTEND_INREG (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_TRUNCATE          (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_ZERO_EXTEND       (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_FP_TO_SINT        (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_FP_TO_UINT        (SDNode *N, SDValue &Lo, SDValue &Hi);

  void ExpandIntRes_Logical           (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_ADDSUB            (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_ADDSUBC           (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_ADDSUBE           (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_BSWAP             (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_MUL               (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_SDIV              (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_SREM              (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_UDIV              (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_UREM              (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandIntRes_Shift             (SDNode *N, SDValue &Lo, SDValue &Hi);

  void ExpandShiftByConstant(SDNode *N, unsigned Amt,
                             SDValue &Lo, SDValue &Hi);
  bool ExpandShiftWithKnownAmountBit(SDNode *N, SDValue &Lo, SDValue &Hi);

  // Integer Operand Expansion.
  bool ExpandIntegerOperand(SDNode *N, unsigned OperandNo);
  SDValue ExpandIntOp_BIT_CONVERT(SDNode *N);
  SDValue ExpandIntOp_BR_CC(SDNode *N);
  SDValue ExpandIntOp_BUILD_VECTOR(SDNode *N);
  SDValue ExpandIntOp_EXTRACT_ELEMENT(SDNode *N);
  SDValue ExpandIntOp_SELECT_CC(SDNode *N);
  SDValue ExpandIntOp_SETCC(SDNode *N);
  SDValue ExpandIntOp_SINT_TO_FP(SDNode *N);
  SDValue ExpandIntOp_STORE(StoreSDNode *N, unsigned OpNo);
  SDValue ExpandIntOp_TRUNCATE(SDNode *N);
  SDValue ExpandIntOp_UINT_TO_FP(SDNode *N);

  void IntegerExpandSetCCOperands(SDValue &NewLHS, SDValue &NewRHS,
                                  ISD::CondCode &CCCode);

  //===--------------------------------------------------------------------===//
  // Float to Integer Conversion Support: LegalizeFloatTypes.cpp
  //===--------------------------------------------------------------------===//

  SDValue GetSoftenedFloat(SDValue Op) {
    SDValue &SoftenedOp = SoftenedFloats[Op];
    RemapValue(SoftenedOp);
    assert(SoftenedOp.getNode() && "Operand wasn't converted to integer?");
    return SoftenedOp;
  }
  void SetSoftenedFloat(SDValue Op, SDValue Result);

  // Result Float to Integer Conversion.
  void SoftenFloatResult(SDNode *N, unsigned OpNo);
  SDValue SoftenFloatRes_BIT_CONVERT(SDNode *N);
  SDValue SoftenFloatRes_BUILD_PAIR(SDNode *N);
  SDValue SoftenFloatRes_ConstantFP(ConstantFPSDNode *N);
  SDValue SoftenFloatRes_FABS(SDNode *N);
  SDValue SoftenFloatRes_FADD(SDNode *N);
  SDValue SoftenFloatRes_FCOPYSIGN(SDNode *N);
  SDValue SoftenFloatRes_FDIV(SDNode *N);
  SDValue SoftenFloatRes_FMUL(SDNode *N);
  SDValue SoftenFloatRes_FP_EXTEND(SDNode *N);
  SDValue SoftenFloatRes_FP_ROUND(SDNode *N);
  SDValue SoftenFloatRes_FPOW(SDNode *N);
  SDValue SoftenFloatRes_FPOWI(SDNode *N);
  SDValue SoftenFloatRes_FSUB(SDNode *N);
  SDValue SoftenFloatRes_LOAD(SDNode *N);
  SDValue SoftenFloatRes_SELECT(SDNode *N);
  SDValue SoftenFloatRes_SELECT_CC(SDNode *N);
  SDValue SoftenFloatRes_SINT_TO_FP(SDNode *N);
  SDValue SoftenFloatRes_UINT_TO_FP(SDNode *N);

  // Operand Float to Integer Conversion.
  bool SoftenFloatOperand(SDNode *N, unsigned OpNo);
  SDValue SoftenFloatOp_BIT_CONVERT(SDNode *N);
  SDValue SoftenFloatOp_BR_CC(SDNode *N);
  SDValue SoftenFloatOp_FP_ROUND(SDNode *N);
  SDValue SoftenFloatOp_FP_TO_SINT(SDNode *N);
  SDValue SoftenFloatOp_FP_TO_UINT(SDNode *N);
  SDValue SoftenFloatOp_SELECT_CC(SDNode *N);
  SDValue SoftenFloatOp_SETCC(SDNode *N);
  SDValue SoftenFloatOp_STORE(SDNode *N, unsigned OpNo);

  void SoftenSetCCOperands(SDValue &NewLHS, SDValue &NewRHS,
                           ISD::CondCode &CCCode);

  //===--------------------------------------------------------------------===//
  // Float Expansion Support: LegalizeFloatTypes.cpp
  //===--------------------------------------------------------------------===//

  void GetExpandedFloat(SDValue Op, SDValue &Lo, SDValue &Hi);
  void SetExpandedFloat(SDValue Op, SDValue Lo, SDValue Hi);

  // Float Result Expansion.
  void ExpandFloatResult(SDNode *N, unsigned ResNo);
  void ExpandFloatRes_ConstantFP(SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FABS      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FADD      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FCEIL     (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FCOS      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FDIV      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FEXP      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FEXP2     (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FFLOOR    (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FLOG      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FLOG2     (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FLOG10    (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FMUL      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FNEARBYINT(SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FNEG      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FP_EXTEND (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FPOW      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FPOWI     (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FRINT     (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FSIN      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FSQRT     (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FSUB      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_FTRUNC    (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_LOAD      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandFloatRes_XINT_TO_FP(SDNode *N, SDValue &Lo, SDValue &Hi);

  // Float Operand Expansion.
  bool ExpandFloatOperand(SDNode *N, unsigned OperandNo);
  SDValue ExpandFloatOp_BR_CC(SDNode *N);
  SDValue ExpandFloatOp_FP_ROUND(SDNode *N);
  SDValue ExpandFloatOp_FP_TO_SINT(SDNode *N);
  SDValue ExpandFloatOp_FP_TO_UINT(SDNode *N);
  SDValue ExpandFloatOp_SELECT_CC(SDNode *N);
  SDValue ExpandFloatOp_SETCC(SDNode *N);
  SDValue ExpandFloatOp_STORE(SDNode *N, unsigned OpNo);

  void FloatExpandSetCCOperands(SDValue &NewLHS, SDValue &NewRHS,
                                ISD::CondCode &CCCode);

  //===--------------------------------------------------------------------===//
  // Scalarization Support: LegalizeVectorTypes.cpp
  //===--------------------------------------------------------------------===//

  SDValue GetScalarizedVector(SDValue Op) {
    SDValue &ScalarizedOp = ScalarizedVectors[Op];
    RemapValue(ScalarizedOp);
    assert(ScalarizedOp.getNode() && "Operand wasn't scalarized?");
    return ScalarizedOp;
  }
  void SetScalarizedVector(SDValue Op, SDValue Result);

  // Vector Result Scalarization: <1 x ty> -> ty.
  void ScalarizeVectorResult(SDNode *N, unsigned OpNo);
  SDValue ScalarizeVecRes_BinOp(SDNode *N);
  SDValue ScalarizeVecRes_UnaryOp(SDNode *N);

  SDValue ScalarizeVecRes_BIT_CONVERT(SDNode *N);
  SDValue ScalarizeVecRes_EXTRACT_SUBVECTOR(SDNode *N);
  SDValue ScalarizeVecRes_FPOWI(SDNode *N);
  SDValue ScalarizeVecRes_INSERT_VECTOR_ELT(SDNode *N);
  SDValue ScalarizeVecRes_LOAD(LoadSDNode *N);
  SDValue ScalarizeVecRes_SELECT(SDNode *N);
  SDValue ScalarizeVecRes_UNDEF(SDNode *N);
  SDValue ScalarizeVecRes_VECTOR_SHUFFLE(SDNode *N);
  SDValue ScalarizeVecRes_VSETCC(SDNode *N);

  // Vector Operand Scalarization: <1 x ty> -> ty.
  bool ScalarizeVectorOperand(SDNode *N, unsigned OpNo);
  SDValue ScalarizeVecOp_BIT_CONVERT(SDNode *N);
  SDValue ScalarizeVecOp_CONCAT_VECTORS(SDNode *N);
  SDValue ScalarizeVecOp_EXTRACT_VECTOR_ELT(SDNode *N);
  SDValue ScalarizeVecOp_STORE(StoreSDNode *N, unsigned OpNo);

  //===--------------------------------------------------------------------===//
  // Vector Splitting Support: LegalizeVectorTypes.cpp
  //===--------------------------------------------------------------------===//

  void GetSplitVector(SDValue Op, SDValue &Lo, SDValue &Hi);
  void SetSplitVector(SDValue Op, SDValue Lo, SDValue Hi);

  // Vector Result Splitting: <128 x ty> -> 2 x <64 x ty>.
  void SplitVectorResult(SDNode *N, unsigned OpNo);
  void SplitVecRes_BinOp(SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitVecRes_UnaryOp(SDNode *N, SDValue &Lo, SDValue &Hi);

  void SplitVecRes_BIT_CONVERT(SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitVecRes_BUILD_PAIR(SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitVecRes_BUILD_VECTOR(SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitVecRes_CONCAT_VECTORS(SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitVecRes_FPOWI(SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitVecRes_INSERT_VECTOR_ELT(SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitVecRes_LOAD(LoadSDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitVecRes_UNDEF(SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitVecRes_VECTOR_SHUFFLE(SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitVecRes_VSETCC(SDNode *N, SDValue &Lo, SDValue &Hi);

  // Vector Operand Splitting: <128 x ty> -> 2 x <64 x ty>.
  bool SplitVectorOperand(SDNode *N, unsigned OpNo);
  SDValue SplitVecOp_UnaryOp(SDNode *N);

  SDValue SplitVecOp_BIT_CONVERT(SDNode *N);
  SDValue SplitVecOp_EXTRACT_SUBVECTOR(SDNode *N);
  SDValue SplitVecOp_EXTRACT_VECTOR_ELT(SDNode *N);
  SDValue SplitVecOp_STORE(StoreSDNode *N, unsigned OpNo);
  SDValue SplitVecOp_VECTOR_SHUFFLE(SDNode *N, unsigned OpNo);

  //===--------------------------------------------------------------------===//
  // Generic Splitting: LegalizeTypesGeneric.cpp
  //===--------------------------------------------------------------------===//

  // Legalization methods which only use that the illegal type is split into two
  // not necessarily identical types.  As such they can be used for splitting
  // vectors and expanding integers and floats.

  void GetSplitOp(SDValue Op, SDValue &Lo, SDValue &Hi) {
    if (Op.getValueType().isVector())
      GetSplitVector(Op, Lo, Hi);
    else if (Op.getValueType().isInteger())
      GetExpandedInteger(Op, Lo, Hi);
    else
      GetExpandedFloat(Op, Lo, Hi);
  }

  /// GetSplitDestVTs - Compute the VTs needed for the low/hi parts of a type
  /// which is split (or expanded) into two not necessarily identical pieces.
  void GetSplitDestVTs(MVT InVT, MVT &LoVT, MVT &HiVT);

  // Generic Result Splitting.
  void SplitRes_MERGE_VALUES(SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitRes_SELECT      (SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitRes_SELECT_CC   (SDNode *N, SDValue &Lo, SDValue &Hi);
  void SplitRes_UNDEF       (SDNode *N, SDValue &Lo, SDValue &Hi);

  //===--------------------------------------------------------------------===//
  // Generic Expansion: LegalizeTypesGeneric.cpp
  //===--------------------------------------------------------------------===//

  // Legalization methods which only use that the illegal type is split into two
  // identical types of half the size, and that the Lo/Hi part is stored first
  // in memory on little/big-endian machines, followed by the Hi/Lo part.  As
  // such they can be used for expanding integers and floats.

  void GetExpandedOp(SDValue Op, SDValue &Lo, SDValue &Hi) {
    if (Op.getValueType().isInteger())
      GetExpandedInteger(Op, Lo, Hi);
    else
      GetExpandedFloat(Op, Lo, Hi);
  }

  // Generic Result Expansion.
  void ExpandRes_BIT_CONVERT       (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandRes_BUILD_PAIR        (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandRes_EXTRACT_ELEMENT   (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandRes_EXTRACT_VECTOR_ELT(SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandRes_NormalLoad        (SDNode *N, SDValue &Lo, SDValue &Hi);
  void ExpandRes_VAARG             (SDNode *N, SDValue &Lo, SDValue &Hi);

  // Generic Operand Expansion.
  SDValue ExpandOp_BIT_CONVERT    (SDNode *N);
  SDValue ExpandOp_BUILD_VECTOR   (SDNode *N);
  SDValue ExpandOp_EXTRACT_ELEMENT(SDNode *N);
  SDValue ExpandOp_NormalStore    (SDNode *N, unsigned OpNo);

};

} // end namespace llvm.

#endif
