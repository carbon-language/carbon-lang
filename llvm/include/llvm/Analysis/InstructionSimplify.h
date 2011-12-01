//===-- InstructionSimplify.h - Fold instructions into simpler forms ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares routines for folding instructions into simpler forms
// that do not require creating new instructions.  This does constant folding
// ("add i32 1, 1" -> "2") but can also handle non-constant operands, either
// returning a constant ("and i32 %x, 0" -> "0") or an already existing value
// ("and i32 %x, %x" -> "%x").  If the simplification is also an instruction
// then it dominates the original instruction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_INSTRUCTIONSIMPLIFY_H
#define LLVM_ANALYSIS_INSTRUCTIONSIMPLIFY_H

namespace llvm {
  class DominatorTree;
  class Instruction;
  class Value;
  class TargetData;
  class TargetLibraryInfo;
  template<typename T>
  class ArrayRef;

  /// SimplifyAddInst - Given operands for an Add, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyAddInst(Value *LHS, Value *RHS, bool isNSW, bool isNUW,
                         const TargetData *TD = 0,
                         const TargetLibraryInfo *TLI = 0,
                         const DominatorTree *DT = 0);

  /// SimplifySubInst - Given operands for a Sub, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifySubInst(Value *LHS, Value *RHS, bool isNSW, bool isNUW,
                         const TargetData *TD = 0,
                         const TargetLibraryInfo *TLI = 0,
                         const DominatorTree *DT = 0);

  /// SimplifyMulInst - Given operands for a Mul, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyMulInst(Value *LHS, Value *RHS, const TargetData *TD = 0,
                         const TargetLibraryInfo *TLI = 0,
                         const DominatorTree *DT = 0);

  /// SimplifySDivInst - Given operands for an SDiv, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifySDivInst(Value *LHS, Value *RHS, const TargetData *TD = 0,
                          const TargetLibraryInfo *TLI = 0,
                          const DominatorTree *DT = 0);

  /// SimplifyUDivInst - Given operands for a UDiv, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyUDivInst(Value *LHS, Value *RHS, const TargetData *TD = 0, 
                          const TargetLibraryInfo *TLI = 0,
                          const DominatorTree *DT = 0);

  /// SimplifyFDivInst - Given operands for an FDiv, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyFDivInst(Value *LHS, Value *RHS, const TargetData *TD = 0,
                          const TargetLibraryInfo *TLI = 0,
                          const DominatorTree *DT = 0);

  /// SimplifySRemInst - Given operands for an SRem, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifySRemInst(Value *LHS, Value *RHS, const TargetData *TD = 0, 
                          const TargetLibraryInfo *TLI = 0,
                          const DominatorTree *DT = 0);

  /// SimplifyURemInst - Given operands for a URem, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyURemInst(Value *LHS, Value *RHS, const TargetData *TD = 0,
                          const TargetLibraryInfo *TLI = 0,
                          const DominatorTree *DT = 0);

  /// SimplifyFRemInst - Given operands for an FRem, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyFRemInst(Value *LHS, Value *RHS, const TargetData *TD = 0,
                          const TargetLibraryInfo *TLI = 0,
                          const DominatorTree *DT = 0);

  /// SimplifyShlInst - Given operands for a Shl, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyShlInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                         const TargetData *TD = 0, 
                         const TargetLibraryInfo *TLI = 0,
                         const DominatorTree *DT = 0);

  /// SimplifyLShrInst - Given operands for a LShr, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyLShrInst(Value *Op0, Value *Op1, bool isExact,
                          const TargetData *TD = 0,
                          const TargetLibraryInfo *TLI = 0,
                          const DominatorTree *DT = 0);

  /// SimplifyAShrInst - Given operands for a AShr, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyAShrInst(Value *Op0, Value *Op1, bool isExact,
                          const TargetData *TD = 0,
                          const TargetLibraryInfo *TLI = 0,
                          const DominatorTree *DT = 0);

  /// SimplifyAndInst - Given operands for an And, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyAndInst(Value *LHS, Value *RHS, const TargetData *TD = 0,
                         const TargetLibraryInfo *TLI = 0,
                         const DominatorTree *DT = 0);

  /// SimplifyOrInst - Given operands for an Or, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyOrInst(Value *LHS, Value *RHS, const TargetData *TD = 0,
                        const TargetLibraryInfo *TLI = 0,
                        const DominatorTree *DT = 0);

  /// SimplifyXorInst - Given operands for a Xor, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyXorInst(Value *LHS, Value *RHS, const TargetData *TD = 0,
                         const TargetLibraryInfo *TLI = 0,
                         const DominatorTree *DT = 0);

  /// SimplifyICmpInst - Given operands for an ICmpInst, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                          const TargetData *TD = 0, 
                          const TargetLibraryInfo *TLI = 0,
                          const DominatorTree *DT = 0);

  /// SimplifyFCmpInst - Given operands for an FCmpInst, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyFCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                          const TargetData *TD = 0, 
                          const TargetLibraryInfo *TLI = 0,
                          const DominatorTree *DT = 0);

  /// SimplifySelectInst - Given operands for a SelectInst, see if we can fold
  /// the result.  If not, this returns null.
  Value *SimplifySelectInst(Value *Cond, Value *TrueVal, Value *FalseVal,
                            const TargetData *TD = 0,
                            const DominatorTree *DT = 0);

  /// SimplifyGEPInst - Given operands for an GetElementPtrInst, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyGEPInst(ArrayRef<Value *> Ops, const TargetData *TD = 0,
                         const DominatorTree *DT = 0);

  /// SimplifyInsertValueInst - Given operands for an InsertValueInst, see if we
  /// can fold the result.  If not, this returns null.
  Value *SimplifyInsertValueInst(Value *Agg, Value *Val,
                                 ArrayRef<unsigned> Idxs,
                                 const TargetData *TD = 0,
                                 const DominatorTree *DT = 0);

  //=== Helper functions for higher up the class hierarchy.


  /// SimplifyCmpInst - Given operands for a CmpInst, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                         const TargetData *TD = 0,
                         const TargetLibraryInfo *TLI = 0,
                         const DominatorTree *DT = 0);

  /// SimplifyBinOp - Given operands for a BinaryOperator, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                       const TargetData *TD = 0, 
                       const TargetLibraryInfo *TLI = 0,
                       const DominatorTree *DT = 0);

  /// SimplifyInstruction - See if we can compute a simplified version of this
  /// instruction.  If not, this returns null.
  Value *SimplifyInstruction(Instruction *I, const TargetData *TD = 0,
                             const TargetLibraryInfo *TLI = 0,
                             const DominatorTree *DT = 0);


  /// ReplaceAndSimplifyAllUses - Perform From->replaceAllUsesWith(To) and then
  /// delete the From instruction.  In addition to a basic RAUW, this does a
  /// recursive simplification of the updated instructions.  This catches
  /// things where one simplification exposes other opportunities.  This only
  /// simplifies and deletes scalar operations, it does not change the CFG.
  ///
  void ReplaceAndSimplifyAllUses(Instruction *From, Value *To,
                                 const TargetData *TD = 0,
                                 const TargetLibraryInfo *TLI = 0,
                                 const DominatorTree *DT = 0);
} // end namespace llvm

#endif

