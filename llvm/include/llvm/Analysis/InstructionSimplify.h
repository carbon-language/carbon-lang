//===-- InstructionSimplify.h - Fold instructions into simpler forms ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares routines for folding instructions into simpler forms that
// do not require creating new instructions.  For example, this does constant
// folding, and can handle identities like (X&0)->0.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_INSTRUCTIONSIMPLIFY_H
#define LLVM_ANALYSIS_INSTRUCTIONSIMPLIFY_H

namespace llvm {
  class Value;
  class TargetData;
  
  /// SimplifyCompare - Given operands for a CmpInst, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyCompare(unsigned Predicate, Value *LHS, Value *RHS,
                         const TargetData *TD = 0);
  
  
  /// SimplifyBinOp - Given operands for a BinaryOperator, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS, 
                       const TargetData *TD = 0);
  
} // end namespace llvm

#endif

