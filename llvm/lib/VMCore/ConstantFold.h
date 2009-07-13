//===-- ConstantFolding.h - Internal Constant Folding Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the (internal) constant folding interfaces for LLVM.  These
// interfaces are used by the ConstantExpr::get* methods to automatically fold
// constants when possible.
//
// These operators may return a null object if they don't know how to perform
// the specified operation on the specified constant types.
//
//===----------------------------------------------------------------------===//

#ifndef CONSTANTFOLDING_H
#define CONSTANTFOLDING_H

namespace llvm {
  class Value;
  class Constant;
  class Type;
  class LLVMContext;

  // Constant fold various types of instruction...
  Constant *ConstantFoldCastInstruction(
    LLVMContext &Context,
    unsigned opcode,     ///< The opcode of the cast
    const Constant *V,   ///< The source constant
    const Type *DestTy   ///< The destination type
  );
  Constant *ConstantFoldSelectInstruction(LLVMContext &Context,
                                          const Constant *Cond,
                                          const Constant *V1,
                                          const Constant *V2);
  Constant *ConstantFoldExtractElementInstruction(LLVMContext &Context,
                                                  const Constant *Val,
                                                  const Constant *Idx);
  Constant *ConstantFoldInsertElementInstruction(LLVMContext &Context,
                                                 const Constant *Val,
                                                 const Constant *Elt,
                                                 const Constant *Idx);
  Constant *ConstantFoldShuffleVectorInstruction(LLVMContext &Context,
                                                 const Constant *V1,
                                                 const Constant *V2,
                                                 const Constant *Mask);
  Constant *ConstantFoldExtractValueInstruction(LLVMContext &Context,
                                                const Constant *Agg,
                                                const unsigned *Idxs,
                                                unsigned NumIdx);
  Constant *ConstantFoldInsertValueInstruction(LLVMContext &Context,
                                               const Constant *Agg,
                                               const Constant *Val,
                                               const unsigned* Idxs,
                                               unsigned NumIdx);
  Constant *ConstantFoldBinaryInstruction(LLVMContext &Context,
                                          unsigned Opcode, const Constant *V1,
                                          const Constant *V2);
  Constant *ConstantFoldCompareInstruction(LLVMContext &Context,
                                           unsigned short predicate, 
                                           const Constant *C1, 
                                           const Constant *C2);
  Constant *ConstantFoldGetElementPtr(LLVMContext &Context, const Constant *C,
                                      Constant* const *Idxs, unsigned NumIdx);
} // End llvm namespace

#endif
