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
    Constant *V,         ///< The source constant
    const Type *DestTy   ///< The destination type
  );
  Constant *ConstantFoldSelectInstruction(LLVMContext &Context,
                                          Constant *Cond,
                                          Constant *V1, Constant *V2);
  Constant *ConstantFoldExtractElementInstruction(LLVMContext &Context,
                                                  Constant *Val,
                                                  Constant *Idx);
  Constant *ConstantFoldInsertElementInstruction(LLVMContext &Context,
                                                 Constant *Val,
                                                 Constant *Elt,
                                                 Constant *Idx);
  Constant *ConstantFoldShuffleVectorInstruction(LLVMContext &Context,
                                                 Constant *V1,
                                                 Constant *V2,
                                                 Constant *Mask);
  Constant *ConstantFoldExtractValueInstruction(LLVMContext &Context,
                                                Constant *Agg,
                                                const unsigned *Idxs,
                                                unsigned NumIdx);
  Constant *ConstantFoldInsertValueInstruction(LLVMContext &Context,
                                               Constant *Agg,
                                               Constant *Val,
                                               const unsigned *Idxs,
                                               unsigned NumIdx);
  Constant *ConstantFoldBinaryInstruction(LLVMContext &Context,
                                          unsigned Opcode, Constant *V1,
                                          Constant *V2);
  Constant *ConstantFoldCompareInstruction(LLVMContext &Context,
                                           unsigned short predicate, 
                                           Constant *C1, Constant *C2);
  Constant *ConstantFoldGetElementPtr(LLVMContext &Context, Constant *C,
                                      bool inBounds,
                                      Constant* const *Idxs, unsigned NumIdx);
} // End llvm namespace

#endif
