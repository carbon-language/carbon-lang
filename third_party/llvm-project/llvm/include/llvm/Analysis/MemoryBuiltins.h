//==- llvm/Analysis/MemoryBuiltins.h - Calls to memory builtins --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This family of functions identifies calls to builtin functions that allocate
// or free memory.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORYBUILTINS_H
#define LLVM_ANALYSIS_MEMORYBUILTINS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/ValueHandle.h"
#include <cstdint>
#include <utility>

namespace llvm {

class AllocaInst;
class Argument;
class CallInst;
class ConstantInt;
class ConstantPointerNull;
class DataLayout;
class ExtractElementInst;
class ExtractValueInst;
class GEPOperator;
class GlobalAlias;
class GlobalVariable;
class Instruction;
class IntegerType;
class IntrinsicInst;
class IntToPtrInst;
class LLVMContext;
class LoadInst;
class PHINode;
class PointerType;
class SelectInst;
class Type;
class UndefValue;
class Value;

/// Tests if a value is a call or invoke to a library function that
/// allocates or reallocates memory (either malloc, calloc, realloc, or strdup
/// like).
bool isAllocationFn(const Value *V, const TargetLibraryInfo *TLI,
                    bool LookThroughBitCast = false);
bool isAllocationFn(const Value *V,
                    function_ref<const TargetLibraryInfo &(Function &)> GetTLI,
                    bool LookThroughBitCast = false);

/// Tests if a value is a call or invoke to a function that returns a
/// NoAlias pointer (including malloc/calloc/realloc/strdup-like functions).
bool isNoAliasFn(const Value *V, const TargetLibraryInfo *TLI,
                 bool LookThroughBitCast = false);

/// Tests if a value is a call or invoke to a library function that
/// allocates uninitialized memory (such as malloc).
bool isMallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                    bool LookThroughBitCast = false);
bool isMallocLikeFn(const Value *V,
                    function_ref<const TargetLibraryInfo &(Function &)> GetTLI,
                    bool LookThroughBitCast = false);

/// Tests if a value is a call or invoke to a library function that
/// allocates uninitialized memory with alignment (such as aligned_alloc).
bool isAlignedAllocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                          bool LookThroughBitCast = false);
bool isAlignedAllocLikeFn(
    const Value *V, function_ref<const TargetLibraryInfo &(Function &)> GetTLI,
    bool LookThroughBitCast = false);

/// Tests if a value is a call or invoke to a library function that
/// allocates zero-filled memory (such as calloc).
bool isCallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                    bool LookThroughBitCast = false);

/// Tests if a value is a call or invoke to a library function that
/// allocates memory similar to malloc or calloc.
bool isMallocOrCallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                            bool LookThroughBitCast = false);

/// Tests if a value is a call or invoke to a library function that
/// allocates memory (either malloc, calloc, or strdup like).
bool isAllocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                   bool LookThroughBitCast = false);

/// Tests if a value is a call or invoke to a library function that
/// reallocates memory (e.g., realloc).
bool isReallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                     bool LookThroughBitCast = false);

/// Tests if a function is a call or invoke to a library function that
/// reallocates memory (e.g., realloc).
bool isReallocLikeFn(const Function *F, const TargetLibraryInfo *TLI);

/// Tests if a value is a call or invoke to a library function that
/// allocates memory and throws if an allocation failed (e.g., new).
bool isOpNewLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                     bool LookThroughBitCast = false);

/// Tests if a value is a call or invoke to a library function that
/// allocates memory (strdup, strndup).
bool isStrdupLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                     bool LookThroughBitCast = false);

//===----------------------------------------------------------------------===//
//  malloc Call Utility Functions.
//

/// extractMallocCall - Returns the corresponding CallInst if the instruction
/// is a malloc call.  Since CallInst::CreateMalloc() only creates calls, we
/// ignore InvokeInst here.
const CallInst *
extractMallocCall(const Value *I,
                  function_ref<const TargetLibraryInfo &(Function &)> GetTLI);
inline CallInst *
extractMallocCall(Value *I,
                  function_ref<const TargetLibraryInfo &(Function &)> GetTLI) {
  return const_cast<CallInst *>(extractMallocCall((const Value *)I, GetTLI));
}

/// getMallocType - Returns the PointerType resulting from the malloc call.
/// The PointerType depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the malloc calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
PointerType *getMallocType(const CallInst *CI, const TargetLibraryInfo *TLI);

/// getMallocAllocatedType - Returns the Type allocated by malloc call.
/// The Type depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the malloc calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
Type *getMallocAllocatedType(const CallInst *CI, const TargetLibraryInfo *TLI);

/// getMallocArraySize - Returns the array size of a malloc call.  If the
/// argument passed to malloc is a multiple of the size of the malloced type,
/// then return that multiple.  For non-array mallocs, the multiple is
/// constant 1.  Otherwise, return NULL for mallocs whose array size cannot be
/// determined.
Value *getMallocArraySize(CallInst *CI, const DataLayout &DL,
                          const TargetLibraryInfo *TLI,
                          bool LookThroughSExt = false);

//===----------------------------------------------------------------------===//
//  calloc Call Utility Functions.
//

/// extractCallocCall - Returns the corresponding CallInst if the instruction
/// is a calloc call.
const CallInst *extractCallocCall(const Value *I, const TargetLibraryInfo *TLI);
inline CallInst *extractCallocCall(Value *I, const TargetLibraryInfo *TLI) {
  return const_cast<CallInst*>(extractCallocCall((const Value*)I, TLI));
}


//===----------------------------------------------------------------------===//
//  free Call Utility Functions.
//

/// isLibFreeFunction - Returns true if the function is a builtin free()
bool isLibFreeFunction(const Function *F, const LibFunc TLIFn);

/// isFreeCall - Returns non-null if the value is a call to the builtin free()
const CallInst *isFreeCall(const Value *I, const TargetLibraryInfo *TLI);

inline CallInst *isFreeCall(Value *I, const TargetLibraryInfo *TLI) {
  return const_cast<CallInst*>(isFreeCall((const Value*)I, TLI));
}

//===----------------------------------------------------------------------===//
//  Utility functions to compute size of objects.
//

/// Various options to control the behavior of getObjectSize.
struct ObjectSizeOpts {
  /// Controls how we handle conditional statements with unknown conditions.
  enum class Mode : uint8_t {
    /// Fail to evaluate an unknown condition.
    Exact,
    /// Evaluate all branches of an unknown condition. If all evaluations
    /// succeed, pick the minimum size.
    Min,
    /// Same as Min, except we pick the maximum size of all of the branches.
    Max
  };

  /// How we want to evaluate this object's size.
  Mode EvalMode = Mode::Exact;
  /// Whether to round the result up to the alignment of allocas, byval
  /// arguments, and global variables.
  bool RoundToAlign = false;
  /// If this is true, null pointers in address space 0 will be treated as
  /// though they can't be evaluated. Otherwise, null is always considered to
  /// point to a 0 byte region of memory.
  bool NullIsUnknownSize = false;
};

/// Compute the size of the object pointed by Ptr. Returns true and the
/// object size in Size if successful, and false otherwise. In this context, by
/// object we mean the region of memory starting at Ptr to the end of the
/// underlying object pointed to by Ptr.
///
/// WARNING: The object size returned is the allocation size.  This does not
/// imply dereferenceability at site of use since the object may be freeed in
/// between.
bool getObjectSize(const Value *Ptr, uint64_t &Size, const DataLayout &DL,
                   const TargetLibraryInfo *TLI, ObjectSizeOpts Opts = {});

/// Try to turn a call to \@llvm.objectsize into an integer value of the given
/// Type. Returns null on failure. If MustSucceed is true, this function will
/// not return null, and may return conservative values governed by the second
/// argument of the call to objectsize.
Value *lowerObjectSizeCall(IntrinsicInst *ObjectSize, const DataLayout &DL,
                           const TargetLibraryInfo *TLI, bool MustSucceed);



using SizeOffsetType = std::pair<APInt, APInt>;

/// Evaluate the size and offset of an object pointed to by a Value*
/// statically. Fails if size or offset are not known at compile time.
class ObjectSizeOffsetVisitor
  : public InstVisitor<ObjectSizeOffsetVisitor, SizeOffsetType> {
  const DataLayout &DL;
  const TargetLibraryInfo *TLI;
  ObjectSizeOpts Options;
  unsigned IntTyBits;
  APInt Zero;
  SmallPtrSet<Instruction *, 8> SeenInsts;

  APInt align(APInt Size, uint64_t Align);

  SizeOffsetType unknown() {
    return std::make_pair(APInt(), APInt());
  }

public:
  ObjectSizeOffsetVisitor(const DataLayout &DL, const TargetLibraryInfo *TLI,
                          LLVMContext &Context, ObjectSizeOpts Options = {});

  SizeOffsetType compute(Value *V);

  static bool knownSize(const SizeOffsetType &SizeOffset) {
    return SizeOffset.first.getBitWidth() > 1;
  }

  static bool knownOffset(const SizeOffsetType &SizeOffset) {
    return SizeOffset.second.getBitWidth() > 1;
  }

  static bool bothKnown(const SizeOffsetType &SizeOffset) {
    return knownSize(SizeOffset) && knownOffset(SizeOffset);
  }

  // These are "private", except they can't actually be made private. Only
  // compute() should be used by external users.
  SizeOffsetType visitAllocaInst(AllocaInst &I);
  SizeOffsetType visitArgument(Argument &A);
  SizeOffsetType visitCallBase(CallBase &CB);
  SizeOffsetType visitConstantPointerNull(ConstantPointerNull&);
  SizeOffsetType visitExtractElementInst(ExtractElementInst &I);
  SizeOffsetType visitExtractValueInst(ExtractValueInst &I);
  SizeOffsetType visitGEPOperator(GEPOperator &GEP);
  SizeOffsetType visitGlobalAlias(GlobalAlias &GA);
  SizeOffsetType visitGlobalVariable(GlobalVariable &GV);
  SizeOffsetType visitIntToPtrInst(IntToPtrInst&);
  SizeOffsetType visitLoadInst(LoadInst &I);
  SizeOffsetType visitPHINode(PHINode&);
  SizeOffsetType visitSelectInst(SelectInst &I);
  SizeOffsetType visitUndefValue(UndefValue&);
  SizeOffsetType visitInstruction(Instruction &I);

private:
  bool CheckedZextOrTrunc(APInt &I);
};

using SizeOffsetEvalType = std::pair<Value *, Value *>;

/// Evaluate the size and offset of an object pointed to by a Value*.
/// May create code to compute the result at run-time.
class ObjectSizeOffsetEvaluator
  : public InstVisitor<ObjectSizeOffsetEvaluator, SizeOffsetEvalType> {
  using BuilderTy = IRBuilder<TargetFolder, IRBuilderCallbackInserter>;
  using WeakEvalType = std::pair<WeakTrackingVH, WeakTrackingVH>;
  using CacheMapTy = DenseMap<const Value *, WeakEvalType>;
  using PtrSetTy = SmallPtrSet<const Value *, 8>;

  const DataLayout &DL;
  const TargetLibraryInfo *TLI;
  LLVMContext &Context;
  BuilderTy Builder;
  IntegerType *IntTy;
  Value *Zero;
  CacheMapTy CacheMap;
  PtrSetTy SeenVals;
  ObjectSizeOpts EvalOpts;
  SmallPtrSet<Instruction *, 8> InsertedInstructions;

  SizeOffsetEvalType compute_(Value *V);

public:
  static SizeOffsetEvalType unknown() {
    return std::make_pair(nullptr, nullptr);
  }

  ObjectSizeOffsetEvaluator(const DataLayout &DL, const TargetLibraryInfo *TLI,
                            LLVMContext &Context, ObjectSizeOpts EvalOpts = {});

  SizeOffsetEvalType compute(Value *V);

  bool knownSize(SizeOffsetEvalType SizeOffset) {
    return SizeOffset.first;
  }

  bool knownOffset(SizeOffsetEvalType SizeOffset) {
    return SizeOffset.second;
  }

  bool anyKnown(SizeOffsetEvalType SizeOffset) {
    return knownSize(SizeOffset) || knownOffset(SizeOffset);
  }

  bool bothKnown(SizeOffsetEvalType SizeOffset) {
    return knownSize(SizeOffset) && knownOffset(SizeOffset);
  }

  // The individual instruction visitors should be treated as private.
  SizeOffsetEvalType visitAllocaInst(AllocaInst &I);
  SizeOffsetEvalType visitCallBase(CallBase &CB);
  SizeOffsetEvalType visitExtractElementInst(ExtractElementInst &I);
  SizeOffsetEvalType visitExtractValueInst(ExtractValueInst &I);
  SizeOffsetEvalType visitGEPOperator(GEPOperator &GEP);
  SizeOffsetEvalType visitIntToPtrInst(IntToPtrInst&);
  SizeOffsetEvalType visitLoadInst(LoadInst &I);
  SizeOffsetEvalType visitPHINode(PHINode &PHI);
  SizeOffsetEvalType visitSelectInst(SelectInst &I);
  SizeOffsetEvalType visitInstruction(Instruction &I);
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_MEMORYBUILTINS_H
