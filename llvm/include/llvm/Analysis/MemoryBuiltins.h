//===- llvm/Analysis/MemoryBuiltins.h- Calls to memory builtins -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions identifies calls to builtin functions that allocate
// or free memory.  
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORYBUILTINS_H
#define LLVM_ANALYSIS_MEMORYBUILTINS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Operator.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/TargetFolder.h"

namespace llvm {
class CallInst;
class PointerType;
class TargetData;
class Type;
class Value;


/// \brief Tests if a value is a call or invoke to a library function that
/// allocates or reallocates memory (either malloc, calloc, realloc, or strdup
/// like).
bool isAllocationFn(const Value *V, bool LookThroughBitCast = false);

/// \brief Tests if a value is a call or invoke to a function that returns a
/// NoAlias pointer (including malloc/calloc/strdup-like functions).
bool isNoAliasFn(const Value *V, bool LookThroughBitCast = false);

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates uninitialized memory (such as malloc).
bool isMallocLikeFn(const Value *V, bool LookThroughBitCast = false);

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates zero-filled memory (such as calloc).
bool isCallocLikeFn(const Value *V, bool LookThroughBitCast = false);

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates memory (either malloc, calloc, or strdup like).
bool isAllocLikeFn(const Value *V, bool LookThroughBitCast = false);

/// \brief Tests if a value is a call or invoke to a library function that
/// reallocates memory (such as realloc).
bool isReallocLikeFn(const Value *V, bool LookThroughBitCast = false);


//===----------------------------------------------------------------------===//
//  malloc Call Utility Functions.
//

/// extractMallocCall - Returns the corresponding CallInst if the instruction
/// is a malloc call.  Since CallInst::CreateMalloc() only creates calls, we
/// ignore InvokeInst here.
const CallInst *extractMallocCall(const Value *I);
static inline CallInst *extractMallocCall(Value *I) {
  return const_cast<CallInst*>(extractMallocCall((const Value*)I));
}

/// isArrayMalloc - Returns the corresponding CallInst if the instruction 
/// is a call to malloc whose array size can be determined and the array size
/// is not constant 1.  Otherwise, return NULL.
const CallInst *isArrayMalloc(const Value *I, const TargetData *TD);

/// getMallocType - Returns the PointerType resulting from the malloc call.
/// The PointerType depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the malloc calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
PointerType *getMallocType(const CallInst *CI);

/// getMallocAllocatedType - Returns the Type allocated by malloc call.
/// The Type depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the malloc calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
Type *getMallocAllocatedType(const CallInst *CI);

/// getMallocArraySize - Returns the array size of a malloc call.  If the 
/// argument passed to malloc is a multiple of the size of the malloced type,
/// then return that multiple.  For non-array mallocs, the multiple is
/// constant 1.  Otherwise, return NULL for mallocs whose array size cannot be
/// determined.
Value *getMallocArraySize(CallInst *CI, const TargetData *TD,
                          bool LookThroughSExt = false);


//===----------------------------------------------------------------------===//
//  calloc Call Utility Functions.
//

/// extractCallocCall - Returns the corresponding CallInst if the instruction
/// is a calloc call.
const CallInst *extractCallocCall(const Value *I);
static inline CallInst *extractCallocCall(Value *I) {
  return const_cast<CallInst*>(extractCallocCall((const Value*)I));
}


//===----------------------------------------------------------------------===//
//  free Call Utility Functions.
//

/// isFreeCall - Returns non-null if the value is a call to the builtin free()
const CallInst *isFreeCall(const Value *I);
  
static inline CallInst *isFreeCall(Value *I) {
  return const_cast<CallInst*>(isFreeCall((const Value*)I));
}

  
//===----------------------------------------------------------------------===//
//  Utility functions to compute size of objects.
//

/// \brief Compute the size of the object pointed by Ptr. Returns true and the
/// object size in Size if successful, and false otherwise.
/// If RoundToAlign is true, then Size is rounded up to the aligment of allocas,
/// byval arguments, and global variables.
bool getObjectSize(const Value *Ptr, uint64_t &Size, const TargetData *TD,
                   bool RoundToAlign = false);



typedef std::pair<APInt, APInt> SizeOffsetType;

/// \brief Evaluate the size and offset of an object ponted by a Value*
/// statically. Fails if size or offset are not known at compile time.
class ObjectSizeOffsetVisitor
  : public InstVisitor<ObjectSizeOffsetVisitor, SizeOffsetType> {

  const TargetData *TD;
  bool RoundToAlign;
  unsigned IntTyBits;
  APInt Zero;

  APInt align(APInt Size, uint64_t Align);

  SizeOffsetType unknown() {
    return std::make_pair(APInt(), APInt());
  }

public:
  ObjectSizeOffsetVisitor(const TargetData *TD, LLVMContext &Context,
                          bool RoundToAlign = false);

  SizeOffsetType compute(Value *V);

  bool knownSize(SizeOffsetType &SizeOffset) {
    return SizeOffset.first.getBitWidth() > 1;
  }

  bool knownOffset(SizeOffsetType &SizeOffset) {
    return SizeOffset.second.getBitWidth() > 1;
  }

  bool bothKnown(SizeOffsetType &SizeOffset) {
    return knownSize(SizeOffset) && knownOffset(SizeOffset);
  }

  SizeOffsetType visitAllocaInst(AllocaInst &I);
  SizeOffsetType visitArgument(Argument &A);
  SizeOffsetType visitCallSite(CallSite CS);
  SizeOffsetType visitConstantPointerNull(ConstantPointerNull&);
  SizeOffsetType visitExtractValueInst(ExtractValueInst &I);
  SizeOffsetType visitGEPOperator(GEPOperator &GEP);
  SizeOffsetType visitGlobalVariable(GlobalVariable &GV);
  SizeOffsetType visitIntToPtrInst(IntToPtrInst&);
  SizeOffsetType visitLoadInst(LoadInst &I);
  SizeOffsetType visitPHINode(PHINode&);
  SizeOffsetType visitSelectInst(SelectInst &I);
  SizeOffsetType visitUndefValue(UndefValue&);
  SizeOffsetType visitInstruction(Instruction &I);
};

typedef std::pair<Value*, Value*> SizeOffsetEvalType;


/// \brief Evaluate the size and offset of an object ponted by a Value*.
/// May create code to compute the result at run-time.
class ObjectSizeOffsetEvaluator
  : public InstVisitor<ObjectSizeOffsetEvaluator, SizeOffsetEvalType> {

  typedef IRBuilder<true, TargetFolder> BuilderTy;
  typedef DenseMap<const Value*, SizeOffsetEvalType> CacheMapTy;
  typedef SmallPtrSet<const Value*, 8> PtrSetTy;

  const TargetData *TD;
  LLVMContext &Context;
  BuilderTy Builder;
  ObjectSizeOffsetVisitor Visitor;
  IntegerType *IntTy;
  Value *Zero;
  CacheMapTy CacheMap;
  PtrSetTy SeenVals;

  SizeOffsetEvalType unknown() {
    return std::make_pair((Value*)0, (Value*)0);
  }
  SizeOffsetEvalType compute_(Value *V);

public:
  ObjectSizeOffsetEvaluator(const TargetData *TD, LLVMContext &Context);
  SizeOffsetEvalType compute(Value *V);

  bool knownSize(SizeOffsetEvalType &SizeOffset) {
    return SizeOffset.first;
  }

  bool knownOffset(SizeOffsetEvalType &SizeOffset) {
    return SizeOffset.second;
  }

  bool anyKnown(SizeOffsetEvalType &SizeOffset) {
    return knownSize(SizeOffset) || knownOffset(SizeOffset);
  }

  bool bothKnown(SizeOffsetEvalType &SizeOffset) {
    return knownSize(SizeOffset) && knownOffset(SizeOffset);
  }

  SizeOffsetEvalType visitAllocaInst(AllocaInst &I);
  SizeOffsetEvalType visitCallSite(CallSite CS);
  SizeOffsetEvalType visitGEPOperator(GEPOperator &GEP);
  SizeOffsetEvalType visitIntToPtrInst(IntToPtrInst&);
  SizeOffsetEvalType visitLoadInst(LoadInst &I);
  SizeOffsetEvalType visitPHINode(PHINode &PHI);
  SizeOffsetEvalType visitSelectInst(SelectInst &I);
  SizeOffsetEvalType visitInstruction(Instruction &I);
};

} // End llvm namespace

#endif
