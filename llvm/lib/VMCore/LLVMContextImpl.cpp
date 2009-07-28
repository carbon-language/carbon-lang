//===--------------- LLVMContextImpl.cpp - Implementation ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContextImpl, the opaque implementation 
//  of LLVMContext.
//
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/MDNode.h"
using namespace llvm;

static char getValType(ConstantAggregateZero *CPZ) { return 0; }

static std::vector<Constant*> getValType(ConstantVector *CP) {
  std::vector<Constant*> Elements;
  Elements.reserve(CP->getNumOperands());
  for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
    Elements.push_back(CP->getOperand(i));
  return Elements;
}


LLVMContextImpl::LLVMContextImpl(LLVMContext &C) :
    Context(C), TheTrueVal(0), TheFalseVal(0) { }

MDString *LLVMContextImpl::getMDString(const char *StrBegin,
                                       unsigned StrLength) {
  sys::SmartScopedWriter<true> Writer(ConstantsLock);
  StringMapEntry<MDString *> &Entry = 
    MDStringCache.GetOrCreateValue(StringRef(StrBegin, StrLength));
  MDString *&S = Entry.getValue();
  if (!S) S = new MDString(Entry.getKeyData(),
                           Entry.getKeyLength());

  return S;
}

MDNode *LLVMContextImpl::getMDNode(Value*const* Vals, unsigned NumVals) {
  FoldingSetNodeID ID;
  for (unsigned i = 0; i != NumVals; ++i)
    ID.AddPointer(Vals[i]);

  ConstantsLock.reader_acquire();
  void *InsertPoint;
  MDNode *N = MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
  ConstantsLock.reader_release();
  
  if (!N) {
    sys::SmartScopedWriter<true> Writer(ConstantsLock);
    N = MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
    if (!N) {
      // InsertPoint will have been set by the FindNodeOrInsertPos call.
      N = new MDNode(Vals, NumVals);
      MDNodeSet.InsertNode(N, InsertPoint);
    }
  }

  return N;
}

ConstantAggregateZero*
LLVMContextImpl::getConstantAggregateZero(const Type *Ty) {
  assert((isa<StructType>(Ty) || isa<ArrayType>(Ty) || isa<VectorType>(Ty)) &&
         "Cannot create an aggregate zero of non-aggregate type!");

  // Implicitly locked.
  return AggZeroConstants.getOrCreate(Ty, 0);
}

Constant *LLVMContextImpl::getConstantVector(const VectorType *Ty,
                              const std::vector<Constant*> &V) {
  assert(!V.empty() && "Vectors can't be empty");
  // If this is an all-undef or alll-zero vector, return a
  // ConstantAggregateZero or UndefValue.
  Constant *C = V[0];
  bool isZero = C->isNullValue();
  bool isUndef = isa<UndefValue>(C);

  if (isZero || isUndef) {
    for (unsigned i = 1, e = V.size(); i != e; ++i)
      if (V[i] != C) {
        isZero = isUndef = false;
        break;
      }
  }
  
  if (isZero)
    return Context.getConstantAggregateZero(Ty);
  if (isUndef)
    return Context.getUndef(Ty);
    
  // Implicitly locked.
  return VectorConstants.getOrCreate(Ty, V);
}

// *** erase methods ***

void LLVMContextImpl::erase(MDString *M) {
  sys::SmartScopedWriter<true> Writer(ConstantsLock);
  MDStringCache.erase(MDStringCache.find(M->getString()));
}

void LLVMContextImpl::erase(MDNode *M) {
  sys::SmartScopedWriter<true> Writer(ConstantsLock);
  MDNodeSet.RemoveNode(M);
}

void LLVMContextImpl::erase(ConstantAggregateZero *Z) {
  AggZeroConstants.remove(Z);
}

void LLVMContextImpl::erase(ConstantVector *V) {
  VectorConstants.remove(V);
}
