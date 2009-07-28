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
#include "llvm/Metadata.h"
using namespace llvm;

static char getValType(ConstantAggregateZero *CPZ) { return 0; }

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
