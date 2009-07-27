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

static std::vector<Constant*> getValType(ConstantArray *CA) {
  std::vector<Constant*> Elements;
  Elements.reserve(CA->getNumOperands());
  for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
    Elements.push_back(cast<Constant>(CA->getOperand(i)));
  return Elements;
}

static std::vector<Constant*> getValType(ConstantStruct *CS) {
  std::vector<Constant*> Elements;
  Elements.reserve(CS->getNumOperands());
  for (unsigned i = 0, e = CS->getNumOperands(); i != e; ++i)
    Elements.push_back(cast<Constant>(CS->getOperand(i)));
  return Elements;
}

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

Constant *LLVMContextImpl::getConstantArray(const ArrayType *Ty,
                             const std::vector<Constant*> &V) {
  // If this is an all-zero array, return a ConstantAggregateZero object
  if (!V.empty()) {
    Constant *C = V[0];
    if (!C->isNullValue()) {
      // Implicitly locked.
      return ArrayConstants.getOrCreate(Ty, V);
    }
    for (unsigned i = 1, e = V.size(); i != e; ++i)
      if (V[i] != C) {
        // Implicitly locked.
        return ArrayConstants.getOrCreate(Ty, V);
      }
  }
  
  return Context.getConstantAggregateZero(Ty);
}

Constant *LLVMContextImpl::getConstantStruct(const StructType *Ty,
                              const std::vector<Constant*> &V) {
  // Create a ConstantAggregateZero value if all elements are zeros...
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    if (!V[i]->isNullValue())
      // Implicitly locked.
      return StructConstants.getOrCreate(Ty, V);

  return Context.getConstantAggregateZero(Ty);
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

void LLVMContextImpl::erase(ConstantArray *C) {
  ArrayConstants.remove(C);
}

void LLVMContextImpl::erase(ConstantStruct *S) {
  StructConstants.remove(S);
}

void LLVMContextImpl::erase(ConstantVector *V) {
  VectorConstants.remove(V);
}

// *** RAUW helpers ***

Constant *LLVMContextImpl::replaceUsesOfWithOnConstant(ConstantArray *CA,
                                               Value *From, Value *To, Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  std::pair<ArrayConstantsTy::MapKey, Constant*> Lookup;
  Lookup.first.first = CA->getType();
  Lookup.second = CA;

  std::vector<Constant*> &Values = Lookup.first.second;
  Values.reserve(CA->getNumOperands());  // Build replacement array.

  // Fill values with the modified operands of the constant array.  Also, 
  // compute whether this turns into an all-zeros array.
  bool isAllZeros = false;
  unsigned NumUpdated = 0;
  if (!ToC->isNullValue()) {
    for (Use *O = CA->OperandList, *E = CA->OperandList + CA->getNumOperands();
         O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      if (Val == From) {
        Val = ToC;
        ++NumUpdated;
      }
      Values.push_back(Val);
    }
  } else {
    isAllZeros = true;
    for (Use *O = CA->OperandList, *E = CA->OperandList + CA->getNumOperands();
         O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      if (Val == From) {
        Val = ToC;
        ++NumUpdated;
      }
      Values.push_back(Val);
      if (isAllZeros) isAllZeros = Val->isNullValue();
    }
  }
  
  Constant *Replacement = 0;
  if (isAllZeros) {
    Replacement = Context.getConstantAggregateZero(CA->getType());
  } else {
    // Check to see if we have this array type already.
    sys::SmartScopedWriter<true> Writer(ConstantsLock);
    bool Exists;
    ArrayConstantsTy::MapTy::iterator I =
      ArrayConstants.InsertOrGetItem(Lookup, Exists);
    
    if (Exists) {
      Replacement = I->second;
    } else {
      // Okay, the new shape doesn't exist in the system yet.  Instead of
      // creating a new constant array, inserting it, replaceallusesof'ing the
      // old with the new, then deleting the old... just update the current one
      // in place!
      ArrayConstants.MoveConstantToNewSlot(CA, I);
      
      // Update to the new value.  Optimize for the case when we have a single
      // operand that we're changing, but handle bulk updates efficiently.
      if (NumUpdated == 1) {
        unsigned OperandToUpdate = U - CA->OperandList;
        assert(CA->getOperand(OperandToUpdate) == From &&
               "ReplaceAllUsesWith broken!");
        CA->setOperand(OperandToUpdate, ToC);
      } else {
        for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
          if (CA->getOperand(i) == From)
            CA->setOperand(i, ToC);
      }
      return 0;
    }
  }
  
  return Replacement;
}

Constant *LLVMContextImpl::replaceUsesOfWithOnConstant(ConstantStruct *CS,
                                               Value *From, Value *To, Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  unsigned OperandToUpdate = U - CS->OperandList;
  assert(CS->getOperand(OperandToUpdate) == From &&
         "ReplaceAllUsesWith broken!");

  std::pair<StructConstantsTy::MapKey, Constant*> Lookup;
  Lookup.first.first = CS->getType();
  Lookup.second = CS;
  std::vector<Constant*> &Values = Lookup.first.second;
  Values.reserve(CS->getNumOperands());  // Build replacement struct.
  
  
  // Fill values with the modified operands of the constant struct.  Also, 
  // compute whether this turns into an all-zeros struct.
  bool isAllZeros = false;
  if (!ToC->isNullValue()) {
    for (Use *O = CS->OperandList, *E = CS->OperandList + CS->getNumOperands(); 
         O != E; ++O)
      Values.push_back(cast<Constant>(O->get()));
  } else {
    isAllZeros = true;
    for (Use *O = CS->OperandList, *E = CS->OperandList + CS->getNumOperands(); 
         O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      Values.push_back(Val);
      if (isAllZeros) isAllZeros = Val->isNullValue();
    }
  }
  Values[OperandToUpdate] = ToC;
  
  Constant *Replacement = 0;
  if (isAllZeros) {
    Replacement = Context.getConstantAggregateZero(CS->getType());
  } else {
    // Check to see if we have this array type already.
    sys::SmartScopedWriter<true> Writer(ConstantsLock);
    bool Exists;
    StructConstantsTy::MapTy::iterator I =
      StructConstants.InsertOrGetItem(Lookup, Exists);
    
    if (Exists) {
      Replacement = I->second;
    } else {
      // Okay, the new shape doesn't exist in the system yet.  Instead of
      // creating a new constant struct, inserting it, replaceallusesof'ing the
      // old with the new, then deleting the old... just update the current one
      // in place!
      StructConstants.MoveConstantToNewSlot(CS, I);
      
      // Update to the new value.
      CS->setOperand(OperandToUpdate, ToC);
      return 0;
    }
  }
  
  assert(Replacement != CS && "I didn't contain From!");
  
  return Replacement;
}
