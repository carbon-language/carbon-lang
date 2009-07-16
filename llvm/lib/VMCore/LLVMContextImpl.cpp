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
using namespace llvm;

// Get a ConstantInt from an APInt. Note that the value stored in the DenseMap 
// as the key, is a DenseMapAPIntKeyInfo::KeyTy which has provided the
// operator== and operator!= to ensure that the DenseMap doesn't attempt to
// compare APInt's of different widths, which would violate an APInt class
// invariant which generates an assertion.
ConstantInt *LLVMContextImpl::getConstantInt(const APInt& V) {
  // Get the corresponding integer type for the bit width of the value.
  const IntegerType *ITy = Context.getIntegerType(V.getBitWidth());
  // get an existing value or the insertion position
  DenseMapAPIntKeyInfo::KeyTy Key(V, ITy);
  
  ConstantsLock.reader_acquire();
  ConstantInt *&Slot = IntConstants[Key]; 
  ConstantsLock.reader_release();
    
  if (!Slot) {
    sys::SmartScopedWriter<true> Writer(ConstantsLock);
    ConstantInt *&NewSlot = IntConstants[Key]; 
    if (!Slot) {
      NewSlot = new ConstantInt(ITy, V);
    }
    
    return NewSlot;
  } else {
    return Slot;
  }
}

ConstantFP *LLVMContextImpl::getConstantFP(const APFloat &V) {
  DenseMapAPFloatKeyInfo::KeyTy Key(V);
  
  ConstantsLock.reader_acquire();
  ConstantFP *&Slot = FPConstants[Key];
  ConstantsLock.reader_release();
    
  if (!Slot) {
    sys::SmartScopedWriter<true> Writer(ConstantsLock);
    ConstantFP *&NewSlot = FPConstants[Key];
    if (!NewSlot) {
      const Type *Ty;
      if (&V.getSemantics() == &APFloat::IEEEsingle)
        Ty = Type::FloatTy;
      else if (&V.getSemantics() == &APFloat::IEEEdouble)
        Ty = Type::DoubleTy;
      else if (&V.getSemantics() == &APFloat::x87DoubleExtended)
        Ty = Type::X86_FP80Ty;
      else if (&V.getSemantics() == &APFloat::IEEEquad)
        Ty = Type::FP128Ty;
      else {
        assert(&V.getSemantics() == &APFloat::PPCDoubleDouble && 
               "Unknown FP format");
        Ty = Type::PPC_FP128Ty;
      }
      NewSlot = new ConstantFP(Ty, V);
    }
    
    return NewSlot;
  }
  
  return Slot;
}

MDString *LLVMContextImpl::getMDString(const char *StrBegin,
                                       const char *StrEnd) {
  sys::SmartScopedWriter<true> Writer(ConstantsLock);
  StringMapEntry<MDString *> &Entry = MDStringCache.GetOrCreateValue(
                                        StrBegin, StrEnd);
  MDString *&S = Entry.getValue();
  if (!S) S = new MDString(Entry.getKeyData(),
                           Entry.getKeyData() + Entry.getKeyLength());

  return S;
}

// *** erase methods ***

void LLVMContextImpl::erase(MDString *M) {
  sys::SmartScopedWriter<true> Writer(ConstantsLock);
  MDStringCache.erase(MDStringCache.find(M->StrBegin, M->StrEnd));
}
