//===- NameMangling.cpp - Name Mangling for LLVM ----------------------------=//
//
// This file implements a consistent scheme for name mangling symbols.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/NameMangling.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
using std::string;

// MangleTypeName - Implement a consistent name-mangling scheme for
//                  a given type.
// 
string MangleTypeName(const Type *Ty) {
  string mangledName;
  if (Ty->isPrimitiveType()) {
    const string &longName = Ty->getDescription();
    return string(longName.c_str(), (longName.length() < 2) ? 1 : 2);
  } else if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
    mangledName = string("P_" + MangleTypeName(PTy->getElementType()));
  } else if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    mangledName = string("S_");
    for (unsigned i=0; i < STy->getNumContainedTypes(); ++i)
      mangledName += MangleTypeName(STy->getContainedType(i));
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    mangledName = string("A_" +MangleTypeName(ATy->getElementType()));
  } else if (const FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
    mangledName = string("M_") + MangleTypeName(FTy->getReturnType());
    for (unsigned i = 1; i < FTy->getNumContainedTypes(); ++i)
      mangledName += string(MangleTypeName(FTy->getContainedType(i)));
  }
  
  return mangledName;
}

// mangleName - implement a consistent name-mangling scheme for all
// externally visible (i.e., global) objects.
// privateName should be unique within the module.
// 
string MangleName(const string &privateName, const Value *V) {
  // Lets drop the P_ before every global name since all globals are ptrs
  return privateName + "_" +
    MangleTypeName(isa<GlobalValue>(V)
                   ? cast<GlobalValue>(V)->getType()->getElementType()
                   : V->getType());
}
