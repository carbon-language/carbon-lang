//===-- InternalGlobalMapper.cpp - Mapping Info for Internal Globals ------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// InternalGlobalMapper is a pass that helps the runtime trace optimizer map
// the names of internal GlobalValues (which may have mangled,
// unreconstructible names in the executable) to pointers. If the name mangler
// is changed at some point in the future to allow its results to be
// reconstructible (for instance, by making the type mangling symbolic instead
// of using a UniqueID) this pass should probably be phased out.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
using namespace llvm;

namespace llvm {

typedef std::vector<Constant *> GVVectorTy;

class InternalGlobalMapper : public Pass {
public:
  bool run (Module &M);
};

Pass *llvm::createInternalGlobalMapperPass () {
  return new InternalGlobalMapper ();
}

static void maybeAddInternalValueToVector (GVVectorTy &Vector, GlobalValue &GV){
  // If it's a GlobalValue with internal linkage and a name (i.e. it's going to
  // be mangled), then put the GV, casted to sbyte*, in the vector. Otherwise
  // add a null.
  if (GV.hasInternalLinkage () && GV.hasName ())
    Vector.push_back (ConstantExpr::getCast
      (ConstantPointerRef::get (&GV), PointerType::get (Type::SByteTy)));
  else
    Vector.push_back (ConstantPointerNull::get (PointerType::get
                                                (Type::SByteTy)));
}

bool InternalGlobalMapper::run (Module &M) {
  GVVectorTy gvvector;

  // Populate the vector with internal global values and their names.
  for (Module::giterator i = M.gbegin (), e = M.gend (); i != e; ++i)
    maybeAddInternalValueToVector (gvvector, *i);
  // Add an extra global for _llvm_internalGlobals itself (null,
  // because it's not internal)
  gvvector.push_back (ConstantPointerNull::get
    (PointerType::get (Type::SByteTy)));
  for (Module::iterator i = M.begin (), e = M.end (); i != e; ++i)
    maybeAddInternalValueToVector (gvvector, *i);

  // Convert the vector to a constant struct of type {Size, [Size x sbyte*]}.
  ArrayType *ATy = ArrayType::get (PointerType::get (Type::SByteTy),
                                  gvvector.size ());
  std::vector<const Type *> FieldTypes;
  FieldTypes.push_back (Type::UIntTy);
  FieldTypes.push_back (ATy);
  StructType *STy = StructType::get (FieldTypes);
  std::vector<Constant *> FieldValues;
  FieldValues.push_back (ConstantUInt::get (Type::UIntTy, gvvector.size ()));
  FieldValues.push_back (ConstantArray::get (ATy, gvvector));
  
  // Add the constant struct to M as an external global symbol named
  // "_llvm_internalGlobals".
  new GlobalVariable (STy, true, GlobalValue::ExternalLinkage,
                      ConstantStruct::get (STy, FieldValues),
                      "_llvm_internalGlobals", &M);

  return true; // Module was modified.
}

} // end namespace llvm
