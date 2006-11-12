//===--- ASTContext.cpp - Context to hold long-lived AST nodes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the ASTContext interface.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/Lex/Preprocessor.h"
using namespace llvm;
using namespace clang;

ASTContext::ASTContext(Preprocessor &pp)
  : PP(pp), Target(pp.getTargetInfo()) {
  InitBuiltinTypes();
}

ASTContext::~ASTContext() {
  // Deallocate all the types.
  while (!Types.empty()) {
    delete Types.back();
    Types.pop_back();
  }
}

void ASTContext::InitBuiltinType(TypeRef &R, const char *Name) {
  Types.push_back((R = new BuiltinType(Name)).getTypePtr());
}


void ASTContext::InitBuiltinTypes() {
  assert(VoidTy.isNull() && "Context reinitialized?");
  
  // C99 6.2.5p19.
  InitBuiltinType(VoidTy, "void");
  
  // C99 6.2.5p2.
  InitBuiltinType(BoolTy, "_Bool");
  // C99 6.2.5p3.
  InitBuiltinType(CharTy, "char");
  // C99 6.2.5p4.
  InitBuiltinType(SignedCharTy, "signed char");
  InitBuiltinType(ShortTy, "short");
  InitBuiltinType(IntTy, "int");
  InitBuiltinType(LongTy, "long");
  InitBuiltinType(LongLongTy, "long long");
  
  // C99 6.2.5p6.
  InitBuiltinType(UnsignedCharTy, "unsigned char");
  InitBuiltinType(UnsignedShortTy, "unsigned short");
  InitBuiltinType(UnsignedIntTy, "unsigned int");
  InitBuiltinType(UnsignedLongTy, "unsigned long");
  InitBuiltinType(UnsignedLongLongTy, "unsigned long long");
  
  // C99 6.2.5p10.
  InitBuiltinType(FloatTy, "float");
  InitBuiltinType(DoubleTy, "double");
  InitBuiltinType(LongDoubleTy, "long double");
  
  // C99 6.2.5p11.
  InitBuiltinType(FloatComplexTy, "float _Complex");
  InitBuiltinType(DoubleComplexTy, "double _Complex");
  InitBuiltinType(LongDoubleComplexTy, "long double _Complex");
}

/// getPointerType - Return the uniqued reference to the type for a pointer to
/// the specified type.
TypeRef ASTContext::getPointerType(TypeRef T) {
  // FIXME: This is obviously braindead!
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    if (PointerType *PTy = dyn_cast<PointerType>(Types[i]))
      if (PTy->getPointeeType() == T)
        return Types[i];
  
  // If the pointee type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  Type *Canonical = 0;
  if (!T->isCanonical())
    Canonical = getPointerType(T.getCanonicalType()).getTypePtr();
  
  Types.push_back(new PointerType(T, Canonical));
  return Types.back();
}

/// getArrayType - Return the unique reference to the type for an array of the
/// specified element type.
TypeRef ASTContext::getArrayType(TypeRef EltTy,ArrayType::ArraySizeModifier ASM,
                                 unsigned EltTypeQuals, void *NumElts) {
#warning "IGNORING SIZE"
  
  // FIXME: This is obviously braindead!
  // Unique array, to guarantee there is only one array of a particular
  // structure.
  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    if (ArrayType *ATy = dyn_cast<ArrayType>(Types[i]))
      if (ATy->getElementType() == EltTy &&
          ATy->getSizeModifier() == ASM &&
          ATy->getIndexTypeQualifier() == EltTypeQuals)
        return Types[i];
  
  // If the element type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  Type *Canonical = 0;
  if (!EltTy->isCanonical())
    Canonical = getArrayType(EltTy.getCanonicalType(), ASM, EltTypeQuals,
                             NumElts).getTypePtr();
  
  Types.push_back(new ArrayType(EltTy, ASM, EltTypeQuals, Canonical));
  return Types.back();
}


