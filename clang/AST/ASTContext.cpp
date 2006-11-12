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
TypeRef ASTContext::getPointerType(const TypeRef &T) {
  // FIXME: This is obviously braindead!
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    if (PointerType *PTy = dyn_cast<PointerType>(Types[i]))
      if (PTy->getPointee() == T)
        return Types[i];
  
  
  Type *Canonical = 0;
  if (!T->isCanonical())
    Canonical = getPointerType(T.getCanonicalType()).getTypePtr();
  
  Types.push_back(new PointerType(T, Canonical));
  return Types.back();
}


