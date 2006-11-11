//===--- SemaType.cpp - Semantic Analysis for Types -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements type-related semantic analysis.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
using namespace llvm;
using namespace clang;

namespace {
  /// BuiltinType - This class is used for builtin types like 'int'.  Builtin
  /// types are always canonical and have a literal name field.
  class BuiltinType : public Type {
    const char *Name;
  public:
    BuiltinType(const char *name) : Name(name) {}
    
    virtual void dump() const;
  };
}

// FIXME: REMOVE
#include <iostream>

void BuiltinType::dump() const {
  std::cerr << Name;
}


void Sema::InitializeBuiltinTypes() {
  assert(Context.VoidTy.isNull() && "Context reinitialized?");
  
  // C99 6.2.5p19.
  Context.VoidTy = new BuiltinType("void");
  
  // C99 6.2.5p2.
  Context.BoolTy = new BuiltinType("_Bool");
  // C99 6.2.5p3.
  Context.CharTy = new BuiltinType("char");
  // C99 6.2.5p4.
  Context.SignedCharTy = new BuiltinType("signed char");
  Context.ShortTy = new BuiltinType("short");
  Context.IntTy = new BuiltinType("int");
  Context.LongTy = new BuiltinType("long");
  Context.LongLongTy = new BuiltinType("long long");
  
  // C99 6.2.5p6.
  Context.UnsignedCharTy = new BuiltinType("unsigned char");
  Context.UnsignedShortTy = new BuiltinType("unsigned short");
  Context.UnsignedIntTy = new BuiltinType("unsigned int");
  Context.UnsignedLongTy = new BuiltinType("unsigned long");
  Context.UnsignedLongLongTy = new BuiltinType("unsigned long long");
  
  // C99 6.2.5p10.
  Context.FloatTy = new BuiltinType("float");
  Context.DoubleTy = new BuiltinType("double");
  Context.LongDoubleTy = new BuiltinType("long double");
  
  // C99 6.2.5p11.
  Context.FloatComplexTy = new BuiltinType("float _Complex");
  Context.DoubleComplexTy = new BuiltinType("double _Complex");
  Context.LongDoubleComplexTy = new BuiltinType("long double _Complex");
}


TypeRef Sema::GetTypeForDeclarator(Declarator &D, Scope *S) {
  return TypeRef();
}
