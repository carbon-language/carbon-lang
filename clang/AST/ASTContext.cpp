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
#include "clang/Lex/Preprocessor.h"
using namespace llvm;
using namespace clang;







namespace {
  /// BuiltinType - This class is used for builtin types like 'int'.  Builtin
  /// types are always canonical and have a literal name field.
  class BuiltinType : public Type {
    const char *Name;
  public:
    BuiltinType(const char *name) : Name(name) {}
  };
}

ASTContext::ASTContext(Preprocessor &pp)
  : PP(pp), Target(pp.getTargetInfo()) {

  // C99 6.2.5p19.
  VoidTy = new BuiltinType("void");
    
  // C99 6.2.5p2.
  BoolTy = new BuiltinType("_Bool");
  // C99 6.2.5p3.
  CharTy = new BuiltinType("char");
  // C99 6.2.5p4.
  SignedCharTy = new BuiltinType("signed char");
  ShortTy = new BuiltinType("short");
  IntTy = new BuiltinType("int");
  LongTy = new BuiltinType("long");
  LongLongTy = new BuiltinType("long long");

  // C99 6.2.5p6.
  UnsignedCharTy = new BuiltinType("unsigned char");
  UnsignedShortTy = new BuiltinType("unsigned short");
  UnsignedIntTy = new BuiltinType("unsigned int");
  UnsignedLongTy = new BuiltinType("unsigned long");
  UnsignedLongLongTy = new BuiltinType("unsigned long long");
  
  // C99 6.2.5p10.
  FloatTy = new BuiltinType("float");
  DoubleTy = new BuiltinType("double");
  LongDoubleTy = new BuiltinType("long double");

  // C99 6.2.5p11.
  FloatComplexTy = new BuiltinType("float _Complex");
  DoubleComplexTy = new BuiltinType("double _Complex");
  LongDoubleComplexTy = new BuiltinType("long double _Complex");
  
}


