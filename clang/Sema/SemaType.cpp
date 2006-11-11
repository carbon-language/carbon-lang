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

/// ConvertDeclSpecToType - Convert the specified declspec to the appropriate
/// type object.  This returns null on error.
static TypeRef ConvertDeclSpecToType(const DeclSpec &DS, ASTContext &Ctx) {
  // FIXME: Should move the logic from DeclSpec::Finish to here for validity
  // checking.
  
  switch (DS.TypeSpecType) {
  default: return TypeRef(); // FIXME: Handle unimp cases!
  case DeclSpec::TST_void: return Ctx.VoidTy;
  case DeclSpec::TST_char:
    if (DS.TypeSpecSign == DeclSpec::TSS_unspecified)
      return Ctx.CharTy;
    else if (DS.TypeSpecSign == DeclSpec::TSS_signed)
      return Ctx.SignedCharTy;
    else {
      assert(DS.TypeSpecSign == DeclSpec::TSS_unsigned && "Unknown TSS value");
      return Ctx.UnsignedCharTy;
    }
  case DeclSpec::TST_int:
    if (DS.TypeSpecSign != DeclSpec::TSS_unsigned) {
      switch (DS.TypeSpecWidth) {
      case DeclSpec::TSW_unspecified: return Ctx.IntTy;
      case DeclSpec::TSW_short:       return Ctx.ShortTy;
      case DeclSpec::TSW_long:        return Ctx.LongTy;
      case DeclSpec::TSW_longlong:    return Ctx.LongLongTy;
      }
    } else {
      switch (DS.TypeSpecWidth) {
      case DeclSpec::TSW_unspecified: return Ctx.UnsignedIntTy;
      case DeclSpec::TSW_short:       return Ctx.UnsignedShortTy;
      case DeclSpec::TSW_long:        return Ctx.UnsignedLongTy;
      case DeclSpec::TSW_longlong:    return Ctx.UnsignedLongLongTy;
      }
    }
  case DeclSpec::TST_float:
    if (DS.TypeSpecComplex == DeclSpec::TSC_unspecified)
      return Ctx.FloatTy;
    assert(DS.TypeSpecComplex == DeclSpec::TSC_complex &&
           "FIXME: imaginary types not supported yet!");
    return Ctx.FloatComplexTy;
    
  case DeclSpec::TST_double: {
    bool isLong = DS.TypeSpecWidth == DeclSpec::TSW_long;
    if (DS.TypeSpecComplex == DeclSpec::TSC_unspecified)
      return isLong ? Ctx.LongDoubleTy : Ctx.DoubleTy;
    assert(DS.TypeSpecComplex == DeclSpec::TSC_complex &&
           "FIXME: imaginary types not supported yet!");
    return isLong ? Ctx.LongDoubleComplexTy : Ctx.DoubleComplexTy;
  }
  case DeclSpec::TST_bool:         // _Bool
    return Ctx.BoolTy;
  case DeclSpec::TST_decimal32:    // _Decimal32
  case DeclSpec::TST_decimal64:    // _Decimal64
  case DeclSpec::TST_decimal128:   // _Decimal128
    assert(0 && "FIXME: GNU decimal extensions not supported yet!"); 
    //DeclSpec::TST_enum:
    //DeclSpec::TST_union:
    //DeclSpec::TST_struct:
    //DeclSpec::TST_typedef:
      
  }
}

/// GetTypeForDeclarator - Convert the type for the specified declarator to Type
/// instances.
TypeRef Sema::GetTypeForDeclarator(Declarator &D, Scope *S) {
  TypeRef T = ConvertDeclSpecToType(D.getDeclSpec(), Context);
  
  // FIXME: Apply const/volatile/restrict qualifiers to T.
  
  return T;
  return TypeRef();
}
