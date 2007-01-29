//===--- ASTContext.h - Context to hold long-lived AST nodes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTContext interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTCONTEXT_H
#define LLVM_CLANG_AST_ASTCONTEXT_H

#include "clang/AST/Builtins.h"
#include "clang/AST/Type.h"
#include <vector>

namespace llvm {
namespace clang {
  class Preprocessor;
  class TargetInfo;
  
/// ASTContext - This class holds long-lived AST nodes (such as types and
/// decls) that can be referred to throughout the semantic analysis of a file.
class ASTContext {
  std::vector<Type*> Types;
  FoldingSet<PointerType> PointerTypes;
  FoldingSet<ArrayType> ArrayTypes;
  FoldingSet<FunctionTypeNoProto> FunctionTypeNoProtos;
  FoldingSet<FunctionTypeProto> FunctionTypeProtos;
public:
  Preprocessor &PP;
  TargetInfo &Target;
  Builtin::Context BuiltinInfo;

  // Builtin Types.
  TypeRef VoidTy;
  TypeRef BoolTy;
  TypeRef CharTy;
  TypeRef SignedCharTy, ShortTy, IntTy, LongTy, LongLongTy;
  TypeRef UnsignedCharTy, UnsignedShortTy, UnsignedIntTy, UnsignedLongTy;
  TypeRef UnsignedLongLongTy;
  TypeRef FloatTy, DoubleTy, LongDoubleTy;
  TypeRef FloatComplexTy, DoubleComplexTy, LongDoubleComplexTy;
  
  ASTContext(Preprocessor &pp);
  ~ASTContext();
  
  void PrintStats() const;
  
  
  /// getPointerType - Return the uniqued reference to the type for a pointer to
  /// the specified type.
  TypeRef getPointerType(TypeRef T);
  
  /// getArrayType - Return the unique reference to the type for an array of the
  /// specified element type.
  TypeRef getArrayType(TypeRef EltTy, ArrayType::ArraySizeModifier ASM,
                       unsigned EltTypeQuals, void *NumElts);

  /// getFunctionTypeNoProto - Return a K&R style C function type like 'int()'.
  ///
  TypeRef getFunctionTypeNoProto(TypeRef ResultTy);
  
  /// getFunctionType - Return a normal function type with a typed argument
  /// list.  isVariadic indicates whether the argument list includes '...'.
  TypeRef getFunctionType(TypeRef ResultTy, TypeRef *ArgArray,
                          unsigned NumArgs, bool isVariadic);
  
  /// getTypedefType - Return the unique reference to the type for the
  /// specified typename decl.
  TypeRef getTypedefType(TypedefDecl *Decl);

  /// getTagDeclType - Return the unique reference to the type for the
  /// specified TagDecl (struct/union/class/enum) decl.
  TypeRef getTagDeclType(TagDecl *Decl);
  
private:
  void InitBuiltinTypes();
  void InitBuiltinType(TypeRef &R, BuiltinType::Kind K);
};
  
}  // end namespace clang
}  // end namespace llvm

#endif
