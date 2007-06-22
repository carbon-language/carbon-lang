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
#include "clang/AST/Expr.h"
#include <vector>

namespace clang {
  class TargetInfo;
  
/// ASTContext - This class holds long-lived AST nodes (such as types and
/// decls) that can be referred to throughout the semantic analysis of a file.
class ASTContext {
  std::vector<Type*> Types;
  llvm::FoldingSet<ComplexType> ComplexTypes;
  llvm::FoldingSet<PointerType> PointerTypes;
  llvm::FoldingSet<ReferenceType> ReferenceTypes;
  llvm::FoldingSet<ArrayType> ArrayTypes;
  llvm::FoldingSet<FunctionTypeNoProto> FunctionTypeNoProtos;
  llvm::FoldingSet<FunctionTypeProto> FunctionTypeProtos;
public:
  TargetInfo &Target;
  Builtin::Context BuiltinInfo;

  // Builtin Types.
  QualType VoidTy;
  QualType BoolTy;
  QualType CharTy;
  QualType SignedCharTy, ShortTy, IntTy, LongTy, LongLongTy;
  QualType UnsignedCharTy, UnsignedShortTy, UnsignedIntTy, UnsignedLongTy;
  QualType UnsignedLongLongTy;
  QualType FloatTy, DoubleTy, LongDoubleTy;
  QualType FloatComplexTy, DoubleComplexTy, LongDoubleComplexTy;
  
  ASTContext(TargetInfo &t, IdentifierTable &idents) : Target(t) {
    InitBuiltinTypes();
    BuiltinInfo.InitializeBuiltins(idents, Target);
  }    
  ~ASTContext();
  
  void PrintStats() const;

  /// getComplexType - Return the uniqued reference to the type for a complex
  /// number with the specified element type.
  QualType getComplexType(QualType T);
  
  /// getPointerType - Return the uniqued reference to the type for a pointer to
  /// the specified type.
  QualType getPointerType(QualType T);
  
  /// getReferenceType - Return the uniqued reference to the type for a
  /// reference to the specified type.
  QualType getReferenceType(QualType T);
  
  /// getArrayType - Return the unique reference to the type for an array of the
  /// specified element type.
  QualType getArrayType(QualType EltTy, ArrayType::ArraySizeModifier ASM,
                        unsigned EltTypeQuals, Expr *NumElts);

  /// getFunctionTypeNoProto - Return a K&R style C function type like 'int()'.
  ///
  QualType getFunctionTypeNoProto(QualType ResultTy);
  
  /// getFunctionType - Return a normal function type with a typed argument
  /// list.  isVariadic indicates whether the argument list includes '...'.
  QualType getFunctionType(QualType ResultTy, QualType *ArgArray,
                           unsigned NumArgs, bool isVariadic);
  
  /// getTypedefType - Return the unique reference to the type for the
  /// specified typename decl.
  QualType getTypedefType(TypedefDecl *Decl);

  /// getTagDeclType - Return the unique reference to the type for the
  /// specified TagDecl (struct/union/class/enum) decl.
  QualType getTagDeclType(TagDecl *Decl);
  
  /// getSizeType - Return the unique type for "size_t" (C99 7.17), defined
  /// in <stddef.h>. The sizeof operator requires this (C99 6.5.3.4p4).
  QualType getSizeType() const;
  
  /// getIntegerBitwidth - Return the bitwidth of the specified integer type
  /// according to the target.  'Loc' specifies the source location that
  /// requires evaluation of this property.
  unsigned getIntegerBitwidth(QualType T, SourceLocation Loc);

  // maxIntegerType - Returns the highest ranked integer type. Handles 3
  // different type combos: unsigned/unsigned, signed/signed, signed/unsigned.
  static QualType maxIntegerType(QualType lhs, QualType rhs);
  
  // maxFloatingType - Returns the highest ranked float type. Both input 
  // types are required to be floats.
  static QualType maxFloatingType(QualType lt, QualType rt);

  // maxComplexType - Returns the highest ranked complex type. Handles 3
  // different type combos: complex/complex, complex/float, float/complex. 
  QualType maxComplexType(QualType lt, QualType rt) const;
  
private:
  ASTContext(const ASTContext&); // DO NOT IMPLEMENT
  void operator=(const ASTContext&); // DO NOT IMPLEMENT
  
  void InitBuiltinTypes();
  void InitBuiltinType(QualType &R, BuiltinType::Kind K);
};
  
}  // end namespace clang

#endif
