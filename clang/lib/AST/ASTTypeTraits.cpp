//===--- ASTTypeTraits.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Provides a dynamic type identifier and a dynamically typed node container
//  that can be used to store an AST base node at runtime in the same storage in
//  a type safe way.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTTypeTraits.h"

namespace clang {
namespace ast_type_traits {

const ASTNodeKind::KindInfo ASTNodeKind::AllKindInfo[] = {
  { NKI_None, "<None>" },
  { NKI_None, "CXXCtorInitializer" },
  { NKI_None, "TemplateArgument" },
  { NKI_None, "NestedNameSpecifier" },
  { NKI_None, "NestedNameSpecifierLoc" },
  { NKI_None, "QualType" },
  { NKI_None, "TypeLoc" },
  { NKI_None, "Decl" },
#define DECL(DERIVED, BASE) { NKI_##BASE, #DERIVED "Decl" },
#include "clang/AST/DeclNodes.inc"
  { NKI_None, "Stmt" },
#define STMT(DERIVED, BASE) { NKI_##BASE, #DERIVED },
#include "clang/AST/StmtNodes.inc"
  { NKI_None, "Type" },
#define TYPE(DERIVED, BASE) { NKI_##BASE, #DERIVED "Type" },
#include "clang/AST/TypeNodes.def"
};

bool ASTNodeKind::isBaseOf(ASTNodeKind Other) const {
  return isBaseOf(KindId, Other.KindId);
}

bool ASTNodeKind::isSame(ASTNodeKind Other) const {
  return KindId != NKI_None && KindId == Other.KindId;
}

bool ASTNodeKind::isBaseOf(NodeKindId Base, NodeKindId Derived) {
  if (Base == NKI_None || Derived == NKI_None) return false;
  while (Derived != Base && Derived != NKI_None)
    Derived = AllKindInfo[Derived].ParentId;
  return Derived == Base;
}

StringRef ASTNodeKind::asStringRef() const { return AllKindInfo[KindId].Name; }

} // end namespace ast_type_traits
} // end namespace clang
