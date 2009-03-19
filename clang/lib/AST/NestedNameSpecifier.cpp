//===--- NestedNameSpecifier.cpp - C++ nested name specifiers -----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the NestedNameSpecifier class, which represents
//  a C++ nested-name-specifier.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

DeclContext *
NestedNameSpecifier::computeDeclContext(ASTContext &Context) const {
  // The simple case: we're storing a DeclContext
  if ((Data & 0x01) == 0)
    return reinterpret_cast<DeclContext *>(Data);

  Type *T = getAsType();
  if (!T)
    return 0;

  // Retrieve the DeclContext associated with this type.
  const TagType *TagT = T->getAsTagType();
  assert(TagT && "No DeclContext from a non-tag type");
  return TagT->getDecl();
}

void NestedNameSpecifier::Print(llvm::raw_ostream &OS, 
                                const NestedNameSpecifier *First,
                                const NestedNameSpecifier *Last) {
  for (; First != Last; ++First) {
    if (Type *T = First->getAsType()) {
      std::string TypeStr;

      // If this is a qualified name type, suppress the qualification:
      // it's part of our nested-name-specifier sequence anyway.
      if (const QualifiedNameType *QualT = dyn_cast<QualifiedNameType>(T))
        T = QualT->getNamedType().getTypePtr();

      if (const TagType *TagT = dyn_cast<TagType>(T))
        TagT->getAsStringInternal(TypeStr, true);
      else
        T->getAsStringInternal(TypeStr);
      OS << TypeStr;
    } else if (NamedDecl *NamedDC 
                 = dyn_cast_or_null<NamedDecl>(First->getAsDeclContext()))
      OS << NamedDC->getNameAsString();
  
    OS <<  "::";
  }
}
