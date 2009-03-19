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
