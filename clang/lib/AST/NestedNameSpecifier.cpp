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
#include <cassert>

using namespace clang;

NestedNameSpecifier *
NestedNameSpecifier::FindOrInsert(ASTContext &Context, 
                                  const NestedNameSpecifier &Mockup) {
  llvm::FoldingSetNodeID ID;
  Mockup.Profile(ID);

  void *InsertPos = 0;
  NestedNameSpecifier *NNS 
    = Context.NestedNameSpecifiers.FindNodeOrInsertPos(ID, InsertPos);
  if (!NNS) {
    NNS = new (Context, 4) NestedNameSpecifier(Mockup);
    Context.NestedNameSpecifiers.InsertNode(NNS, InsertPos);
  }

  return NNS;
}

NestedNameSpecifier *
NestedNameSpecifier::Create(ASTContext &Context, NestedNameSpecifier *Prefix, 
                            IdentifierInfo *II) {
  assert(II && "Identifier cannot be NULL");
  assert(Prefix && Prefix->isDependent() && "Prefix must be dependent");

  NestedNameSpecifier Mockup;
  Mockup.Prefix.setPointer(Prefix);
  Mockup.Prefix.setInt(Identifier);
  Mockup.Specifier = II;
  return FindOrInsert(Context, Mockup);
}

NestedNameSpecifier *
NestedNameSpecifier::Create(ASTContext &Context, NestedNameSpecifier *Prefix, 
                            NamespaceDecl *NS) {
  assert(NS && "Namespace cannot be NULL");
  assert((!Prefix || 
          (Prefix->getAsType() == 0 && Prefix->getAsIdentifier() == 0)) &&
         "Broken nested name specifier");
  NestedNameSpecifier Mockup;
  Mockup.Prefix.setPointer(Prefix);
  Mockup.Prefix.setInt(Namespace);
  Mockup.Specifier = NS;
  return FindOrInsert(Context, Mockup);
}

NestedNameSpecifier *
NestedNameSpecifier::Create(ASTContext &Context, NestedNameSpecifier *Prefix,
                            bool Template, Type *T) {
  assert(T && "Type cannot be NULL");
  NestedNameSpecifier Mockup;
  Mockup.Prefix.setPointer(Prefix);
  Mockup.Prefix.setInt(Template? TypeSpecWithTemplate : TypeSpec);
  Mockup.Specifier = T;
  return FindOrInsert(Context, Mockup);
}
  
NestedNameSpecifier *NestedNameSpecifier::GlobalSpecifier(ASTContext &Context) {
  if (!Context.GlobalNestedNameSpecifier)
    Context.GlobalNestedNameSpecifier = new (Context, 4) NestedNameSpecifier();
  return Context.GlobalNestedNameSpecifier;
}

/// \brief Whether this nested name specifier refers to a dependent
/// type or not.
bool NestedNameSpecifier::isDependent() const {
  switch (getKind()) {
  case Identifier:
    // Identifier specifiers always represent dependent types
    return true;

  case Namespace:
  case Global:
    return false;

  case TypeSpec:
  case TypeSpecWithTemplate:
    return getAsType()->isDependentType();
  }

  // Necessary to suppress a GCC warning.
  return false;
}

/// \brief Print this nested name specifier to the given output
/// stream.
void NestedNameSpecifier::print(llvm::raw_ostream &OS) const {
  if (getPrefix())
    getPrefix()->print(OS);

  switch (getKind()) {
  case Identifier:
    OS << getAsIdentifier()->getName();
    break;

  case Namespace:
    OS << getAsNamespace()->getIdentifier()->getName();
    break;

  case Global:
    break;

  case TypeSpecWithTemplate:
    OS << "template ";
    // Fall through to print the type.

  case TypeSpec: {
    std::string TypeStr;
    Type *T = getAsType();

    // If this is a qualified name type, suppress the qualification:
    // it's part of our nested-name-specifier sequence anyway.  FIXME:
    // We should be able to assert that this doesn't happen.
    if (const QualifiedNameType *QualT = dyn_cast<QualifiedNameType>(T))
      T = QualT->getNamedType().getTypePtr();
    
    if (const TagType *TagT = dyn_cast<TagType>(T))
      TagT->getAsStringInternal(TypeStr, true);
    else
      T->getAsStringInternal(TypeStr);
    OS << TypeStr;
    break;
  }
  }

  OS << "::";
}

void NestedNameSpecifier::Destroy(ASTContext &Context) {
  this->~NestedNameSpecifier();
  Context.Deallocate((void *)this);
}

void NestedNameSpecifier::dump() {
  print(llvm::errs());
}
