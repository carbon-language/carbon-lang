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
#include "clang/AST/PrettyPrinter.h"
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

NestedNameSpecifier *
NestedNameSpecifier::Create(ASTContext &Context, IdentifierInfo *II) {
  assert(II && "Identifier cannot be NULL");
  NestedNameSpecifier Mockup;
  Mockup.Prefix.setPointer(0);
  Mockup.Prefix.setInt(Identifier);
  Mockup.Specifier = II;
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
void 
NestedNameSpecifier::print(llvm::raw_ostream &OS, 
                           const PrintingPolicy &Policy) const {
  if (getPrefix())
    getPrefix()->print(OS, Policy);

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

    PrintingPolicy InnerPolicy(Policy);
    InnerPolicy.SuppressTagKind = true;
    
    // Nested-name-specifiers are intended to contain minimally-qualified
    // types. An actual QualifiedNameType will not occur, since we'll store
    // just the type that is referred to in the nested-name-specifier (e.g.,
    // a TypedefType, TagType, etc.). However, when we are dealing with
    // dependent template-id types (e.g., Outer<T>::template Inner<U>), 
    // the type requires its own nested-name-specifier for uniqueness, so we
    // suppress that nested-name-specifier during printing.
    assert(!isa<QualifiedNameType>(T) && 
           "Qualified name type in nested-name-specifier");
    if (const TemplateSpecializationType *SpecType
          = dyn_cast<TemplateSpecializationType>(T)) {
      // Print the template name without its corresponding 
      // nested-name-specifier.
      SpecType->getTemplateName().print(OS, InnerPolicy, true);
      
      // Print the template argument list.
      TypeStr = TemplateSpecializationType::PrintTemplateArgumentList(
                                                          SpecType->getArgs(), 
                                                       SpecType->getNumArgs(), 
                                                                 InnerPolicy);
    } else {
      // Print the type normally
      T->getAsStringInternal(TypeStr, InnerPolicy);
    }
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

void NestedNameSpecifier::dump(const LangOptions &LO) {
  print(llvm::errs(), PrintingPolicy(LO));
}
