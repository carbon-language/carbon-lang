//===--- DeclVisitor.h - Visitor for Decl subclasses ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the DeclVisitor interface.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_DECLVISITOR_H
#define LLVM_CLANG_AST_DECLVISITOR_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclTemplate.h"

namespace clang {

#define DISPATCH(NAME, CLASS) \
  return static_cast<ImplClass*>(this)-> Visit##NAME(static_cast<CLASS*>(D))

/// \brief A simple visitor class that helps create declaration visitors.
template<typename ImplClass, typename RetTy=void>
class DeclVisitor {
public:
  RetTy Visit(Decl *D) {
    switch (D->getKind()) {
      default: assert(false && "Decl that isn't part of DeclNodes.def!");
#define DECL(Derived, Base) \
      case Decl::Derived: DISPATCH(Derived##Decl, Derived##Decl);
#define ABSTRACT_DECL(Derived, Base)
#include "clang/AST/DeclNodes.def"
    }
  }

  // If the implementation chooses not to implement a certain visit
  // method, fall back to the parent.
#define DECL(Derived, Base)                                             \
  RetTy Visit##Derived##Decl(Derived##Decl *D) { DISPATCH(Base, Base); }
#define ABSTRACT_DECL(Derived, Base) DECL(Derived, Base)
#include "clang/AST/DeclNodes.def"

  RetTy VisitDecl(Decl *D) { return RetTy(); }
};

#undef DISPATCH

}  // end namespace clang

#endif // LLVM_CLANG_AST_DECLVISITOR_H
