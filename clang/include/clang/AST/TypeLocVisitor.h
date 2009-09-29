//===--- TypeLocVisitor.h - Visitor for TypeLoc subclasses ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TypeLocVisitor interface.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_TYPELOCVISITOR_H
#define LLVM_CLANG_AST_TYPELOCVISITOR_H

#include "clang/AST/TypeLoc.h"
#include "clang/AST/TypeVisitor.h"

namespace clang {

#define DISPATCH(CLASS) \
  return static_cast<ImplClass*>(this)->Visit ## CLASS(cast<CLASS>(TyLoc))

template<typename ImplClass, typename RetTy=void>
class TypeLocVisitor {
  class TypeDispatch : public TypeVisitor<TypeDispatch, RetTy> {
    ImplClass *Impl;
    TypeLoc TyLoc;

  public:
    TypeDispatch(ImplClass *impl, TypeLoc &tyLoc) : Impl(impl), TyLoc(tyLoc) { }
#define ABSTRACT_TYPELOC(CLASS)
#define TYPELOC(CLASS, PARENT, TYPE)                              \
    RetTy Visit##TYPE(TYPE *) {                                   \
      return Impl->Visit##CLASS(reinterpret_cast<CLASS&>(TyLoc)); \
    }
#include "clang/AST/TypeLocNodes.def"
  };

public:
  RetTy Visit(TypeLoc TyLoc) {
    TypeDispatch TD(static_cast<ImplClass*>(this), TyLoc);
    return TD.Visit(TyLoc.getSourceType().getTypePtr());
  }

#define TYPELOC(CLASS, PARENT, TYPE) RetTy Visit##CLASS(CLASS TyLoc) {       \
  DISPATCH(PARENT);                                                          \
}
#include "clang/AST/TypeLocNodes.def"

  RetTy VisitTypeLoc(TypeLoc TyLoc) { return RetTy(); }
};

#undef DISPATCH

}  // end namespace clang

#endif // LLVM_CLANG_AST_TYPELOCVISITOR_H
