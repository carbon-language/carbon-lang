//===--- TypeVisitor.h - Visitor for Stmt subclasses ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TypeVisitor interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TYPEVISITOR_H
#define LLVM_CLANG_AST_TYPEVISITOR_H

#include "clang/AST/Type.h"

namespace clang {

#define DISPATCH(CLASS) \
  return static_cast<ImplClass*>(this)->Visit ## CLASS(static_cast<CLASS*>(T))

template<typename ImplClass, typename RetTy=void>
class TypeVisitor {
public:
  RetTy Visit(Type *T) {
    // Top switch stmt: dispatch to VisitFooStmt for each FooStmt.
    switch (T->getTypeClass()) {
    default: assert(0 && "Unknown type class!");
#define ABSTRACT_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) case Type::CLASS: DISPATCH(CLASS##Type);
#include "clang/AST/TypeNodes.def"
    }
  }

  // If the implementation chooses not to implement a certain visit method, fall
  // back on superclass.
#define TYPE(CLASS, PARENT) RetTy Visit##CLASS##Type(CLASS##Type *T) {       \
  DISPATCH(PARENT);                                                          \
}
#include "clang/AST/TypeNodes.def"

  // Base case, ignore it. :)
  RetTy VisitType(Type*) { return RetTy(); }
};

#undef DISPATCH

}  // end namespace clang

#endif
