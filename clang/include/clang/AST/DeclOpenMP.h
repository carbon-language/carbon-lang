//===--- OpenMP.h - Classes for representing OpenMP directives ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines OpenMP nodes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_OPENMP_H
#define LLVM_CLANG_AST_OPENMP_H

#include "clang/AST/DeclBase.h"
#include "llvm/ADT/ArrayRef.h"

namespace clang {

class DeclRefExpr;

/// \brief This represents '#pragma omp threadprivate ...' directive.
/// For example, in the following, both 'a' and 'A::b' are threadprivate:
///
/// \code
/// int a;
/// #pragma omp threadprivate(a)
/// struct A {
///   static int b;
/// #pragma omp threadprivate(b)
/// };
/// \endcode
///
class OMPThreadPrivateDecl : public Decl {
  friend class ASTDeclReader;
  unsigned NumVars;

  virtual void anchor();

  OMPThreadPrivateDecl(Kind DK, DeclContext *DC, SourceLocation L) :
    Decl(DK, DC, L), NumVars(0) { }

  ArrayRef<const DeclRefExpr *> getVars() const {
    return ArrayRef<const DeclRefExpr *>(
                   reinterpret_cast<const DeclRefExpr * const *>(this + 1),
                   NumVars);
  }

  llvm::MutableArrayRef<DeclRefExpr *> getVars() {
    return llvm::MutableArrayRef<DeclRefExpr *>(
                                 reinterpret_cast<DeclRefExpr **>(this + 1),
                                 NumVars);
  }

  void setVars(ArrayRef<DeclRefExpr *> VL);

public:
  static OMPThreadPrivateDecl *Create(ASTContext &C, DeclContext *DC,
                                      SourceLocation L,
                                      ArrayRef<DeclRefExpr *> VL);
  static OMPThreadPrivateDecl *CreateDeserialized(ASTContext &C,
                                                  unsigned ID, unsigned N);

  typedef llvm::MutableArrayRef<DeclRefExpr *>::iterator varlist_iterator;
  typedef ArrayRef<const DeclRefExpr *>::iterator varlist_const_iterator;

  unsigned varlist_size() const { return NumVars; }
  bool varlist_empty() const { return NumVars == 0; }
  varlist_iterator varlist_begin() { return getVars().begin(); }
  varlist_iterator varlist_end() { return getVars().end(); }
  varlist_const_iterator varlist_begin() const { return getVars().begin(); }
  varlist_const_iterator varlist_end() const { return getVars().end(); }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == OMPThreadPrivate; }
};

}  // end namespace clang

#endif
