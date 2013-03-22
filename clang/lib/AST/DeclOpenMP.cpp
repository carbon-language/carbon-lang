//===--- DeclOpenMP.cpp - Declaration OpenMP AST Node Implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements OMPThreadPrivateDecl class.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/Expr.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// OMPThreadPrivateDecl Implementation.
//===----------------------------------------------------------------------===//

void OMPThreadPrivateDecl::anchor() { }

OMPThreadPrivateDecl *OMPThreadPrivateDecl::Create(ASTContext &C,
                                                   DeclContext *DC,
                                                   SourceLocation L,
                                                   ArrayRef<DeclRefExpr *> VL) {
  unsigned Size = sizeof(OMPThreadPrivateDecl) +
                  (VL.size() * sizeof(DeclRefExpr *));

  void *Mem = C.Allocate(Size, llvm::alignOf<OMPThreadPrivateDecl>());
  OMPThreadPrivateDecl *D = new (Mem) OMPThreadPrivateDecl(OMPThreadPrivate,
                                                           DC, L);
  D->NumVars = VL.size();
  D->setVars(VL);
  return D;
}

OMPThreadPrivateDecl *OMPThreadPrivateDecl::CreateDeserialized(ASTContext &C,
                                                               unsigned ID,
                                                               unsigned N) {
  unsigned Size = sizeof(OMPThreadPrivateDecl) + (N * sizeof(DeclRefExpr *));

  void *Mem = AllocateDeserializedDecl(C, ID, Size);
  OMPThreadPrivateDecl *D = new (Mem) OMPThreadPrivateDecl(OMPThreadPrivate,
                                                           0, SourceLocation());
  D->NumVars = N;
  return D;
}

void OMPThreadPrivateDecl::setVars(ArrayRef<DeclRefExpr *> VL) {
  assert(VL.size() == NumVars &&
         "Number of variables is not the same as the preallocated buffer");
  DeclRefExpr **Vars = reinterpret_cast<DeclRefExpr **>(this + 1);
  std::copy(VL.begin(), VL.end(), Vars);
}
