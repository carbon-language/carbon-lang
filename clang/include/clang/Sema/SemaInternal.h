//===--- SemaInternal.h - Internal Sema Interfaces --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides common API and #includes for the internal
// implementation of Sema.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMA_INTERNAL_H
#define LLVM_CLANG_SEMA_SEMA_INTERNAL_H

#include "clang/AST/ASTContext.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

namespace clang {

inline PartialDiagnostic Sema::PDiag(unsigned DiagID) {
  return PartialDiagnostic(DiagID, Context.getDiagAllocator());
}


// This requires the variable to be non-dependent and the initializer
// to not be value dependent.
inline bool IsVariableAConstantExpression(VarDecl *Var, ASTContext &Context) {
  const VarDecl *DefVD = 0;
  return !isa<ParmVarDecl>(Var) &&
    Var->isUsableInConstantExpressions(Context) &&
    Var->getAnyInitializer(DefVD) && DefVD->checkInitIsICE(); 
}

// Directly mark a variable odr-used. Given a choice, prefer to use 
// MarkVariableReferenced since it does additional checks and then 
// calls MarkVarDeclODRUsed.
// If the variable must be captured:
//  - if FunctionScopeIndexToStopAt is null, capture it in the CurContext
//  - else capture it in the DeclContext that maps to the 
//    *FunctionScopeIndexToStopAt on the FunctionScopeInfo stack.  
inline void MarkVarDeclODRUsed(VarDecl *Var,
    SourceLocation Loc, Sema &SemaRef,
    const unsigned *const FunctionScopeIndexToStopAt) {
  // Keep track of used but undefined variables.
  // FIXME: We shouldn't suppress this warning for static data members.
  if (Var->hasDefinition(SemaRef.Context) == VarDecl::DeclarationOnly &&
    !Var->isExternallyVisible() &&
    !(Var->isStaticDataMember() && Var->hasInit())) {
      SourceLocation &old = SemaRef.UndefinedButUsed[Var->getCanonicalDecl()];
      if (old.isInvalid()) old = Loc;
  }
  QualType CaptureType, DeclRefType;
  SemaRef.tryCaptureVariable(Var, Loc, Sema::TryCapture_Implicit, 
    /*EllipsisLoc*/ SourceLocation(),
    /*BuildAndDiagnose*/ true, 
    CaptureType, DeclRefType, 
    FunctionScopeIndexToStopAt);

  Var->markUsed(SemaRef.Context);
}
}

#endif
