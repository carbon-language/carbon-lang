//===---------------- SemaCodeComplete.cpp - Code Completion ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the code-completion semantic actions.
//
//===----------------------------------------------------------------------===//
#include "Sema.h"
#include "clang/Sema/CodeCompleteConsumer.h"

using namespace clang;

/// \brief Set the code-completion consumer for semantic analysis.
void Sema::setCodeCompleteConsumer(CodeCompleteConsumer *CCC) {
  assert(((CodeCompleter != 0) != (CCC != 0)) && 
         "Already set or cleared a code-completion consumer?");
  CodeCompleter = CCC;
}

void Sema::CodeCompleteMemberReferenceExpr(Scope *S, ExprTy *BaseE,
                                           SourceLocation OpLoc,
                                           bool IsArrow) {
  if (!BaseE || !CodeCompleter)
    return;
  
  Expr *Base = static_cast<Expr *>(BaseE);
  QualType BaseType = Base->getType();
   
  CodeCompleter->CodeCompleteMemberReferenceExpr(S, BaseType, IsArrow);
}

void Sema::CodeCompleteQualifiedId(Scope *S, const CXXScopeSpec &SS,
                                   bool EnteringContext) {
  if (!SS.getScopeRep() || !CodeCompleter)
    return;
  
  CodeCompleter->CodeCompleteQualifiedId(S, 
                                      (NestedNameSpecifier *)SS.getScopeRep(),
                                         EnteringContext);
}
