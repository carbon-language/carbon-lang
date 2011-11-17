//== CheckerContext.cpp - Context info for path-sensitive checkers-----------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines CheckerContext that provides contextual info for
//  path-sensitive checkers.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
using namespace clang;
using namespace ento;

StringRef CheckerContext::getCalleeName(const CallExpr *CE) const {
  const ProgramState *State = getState();
  const Expr *Callee = CE->getCallee();
  SVal L = State->getSVal(Callee);

  const FunctionDecl *funDecl = L.getAsFunctionDecl();
  if (!funDecl)
    return StringRef();
  IdentifierInfo *funI = funDecl->getIdentifier();
  if (!funI)
    return StringRef();
  return funI->getName();
}
