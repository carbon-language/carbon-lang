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
#include "clang/Basic/Builtins.h"

using namespace clang;
using namespace ento;

const FunctionDecl *CheckerContext::getCalleeDecl(const CallExpr *CE) const {
  const ProgramState *State = getState();
  const Expr *Callee = CE->getCallee();
  SVal L = State->getSVal(Callee, Pred->getLocationContext());
  return L.getAsFunctionDecl();
}

StringRef CheckerContext::getCalleeName(const FunctionDecl *FunDecl) const {
  if (!FunDecl)
    return StringRef();
  IdentifierInfo *funI = FunDecl->getIdentifier();
  if (!funI)
    return StringRef();
  return funI->getName();
}


bool CheckerContext::isCLibraryFunction(const FunctionDecl *FD,
                                        StringRef Name){
  // To avoid false positives (Ex: finding user defined functions with
  // similar names), only perform fuzzy name matching when it's a builtin.
  // Using a string compare is slow, we might want to switch on BuiltinID here.
  unsigned BId = FD->getBuiltinID();
  if (BId != 0) {
    ASTContext &Context = getASTContext();
    StringRef BName = Context.BuiltinInfo.GetName(BId);
    if (StringRef(BName).find(Name) != StringRef::npos)
      return true;
  }

  if (FD->isExternC() && FD->getIdentifier()->getName().equals(Name))
    return true;

  return false;
}
