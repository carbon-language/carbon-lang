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
#include "clang/Lex/Lexer.h"

using namespace clang;
using namespace ento;

const FunctionDecl *CheckerContext::getCalleeDecl(const CallExpr *CE) const {
  ProgramStateRef State = getState();
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
                                        StringRef Name) {
  return isCLibraryFunction(FD, Name, getASTContext());
}

bool CheckerContext::isCLibraryFunction(const FunctionDecl *FD,
                                        StringRef Name, ASTContext &Context) {
  // To avoid false positives (Ex: finding user defined functions with
  // similar names), only perform fuzzy name matching when it's a builtin.
  // Using a string compare is slow, we might want to switch on BuiltinID here.
  unsigned BId = FD->getBuiltinID();
  if (BId != 0) {
    StringRef BName = Context.BuiltinInfo.GetName(BId);
    if (BName.find(Name) != StringRef::npos)
      return true;
  }

  const IdentifierInfo *II = FD->getIdentifier();
  // If this is a special C++ name without IdentifierInfo, it can't be a
  // C library function.
  if (!II)
    return false;

  StringRef FName = II->getName();
  if (FName.equals(Name))
    return true;

  if (FName.startswith("__inline") && (FName.find(Name) != StringRef::npos))
    return true;

  if (FName.startswith("__") && FName.endswith("_chk") &&
      FName.find(Name) != StringRef::npos)
    return true;

  return false;
}

StringRef CheckerContext::getMacroNameOrSpelling(SourceLocation &Loc) {
  if (Loc.isMacroID())
    return Lexer::getImmediateMacroName(Loc, getSourceManager(),
                                             getLangOpts());
  SmallVector<char, 16> buf;
  return Lexer::getSpelling(Loc, buf, getSourceManager(), getLangOpts());
}

