//===--- Sema.cpp - AST Builder and Semantic Analysis Implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the actions class which performs semantic analysis and
// builds an AST out of a parse stream.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
using namespace llvm;
using namespace clang;

Sema::Sema(Preprocessor &pp, ASTContext &ctxt, std::vector<Decl*> &prevInGroup)
  : PP(pp), Context(ctxt), CurFunctionDecl(0), LastInGroupList(prevInGroup) {
}

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

bool Sema::Diag(SourceLocation Loc, unsigned DiagID) {
  PP.getDiagnostics().Report(Loc, DiagID);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg) {
  PP.getDiagnostics().Report(Loc, DiagID, &Msg, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1,
                const std::string &Msg2) {
  std::string MsgArr[] = { Msg1, Msg2 };
  PP.getDiagnostics().Report(Loc, DiagID, MsgArr, 2);
  return true;
}

bool Sema::Diag(const LexerToken &Tok, unsigned DiagID) {
  PP.getDiagnostics().Report(Tok.getLocation(), DiagID);
  return true;
}

bool Sema::Diag(const LexerToken &Tok, unsigned DiagID, const std::string &M) {
  PP.getDiagnostics().Report(Tok.getLocation(), DiagID, &M, 1);
  return true;
}

const LangOptions &Sema::getLangOptions() const {
  return PP.getLangOptions();
}
