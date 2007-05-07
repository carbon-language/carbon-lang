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
using namespace llvm;
using namespace clang;

Sema::Sema(Preprocessor &pp, ASTContext &ctxt, std::vector<Decl*> &prevInGroup)
  : PP(pp), Context(ctxt), CurFunctionDecl(0), LastInGroupList(prevInGroup) {
  constantOne = new IntegerLiteral(1, ctxt.IntTy);
}

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg) {
  PP.Diag(Loc, DiagID, Msg);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, QualType t) {
  std::string Name;
  t.getAsString(Name);
  PP.Diag(Loc, DiagID, Name);
  return true;
}

bool Sema::Diag(const LexerToken &Tok, unsigned DiagID, const std::string &M) {
  Diag(Tok.getLocation(), DiagID, M);
  return true;
}

const LangOptions &Sema::getLangOptions() const {
  return PP.getLangOptions();
}
