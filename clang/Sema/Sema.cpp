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

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, SourceRange Range) {
  PP.getDiagnostics().Report(Loc, DiagID, 0, 0, &Range, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg,
                SourceRange Range) {
  PP.getDiagnostics().Report(Loc, DiagID, &Msg, 1, &Range, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1,
                const std::string &Msg2, SourceRange Range) {
  std::string MsgArr[] = { Msg1, Msg2 };
  PP.getDiagnostics().Report(Loc, DiagID, MsgArr, 2, &Range, 1);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID,
                SourceRange R1, SourceRange R2) {
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(Loc, DiagID, 0, 0, RangeArr, 2);
  return true;
}

bool Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg,
                SourceRange R1, SourceRange R2) {
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(Loc, DiagID, &Msg, 1, RangeArr, 2);
  return true;
}

bool Sema::Diag(SourceLocation Range, unsigned DiagID, const std::string &Msg1,
                const std::string &Msg2, SourceRange R1, SourceRange R2) {
  std::string MsgArr[] = { Msg1, Msg2 };
  SourceRange RangeArr[] = { R1, R2 };
  PP.getDiagnostics().Report(Range, DiagID, MsgArr, 2, RangeArr, 2);
  return true;
}

const LangOptions &Sema::getLangOptions() const {
  return PP.getLangOptions();
}
