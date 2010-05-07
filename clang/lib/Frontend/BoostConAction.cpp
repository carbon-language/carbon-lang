//===-- BoostConAction.cpp - BoostCon Workshop Action -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "clang/Frontend/FrontendActions.h"
#include "clang/AST/ASTConsumer.h"
using namespace clang;

namespace {
  class BoostConASTConsumer : public ASTConsumer {
  public:
    /// HandleTranslationUnit - This method is called when the ASTs for entire
    /// translation unit have been parsed.
    virtual void HandleTranslationUnit(ASTContext &Ctx);
  };
}

ASTConsumer *BoostConAction::CreateASTConsumer(CompilerInstance &CI,
                                               llvm::StringRef InFile) {
  return new BoostConASTConsumer();
}

void BoostConASTConsumer::HandleTranslationUnit(ASTContext &Ctx) {
  fprintf(stderr, "Welcome to BoostCon!\n");
}
