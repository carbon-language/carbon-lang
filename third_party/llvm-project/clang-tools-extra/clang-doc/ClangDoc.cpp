//===-- ClangDoc.cpp - ClangDoc ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the main entry point for the clang-doc tool. It runs
// the clang-doc mapper on a given set of source code files using a
// FrontendActionFactory.
//
//===----------------------------------------------------------------------===//

#include "ClangDoc.h"
#include "Mapper.h"
#include "Representation.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"

namespace clang {
namespace doc {

class MapperActionFactory : public tooling::FrontendActionFactory {
public:
  MapperActionFactory(ClangDocContext CDCtx) : CDCtx(CDCtx) {}
  std::unique_ptr<FrontendAction> create() override;

private:
  ClangDocContext CDCtx;
};

std::unique_ptr<FrontendAction> MapperActionFactory::create() {
  class ClangDocAction : public clang::ASTFrontendAction {
  public:
    ClangDocAction(ClangDocContext CDCtx) : CDCtx(CDCtx) {}

    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &Compiler,
                      llvm::StringRef InFile) override {
      return std::make_unique<MapASTVisitor>(&Compiler.getASTContext(), CDCtx);
    }

  private:
    ClangDocContext CDCtx;
  };
  return std::make_unique<ClangDocAction>(CDCtx);
}

std::unique_ptr<tooling::FrontendActionFactory>
newMapperActionFactory(ClangDocContext CDCtx) {
  return std::make_unique<MapperActionFactory>(CDCtx);
}

} // namespace doc
} // namespace clang
