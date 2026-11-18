// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/context.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CompilerInstance.h"

namespace Carbon::Check {

CppContext::CppContext(SemIR::CppDomain& domain,
                       std::unique_ptr<CppDiagnosticListener> listener)
    : domain_(&domain), diagnostic_listener_(std::move(listener)) {}

CppContext::~CppContext() = default;

auto CppContext::ast_context() -> clang::ASTContext& {
  return domain_->clang_instance().getASTContext();
}

auto CppContext::sema() -> clang::Sema& {
  return domain_->clang_instance().getSema();
}

auto CppContext::clang_mangle_context() -> clang::MangleContext& {
  if (!clang_mangle_context_) {
    clang_mangle_context_.reset(ast_context().createMangleContext());
  }
  return *clang_mangle_context_;
}

}  // namespace Carbon::Check
