// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/context.h"

#include "clang/AST/Mangle.h"

namespace Carbon::Check {

CppContext::CppContext(clang::CompilerInstance& instance,
                       std::unique_ptr<clang::Parser> parser)
    : ast_context_(&instance.getASTContext()),
      sema_(&instance.getSema()),
      parser_(std::move(parser)) {}

CppContext::~CppContext() = default;

auto CppContext::clang_mangle_context() -> clang::MangleContext& {
  if (!clang_mangle_context_) {
    clang_mangle_context_.reset(ast_context().createMangleContext());
  }
  return *clang_mangle_context_;
}

}  // namespace Carbon::Check
