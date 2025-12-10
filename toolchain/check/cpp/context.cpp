// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/context.h"
#include "clang/AST/Mangle.h"

namespace Carbon::Check {

auto CppContext::clang_mangle_context() -> clang::MangleContext& {
  if (!clang_mangle_context_) {
    clang_mangle_context_.reset(ast_context().createMangleContext());
  }
  return *clang_mangle_context_;
}

}  // namespace Carbon::Check
