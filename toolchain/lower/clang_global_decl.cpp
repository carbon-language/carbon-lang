// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/clang_global_decl.h"
namespace Carbon::Lower {

auto CreateGlobalDecl(const clang::NamedDecl* decl) -> clang::GlobalDecl {
  if (const auto* constructor_decl =
          dyn_cast<clang::CXXConstructorDecl>(decl)) {
    return clang::GlobalDecl(constructor_decl,
                             clang::CXXCtorType::Ctor_Complete);
  }

  return clang::GlobalDecl(decl);
}

}  // namespace Carbon::Lower
