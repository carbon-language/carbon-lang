// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_CLANG_GLOBAL_DECL_H_
#define CARBON_TOOLCHAIN_LOWER_CLANG_GLOBAL_DECL_H_

#include "clang/AST/Decl.h"
#include "clang/AST/GlobalDecl.h"

namespace Carbon::Lower {

// Returns `clang::GlobalDecl` with special handling of constructors, assuming
// they're complete.
auto CreateGlobalDecl(const clang::NamedDecl* decl) -> clang::GlobalDecl;

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_CLANG_GLOBAL_DECL_H_
