// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_ACCESS_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_ACCESS_H_

#include "toolchain/sem_ir/name_scope.h"

namespace Carbon::Check {

// Calculates the effective access kind from the given (declaration, lookup
// access) pair.
auto MapCppAccess(clang::DeclAccessPair access_pair) -> SemIR::AccessKind;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_ACCESS_H_
