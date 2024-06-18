// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/version.h"

namespace Carbon {

#pragma clang attribute push
// If requested, make the contents of this file weak.
#if $MAKE_WEAK
#pragma clang attribute(__attribute__((weak)), \
                        apply_to = any(function, variable))
#endif

constexpr llvm::StringLiteral Version::String =
    "$VERSION+$GIT_COMMIT_SHA$GIT_DIRTY_SUFFIX";

constexpr llvm::StringLiteral Version::ToolchainInfo = R"""(
Carbon Language toolchain version: $VERSION+$GIT_COMMIT_SHA$GIT_DIRTY_SUFFIX
)""";

#pragma clang attribute pop

}  // namespace Carbon
