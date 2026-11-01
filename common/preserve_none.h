// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_PRESERVE_NONE_H_
#define CARBON_COMMON_PRESERVE_NONE_H_

#include "llvm/Support/Compiler.h"

// Applies `[[clang::preserve_none]]` when supported and safe.
//
// Disabled when building with AddressSanitizer on AArch64 to work around an
// LLVM backend bug: when sanitizer stack instrumentation combines dynamic stack
// allocation with stack realignment, AArch64 reserves `x19` as the stack base
// pointer (`hasBasePointer`) and fails to save and restore `x19` across calls
// to `[[clang::preserve_none]]` callees that clobber `x19`.
#if __has_cpp_attribute(clang::preserve_none) &&     \
    !((defined(__aarch64__) || defined(_M_ARM64)) && \
      (LLVM_ADDRESS_SANITIZER_BUILD || LLVM_HWADDRESS_SANITIZER_BUILD))
#define CARBON_PRESERVE_NONE [[clang::preserve_none]]
#else
#define CARBON_PRESERVE_NONE
#endif

#endif  // CARBON_COMMON_PRESERVE_NONE_H_
