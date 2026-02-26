// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Header file template expanded with strings describing build info for Carbon's
// runtimes.
//
// See toolchain/base/runtimes_build_info.bzl for more details.

#ifndef CARBON_TOOLCHAIN_BASE_RUNTIMES_BUILD_INFO_TPL_H_
#define CARBON_TOOLCHAIN_BASE_RUNTIMES_BUILD_INFO_TPL_H_

#include "llvm/ADT/StringRef.h"

namespace Carbon::RuntimesBuildInfo {

constexpr inline llvm::StringLiteral CrtBegin = CRTBEGIN_SRC;
constexpr inline llvm::StringLiteral CrtEnd = CRTEND_SRC;

constexpr inline llvm::StringLiteral CrtCopts[] = {CRT_COPTS};

// Prevent wrapping these lines -- the expansion of the variables will add line
// breaks.
//
// clang-format off
constexpr inline llvm::StringLiteral BuiltinsGenericSrcs[] = {BUILTINS_GENERIC_SRCS};
constexpr inline llvm::StringLiteral BuiltinsMacosSrcs[] = {BUILTINS_MACOS_SRCS};
constexpr inline llvm::StringLiteral BuiltinsBf16Srcs[] = {BUILTINS_BF16_SRCS};
constexpr inline llvm::StringLiteral BuiltinsTfSrcs[] = {BUILTINS_TF_SRCS};
constexpr inline llvm::StringLiteral BuiltinsX86ArchSrcs[] = {BUILTINS_X86_ARCH_SRCS};
constexpr inline llvm::StringLiteral BuiltinsX86Fp80Srcs[] = {BUILTINS_X86_FP80_SRCS};
constexpr inline llvm::StringLiteral BuiltinsAarch64Srcs[] = {BUILTINS_AARCH64_SRCS};
// NOLINTNEXTLINE(readability-identifier-naming)
constexpr inline llvm::StringLiteral BuiltinsX86_64Srcs[] = {BUILTINS_X86_64_SRCS};
constexpr inline llvm::StringLiteral BuiltinsI386Srcs[] = {BUILTINS_I386_SRCS};
// clang-format on

constexpr inline llvm::StringLiteral LibcxxLinuxSrcs[] = {LIBCXX_LINUX_SRCS};
constexpr inline llvm::StringLiteral LibcxxMacosSrcs[] = {LIBCXX_MACOS_SRCS};
constexpr inline llvm::StringLiteral LibcxxWin32Srcs[] = {LIBCXX_WIN32_SRCS};
constexpr inline llvm::StringLiteral LibcxxabiSrcs[] = {LIBCXXABI_SRCS};
constexpr inline llvm::StringLiteral LibcxxCopts[] = {LIBCXX_AND_ABI_COPTS};

constexpr inline llvm::StringLiteral LibunwindSrcs[] = {LIBUNWIND_SRCS};
constexpr inline llvm::StringLiteral LibunwindCopts[] = {LIBUNWIND_COPTS};

}  // namespace Carbon::RuntimesBuildInfo

#endif  // CARBON_TOOLCHAIN_BASE_RUNTIMES_BUILD_INFO_TPL_H_
