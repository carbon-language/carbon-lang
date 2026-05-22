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

inline constexpr llvm::StringLiteral CrtBegin = CRTBEGIN_SRC;
inline constexpr llvm::StringLiteral CrtEnd = CRTEND_SRC;

inline constexpr llvm::StringLiteral CrtCopts[] = {CRT_COPTS};

// Prevent wrapping these lines -- the expansion of the variables will add line
// breaks.
//
// clang-format off
inline constexpr llvm::StringLiteral BuiltinsAarch64Srcs[] = {BUILTINS_AARCH64_SRCS};
// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr llvm::StringLiteral BuiltinsX86_64Srcs[] = {BUILTINS_X86_64_SRCS};
inline constexpr llvm::StringLiteral BuiltinsI386Srcs[] = {BUILTINS_I386_SRCS};
// clang-format on

inline constexpr llvm::StringLiteral BuiltinsCopts[] = {BUILTINS_COPTS};

inline constexpr llvm::StringLiteral LibcxxLinuxSrcs[] = {LIBCXX_LINUX_SRCS};
inline constexpr llvm::StringLiteral LibcxxMacosSrcs[] = {LIBCXX_MACOS_SRCS};
inline constexpr llvm::StringLiteral LibcxxWin32Srcs[] = {LIBCXX_WIN32_SRCS};
inline constexpr llvm::StringLiteral LibcxxabiSrcs[] = {LIBCXXABI_SRCS};
inline constexpr llvm::StringLiteral LibcxxCopts[] = {LIBCXX_AND_ABI_COPTS};

inline constexpr llvm::StringLiteral LibunwindSrcs[] = {LIBUNWIND_SRCS};
inline constexpr llvm::StringLiteral LibunwindCopts[] = {LIBUNWIND_COPTS};

}  // namespace Carbon::RuntimesBuildInfo

#endif  // CARBON_TOOLCHAIN_BASE_RUNTIMES_BUILD_INFO_TPL_H_
