# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Starlark file exporting Carbon toolchain runtimes build info variables.

This file is a template that is expanded into a starlark file for the installed
Carbon toolchain that provides a trivial textual definition of the relevant
build info in variables.
"""

llvm_version_major = LLVM_VERSION_MAJOR

crtbegin_src = CRTBEGIN_SRC
crtend_src = CRTEND_SRC

crt_copts = [CRT_COPTS]

builtins_aarch64_srcs = [BUILTINS_AARCH64_SRCS]
builtins_x86_64_srcs = [BUILTINS_X86_64_SRCS]
builtins_i386_srcs = [BUILTINS_I386_SRCS]

builtins_aarch64_textual_srcs = [BUILTINS_AARCH64_TEXTUAL_SRCS]
builtins_x86_64_textual_srcs = [BUILTINS_X86_64_TEXTUAL_SRCS]
builtins_i386_textual_srcs = [BUILTINS_I386_TEXTUAL_SRCS]

builtins_copts = [BUILTINS_COPTS]

libcxx_hdrs = [LIBCXX_HDRS]
libcxx_linux_srcs = [LIBCXX_LINUX_SRCS]
libcxx_macos_srcs = [LIBCXX_MACOS_SRCS]
libcxx_win32_srcs = [LIBCXX_WIN32_SRCS]

libc_internal_libcxx_hdrs = [LIBCXX_SHARED_HEADERS_HDRS]

libcxxabi_hdrs = [LIBCXXABI_HDRS]
libcxxabi_srcs = [LIBCXXABI_SRCS]
libcxxabi_textual_srcs = [LIBCXXABI_TEXTUAL_SRCS]

libcxx_copts = [LIBCXX_AND_ABI_COPTS]

libunwind_hdrs = [LIBUNWIND_HDRS]
libunwind_srcs = [LIBUNWIND_SRCS]

libunwind_copts = [LIBUNWIND_COPTS]
