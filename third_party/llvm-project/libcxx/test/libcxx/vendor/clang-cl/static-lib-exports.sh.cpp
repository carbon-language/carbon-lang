//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: msvc

// This file checks that the built static library doesn't contain dllexport
// directives in clang-cl builds.

// RUN: llvm-readobj --coff-directives "%{lib}/libc++.lib" | not grep -i "export:" > /dev/null
