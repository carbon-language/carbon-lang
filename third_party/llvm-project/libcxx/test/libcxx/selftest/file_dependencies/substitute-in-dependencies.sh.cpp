//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that lit substitutions are expanded inside FILE_DEPENDENCIES lines.

// FILE_DEPENDENCIES: %s
// RUN: test -e %T/substitute-in-dependencies.sh.cpp
