// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

�

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: executable_semantics/testdata/invalid_char.carbon:8: invalid character '\xEF' in source file.
