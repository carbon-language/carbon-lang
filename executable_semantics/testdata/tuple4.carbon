// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

fn main() -> i32 {
  var t: auto = (.x = 2, 3);
  return 0;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: PROGRAM ERROR: executable_semantics/testdata/tuple4.carbon:11: positional members must come before named members
