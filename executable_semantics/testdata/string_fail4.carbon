// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

fn main() -> i32 {
  Print("str\01");
  return 0;
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: COMPILATION ERROR: executable_semantics/testdata/string_fail4.carbon:11: Invalid escaping in string: "str\01"
