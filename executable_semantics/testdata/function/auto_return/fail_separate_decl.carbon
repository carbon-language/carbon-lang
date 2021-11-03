// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/function/auto_return/fail_separate_decl.carbon:15: syntax error, unexpected SEMICOLON, expecting LEFT_CURLY_BRACE

package ExecutableSemanticsTest api;

// This declaration is not allowed.
fn Add(x: i32, y: i32) -> auto;

fn Add(x: i32, y: i32) -> auto { return x + y; }

fn Main() -> i32 {
  return Add(1, 2) - 3;
}
