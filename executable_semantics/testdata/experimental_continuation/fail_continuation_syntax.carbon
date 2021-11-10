// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/experimental_continuation/fail_continuation_syntax.carbon:16: syntax error, unexpected identifier, expecting LEFT_CURLY_BRACE

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  var x: i32 = 0;
  __continuation k x = 3;
  __run k;
  return x;
}
