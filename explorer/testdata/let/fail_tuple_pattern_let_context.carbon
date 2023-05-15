// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  let (var a: auto, b: auto, c: auto, d: auto) = (1, 2, 3, 4);
  a = 0;
  // should fail
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/let/fail_tuple_pattern_let_context.carbon:[[@LINE+1]]: Only a reference expression can be assigned to, but got `c`
  c = 0;
  return 0;
}
