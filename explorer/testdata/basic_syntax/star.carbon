// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

// Never actually called, so this only tests typechecking and semantic analysis.
fn F(n: i32, p: i32*, q: i32***) -> i32* {
  var a: i32 = n * *p;
  var b: i32 = a*n;
  *p = b*(*p);
  **q = p;
  return **q;
}

fn Main() -> i32 {
  return 0;
}
