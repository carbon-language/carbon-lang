// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Never actually called, so this only tests typechecking and semantic analysis.
fn F(n: Int, p: Int*, q: Int***) -> Int* {
  var a: Int = n * *p;
  var b: Int = a*n;
  *p = b*(*p);
  **q = p;
  return **q;
}

fn main() -> Int {
  return 0;
}
