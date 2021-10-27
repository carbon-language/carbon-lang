// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: split-file %s %t
// RUN: executable_semantics --unparse %t/a.carbon 2>&1 > %t.a.unparsed
// RUN: executable_semantics --unparse %t/b.carbon 2>&1 | diff %t.a.unparsed -

//--- a.carbon

package ExecutableSemanticsTest api;

fn main() -> i32 {
  return 3 * if cond then 1 + 1 else 2 + 4 * 6;
}

//--- b.carbon

package ExecutableSemanticsTest api;

fn main() -> i32 {
  return 3 * (if cond then (1 + 1) else (2 + (4 * 6)));
}
