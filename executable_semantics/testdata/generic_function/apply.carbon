// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: result: -2

package ExecutableSemanticsTest api;

fn apply[T:! Type, U:! Type](f: __Fn (T) -> U, x: T) -> U {
  return f(x);
}

fn positive(x: Bool) -> i32 {
  if (x) {
    return 2;
  } else {
    return -2;
  }
}

fn Main() -> i32 {
  return apply(positive, false);
}
