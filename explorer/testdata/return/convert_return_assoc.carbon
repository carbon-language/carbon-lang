// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface I {
  let Assoc:! i32;
}

class C {
  impl Self as I where .Assoc = 1 {}
}

alias A = I where .Assoc == 1;

fn F[T:! A](x: T) -> i32 {
  return x.(A.Assoc);
}

fn Main() -> i32 {
  var a: C = {};
  return F(a);
}

// CHECK:STDOUT: result: 1
