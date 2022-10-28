// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: Carrot.G
// CHECK:STDOUT: result: 5

package ExplorerTest api;

interface HasFoo {
  fn Foo();
}

constraint ImplAsHasFoo {
  impl as HasFoo;
}

fn CallFoo[T:! ImplAsHasFoo](x: T) {
  // OK, T is HasFoo.
  x.(HasFoo.Foo)();
  x.Foo();
}

fn Main() -> i32 { return 0; }
