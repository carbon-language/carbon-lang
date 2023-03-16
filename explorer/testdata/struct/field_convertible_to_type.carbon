// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 3

package ExplorerTest api;

class TypeLike {
  impl as ImplicitAs(type) {
    fn Convert[self: Self]() -> type { return i32; }
  }
  fn Make() -> Self { return {}; }
}

fn Main() -> TypeLike.Make() {
  var x: {.a: TypeLike.Make(), .b: TypeLike.Make()} = {.a = 1, .b = 2};
  return x.a + x.b;
}
