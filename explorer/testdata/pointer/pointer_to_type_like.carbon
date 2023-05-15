// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 1

package ExplorerTest api;

class TypeLike {
  var v: type;
  impl as ImplicitAs(type) {
    fn Convert[self: Self]() -> type { return i32; }
  }
}

fn Almosti32() -> TypeLike { return {.v = i32}; }

fn Main() -> i32 {
  var a: Almosti32() = 1;
  var p: Almosti32()* = &a;
  return *p;
}
