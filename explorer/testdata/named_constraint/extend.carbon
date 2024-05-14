// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface Maker(T:! type) {
  fn Make[self: Self]() -> T;
}

constraint IndirectMaker(T:! type) {
  extend Maker(T*);
}

constraint MoreIndirectMaker {
  extend IndirectMaker(i32);
}

class PointerFactory {
  var p: i32*;
  impl as Maker(i32*) {
    fn Make[self: Self]() -> i32* { return self.p; }
  }
}

fn CallIndirect[T:! IndirectMaker(i32)](x: T) -> i32 {
  return *x.Make();
}

fn CallMoreIndirect[T:! MoreIndirectMaker](x: T) -> i32 {
  return *x.Make();
}

fn Main() -> i32 {
  var n: i32 = 5;
  var f: PointerFactory = {.p = &n};
  Print("{0}", *f.(Maker(i32*).Make)());
  Print("{0}", *f.(IndirectMaker(i32).Make)());
  Print("{0}", *f.(MoreIndirectMaker.Make)());
  Print("{0}", CallIndirect(f));
  Print("{0}", CallMoreIndirect(f));
  return 0;
}

// CHECK:STDOUT: 5
// CHECK:STDOUT: 5
// CHECK:STDOUT: 5
// CHECK:STDOUT: 5
// CHECK:STDOUT: 5
// CHECK:STDOUT: result: 0
