// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class Cell(T:! Type) {
  fn Create(x: T) -> Cell(T) { return { .data = x }; }

  fn Get[me: Self]() -> T {
    return me.data;
  }
  fn Put[addr me: Self*](x: T) {
    (*me).data = x;
  }
  fn CreateOther[me: Self, U:! Type](x: U) -> Cell(U) {
    return {.data = x};
  }
  var data: T;
}

class Integer {
  var int: i32;
}

fn Main() -> i32 {
  var i: Integer = {.int = 1};
  var c: Cell(Integer) = Cell(Integer).Create(i);
  i = {.int = 0};
  c.Put(i);
  var j: Integer = c.Get();
  var d: Cell(Integer) = c.CreateOther(j);
  return c.data.int + d.data.int;
}
