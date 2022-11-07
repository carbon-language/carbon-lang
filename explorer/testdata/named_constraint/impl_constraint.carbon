// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 12

package ExplorerTest api;

interface Runnable {
  fn Run[me: Self]() -> i32;
}

interface Walkable {
  fn Walk[me: Self]() -> i32;
}

constraint Traversible {
  extends Runnable;
  extends Walkable;
}

external impl i32 as Traversible {
  fn Run[me: i32]() -> i32 {
    return 10 * me;
  }
  fn Walk[me: i32]() -> i32 {
    return me + 1;
  }
}

fn Main() -> i32 {
  var n: i32 = 1;
  return n.(Runnable.Run)() + n.(Walkable.Walk)();
}
