// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}

package ExplorerTest api;

abstract class A {
    abstract fn Foo[self: Self]();
}

fn Main() -> i32 {
  return 0;
}

// CHECK:STDOUT: result: 0
