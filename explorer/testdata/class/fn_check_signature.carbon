// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class B {
    virtual fn foo[self: Self](i: i32) -> i32 { return i; }
}

class D1 extends B {
    impl fn foo[self: Self](i: i32) -> i32 { return 0; }
}

fn Main() -> i32 {
    return 0;
}
