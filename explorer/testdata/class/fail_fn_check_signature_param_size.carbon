// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

base class B {
    virtual fn foo[self: Self](i: i32) -> i32 { return i; }
}

class D extends B {
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_fn_check_signature_param_size.carbon:[[@LINE+3]]: type error in impl method
    // CHECK:STDERR: expected: fn (i32) -> i32
    // CHECK:STDERR: actual: fn (i32, i32) -> i32
    impl fn foo[self: Self](a: i32, b: i32) -> i32 { return a + b; }
}

fn Main() -> i32 {
    return 0;
}
