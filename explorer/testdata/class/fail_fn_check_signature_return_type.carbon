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
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_fn_check_signature_return_type.carbon:[[@LINE+1]]: CheckExpressionEquality:4520 failed. kind mismatch
    impl fn foo[self: Self](i: i32) -> String { return ""; }
}

fn Main() -> i32 {
    return 0;
}
