// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn F() {}
alias A = F;

// CHECK:STDERR: COMPILATION ERROR: explorer/testdata/alias/fail_wrong_fn_alias_use.carbon:[[@LINE+1]]: alias A cannot be used as a name qualifier
fn A.Foo() {}

fn Main() -> i32 {
    return 0;
}
