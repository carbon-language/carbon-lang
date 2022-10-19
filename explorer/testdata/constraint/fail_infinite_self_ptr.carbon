// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/fail_infinite_self_ptr.carbon:[[@LINE+1]]: `.Self` used in type of non-type binding `T`
fn F[T:! .Self*]() {}

fn Main() -> i32 { return 0; }
