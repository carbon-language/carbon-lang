// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface FrobWith(T:! Type) {}

fn F[T:! FrobWith(.Self),
     // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/fail_dot_self_after_scope_2.carbon:[[@LINE+1]]: `.Self` used in type of non-type binding `U`
     U:! .Self]() {
}

fn Main() -> i32 {
  return 0;
}
