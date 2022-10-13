// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/impl/fail_impl_as_not_constraint.carbon:[[@LINE+1]]: expected a constraint in impl declaration, found i32
external impl i32 as i32 {}

fn Main() -> i32 {
}
