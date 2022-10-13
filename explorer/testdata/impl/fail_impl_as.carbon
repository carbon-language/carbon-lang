// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/impl/fail_impl_as.carbon:[[@LINE+1]]: expected constraint after `as`, found value of type i32
external impl i32 as i32 {}

fn Main() -> i32 {
}
