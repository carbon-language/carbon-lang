// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface Vector {
  fn Zero() -> i32;
}
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/interface/fail_impl_not_type.carbon:[[@LINE+1]]: Expected a type, but got "hello"
impl "hello" as Vector {
  fn Zero() -> i32 { return 0; }
}

fn Main() -> i32 {
  return 0;
}
