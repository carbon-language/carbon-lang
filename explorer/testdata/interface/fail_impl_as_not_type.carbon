// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface A {}

interface B {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/interface/fail_impl_as_not_type.carbon:[[@LINE+1]]: type error in type expression: 'i32' is not implicitly convertible to 'Type'
  impl 5 as A;
}

fn Main() -> i32 {
  return 0;
}
