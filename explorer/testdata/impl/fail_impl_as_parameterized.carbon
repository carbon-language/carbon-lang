// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface Vector(Scalar:! Type) {
}

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/impl/fail_impl_as_parameterized.carbon:[[@LINE+1]]: 'Vector' must be given an argument list
external impl i32 as Vector {}

fn Main() -> i32 {
}
