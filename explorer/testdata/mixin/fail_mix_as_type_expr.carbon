// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

__mixin Operations {}

fn Main() -> i32 {
   // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_mix_as_type_expr.carbon:[[@LINE+1]]: Expected a type, but got mixin Operations
   var a: Operations;
   return 0;
}
