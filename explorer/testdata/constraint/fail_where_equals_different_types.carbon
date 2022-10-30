// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface A {}

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/fail_where_equals_different_types.carbon:[[@LINE+3]]: type mismatch between values in `where LHS == RHS`
// CHECK:STDERR:   LHS type: i32
// CHECK:STDERR:   RHS type: Type
alias B = A where 4 == i32;

fn Main() -> i32 { return 0; }
