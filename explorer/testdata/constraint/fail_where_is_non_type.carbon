// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface A {}

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/fail_where_is_non_type.carbon:[[@LINE+1]]: expression after `is` does not resolve to a constraint, found i32
alias B = A where i32 is 5;

fn Main() -> i32 { return 0; }
