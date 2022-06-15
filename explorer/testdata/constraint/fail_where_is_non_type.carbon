// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

interface A {}

// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/fail_where_is_non_type.carbon:[[@LINE+1]]: expression after `is` does not resolve to a constraint, found i32
alias B = A where i32 is 5;

fn Main() -> i32 { return 0; }
