// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface A {}

// CHECK:STDERR: COMPILATION ERROR: explorer/testdata/constraint/fail_where_impls_non_constraint.carbon:[[@LINE+1]]: expected a constraint in expression after `impls`, found i32
alias B = A where i32 impls i32;

fn Main() -> i32 { return 0; }
