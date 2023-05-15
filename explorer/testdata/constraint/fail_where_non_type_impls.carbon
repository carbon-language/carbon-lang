// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface A {}

// CHECK:STDERR: COMPILATION ERROR: explorer/testdata/constraint/fail_where_non_type_impls.carbon:[[@LINE+1]]: type error in type expression: 'i32' is not implicitly convertible to 'type'
alias B = A where 4 impls A;

fn Main() -> i32 { return 0; }
