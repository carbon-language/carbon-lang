// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

// CHECK:STDERR: COMPILATION ERROR: explorer/testdata/var/global/fail_invalid_var_expression.carbon:[[@LINE+1]]: Expected expression for variable type
var a: auto;

fn Main() -> i32 { return 0; }
