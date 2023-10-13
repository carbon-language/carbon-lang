// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_dereference_error.carbon:[[@LINE+3]]:15: ERROR: Name `undeclared` not found.
// CHECK:STDERR: let n: i32 = *undeclared;
// CHECK:STDERR:               ^
let n: i32 = *undeclared;

// CHECK:STDOUT: file "fail_dereference_error.carbon" {
// CHECK:STDOUT:   %undeclared.ref: <error> = name_reference "undeclared", <error>
// CHECK:STDOUT:   %.loc10: ref <error> = dereference <error>
// CHECK:STDOUT:   %n: i32 = bind_name "n", <error>
// CHECK:STDOUT: }
