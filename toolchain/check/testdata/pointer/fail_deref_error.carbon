// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_deref_error.carbon:[[@LINE+4]]:15: ERROR: Name `undeclared` not found.
// CHECK:STDERR: let n: i32 = *undeclared;
// CHECK:STDERR:               ^~~~~~~~~~
// CHECK:STDERR:
let n: i32 = *undeclared;
// CHECK:STDERR: fail_deref_error.carbon:[[@LINE+3]]:15: ERROR: Name `undeclared` not found.
// CHECK:STDERR: let n2: i32 = undeclared->foo;
// CHECK:STDERR:               ^~~~~~~~~~
let n2: i32 = undeclared->foo;

// CHECK:STDOUT: --- fail_deref_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %undeclared.ref.loc11: <error> = name_ref undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc11: ref <error> = deref <error>
// CHECK:STDOUT:   %n: i32 = bind_name n, <error>
// CHECK:STDOUT:   %undeclared.ref.loc15: <error> = name_ref undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc15: ref <error> = deref <error>
// CHECK:STDOUT:   %n2: i32 = bind_name n2, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
