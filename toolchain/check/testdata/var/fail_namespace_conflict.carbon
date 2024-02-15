// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace A;

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+9]]:5: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: var A: b;
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-5]]:1: Name is previously declared here.
// CHECK:STDERR: namespace A;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+3]]:8: ERROR: Name `b` not found.
// CHECK:STDERR: var A: b;
// CHECK:STDERR:        ^
var A: b;

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+9]]:5: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: var A: b = 1;
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-16]]:1: Name is previously declared here.
// CHECK:STDERR: namespace A;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+3]]:8: ERROR: Name `b` not found.
// CHECK:STDERR: var A: b = 1;
// CHECK:STDERR:        ^
var A: b = 1;

// CHECK:STDOUT: --- fail_namespace_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.A = %.loc7} [template]
// CHECK:STDOUT:   %.loc7: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %b.ref.loc18: <error> = name_ref b, <error> [template = <error>]
// CHECK:STDOUT:   %A.var.loc18: ref <error> = var A
// CHECK:STDOUT:   %A.loc18: ref <error> = bind_name A, %A.var.loc18
// CHECK:STDOUT:   %b.ref.loc29: <error> = name_ref b, <error> [template = <error>]
// CHECK:STDOUT:   %A.var.loc29: ref <error> = var A
// CHECK:STDOUT:   %A.loc29: ref <error> = bind_name A, %A.var.loc29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc29: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   assign file.%A.var.loc29, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
