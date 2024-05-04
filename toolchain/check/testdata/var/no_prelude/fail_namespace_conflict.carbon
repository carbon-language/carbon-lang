// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace A;

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+7]]:5: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: var A: ();
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-5]]:1: Name is previously declared here.
// CHECK:STDERR: namespace A;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
var A: ();

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+6]]:5: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: var A: () = ();
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-14]]:1: Name is previously declared here.
// CHECK:STDERR: namespace A;
// CHECK:STDERR: ^~~~~~~~~~~~
var A: () = ();

// CHECK:STDOUT: --- fail_namespace_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc7: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc16_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc16_9.2: type = converted %.loc16_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %A.var.loc16: ref () = var A
// CHECK:STDOUT:   %A.loc16: ref () = bind_name A, %A.var.loc16
// CHECK:STDOUT:   %.loc24_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc24_9.2: type = converted %.loc24_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %A.var.loc24: ref () = var A
// CHECK:STDOUT:   %A.loc24: ref () = bind_name A, %A.var.loc24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_14.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc24_14.2: init () = tuple_init () to file.%A.var.loc24 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc24_15: init () = converted %.loc24_14.1, %.loc24_14.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%A.var.loc24, %.loc24_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
