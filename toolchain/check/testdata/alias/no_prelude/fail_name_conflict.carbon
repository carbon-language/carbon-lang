// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {}

alias a = C;
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE+7]]:5: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: var a: C = {};
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE-4]]:7: Name is previously declared here.
// CHECK:STDERR: alias a = C;
// CHECK:STDERR:       ^
// CHECK:STDERR:
var a: C = {};

var b: C = {};
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE+6]]:7: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: alias b = C;
// CHECK:STDERR:       ^
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE-4]]:5: Name is previously declared here.
// CHECK:STDERR: var b: C = {};
// CHECK:STDERR:     ^
alias b = C;

// CHECK:STDOUT: --- fail_name_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct: C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .a = %a.loc9
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref.loc9: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.loc9: type = bind_alias a, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %C.ref.loc17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.var: ref C = var a
// CHECK:STDOUT:   %a.loc17: ref C = bind_name a, %a.var
// CHECK:STDOUT:   %C.ref.loc19: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %b.var: ref C = var b
// CHECK:STDOUT:   %b: ref C = bind_name b, %b.var
// CHECK:STDOUT:   %C.ref.loc26: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc26: type = bind_alias <invalid>, %C.decl [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc17_13.2: init C = class_init (), file.%a.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc17_14: init C = converted %.loc17_13.1, %.loc17_13.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%a.var, %.loc17_14
// CHECK:STDOUT:   %.loc19_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc19_13.2: init C = class_init (), file.%b.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc19_14: init C = converted %.loc19_13.1, %.loc19_13.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%b.var, %.loc19_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
