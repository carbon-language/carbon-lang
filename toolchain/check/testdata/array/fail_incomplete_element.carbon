// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Incomplete;

// CHECK:STDERR: fail_incomplete_element.carbon:[[@LINE+6]]:8: ERROR: Variable has incomplete type `[Incomplete; 1]`.
// CHECK:STDERR: var a: [Incomplete; 1];
// CHECK:STDERR:        ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_incomplete_element.carbon:[[@LINE-5]]:1: Class was forward declared here.
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
var a: [Incomplete; 1];

var p: Incomplete* = &a[0];

// CHECK:STDOUT: --- fail_incomplete_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, Incomplete [template]
// CHECK:STDOUT:   %.3: type = ptr_type Incomplete [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Incomplete = %Incomplete.decl, .a = %a, .p = %p} [template]
// CHECK:STDOUT:   %Incomplete.decl = class_decl @Incomplete, ()
// CHECK:STDOUT:   %Incomplete.ref.loc15: type = name_ref Incomplete, constants.%Incomplete [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc15_21: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc15_22: type = array_type %.loc15_21, Incomplete [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT:   %Incomplete.ref.loc17: type = name_ref Incomplete, constants.%Incomplete [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc17_18: type = ptr_type Incomplete [template = constants.%.3]
// CHECK:STDOUT:   %p.var: ref Incomplete* = var p
// CHECK:STDOUT:   %p: ref Incomplete* = bind_name p, %p.var
// CHECK:STDOUT:   %a.ref: ref <error> = name_ref a, %a
// CHECK:STDOUT:   %.loc17_25: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc17_22: <error> = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %p.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
