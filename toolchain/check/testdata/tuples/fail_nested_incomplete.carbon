// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Incomplete;

// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE+6]]:8: ERROR: Variable has incomplete type `(i32, Incomplete)`.
// CHECK:STDERR: var t: (i32, Incomplete);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE-5]]:1: Class was forward declared here.
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
var t: (i32, Incomplete);

var p: Incomplete* = &t[1];

// CHECK:STDOUT: --- fail_nested_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, Incomplete) [template]
// CHECK:STDOUT:   %.3: type = ptr_type Incomplete [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Incomplete = %Incomplete.decl, .t = %t, .p = %p} [template]
// CHECK:STDOUT:   %Incomplete.decl = class_decl @Incomplete, ()
// CHECK:STDOUT:   %Incomplete.ref.loc15: type = name_ref Incomplete, constants.%Incomplete [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc15_24.1: (type, type) = tuple_literal (i32, %Incomplete.ref.loc15)
// CHECK:STDOUT:   %.loc15_24.2: type = converted %.loc15_24.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %t.var: ref <error> = var t
// CHECK:STDOUT:   %t: ref <error> = bind_name t, %t.var
// CHECK:STDOUT:   %Incomplete.ref.loc17: type = name_ref Incomplete, constants.%Incomplete [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc17_18: type = ptr_type Incomplete [template = constants.%.3]
// CHECK:STDOUT:   %p.var: ref Incomplete* = var p
// CHECK:STDOUT:   %p: ref Incomplete* = bind_name p, %p.var
// CHECK:STDOUT:   %t.ref: ref <error> = name_ref t, %t
// CHECK:STDOUT:   %.loc17_25: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc17_22: <error> = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %p.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
