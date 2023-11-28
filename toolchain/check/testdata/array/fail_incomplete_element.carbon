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
// CHECK:STDERR: ^~~~~
var a: [Incomplete; 1];

// CHECK:STDERR: fail_incomplete_element.carbon:[[@LINE+3]]:1: ERROR: Cannot implicitly convert from `<error>*` to `Incomplete*`.
// CHECK:STDERR: var p: Incomplete* = &a[0];
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
var p: Incomplete* = &a[0];

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc20: type = ptr_type <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_incomplete_element.carbon" {
// CHECK:STDOUT:   class_decl @Incomplete, ()
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete
// CHECK:STDOUT:   %Incomplete.ref.loc15: type = name_ref Incomplete, %Incomplete
// CHECK:STDOUT:   %.loc15_21: i32 = int_literal 1
// CHECK:STDOUT:   %.loc15_22: type = array_type %.loc15_21, Incomplete
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT:   %Incomplete.ref.loc20: type = name_ref Incomplete, %Incomplete
// CHECK:STDOUT:   %.loc20_18: type = ptr_type Incomplete
// CHECK:STDOUT:   %p.var: ref Incomplete* = var p
// CHECK:STDOUT:   %p: ref Incomplete* = bind_name p, %p.var
// CHECK:STDOUT:   %a.ref: ref <error> = name_ref a, %a
// CHECK:STDOUT:   %.loc20_25: i32 = int_literal 0
// CHECK:STDOUT:   %.loc20_22: <error>* = address_of <error>
// CHECK:STDOUT:   assign %p.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
