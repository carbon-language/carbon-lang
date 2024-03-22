// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  returned var v: i32 = 0;
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE+7]]:3: ERROR: Can only `return var;` in the scope of a `returned var`.
  // CHECK:STDERR:   return 1;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE-4]]:16: `returned var` was declared here.
  // CHECK:STDERR:   returned var v: i32 = 0;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  return 1;
}

class C { var a: i32; var b: i32; }
fn G() -> C {
  returned var c: C = {.a = 1, .b = 2};
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE+6]]:3: ERROR: Can only `return var;` in the scope of a `returned var`.
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE-4]]:16: `returned var` was declared here.
  // CHECK:STDERR:   returned var c: C = {.a = 1, .b = 2};
  // CHECK:STDERR:                ^
  return c;
}

// CHECK:STDOUT: --- fail_return_with_returned_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type C, i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: C = struct_value (%.2, %.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @G.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc19_16: <unbound element of class C> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc19_28: <unbound element of class C> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc19_16
// CHECK:STDOUT:   .b = %.loc19_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %v.var, %.loc8
// CHECK:STDOUT:   %.loc16: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c: ref C = bind_name c, %return
// CHECK:STDOUT:   %.loc21_29: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc21_37: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc21_38.1: {.a: i32, .b: i32} = struct_literal (%.loc21_29, %.loc21_37)
// CHECK:STDOUT:   %.loc21_38.2: ref i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc21_38.3: init i32 = initialize_from %.loc21_29 to %.loc21_38.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc21_38.4: ref i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc21_38.5: init i32 = initialize_from %.loc21_37 to %.loc21_38.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc21_38.6: init C = class_init (%.loc21_38.3, %.loc21_38.5), %return [template = constants.%.7]
// CHECK:STDOUT:   %.loc21_38.7: init C = converted %.loc21_38.1, %.loc21_38.6 [template = constants.%.7]
// CHECK:STDOUT:   assign %return, %.loc21_38.7
// CHECK:STDOUT:   %c.ref: ref C = name_ref c, %c
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
