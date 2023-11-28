// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  returned var v: i32 = 0;
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE+6]]:3: ERROR: Can only `return var;` in the scope of a `returned var`.
  // CHECK:STDERR:   return 1;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE-4]]:16: `returned var` was declared here.
  // CHECK:STDERR:   returned var v: i32 = 0;
  // CHECK:STDERR:                ^
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

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc18_35.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc18_35.2: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_return_with_returned_var.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc18_16.1: type = unbound_field_type C, i32
// CHECK:STDOUT:   %.loc18_16.2: <unbound field of class C> = field a, member0
// CHECK:STDOUT:   %a: <unbound field of class C> = bind_name a, %.loc18_16.2
// CHECK:STDOUT:   %.loc18_28.1: type = unbound_field_type C, i32
// CHECK:STDOUT:   %.loc18_28.2: <unbound field of class C> = field b, member1
// CHECK:STDOUT:   %b: <unbound field of class C> = bind_name b, %.loc18_28.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .b = %b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0
// CHECK:STDOUT:   assign %v.var, %.loc8
// CHECK:STDOUT:   %.loc15: i32 = int_literal 1
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C
// CHECK:STDOUT:   %c: ref C = bind_name c, %return
// CHECK:STDOUT:   %.loc20_29: i32 = int_literal 1
// CHECK:STDOUT:   %.loc20_37: i32 = int_literal 2
// CHECK:STDOUT:   %.loc20_38.1: {.a: i32, .b: i32} = struct_literal (%.loc20_29, %.loc20_37)
// CHECK:STDOUT:   %.loc20_38.2: ref i32 = class_field_access %return, member0
// CHECK:STDOUT:   %.loc20_38.3: init i32 = initialize_from %.loc20_29 to %.loc20_38.2
// CHECK:STDOUT:   %.loc20_38.4: ref i32 = class_field_access %return, member1
// CHECK:STDOUT:   %.loc20_38.5: init i32 = initialize_from %.loc20_37 to %.loc20_38.4
// CHECK:STDOUT:   %.loc20_38.6: init C = class_init (%.loc20_38.3, %.loc20_38.5), %return
// CHECK:STDOUT:   %.loc20_38.7: init C = converted %.loc20_38.1, %.loc20_38.6
// CHECK:STDOUT:   assign %return, %.loc20_38.7
// CHECK:STDOUT:   %c.ref: ref C = name_ref c, %c
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
