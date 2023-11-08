// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  var a: i32;
  var b: i32;
}

fn F() -> C {
  returned var result: C = {.a = 1, .b = 2};
  // TODO: return var;
  // CHECK:STDERR: returned_var.carbon:[[@LINE+3]]:10: ERROR: Cannot copy value of type `C`.
  // CHECK:STDERR:   return result;
  // CHECK:STDERR:          ^
  return result;
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_1.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc10_1.2: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "returned_var.carbon" {
// CHECK:STDOUT:   class_declaration @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_field_type C, i32
// CHECK:STDOUT:   %.loc8_8.2: <unbound field of class C> = field "a", member0
// CHECK:STDOUT:   %a: <unbound field of class C> = bind_name "a", %.loc8_8.2
// CHECK:STDOUT:   %.loc9_8.1: type = unbound_field_type C, i32
// CHECK:STDOUT:   %.loc9_8.2: <unbound field of class C> = field "b", member1
// CHECK:STDOUT:   %b: <unbound field of class C> = bind_name "b", %.loc9_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .b = %b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_reference "C", file.%C
// CHECK:STDOUT:   %result: ref C = bind_name "result", %return
// CHECK:STDOUT:   %.loc13_34: i32 = int_literal 1
// CHECK:STDOUT:   %.loc13_42: i32 = int_literal 2
// CHECK:STDOUT:   %.loc13_43.1: {.a: i32, .b: i32} = struct_literal (%.loc13_34, %.loc13_42)
// CHECK:STDOUT:   %.loc13_43.2: ref i32 = class_field_access %return, member0
// CHECK:STDOUT:   %.loc13_43.3: init i32 = initialize_from %.loc13_34 to %.loc13_43.2
// CHECK:STDOUT:   %.loc13_43.4: ref i32 = class_field_access %return, member1
// CHECK:STDOUT:   %.loc13_43.5: init i32 = initialize_from %.loc13_42 to %.loc13_43.4
// CHECK:STDOUT:   %.loc13_43.6: init C = class_init %.loc13_43.1, (%.loc13_43.3, %.loc13_43.5)
// CHECK:STDOUT:   assign %return, %.loc13_43.6
// CHECK:STDOUT:   %result.ref: ref C = name_reference "result", %result
// CHECK:STDOUT:   %.loc18: C = bind_value %result.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
