// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(s: {.a: i32, .b: i32});

fn F(cond: bool) {
  var a: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  G(if cond then a else a);
}

// CHECK:STDOUT: file "struct.carbon" {
// CHECK:STDOUT:   %G = fn_decl @G
// CHECK:STDOUT:   %F = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s: {.a: i32, .b: i32});
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%cond: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_27: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %a: ref {.a: i32, .b: i32} = var "a"
// CHECK:STDOUT:   %.loc10_37: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_45: i32 = int_literal 2
// CHECK:STDOUT:   %.loc10_46.1: {.a: i32, .b: i32} = struct_literal (%.loc10_37, %.loc10_45)
// CHECK:STDOUT:   %.loc10_46.2: ref i32 = struct_access %a, member0
// CHECK:STDOUT:   %.loc10_46.3: init i32 = initialize_from %.loc10_37 to %.loc10_46.2
// CHECK:STDOUT:   %.loc10_46.4: ref i32 = struct_access %a, member1
// CHECK:STDOUT:   %.loc10_46.5: init i32 = initialize_from %.loc10_45 to %.loc10_46.4
// CHECK:STDOUT:   %.loc10_46.6: init {.a: i32, .b: i32} = struct_init %.loc10_46.1, (%.loc10_46.3, %.loc10_46.5)
// CHECK:STDOUT:   assign %a, %.loc10_46.6
// CHECK:STDOUT:   if %cond br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %.loc10_7.1: ref i32 = struct_access %a, member0
// CHECK:STDOUT:   %.loc10_7.2: i32 = bind_value %.loc10_7.1
// CHECK:STDOUT:   %.loc10_7.3: ref i32 = struct_access %a, member1
// CHECK:STDOUT:   %.loc10_7.4: i32 = bind_value %.loc10_7.3
// CHECK:STDOUT:   %.loc10_7.5: {.a: i32, .b: i32} = struct_value %a, (%.loc10_7.2, %.loc10_7.4)
// CHECK:STDOUT:   br !if.expr.result(%.loc10_7.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc10_7.6: ref i32 = struct_access %a, member0
// CHECK:STDOUT:   %.loc10_7.7: i32 = bind_value %.loc10_7.6
// CHECK:STDOUT:   %.loc10_7.8: ref i32 = struct_access %a, member1
// CHECK:STDOUT:   %.loc10_7.9: i32 = bind_value %.loc10_7.8
// CHECK:STDOUT:   %.loc10_7.10: {.a: i32, .b: i32} = struct_value %a, (%.loc10_7.7, %.loc10_7.9)
// CHECK:STDOUT:   br !if.expr.result(%.loc10_7.10)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc11_5: {.a: i32, .b: i32} = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc11_4.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc11_4.2: init () = call @G(%.loc11_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
