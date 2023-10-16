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
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s: {.a: i32, .b: i32});
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%cond: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_27: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc7: type = ptr_type {.a: i32, .b: i32}
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
// CHECK:STDOUT:   %G.ref: <function> = name_reference "G", package.%G
// CHECK:STDOUT:   %cond.ref: bool = name_reference "cond", %cond
// CHECK:STDOUT:   if %cond.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %a.ref.loc11_18: ref {.a: i32, .b: i32} = name_reference "a", %a
// CHECK:STDOUT:   %.loc11_18.1: ref i32 = struct_access %a.ref.loc11_18, member0
// CHECK:STDOUT:   %.loc11_18.2: i32 = bind_value %.loc11_18.1
// CHECK:STDOUT:   %.loc11_18.3: ref i32 = struct_access %a.ref.loc11_18, member1
// CHECK:STDOUT:   %.loc11_18.4: i32 = bind_value %.loc11_18.3
// CHECK:STDOUT:   %.loc11_18.5: {.a: i32, .b: i32} = struct_value %a.ref.loc11_18, (%.loc11_18.2, %.loc11_18.4)
// CHECK:STDOUT:   br !if.expr.result(%.loc11_18.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc11_25: ref {.a: i32, .b: i32} = name_reference "a", %a
// CHECK:STDOUT:   %.loc11_25.1: ref i32 = struct_access %a.ref.loc11_25, member0
// CHECK:STDOUT:   %.loc11_25.2: i32 = bind_value %.loc11_25.1
// CHECK:STDOUT:   %.loc11_25.3: ref i32 = struct_access %a.ref.loc11_25, member1
// CHECK:STDOUT:   %.loc11_25.4: i32 = bind_value %.loc11_25.3
// CHECK:STDOUT:   %.loc11_25.5: {.a: i32, .b: i32} = struct_value %a.ref.loc11_25, (%.loc11_25.2, %.loc11_25.4)
// CHECK:STDOUT:   br !if.expr.result(%.loc11_25.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc11_5: {.a: i32, .b: i32} = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc11_4.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc11_4.2: init () = call %G.ref(%.loc11_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
