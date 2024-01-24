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

// CHECK:STDOUT: --- struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.2: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: {.a: i32, .b: i32} = struct_value (%.3, %.4) [template]
// CHECK:STDOUT:   %.6: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.G = %G, .F = %F} [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s: {.a: i32, .b: i32});
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%cond: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref {.a: i32, .b: i32} = var a
// CHECK:STDOUT:   %a: ref {.a: i32, .b: i32} = bind_name a, %a.var
// CHECK:STDOUT:   %.loc10_37: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_45: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_46.1: {.a: i32, .b: i32} = struct_literal (%.loc10_37, %.loc10_45)
// CHECK:STDOUT:   %.loc10_46.2: ref i32 = struct_access %a.var, element0
// CHECK:STDOUT:   %.loc10_46.3: init i32 = initialize_from %.loc10_37 to %.loc10_46.2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_46.4: ref i32 = struct_access %a.var, element1
// CHECK:STDOUT:   %.loc10_46.5: init i32 = initialize_from %.loc10_45 to %.loc10_46.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_46.6: init {.a: i32, .b: i32} = struct_init (%.loc10_46.3, %.loc10_46.5) to %a.var [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_46.7: init {.a: i32, .b: i32} = converted %.loc10_46.1, %.loc10_46.6 [template = constants.%.5]
// CHECK:STDOUT:   assign %a.var, %.loc10_46.7
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, file.%G [template = file.%G]
// CHECK:STDOUT:   %cond.ref: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %a.ref.loc11_18: ref {.a: i32, .b: i32} = name_ref a, %a
// CHECK:STDOUT:   %.loc11_18.1: ref i32 = struct_access %a.ref.loc11_18, element0
// CHECK:STDOUT:   %.loc11_18.2: i32 = bind_value %.loc11_18.1
// CHECK:STDOUT:   %.loc11_18.3: ref i32 = struct_access %a.ref.loc11_18, element1
// CHECK:STDOUT:   %.loc11_18.4: i32 = bind_value %.loc11_18.3
// CHECK:STDOUT:   %.loc11_18.5: {.a: i32, .b: i32} = struct_value (%.loc11_18.2, %.loc11_18.4)
// CHECK:STDOUT:   %.loc11_18.6: {.a: i32, .b: i32} = converted %a.ref.loc11_18, %.loc11_18.5
// CHECK:STDOUT:   br !if.expr.result(%.loc11_18.6)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc11_25: ref {.a: i32, .b: i32} = name_ref a, %a
// CHECK:STDOUT:   %.loc11_25.1: ref i32 = struct_access %a.ref.loc11_25, element0
// CHECK:STDOUT:   %.loc11_25.2: i32 = bind_value %.loc11_25.1
// CHECK:STDOUT:   %.loc11_25.3: ref i32 = struct_access %a.ref.loc11_25, element1
// CHECK:STDOUT:   %.loc11_25.4: i32 = bind_value %.loc11_25.3
// CHECK:STDOUT:   %.loc11_25.5: {.a: i32, .b: i32} = struct_value (%.loc11_25.2, %.loc11_25.4)
// CHECK:STDOUT:   %.loc11_25.6: {.a: i32, .b: i32} = converted %a.ref.loc11_25, %.loc11_25.5
// CHECK:STDOUT:   br !if.expr.result(%.loc11_25.6)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc11_5: {.a: i32, .b: i32} = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc11_4: init () = call %G.ref(%.loc11_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
