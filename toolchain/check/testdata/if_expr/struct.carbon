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
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: G = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct.3: {.a: i32, .b: i32} = struct_value (%.4, %.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.1] {
// CHECK:STDOUT:     %.loc7: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:     %s.loc7_6.1: {.a: i32, .b: i32} = param s
// CHECK:STDOUT:     @G.%s: {.a: i32, .b: i32} = bind_name s, %s.loc7_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %cond.loc9_6.1: bool = param cond
// CHECK:STDOUT:     @F.%cond: bool = bind_name cond, %cond.loc9_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s: {.a: i32, .b: i32});
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%cond: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref {.a: i32, .b: i32} = var a
// CHECK:STDOUT:   %a: ref {.a: i32, .b: i32} = bind_name a, %a.var
// CHECK:STDOUT:   %.loc10_37: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_45: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_46.1: {.a: i32, .b: i32} = struct_literal (%.loc10_37, %.loc10_45)
// CHECK:STDOUT:   %.loc10_46.2: ref i32 = struct_access %a.var, element0
// CHECK:STDOUT:   %.loc10_46.3: init i32 = initialize_from %.loc10_37 to %.loc10_46.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_46.4: ref i32 = struct_access %a.var, element1
// CHECK:STDOUT:   %.loc10_46.5: init i32 = initialize_from %.loc10_45 to %.loc10_46.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_46.6: init {.a: i32, .b: i32} = struct_init (%.loc10_46.3, %.loc10_46.5) to %a.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc10_47: init {.a: i32, .b: i32} = converted %.loc10_46.1, %.loc10_46.6 [template = constants.%struct.3]
// CHECK:STDOUT:   assign %a.var, %.loc10_47
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %cond.ref: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %a.ref.loc11_18: ref {.a: i32, .b: i32} = name_ref a, %a
// CHECK:STDOUT:   %.loc11_18.1: ref i32 = struct_access %a.ref.loc11_18, element0
// CHECK:STDOUT:   %.loc11_18.2: i32 = bind_value %.loc11_18.1
// CHECK:STDOUT:   %.loc11_18.3: ref i32 = struct_access %a.ref.loc11_18, element1
// CHECK:STDOUT:   %.loc11_18.4: i32 = bind_value %.loc11_18.3
// CHECK:STDOUT:   %struct.loc11_18: {.a: i32, .b: i32} = struct_value (%.loc11_18.2, %.loc11_18.4)
// CHECK:STDOUT:   %.loc11_18.5: {.a: i32, .b: i32} = converted %a.ref.loc11_18, %struct.loc11_18
// CHECK:STDOUT:   br !if.expr.result(%.loc11_18.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc11_25: ref {.a: i32, .b: i32} = name_ref a, %a
// CHECK:STDOUT:   %.loc11_25.1: ref i32 = struct_access %a.ref.loc11_25, element0
// CHECK:STDOUT:   %.loc11_25.2: i32 = bind_value %.loc11_25.1
// CHECK:STDOUT:   %.loc11_25.3: ref i32 = struct_access %a.ref.loc11_25, element1
// CHECK:STDOUT:   %.loc11_25.4: i32 = bind_value %.loc11_25.3
// CHECK:STDOUT:   %struct.loc11_25: {.a: i32, .b: i32} = struct_value (%.loc11_25.2, %.loc11_25.4)
// CHECK:STDOUT:   %.loc11_20: {.a: i32, .b: i32} = converted %a.ref.loc11_25, %struct.loc11_25
// CHECK:STDOUT:   br !if.expr.result(%.loc11_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc11_5: {.a: i32, .b: i32} = block_arg !if.expr.result
// CHECK:STDOUT:   %G.call: init () = call %G.ref(%.loc11_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
