// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/struct.carbon

fn G(s: {.a: i32, .b: i32});

fn F(cond: bool) {
  var a: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  G(if cond then a else a);
}

// CHECK:STDOUT: --- struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %Bool: type = fn_type @Bool [template]
// CHECK:STDOUT:   %struct.3: Bool = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.4: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct.5: {.a: i32, .b: i32} = struct_value (%.4, %.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc11_14: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_14.1: type = value_of_initializer %int.make_type_32.loc11_14 [template = i32]
// CHECK:STDOUT:     %.loc11_14.2: type = converted %int.make_type_32.loc11_14, %.loc11_14.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_23: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_23.1: type = value_of_initializer %int.make_type_32.loc11_23 [template = i32]
// CHECK:STDOUT:     %.loc11_23.2: type = converted %int.make_type_32.loc11_23, %.loc11_23.1 [template = i32]
// CHECK:STDOUT:     %.loc11_26: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:     %s.loc11_6.1: {.a: i32, .b: i32} = param s
// CHECK:STDOUT:     @G.%s: {.a: i32, .b: i32} = bind_name s, %s.loc11_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.4] {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%struct.3() [template = bool]
// CHECK:STDOUT:     %.loc13_12.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc13_12.2: type = converted %bool.make_type, %.loc13_12.1 [template = bool]
// CHECK:STDOUT:     %cond.loc13_6.1: bool = param cond
// CHECK:STDOUT:     @F.%cond: bool = bind_name cond, %cond.loc13_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s: {.a: i32, .b: i32});
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%cond: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc14_15: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc14_15.1: type = value_of_initializer %int.make_type_32.loc14_15 [template = i32]
// CHECK:STDOUT:   %.loc14_15.2: type = converted %int.make_type_32.loc14_15, %.loc14_15.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_24: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc14_24.1: type = value_of_initializer %int.make_type_32.loc14_24 [template = i32]
// CHECK:STDOUT:   %.loc14_24.2: type = converted %int.make_type_32.loc14_24, %.loc14_24.1 [template = i32]
// CHECK:STDOUT:   %.loc14_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref {.a: i32, .b: i32} = var a
// CHECK:STDOUT:   %a: ref {.a: i32, .b: i32} = bind_name a, %a.var
// CHECK:STDOUT:   %.loc14_37: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_45: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_46.1: {.a: i32, .b: i32} = struct_literal (%.loc14_37, %.loc14_45)
// CHECK:STDOUT:   %.loc14_46.2: ref i32 = struct_access %a.var, element0
// CHECK:STDOUT:   %.loc14_46.3: init i32 = initialize_from %.loc14_37 to %.loc14_46.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_46.4: ref i32 = struct_access %a.var, element1
// CHECK:STDOUT:   %.loc14_46.5: init i32 = initialize_from %.loc14_45 to %.loc14_46.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_46.6: init {.a: i32, .b: i32} = struct_init (%.loc14_46.3, %.loc14_46.5) to %a.var [template = constants.%struct.5]
// CHECK:STDOUT:   %.loc14_47: init {.a: i32, .b: i32} = converted %.loc14_46.1, %.loc14_46.6 [template = constants.%struct.5]
// CHECK:STDOUT:   assign %a.var, %.loc14_47
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %cond.ref: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %a.ref.loc15_18: ref {.a: i32, .b: i32} = name_ref a, %a
// CHECK:STDOUT:   %.loc15_18.1: ref i32 = struct_access %a.ref.loc15_18, element0
// CHECK:STDOUT:   %.loc15_18.2: i32 = bind_value %.loc15_18.1
// CHECK:STDOUT:   %.loc15_18.3: ref i32 = struct_access %a.ref.loc15_18, element1
// CHECK:STDOUT:   %.loc15_18.4: i32 = bind_value %.loc15_18.3
// CHECK:STDOUT:   %struct.loc15_18: {.a: i32, .b: i32} = struct_value (%.loc15_18.2, %.loc15_18.4)
// CHECK:STDOUT:   %.loc15_18.5: {.a: i32, .b: i32} = converted %a.ref.loc15_18, %struct.loc15_18
// CHECK:STDOUT:   br !if.expr.result(%.loc15_18.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc15_25: ref {.a: i32, .b: i32} = name_ref a, %a
// CHECK:STDOUT:   %.loc15_25.1: ref i32 = struct_access %a.ref.loc15_25, element0
// CHECK:STDOUT:   %.loc15_25.2: i32 = bind_value %.loc15_25.1
// CHECK:STDOUT:   %.loc15_25.3: ref i32 = struct_access %a.ref.loc15_25, element1
// CHECK:STDOUT:   %.loc15_25.4: i32 = bind_value %.loc15_25.3
// CHECK:STDOUT:   %struct.loc15_25: {.a: i32, .b: i32} = struct_value (%.loc15_25.2, %.loc15_25.4)
// CHECK:STDOUT:   %.loc15_20: {.a: i32, .b: i32} = converted %a.ref.loc15_25, %struct.loc15_25
// CHECK:STDOUT:   br !if.expr.result(%.loc15_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc15_5: {.a: i32, .b: i32} = block_arg !if.expr.result
// CHECK:STDOUT:   %G.call: init () = call %G.ref(%.loc15_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
