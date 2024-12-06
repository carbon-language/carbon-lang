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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %struct_type.a.b.1: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.a.b.2: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %struct_type.a.b.1 = struct_value (%int_1.2, %int_2.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %s.patt: %struct_type.a.b.1 = binding_pattern s
// CHECK:STDOUT:     %s.param_patt: %struct_type.a.b.1 = value_param_pattern %s.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_14: init type = call constants.%Int(%int_32.loc11_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_14.1: type = value_of_initializer %int.make_type_signed.loc11_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_14.2: type = converted %int.make_type_signed.loc11_14, %.loc11_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_23: init type = call constants.%Int(%int_32.loc11_23) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_23.1: type = value_of_initializer %int.make_type_signed.loc11_23 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_23.2: type = converted %int.make_type_signed.loc11_23, %.loc11_23.1 [template = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:     %s.param: %struct_type.a.b.1 = value_param runtime_param0
// CHECK:STDOUT:     %s: %struct_type.a.b.1 = bind_name s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %cond.patt: bool = binding_pattern cond
// CHECK:STDOUT:     %cond.param_patt: bool = value_param_pattern %cond.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc13_12.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc13_12.2: type = converted %bool.make_type, %.loc13_12.1 [template = bool]
// CHECK:STDOUT:     %cond.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %cond: bool = bind_name cond, %cond.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s.param_patt: %struct_type.a.b.1);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%cond.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc14_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_15: init type = call constants.%Int(%int_32.loc14_15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15.1: type = value_of_initializer %int.make_type_signed.loc14_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15.2: type = converted %int.make_type_signed.loc14_15, %.loc14_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_24: init type = call constants.%Int(%int_32.loc14_24) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.1: type = value_of_initializer %int.make_type_signed.loc14_24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.2: type = converted %int.make_type_signed.loc14_24, %.loc14_24.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:   %a.var: ref %struct_type.a.b.1 = var a
// CHECK:STDOUT:   %a: ref %struct_type.a.b.1 = bind_name a, %a.var
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc14_46.1: %struct_type.a.b.2 = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc14_46.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc14_46.1: <bound method> = bound_method %int_1, %impl.elem0.loc14_46.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_46.1: <specific function> = specific_function %Convert.bound.loc14_46.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_46.1: init %i32 = call %Convert.specific_fn.loc14_46.1(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_46.2: init %i32 = converted %int_1, %int.convert_checked.loc14_46.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_46.3: ref %i32 = struct_access %a.var, element0
// CHECK:STDOUT:   %.loc14_46.4: init %i32 = initialize_from %.loc14_46.2 to %.loc14_46.3 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc14_46.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc14_46.2: <bound method> = bound_method %int_2, %impl.elem0.loc14_46.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_46.2: <specific function> = specific_function %Convert.bound.loc14_46.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc14_46.2: init %i32 = call %Convert.specific_fn.loc14_46.2(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_46.5: init %i32 = converted %int_2, %int.convert_checked.loc14_46.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_46.6: ref %i32 = struct_access %a.var, element1
// CHECK:STDOUT:   %.loc14_46.7: init %i32 = initialize_from %.loc14_46.5 to %.loc14_46.6 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_46.8: init %struct_type.a.b.1 = struct_init (%.loc14_46.4, %.loc14_46.7) to %a.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc14_47: init %struct_type.a.b.1 = converted %.loc14_46.1, %.loc14_46.8 [template = constants.%struct]
// CHECK:STDOUT:   assign %a.var, %.loc14_47
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %cond.ref: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %a.ref.loc15_18: ref %struct_type.a.b.1 = name_ref a, %a
// CHECK:STDOUT:   %.loc15_18.1: ref %i32 = struct_access %a.ref.loc15_18, element0
// CHECK:STDOUT:   %.loc15_18.2: %i32 = bind_value %.loc15_18.1
// CHECK:STDOUT:   %.loc15_18.3: ref %i32 = struct_access %a.ref.loc15_18, element1
// CHECK:STDOUT:   %.loc15_18.4: %i32 = bind_value %.loc15_18.3
// CHECK:STDOUT:   %struct.loc15_18: %struct_type.a.b.1 = struct_value (%.loc15_18.2, %.loc15_18.4)
// CHECK:STDOUT:   %.loc15_18.5: %struct_type.a.b.1 = converted %a.ref.loc15_18, %struct.loc15_18
// CHECK:STDOUT:   br !if.expr.result(%.loc15_18.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc15_25: ref %struct_type.a.b.1 = name_ref a, %a
// CHECK:STDOUT:   %.loc15_25.1: ref %i32 = struct_access %a.ref.loc15_25, element0
// CHECK:STDOUT:   %.loc15_25.2: %i32 = bind_value %.loc15_25.1
// CHECK:STDOUT:   %.loc15_25.3: ref %i32 = struct_access %a.ref.loc15_25, element1
// CHECK:STDOUT:   %.loc15_25.4: %i32 = bind_value %.loc15_25.3
// CHECK:STDOUT:   %struct.loc15_25: %struct_type.a.b.1 = struct_value (%.loc15_25.2, %.loc15_25.4)
// CHECK:STDOUT:   %.loc15_20: %struct_type.a.b.1 = converted %a.ref.loc15_25, %struct.loc15_25
// CHECK:STDOUT:   br !if.expr.result(%.loc15_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc15_5: %struct_type.a.b.1 = block_arg !if.expr.result
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref(%.loc15_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
