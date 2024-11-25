// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/more_param_ir.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/more_param_ir.carbon

fn Foo(a: i32, b: i32) {}

fn Main() {
  var x: (i32,) = (1,);
  // Generates multiple IR instructions for the first parameter.
  Foo(x.0, 6);
}

// CHECK:STDOUT: --- more_param_ir.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.29) [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.31, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_11: init type = call constants.%Int(%.loc11_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.2: type = value_of_initializer %int.make_type_signed.loc11_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.3: type = converted %int.make_type_signed.loc11_11, %.loc11_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_19: init type = call constants.%Int(%.loc11_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.2: type = value_of_initializer %int.make_type_signed.loc11_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.3: type = converted %int.make_type_signed.loc11_19, %.loc11_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a.param_patt: %i32, %b.param_patt: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_11: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc14_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15.1: %tuple.type.1 = tuple_literal (%int.make_type_signed)
// CHECK:STDOUT:   %.loc14_15.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15.3: type = converted %int.make_type_signed, %.loc14_15.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15.4: type = converted %.loc14_15.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %x.var: ref %tuple.type.2 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.2 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc14_20: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_22.1: %tuple.type.3 = tuple_literal (%.loc14_20)
// CHECK:STDOUT:   %.loc14_22.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_22.3: <bound method> = bound_method %.loc14_20, %.loc14_22.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc14_22.4: <specific function> = specific_function %.loc14_22.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32 = call %.loc14_22.4(%.loc14_20) [template = constants.%.29]
// CHECK:STDOUT:   %.loc14_22.5: init %i32 = converted %.loc14_20, %int.convert_checked.loc14 [template = constants.%.29]
// CHECK:STDOUT:   %.loc14_22.6: init %tuple.type.2 = tuple_init (%.loc14_22.5) to %x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_23: init %tuple.type.2 = converted %.loc14_22.1, %.loc14_22.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.var, %.loc14_23
// CHECK:STDOUT:   %Foo.ref: %Foo.type = name_ref Foo, file.%Foo.decl [template = constants.%Foo]
// CHECK:STDOUT:   %x.ref: ref %tuple.type.2 = name_ref x, %x
// CHECK:STDOUT:   %.loc16_9: Core.IntLiteral = int_value 0 [template = constants.%.30]
// CHECK:STDOUT:   %.loc16_8.1: ref %i32 = tuple_access %x.ref, element0
// CHECK:STDOUT:   %.loc16_12.1: Core.IntLiteral = int_value 6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_8.2: %i32 = bind_value %.loc16_8.1
// CHECK:STDOUT:   %.loc16_12.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_12.3: <bound method> = bound_method %.loc16_12.1, %.loc16_12.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc16_12.4: <specific function> = specific_function %.loc16_12.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %.loc16_12.4(%.loc16_12.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc16_12.5: %i32 = value_of_initializer %int.convert_checked.loc16 [template = constants.%.34]
// CHECK:STDOUT:   %.loc16_12.6: %i32 = converted %.loc16_12.1, %.loc16_12.5 [template = constants.%.34]
// CHECK:STDOUT:   %Foo.call: init %empty_tuple.type = call %Foo.ref(%.loc16_8.2, %.loc16_12.6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
