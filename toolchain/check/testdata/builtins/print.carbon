// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/print.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/print.carbon

fn Print(a: i32) = "print.int";

fn Main() {
  Print(1);

  Core.Print(2);
}

// CHECK:STDOUT: --- print.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Print.type.1: type = fn_type @Print.1 [template]
// CHECK:STDOUT:   %Print.1: %Print.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Print.type.2: type = fn_type @Print.2 [template]
// CHECK:STDOUT:   %Print.2: %Print.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Print = %import_ref.38
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.38: %Print.type.2 = import_ref Core//prelude, inst+57, loaded [template = constants.%Print.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Print = %Print.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Print.decl: %Print.type.1 = fn_decl @Print.1 [template = constants.%Print.1] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc11_13.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_13.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_13.3: type = converted %int.make_type_signed, %.loc11_13.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Print.1(%a.param_patt: %i32) = "print.int";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Print.ref.loc14: %Print.type.1 = name_ref Print, file.%Print.decl [template = constants.%Print.1]
// CHECK:STDOUT:   %.loc14_9.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_9.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_9.3: <bound method> = bound_method %.loc14_9.1, %.loc14_9.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc14_9.4: <specific function> = specific_function %.loc14_9.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32 = call %.loc14_9.4(%.loc14_9.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc14_9.5: %i32 = value_of_initializer %int.convert_checked.loc14 [template = constants.%.29]
// CHECK:STDOUT:   %.loc14_9.6: %i32 = converted %.loc14_9.1, %.loc14_9.5 [template = constants.%.29]
// CHECK:STDOUT:   %print.int.loc14: init %empty_tuple.type = call %Print.ref.loc14(%.loc14_9.6)
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %Print.ref.loc16: %Print.type.2 = name_ref Print, imports.%import_ref.38 [template = constants.%Print.2]
// CHECK:STDOUT:   %.loc16_14.1: Core.IntLiteral = int_value 2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc16_14.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_14.3: <bound method> = bound_method %.loc16_14.1, %.loc16_14.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_14.4: <specific function> = specific_function %.loc16_14.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %.loc16_14.4(%.loc16_14.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc16_14.5: %i32 = value_of_initializer %int.convert_checked.loc16 [template = constants.%.33]
// CHECK:STDOUT:   %.loc16_14.6: %i32 = converted %.loc16_14.1, %.loc16_14.5 [template = constants.%.33]
// CHECK:STDOUT:   %print.int.loc16: init %empty_tuple.type = call %Print.ref.loc16(%.loc16_14.6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
