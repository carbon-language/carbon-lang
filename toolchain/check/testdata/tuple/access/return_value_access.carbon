// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/return_value_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/return_value_access.carbon

fn F() -> (i32,) { return (0,); }

fn Run() -> i32 {
  return F().0;
}

// CHECK:STDOUT: --- return_value_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.29) [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_12: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc11_12) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_16.1: %tuple.type.1 = tuple_literal (%int.make_type_signed)
// CHECK:STDOUT:     %.loc11_16.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_16.3: type = converted %int.make_type_signed, %.loc11_16.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_16.4: type = converted %.loc11_16.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc13_13.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_13.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_13.3: type = converted %int.make_type_signed, %.loc13_13.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %tuple.type.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_28: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_30.1: %tuple.type.3 = tuple_literal (%.loc11_28)
// CHECK:STDOUT:   %.loc11_30.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_30.3: <bound method> = bound_method %.loc11_28, %.loc11_30.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_30.4: <specific function> = specific_function %.loc11_30.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc11_30.4(%.loc11_28) [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_30.5: %i32 = value_of_initializer %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_30.6: %i32 = converted %.loc11_28, %.loc11_30.5 [template = constants.%.29]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc11_30.6) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_31: %tuple.type.2 = converted %.loc11_30.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc11_31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %tuple.type.2 = call %F.ref()
// CHECK:STDOUT:   %.loc14_14: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_11.1: ref %tuple.type.2 = temporary_storage
// CHECK:STDOUT:   %.loc14_11.2: ref %tuple.type.2 = temporary %.loc14_11.1, %F.call
// CHECK:STDOUT:   %.loc14_13.1: ref %i32 = tuple_access %.loc14_11.2, element0
// CHECK:STDOUT:   %.loc14_13.2: %i32 = bind_value %.loc14_13.1
// CHECK:STDOUT:   return %.loc14_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
