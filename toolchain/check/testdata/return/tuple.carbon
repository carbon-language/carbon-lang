// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/tuple.carbon


fn Main() -> (i32, i32) {
  return (15, 35);
}

// CHECK:STDOUT: --- tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 15 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 35 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.15 [template]
// CHECK:STDOUT:   %.29: i32 = int_value 15 [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.3, %Convert.15 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 35 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.29, %.31) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc12_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc12_20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_23.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc12_15, %int.make_type_32.loc12_20)
// CHECK:STDOUT:     %.loc12_23.2: type = value_of_initializer %int.make_type_32.loc12_15 [template = i32]
// CHECK:STDOUT:     %.loc12_23.3: type = converted %int.make_type_32.loc12_15, %.loc12_23.2 [template = i32]
// CHECK:STDOUT:     %.loc12_23.4: type = value_of_initializer %int.make_type_32.loc12_20 [template = i32]
// CHECK:STDOUT:     %.loc12_23.5: type = converted %int.make_type_32.loc12_20, %.loc12_23.4 [template = i32]
// CHECK:STDOUT:     %.loc12_23.6: type = converted %.loc12_23.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %return: %tuple.type.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_11: Core.IntLiteral = int_value 15 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_15: Core.IntLiteral = int_value 35 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_17.1: %tuple.type.3 = tuple_literal (%.loc13_11, %.loc13_15)
// CHECK:STDOUT:   %.loc13_17.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_17.3: <bound method> = bound_method %.loc13_11, %.loc13_17.2 [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc13_17.1: init i32 = call %.loc13_17.3(%.loc13_11) [template = constants.%.29]
// CHECK:STDOUT:   %.loc13_17.4: init i32 = converted %.loc13_11, %int.convert_checked.loc13_17.1 [template = constants.%.29]
// CHECK:STDOUT:   %.loc13_17.5: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc13_17.6: init i32 = initialize_from %.loc13_17.4 to %.loc13_17.5 [template = constants.%.29]
// CHECK:STDOUT:   %.loc13_17.7: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_17.8: <bound method> = bound_method %.loc13_15, %.loc13_17.7 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc13_17.2: init i32 = call %.loc13_17.8(%.loc13_15) [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_17.9: init i32 = converted %.loc13_15, %int.convert_checked.loc13_17.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_17.10: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc13_17.11: init i32 = initialize_from %.loc13_17.9 to %.loc13_17.10 [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_17.12: init %tuple.type.2 = tuple_init (%.loc13_17.6, %.loc13_17.11) to %return [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_18: init %tuple.type.2 = converted %.loc13_17.1, %.loc13_17.12 [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc13_18 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
