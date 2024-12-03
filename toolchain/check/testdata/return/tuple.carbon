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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int.15.1: Core.IntLiteral = int_value 15 [template]
// CHECK:STDOUT:   %int.35.1: Core.IntLiteral = int_value 35 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %int.15.1, %Convert.14 [template]
// CHECK:STDOUT:   %.27: <specific function> = specific_function %.26, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.15.2: %i32 = int_value 15 [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %int.35.1, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.35.2: %i32 = int_value 35 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int.15.2, %int.35.2) [template]
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc12_15: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc12_15: init type = call constants.%Int(%int.32.loc12_15) [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc12_20: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc12_20: init type = call constants.%Int(%int.32.loc12_20) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_23.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc12_15, %int.make_type_signed.loc12_20)
// CHECK:STDOUT:     %.loc12_23.2: type = value_of_initializer %int.make_type_signed.loc12_15 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_23.3: type = converted %int.make_type_signed.loc12_15, %.loc12_23.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_23.4: type = value_of_initializer %int.make_type_signed.loc12_20 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_23.5: type = converted %int.make_type_signed.loc12_20, %.loc12_23.4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_23.6: type = converted %.loc12_23.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %return: %tuple.type.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.15: Core.IntLiteral = int_value 15 [template = constants.%int.15.1]
// CHECK:STDOUT:   %int.35: Core.IntLiteral = int_value 35 [template = constants.%int.35.1]
// CHECK:STDOUT:   %.loc13_17.1: %tuple.type.3 = tuple_literal (%int.15, %int.35)
// CHECK:STDOUT:   %.loc13_17.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_17.3: <bound method> = bound_method %int.15, %.loc13_17.2 [template = constants.%.26]
// CHECK:STDOUT:   %.loc13_17.4: <specific function> = specific_function %.loc13_17.3, @Convert.2(constants.%int.32) [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc13_17.1: init %i32 = call %.loc13_17.4(%int.15) [template = constants.%int.15.2]
// CHECK:STDOUT:   %.loc13_17.5: init %i32 = converted %int.15, %int.convert_checked.loc13_17.1 [template = constants.%int.15.2]
// CHECK:STDOUT:   %.loc13_17.6: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc13_17.7: init %i32 = initialize_from %.loc13_17.5 to %.loc13_17.6 [template = constants.%int.15.2]
// CHECK:STDOUT:   %.loc13_17.8: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_17.9: <bound method> = bound_method %int.35, %.loc13_17.8 [template = constants.%.28]
// CHECK:STDOUT:   %.loc13_17.10: <specific function> = specific_function %.loc13_17.9, @Convert.2(constants.%int.32) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc13_17.2: init %i32 = call %.loc13_17.10(%int.35) [template = constants.%int.35.2]
// CHECK:STDOUT:   %.loc13_17.11: init %i32 = converted %int.35, %int.convert_checked.loc13_17.2 [template = constants.%int.35.2]
// CHECK:STDOUT:   %.loc13_17.12: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc13_17.13: init %i32 = initialize_from %.loc13_17.11 to %.loc13_17.12 [template = constants.%int.35.2]
// CHECK:STDOUT:   %.loc13_17.14: init %tuple.type.2 = tuple_init (%.loc13_17.7, %.loc13_17.13) to %return [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_18: init %tuple.type.2 = converted %.loc13_17.1, %.loc13_17.14 [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc13_18 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
