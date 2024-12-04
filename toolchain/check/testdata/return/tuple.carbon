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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int_15.1: Core.IntLiteral = int_value 15 [template]
// CHECK:STDOUT:   %int_35.1: Core.IntLiteral = int_value 35 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_15.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_15.2: %i32 = int_value 15 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_35.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_35.2: %i32 = int_value 35 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int_15.2, %int_35.2) [template]
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
// CHECK:STDOUT:     %int_32.loc12_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc12_15: init type = call constants.%Int(%int_32.loc12_15) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc12_20: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc12_20: init type = call constants.%Int(%int_32.loc12_20) [template = constants.%i32]
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
// CHECK:STDOUT: fn @Main() -> %return.param_patt: %tuple.type.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [template = constants.%int_15.1]
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [template = constants.%int_35.1]
// CHECK:STDOUT:   %.loc13_17.1: %tuple.type.3 = tuple_literal (%int_15, %int_35)
// CHECK:STDOUT:   %impl.elem0.loc13_17.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_17.1: <bound method> = bound_method %int_15, %impl.elem0.loc13_17.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_17.1: <specific function> = specific_function %Convert.bound.loc13_17.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_17.1: init %i32 = call %Convert.specific_fn.loc13_17.1(%int_15) [template = constants.%int_15.2]
// CHECK:STDOUT:   %.loc13_17.2: init %i32 = converted %int_15, %int.convert_checked.loc13_17.1 [template = constants.%int_15.2]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc13_17.3: init %i32 = initialize_from %.loc13_17.2 to %tuple.elem0 [template = constants.%int_15.2]
// CHECK:STDOUT:   %impl.elem0.loc13_17.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_17.2: <bound method> = bound_method %int_35, %impl.elem0.loc13_17.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_17.2: <specific function> = specific_function %Convert.bound.loc13_17.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc13_17.2: init %i32 = call %Convert.specific_fn.loc13_17.2(%int_35) [template = constants.%int_35.2]
// CHECK:STDOUT:   %.loc13_17.4: init %i32 = converted %int_35, %int.convert_checked.loc13_17.2 [template = constants.%int_35.2]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc13_17.5: init %i32 = initialize_from %.loc13_17.4 to %tuple.elem1 [template = constants.%int_35.2]
// CHECK:STDOUT:   %.loc13_17.6: init %tuple.type.2 = tuple_init (%.loc13_17.3, %.loc13_17.5) to %return [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_18: init %tuple.type.2 = converted %.loc13_17.1, %.loc13_17.6 [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc13_18 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
