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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.471: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int_15.447: Core.IntLiteral = int_value 15 [template]
// CHECK:STDOUT:   %int_35.c79: Core.IntLiteral = int_value 35 [template]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.bc3: <bound method> = bound_method %int_15.447, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.098: <specific function> = specific_function %Convert.bound.bc3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_15.761: %i32 = int_value 15 [template]
// CHECK:STDOUT:   %Convert.bound.9e7: <bound method> = bound_method %int_35.c79, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.dba: <specific function> = specific_function %Convert.bound.9e7, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_35.545: %i32 = int_value 35 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.471 = tuple_value (%int_15.761, %int_35.545) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
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
// CHECK:STDOUT:     %return.patt: %tuple.type.471 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.471 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_15: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc12_20: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_20: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_23.1: %tuple.type.24b = tuple_literal (%i32.loc12_15, %i32.loc12_20)
// CHECK:STDOUT:     %.loc12_23.2: type = converted %.loc12_23.1, constants.%tuple.type.471 [template = constants.%tuple.type.471]
// CHECK:STDOUT:     %return.param: ref %tuple.type.471 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %tuple.type.471 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %return.param_patt: %tuple.type.471 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [template = constants.%int_15.447]
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [template = constants.%int_35.c79]
// CHECK:STDOUT:   %.loc13_17.1: %tuple.type.f94 = tuple_literal (%int_15, %int_35)
// CHECK:STDOUT:   %impl.elem0.loc13_17.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc13_17.1: <bound method> = bound_method %int_15, %impl.elem0.loc13_17.1 [template = constants.%Convert.bound.bc3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_17.1: <specific function> = specific_function %Convert.bound.loc13_17.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.098]
// CHECK:STDOUT:   %int.convert_checked.loc13_17.1: init %i32 = call %Convert.specific_fn.loc13_17.1(%int_15) [template = constants.%int_15.761]
// CHECK:STDOUT:   %.loc13_17.2: init %i32 = converted %int_15, %int.convert_checked.loc13_17.1 [template = constants.%int_15.761]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc13_17.3: init %i32 = initialize_from %.loc13_17.2 to %tuple.elem0 [template = constants.%int_15.761]
// CHECK:STDOUT:   %impl.elem0.loc13_17.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc13_17.2: <bound method> = bound_method %int_35, %impl.elem0.loc13_17.2 [template = constants.%Convert.bound.9e7]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_17.2: <specific function> = specific_function %Convert.bound.loc13_17.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.dba]
// CHECK:STDOUT:   %int.convert_checked.loc13_17.2: init %i32 = call %Convert.specific_fn.loc13_17.2(%int_35) [template = constants.%int_35.545]
// CHECK:STDOUT:   %.loc13_17.4: init %i32 = converted %int_35, %int.convert_checked.loc13_17.2 [template = constants.%int_35.545]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc13_17.5: init %i32 = initialize_from %.loc13_17.4 to %tuple.elem1 [template = constants.%int_35.545]
// CHECK:STDOUT:   %.loc13_17.6: init %tuple.type.471 = tuple_init (%.loc13_17.3, %.loc13_17.5) to %return [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_18: init %tuple.type.471 = converted %.loc13_17.1, %.loc13_17.6 [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc13_18 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
