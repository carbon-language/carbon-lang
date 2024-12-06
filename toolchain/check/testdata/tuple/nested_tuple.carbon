// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/nested_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/nested_tuple.carbon

var x: ((i32, i32), i32) = ((12, 76), 6);

// CHECK:STDOUT: --- nested_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%tuple.type.1, type) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (%tuple.type.3, %i32) [template]
// CHECK:STDOUT:   %int_12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_76.1: Core.IntLiteral = int_value 76 [template]
// CHECK:STDOUT:   %tuple.type.6: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_6.1: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %tuple.type.7: type = tuple_type (%tuple.type.6, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_12.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_76.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_76.2: %i32 = int_value 76 [template]
// CHECK:STDOUT:   %tuple.1: %tuple.type.3 = tuple_value (%int_12.2, %int_76.2) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_6.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.2: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %tuple.2: %tuple.type.4 = tuple_value (%tuple.1, %int_6.2) [template]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32.loc11_10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_10: init type = call constants.%Int(%int_32.loc11_10) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc11_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_15: init type = call constants.%Int(%int_32.loc11_15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_18: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_10, %int.make_type_signed.loc11_15)
// CHECK:STDOUT:   %int_32.loc11_21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_21: init type = call constants.%Int(%int_32.loc11_21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_24.1: %tuple.type.2 = tuple_literal (%.loc11_18, %int.make_type_signed.loc11_21)
// CHECK:STDOUT:   %.loc11_24.2: type = value_of_initializer %int.make_type_signed.loc11_10 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_24.3: type = converted %int.make_type_signed.loc11_10, %.loc11_24.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_24.4: type = value_of_initializer %int.make_type_signed.loc11_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_24.5: type = converted %int.make_type_signed.loc11_15, %.loc11_24.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_24.6: type = converted %.loc11_18, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:   %.loc11_24.7: type = value_of_initializer %int.make_type_signed.loc11_21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_24.8: type = converted %int.make_type_signed.loc11_21, %.loc11_24.7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_24.9: type = converted %.loc11_24.1, constants.%tuple.type.4 [template = constants.%tuple.type.4]
// CHECK:STDOUT:   %x.var: ref %tuple.type.4 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.4 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.1]
// CHECK:STDOUT:   %int_76: Core.IntLiteral = int_value 76 [template = constants.%int_76.1]
// CHECK:STDOUT:   %.loc11_36.1: %tuple.type.6 = tuple_literal (%int_12, %int_76)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.1]
// CHECK:STDOUT:   %.loc11_40.1: %tuple.type.7 = tuple_literal (%.loc11_36.1, %int_6)
// CHECK:STDOUT:   %impl.elem0.loc11_36.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_36.1: <bound method> = bound_method %int_12, %impl.elem0.loc11_36.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_36.1: <specific function> = specific_function %Convert.bound.loc11_36.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_36.1: init %i32 = call %Convert.specific_fn.loc11_36.1(%int_12) [template = constants.%int_12.2]
// CHECK:STDOUT:   %.loc11_36.2: init %i32 = converted %int_12, %int.convert_checked.loc11_36.1 [template = constants.%int_12.2]
// CHECK:STDOUT:   %tuple.elem0.loc11_40: ref %tuple.type.3 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc11_36: ref %i32 = tuple_access %tuple.elem0.loc11_40, element0
// CHECK:STDOUT:   %.loc11_36.3: init %i32 = initialize_from %.loc11_36.2 to %tuple.elem0.loc11_36 [template = constants.%int_12.2]
// CHECK:STDOUT:   %impl.elem0.loc11_36.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_36.2: <bound method> = bound_method %int_76, %impl.elem0.loc11_36.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_36.2: <specific function> = specific_function %Convert.bound.loc11_36.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_36.2: init %i32 = call %Convert.specific_fn.loc11_36.2(%int_76) [template = constants.%int_76.2]
// CHECK:STDOUT:   %.loc11_36.4: init %i32 = converted %int_76, %int.convert_checked.loc11_36.2 [template = constants.%int_76.2]
// CHECK:STDOUT:   %tuple.elem1.loc11_36: ref %i32 = tuple_access %tuple.elem0.loc11_40, element1
// CHECK:STDOUT:   %.loc11_36.5: init %i32 = initialize_from %.loc11_36.4 to %tuple.elem1.loc11_36 [template = constants.%int_76.2]
// CHECK:STDOUT:   %.loc11_36.6: init %tuple.type.3 = tuple_init (%.loc11_36.3, %.loc11_36.5) to %tuple.elem0.loc11_40 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc11_40.2: init %tuple.type.3 = converted %.loc11_36.1, %.loc11_36.6 [template = constants.%tuple.1]
// CHECK:STDOUT:   %impl.elem0.loc11_40: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_40: <bound method> = bound_method %int_6, %impl.elem0.loc11_40 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_40: <specific function> = specific_function %Convert.bound.loc11_40, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_40: init %i32 = call %Convert.specific_fn.loc11_40(%int_6) [template = constants.%int_6.2]
// CHECK:STDOUT:   %.loc11_40.3: init %i32 = converted %int_6, %int.convert_checked.loc11_40 [template = constants.%int_6.2]
// CHECK:STDOUT:   %tuple.elem1.loc11_40: ref %i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc11_40.4: init %i32 = initialize_from %.loc11_40.3 to %tuple.elem1.loc11_40 [template = constants.%int_6.2]
// CHECK:STDOUT:   %.loc11_40.5: init %tuple.type.4 = tuple_init (%.loc11_40.2, %.loc11_40.4) to file.%x.var [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc11_41: init %tuple.type.4 = converted %.loc11_40.1, %.loc11_40.5 [template = constants.%tuple.2]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_41
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
