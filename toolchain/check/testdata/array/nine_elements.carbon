// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/nine_elements.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/nine_elements.carbon

var a: [i32; 9] = (1, 2, 3, 4, 5, 6, 7, 8, 9);

// CHECK:STDOUT: --- nine_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.9.1: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %.1: type = array_type %int.9.1, %i32 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int.3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int.4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %int.5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int.6.1: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %int.7.1: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %int.8.1: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int.3.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int.4.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int.5.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int.6.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.6.2: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int.7.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.7, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.7.2: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %Convert.bound.8: <bound method> = bound_method %int.8.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %Convert.bound.8, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.8.2: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %Convert.bound.9: <bound method> = bound_method %int.9.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.9: <specific function> = specific_function %Convert.bound.9, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.9.2: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %array: %.1 = tuple_value (%int.1.2, %int.2.2, %int.3.2, %int.4.2, %int.5.2, %int.6.2, %int.7.2, %int.8.2, %int.9.2) [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %int.9: Core.IntLiteral = int_value 9 [template = constants.%int.9.1]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_signed, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %int.9, %i32 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.1.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.2.loc11_23: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %int.3.loc11_26: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %int.4.loc11_29: Core.IntLiteral = int_value 4 [template = constants.%int.4.1]
// CHECK:STDOUT:   %int.5.loc11_32: Core.IntLiteral = int_value 5 [template = constants.%int.5.1]
// CHECK:STDOUT:   %int.6.loc11_35: Core.IntLiteral = int_value 6 [template = constants.%int.6.1]
// CHECK:STDOUT:   %int.7.loc11_38: Core.IntLiteral = int_value 7 [template = constants.%int.7.1]
// CHECK:STDOUT:   %int.8.loc11_41: Core.IntLiteral = int_value 8 [template = constants.%int.8.1]
// CHECK:STDOUT:   %int.9: Core.IntLiteral = int_value 9 [template = constants.%int.9.1]
// CHECK:STDOUT:   %.loc11_45.1: %tuple.type = tuple_literal (%int.1.loc11_20, %int.2.loc11_23, %int.3.loc11_26, %int.4.loc11_29, %int.5.loc11_32, %int.6.loc11_35, %int.7.loc11_38, %int.8.loc11_41, %int.9)
// CHECK:STDOUT:   %.loc11_45.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.1: <bound method> = bound_method %int.1.loc11_20, %.loc11_45.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.1: <specific function> = specific_function %Convert.bound.loc11_45.1, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.1: init %i32 = call %Convert.specific_fn.loc11_45.1(%int.1.loc11_20) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_45.3: init %i32 = converted %int.1.loc11_20, %int.convert_checked.loc11_45.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc11_45.4: ref %i32 = array_index file.%a.var, %int.0
// CHECK:STDOUT:   %.loc11_45.5: init %i32 = initialize_from %.loc11_45.3 to %.loc11_45.4 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_45.6: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.2: <bound method> = bound_method %int.2.loc11_23, %.loc11_45.6 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.2: <specific function> = specific_function %Convert.bound.loc11_45.2, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.2: init %i32 = call %Convert.specific_fn.loc11_45.2(%int.2.loc11_23) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_45.7: init %i32 = converted %int.2.loc11_23, %int.convert_checked.loc11_45.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %int.1.loc11_45: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc11_45.8: ref %i32 = array_index file.%a.var, %int.1.loc11_45
// CHECK:STDOUT:   %.loc11_45.9: init %i32 = initialize_from %.loc11_45.7 to %.loc11_45.8 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_45.10: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.3: <bound method> = bound_method %int.3.loc11_26, %.loc11_45.10 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.3: <specific function> = specific_function %Convert.bound.loc11_45.3, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.3: init %i32 = call %Convert.specific_fn.loc11_45.3(%int.3.loc11_26) [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc11_45.11: init %i32 = converted %int.3.loc11_26, %int.convert_checked.loc11_45.3 [template = constants.%int.3.2]
// CHECK:STDOUT:   %int.2.loc11_45: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc11_45.12: ref %i32 = array_index file.%a.var, %int.2.loc11_45
// CHECK:STDOUT:   %.loc11_45.13: init %i32 = initialize_from %.loc11_45.11 to %.loc11_45.12 [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc11_45.14: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.4: <bound method> = bound_method %int.4.loc11_29, %.loc11_45.14 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.4: <specific function> = specific_function %Convert.bound.loc11_45.4, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.4: init %i32 = call %Convert.specific_fn.loc11_45.4(%int.4.loc11_29) [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc11_45.15: init %i32 = converted %int.4.loc11_29, %int.convert_checked.loc11_45.4 [template = constants.%int.4.2]
// CHECK:STDOUT:   %int.3.loc11_45: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc11_45.16: ref %i32 = array_index file.%a.var, %int.3.loc11_45
// CHECK:STDOUT:   %.loc11_45.17: init %i32 = initialize_from %.loc11_45.15 to %.loc11_45.16 [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc11_45.18: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.5: <bound method> = bound_method %int.5.loc11_32, %.loc11_45.18 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.5: <specific function> = specific_function %Convert.bound.loc11_45.5, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.5: init %i32 = call %Convert.specific_fn.loc11_45.5(%int.5.loc11_32) [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc11_45.19: init %i32 = converted %int.5.loc11_32, %int.convert_checked.loc11_45.5 [template = constants.%int.5.2]
// CHECK:STDOUT:   %int.4.loc11_45: Core.IntLiteral = int_value 4 [template = constants.%int.4.1]
// CHECK:STDOUT:   %.loc11_45.20: ref %i32 = array_index file.%a.var, %int.4.loc11_45
// CHECK:STDOUT:   %.loc11_45.21: init %i32 = initialize_from %.loc11_45.19 to %.loc11_45.20 [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc11_45.22: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.6: <bound method> = bound_method %int.6.loc11_35, %.loc11_45.22 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.6: <specific function> = specific_function %Convert.bound.loc11_45.6, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.6: init %i32 = call %Convert.specific_fn.loc11_45.6(%int.6.loc11_35) [template = constants.%int.6.2]
// CHECK:STDOUT:   %.loc11_45.23: init %i32 = converted %int.6.loc11_35, %int.convert_checked.loc11_45.6 [template = constants.%int.6.2]
// CHECK:STDOUT:   %int.5.loc11_45: Core.IntLiteral = int_value 5 [template = constants.%int.5.1]
// CHECK:STDOUT:   %.loc11_45.24: ref %i32 = array_index file.%a.var, %int.5.loc11_45
// CHECK:STDOUT:   %.loc11_45.25: init %i32 = initialize_from %.loc11_45.23 to %.loc11_45.24 [template = constants.%int.6.2]
// CHECK:STDOUT:   %.loc11_45.26: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.7: <bound method> = bound_method %int.7.loc11_38, %.loc11_45.26 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.7: <specific function> = specific_function %Convert.bound.loc11_45.7, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.7: init %i32 = call %Convert.specific_fn.loc11_45.7(%int.7.loc11_38) [template = constants.%int.7.2]
// CHECK:STDOUT:   %.loc11_45.27: init %i32 = converted %int.7.loc11_38, %int.convert_checked.loc11_45.7 [template = constants.%int.7.2]
// CHECK:STDOUT:   %int.6.loc11_45: Core.IntLiteral = int_value 6 [template = constants.%int.6.1]
// CHECK:STDOUT:   %.loc11_45.28: ref %i32 = array_index file.%a.var, %int.6.loc11_45
// CHECK:STDOUT:   %.loc11_45.29: init %i32 = initialize_from %.loc11_45.27 to %.loc11_45.28 [template = constants.%int.7.2]
// CHECK:STDOUT:   %.loc11_45.30: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.8: <bound method> = bound_method %int.8.loc11_41, %.loc11_45.30 [template = constants.%Convert.bound.8]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.8: <specific function> = specific_function %Convert.bound.loc11_45.8, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.8: init %i32 = call %Convert.specific_fn.loc11_45.8(%int.8.loc11_41) [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc11_45.31: init %i32 = converted %int.8.loc11_41, %int.convert_checked.loc11_45.8 [template = constants.%int.8.2]
// CHECK:STDOUT:   %int.7.loc11_45: Core.IntLiteral = int_value 7 [template = constants.%int.7.1]
// CHECK:STDOUT:   %.loc11_45.32: ref %i32 = array_index file.%a.var, %int.7.loc11_45
// CHECK:STDOUT:   %.loc11_45.33: init %i32 = initialize_from %.loc11_45.31 to %.loc11_45.32 [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc11_45.34: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_45.9: <bound method> = bound_method %int.9, %.loc11_45.34 [template = constants.%Convert.bound.9]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.9: <specific function> = specific_function %Convert.bound.loc11_45.9, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.9]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.9: init %i32 = call %Convert.specific_fn.loc11_45.9(%int.9) [template = constants.%int.9.2]
// CHECK:STDOUT:   %.loc11_45.35: init %i32 = converted %int.9, %int.convert_checked.loc11_45.9 [template = constants.%int.9.2]
// CHECK:STDOUT:   %int.8.loc11_45: Core.IntLiteral = int_value 8 [template = constants.%int.8.1]
// CHECK:STDOUT:   %.loc11_45.36: ref %i32 = array_index file.%a.var, %int.8.loc11_45
// CHECK:STDOUT:   %.loc11_45.37: init %i32 = initialize_from %.loc11_45.35 to %.loc11_45.36 [template = constants.%int.9.2]
// CHECK:STDOUT:   %.loc11_45.38: init %.1 = array_init (%.loc11_45.5, %.loc11_45.9, %.loc11_45.13, %.loc11_45.17, %.loc11_45.21, %.loc11_45.25, %.loc11_45.29, %.loc11_45.33, %.loc11_45.37) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_46: init %.1 = converted %.loc11_45.1, %.loc11_45.38 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_46
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
