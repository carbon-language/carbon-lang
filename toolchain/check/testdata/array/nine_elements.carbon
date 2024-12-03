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
// CHECK:STDOUT:   %.18: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %.18, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.19: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %.19, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.20: <bound method> = bound_method %int.3.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %.20, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.21: <bound method> = bound_method %int.4.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %.21, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %.22: <bound method> = bound_method %int.5.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %.22, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %.23: <bound method> = bound_method %int.6.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %.23, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.6.2: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %.24: <bound method> = bound_method %int.7.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %.24, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.7.2: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %.25: <bound method> = bound_method %int.8.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %.25, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.8.2: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %int.9.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.9: <specific function> = specific_function %.26, @Convert.2(%int.32) [template]
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
// CHECK:STDOUT:   %.loc11_45.3: <bound method> = bound_method %int.1.loc11_20, %.loc11_45.2 [template = constants.%.18]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.1: <specific function> = specific_function %.loc11_45.3, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.1: init %i32 = call %Convert.specific_fn.loc11_45.1(%int.1.loc11_20) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_45.4: init %i32 = converted %int.1.loc11_20, %int.convert_checked.loc11_45.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc11_45.5: ref %i32 = array_index file.%a.var, %int.0
// CHECK:STDOUT:   %.loc11_45.6: init %i32 = initialize_from %.loc11_45.4 to %.loc11_45.5 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_45.7: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.8: <bound method> = bound_method %int.2.loc11_23, %.loc11_45.7 [template = constants.%.19]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.2: <specific function> = specific_function %.loc11_45.8, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.2: init %i32 = call %Convert.specific_fn.loc11_45.2(%int.2.loc11_23) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_45.9: init %i32 = converted %int.2.loc11_23, %int.convert_checked.loc11_45.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %int.1.loc11_45: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc11_45.10: ref %i32 = array_index file.%a.var, %int.1.loc11_45
// CHECK:STDOUT:   %.loc11_45.11: init %i32 = initialize_from %.loc11_45.9 to %.loc11_45.10 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_45.12: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.13: <bound method> = bound_method %int.3.loc11_26, %.loc11_45.12 [template = constants.%.20]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.3: <specific function> = specific_function %.loc11_45.13, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.3: init %i32 = call %Convert.specific_fn.loc11_45.3(%int.3.loc11_26) [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc11_45.14: init %i32 = converted %int.3.loc11_26, %int.convert_checked.loc11_45.3 [template = constants.%int.3.2]
// CHECK:STDOUT:   %int.2.loc11_45: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc11_45.15: ref %i32 = array_index file.%a.var, %int.2.loc11_45
// CHECK:STDOUT:   %.loc11_45.16: init %i32 = initialize_from %.loc11_45.14 to %.loc11_45.15 [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc11_45.17: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.18: <bound method> = bound_method %int.4.loc11_29, %.loc11_45.17 [template = constants.%.21]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.4: <specific function> = specific_function %.loc11_45.18, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.4: init %i32 = call %Convert.specific_fn.loc11_45.4(%int.4.loc11_29) [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc11_45.19: init %i32 = converted %int.4.loc11_29, %int.convert_checked.loc11_45.4 [template = constants.%int.4.2]
// CHECK:STDOUT:   %int.3.loc11_45: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc11_45.20: ref %i32 = array_index file.%a.var, %int.3.loc11_45
// CHECK:STDOUT:   %.loc11_45.21: init %i32 = initialize_from %.loc11_45.19 to %.loc11_45.20 [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc11_45.22: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.23: <bound method> = bound_method %int.5.loc11_32, %.loc11_45.22 [template = constants.%.22]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.5: <specific function> = specific_function %.loc11_45.23, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.5: init %i32 = call %Convert.specific_fn.loc11_45.5(%int.5.loc11_32) [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc11_45.24: init %i32 = converted %int.5.loc11_32, %int.convert_checked.loc11_45.5 [template = constants.%int.5.2]
// CHECK:STDOUT:   %int.4.loc11_45: Core.IntLiteral = int_value 4 [template = constants.%int.4.1]
// CHECK:STDOUT:   %.loc11_45.25: ref %i32 = array_index file.%a.var, %int.4.loc11_45
// CHECK:STDOUT:   %.loc11_45.26: init %i32 = initialize_from %.loc11_45.24 to %.loc11_45.25 [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc11_45.27: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.28: <bound method> = bound_method %int.6.loc11_35, %.loc11_45.27 [template = constants.%.23]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.6: <specific function> = specific_function %.loc11_45.28, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.6: init %i32 = call %Convert.specific_fn.loc11_45.6(%int.6.loc11_35) [template = constants.%int.6.2]
// CHECK:STDOUT:   %.loc11_45.29: init %i32 = converted %int.6.loc11_35, %int.convert_checked.loc11_45.6 [template = constants.%int.6.2]
// CHECK:STDOUT:   %int.5.loc11_45: Core.IntLiteral = int_value 5 [template = constants.%int.5.1]
// CHECK:STDOUT:   %.loc11_45.30: ref %i32 = array_index file.%a.var, %int.5.loc11_45
// CHECK:STDOUT:   %.loc11_45.31: init %i32 = initialize_from %.loc11_45.29 to %.loc11_45.30 [template = constants.%int.6.2]
// CHECK:STDOUT:   %.loc11_45.32: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.33: <bound method> = bound_method %int.7.loc11_38, %.loc11_45.32 [template = constants.%.24]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.7: <specific function> = specific_function %.loc11_45.33, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.7: init %i32 = call %Convert.specific_fn.loc11_45.7(%int.7.loc11_38) [template = constants.%int.7.2]
// CHECK:STDOUT:   %.loc11_45.34: init %i32 = converted %int.7.loc11_38, %int.convert_checked.loc11_45.7 [template = constants.%int.7.2]
// CHECK:STDOUT:   %int.6.loc11_45: Core.IntLiteral = int_value 6 [template = constants.%int.6.1]
// CHECK:STDOUT:   %.loc11_45.35: ref %i32 = array_index file.%a.var, %int.6.loc11_45
// CHECK:STDOUT:   %.loc11_45.36: init %i32 = initialize_from %.loc11_45.34 to %.loc11_45.35 [template = constants.%int.7.2]
// CHECK:STDOUT:   %.loc11_45.37: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.38: <bound method> = bound_method %int.8.loc11_41, %.loc11_45.37 [template = constants.%.25]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.8: <specific function> = specific_function %.loc11_45.38, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.8: init %i32 = call %Convert.specific_fn.loc11_45.8(%int.8.loc11_41) [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc11_45.39: init %i32 = converted %int.8.loc11_41, %int.convert_checked.loc11_45.8 [template = constants.%int.8.2]
// CHECK:STDOUT:   %int.7.loc11_45: Core.IntLiteral = int_value 7 [template = constants.%int.7.1]
// CHECK:STDOUT:   %.loc11_45.40: ref %i32 = array_index file.%a.var, %int.7.loc11_45
// CHECK:STDOUT:   %.loc11_45.41: init %i32 = initialize_from %.loc11_45.39 to %.loc11_45.40 [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc11_45.42: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.43: <bound method> = bound_method %int.9, %.loc11_45.42 [template = constants.%.26]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_45.9: <specific function> = specific_function %.loc11_45.43, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.9]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.9: init %i32 = call %Convert.specific_fn.loc11_45.9(%int.9) [template = constants.%int.9.2]
// CHECK:STDOUT:   %.loc11_45.44: init %i32 = converted %int.9, %int.convert_checked.loc11_45.9 [template = constants.%int.9.2]
// CHECK:STDOUT:   %int.8.loc11_45: Core.IntLiteral = int_value 8 [template = constants.%int.8.1]
// CHECK:STDOUT:   %.loc11_45.45: ref %i32 = array_index file.%a.var, %int.8.loc11_45
// CHECK:STDOUT:   %.loc11_45.46: init %i32 = initialize_from %.loc11_45.44 to %.loc11_45.45 [template = constants.%int.9.2]
// CHECK:STDOUT:   %.loc11_45.47: init %.1 = array_init (%.loc11_45.6, %.loc11_45.11, %.loc11_45.16, %.loc11_45.21, %.loc11_45.26, %.loc11_45.31, %.loc11_45.36, %.loc11_45.41, %.loc11_45.46) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_46: init %.1 = converted %.loc11_45.1, %.loc11_45.47 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_46
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
