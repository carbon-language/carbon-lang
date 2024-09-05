// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/function_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/function_param.carbon

fn F(arr: [i32; 3], i: i32) -> i32 {
  return arr[i];
}

fn G() -> i32 {
  return F((1, 2, 3), 1);
}

// CHECK:STDOUT: --- function_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.5, %.6, %.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32.loc11_12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_17: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_12.1: type = value_of_initializer %int.make_type_32.loc11_12 [template = i32]
// CHECK:STDOUT:     %.loc11_12.2: type = converted %int.make_type_32.loc11_12, %.loc11_12.1 [template = i32]
// CHECK:STDOUT:     %.loc11_18: type = array_type %.loc11_17, i32 [template = constants.%.3]
// CHECK:STDOUT:     %arr.loc11_6.1: %.3 = param arr
// CHECK:STDOUT:     @F.%arr: %.3 = bind_name arr, %arr.loc11_6.1
// CHECK:STDOUT:     %int.make_type_32.loc11_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_24.1: type = value_of_initializer %int.make_type_32.loc11_24 [template = i32]
// CHECK:STDOUT:     %.loc11_24.2: type = converted %int.make_type_32.loc11_24, %.loc11_24.1 [template = i32]
// CHECK:STDOUT:     %i.loc11_21.1: i32 = param i
// CHECK:STDOUT:     @F.%i: i32 = bind_name i, %i.loc11_21.1
// CHECK:STDOUT:     %int.make_type_32.loc11_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_32.1: type = value_of_initializer %int.make_type_32.loc11_32 [template = i32]
// CHECK:STDOUT:     %.loc11_32.2: type = converted %int.make_type_32.loc11_32, %.loc11_32.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_11.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:     %.loc15_11.2: type = converted %int.make_type_32.loc15, %.loc15_11.1 [template = i32]
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%arr: %.3, %i: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: %.3 = name_ref arr, %arr
// CHECK:STDOUT:   %i.ref: i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc12_15.1: ref %.3 = value_as_ref %arr.ref
// CHECK:STDOUT:   %.loc12_15.2: ref i32 = array_index %.loc12_15.1, %i.ref
// CHECK:STDOUT:   %.loc12_15.3: i32 = bind_value %.loc12_15.2
// CHECK:STDOUT:   return %.loc12_15.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc16_13: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_16: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc16_19: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_20.1: %.7 = tuple_literal (%.loc16_13, %.loc16_16, %.loc16_19)
// CHECK:STDOUT:   %.loc16_23: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_20.2: ref %.3 = temporary_storage
// CHECK:STDOUT:   %.loc16_20.3: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc16_20.4: ref i32 = array_index %.loc16_20.2, %.loc16_20.3
// CHECK:STDOUT:   %.loc16_20.5: init i32 = initialize_from %.loc16_13 to %.loc16_20.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_20.6: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_20.7: ref i32 = array_index %.loc16_20.2, %.loc16_20.6
// CHECK:STDOUT:   %.loc16_20.8: init i32 = initialize_from %.loc16_16 to %.loc16_20.7 [template = constants.%.6]
// CHECK:STDOUT:   %.loc16_20.9: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc16_20.10: ref i32 = array_index %.loc16_20.2, %.loc16_20.9
// CHECK:STDOUT:   %.loc16_20.11: init i32 = initialize_from %.loc16_19 to %.loc16_20.10 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_20.12: init %.3 = array_init (%.loc16_20.5, %.loc16_20.8, %.loc16_20.11) to %.loc16_20.2 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_11.1: init %.3 = converted %.loc16_20.1, %.loc16_20.12 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_11.2: ref %.3 = temporary %.loc16_20.2, %.loc16_11.1
// CHECK:STDOUT:   %.loc16_11.3: %.3 = bind_value %.loc16_11.2
// CHECK:STDOUT:   %F.call: init i32 = call %F.ref(%.loc16_11.3, %.loc16_23)
// CHECK:STDOUT:   %.loc16_25.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc16_25.2: i32 = converted %F.call, %.loc16_25.1
// CHECK:STDOUT:   return %.loc16_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
