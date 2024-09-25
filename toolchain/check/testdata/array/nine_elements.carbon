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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 8 [template]
// CHECK:STDOUT:   %.13: type = tuple_type (i32, i32, i32, i32, i32, i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.5, %.6, %.7, %.8, %.9, %.10, %.11, %.12, %.2) [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_14: i32 = int_literal 9 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_32, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_23: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_26: i32 = int_literal 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_29: i32 = int_literal 4 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_32: i32 = int_literal 5 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_35: i32 = int_literal 6 [template = constants.%.10]
// CHECK:STDOUT:   %.loc11_38: i32 = int_literal 7 [template = constants.%.11]
// CHECK:STDOUT:   %.loc11_41: i32 = int_literal 8 [template = constants.%.12]
// CHECK:STDOUT:   %.loc11_44: i32 = int_literal 9 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_45.1: %.13 = tuple_literal (%.loc11_20, %.loc11_23, %.loc11_26, %.loc11_29, %.loc11_32, %.loc11_35, %.loc11_38, %.loc11_41, %.loc11_44)
// CHECK:STDOUT:   %.loc11_45.2: i32 = int_literal 0 [template = constants.%.14]
// CHECK:STDOUT:   %.loc11_45.3: ref i32 = array_index file.%a.var, %.loc11_45.2
// CHECK:STDOUT:   %.loc11_45.4: init i32 = initialize_from %.loc11_20 to %.loc11_45.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_45.5: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_45.6: ref i32 = array_index file.%a.var, %.loc11_45.5
// CHECK:STDOUT:   %.loc11_45.7: init i32 = initialize_from %.loc11_23 to %.loc11_45.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_45.8: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_45.9: ref i32 = array_index file.%a.var, %.loc11_45.8
// CHECK:STDOUT:   %.loc11_45.10: init i32 = initialize_from %.loc11_26 to %.loc11_45.9 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_45.11: i32 = int_literal 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_45.12: ref i32 = array_index file.%a.var, %.loc11_45.11
// CHECK:STDOUT:   %.loc11_45.13: init i32 = initialize_from %.loc11_29 to %.loc11_45.12 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_45.14: i32 = int_literal 4 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_45.15: ref i32 = array_index file.%a.var, %.loc11_45.14
// CHECK:STDOUT:   %.loc11_45.16: init i32 = initialize_from %.loc11_32 to %.loc11_45.15 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_45.17: i32 = int_literal 5 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_45.18: ref i32 = array_index file.%a.var, %.loc11_45.17
// CHECK:STDOUT:   %.loc11_45.19: init i32 = initialize_from %.loc11_35 to %.loc11_45.18 [template = constants.%.10]
// CHECK:STDOUT:   %.loc11_45.20: i32 = int_literal 6 [template = constants.%.10]
// CHECK:STDOUT:   %.loc11_45.21: ref i32 = array_index file.%a.var, %.loc11_45.20
// CHECK:STDOUT:   %.loc11_45.22: init i32 = initialize_from %.loc11_38 to %.loc11_45.21 [template = constants.%.11]
// CHECK:STDOUT:   %.loc11_45.23: i32 = int_literal 7 [template = constants.%.11]
// CHECK:STDOUT:   %.loc11_45.24: ref i32 = array_index file.%a.var, %.loc11_45.23
// CHECK:STDOUT:   %.loc11_45.25: init i32 = initialize_from %.loc11_41 to %.loc11_45.24 [template = constants.%.12]
// CHECK:STDOUT:   %.loc11_45.26: i32 = int_literal 8 [template = constants.%.12]
// CHECK:STDOUT:   %.loc11_45.27: ref i32 = array_index file.%a.var, %.loc11_45.26
// CHECK:STDOUT:   %.loc11_45.28: init i32 = initialize_from %.loc11_44 to %.loc11_45.27 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_45.29: init %.3 = array_init (%.loc11_45.4, %.loc11_45.7, %.loc11_45.10, %.loc11_45.13, %.loc11_45.16, %.loc11_45.19, %.loc11_45.22, %.loc11_45.25, %.loc11_45.28) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_46: init %.3 = converted %.loc11_45.1, %.loc11_45.29 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_46
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
