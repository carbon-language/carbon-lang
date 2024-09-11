// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_out_of_bound_non_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_out_of_bound_non_literal.carbon

var a: [i32; 3] = (1, 2, 3);
// CHECK:STDERR: fail_out_of_bound_non_literal.carbon:[[@LINE+3]]:16: ERROR: Array index `3` is past the end of type `[i32; 3]`.
// CHECK:STDERR: var b: i32 = a[{.index = 3}.index];
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~
var b: i32 = a[{.index = 3}.index];

// CHECK:STDOUT: --- fail_out_of_bound_non_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.5, %.6, %.2) [template]
// CHECK:STDOUT:   %.9: type = struct_type {.index: i32} [template]
// CHECK:STDOUT:   %struct: %.9 = struct_value (%.2) [template]
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
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_14: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_32.loc11, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32.loc15, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_23: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_26: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_27.1: %.7 = tuple_literal (%.loc11_20, %.loc11_23, %.loc11_26)
// CHECK:STDOUT:   %.loc11_27.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_27.3: ref i32 = array_index file.%a.var, %.loc11_27.2
// CHECK:STDOUT:   %.loc11_27.4: init i32 = initialize_from %.loc11_20 to %.loc11_27.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_27.5: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_27.6: ref i32 = array_index file.%a.var, %.loc11_27.5
// CHECK:STDOUT:   %.loc11_27.7: init i32 = initialize_from %.loc11_23 to %.loc11_27.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_27.8: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_27.9: ref i32 = array_index file.%a.var, %.loc11_27.8
// CHECK:STDOUT:   %.loc11_27.10: init i32 = initialize_from %.loc11_26 to %.loc11_27.9 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_27.11: init %.3 = array_init (%.loc11_27.4, %.loc11_27.7, %.loc11_27.10) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_28: init %.3 = converted %.loc11_27.1, %.loc11_27.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_28
// CHECK:STDOUT:   %a.ref: ref %.3 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc15_26: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_27.1: %.9 = struct_literal (%.loc15_26)
// CHECK:STDOUT:   %struct: %.9 = struct_value (%.loc15_26) [template = constants.%struct]
// CHECK:STDOUT:   %.loc15_27.2: %.9 = converted %.loc15_27.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %.loc15_28: i32 = struct_access %.loc15_27.2, element0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_34.1: ref i32 = array_index %a.ref, %.loc15_28 [template = <error>]
// CHECK:STDOUT:   %.loc15_34.2: i32 = bind_value %.loc15_34.1
// CHECK:STDOUT:   assign file.%b.var, %.loc15_34.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
