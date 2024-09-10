// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/assign_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/assign_var.carbon

var a: (i32, i32, i32) = (1, 2, 3);
var b: [i32; 3] = a;

// CHECK:STDOUT: --- assign_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.5, %.6, %.7) [template]
// CHECK:STDOUT:   %.8: type = array_type %.7, i32 [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 0 [template]
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
// CHECK:STDOUT:   %int.make_type_32.loc11_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_22.1: %.2 = tuple_literal (%int.make_type_32.loc11_9, %int.make_type_32.loc11_14, %int.make_type_32.loc11_19)
// CHECK:STDOUT:   %.loc11_22.2: type = value_of_initializer %int.make_type_32.loc11_9 [template = i32]
// CHECK:STDOUT:   %.loc11_22.3: type = converted %int.make_type_32.loc11_9, %.loc11_22.2 [template = i32]
// CHECK:STDOUT:   %.loc11_22.4: type = value_of_initializer %int.make_type_32.loc11_14 [template = i32]
// CHECK:STDOUT:   %.loc11_22.5: type = converted %int.make_type_32.loc11_14, %.loc11_22.4 [template = i32]
// CHECK:STDOUT:   %.loc11_22.6: type = value_of_initializer %int.make_type_32.loc11_19 [template = i32]
// CHECK:STDOUT:   %.loc11_22.7: type = converted %int.make_type_32.loc11_19, %.loc11_22.6 [template = i32]
// CHECK:STDOUT:   %.loc11_22.8: type = converted %.loc11_22.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_14: i32 = int_literal 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc12_9.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_9.2: type = converted %int.make_type_32.loc12, %.loc12_9.1 [template = i32]
// CHECK:STDOUT:   %.loc12_15: type = array_type %.loc12_14, i32 [template = constants.%.8]
// CHECK:STDOUT:   %b.var: ref %.8 = var b
// CHECK:STDOUT:   %b: ref %.8 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_27: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_30: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_33: i32 = int_literal 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_34.1: %.3 = tuple_literal (%.loc11_27, %.loc11_30, %.loc11_33)
// CHECK:STDOUT:   %.loc11_34.2: ref i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_34.3: init i32 = initialize_from %.loc11_27 to %.loc11_34.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_34.4: ref i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_34.5: init i32 = initialize_from %.loc11_30 to %.loc11_34.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_34.6: ref i32 = tuple_access file.%a.var, element2
// CHECK:STDOUT:   %.loc11_34.7: init i32 = initialize_from %.loc11_33 to %.loc11_34.6 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_34.8: init %.3 = tuple_init (%.loc11_34.3, %.loc11_34.5, %.loc11_34.7) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_35: init %.3 = converted %.loc11_34.1, %.loc11_34.8 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_35
// CHECK:STDOUT:   %a.ref: ref %.3 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc12_19.1: ref i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc12_19.2: i32 = bind_value %.loc12_19.1
// CHECK:STDOUT:   %.loc12_19.3: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %.loc12_19.4: ref i32 = array_index file.%b.var, %.loc12_19.3
// CHECK:STDOUT:   %.loc12_19.5: init i32 = initialize_from %.loc12_19.2 to %.loc12_19.4
// CHECK:STDOUT:   %.loc12_19.6: ref i32 = tuple_access %a.ref, element1
// CHECK:STDOUT:   %.loc12_19.7: i32 = bind_value %.loc12_19.6
// CHECK:STDOUT:   %.loc12_19.8: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_19.9: ref i32 = array_index file.%b.var, %.loc12_19.8
// CHECK:STDOUT:   %.loc12_19.10: init i32 = initialize_from %.loc12_19.7 to %.loc12_19.9
// CHECK:STDOUT:   %.loc12_19.11: ref i32 = tuple_access %a.ref, element2
// CHECK:STDOUT:   %.loc12_19.12: i32 = bind_value %.loc12_19.11
// CHECK:STDOUT:   %.loc12_19.13: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_19.14: ref i32 = array_index file.%b.var, %.loc12_19.13
// CHECK:STDOUT:   %.loc12_19.15: init i32 = initialize_from %.loc12_19.12 to %.loc12_19.14
// CHECK:STDOUT:   %.loc12_19.16: init %.8 = array_init (%.loc12_19.5, %.loc12_19.10, %.loc12_19.15) to file.%b.var
// CHECK:STDOUT:   %.loc12_20: init %.8 = converted %a.ref, %.loc12_19.16
// CHECK:STDOUT:   assign file.%b.var, %.loc12_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
