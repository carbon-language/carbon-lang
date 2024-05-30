// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuples/fail_assign_nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/tuples/fail_assign_nested.carbon

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:36: ERROR: Cannot initialize tuple of 2 element(s) from tuple with 3 element(s).
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                    ^~~~~~~~~
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));

// CHECK:STDOUT: --- fail_assign_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type ((type, type), (type, type)) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.5: type = tuple_type ((i32, i32), (i32, i32)) [template]
// CHECK:STDOUT:   %.6: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.7: type = tuple_type ((i32, i32)*, (i32, i32)*) [template]
// CHECK:STDOUT:   %.8: type = ptr_type ((i32, i32)*, (i32, i32)*) [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.12: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.16: type = tuple_type ((i32, i32, i32), (i32, i32, i32)) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc14_10: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc14_15: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc14_18: (type, type) = tuple_literal (%int.make_type_32.loc14_10, %int.make_type_32.loc14_15)
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc14_22: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc14_27: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc14_30: (type, type) = tuple_literal (%int.make_type_32.loc14_22, %int.make_type_32.loc14_27)
// CHECK:STDOUT:   %.loc14_31.1: ((type, type), (type, type)) = tuple_literal (%.loc14_18, %.loc14_30)
// CHECK:STDOUT:   %.loc14_31.2: type = value_of_initializer %int.make_type_32.loc14_10 [template = i32]
// CHECK:STDOUT:   %.loc14_31.3: type = converted %int.make_type_32.loc14_10, %.loc14_31.2 [template = i32]
// CHECK:STDOUT:   %.loc14_31.4: type = value_of_initializer %int.make_type_32.loc14_15 [template = i32]
// CHECK:STDOUT:   %.loc14_31.5: type = converted %int.make_type_32.loc14_15, %.loc14_31.4 [template = i32]
// CHECK:STDOUT:   %.loc14_31.6: type = converted %.loc14_18, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_31.7: type = value_of_initializer %int.make_type_32.loc14_22 [template = i32]
// CHECK:STDOUT:   %.loc14_31.8: type = converted %int.make_type_32.loc14_22, %.loc14_31.7 [template = i32]
// CHECK:STDOUT:   %.loc14_31.9: type = value_of_initializer %int.make_type_32.loc14_27 [template = i32]
// CHECK:STDOUT:   %.loc14_31.10: type = converted %int.make_type_32.loc14_27, %.loc14_31.9 [template = i32]
// CHECK:STDOUT:   %.loc14_31.11: type = converted %.loc14_30, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_31.12: type = converted %.loc14_31.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %x.var: ref ((i32, i32), (i32, i32)) = var x
// CHECK:STDOUT:   %x: ref ((i32, i32), (i32, i32)) = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_37: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc14_40: i32 = int_literal 2 [template = constants.%.10]
// CHECK:STDOUT:   %.loc14_43: i32 = int_literal 3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc14_44: (i32, i32, i32) = tuple_literal (%.loc14_37, %.loc14_40, %.loc14_43)
// CHECK:STDOUT:   %.loc14_48: i32 = int_literal 4 [template = constants.%.13]
// CHECK:STDOUT:   %.loc14_51: i32 = int_literal 5 [template = constants.%.14]
// CHECK:STDOUT:   %.loc14_54: i32 = int_literal 6 [template = constants.%.15]
// CHECK:STDOUT:   %.loc14_55: (i32, i32, i32) = tuple_literal (%.loc14_48, %.loc14_51, %.loc14_54)
// CHECK:STDOUT:   %.loc14_56: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%.loc14_44, %.loc14_55)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
