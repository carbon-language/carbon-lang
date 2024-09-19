// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/fail_assign_nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/fail_assign_nested.carbon

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:36: error: cannot initialize tuple of 2 element(s) from tuple with 3 element(s).
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                    ^~~~~~~~~
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));

// CHECK:STDOUT: --- fail_assign_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (%.2, %.2) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (%.4, %.4) [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (%.6, %.6) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.12: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.16: type = tuple_type (%.12, %.12) [template]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc14_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_18: %.2 = tuple_literal (%int.make_type_32.loc14_10, %int.make_type_32.loc14_15)
// CHECK:STDOUT:   %int.make_type_32.loc14_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_30: %.2 = tuple_literal (%int.make_type_32.loc14_22, %int.make_type_32.loc14_27)
// CHECK:STDOUT:   %.loc14_31.1: %.3 = tuple_literal (%.loc14_18, %.loc14_30)
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
// CHECK:STDOUT:   %x.var: ref %.5 = var x
// CHECK:STDOUT:   %x: ref %.5 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_37: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc14_40: i32 = int_literal 2 [template = constants.%.10]
// CHECK:STDOUT:   %.loc14_43: i32 = int_literal 3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc14_44: %.12 = tuple_literal (%.loc14_37, %.loc14_40, %.loc14_43)
// CHECK:STDOUT:   %.loc14_48: i32 = int_literal 4 [template = constants.%.13]
// CHECK:STDOUT:   %.loc14_51: i32 = int_literal 5 [template = constants.%.14]
// CHECK:STDOUT:   %.loc14_54: i32 = int_literal 6 [template = constants.%.15]
// CHECK:STDOUT:   %.loc14_55: %.12 = tuple_literal (%.loc14_48, %.loc14_51, %.loc14_54)
// CHECK:STDOUT:   %.loc14_56: %.16 = tuple_literal (%.loc14_44, %.loc14_55)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
