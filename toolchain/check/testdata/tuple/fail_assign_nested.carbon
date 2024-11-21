// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/fail_assign_nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/fail_assign_nested.carbon

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:36: error: cannot initialize tuple of 2 elements from tuple with 3 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                    ^~~~~~~~~
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));

// CHECK:STDOUT: --- fail_assign_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%tuple.type.1, %tuple.type.1) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (%tuple.type.3, %tuple.type.3) [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.6: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.9: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %tuple.type.7: type = tuple_type (%tuple.type.6, %tuple.type.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
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
// CHECK:STDOUT:   %.loc14_10: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_10: init type = call constants.%Int(%.loc14_10) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_15: init type = call constants.%Int(%.loc14_15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_18: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc14_10, %int.make_type_signed.loc14_15)
// CHECK:STDOUT:   %.loc14_22: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_22: init type = call constants.%Int(%.loc14_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_27: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_27: init type = call constants.%Int(%.loc14_27) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_30: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc14_22, %int.make_type_signed.loc14_27)
// CHECK:STDOUT:   %.loc14_31.1: %tuple.type.2 = tuple_literal (%.loc14_18, %.loc14_30)
// CHECK:STDOUT:   %.loc14_31.2: type = value_of_initializer %int.make_type_signed.loc14_10 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_31.3: type = converted %int.make_type_signed.loc14_10, %.loc14_31.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_31.4: type = value_of_initializer %int.make_type_signed.loc14_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_31.5: type = converted %int.make_type_signed.loc14_15, %.loc14_31.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_31.6: type = converted %.loc14_18, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:   %.loc14_31.7: type = value_of_initializer %int.make_type_signed.loc14_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_31.8: type = converted %int.make_type_signed.loc14_22, %.loc14_31.7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_31.9: type = value_of_initializer %int.make_type_signed.loc14_27 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_31.10: type = converted %int.make_type_signed.loc14_27, %.loc14_31.9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_31.11: type = converted %.loc14_30, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:   %.loc14_31.12: type = converted %.loc14_31.1, constants.%tuple.type.4 [template = constants.%tuple.type.4]
// CHECK:STDOUT:   %x.var: ref %tuple.type.4 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.4 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_37: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_40: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_43: Core.IntLiteral = int_value 3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_44: %tuple.type.6 = tuple_literal (%.loc14_37, %.loc14_40, %.loc14_43)
// CHECK:STDOUT:   %.loc14_48: Core.IntLiteral = int_value 4 [template = constants.%.7]
// CHECK:STDOUT:   %.loc14_51: Core.IntLiteral = int_value 5 [template = constants.%.8]
// CHECK:STDOUT:   %.loc14_54: Core.IntLiteral = int_value 6 [template = constants.%.9]
// CHECK:STDOUT:   %.loc14_55: %tuple.type.6 = tuple_literal (%.loc14_48, %.loc14_51, %.loc14_54)
// CHECK:STDOUT:   %.loc14_56: %tuple.type.7 = tuple_literal (%.loc14_44, %.loc14_55)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
