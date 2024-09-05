// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/fail_aggregate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/fail_aggregate.carbon

// TODO: This creates an array temporary, which we don't yet support evaluating.

// CHECK:STDERR: fail_aggregate.carbon:[[@LINE+3]]:43: ERROR: Array bound is not a constant.
// CHECK:STDERR: var array_index: [i32; 1] = (0,) as [i32; ((5, 7, 1, 9) as [i32; 4])[2]];
// CHECK:STDERR:                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var array_index: [i32; 1] = (0,) as [i32; ((5, 7, 1, 9) as [i32; 4])[2]];

// CHECK:STDOUT: --- fail_aggregate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.10: type = tuple_type (i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.12: type = array_type %.11, i32 [template]
// CHECK:STDOUT:   %.13: type = ptr_type %.12 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %array: %.12 = tuple_value (%.7, %.8, %.2, %.9) [template]
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
// CHECK:STDOUT:     .array_index = %array_index
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_24: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_19.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc16_19.2: type = converted %int.make_type_32, %.loc16_19.1 [template = i32]
// CHECK:STDOUT:   %.loc16_25: type = array_type %.loc16_24, i32 [template = constants.%.3]
// CHECK:STDOUT:   %array_index.var: ref %.3 = var array_index
// CHECK:STDOUT:   %array_index: ref %.3 = bind_name array_index, %array_index.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_30: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_32: %.6 = tuple_literal (%.loc16_30)
// CHECK:STDOUT:   %int.make_type_32.loc16_38: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_45: i32 = int_literal 5 [template = constants.%.7]
// CHECK:STDOUT:   %.loc16_48: i32 = int_literal 7 [template = constants.%.8]
// CHECK:STDOUT:   %.loc16_51: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_54: i32 = int_literal 9 [template = constants.%.9]
// CHECK:STDOUT:   %.loc16_55.1: %.10 = tuple_literal (%.loc16_45, %.loc16_48, %.loc16_51, %.loc16_54)
// CHECK:STDOUT:   %int.make_type_32.loc16_61: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_66: i32 = int_literal 4 [template = constants.%.11]
// CHECK:STDOUT:   %.loc16_61.1: type = value_of_initializer %int.make_type_32.loc16_61 [template = i32]
// CHECK:STDOUT:   %.loc16_61.2: type = converted %int.make_type_32.loc16_61, %.loc16_61.1 [template = i32]
// CHECK:STDOUT:   %.loc16_67: type = array_type %.loc16_66, i32 [template = constants.%.12]
// CHECK:STDOUT:   %.loc16_55.2: ref %.12 = temporary_storage
// CHECK:STDOUT:   %.loc16_55.3: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_55.4: ref i32 = array_index %.loc16_55.2, %.loc16_55.3
// CHECK:STDOUT:   %.loc16_55.5: init i32 = initialize_from %.loc16_45 to %.loc16_55.4 [template = constants.%.7]
// CHECK:STDOUT:   %.loc16_55.6: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_55.7: ref i32 = array_index %.loc16_55.2, %.loc16_55.6
// CHECK:STDOUT:   %.loc16_55.8: init i32 = initialize_from %.loc16_48 to %.loc16_55.7 [template = constants.%.8]
// CHECK:STDOUT:   %.loc16_55.9: i32 = int_literal 2 [template = constants.%.14]
// CHECK:STDOUT:   %.loc16_55.10: ref i32 = array_index %.loc16_55.2, %.loc16_55.9
// CHECK:STDOUT:   %.loc16_55.11: init i32 = initialize_from %.loc16_51 to %.loc16_55.10 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_55.12: i32 = int_literal 3 [template = constants.%.15]
// CHECK:STDOUT:   %.loc16_55.13: ref i32 = array_index %.loc16_55.2, %.loc16_55.12
// CHECK:STDOUT:   %.loc16_55.14: init i32 = initialize_from %.loc16_54 to %.loc16_55.13 [template = constants.%.9]
// CHECK:STDOUT:   %.loc16_55.15: init %.12 = array_init (%.loc16_55.5, %.loc16_55.8, %.loc16_55.11, %.loc16_55.14) to %.loc16_55.2 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_57.1: init %.12 = converted %.loc16_55.1, %.loc16_55.15 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_70: i32 = int_literal 2 [template = constants.%.14]
// CHECK:STDOUT:   %.loc16_57.2: ref %.12 = temporary %.loc16_55.2, %.loc16_57.1
// CHECK:STDOUT:   %.loc16_71.1: ref i32 = array_index %.loc16_57.2, %.loc16_70
// CHECK:STDOUT:   %.loc16_71.2: i32 = bind_value %.loc16_71.1
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
