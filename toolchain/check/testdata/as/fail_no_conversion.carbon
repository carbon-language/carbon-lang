// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/fail_no_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/fail_no_conversion.carbon

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+3]]:21: ERROR: Cannot convert from `i32` to `(i32, i32)` with `as`.
// CHECK:STDERR: let n: (i32, i32) = 1 as (i32, i32);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~
let n: (i32, i32) = 1 as (i32, i32);

// CHECK:STDOUT: --- fail_no_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .n = @__global_init.%n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc14_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc14_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_17.1: %.2 = tuple_literal (%int.make_type_32.loc14_9, %int.make_type_32.loc14_14)
// CHECK:STDOUT:   %.loc14_17.2: type = value_of_initializer %int.make_type_32.loc14_9 [template = i32]
// CHECK:STDOUT:   %.loc14_17.3: type = converted %int.make_type_32.loc14_9, %.loc14_17.2 [template = i32]
// CHECK:STDOUT:   %.loc14_17.4: type = value_of_initializer %int.make_type_32.loc14_14 [template = i32]
// CHECK:STDOUT:   %.loc14_17.5: type = converted %int.make_type_32.loc14_14, %.loc14_17.4 [template = i32]
// CHECK:STDOUT:   %.loc14_17.6: type = converted %.loc14_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %import_ref.3: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.4: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_21: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %int.make_type_32.loc14_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_35.1: %.2 = tuple_literal (%int.make_type_32.loc14_27, %int.make_type_32.loc14_32)
// CHECK:STDOUT:   %.loc14_35.2: type = value_of_initializer %int.make_type_32.loc14_27 [template = i32]
// CHECK:STDOUT:   %.loc14_35.3: type = converted %int.make_type_32.loc14_27, %.loc14_35.2 [template = i32]
// CHECK:STDOUT:   %.loc14_35.4: type = value_of_initializer %int.make_type_32.loc14_32 [template = i32]
// CHECK:STDOUT:   %.loc14_35.5: type = converted %int.make_type_32.loc14_32, %.loc14_35.4 [template = i32]
// CHECK:STDOUT:   %.loc14_35.6: type = converted %.loc14_35.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %n: %.3 = bind_name n, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
