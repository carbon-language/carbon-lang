// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/fail_element_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/fail_element_type_mismatch.carbon

// CHECK:STDERR: fail_element_type_mismatch.carbon:[[@LINE+3]]:21: ERROR: Cannot implicitly convert from `f64` to `i32`.
// CHECK:STDERR: var x: (i32, i32) = (2, 65.89);
// CHECK:STDERR:                     ^~~~~~~~~~
var x: (i32, i32) = (2, 65.89);

// CHECK:STDOUT: --- fail_element_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: f64 = float_literal 65.890000000000001 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, f64) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
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
// CHECK:STDOUT:   %int.make_type_32.loc14_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_17.1: %.2 = tuple_literal (%int.make_type_32.loc14_9, %int.make_type_32.loc14_14)
// CHECK:STDOUT:   %.loc14_17.2: type = value_of_initializer %int.make_type_32.loc14_9 [template = i32]
// CHECK:STDOUT:   %.loc14_17.3: type = converted %int.make_type_32.loc14_9, %.loc14_17.2 [template = i32]
// CHECK:STDOUT:   %.loc14_17.4: type = value_of_initializer %int.make_type_32.loc14_14 [template = i32]
// CHECK:STDOUT:   %.loc14_17.5: type = converted %int.make_type_32.loc14_14, %.loc14_17.4 [template = i32]
// CHECK:STDOUT:   %.loc14_17.6: type = converted %.loc14_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %x.var: ref %.3 = var x
// CHECK:STDOUT:   %x: ref %.3 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_22: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_25: f64 = float_literal 65.890000000000001 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_30.1: %.7 = tuple_literal (%.loc14_22, %.loc14_25)
// CHECK:STDOUT:   %.loc14_30.2: ref i32 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_30.3: init i32 = initialize_from %.loc14_22 to %.loc14_30.2 [template = constants.%.5]
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
