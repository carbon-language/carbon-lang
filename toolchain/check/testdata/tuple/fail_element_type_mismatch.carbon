// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/fail_element_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/fail_element_type_mismatch.carbon

// CHECK:STDERR: fail_element_type_mismatch.carbon:[[@LINE+6]]:21: error: cannot implicitly convert from `f64` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: var x: (i32, i32) = (2, 65.89);
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: fail_element_type_mismatch.carbon:[[@LINE+3]]:21: note: type `f64` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: (i32, i32) = (2, 65.89);
// CHECK:STDERR:                     ^~~~~~~~~~
var x: (i32, i32) = (2, 65.89);

// CHECK:STDOUT: --- fail_element_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.2: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.3: f64 = float_literal 65.890000000000001 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (i32, f64) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:   %int.make_type_32.loc17_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc17_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_17.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc17_9, %int.make_type_32.loc17_14)
// CHECK:STDOUT:   %.loc17_17.2: type = value_of_initializer %int.make_type_32.loc17_9 [template = i32]
// CHECK:STDOUT:   %.loc17_17.3: type = converted %int.make_type_32.loc17_9, %.loc17_17.2 [template = i32]
// CHECK:STDOUT:   %.loc17_17.4: type = value_of_initializer %int.make_type_32.loc17_14 [template = i32]
// CHECK:STDOUT:   %.loc17_17.5: type = converted %int.make_type_32.loc17_14, %.loc17_17.4 [template = i32]
// CHECK:STDOUT:   %.loc17_17.6: type = converted %.loc17_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %x.var: ref %tuple.type.2 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.2 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_22: i32 = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc17_25: f64 = float_literal 65.890000000000001 [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_30.1: %tuple.type.3 = tuple_literal (%.loc17_22, %.loc17_25)
// CHECK:STDOUT:   %.loc17_30.2: ref i32 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc17_30.3: init i32 = initialize_from %.loc17_22 to %.loc17_30.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc17_30.4: i32 = converted %.loc17_25, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
