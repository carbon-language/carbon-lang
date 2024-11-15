// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_invalid_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_invalid_type.carbon

// CHECK:STDERR: fail_invalid_type.carbon:[[@LINE+6]]:9: error: cannot implicitly convert from `i32` to `type` [ImplicitAsConversionFailure]
// CHECK:STDERR: var a: [1; 1];
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_invalid_type.carbon:[[@LINE+3]]:9: note: type `i32` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var a: [1; 1];
// CHECK:STDERR:         ^
var a: [1; 1];

// CHECK:STDOUT: --- fail_invalid_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.1, %Convert.16 [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc17_9.1: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc17_12.1: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc17_9.2: type = converted %.loc17_9.1, <error> [template = <error>]
// CHECK:STDOUT:   %.loc17_12.2: %Convert.type.6 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %.loc17_12.3: <bound method> = bound_method %.loc17_12.1, %.loc17_12.2 [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.loc17_12.3(%.loc17_12.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc17_12.4: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   %.loc17_12.5: Core.IntLiteral = converted %.loc17_12.1, %.loc17_12.4 [template = constants.%.29]
// CHECK:STDOUT:   %.loc17_13: type = array_type %.loc17_12.5, <error> [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
