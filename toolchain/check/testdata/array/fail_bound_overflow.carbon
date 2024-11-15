// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_bound_overflow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_bound_overflow.carbon

// TODO: Once we preserve the full value of integer literals in SemIR, check
// that we reject the array bound being too large.

// CHECK:STDERR: fail_bound_overflow.carbon:[[@LINE+4]]:14: error: integer literal with value 39999999999999999993 does not fit in i32 [IntLiteralTooLargeForI32]
// CHECK:STDERR: var a: [i32; 39999999999999999993];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: [i32; 39999999999999999993];

// CHECK:STDERR: fail_bound_overflow.carbon:[[@LINE+10]]:9: error: cannot implicitly convert from `i32` to `type` [ImplicitAsConversionFailure]
// CHECK:STDERR: var b: [1; 39999999999999999993];
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_bound_overflow.carbon:[[@LINE+7]]:9: note: type `i32` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: [1; 39999999999999999993];
// CHECK:STDERR:         ^
// CHECK:STDERR:
// CHECK:STDERR: fail_bound_overflow.carbon:[[@LINE+3]]:12: error: integer literal with value 39999999999999999993 does not fit in i32 [IntLiteralTooLargeForI32]
// CHECK:STDERR: var b: [1; 39999999999999999993];
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~
var b: [1; 39999999999999999993];

// CHECK:STDOUT: --- fail_bound_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 1 [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_9.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc18_9.2: type = converted %int.make_type_32, %.loc18_9.1 [template = i32]
// CHECK:STDOUT:   %.loc18_34: type = array_type <error>, i32 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT:   %.loc30_9.1: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc30_9.2: type = converted %.loc30_9.1, <error> [template = <error>]
// CHECK:STDOUT:   %.loc30_32: type = array_type <error>, <error> [template = <error>]
// CHECK:STDOUT:   %b.var: ref <error> = var b
// CHECK:STDOUT:   %b: ref <error> = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
