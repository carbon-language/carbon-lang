// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_value_as_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_value_as_type.carbon

// CHECK:STDERR: fail_value_as_type.carbon:[[@LINE+6]]:8: error: cannot implicitly convert from `{.a: Core.IntLiteral}` to `type` [ImplicitAsConversionFailure]
// CHECK:STDERR: var x: {.a = 1};
// CHECK:STDERR:        ^~~~~~~~
// CHECK:STDERR: fail_value_as_type.carbon:[[@LINE+3]]:8: note: type `{.a: Core.IntLiteral}` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: {.a = 1};
// CHECK:STDERR:        ^~~~~~~~
var x: {.a = 1};

// CHECK:STDOUT: --- fail_value_as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: Core.IntLiteral} [template]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc17_14: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc17_15.1: %.2 = struct_literal (%.loc17_14)
// CHECK:STDOUT:   %.loc17_15.2: type = converted %.loc17_15.1, <error> [template = <error>]
// CHECK:STDOUT:   %x.var: ref <error> = var x
// CHECK:STDOUT:   %x: ref <error> = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
