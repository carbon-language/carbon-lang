// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/while/fail_bad_condition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/while/fail_bad_condition.carbon

fn While() {
  // CHECK:STDERR: fail_bad_condition.carbon:[[@LINE+7]]:9: error: cannot implicitly convert value of type `String` to `bool` [ConversionFailure]
  // CHECK:STDERR:   while ("Hello") {}
  // CHECK:STDERR:         ^~~~~~~~~
  // CHECK:STDERR: fail_bad_condition.carbon:[[@LINE+4]]:9: note: type `String` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   while ("Hello") {}
  // CHECK:STDERR:         ^~~~~~~~~
  // CHECK:STDERR:
  while ("Hello") {}
}

// CHECK:STDOUT: --- fail_bad_condition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %While.type: type = fn_type @While [concrete]
// CHECK:STDOUT:   %While: %While.type = struct_value () [concrete]
// CHECK:STDOUT:   %str: String = string_literal "Hello" [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .While = %While.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %While.decl: %While.type = fn_decl @While [concrete = constants.%While] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %str: String = string_literal "Hello" [concrete = constants.%str]
// CHECK:STDOUT:   %.loc19: bool = converted %str, <error> [concrete = <error>]
// CHECK:STDOUT:   if <error> br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
