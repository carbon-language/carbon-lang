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
  // CHECK:STDERR: fail_bad_condition.carbon:[[@LINE+3]]:9: ERROR: Cannot implicitly convert from `String` to `bool`.
  // CHECK:STDERR:   while ("Hello") {}
  // CHECK:STDERR:         ^~~~~~~~~
  while ("Hello") {}
}

// CHECK:STDOUT: --- fail_bad_condition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %While.type: type = fn_type @While [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %While: %While.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type String [template]
// CHECK:STDOUT:   %.3: String = string_literal "Hello" [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .While = %While.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %While.decl: %While.type = fn_decl @While [template = constants.%While] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %.loc15: String = string_literal "Hello" [template = constants.%.3]
// CHECK:STDOUT:   if <error> br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
