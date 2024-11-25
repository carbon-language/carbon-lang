// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_storage_is_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_storage_is_literal.carbon

fn Main() {
  // CHECK:STDERR: fail_storage_is_literal.carbon:[[@LINE+6]]:10: error: cannot implicitly convert from `Core.IntLiteral` to `type` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   var x: 1 = 1;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_storage_is_literal.carbon:[[@LINE+3]]:10: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var x: 1 = 1;
  // CHECK:STDERR:          ^
  var x: 1 = 1;
}

// CHECK:STDOUT: --- fail_storage_is_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_10.1: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_10.2: type = converted %.loc18_10.1, <error> [template = <error>]
// CHECK:STDOUT:   %x.var: ref <error> = var x
// CHECK:STDOUT:   %x: ref <error> = bind_name x, %x.var
// CHECK:STDOUT:   %.loc18_14: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
