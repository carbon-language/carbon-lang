// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_type_mismatch.carbon

fn Main() {
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+6]]:17: error: cannot implicitly convert from `Core.IntLiteral` to `bool` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   var x: bool = not 12;
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:17: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var x: bool = not 12;
  // CHECK:STDERR:                 ^~~~~~
  var x: bool = not 12;
}

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.f7c
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
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
// CHECK:STDOUT:   %x.var: ref bool = var x
// CHECK:STDOUT:   %x: ref bool = bind_name x, %x.var
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12]
// CHECK:STDOUT:   %.loc18_17.1: bool = converted %int_12, <error> [template = <error>]
// CHECK:STDOUT:   %.loc18_17.2: <error> = not <error> [template = <error>]
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
