// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_duplicate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_duplicate.carbon

namespace Foo;

fn Foo.Baz() {
}

// CHECK:STDERR: fail_duplicate.carbon:[[@LINE+7]]:1: error: redefinition of `fn Baz` [RedeclRedef]
// CHECK:STDERR: fn Foo.Baz() {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_duplicate.carbon:[[@LINE-6]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn Foo.Baz() {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn Foo.Baz() {
}

// CHECK:STDOUT: --- fail_duplicate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Baz.type.8987ba.1: type = fn_type @Baz.1 [concrete]
// CHECK:STDOUT:   %Baz.eb4c34.1: %Baz.type.8987ba.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Baz.type.8987ba.2: type = fn_type @Baz.2 [concrete]
// CHECK:STDOUT:   %Baz.eb4c34.2: %Baz.type.8987ba.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Foo: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Baz = %Baz.decl.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Baz.decl.loc13: %Baz.type.8987ba.1 = fn_decl @Baz.1 [concrete = constants.%Baz.eb4c34.1] {} {}
// CHECK:STDOUT:   %Baz.decl.loc23: %Baz.type.8987ba.2 = fn_decl @Baz.2 [concrete = constants.%Baz.eb4c34.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
