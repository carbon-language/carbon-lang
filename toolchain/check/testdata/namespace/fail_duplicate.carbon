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
// CHECK:STDOUT:   %Baz.type: type = fn_type @Baz [concrete]
// CHECK:STDOUT:   %Baz: %Baz.type = struct_value () [concrete]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [concrete]
// CHECK:STDOUT:   %.700: %.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Baz = %Baz.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Baz.decl: %Baz.type = fn_decl @Baz [concrete = constants.%Baz] {} {}
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [concrete = constants.%.700] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
