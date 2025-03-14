// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/no_prelude/multi_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/no_prelude/multi_error.carbon

// --- fail_multi_error.carbon

// When naming insts, the `error` instruction generated is added for naming.
// This regression test ensures that reuse of `<error>` in different scopes
// works.

// CHECK:STDERR: fail_multi_error.carbon:[[@LINE+4]]:8: error: implicit parameters of functions must be constant or `self` [ImplictParamMustBeConstant]
// CHECK:STDERR: fn Foo[a: ()]();
// CHECK:STDERR:        ^~~~~
// CHECK:STDERR:
fn Foo[a: ()]();
// CHECK:STDERR: fail_multi_error.carbon:[[@LINE+4]]:8: error: implicit parameters of functions must be constant or `self` [ImplictParamMustBeConstant]
// CHECK:STDERR: fn Boo[a: ()]() {}
// CHECK:STDERR:        ^~~~~
// CHECK:STDERR:
fn Boo[a: ()]() {}

// CHECK:STDOUT: --- fail_multi_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Boo.type: type = fn_type @Boo [concrete]
// CHECK:STDOUT:   %Boo: %Boo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Boo = %Boo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {} {}
// CHECK:STDOUT:   %Boo.decl: %Boo.type = fn_decl @Boo [concrete = constants.%Boo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo[<error>: <error>]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Boo[<error>: <error>]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
