// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/no_prelude/order.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/no_prelude/order.carbon

fn Foo() {}
fn Bar() {}
fn Baz() {}

// CHECK:STDOUT: --- order.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type: type = fn_type @Bar [concrete]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %Baz.type: type = fn_type @Baz [concrete]
// CHECK:STDOUT:   %Baz: %Baz.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:     .Baz = %Baz.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {} {}
// CHECK:STDOUT:   %Bar.decl: %Bar.type = fn_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:   %Baz.decl: %Baz.type = fn_decl @Baz [concrete = constants.%Baz] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
