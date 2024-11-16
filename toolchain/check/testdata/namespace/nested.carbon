// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/nested.carbon

namespace Foo;
namespace Foo.Bar;

fn Foo.Bar.Wiz() {
}

fn Foo.Bar.Baz() {
  Foo.Bar.Wiz();
}

// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Wiz.type: type = fn_type @Wiz [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Wiz: %Wiz.type = struct_value () [template]
// CHECK:STDOUT:   %Baz.type: type = fn_type @Baz [template]
// CHECK:STDOUT:   %Baz: %Baz.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Foo: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Bar = %Bar
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Wiz = %Wiz.decl
// CHECK:STDOUT:     .Baz = %Baz.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Wiz.decl: %Wiz.type = fn_decl @Wiz [template = constants.%Wiz] {} {}
// CHECK:STDOUT:   %Baz.decl: %Baz.type = fn_decl @Baz [template = constants.%Baz] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Wiz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Foo.ref: <namespace> = name_ref Foo, file.%Foo [template = file.%Foo]
// CHECK:STDOUT:   %Bar.ref: <namespace> = name_ref Bar, file.%Bar [template = file.%Bar]
// CHECK:STDOUT:   %Wiz.ref: %Wiz.type = name_ref Wiz, file.%Wiz.decl [template = constants.%Wiz]
// CHECK:STDOUT:   %Wiz.call: init %empty_tuple.type = call %Wiz.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
