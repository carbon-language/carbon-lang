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
// CHECK:STDOUT:   %Wiz.type: type = fn_type @Wiz [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Wiz: %Wiz.type = struct_value () [concrete]
// CHECK:STDOUT:   %Baz.type: type = fn_type @Baz [concrete]
// CHECK:STDOUT:   %Baz: %Baz.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Bar = %Bar
// CHECK:STDOUT:     .Foo = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Wiz = %Wiz.decl
// CHECK:STDOUT:     .Baz = %Baz.decl
// CHECK:STDOUT:     .Foo = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Wiz.decl: %Wiz.type = fn_decl @Wiz [concrete = constants.%Wiz] {} {}
// CHECK:STDOUT:   %Baz.decl: %Baz.type = fn_decl @Baz [concrete = constants.%Baz] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Wiz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Foo.ref: <namespace> = name_ref Foo, file.%Foo [concrete = file.%Foo]
// CHECK:STDOUT:   %Bar.ref: <namespace> = name_ref Bar, file.%Bar [concrete = file.%Bar]
// CHECK:STDOUT:   %Wiz.ref: %Wiz.type = name_ref Wiz, file.%Wiz.decl [concrete = constants.%Wiz]
// CHECK:STDOUT:   %Wiz.call: init %empty_tuple.type = call %Wiz.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
