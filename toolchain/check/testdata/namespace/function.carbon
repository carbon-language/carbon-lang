// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/function.carbon

namespace Foo;

// Never called, just here to help catch bugs in name lookup.
fn Baz() {
}

fn Foo.Baz() {
}

fn Bar() {
  Foo.Baz();
}

// CHECK:STDOUT: --- function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Baz.type.03f: type = fn_type @Baz.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Baz.5ea: %Baz.type.03f = struct_value () [concrete]
// CHECK:STDOUT:   %Baz.type.898: type = fn_type @Baz.2 [concrete]
// CHECK:STDOUT:   %Baz.eb4: %Baz.type.898 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type: type = fn_type @Bar [concrete]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Baz = %Baz.decl.loc14
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Foo: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Baz = %Baz.decl.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Baz.decl.loc14: %Baz.type.03f = fn_decl @Baz.1 [concrete = constants.%Baz.5ea] {} {}
// CHECK:STDOUT:   %Baz.decl.loc17: %Baz.type.898 = fn_decl @Baz.2 [concrete = constants.%Baz.eb4] {} {}
// CHECK:STDOUT:   %Bar.decl: %Bar.type = fn_decl @Bar [concrete = constants.%Bar] {} {}
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
// CHECK:STDOUT: fn @Bar() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Foo.ref: <namespace> = name_ref Foo, file.%Foo [concrete = file.%Foo]
// CHECK:STDOUT:   %Baz.ref: %Baz.type.898 = name_ref Baz, file.%Baz.decl.loc17 [concrete = constants.%Baz.eb4]
// CHECK:STDOUT:   %Baz.call: init %empty_tuple.type = call %Baz.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
