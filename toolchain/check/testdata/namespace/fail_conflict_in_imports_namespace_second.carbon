// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_conflict_in_imports_namespace_second.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_conflict_in_imports_namespace_second.carbon

// --- fn.carbon

package Example library "fn";

fn NS() {}

// --- namespace.carbon

package Example library "namespace";

namespace NS;
fn NS.Foo() {}

// --- fail_conflict.carbon

package Example library "fn_then_namespace";

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+12]]:1: In import.
// CHECK:STDERR: import library "fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: namespace.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+6]]:1: In import.
// CHECK:STDERR: import library "fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fn.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn NS() {}
// CHECK:STDERR: ^~~~~~~~~
import library "fn";
import library "namespace";

fn NS.Bar() {}

// CHECK:STDOUT: --- fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NS.type: type = fn_type @NS [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %NS: %NS.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .NS = %NS.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {}
// CHECK:STDOUT:   %NS.decl: %NS.type = fn_decl @NS [template = constants.%NS] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NS() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar.type: type = fn_type @Bar [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir2, inst+3, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .Foo = %import_ref.3
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+4, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {}
// CHECK:STDOUT:   %Bar.decl: %Bar.type = fn_decl @Bar [template = constants.%Bar] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
