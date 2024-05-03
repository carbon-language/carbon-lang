// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- namespace.carbon

package Example library "namespace" api;

namespace NS;
fn NS.Foo() {}

// --- fn.carbon

package Example library "fn" api;

fn NS() {}

// --- fail_conflict.carbon

package Example library "namespace_then_fn" api;

import library "namespace";
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+12]]:1: In import.
// CHECK:STDERR: import library "fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fn.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn NS() {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "namespace";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: namespace.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
import library "fn";

fn NS.Bar() {}

// CHECK:STDOUT: --- namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Foo: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Foo = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: Foo = fn_decl @Foo [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NS: type = fn_type @NS [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: NS = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .NS = %NS.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NS.decl: NS = fn_decl @NS [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NS() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar: type = fn_type @Bar [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Bar = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref ir1, inst+2, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.1, [template] {
// CHECK:STDOUT:     .Foo = %import_ref.2
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Bar.decl: Bar = fn_decl @Bar [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
