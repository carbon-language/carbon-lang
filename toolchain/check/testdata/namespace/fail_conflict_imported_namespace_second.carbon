// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fn.carbon

package Example library "fn" api;

fn NS();

// --- fail_conflict.carbon

package Example api;

import library "fn";

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+10]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-5]]:1: In import.
// CHECK:STDERR: import library "fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fn.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn NS();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
namespace NS;

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+6]]:7: ERROR: Name qualifiers are only allowed for entities that provide a scope.
// CHECK:STDERR: fn NS.Foo();
// CHECK:STDERR:       ^~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+3]]:4: Non-scope entity referenced here.
// CHECK:STDERR: fn NS.Foo();
// CHECK:STDERR:    ^~
fn NS.Foo();

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
// CHECK:STDOUT: fn @NS();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NS: type = fn_type @NS [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: NS = struct_value () [template]
// CHECK:STDOUT:   %.2: type = fn_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: NS = import_ref ir1, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc16: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NS();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
