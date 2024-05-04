// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace NS;

alias ns = NS;

// Aliases can't be used when declaring names.
// CHECK:STDERR: fail_decl_in_alias.carbon:[[@LINE+6]]:7: ERROR: Name qualifiers are only allowed for entities that provide a scope.
// CHECK:STDERR: fn ns.A() -> i32 { return 0; }
// CHECK:STDERR:       ^
// CHECK:STDERR: fail_decl_in_alias.carbon:[[@LINE+3]]:4: Non-scope entity referenced here.
// CHECK:STDERR: fn ns.A() -> i32 { return 0; }
// CHECK:STDERR:    ^~
fn ns.A() -> i32 { return 0; }

// CHECK:STDOUT: --- fail_decl_in_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .ns = %ns
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %ns: <namespace> = bind_alias ns, %NS [template = %NS]
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct] {
// CHECK:STDOUT:     @.1.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
