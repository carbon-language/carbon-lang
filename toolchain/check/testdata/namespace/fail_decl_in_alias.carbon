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
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .ns = %ns
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %ns: <namespace> = bind_alias ns, %NS [template = %NS]
// CHECK:STDOUT:   %.loc18: <function> = fn_decl @.1 [template] {
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   return %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
