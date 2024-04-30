// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_unknown.carbon:[[@LINE+3]]:23: ERROR: Unknown builtin function name "unknown.builtin.name".
// CHECK:STDERR: fn UnknownBuiltin() = "unknown.builtin.name";
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~~~~
fn UnknownBuiltin() = "unknown.builtin.name";

// CHECK:STDOUT: --- fail_unknown.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UnknownBuiltin: type = fn_type @UnknownBuiltin [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: UnknownBuiltin = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .UnknownBuiltin = %UnknownBuiltin.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %UnknownBuiltin.decl: UnknownBuiltin = fn_decl @UnknownBuiltin [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UnknownBuiltin();
// CHECK:STDOUT:
