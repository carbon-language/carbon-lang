// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_param_redecl.carbon:[[@LINE+6]]:14: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn F(n: i32, n: i32);
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_param_redecl.carbon:[[@LINE+3]]:6: Name is previously declared here.
// CHECK:STDERR: fn F(n: i32, n: i32);
// CHECK:STDERR:      ^
fn F(n: i32, n: i32);

// CHECK:STDOUT: --- fail_param_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc13_9: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc13_9.1: type = value_of_initializer %int.make_type_32.loc13_9 [template = i32]
// CHECK:STDOUT:     %.loc13_9.2: type = converted %int.make_type_32.loc13_9, %.loc13_9.1 [template = i32]
// CHECK:STDOUT:     %n.loc13_6.1: i32 = param n
// CHECK:STDOUT:     @F.%n.loc13_6: i32 = bind_name n, %n.loc13_6.1
// CHECK:STDOUT:     %int.make_type_32.loc13_17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc13_17.1: type = value_of_initializer %int.make_type_32.loc13_17 [template = i32]
// CHECK:STDOUT:     %.loc13_17.2: type = converted %int.make_type_32.loc13_17, %.loc13_17.1 [template = i32]
// CHECK:STDOUT:     %n.loc13_14.1: i32 = param n
// CHECK:STDOUT:     @F.%n.loc13_14: i32 = bind_name n, %n.loc13_14.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.loc13_6: i32, %n.loc13_14: i32);
// CHECK:STDOUT:
