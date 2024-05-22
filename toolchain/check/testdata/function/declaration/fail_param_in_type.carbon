// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_param_in_type.carbon:[[@LINE+3]]:23: ERROR: Array bound is not a constant.
// CHECK:STDERR: fn F(n: i32, a: [i32; n]*);
// CHECK:STDERR:                       ^
fn F(n: i32, a: [i32; n]*);

// CHECK:STDOUT: --- fail_param_in_type.carbon
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
// CHECK:STDOUT:     %int.make_type_32.loc10_9: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc10_9.1: type = value_of_initializer %int.make_type_32.loc10_9 [template = i32]
// CHECK:STDOUT:     %.loc10_9.2: type = converted %int.make_type_32.loc10_9, %.loc10_9.1 [template = i32]
// CHECK:STDOUT:     %n.loc10_6.1: i32 = param n
// CHECK:STDOUT:     @F.%n: i32 = bind_name n, %n.loc10_6.1
// CHECK:STDOUT:     %int.make_type_32.loc10_18: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %n.ref: i32 = name_ref n, @F.%n
// CHECK:STDOUT:     %.loc10_25: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:     %a.loc10_14.1: <error> = param a
// CHECK:STDOUT:     @F.%a: <error> = bind_name a, %a.loc10_14.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n: i32, %a: <error>);
// CHECK:STDOUT:
