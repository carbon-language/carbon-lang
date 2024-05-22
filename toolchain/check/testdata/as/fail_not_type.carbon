// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_not_type.carbon:[[@LINE+3]]:19: ERROR: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: let n: i32 = 1 as 2;
// CHECK:STDERR:                   ^
let n: i32 = 1 as 2;

// CHECK:STDOUT: --- fail_not_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc10_8.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc10_8.2: type = converted %int.make_type_32, %.loc10_8.1 [template = i32]
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_19: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %n: i32 = bind_name n, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
