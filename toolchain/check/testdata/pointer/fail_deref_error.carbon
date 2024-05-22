// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_deref_error.carbon:[[@LINE+4]]:15: ERROR: Name `undeclared` not found.
// CHECK:STDERR: let n: i32 = *undeclared;
// CHECK:STDERR:               ^~~~~~~~~~
// CHECK:STDERR:
let n: i32 = *undeclared;
// CHECK:STDERR: fail_deref_error.carbon:[[@LINE+3]]:15: ERROR: Name `undeclared` not found.
// CHECK:STDERR: let n2: i32 = undeclared->foo;
// CHECK:STDERR:               ^~~~~~~~~~
let n2: i32 = undeclared->foo;

// CHECK:STDOUT: --- fail_deref_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc11_8.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_8.2: type = converted %int.make_type_32.loc11, %.loc11_8.1 [template = i32]
// CHECK:STDOUT:   %undeclared.ref.loc11: <error> = name_ref undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc11_14: ref <error> = deref <error>
// CHECK:STDOUT:   %n: i32 = bind_name n, <error>
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc15_9.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_9.2: type = converted %int.make_type_32.loc15, %.loc15_9.1 [template = i32]
// CHECK:STDOUT:   %undeclared.ref.loc15: <error> = name_ref undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc15_25: ref <error> = deref <error>
// CHECK:STDOUT:   %n2: i32 = bind_name n2, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
