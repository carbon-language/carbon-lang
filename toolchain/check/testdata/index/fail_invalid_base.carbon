// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace N;
// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+4]]:14: ERROR: Expression cannot be used as a value.
// CHECK:STDERR: var a: i32 = N[0];
// CHECK:STDERR:              ^
// CHECK:STDERR:
var a: i32 = N[0];

fn F();
// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+4]]:14: ERROR: Expression cannot be used as a value.
// CHECK:STDERR: var b: i32 = F[1];
// CHECK:STDERR:              ^
// CHECK:STDERR:
var b: i32 = F[1];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+4]]:14: ERROR: Type `{.a: i32, .b: i32}` does not support indexing.
// CHECK:STDERR: var c: i32 = {.a = 1, .b = 2}[0];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c: i32 = {.a = 1, .b = 2}[0];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:14: ERROR: Type `type` does not support indexing.
// CHECK:STDERR: var d: i32 = {.a: i32, .b: i32}[0];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
var d: i32 = {.a: i32, .b: i32}[0];

// CHECK:STDOUT: --- fail_invalid_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.6: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %struct.3: {.a: i32, .b: i32} = struct_value (%.3, %.4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc12_8.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_8.2: type = converted %int.make_type_32.loc12, %.loc12_8.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {}
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc19_8.1: type = value_of_initializer %int.make_type_32.loc19 [template = i32]
// CHECK:STDOUT:   %.loc19_8.2: type = converted %int.make_type_32.loc19, %.loc19_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc25: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc25_8.1: type = value_of_initializer %int.make_type_32.loc25 [template = i32]
// CHECK:STDOUT:   %.loc25_8.2: type = converted %int.make_type_32.loc25, %.loc25_8.1 [template = i32]
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc30: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc30_8.1: type = value_of_initializer %int.make_type_32.loc30 [template = i32]
// CHECK:STDOUT:   %.loc30_8.2: type = converted %int.make_type_32.loc30, %.loc30_8.1 [template = i32]
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %N.ref: <namespace> = name_ref N, file.%N [template = file.%N]
// CHECK:STDOUT:   %.loc12: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc19: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %.loc25_20: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc25_28: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc25_29.1: {.a: i32, .b: i32} = struct_literal (%.loc25_20, %.loc25_28)
// CHECK:STDOUT:   %.loc25_31: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %struct: {.a: i32, .b: i32} = struct_value (%.loc25_20, %.loc25_28) [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc25_29.2: {.a: i32, .b: i32} = converted %.loc25_29.1, %struct [template = constants.%struct.3]
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %int.make_type_32.loc30_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc30_19.1: type = value_of_initializer %int.make_type_32.loc30_19 [template = i32]
// CHECK:STDOUT:   %.loc30_19.2: type = converted %int.make_type_32.loc30_19, %.loc30_19.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc30_28: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc30_28.1: type = value_of_initializer %int.make_type_32.loc30_28 [template = i32]
// CHECK:STDOUT:   %.loc30_28.2: type = converted %int.make_type_32.loc30_28, %.loc30_28.1 [template = i32]
// CHECK:STDOUT:   %.loc30_31: type = struct_type {.a: i32, .b: i32} [template = constants.%.5]
// CHECK:STDOUT:   %.loc30_33: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
