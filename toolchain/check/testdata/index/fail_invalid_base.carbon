// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_invalid_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_invalid_base.carbon

namespace N;
// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+4]]:14: error: expression cannot be used as a value
// CHECK:STDERR: var a: i32 = N[0];
// CHECK:STDERR:              ^
// CHECK:STDERR:
var a: i32 = N[0];

fn F();
// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+4]]:14: error: expression cannot be used as a value
// CHECK:STDERR: var b: i32 = F[1];
// CHECK:STDERR:              ^
// CHECK:STDERR:
var b: i32 = F[1];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+4]]:14: error: type `{.a: i32, .b: i32}` does not support indexing
// CHECK:STDERR: var c: i32 = {.a = 1, .b = 2}[0];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c: i32 = {.a = 1, .b = 2}[0];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:14: error: type `type` does not support indexing
// CHECK:STDERR: var d: i32 = {.a: i32, .b: i32}[0];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
var d: i32 = {.a: i32, .b: i32}[0];

// CHECK:STDOUT: --- fail_invalid_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %struct: %.5 = struct_value (%.3, %.4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_8.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_8.2: type = converted %int.make_type_32.loc16, %.loc16_8.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %int.make_type_32.loc23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_8.1: type = value_of_initializer %int.make_type_32.loc23 [template = i32]
// CHECK:STDOUT:   %.loc23_8.2: type = converted %int.make_type_32.loc23, %.loc23_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc29_8.1: type = value_of_initializer %int.make_type_32.loc29 [template = i32]
// CHECK:STDOUT:   %.loc29_8.2: type = converted %int.make_type_32.loc29, %.loc29_8.1 [template = i32]
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %int.make_type_32.loc34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc34_8.1: type = value_of_initializer %int.make_type_32.loc34 [template = i32]
// CHECK:STDOUT:   %.loc34_8.2: type = converted %int.make_type_32.loc34, %.loc34_8.1 [template = i32]
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %N.ref: <namespace> = name_ref N, file.%N [template = file.%N]
// CHECK:STDOUT:   %.loc16: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc23: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %.loc29_20: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc29_28: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc29_29.1: %.5 = struct_literal (%.loc29_20, %.loc29_28)
// CHECK:STDOUT:   %.loc29_31: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %struct: %.5 = struct_value (%.loc29_20, %.loc29_28) [template = constants.%struct]
// CHECK:STDOUT:   %.loc29_29.2: %.5 = converted %.loc29_29.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %int.make_type_32.loc34_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc34_19.1: type = value_of_initializer %int.make_type_32.loc34_19 [template = i32]
// CHECK:STDOUT:   %.loc34_19.2: type = converted %int.make_type_32.loc34_19, %.loc34_19.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc34_28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc34_28.1: type = value_of_initializer %int.make_type_32.loc34_28 [template = i32]
// CHECK:STDOUT:   %.loc34_28.2: type = converted %int.make_type_32.loc34_28, %.loc34_28.1 [template = i32]
// CHECK:STDOUT:   %.loc34_31: type = struct_type {.a: i32, .b: i32} [template = constants.%.5]
// CHECK:STDOUT:   %.loc34_33: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
