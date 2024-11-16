// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_field_name_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_field_name_mismatch.carbon

// CHECK:STDERR: fail_field_name_mismatch.carbon:[[@LINE+4]]:20: error: missing value for field `a` in struct initialization [StructInitMissingFieldInLiteral]
// CHECK:STDERR: var x: {.a: i32} = {.b = 1};
// CHECK:STDERR:                    ^~~~~~~~
// CHECK:STDERR:
var x: {.a: i32} = {.b = 1};

// CHECK:STDERR: fail_field_name_mismatch.carbon:[[@LINE+3]]:20: error: cannot convert from struct type `{.a: i32}` to `{.b: i32}`: missing field `b` in source type [StructInitMissingFieldInConversion]
// CHECK:STDERR: var y: {.b: i32} = x;
// CHECK:STDERR:                    ^
var y: {.b: i32} = x;

// CHECK:STDOUT: --- fail_field_name_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.5: type = struct_type {.b: %i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc15_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%.loc15_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_13.2: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_13.3: type = converted %int.make_type_signed.loc15, %.loc15_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_16: type = struct_type {.a: %i32} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref %.2 = var x
// CHECK:STDOUT:   %x: ref %.2 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc20_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc20: init type = call constants.%Int(%.loc20_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_13.2: type = value_of_initializer %int.make_type_signed.loc20 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_13.3: type = converted %int.make_type_signed.loc20, %.loc20_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_16: type = struct_type {.b: %i32} [template = constants.%.5]
// CHECK:STDOUT:   %y.var: ref %.5 = var y
// CHECK:STDOUT:   %y: ref %.5 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_26: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_27: %.4 = struct_literal (%.loc15_26)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   %x.ref: ref %.2 = name_ref x, file.%x
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
