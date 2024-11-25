// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_too_few_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_too_few_values.carbon

// CHECK:STDERR: fail_too_few_values.carbon:[[@LINE+3]]:29: error: cannot initialize struct with 2 fields from struct with 1 field [StructInitElementCountMismatch]
// CHECK:STDERR: var x: {.a: i32, .b: i32} = {.a = 1};
// CHECK:STDERR:                             ^~~~~~~~
var x: {.a: i32, .b: i32} = {.a = 1};

// CHECK:STDOUT: --- fail_too_few_values.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: Core.IntLiteral} [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc14_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_13: init type = call constants.%Int(%.loc14_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_13.2: type = value_of_initializer %int.make_type_signed.loc14_13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_13.3: type = converted %int.make_type_signed.loc14_13, %.loc14_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_22: init type = call constants.%Int(%.loc14_22.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_22.2: type = value_of_initializer %int.make_type_signed.loc14_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_22.3: type = converted %int.make_type_signed.loc14_22, %.loc14_22.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_25: type = struct_type {.a: %i32, .b: %i32} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref %.2 = var x
// CHECK:STDOUT:   %x: ref %.2 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_35: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_36: %.5 = struct_literal (%.loc14_35)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
