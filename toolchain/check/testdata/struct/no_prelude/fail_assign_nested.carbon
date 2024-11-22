// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/no_prelude/fail_assign_nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/no_prelude/fail_assign_nested.carbon

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:19: error: missing value for field `a` in struct initialization [StructInitMissingFieldInLiteral]
// CHECK:STDERR: var x: {.a: {}} = {.b = {}};
// CHECK:STDERR:                   ^~~~~~~~~
var x: {.a: {}} = {.b = {}};

// CHECK:STDOUT: --- fail_assign_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: %.1} [template]
// CHECK:STDOUT:   %.4: type = struct_type {.b: %.1} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_14.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc14_14.2: type = converted %.loc14_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_15: type = struct_type {.a: %.1} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref %.2 = var x
// CHECK:STDOUT:   %x: ref %.2 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_26: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc14_27: %.4 = struct_literal (%.loc14_26)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
