// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/no_prelude/fail_assign_nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/no_prelude/fail_assign_nested.carbon

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+4]]:19: error: missing value for field `a` in struct initialization [StructInitMissingFieldInLiteral]
// CHECK:STDERR: var x: {.a: {}} = {.b = {}};
// CHECK:STDERR:                   ^~~~~~~~~
// CHECK:STDERR:
var x: {.a: {}} = {.b = {}};

// CHECK:STDOUT: --- fail_assign_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.a.225: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %struct_type.a.225 = binding_pattern x
// CHECK:STDOUT:     %.loc15_1: %struct_type.a.225 = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a.225 = var x
// CHECK:STDOUT:   %.loc15_15: type = splice_block %struct_type.a [concrete = constants.%struct_type.a.225] {
// CHECK:STDOUT:     %.loc15_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc15_14.2: type = converted %.loc15_14.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_struct_type} [concrete = constants.%struct_type.a.225]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.225 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc15_27: %struct_type.b = struct_literal (%.loc15_26)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
