// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/fail_assign_nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/fail_assign_nested.carbon

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:36: error: cannot initialize tuple of 2 elements from tuple with 3 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                    ^~~~~~~~~
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));

// CHECK:STDOUT: --- fail_assign_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.ff1: type = tuple_type (%tuple.type.d07, %tuple.type.d07) [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %tuple.type.e89: type = tuple_type (%tuple.type.37f, %tuple.type.37f) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.485
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
// CHECK:STDOUT:   %x.var: ref %tuple.type.ff1 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.ff1 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc14_44: %tuple.type.37f = tuple_literal (%int_1, %int_2, %int_3)
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %.loc14_55: %tuple.type.37f = tuple_literal (%int_4, %int_5, %int_6)
// CHECK:STDOUT:   %.loc14_56: %tuple.type.e89 = tuple_literal (%.loc14_44, %.loc14_55)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
