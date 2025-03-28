// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/fail_assign_nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/fail_assign_nested.carbon

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+4]]:36: error: cannot initialize tuple of 2 elements from tuple with 3 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                    ^~~~~~~~~
// CHECK:STDERR:
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));

// CHECK:STDOUT: --- fail_assign_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.1c9: type = tuple_type (%tuple.type.24b, %tuple.type.24b) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff1: type = tuple_type (%tuple.type.d07, %tuple.type.d07) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %tuple.type.e89: type = tuple_type (%tuple.type.37f, %tuple.type.37f) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %tuple.type.ff1 = binding_pattern x
// CHECK:STDOUT:     %.loc15_1: %tuple.type.ff1 = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type.ff1 = var x
// CHECK:STDOUT:   %.loc15_31.1: type = splice_block %.loc15_31.5 [concrete = constants.%tuple.type.ff1] {
// CHECK:STDOUT:     %int_32.loc15_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_18: %tuple.type.24b = tuple_literal (%i32.loc15_10, %i32.loc15_15)
// CHECK:STDOUT:     %int_32.loc15_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_30: %tuple.type.24b = tuple_literal (%i32.loc15_22, %i32.loc15_27)
// CHECK:STDOUT:     %.loc15_31.2: %tuple.type.1c9 = tuple_literal (%.loc15_18, %.loc15_30)
// CHECK:STDOUT:     %.loc15_31.3: type = converted %.loc15_18, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %.loc15_31.4: type = converted %.loc15_30, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %.loc15_31.5: type = converted %.loc15_31.2, constants.%tuple.type.ff1 [concrete = constants.%tuple.type.ff1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.ff1 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc15_44: %tuple.type.37f = tuple_literal (%int_1, %int_2, %int_3)
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %.loc15_55: %tuple.type.37f = tuple_literal (%int_4, %int_5, %int_6)
// CHECK:STDOUT:   %.loc15_56: %tuple.type.e89 = tuple_literal (%.loc15_44, %.loc15_55)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
