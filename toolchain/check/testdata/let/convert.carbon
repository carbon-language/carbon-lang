// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/convert.carbon

// --- convert.carbon

library "[[@TEST_NAME]]";

fn F() -> i32 {
  var v: (i32, i32, i32) = (1, 2, 3);
  // Convert from object representation to value representation.
  //@dump-sem-ir-begin
  let w: (i32, i32, i32) = v;
  //@dump-sem-ir-end
  return w.1;
}

// --- fail_let_bound_to_ref_not_mutable.carbon

library "[[@TEST_NAME]]";

fn G() {
  var a: i32 = 0;
  let n: i32 = a;
  // CHECK:STDERR: fail_let_bound_to_ref_not_mutable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   n = 1;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  n = 1;
}

// CHECK:STDOUT: --- convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.189: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.b5a: type = pattern_type %tuple.type.189 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.b5a = value_binding_pattern w [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.ref: ref %tuple.type.189 = name_ref v, %v
// CHECK:STDOUT:   %.loc8_24.1: type = splice_block %.loc8_24.3 [concrete = constants.%tuple.type.189] {
// CHECK:STDOUT:     %int_32.loc8_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc8_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc8_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_24.2: %tuple.type.ff9 = tuple_literal (%i32.loc8_11, %i32.loc8_16, %i32.loc8_21)
// CHECK:STDOUT:     %.loc8_24.3: type = converted %.loc8_24.2, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.elem0.loc8: ref %i32 = tuple_access %v.ref, element0
// CHECK:STDOUT:   %.loc8_28.1: %i32 = bind_value %tuple.elem0.loc8
// CHECK:STDOUT:   %tuple.elem1.loc8: ref %i32 = tuple_access %v.ref, element1
// CHECK:STDOUT:   %.loc8_28.2: %i32 = bind_value %tuple.elem1.loc8
// CHECK:STDOUT:   %tuple.elem2.loc8: ref %i32 = tuple_access %v.ref, element2
// CHECK:STDOUT:   %.loc8_28.3: %i32 = bind_value %tuple.elem2.loc8
// CHECK:STDOUT:   %tuple: %tuple.type.189 = tuple_value (%.loc8_28.1, %.loc8_28.2, %.loc8_28.3)
// CHECK:STDOUT:   %.loc8_28.4: %tuple.type.189 = converted %v.ref, %tuple
// CHECK:STDOUT:   %w: %tuple.type.189 = value_binding w, %.loc8_28.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
