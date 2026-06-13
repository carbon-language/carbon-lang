// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/ref.carbon

// --- local.carbon

library "[[@TEST_NAME]]";

fn F(a: i32) {
  var b: i32 = a;
  //@dump-sem-ir-begin
  let ref c: i32 = b;
  let (unused ref d: i32, unused ref e: i32) = (b, c);
  //@dump-sem-ir-end
}

// --- fail_ref_to_value.carbon

library "[[@TEST_NAME]]";

fn F(a: i32) {
  // CHECK:STDERR: fail_ref_to_value.carbon:[[@LINE+4]]:27: error: cannot bind durable reference to non-reference value of type `i32` [ConversionFailureNonRefToRef]
  // CHECK:STDERR:   let unused ref b: i32 = a;
  // CHECK:STDERR:                           ^
  // CHECK:STDERR:
  let unused ref b: i32 = a;
}

// --- fail_ref_to_temporary.carbon

library "[[@TEST_NAME]]";

fn G() -> i32;

fn F() {
  // CHECK:STDERR: fail_ref_to_temporary.carbon:[[@LINE+4]]:27: error: cannot bind durable reference to non-reference value of type `i32` [ConversionFailureNonRefToRef]
  // CHECK:STDERR:   let unused ref a: i32 = G();
  // CHECK:STDERR:                           ^~~
  // CHECK:STDERR:
  let unused ref a: i32 = G();
}

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.6b6 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.6b6 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.6b6 = ref_binding_pattern e [concrete]
// CHECK:STDOUT:   %tuple.type.e55: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.394: type = pattern_type %tuple.type.e55 [concrete]
// CHECK:STDOUT:   %.dd8: %pattern_type.394 = tuple_pattern (%d.patt, %e.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.6b6 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.ref.loc7: ref %i32 = name_ref b, %b
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %c: ref %i32 = ref_binding c, %b.ref.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.6b6 = ref_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %e.patt: %pattern_type.6b6 = ref_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:     %.loc8_44: %pattern_type.394 = tuple_pattern (%d.patt, %e.patt) [concrete = constants.%.dd8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.ref.loc8: ref %i32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: ref %i32 = name_ref c, %c
// CHECK:STDOUT:   %.loc8_53: %tuple.type.e55 = tuple_literal (%b.ref.loc8, %c.ref)
// CHECK:STDOUT:   %i32.loc8_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %d: ref %i32 = ref_binding d, %b.ref.loc8
// CHECK:STDOUT:   %i32.loc8_41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %e: ref %i32 = ref_binding e, %c.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
