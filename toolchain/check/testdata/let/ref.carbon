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
  let (ref d: i32, ref e: i32) = (b, c);
  //@dump-sem-ir-end
}

// --- fail_ref_to_value.carbon

library "[[@TEST_NAME]]";

fn F(a: i32) {
  // CHECK:STDERR: fail_ref_to_value.carbon:[[@LINE+4]]:20: error: cannot bind durable reference to non-reference value of type `i32` [ConversionFailureNonRefToRef]
  // CHECK:STDERR:   let ref b: i32 = a;
  // CHECK:STDERR:                    ^
  // CHECK:STDERR:
  let ref b: i32 = a;
}

// --- fail_ref_to_temporary.carbon

library "[[@TEST_NAME]]";

fn G() -> i32;

fn F() {
  // CHECK:STDERR: fail_ref_to_temporary.carbon:[[@LINE+4]]:20: error: cannot bind durable reference to non-reference value of type `i32` [ConversionFailureNonRefToRef]
  // CHECK:STDERR:   let ref a: i32 = G();
  // CHECK:STDERR:                    ^~~
  // CHECK:STDERR:
  let ref a: i32 = G();
}

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7ce = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.ref.loc7: ref %i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc7: type = splice_block %i32.loc7 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %i32 = ref_binding c, %b.ref.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %e.patt: %pattern_type.7ce = ref_binding_pattern e [concrete]
// CHECK:STDOUT:     %.loc8_30: %pattern_type.511 = tuple_pattern (%d.patt, %e.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.ref.loc8: ref %i32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: ref %i32 = name_ref c, %c
// CHECK:STDOUT:   %.loc8_39: %tuple.type.d07 = tuple_literal (%b.ref.loc8, %c.ref)
// CHECK:STDOUT:   %.loc8_15: type = splice_block %i32.loc8_15 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc8_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %i32 = ref_binding d, %b.ref.loc8
// CHECK:STDOUT:   %.loc8_27: type = splice_block %i32.loc8_27 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc8_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %i32 = ref_binding e, %c.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
