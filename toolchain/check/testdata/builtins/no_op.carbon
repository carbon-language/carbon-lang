// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/no_op.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/no_op.carbon

// --- no_op.carbon

library "[[@TEST_NAME]]";

fn NoOp() = "no_op";

fn F() {
  //@dump-sem-ir-begin
  NoOp();
  //@dump-sem-ir-end
}

// --- explicit_return.carbon

library "[[@TEST_NAME]]";

fn NoOp() -> () = "no_op";

fn F() {
  //@dump-sem-ir-begin
  NoOp();
  //@dump-sem-ir-end
}

// --- ignore_args.carbon

library "[[@TEST_NAME]]";

fn NoOp(x: ()) -> () = "no_op";

fn F() {
  //@dump-sem-ir-begin
  NoOp(());
  //@dump-sem-ir-end
}

// --- assign.carbon

library "[[@TEST_NAME]]";

fn NoOp() = "no_op";

//@dump-sem-ir-begin
var x: () = NoOp();
//@dump-sem-ir-end

// --- fail_signature.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_signature.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "no_op" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NoOp() -> {} = "no_op";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NoOp() -> {} = "no_op";

// CHECK:STDOUT: --- no_op.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoOp.type: type = fn_type @NoOp [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NoOp: %NoOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NoOp.ref: %NoOp.type = name_ref NoOp, file.%NoOp.decl [concrete = constants.%NoOp]
// CHECK:STDOUT:   %NoOp.call: init %empty_tuple.type = call %NoOp.ref() [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NoOp.type: type = fn_type @NoOp [concrete]
// CHECK:STDOUT:   %NoOp: %NoOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NoOp.ref: %NoOp.type = name_ref NoOp, file.%NoOp.decl [concrete = constants.%NoOp]
// CHECK:STDOUT:   %NoOp.call: init %empty_tuple.type = call %NoOp.ref() [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ignore_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NoOp.type: type = fn_type @NoOp [concrete]
// CHECK:STDOUT:   %NoOp: %NoOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NoOp.ref: %NoOp.type = name_ref NoOp, file.%NoOp.decl [concrete = constants.%NoOp]
// CHECK:STDOUT:   %.loc8_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_9.2: %empty_tuple.type = converted %.loc8_9.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %NoOp.call: init %empty_tuple.type = call %NoOp.ref(%.loc8_9.2) [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- assign.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoOp.type: type = fn_type @NoOp [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NoOp: %NoOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_9.1: type = splice_block %.loc7_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_9.3: type = converted %.loc7_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NoOp.ref: %NoOp.type = name_ref NoOp, file.%NoOp.decl [concrete = constants.%NoOp]
// CHECK:STDOUT:   %NoOp.call: init %empty_tuple.type = call %NoOp.ref() [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%x.var, %NoOp.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
