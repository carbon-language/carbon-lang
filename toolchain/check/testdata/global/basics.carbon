// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/global/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/global/basics.carbon

// --- decl.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
var a: {};
//@dump-sem-ir-end

// --- simple_init.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
var a: {} = {};
//@dump-sem-ir-end

// --- simple_init_with_fn.carbon
library "[[@TEST_NAME]]";

fn Make() -> {};

//@dump-sem-ir-begin
var a: {} = Make();
//@dump-sem-ir-end

// --- class_init.carbon
library "[[@TEST_NAME]]";

class A {}

//@dump-sem-ir-begin
var a: A = {};
//@dump-sem-ir-end

// --- class_init_from_fn.carbon
library "[[@TEST_NAME]]";

class A {}

fn Make() -> A;

//@dump-sem-ir-begin
var a: A = Make();
//@dump-sem-ir-end

// CHECK:STDOUT: --- decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_struct_type = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_struct_type = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- simple_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_struct_type = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_struct_type = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc4_14.2: init %empty_struct_type = struct_init () to file.%a.var [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc4_1: init %empty_struct_type = converted %.loc4_14.1, %.loc4_14.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   assign file.%a.var, %.loc4_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- simple_init_with_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_struct_type = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_9.1: type = splice_block %.loc6_9.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc6_9.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_9.3: type = converted %.loc6_9.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_struct_type = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %empty_struct_type = call %Make.ref()
// CHECK:STDOUT:   assign file.%a.var, %Make.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- class_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt [concrete]
// CHECK:STDOUT:   %A.ref: type = name_ref A, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc6_13.2: init %A = class_init (), file.%a.var [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc6_1: init %A = converted %.loc6_13.1, %.loc6_13.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign file.%a.var, %.loc6_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- class_init_from_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt [concrete]
// CHECK:STDOUT:   %A.ref: type = name_ref A, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc8: ref %A = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %Make.call: init %A = call %Make.ref() to %.loc8
// CHECK:STDOUT:   assign file.%a.var, %Make.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
