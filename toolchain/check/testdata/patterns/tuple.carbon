// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is mostly checking against crashes for compile time bindings in
// difficult contexts.
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/patterns/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/patterns/tuple.carbon

// --- literal.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let (a: (), b: ()) = ((), ());
//@dump-sem-ir-end

// --- init_expr.carbon

library "[[@TEST_NAME]]";

fn F() -> ((), ());

//@dump-sem-ir-begin
let (a: (), b: ()) = F();
//@dump-sem-ir-end

// --- ref_expr.carbon

library "[[@TEST_NAME]]";

var t: ((), ());
//@dump-sem-ir-begin
let (a: (), b: ()) = t;
//@dump-sem-ir-end

// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc5_18: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_10.1: type = splice_block %.loc5_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_10.3: type = converted %.loc5_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.loc5_24: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_24: %empty_tuple.type = converted @__global_init.%.loc5_24, %empty_tuple.loc5_24 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %a: %empty_tuple.type = value_binding a, %.loc5_24
// CHECK:STDOUT:   %.loc5_17.1: type = splice_block %.loc5_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_17.3: type = converted %.loc5_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.loc5_28: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_28: %empty_tuple.type = converted @__global_init.%.loc5_28, %empty_tuple.loc5_28 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %b: %empty_tuple.type = value_binding b, %.loc5_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_24: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_28: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_29: %tuple.type = tuple_literal (%.loc5_24, %.loc5_28) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_expr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc7_18: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_24.1: ref %tuple.type = temporary @__global_init.%.loc7, @__global_init.%F.call
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access %.loc7_24.1, element0
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %.loc7_24.1, element1
// CHECK:STDOUT:   %.loc7_10.1: type = splice_block %.loc7_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_10.3: type = converted %.loc7_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.loc7_24.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_24.2: %empty_tuple.type = converted %tuple.elem0, %tuple.loc7_24.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %a: %empty_tuple.type = value_binding a, %.loc7_24.2
// CHECK:STDOUT:   %.loc7_17.1: type = splice_block %.loc7_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_17.3: type = converted %.loc7_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.loc7_24.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_24.3: %empty_tuple.type = converted %tuple.elem1, %tuple.loc7_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %b: %empty_tuple.type = value_binding b, %.loc7_24.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7: ref %tuple.type = temporary_storage
// CHECK:STDOUT:   %F.call: init %tuple.type = call %F.ref() to %.loc7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_expr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access file.%t.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access file.%t.var, element1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc6_18: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access @__global_init.%t.ref, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access @__global_init.%t.ref, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc6_10.1: type = splice_block %.loc6_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_10.3: type = converted %.loc6_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.loc6_22.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_22.1: %empty_tuple.type = converted %tuple.elem0, %tuple.loc6_22.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %a: %empty_tuple.type = value_binding a, %.loc6_22.1
// CHECK:STDOUT:   %.loc6_17.1: type = splice_block %.loc6_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_17.3: type = converted %.loc6_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.loc6_22.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_22.2: %empty_tuple.type = converted %tuple.elem1, %tuple.loc6_22.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %b: %empty_tuple.type = value_binding b, %.loc6_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %t.ref: ref %tuple.type = name_ref t, file.%t [concrete = file.%t.var]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
