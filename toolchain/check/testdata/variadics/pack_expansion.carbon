// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/variadics/pack_expansion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/variadics/pack_expansion.carbon

// --- homogeneous.carbon
library "[[@TEST_NAME]]";

fn G(n: i32);

//@dump-sem-ir-begin
fn F(x: (i32, i32, i32)) {
  ... G(expand x);
}
//@dump-sem-ir-end

// --- heterogeneous.carbon
library "[[@TEST_NAME]]";

fn G[T: type](unused t: T) {}

//@dump-sem-ir-begin
fn F(x: (i32, bool), y: (bool, i32)) {
  ... if (true) {
    G(expand x);
    G(expand y);
  }
}
//@dump-sem-ir-end

// --- fail_heterogeneous_type_mismatch.carbon
library "[[@TEST_NAME]]";

fn G(n: i32);

fn F(x: (i32, bool)) {
  // CHECK:STDERR: fail_heterogeneous_type_mismatch.carbon:[[@LINE+13]]:3: error: unable to monomorphize specific `pack expansion element 1` [ResolvingSpecificHere]
  // CHECK:STDERR:   ... G(expand x);
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR: fail_heterogeneous_type_mismatch.carbon:[[@LINE+10]]:7: note: cannot implicitly convert expression of type `bool` to `i32` [ConversionFailure]
  // CHECK:STDERR:   ... G(expand x);
  // CHECK:STDERR:       ^~~~~~~~~~~
  // CHECK:STDERR: fail_heterogeneous_type_mismatch.carbon:[[@LINE+7]]:7: note: type `bool` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   ... G(expand x);
  // CHECK:STDERR:       ^~~~~~~~~~~
  // CHECK:STDERR: fail_heterogeneous_type_mismatch.carbon:[[@LINE-12]]:6: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn G(n: i32);
  // CHECK:STDERR:      ^~~~~~
  // CHECK:STDERR:
  ... G(expand x);
}

// --- generic.carbon
library "[[@TEST_NAME]]";

fn G[T: type](unused t: T) {}

//@dump-sem-ir-begin
fn F[T: type](x: (T, T)) {
  ... G(expand x);
}
//@dump-sem-ir-end

fn Call(a: (i32, i32)) {
  F(a);
}

// --- multiple_expands.carbon
library "[[@TEST_NAME]]";

fn G(a: i32, b: bool);

fn F(x: (i32, i32), y: (bool, bool)) {
  ... G(expand x, expand y);
}

// --- empty.carbon
library "[[@TEST_NAME]]";

fn G(n: i32);

fn F(x: ()) {
  ... G(expand x);
}

// --- fail_expand_outside_pack.carbon
library "[[@TEST_NAME]]";

fn F(x: (i32, i32)) {
  // CHECK:STDERR: fail_expand_outside_pack.carbon:[[@LINE+4]]:3: error: `expand` can only be used in a pack expansion [ExpandOutsidePackExpansion]
  // CHECK:STDERR:   expand x;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  expand x;
}

// --- fail_expand_non_tuple.carbon
library "[[@TEST_NAME]]";

fn G(n: i32);

fn F(x: i32) {
  // CHECK:STDERR: fail_expand_non_tuple.carbon:[[@LINE+4]]:9: error: operand of `expand` has type `i32`, which is not a tuple type [ExpandOperandNotTuple]
  // CHECK:STDERR:   ... G(expand x);
  // CHECK:STDERR:         ^~~~~~~~
  // CHECK:STDERR:
  ... G(expand x);
}

// --- fail_arity_mismatch.carbon
library "[[@TEST_NAME]]";

fn G(a: i32, b: i32);

fn F(x: (i32, i32), y: (i32, i32, i32)) {
  // CHECK:STDERR: fail_arity_mismatch.carbon:[[@LINE+7]]:19: error: operand of `expand` has 3 elements, but pack expansion has 2 elements [ExpandArityMismatch]
  // CHECK:STDERR:   ... G(expand x, expand y);
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR: fail_arity_mismatch.carbon:[[@LINE+4]]:16: note: number of elements of pack expansion determined here [ExpandArityMismatchPrevious]
  // CHECK:STDERR:   ... G(expand x, expand y);
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  ... G(expand x, expand y);
}

// --- fail_no_expand.carbon
library "[[@TEST_NAME]]";

fn G();

fn F() {
  // CHECK:STDERR: fail_no_expand.carbon:[[@LINE+4]]:3: error: pack expansion does not contain an `expand` expression [PackExpansionWithoutExpand]
  // CHECK:STDERR:   ... G();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  ... G();
}

// --- fail_todo_nested.carbon
library "[[@TEST_NAME]]";

fn G(n: i32);

fn F(x: (i32, i32)) {
  // CHECK:STDERR: fail_todo_nested.carbon:[[@LINE+8]]:7: error: semantics TODO: `nested pack expansion` [SemanticsTodo]
  // CHECK:STDERR:   ... ... G(expand x);
  // CHECK:STDERR:       ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_nested.carbon:[[@LINE+4]]:3: error: pack expansion does not contain an `expand` expression [PackExpansionWithoutExpand]
  // CHECK:STDERR:   ... ... G(expand x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ... ... G(expand x);
}

// CHECK:STDOUT: --- homogeneous.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.531: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.276: %tuple.type.531 = tuple_value (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.555: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.777: type = pattern_type %tuple.type.555 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.777 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.777 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pack_index: Core.IntLiteral = symbolic_binding <pack index>, 0 [symbolic]
// CHECK:STDOUT:   %tuple.type.705: type = tuple_type (<instruction>, <instruction>, <instruction>) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %inst.specific_inst.92f: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.610 = specific_inst @F.pack.%entry, @F.pack(%int_0)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %inst.specific_inst.020: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.33d = specific_inst @F.pack.%entry, @F.pack(%int_1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %inst.specific_inst.139: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.d1d = specific_inst @F.pack.%entry, @F.pack(%int_2)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.0dc: %tuple.type.705 = tuple_value (%inst.specific_inst.92f, %inst.specific_inst.020, %inst.specific_inst.139) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.777 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.777 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.555 = value_param call_param0
// CHECK:STDOUT:     %.loc6_23.1: type = splice_block %.loc6_23.3 [concrete = constants.%tuple.type.555] {
// CHECK:STDOUT:       %i32.loc6_10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc6_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc6_20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc6_23.2: %tuple.type.531 = tuple_literal (%i32.loc6_10, %i32.loc6_15, %i32.loc6_20) [concrete = constants.%tuple.276]
// CHECK:STDOUT:       %.loc6_23.3: type = converted %.loc6_23.2, constants.%tuple.type.555 [concrete = constants.%tuple.type.555]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.555 = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %tuple.type.555) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %pack: %tuple.type.705 = pack_expansion @F.pack(%pack_index.loc7_3.2: Core.IntLiteral), %entry [concrete = constants.%tuple.0dc] {
// CHECK:STDOUT:     %pack_index.loc7_3.1: Core.IntLiteral = symbolic_binding <pack index>, 0 [symbolic = %pack_index.loc7_3.1 (constants.%pack_index)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   !pack.entry:
// CHECK:STDOUT:     %pack_index.loc7_3.2: Core.IntLiteral = symbolic_binding <pack index>, 0 [symbolic = %pack_index.loc7_3.1 (constants.%pack_index)]
// CHECK:STDOUT:     %entry = br !pack.body
// CHECK:STDOUT:
// CHECK:STDOUT:   !pack.body:
// CHECK:STDOUT:     %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:     %x.ref: %tuple.type.555 = name_ref x, @F.%x
// CHECK:STDOUT:     %.loc7: %i32 = tuple_index %x.ref, %pack_index.loc7_3.2
// CHECK:STDOUT:     %G.call: init %empty_tuple.type = call %G.ref(%.loc7)
// CHECK:STDOUT:     br_next_index
// CHECK:STDOUT:   }
// CHECK:STDOUT:   splice_br %pack, !pack.done
// CHECK:STDOUT:
// CHECK:STDOUT: !pack.done:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%pack_index) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%pack_index
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%int_0) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%int_0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%int_1) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%int_1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%int_2) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%int_2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- heterogeneous.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.760: %tuple.type.693 = tuple_value (%i32, bool) [concrete]
// CHECK:STDOUT:   %tuple.type.6ed: type = tuple_type (%i32, bool) [concrete]
// CHECK:STDOUT:   %pattern_type.cae: type = pattern_type %tuple.type.6ed [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.cae = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cae = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %tuple.dcd: %tuple.type.693 = tuple_value (bool, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.68a: type = tuple_type (bool, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.541: type = pattern_type %tuple.type.68a [concrete]
// CHECK:STDOUT:   %y.param_patt: %pattern_type.541 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.541 = wrapper_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pack_index: Core.IntLiteral = symbolic_binding <pack index>, 0 [symbolic]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %.2d7: type = tuple_index %tuple.type.6ed, %pack_index [template]
// CHECK:STDOUT:   %.f92: type = type_of_inst @F.pack.%.loc8_15.1 [template]
// CHECK:STDOUT:   %.d40: %.f92 = splice_inst @F.pack.%.loc8_15.1 [template]
// CHECK:STDOUT:   %require_complete.1e6: <witness> = require_complete_type %.f92 [template]
// CHECK:STDOUT:   %.30c: %.f92 = splice_inst @F.pack.%.loc8_15.4 [template]
// CHECK:STDOUT:   %.26e: type = tuple_index %tuple.type.68a, %pack_index [template]
// CHECK:STDOUT:   %.f0b: type = type_of_inst @F.pack.%.loc9_15.1 [template]
// CHECK:STDOUT:   %.a6d: %.f0b = splice_inst @F.pack.%.loc9_15.1 [template]
// CHECK:STDOUT:   %require_complete.cfa: <witness> = require_complete_type %.f0b [template]
// CHECK:STDOUT:   %.ec9: %.f0b = splice_inst @F.pack.%.loc9_15.4 [template]
// CHECK:STDOUT:   %tuple.type.413: type = tuple_type (<instruction>, <instruction>) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %inst.splice_block.008: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.60f: init %empty_tuple.type = splice_block %G.call.da5 {
// CHECK:STDOUT:       %.80a: %i32 = specific_inst @F.pack.%.loc8_7.3, @F.pack(%int_0)
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %G.call.da5: init %empty_tuple.type = call %G.specific_fn.ef124f.3(%.80a)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.782: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT:   %inst.splice_block.876: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.117: init %empty_tuple.type = splice_block %.60f {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.c5b: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.66c: init %empty_tuple.type = splice_block %G.call.92d {
// CHECK:STDOUT:       %.cfc: bool = specific_inst @F.pack.%.loc9_7.3, @F.pack(%int_0)
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %G.call.92d: init %empty_tuple.type = call %G.specific_fn.69fb3f.3(%.cfc)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.d86: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.27d: init %empty_tuple.type = splice_block %.66c {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.specific_inst.93c: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.354 = specific_inst @F.pack.%entry, @F.pack(%int_0)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %inst.splice_block.6c2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.125: init %empty_tuple.type = splice_block %G.call.e25 {
// CHECK:STDOUT:       %.2e7: bool = specific_inst @F.pack.%.loc8_7.3, @F.pack(%int_1)
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %G.call.e25: init %empty_tuple.type = call %G.specific_fn.69fb3f.2(%.2e7)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.cd1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.ea0: init %empty_tuple.type = splice_block %.125 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.d7e: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.01b: init %empty_tuple.type = splice_block %G.call.d2a {
// CHECK:STDOUT:       %.ec2: %i32 = specific_inst @F.pack.%.loc9_7.3, @F.pack(%int_1)
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %G.call.d2a: init %empty_tuple.type = call %G.specific_fn.ef124f.2(%.ec2)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.45d: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.d96: init %empty_tuple.type = splice_block %.01b {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.specific_inst.e54: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.85c = specific_inst @F.pack.%entry, @F.pack(%int_1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.c33: %tuple.type.413 = tuple_value (%inst.specific_inst.93c, %inst.specific_inst.e54) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cae = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.cae = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.541 = value_param_pattern [concrete = constants.%y.param_patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.541 = wrapper_binding_pattern y, %y.param_patt [concrete = constants.%y.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.6ed = value_param call_param0
// CHECK:STDOUT:     %.loc6_19.1: type = splice_block %.loc6_19.3 [concrete = constants.%tuple.type.6ed] {
// CHECK:STDOUT:       %i32.loc6_10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc6_15: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:       %.loc6_19.2: %tuple.type.693 = tuple_literal (%i32.loc6_10, %.loc6_15) [concrete = constants.%tuple.760]
// CHECK:STDOUT:       %.loc6_19.3: type = converted %.loc6_19.2, constants.%tuple.type.6ed [concrete = constants.%tuple.type.6ed]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.6ed = wrapper_binding x, %x.param
// CHECK:STDOUT:     %y.param: %tuple.type.68a = value_param call_param1
// CHECK:STDOUT:     %.loc6_35.1: type = splice_block %.loc6_35.3 [concrete = constants.%tuple.type.68a] {
// CHECK:STDOUT:       %.loc6_26: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:       %i32.loc6_32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc6_35.2: %tuple.type.693 = tuple_literal (%.loc6_26, %i32.loc6_32) [concrete = constants.%tuple.dcd]
// CHECK:STDOUT:       %.loc6_35.3: type = converted %.loc6_35.2, constants.%tuple.type.68a [concrete = constants.%tuple.type.68a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %tuple.type.68a = wrapper_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %tuple.type.6ed, %y.param: %tuple.type.68a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %pack: %tuple.type.413 = pack_expansion @F.pack(%pack_index.loc7_3.2: Core.IntLiteral), %entry [concrete = constants.%tuple.c33] {
// CHECK:STDOUT:     %pack_index.loc7_3.1: Core.IntLiteral = symbolic_binding <pack index>, 0 [symbolic = %pack_index.loc7_3.1 (constants.%pack_index)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !definition:
// CHECK:STDOUT:     %.loc8_7.1: type = tuple_index constants.%tuple.type.6ed, %pack_index.loc7_3.1 [template = %.loc8_7.1 (constants.%.2d7)]
// CHECK:STDOUT:     %.loc8_15.1: <instruction> = call_action (%G.ref.loc8, %.loc8_7.3), false [template]
// CHECK:STDOUT:     %.loc8_15.2: type = type_of_inst %.loc8_15.1 [template = %.loc8_15.2 (constants.%.f92)]
// CHECK:STDOUT:     %.loc8_15.3: @F.pack.%.loc8_15.2 (%.f92) = splice_inst %.loc8_15.1 [template = %.loc8_15.3 (constants.%.d40)]
// CHECK:STDOUT:     %require_complete.loc8: <witness> = require_complete_type %.loc8_15.2 [template = %require_complete.loc8 (constants.%require_complete.1e6)]
// CHECK:STDOUT:     %.loc8_15.4: <instruction> = convert_to_category_action %.loc8_15.7, element9 [template]
// CHECK:STDOUT:     %.loc8_15.5: @F.pack.%.loc8_15.2 (%.f92) = splice_inst %.loc8_15.4 [template = %.loc8_15.5 (constants.%.30c)]
// CHECK:STDOUT:     %.loc9_7.1: type = tuple_index constants.%tuple.type.68a, %pack_index.loc7_3.1 [template = %.loc9_7.1 (constants.%.26e)]
// CHECK:STDOUT:     %.loc9_15.1: <instruction> = call_action (%G.ref.loc9, %.loc9_7.3), false [template]
// CHECK:STDOUT:     %.loc9_15.2: type = type_of_inst %.loc9_15.1 [template = %.loc9_15.2 (constants.%.f0b)]
// CHECK:STDOUT:     %.loc9_15.3: @F.pack.%.loc9_15.2 (%.f0b) = splice_inst %.loc9_15.1 [template = %.loc9_15.3 (constants.%.a6d)]
// CHECK:STDOUT:     %require_complete.loc9: <witness> = require_complete_type %.loc9_15.2 [template = %require_complete.loc9 (constants.%require_complete.cfa)]
// CHECK:STDOUT:     %.loc9_15.4: <instruction> = convert_to_category_action %.loc9_15.7, element9 [template]
// CHECK:STDOUT:     %.loc9_15.5: @F.pack.%.loc9_15.2 (%.f0b) = splice_inst %.loc9_15.4 [template = %.loc9_15.5 (constants.%.ec9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !pack.entry:
// CHECK:STDOUT:     %pack_index.loc7_3.2: Core.IntLiteral = symbolic_binding <pack index>, 0 [symbolic = %pack_index.loc7_3.1 (constants.%pack_index)]
// CHECK:STDOUT:     %entry = br !pack.body
// CHECK:STDOUT:
// CHECK:STDOUT:   !pack.body:
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then:
// CHECK:STDOUT:     %G.ref.loc8: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:     %x.ref: %tuple.type.6ed = name_ref x, @F.%x
// CHECK:STDOUT:     %.loc8_7.2: type = tuple_index constants.%tuple.type.6ed, %pack_index.loc7_3.2 [template = %.loc8_7.1 (constants.%.2d7)]
// CHECK:STDOUT:     %.loc8_7.3: @F.pack.%.loc8_7.1 (%.2d7) = tuple_index %x.ref, %pack_index.loc7_3.2
// CHECK:STDOUT:     %.loc8_15.6: type = type_of_inst %.loc8_15.1 [template = %.loc8_15.2 (constants.%.f92)]
// CHECK:STDOUT:     %.loc8_15.7: @F.pack.%.loc8_15.2 (%.f92) = splice_inst %.loc8_15.1 [template = %.loc8_15.3 (constants.%.d40)]
// CHECK:STDOUT:     %.loc8_15.8: @F.pack.%.loc8_15.2 (%.f92) = splice_inst %.loc8_15.4 [template = %.loc8_15.5 (constants.%.30c)]
// CHECK:STDOUT:     %G.ref.loc9: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:     %y.ref: %tuple.type.68a = name_ref y, @F.%y
// CHECK:STDOUT:     %.loc9_7.2: type = tuple_index constants.%tuple.type.68a, %pack_index.loc7_3.2 [template = %.loc9_7.1 (constants.%.26e)]
// CHECK:STDOUT:     %.loc9_7.3: @F.pack.%.loc9_7.1 (%.26e) = tuple_index %y.ref, %pack_index.loc7_3.2
// CHECK:STDOUT:     %.loc9_15.6: type = type_of_inst %.loc9_15.1 [template = %.loc9_15.2 (constants.%.f0b)]
// CHECK:STDOUT:     %.loc9_15.7: @F.pack.%.loc9_15.2 (%.f0b) = splice_inst %.loc9_15.1 [template = %.loc9_15.3 (constants.%.a6d)]
// CHECK:STDOUT:     %.loc9_15.8: @F.pack.%.loc9_15.2 (%.f0b) = splice_inst %.loc9_15.4 [template = %.loc9_15.5 (constants.%.ec9)]
// CHECK:STDOUT:     br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.else:
// CHECK:STDOUT:     br_next_index
// CHECK:STDOUT:   }
// CHECK:STDOUT:   splice_br %pack, !pack.done
// CHECK:STDOUT:
// CHECK:STDOUT: !pack.done:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%pack_index) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%pack_index
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%int_0) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%int_0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc8_7.1 => constants.%i32
// CHECK:STDOUT:   %.loc8_15.1 => constants.%inst.splice_block.008
// CHECK:STDOUT:   %.loc8_15.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc8_15.3 => invalid
// CHECK:STDOUT:   %require_complete.loc8 => constants.%complete_type.782
// CHECK:STDOUT:   %.loc8_15.4 => constants.%inst.splice_block.876
// CHECK:STDOUT:   %.loc8_15.5 => invalid
// CHECK:STDOUT:   %.loc9_7.1 => bool
// CHECK:STDOUT:   %.loc9_15.1 => constants.%inst.splice_block.c5b
// CHECK:STDOUT:   %.loc9_15.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_15.3 => invalid
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.782
// CHECK:STDOUT:   %.loc9_15.4 => constants.%inst.splice_block.d86
// CHECK:STDOUT:   %.loc9_15.5 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%int_1) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%int_1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc8_7.1 => bool
// CHECK:STDOUT:   %.loc8_15.1 => constants.%inst.splice_block.6c2
// CHECK:STDOUT:   %.loc8_15.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc8_15.3 => invalid
// CHECK:STDOUT:   %require_complete.loc8 => constants.%complete_type.782
// CHECK:STDOUT:   %.loc8_15.4 => constants.%inst.splice_block.cd1
// CHECK:STDOUT:   %.loc8_15.5 => invalid
// CHECK:STDOUT:   %.loc9_7.1 => constants.%i32
// CHECK:STDOUT:   %.loc9_15.1 => constants.%inst.splice_block.d7e
// CHECK:STDOUT:   %.loc9_15.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_15.3 => invalid
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.782
// CHECK:STDOUT:   %.loc9_15.4 => constants.%inst.splice_block.45d
// CHECK:STDOUT:   %.loc9_15.5 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.3df: %tuple.type.693 = tuple_value (%T, %T) [symbolic]
// CHECK:STDOUT:   %tuple.type.07a: type = tuple_type (%T, %T) [symbolic]
// CHECK:STDOUT:   %pattern_type.c3f: type = pattern_type %tuple.type.07a [symbolic]
// CHECK:STDOUT:   %x.param_patt.13d: %pattern_type.c3f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.8f3: %pattern_type.c3f = wrapper_binding_pattern x, %x.param_patt.13d [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.22a: <witness> = require_complete_type %tuple.type.07a [symbolic]
// CHECK:STDOUT:   %pack_index: Core.IntLiteral = symbolic_binding <pack index>, 1 [symbolic]
// CHECK:STDOUT:   %G.specific_fn.058: <specific function> = specific_function %G, @G(%T) [symbolic]
// CHECK:STDOUT:   %tuple.type.413: type = tuple_type (<instruction>, <instruction>) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %inst.specific_inst.11b: <instruction> = inst_value [symbolic] {
// CHECK:STDOUT:     %.1a2 = specific_inst @F.pack.%entry, @F.pack(%T, %int_0)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %inst.specific_inst.2a3: <instruction> = inst_value [symbolic] {
// CHECK:STDOUT:     %.bd4 = specific_inst @F.pack.%entry, @F.pack(%T, %int_1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.95b: %tuple.type.413 = tuple_value (%inst.specific_inst.11b, %inst.specific_inst.2a3) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.94e: %tuple.type.693 = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.e55: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.394: type = pattern_type %tuple.type.e55 [concrete]
// CHECK:STDOUT:   %x.param_patt.8a8: %pattern_type.394 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.7e4: %pattern_type.394 = wrapper_binding_pattern x, %x.param_patt.8a8 [concrete]
// CHECK:STDOUT:   %complete_type.6c4: <witness> = complete_type_witness %tuple.type.e55 [concrete]
// CHECK:STDOUT:   %G.specific_fn.ef1: <specific function> = specific_function %G, @G(%i32) [concrete]
// CHECK:STDOUT:   %inst.specific_inst.238: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.bda = specific_inst @F.pack.%entry, @F.pack(%i32, %int_0)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.specific_inst.169: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.368 = specific_inst @F.pack.%entry, @F.pack(%i32, %int_1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.fcb: %tuple.type.413 = tuple_value (%inst.specific_inst.238, %inst.specific_inst.169) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc6_16.1: @F.%pattern_type (%pattern_type.c3f) = value_param_pattern [symbolic = %x.param_patt.loc6_16.2 (constants.%x.param_patt.13d)]
// CHECK:STDOUT:     %x.patt.loc6_16.1: @F.%pattern_type (%pattern_type.c3f) = wrapper_binding_pattern x, %x.param_patt.loc6_16.1 [symbolic = %x.patt.loc6_16.2 (constants.%x.patt.8f3)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%tuple.type (%tuple.type.07a) = value_param call_param0
// CHECK:STDOUT:     %.loc6_23.1: type = splice_block %.loc6_23.3 [symbolic = %tuple.type (constants.%tuple.type.07a)] {
// CHECK:STDOUT:       %T.ref.loc6_19: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.ref.loc6_22: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %.loc6_23.2: %tuple.type.693 = tuple_literal (%T.ref.loc6_19, %T.ref.loc6_22) [symbolic = %tuple.loc6 (constants.%tuple.3df)]
// CHECK:STDOUT:       %.loc6_23.3: type = converted %.loc6_23.2, constants.%tuple.type.07a [symbolic = %tuple.type (constants.%tuple.type.07a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%tuple.type (%tuple.type.07a) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %tuple.loc6: %tuple.type.693 = tuple_value (%T.loc6_7.1, %T.loc6_7.1) [symbolic = %tuple.loc6 (constants.%tuple.3df)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc6_7.1, %T.loc6_7.1) [symbolic = %tuple.type (constants.%tuple.type.07a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [symbolic = %pattern_type (constants.%pattern_type.c3f)]
// CHECK:STDOUT:   %x.param_patt.loc6_16.2: @F.%pattern_type (%pattern_type.c3f) = value_param_pattern [symbolic = %x.param_patt.loc6_16.2 (constants.%x.param_patt.13d)]
// CHECK:STDOUT:   %x.patt.loc6_16.2: @F.%pattern_type (%pattern_type.c3f) = wrapper_binding_pattern x, %x.param_patt.loc6_16.2 [symbolic = %x.patt.loc6_16.2 (constants.%x.patt.8f3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.22a)]
// CHECK:STDOUT:   %inst.specific_inst.loc7_3.1: <instruction> = inst_value [symbolic = %inst.specific_inst.loc7_3.1 (constants.%inst.specific_inst.11b)] {
// CHECK:STDOUT:     %.loc7_3.1 = specific_inst @F.pack.%entry, @F.pack(%T.loc6_7.1, constants.%int_0)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.specific_inst.loc7_3.2: <instruction> = inst_value [symbolic = %inst.specific_inst.loc7_3.2 (constants.%inst.specific_inst.2a3)] {
// CHECK:STDOUT:     %.loc7_3.2 = specific_inst @F.pack.%entry, @F.pack(%T.loc6_7.1, constants.%int_1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.loc7: %tuple.type.413 = tuple_value (%inst.specific_inst.loc7_3.1, %inst.specific_inst.loc7_3.2) [symbolic = %tuple.loc7 (constants.%tuple.95b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%tuple.type (%tuple.type.07a)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %pack: %tuple.type.413 = pack_expansion @F.pack(@F.%T.loc6_7.2: type, %pack_index.loc7_3.2: Core.IntLiteral), %entry [symbolic = @F.%tuple.loc7 (constants.%tuple.95b)] {
// CHECK:STDOUT:       %pack_index.loc7_3.1: Core.IntLiteral = symbolic_binding <pack index>, 1 [symbolic = %pack_index.loc7_3.1 (constants.%pack_index)]
// CHECK:STDOUT:
// CHECK:STDOUT:     !definition:
// CHECK:STDOUT:       %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %tuple.type: type = tuple_type (%T, %T) [symbolic = %tuple.type (constants.%tuple.type.07a)]
// CHECK:STDOUT:       %G.specific_fn.loc7_7.1: <specific function> = specific_function constants.%G, @G(%T) [symbolic = %G.specific_fn.loc7_7.1 (constants.%G.specific_fn.058)]
// CHECK:STDOUT:
// CHECK:STDOUT:     !pack.entry:
// CHECK:STDOUT:       %pack_index.loc7_3.2: Core.IntLiteral = symbolic_binding <pack index>, 1 [symbolic = %pack_index.loc7_3.1 (constants.%pack_index)]
// CHECK:STDOUT:       %entry = br !pack.body
// CHECK:STDOUT:
// CHECK:STDOUT:     !pack.body:
// CHECK:STDOUT:       %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:       %x.ref: @F.pack.%tuple.type (%tuple.type.07a) = name_ref x, @F.%x
// CHECK:STDOUT:       %.loc7: @F.pack.%T (%T) = tuple_index %x.ref, %pack_index.loc7_3.2
// CHECK:STDOUT:       %G.specific_fn.loc7_7.2: <specific function> = specific_function %G.ref, @G(constants.%T) [symbolic = %G.specific_fn.loc7_7.1 (constants.%G.specific_fn.058)]
// CHECK:STDOUT:       %G.call: init %empty_tuple.type = call %G.specific_fn.loc7_7.2(%.loc7)
// CHECK:STDOUT:       br_next_index
// CHECK:STDOUT:     }
// CHECK:STDOUT:     splice_br %pack, !pack.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !pack.done:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %tuple.loc6 => constants.%tuple.3df
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.07a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c3f
// CHECK:STDOUT:   %x.param_patt.loc6_16.2 => constants.%x.param_patt.13d
// CHECK:STDOUT:   %x.patt.loc6_16.2 => constants.%x.patt.8f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%T, constants.%pack_index) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%pack_index
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%T, constants.%int_0) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%int_0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%T, constants.%int_1) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%int_1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%i32
// CHECK:STDOUT:   %tuple.loc6 => constants.%tuple.94e
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.e55
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.394
// CHECK:STDOUT:   %x.param_patt.loc6_16.2 => constants.%x.param_patt.8a8
// CHECK:STDOUT:   %x.patt.loc6_16.2 => constants.%x.patt.7e4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.6c4
// CHECK:STDOUT:   %inst.specific_inst.loc7_3.1 => constants.%inst.specific_inst.238
// CHECK:STDOUT:   %inst.specific_inst.loc7_3.2 => constants.%inst.specific_inst.169
// CHECK:STDOUT:   %tuple.loc7 => constants.%tuple.fcb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%i32, constants.%int_0) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%int_0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.e55
// CHECK:STDOUT:   %G.specific_fn.loc7_7.1 => constants.%G.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.pack(constants.%i32, constants.%int_1) {
// CHECK:STDOUT:   %pack_index.loc7_3.1 => constants.%int_1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.e55
// CHECK:STDOUT:   %G.specific_fn.loc7_7.1 => constants.%G.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
