// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/basics.carbon

// --- assign_var.carbon

library "[[@TEST_NAME]]";

var a: ((), (), ());
//@dump-sem-ir-begin
var b: array((), 3) = a;
//@dump-sem-ir-end

// --- assign_arity.carbon

var a0: array((), 0) = ();
var a1: array((), 1) = ((),);
var a2: array((), 2) = ((), ());

// --- array_in_place.carbon

library "[[@TEST_NAME]]";

class C {}

fn F() -> (C, C, C);

fn G() {
  //@dump-sem-ir-begin
  var v: array((C, C, C), 2) = (F(), F());
  //@dump-sem-ir-end
}

// --- array_vs_tuple.carbon

library "[[@TEST_NAME]]";

fn G() {
  // These should have two different constant values.
  //@dump-sem-ir-begin
  var a: array((), 3);
  var b: ((), (), ());
  //@dump-sem-ir-end
}

// --- assign_return_value.carbon

library "[[@TEST_NAME]]";

fn F() -> ((),) { return ((),); }

fn Run() {
  //@dump-sem-ir-begin
  var t: array((), 1) = F();
  //@dump-sem-ir-end
}

// --- nine_elements.carbon

library "[[@TEST_NAME]]";

var a: ((), (), (), (), (), (), (), (), ()) =
    ((), (), (), (), (), (), (), (), ());

// Regression test for APInt handling.
//@dump-sem-ir-begin
var b: array((), 9) = a;
//@dump-sem-ir-end

// --- fail_incomplete_element.carbon

library "[[@TEST_NAME]]";

class Incomplete;

// CHECK:STDERR: fail_incomplete_element.carbon:[[@LINE+7]]:8: error: binding pattern has incomplete type `array(Incomplete, 1)` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var a: array(Incomplete, 1);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_incomplete_element.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: array(Incomplete, 1);

var p: Incomplete* = &a[0];

// --- fail_invalid_element.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_invalid_element.carbon:[[@LINE+7]]:14: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: var a: array(1, 1);
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_invalid_element.carbon:[[@LINE+4]]:14: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var a: array(1, 1);
// CHECK:STDERR:              ^
// CHECK:STDERR:
var a: array(1, 1);

// CHECK:STDOUT: --- assign_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.035: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access file.%a.var, element0 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access file.%a.var, element1 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.elem2: ref %empty_tuple.type = tuple_access file.%a.var, element2 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%empty_tuple, %empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.035 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.035 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %array_type = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_19: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %.loc6_15.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %.loc6_15.2: type = converted %.loc6_15.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %.loc6_15.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %array_type = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: ref %tuple.type = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access %a.ref, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc6_23.1: ref %empty_tuple.type = array_index file.%b.var, %int_0
// CHECK:STDOUT:   %.loc6_23.2: init %empty_tuple.type to %.loc6_23.1 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_23.3: init %empty_tuple.type = converted %tuple.elem0, %.loc6_23.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %a.ref, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc6_23.4: ref %empty_tuple.type = array_index file.%b.var, %int_1
// CHECK:STDOUT:   %.loc6_23.5: init %empty_tuple.type to %.loc6_23.4 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_23.6: init %empty_tuple.type = converted %tuple.elem1, %.loc6_23.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem2: ref %empty_tuple.type = tuple_access %a.ref, element2 [concrete = constants.%tuple.elem2]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc6_23.7: ref %empty_tuple.type = array_index file.%b.var, %int_2
// CHECK:STDOUT:   %.loc6_23.8: init %empty_tuple.type to %.loc6_23.7 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_23.9: init %empty_tuple.type = converted %tuple.elem2, %.loc6_23.8 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_23.10: init %array_type to file.%b.var = array_init (%.loc6_23.3, %.loc6_23.6, %.loc6_23.9) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc6_1: init %array_type = converted %a.ref, %.loc6_23.10 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%b.var, %.loc6_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- array_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.ff9 = tuple_value (%C, %C, %C) [concrete]
// CHECK:STDOUT:   %tuple.type.e56: type = tuple_type (%C, %C, %C) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %tuple.type.e56 [concrete]
// CHECK:STDOUT:   %pattern_type.709: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %tuple.type.708: type = tuple_type (%tuple.type.e56, %tuple.type.e56) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.709 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.709 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %array_type = var %v.var_patt
// CHECK:STDOUT:   %F.ref.loc10_33: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc10_41.1: ref %tuple.type.e56 = splice_block %.loc10_41.6 {
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %.loc10_41.6: ref %tuple.type.e56 = array_index %v.var, %int_0
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.call.loc10_35: init %tuple.type.e56 to %.loc10_41.1 = call %F.ref.loc10_33()
// CHECK:STDOUT:   %F.ref.loc10_38: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc10_41.2: ref %tuple.type.e56 = splice_block %.loc10_41.5 {
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %.loc10_41.5: ref %tuple.type.e56 = array_index %v.var, %int_1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.call.loc10_40: init %tuple.type.e56 to %.loc10_41.2 = call %F.ref.loc10_38()
// CHECK:STDOUT:   %.loc10_41.3: %tuple.type.708 = tuple_literal (%F.call.loc10_35, %F.call.loc10_40)
// CHECK:STDOUT:   %.loc10_41.4: init %array_type to %v.var = array_init (%F.call.loc10_35, %F.call.loc10_40)
// CHECK:STDOUT:   %.loc10_3: init %array_type = converted %.loc10_41.3, %.loc10_41.4
// CHECK:STDOUT:   assign %v.var, %.loc10_3
// CHECK:STDOUT:   %.loc10_28: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %C.ref.loc10_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %C.ref.loc10_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %C.ref.loc10_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc10_24.1: %tuple.type.ff9 = tuple_literal (%C.ref.loc10_17, %C.ref.loc10_20, %C.ref.loc10_23) [concrete = constants.%tuple]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %.loc10_24.2: type = converted %.loc10_24.1, constants.%tuple.type.e56 [concrete = constants.%tuple.type.e56]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, %.loc10_24.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %array_type = ref_binding v, %v.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %v.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%v.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- array_vs_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.035: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type.8c1: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc8 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc7 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.035 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.035 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %.loc7_21: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %.loc7_17.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %.loc7_17.2: type = converted %.loc7_17.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %.loc7_17.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.8c1 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.8c1 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type = var %b.var_patt
// CHECK:STDOUT:   %.loc8_21.1: type = splice_block %.loc8_21.6 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc8_12: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_16: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_20: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_21.2: %tuple.type = tuple_literal (%.loc8_12, %.loc8_16, %.loc8_20) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc8_21.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_21.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_21.5: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_21.6: type = converted %.loc8_21.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type = ref_binding b, %b.var
// CHECK:STDOUT:   %DestroyOp.bound.loc8: <bound method> = bound_method %b.var, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc8: init %empty_tuple.type = call %DestroyOp.bound.loc8(%b.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc7: <bound method> = bound_method %a.var, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc7: init %empty_tuple.type = call %DestroyOp.bound.loc7(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc8(%self.param: %tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc7(%self.param: %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- assign_return_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc8_27 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc8_3 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.fe8 = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.fe8 = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %array_type = var %t.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %tuple.type = call %F.ref()
// CHECK:STDOUT:   %.loc8_27.1: ref %tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc8_27.2: ref %tuple.type = temporary %.loc8_27.1, %F.call
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access %.loc8_27.2, element0
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc8_27.3: ref %empty_tuple.type = array_index %t.var, %int_0
// CHECK:STDOUT:   %.loc8_27.4: init %empty_tuple.type to %.loc8_27.3 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_27.5: init %empty_tuple.type = converted %tuple.elem0, %.loc8_27.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_27.6: init %array_type to %t.var = array_init (%.loc8_27.5) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc8_3: init %array_type = converted %F.call, %.loc8_27.6 [concrete = constants.%array]
// CHECK:STDOUT:   assign %t.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_21: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %.loc8_17.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %.loc8_17.2: type = converted %.loc8_17.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %array_type: type = array_type %int_1, %.loc8_17.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %array_type = ref_binding t, %t.var
// CHECK:STDOUT:   %DestroyOp.bound.loc8_27: <bound method> = bound_method %.loc8_27.2, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc8_27: init %empty_tuple.type = call %DestroyOp.bound.loc8_27(%.loc8_27.2)
// CHECK:STDOUT:   %DestroyOp.bound.loc8_3: <bound method> = bound_method %t.var, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc8_3: init %empty_tuple.type = call %DestroyOp.bound.loc8_3(%t.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc8_27(%self.param: %tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc8_3(%self.param: %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- nine_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access file.%a.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access file.%a.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem2: ref %empty_tuple.type = tuple_access file.%a.var, element2 [concrete]
// CHECK:STDOUT:   %tuple.elem3: ref %empty_tuple.type = tuple_access file.%a.var, element3 [concrete]
// CHECK:STDOUT:   %tuple.elem4: ref %empty_tuple.type = tuple_access file.%a.var, element4 [concrete]
// CHECK:STDOUT:   %tuple.elem5: ref %empty_tuple.type = tuple_access file.%a.var, element5 [concrete]
// CHECK:STDOUT:   %tuple.elem6: ref %empty_tuple.type = tuple_access file.%a.var, element6 [concrete]
// CHECK:STDOUT:   %tuple.elem7: ref %empty_tuple.type = tuple_access file.%a.var, element7 [concrete]
// CHECK:STDOUT:   %tuple.elem8: ref %empty_tuple.type = tuple_access file.%a.var, element8 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_9, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.3db: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.3db = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.3db = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %array_type = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc9_19: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %.loc9_15.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:     %.loc9_15.2: type = converted %.loc9_15.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %array_type: type = array_type %int_9, %.loc9_15.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %array_type = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref: ref %tuple.type = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %tuple.elem0.loc9: ref %empty_tuple.type = tuple_access %a.ref, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_23.1: ref %empty_tuple.type = array_index file.%b.var, %int_0
// CHECK:STDOUT:   %.loc9_23.2: init %empty_tuple.type to %.loc9_23.1 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_23.3: init %empty_tuple.type = converted %tuple.elem0.loc9, %.loc9_23.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc9: ref %empty_tuple.type = tuple_access %a.ref, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_23.4: ref %empty_tuple.type = array_index file.%b.var, %int_1
// CHECK:STDOUT:   %.loc9_23.5: init %empty_tuple.type to %.loc9_23.4 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_23.6: init %empty_tuple.type = converted %tuple.elem1.loc9, %.loc9_23.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem2.loc9: ref %empty_tuple.type = tuple_access %a.ref, element2 [concrete = constants.%tuple.elem2]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_23.7: ref %empty_tuple.type = array_index file.%b.var, %int_2
// CHECK:STDOUT:   %.loc9_23.8: init %empty_tuple.type to %.loc9_23.7 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_23.9: init %empty_tuple.type = converted %tuple.elem2.loc9, %.loc9_23.8 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem3.loc9: ref %empty_tuple.type = tuple_access %a.ref, element3 [concrete = constants.%tuple.elem3]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc9_23.10: ref %empty_tuple.type = array_index file.%b.var, %int_3
// CHECK:STDOUT:   %.loc9_23.11: init %empty_tuple.type to %.loc9_23.10 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_23.12: init %empty_tuple.type = converted %tuple.elem3.loc9, %.loc9_23.11 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem4.loc9: ref %empty_tuple.type = tuple_access %a.ref, element4 [concrete = constants.%tuple.elem4]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc9_23.13: ref %empty_tuple.type = array_index file.%b.var, %int_4
// CHECK:STDOUT:   %.loc9_23.14: init %empty_tuple.type to %.loc9_23.13 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_23.15: init %empty_tuple.type = converted %tuple.elem4.loc9, %.loc9_23.14 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem5.loc9: ref %empty_tuple.type = tuple_access %a.ref, element5 [concrete = constants.%tuple.elem5]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc9_23.16: ref %empty_tuple.type = array_index file.%b.var, %int_5
// CHECK:STDOUT:   %.loc9_23.17: init %empty_tuple.type to %.loc9_23.16 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_23.18: init %empty_tuple.type = converted %tuple.elem5.loc9, %.loc9_23.17 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem6.loc9: ref %empty_tuple.type = tuple_access %a.ref, element6 [concrete = constants.%tuple.elem6]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %.loc9_23.19: ref %empty_tuple.type = array_index file.%b.var, %int_6
// CHECK:STDOUT:   %.loc9_23.20: init %empty_tuple.type to %.loc9_23.19 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_23.21: init %empty_tuple.type = converted %tuple.elem6.loc9, %.loc9_23.20 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem7.loc9: ref %empty_tuple.type = tuple_access %a.ref, element7 [concrete = constants.%tuple.elem7]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc9_23.22: ref %empty_tuple.type = array_index file.%b.var, %int_7
// CHECK:STDOUT:   %.loc9_23.23: init %empty_tuple.type to %.loc9_23.22 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_23.24: init %empty_tuple.type = converted %tuple.elem7.loc9, %.loc9_23.23 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem8.loc9: ref %empty_tuple.type = tuple_access %a.ref, element8 [concrete = constants.%tuple.elem8]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %.loc9_23.25: ref %empty_tuple.type = array_index file.%b.var, %int_8
// CHECK:STDOUT:   %.loc9_23.26: init %empty_tuple.type to %.loc9_23.25 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_23.27: init %empty_tuple.type = converted %tuple.elem8.loc9, %.loc9_23.26 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_23.28: init %array_type to file.%b.var = array_init (%.loc9_23.3, %.loc9_23.6, %.loc9_23.9, %.loc9_23.12, %.loc9_23.15, %.loc9_23.18, %.loc9_23.21, %.loc9_23.24, %.loc9_23.27) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_1: init %array_type = converted %a.ref, %.loc9_23.28 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%b.var, %.loc9_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
