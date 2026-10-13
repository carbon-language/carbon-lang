// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_redeclared_no_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_redeclared_no_type.carbon

// Regression test for #7160.

// --- fail_redeclared_type_init.carbon

library "[[@TEST_NAME]]";

class AClassType {
  fn Make() -> AClassType {
    return {};
  }
}

var some_object_defined_earlier: AClassType = AClassType.Make();

// CHECK:STDERR: fail_redeclared_type_init.carbon:[[@LINE+8]]:33: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var some_object_defined_earlier = AClassType;
// CHECK:STDERR:                                 ^
// CHECK:STDERR:
// CHECK:STDERR: fail_redeclared_type_init.carbon:[[@LINE+4]]:33: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: var some_object_defined_earlier = AClassType;
// CHECK:STDERR:                                 ^
// CHECK:STDERR:
var some_object_defined_earlier = AClassType;

// --- fail_redeclared_call_init.carbon

library "[[@TEST_NAME]]";

class AClassType {
  fn Make() -> AClassType {
    return {};
  }
}

var some_object_defined_earlier: AClassType = AClassType.Make();

// CHECK:STDERR: fail_redeclared_call_init.carbon:[[@LINE+8]]:33: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var some_object_defined_earlier = AClassType.Make();
// CHECK:STDERR:                                 ^
// CHECK:STDERR:
// CHECK:STDERR: fail_redeclared_call_init.carbon:[[@LINE+4]]:33: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: var some_object_defined_earlier = AClassType.Make();
// CHECK:STDERR:                                 ^
// CHECK:STDERR:
var some_object_defined_earlier = AClassType.Make();

// --- fail_first_def_no_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_first_def_no_type.carbon:[[@LINE+8]]:7: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var x = ();
// CHECK:STDERR:       ^
// CHECK:STDERR:
// CHECK:STDERR: fail_first_def_no_type.carbon:[[@LINE+4]]:7: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: var x = ();
// CHECK:STDERR:       ^
// CHECK:STDERR:
var x = ();

// --- fail_expr_pattern_in_var.carbon

library "[[@TEST_NAME]]";

class AClassType {
  fn Make() -> AClassType {
    return {};
  }
}

// Exercises `GetFirstBindingNameFromPatternId` returning `EntityNameId::None`
// when a `var` pattern wraps an `ExprPattern` without any bindings.
//@dump-sem-ir-begin
// CHECK:STDERR: fail_expr_pattern_in_var.carbon:[[@LINE+4]]:6: error: semantics TODO: `expression pattern` [SemanticsTodo]
// CHECK:STDERR: var (AClassType.Make()) = AClassType.Make();
// CHECK:STDERR:      ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var (AClassType.Make()) = AClassType.Make();
//@dump-sem-ir-end

// --- fail_tuple_expr_pattern_in_var.carbon

library "[[@TEST_NAME]]";

class AClassType {
  fn Make() -> AClassType {
    return {};
  }
}

// Exercises `GetFirstBindingNameFromPatternId` continuing past a leading
// `ExprPattern` in a `TuplePattern` to find the first binding `x`.
//@dump-sem-ir-begin
// CHECK:STDERR: fail_tuple_expr_pattern_in_var.carbon:[[@LINE+4]]:6: error: semantics TODO: `expression pattern` [SemanticsTodo]
// CHECK:STDERR: var (AClassType.Make(), x: ()) = (AClassType.Make(), ());
// CHECK:STDERR:      ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var (AClassType.Make(), x: ()) = (AClassType.Make(), ());
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_expr_pattern_in_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AClassType: type = class_type @AClassType [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %AClassType [concrete]
// CHECK:STDOUT:   %AClassType.Make.type: type = fn_type @AClassType.Make [concrete]
// CHECK:STDOUT:   %AClassType.Make: %AClassType.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %.var_patt: %pattern_type = var_pattern file.%expr_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.var: ref %AClassType = var_storage %.var_patt
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %expr_patt: %pattern_type = expr_pattern %AClassType.Make.call in [concrete] {
// CHECK:STDOUT:       %AClassType.ref: type = name_ref AClassType, %AClassType.decl [concrete = constants.%AClassType]
// CHECK:STDOUT:       %Make.ref: %AClassType.Make.type = name_ref Make, @AClassType.%AClassType.Make.decl [concrete = constants.%AClassType.Make]
// CHECK:STDOUT:       %.loc17: ref %AClassType = temporary_storage
// CHECK:STDOUT:       %AClassType.Make.call: init %AClassType to %.loc17 = call %Make.ref()
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var_patt: %pattern_type = var_pattern %expr_patt [concrete = constants.%.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AClassType.ref: type = name_ref AClassType, file.%AClassType.decl [concrete = constants.%AClassType]
// CHECK:STDOUT:   %Make.ref: %AClassType.Make.type = name_ref Make, @AClassType.%AClassType.Make.decl [concrete = constants.%AClassType.Make]
// CHECK:STDOUT:   %.loc17: ref %AClassType = splice_block file.%.var {}
// CHECK:STDOUT:   %AClassType.Make.call: init %AClassType to %.loc17 = call %Make.ref()
// CHECK:STDOUT:   assign file.%.var, %AClassType.Make.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_tuple_expr_pattern_in_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AClassType: type = class_type @AClassType [concrete]
// CHECK:STDOUT:   %pattern_type.d11: type = pattern_type %AClassType [concrete]
// CHECK:STDOUT:   %AClassType.Make.type: type = fn_type @AClassType.Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AClassType.Make: %AClassType.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %tuple.type.b2d: type = tuple_type (%AClassType, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.6d3: type = pattern_type %tuple.type.b2d [concrete]
// CHECK:STDOUT:   %.010: %pattern_type.6d3 = tuple_pattern (file.%expr_patt, %x.patt) [concrete]
// CHECK:STDOUT:   %.var_patt: %pattern_type.6d3 = var_pattern %.010 [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %AClassType = tuple_access file.%.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access file.%.var, element1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.var: ref %tuple.type.b2d = var_storage %.var_patt [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %AClassType = tuple_access %.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc17_29.1: type = splice_block %.loc17_29.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc17_29.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17_29.3: type = converted %.loc17_29.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = wrapper_binding x, %tuple.elem1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %expr_patt: %pattern_type.d11 = expr_pattern %AClassType.Make.call in [concrete] {
// CHECK:STDOUT:       %AClassType.ref: type = name_ref AClassType, %AClassType.decl [concrete = constants.%AClassType]
// CHECK:STDOUT:       %Make.ref: %AClassType.Make.type = name_ref Make, @AClassType.%AClassType.Make.decl [concrete = constants.%AClassType.Make]
// CHECK:STDOUT:       %.loc17_22: ref %AClassType = temporary_storage
// CHECK:STDOUT:       %AClassType.Make.call: init %AClassType to %.loc17_22 = call %Make.ref()
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc17_30: %pattern_type.6d3 = tuple_pattern (%expr_patt, %x.patt) [concrete = constants.%.010]
// CHECK:STDOUT:     %.var_patt: %pattern_type.6d3 = var_pattern %.loc17_30 [concrete = constants.%.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AClassType.ref: type = name_ref AClassType, file.%AClassType.decl [concrete = constants.%AClassType]
// CHECK:STDOUT:   %Make.ref: %AClassType.Make.type = name_ref Make, @AClassType.%AClassType.Make.decl [concrete = constants.%AClassType.Make]
// CHECK:STDOUT:   %tuple.elem0: ref %AClassType = tuple_access file.%.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %AClassType.Make.call: init %AClassType to %tuple.elem0 = call %Make.ref()
// CHECK:STDOUT:   %.loc17_55.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_56.1: %tuple.type.b2d = tuple_literal (%AClassType.Make.call, %.loc17_55.1)
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access file.%.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc17_55.2: init %empty_tuple.type to %tuple.elem1 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_56.2: init %empty_tuple.type = converted %.loc17_55.1, %.loc17_55.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_56.3: init %empty_tuple.type to %tuple.elem1 = in_place_init %.loc17_56.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_56.4: init %tuple.type.b2d to file.%.var = tuple_init (%AClassType.Make.call, %.loc17_56.3)
// CHECK:STDOUT:   %.loc17_1: init %tuple.type.b2d = converted %.loc17_56.1, %.loc17_56.4
// CHECK:STDOUT:   assign file.%.var, %.loc17_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
