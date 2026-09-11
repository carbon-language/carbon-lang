// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/default_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/default_values.carbon

// --- fail_value_not_constant.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_value_not_constant.carbon:[[@LINE+4]]:23: error: default value for pattern must be constant [PatternDefaultValueNotConstant]
// CHECK:STDERR: fn H(x: i32, y: i32 = x);
// CHECK:STDERR:                       ^
// CHECK:STDERR:
fn H(x: i32, y: i32 = x);

// --- fail_type_mismatch.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:6: error: pattern type `i32` doesn't match default value expression type `str` [PatternDefaultValueTypeMismatch]
// CHECK:STDERR: fn K(x: i32 = "bazz");
// CHECK:STDERR:      ^~~~~~
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:15: note: default value expression here [PatternDefaultValueTypeMismatchNote]
// CHECK:STDERR: fn K(x: i32 = "bazz");
// CHECK:STDERR:               ^~~~~~
// CHECK:STDERR:
fn K(x: i32 = "bazz");

// --- fail_pattern_defaults_not_in_parameter_list.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_pattern_defaults_not_in_parameter_list.carbon:[[@LINE+4]]:23: error: default values are only supported in parameter lists [PatternDefaultValueNotInParameterList]
// CHECK:STDERR: let (y: i32, x: i32 = 0) = (1, 2);
// CHECK:STDERR:                       ^
// CHECK:STDERR:
let (y: i32, x: i32 = 0) = (1, 2);

// --- fail_required_default_values_missing.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+17]]:39: error: this pattern is missing a required default value. [RequiredPatternDefaultValueMissing]
// CHECK:STDERR: fn Z(v: i32, x: i32 = 3, (y: i32 = 2, z: i32), w: i32 = 4, k: i32);
// CHECK:STDERR:                                       ^~~~~~
// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+14]]:27: note: all patterns to the right of this first pattern with a default value must also specify a default value. [RequiredPatternDefaultValueFirstDefault]
// CHECK:STDERR: fn Z(v: i32, x: i32 = 3, (y: i32 = 2, z: i32), w: i32 = 4, k: i32);
// CHECK:STDERR:                           ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+10]]:26: error: this pattern is missing a required default value. [RequiredPatternDefaultValueMissing]
// CHECK:STDERR: fn Z(v: i32, x: i32 = 3, (y: i32 = 2, z: i32), w: i32 = 4, k: i32);
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+7]]:60: note: this pattern is also missing a required default value. [RequiredPatternDefaultValueMissingAdditional]
// CHECK:STDERR: fn Z(v: i32, x: i32 = 3, (y: i32 = 2, z: i32), w: i32 = 4, k: i32);
// CHECK:STDERR:                                                            ^~~~~~
// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+4]]:14: note: all patterns to the right of this first pattern with a default value must also specify a default value. [RequiredPatternDefaultValueFirstDefault]
// CHECK:STDERR: fn Z(v: i32, x: i32 = 3, (y: i32 = 2, z: i32), w: i32 = 4, k: i32);
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
fn Z(v: i32, x: i32 = 3, (y: i32 = 2, z: i32), w: i32 = 4, k: i32);

// --- fail_nesting_required_default_values_missing.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_nesting_required_default_values_missing.carbon:[[@LINE+14]]:40: error: this pattern is missing a required default value. [RequiredPatternDefaultValueMissing]
// CHECK:STDERR: fn F(a: i32, (b: i32 = 0, (c: i32 = 0, (d: i32, e: i32))));
// CHECK:STDERR:                                        ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nesting_required_default_values_missing.carbon:[[@LINE+11]]:28: note: all patterns to the right of this first pattern with a default value must also specify a default value. [RequiredPatternDefaultValueFirstDefault]
// CHECK:STDERR: fn F(a: i32, (b: i32 = 0, (c: i32 = 0, (d: i32, e: i32))));
// CHECK:STDERR:                            ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_nesting_required_default_values_missing.carbon:[[@LINE+7]]:27: error: this pattern is missing a required default value. [RequiredPatternDefaultValueMissing]
// CHECK:STDERR: fn F(a: i32, (b: i32 = 0, (c: i32 = 0, (d: i32, e: i32))));
// CHECK:STDERR:                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nesting_required_default_values_missing.carbon:[[@LINE+4]]:15: note: all patterns to the right of this first pattern with a default value must also specify a default value. [RequiredPatternDefaultValueFirstDefault]
// CHECK:STDERR: fn F(a: i32, (b: i32 = 0, (c: i32 = 0, (d: i32, e: i32))));
// CHECK:STDERR:               ^~~~~~
// CHECK:STDERR:
fn F(a: i32, (b: i32 = 0, (c: i32 = 0, (d: i32, e: i32))));

// --- nested.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(a: i32, (b: i32, (c: i32, (d: i32, e: i32))) = (1, (2, (3, 4))));
//@dump-sem-ir-end

// --- basic.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(x: i32 = 0);
//@dump-sem-ir-end

// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %b.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.6b6 = wrapper_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.6b6 = wrapper_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %d.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.6b6 = wrapper_binding_pattern d, %d.param_patt [concrete]
// CHECK:STDOUT:   %e.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.6b6 = wrapper_binding_pattern e, %e.param_patt [concrete]
// CHECK:STDOUT:   %tuple.type.e55: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.394: type = pattern_type %tuple.type.e55 [concrete]
// CHECK:STDOUT:   %.70d: %pattern_type.394 = tuple_pattern (%d.patt, %e.patt) [concrete]
// CHECK:STDOUT:   %tuple.type.920: type = tuple_type (%i32, %tuple.type.e55) [concrete]
// CHECK:STDOUT:   %pattern_type.e87: type = pattern_type %tuple.type.920 [concrete]
// CHECK:STDOUT:   %.aa9: %pattern_type.e87 = tuple_pattern (%c.patt, %.70d) [concrete]
// CHECK:STDOUT:   %tuple.type.007: type = tuple_type (%i32, %tuple.type.920) [concrete]
// CHECK:STDOUT:   %pattern_type.802: type = pattern_type %tuple.type.007 [concrete]
// CHECK:STDOUT:   %.203: %pattern_type.802 = tuple_pattern (%b.patt, %.aa9) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.302: %tuple.type.f94 = tuple_value (%int_3.1ba, %int_4.0c1) [concrete]
// CHECK:STDOUT:   %tuple.type.bd0: type = tuple_type (Core.IntLiteral, %tuple.type.f94) [concrete]
// CHECK:STDOUT:   %tuple.b26: %tuple.type.bd0 = tuple_value (%int_2.ecc, %tuple.302) [concrete]
// CHECK:STDOUT:   %tuple.type.c4f: type = tuple_type (Core.IntLiteral, %tuple.type.bd0) [concrete]
// CHECK:STDOUT:   %tuple.326: %tuple.type.c4f = tuple_value (%int_1.5b8, %tuple.b26) [concrete]
// CHECK:STDOUT:   %.19b: %pattern_type.802 = default_value_pattern %.203, index: 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aad: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.b7a: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.4eb: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.e16: %tuple.type.e55 = tuple_value (%int_3.410, %int_4.4eb) [concrete]
// CHECK:STDOUT:   %tuple.094: %tuple.type.920 = tuple_value (%int_2.295, %tuple.e16) [concrete]
// CHECK:STDOUT:   %tuple.4d3: %tuple.type.007 = tuple_value (%int_1.0c6, %tuple.094) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.6b6 = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.6b6 = wrapper_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%d.param_patt]
// CHECK:STDOUT:     %d.patt: %pattern_type.6b6 = wrapper_binding_pattern d, %d.param_patt [concrete = constants.%d.patt]
// CHECK:STDOUT:     %e.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%e.param_patt]
// CHECK:STDOUT:     %e.patt: %pattern_type.6b6 = wrapper_binding_pattern e, %e.param_patt [concrete = constants.%e.patt]
// CHECK:STDOUT:     %.loc4_47: %pattern_type.394 = tuple_pattern (%d.patt, %e.patt) [concrete = constants.%.70d]
// CHECK:STDOUT:     %.loc4_48: %pattern_type.e87 = tuple_pattern (%c.patt, %.loc4_47) [concrete = constants.%.aa9]
// CHECK:STDOUT:     %.loc4_49: %pattern_type.802 = tuple_pattern (%b.patt, %.loc4_48) [concrete = constants.%.203]
// CHECK:STDOUT:     %.loc4_51: %pattern_type.802 = default_value_pattern %.loc4_49, index: 0 [concrete = constants.%.19b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %.loc4_66.1: %tuple.type.f94 = tuple_literal (%int_3, %int_4) [concrete = constants.%tuple.302]
// CHECK:STDOUT:     %.loc4_67.1: %tuple.type.bd0 = tuple_literal (%int_2, %.loc4_66.1) [concrete = constants.%tuple.b26]
// CHECK:STDOUT:     %.loc4_68.1: %tuple.type.c4f = tuple_literal (%int_1, %.loc4_67.1) [concrete = constants.%tuple.326]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc4_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %a: %i32 = wrapper_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc4_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %b: %i32 = wrapper_binding b, %b.param
// CHECK:STDOUT:     %c.param: %i32 = value_param call_param2
// CHECK:STDOUT:     %i32.loc4_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %c: %i32 = wrapper_binding c, %c.param
// CHECK:STDOUT:     %d.param: %i32 = value_param call_param3
// CHECK:STDOUT:     %i32.loc4_36: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %d: %i32 = wrapper_binding d, %d.param
// CHECK:STDOUT:     %e.param: %i32 = value_param call_param4
// CHECK:STDOUT:     %i32.loc4_44: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %e: %i32 = wrapper_binding e, %e.param
// CHECK:STDOUT:     %tuple.loc4_47: %tuple.type.e55 = tuple_value (%d.param, %e.param)
// CHECK:STDOUT:     %tuple.loc4_48: %tuple.type.920 = tuple_value (%c.param, %tuple.loc4_47)
// CHECK:STDOUT:     %tuple.loc4_49: %tuple.type.007 = tuple_value (%b.param, %tuple.loc4_48)
// CHECK:STDOUT:     %impl.elem0.loc4_68: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc4_68.1: <bound method> = bound_method %int_1, %impl.elem0.loc4_68 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:     %specific_fn.loc4_68: <specific function> = specific_function %impl.elem0.loc4_68, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc4_68.2: <bound method> = bound_method %int_1, %specific_fn.loc4_68 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_68: init %i32 = call %bound_method.loc4_68.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %.loc4_68.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_68 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %.loc4_68.3: %i32 = converted %int_1, %.loc4_68.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %impl.elem0.loc4_67: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc4_67.1: <bound method> = bound_method %int_2, %impl.elem0.loc4_67 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:     %specific_fn.loc4_67: <specific function> = specific_function %impl.elem0.loc4_67, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc4_67.2: <bound method> = bound_method %int_2, %specific_fn.loc4_67 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_67: init %i32 = call %bound_method.loc4_67.2(%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %.loc4_67.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_67 [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %.loc4_67.3: %i32 = converted %int_2, %.loc4_67.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %impl.elem0.loc4_66.1: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc4_66.1: <bound method> = bound_method %int_3, %impl.elem0.loc4_66.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:     %specific_fn.loc4_66.1: <specific function> = specific_function %impl.elem0.loc4_66.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc4_66.2: <bound method> = bound_method %int_3, %specific_fn.loc4_66.1 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_66.1: init %i32 = call %bound_method.loc4_66.2(%int_3) [concrete = constants.%int_3.410]
// CHECK:STDOUT:     %.loc4_66.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_66.1 [concrete = constants.%int_3.410]
// CHECK:STDOUT:     %.loc4_66.3: %i32 = converted %int_3, %.loc4_66.2 [concrete = constants.%int_3.410]
// CHECK:STDOUT:     %impl.elem0.loc4_66.2: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc4_66.3: <bound method> = bound_method %int_4, %impl.elem0.loc4_66.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aad]
// CHECK:STDOUT:     %specific_fn.loc4_66.2: <specific function> = specific_function %impl.elem0.loc4_66.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc4_66.4: <bound method> = bound_method %int_4, %specific_fn.loc4_66.2 [concrete = constants.%bound_method.b7a]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_66.2: init %i32 = call %bound_method.loc4_66.4(%int_4) [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:     %.loc4_66.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_66.2 [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:     %.loc4_66.5: %i32 = converted %int_4, %.loc4_66.4 [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:     %tuple.loc4_66: %tuple.type.e55 = tuple_value (%.loc4_66.3, %.loc4_66.5) [concrete = constants.%tuple.e16]
// CHECK:STDOUT:     %.loc4_67.4: %tuple.type.e55 = converted %.loc4_66.1, %tuple.loc4_66 [concrete = constants.%tuple.e16]
// CHECK:STDOUT:     %tuple.loc4_67: %tuple.type.920 = tuple_value (%.loc4_67.3, %.loc4_67.4) [concrete = constants.%tuple.094]
// CHECK:STDOUT:     %.loc4_68.4: %tuple.type.920 = converted %.loc4_67.1, %tuple.loc4_67 [concrete = constants.%tuple.094]
// CHECK:STDOUT:     %tuple.loc4_68: %tuple.type.007 = tuple_value (%.loc4_68.3, %.loc4_68.4) [concrete = constants.%tuple.4d3]
// CHECK:STDOUT:     %.loc4_68.5: %tuple.type.007 = converted %.loc4_68.1, %tuple.loc4_68 [concrete = constants.%tuple.4d3]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %i32, %b.param: %i32, %c.param: %i32, %d.param: %i32, %e.param: %i32) {
// CHECK:STDOUT: !default_values:
// CHECK:STDOUT:   %.loc4_68.1: %tuple.type.c4f = tuple_literal (%int_1, %.loc4_67.1) [concrete = constants.%tuple.326]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %.e53: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc4_13: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete = constants.%.e53]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc4_15.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc4_15.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc4_15.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:     %.loc4_15.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:     %.loc4_15.2: %i32 = converted %int_0, %.loc4_15.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %i32) {
// CHECK:STDOUT: !default_values:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT: }
// CHECK:STDOUT:
