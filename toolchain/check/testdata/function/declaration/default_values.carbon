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

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:6: error: default value expression type `str` doesn't match pattern type `i32` [PatternDefaultValueTypeMismatch]
// CHECK:STDERR: fn K(x: i32 = "bazz");
// CHECK:STDERR:      ^~~~~~~~~~~~~~~
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
// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+7]]:14: note: all patterns to the right of this first pattern with a default value must also specify a default value. [RequiredPatternDefaultValueFirstDefault]
// CHECK:STDERR: fn Z(v: i32, x: i32 = 3, (y: i32 = 2, z: i32), w: i32 = 4, k: i32);
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+4]]:60: note: this pattern is also missing a required default value. [RequiredPatternDefaultValueMissingAdditional]
// CHECK:STDERR: fn Z(v: i32, x: i32 = 3, (y: i32 = 2, z: i32), w: i32 = 4, k: i32);
// CHECK:STDERR:                                                            ^~~~~~
// CHECK:STDERR:
fn Z(v: i32, x: i32 = 3, (y: i32 = 2, z: i32), w: i32 = 4, k: i32);

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
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %.loc4_66: %tuple.type.f94 = tuple_literal (%int_3, %int_4) [concrete = constants.%tuple.302]
// CHECK:STDOUT:     %.loc4_67: %tuple.type.bd0 = tuple_literal (%int_2, %.loc4_66) [concrete = constants.%tuple.b26]
// CHECK:STDOUT:     %.loc4_68: %tuple.type.c4f = tuple_literal (%int_1, %.loc4_67) [concrete = constants.%tuple.326]
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %i32, %b.param: %i32, %c.param: %i32, %d.param: %i32, %e.param: %i32) default_values: {
// CHECK:STDOUT:   constants.%tuple.326: %tuple.type.c4f = tuple_value (constants.%int_1.5b8, constants.%tuple.b26) [concrete]
// CHECK:STDOUT: };
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
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc4: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete = constants.%.e53]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %i32) default_values {
// CHECK:STDOUT:   constants.%int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: };
// CHECK:STDOUT:
