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

// --- basic.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(x: i32 = 0);
//@dump-sem-ir-end

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
// CHECK:STDOUT:     %.loc5: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete = constants.%.e53]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %i32) default_values: {
// CHECK:STDOUT:   constants.%int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: };
// CHECK:STDOUT:
