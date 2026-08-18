// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/default_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/default_values.carbon
//
// --- fail_default_presence_mismatch_redecl.carbon

library "[[@TEST_NAME]]";

fn F(x: i32 = 0, y: i32 = _, z: i32 = _);
// CHECK:STDERR: fail_default_presence_mismatch_redecl.carbon:[[@LINE+7]]:13: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: fn F(unused x: i32, unused y: i32 = 1, unused z: i32 = 2) { }
// CHECK:STDERR:             ^~~~~~
// CHECK:STDERR: fail_default_presence_mismatch_redecl.carbon:[[@LINE-4]]:6: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: fn F(x: i32 = 0, y: i32 = _, z: i32 = _);
// CHECK:STDERR:      ^~~~~~~~~~
// CHECK:STDERR:
fn F(unused x: i32, unused y: i32 = 1, unused z: i32 = 2) { }

// --- fail_defaults_different_values_redecl.carbon

library "[[@TEST_NAME]]";

fn F(x: i32 = 0, y: i32 = 10);
// CHECK:STDERR: fail_defaults_different_values_redecl.carbon:[[@LINE+7]]:41: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn F(unused x: i32 = 0, unused y: i32 = 22) { }
// CHECK:STDERR:                                         ^~
// CHECK:STDERR: fail_defaults_different_values_redecl.carbon:[[@LINE-4]]:27: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: fn F(x: i32 = 0, y: i32 = 10);
// CHECK:STDERR:                           ^~
// CHECK:STDERR:
fn F(unused x: i32 = 0, unused y: i32 = 22) { }

// --- fail_defaults_never_specified_redecl.carbon

library "[[@TEST_NAME]]";

fn F(x: i32 = _);
// CHECK:STDERR: fail_defaults_never_specified_redecl.carbon:[[@LINE+7]]:6: error: no value for default number 0 is ever specified. [PatternDefaultValueNeverSpecified]
// CHECK:STDERR: fn F(unused x: i32 = _) { }
// CHECK:STDERR:      ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_defaults_never_specified_redecl.carbon:[[@LINE-4]]:6: note: previous declaration here. [PatternDefaultValueNeverSpecifiedNote]
// CHECK:STDERR: fn F(x: i32 = _);
// CHECK:STDERR:      ^~~~~~~~~~
// CHECK:STDERR:
fn F(unused x: i32 = _) { }

// --- default_presence_mismatch.carbon

library "[[@TEST_NAME]]";

fn F(x: i32 = 0, y: i32 = _, z: i32 = _);

// --- fail_default_presence_mismatch.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_default_presence_mismatch.impl.carbon:[[@LINE+8]]:13: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: fn F(unused x: i32, unused y: i32 = 1, unused z: i32 = 2) { }
// CHECK:STDERR:             ^~~~~~
// CHECK:STDERR: fail_default_presence_mismatch.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: default_presence_mismatch.carbon:4:6: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: fn F(x: i32 = 0, y: i32 = _, z: i32 = _);
// CHECK:STDERR:      ^~~~~~~~~~
// CHECK:STDERR:
fn F(unused x: i32, unused y: i32 = 1, unused z: i32 = 2) { }

// --- defaults_different_values.carbon

library "[[@TEST_NAME]]";

fn F(x: i32 = 0, y: i32 = 10);

// --- fail_defaults_different_values.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_defaults_different_values.impl.carbon:[[@LINE+4]]:25: error: default value differs from the previous declaration. [PatternDefaultValueDiffers]
// CHECK:STDERR: fn F(unused x: i32 = 0, unused y: i32 = 22) { }
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(unused x: i32 = 0, unused y: i32 = 22) { }

// --- defaults_never_specified.carbon

library "[[@TEST_NAME]]";

fn F(x: i32 = _);

// --- fail_defaults_never_specified.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_defaults_never_specified.impl.carbon:[[@LINE+4]]:6: error: no value for default number 0 is ever specified. [PatternDefaultValueNeverSpecified]
// CHECK:STDERR: fn F(unused x: i32 = _) { }
// CHECK:STDERR:      ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(unused x: i32 = _) { }

// --- redecl.carbon

library "[[@TEST_NAME]]";

fn F(x: i32 = 0) -> i32;
fn G(x: i32 = _) -> i32;
fn H(x: i32 = 0) -> i32;

//@dump-sem-ir-begin
fn F(x: i32 = 0) -> i32 { return x; }
fn G(x: i32 = 0) -> i32 { return x; }
fn H(x: i32 = _) -> i32 { return x; }
//@dump-sem-ir-end

// --- repeated_defaults.carbon

library "[[@TEST_NAME]]";

fn F(x: i32 = 0) -> i32;

// --- repeated_defaults.impl.carbon

impl library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(x: i32 = 0) -> i32 { return x; }
//@dump-sem-ir-end

// --- elided_defaults_in_def.carbon

library "[[@TEST_NAME]]";

fn F(x: i32 = 0) -> i32;

// --- elided_defaults_in_def.impl.carbon

impl library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(x: i32 = _) -> i32 { return x; }
//@dump-sem-ir-end

// --- elided_defaults_in_decl.carbon

library "[[@TEST_NAME]]";

fn F(x: i32 = _) -> i32;

// --- elided_defaults_in_decl.impl.carbon

impl library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(x: i32 = 0) -> i32 { return x; }
//@dump-sem-ir-end

// CHECK:STDOUT: --- redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %.e53: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl.loc9: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc9_13: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete = constants.%.e53]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc9_21 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_0.loc9: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %i32.loc9_21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc9_21: Core.Form = init_form %i32.loc9_21 [concrete = constants.%.795f]
// CHECK:STDOUT:     %x.param.loc9: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc9_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x.loc9: %i32 = wrapper_binding x, %x.param.loc9
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %return.param.loc9: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc9: ref %i32 = return_slot %return.param.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl.loc10: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc10_13: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete = constants.%.e53]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc10_21 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %i32.loc10_21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_21: Core.Form = init_form %i32.loc10_21 [concrete = constants.%.795f]
// CHECK:STDOUT:     %x.param.loc10: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc10_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x.loc10: %i32 = wrapper_binding x, %x.param.loc10
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %return.param.loc10: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc10: ref %i32 = return_slot %return.param.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl.loc11: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc11_13: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete = constants.%.e53]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc11_21 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc11_21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_21: Core.Form = init_form %i32.loc11_21 [concrete = constants.%.795f]
// CHECK:STDOUT:     %x.param.loc11: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc11_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x.loc11: %i32 = wrapper_binding x, %x.param.loc11
// CHECK:STDOUT:     %return.param.loc11: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc11: ref %i32 = return_slot %return.param.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param.loc9: %i32) -> out %return.param.loc9: %i32 default_values: {
// CHECK:STDOUT:   constants.%int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: } {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x.loc9
// CHECK:STDOUT:   %impl.elem0.loc9: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %x.ref, %impl.elem0.loc9
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %x.ref, %specific_fn.loc9
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc9_34.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param.loc10: %i32) -> out %return.param.loc10: %i32 default_values: {
// CHECK:STDOUT:   constants.%int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: } {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x.loc10
// CHECK:STDOUT:   %impl.elem0.loc10: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %x.ref, %impl.elem0.loc10
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %x.ref, %specific_fn.loc10
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc10_34.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%x.param.loc11: %i32) -> out %return.param.loc11: %i32 default_values: {
// CHECK:STDOUT:   constants.%int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: } {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc11_34.1: <bound method> = bound_method %x.ref, %impl.elem0.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_34.2: <bound method> = bound_method %x.ref, %specific_fn.loc11
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc11_34.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated_defaults.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %.e53: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc5_13: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete = constants.%.e53]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc5_21 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %i32.loc5_21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_21: Core.Form = init_form %i32.loc5_21 [concrete = constants.%.795f]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc5_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %i32) -> out %return.param: %i32 default_values: [from "repeated_defaults.carbon"] {
// CHECK:STDOUT:   constants.%int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: } {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %impl.elem0.loc5: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc5_34.1: <bound method> = bound_method %x.ref, %impl.elem0.loc5
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_34.2: <bound method> = bound_method %x.ref, %specific_fn.loc5
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc5_34.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- elided_defaults_in_def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %.e53: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc5_13: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete = constants.%.e53]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc5_21 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5_21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_21: Core.Form = init_form %i32.loc5_21 [concrete = constants.%.795f]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc5_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %i32) -> out %return.param: %i32 default_values: [from "elided_defaults_in_def.carbon"] {
// CHECK:STDOUT:   constants.%int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: } {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %impl.elem0: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc5_34.1: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_34.2: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc5_34.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- elided_defaults_in_decl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %.e53: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc5_13: %pattern_type.6b6 = default_value_pattern %x.patt, index: 0 [concrete = constants.%.e53]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc5_21 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %i32.loc5_21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_21: Core.Form = init_form %i32.loc5_21 [concrete = constants.%.795f]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc5_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %i32) -> out %return.param: %i32 default_values: [from "elided_defaults_in_decl.carbon"] {
// CHECK:STDOUT:   constants.%int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: } {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %impl.elem0.loc5: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc5_34.1: <bound method> = bound_method %x.ref, %impl.elem0.loc5
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_34.2: <bound method> = bound_method %x.ref, %specific_fn.loc5
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc5_34.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
