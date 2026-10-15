// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/default_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/default_values.carbon
//
// --- fail_defaults_different_values_redecl.carbon
library "[[@TEST_NAME]]";

fn F(x: i32 = 0, y: i32 = 10);
// CHECK:STDERR: fail_defaults_different_values_redecl.carbon:[[@LINE+3]]:25-42: error: redeclaration differs at parameter 2 [RedeclParamDiffers]
// CHECK:STDERR: fail_defaults_different_values_redecl.carbon:[[@LINE-2]]:18-28: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR:
fn F(unused x: i32 = 0, unused y: i32 = 22) { }

// --- imported_default_presence_mismatch.carbon
library "[[@TEST_NAME]]";

fn F(x: i32 = 0, y: i32 = 1, z: i32 = 2);

// --- fail_imported_default_presence_mismatch.impl.carbon
impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_imported_default_presence_mismatch.impl.carbon:[[@LINE+4]]:13-18: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: imported from: fail_imported_default_presence_mismatch.impl.carbon:[[@LINE-3]]:1 [InImport]
// CHECK:STDERR: imported_default_presence_mismatch.carbon:3:6-15: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR:
fn F(unused x: i32, unused y: i32 = 1, unused z: i32 = 2) { }

// --- imported_defaults_different_values.carbon
library "[[@TEST_NAME]]";

fn F(x: i32 = 0, y: i32 = 10);

// --- fail_imported_defaults_different_values.impl.carbon
impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_imported_defaults_different_values.impl.carbon:[[@LINE+4]]:25-42: error: redeclaration differs at parameter 2 [RedeclParamDiffers]
// CHECK:STDERR: imported from: fail_imported_defaults_different_values.impl.carbon:[[@LINE-3]]:1 [InImport]
// CHECK:STDERR: imported_defaults_different_values.carbon:3:18-28: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR:
fn F(unused x: i32 = 0, unused y: i32 = 22) { }

// --- redecl.carbon
library "[[@TEST_NAME]]";

fn F(x: i32 = 0) -> i32;

//@dump-sem-ir-begin
fn F(x: i32 = 0) -> i32 { return x; }
//@dump-sem-ir-end

// --- imported_repeated_defaults.carbon
library "[[@TEST_NAME]]";

fn F(x: i32 = 0) -> i32;

// --- imported_repeated_defaults.impl.carbon
impl library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(x: i32 = 0) -> i32 { return x; }
//@dump-sem-ir-end

// CHECK:STDOUT: --- redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %.bb4: %pattern_type.d94 = default_value_pattern %x.patt, %int_0.753 [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl.loc6: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc6_13: %pattern_type.d94 = default_value_pattern %x.patt, %.loc6_15.2 [concrete = constants.%.bb4]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc6_21 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_0.loc6: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0.loc6_15: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc6_15.1: <bound method> = bound_method %int_0.loc6, %impl.elem0.loc6_15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc6_15: <specific function> = specific_function %impl.elem0.loc6_15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_15.2: <bound method> = bound_method %int_0.loc6, %specific_fn.loc6_15 [concrete = constants.%bound_method]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_15.2(%int_0.loc6) [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %.loc6_15.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %.loc6_15.2: %i32 = converted %int_0.loc6, %.loc6_15.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %i32.loc6_21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_21: Core.Form = init_form %i32.loc6_21 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %x.param.loc6: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc6_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x.loc6: %i32 = wrapper_binding x, %x.param.loc6
// CHECK:STDOUT:     %return.param.loc6: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc6: ref %i32 = return_slot %return.param.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param.loc6: %i32) -> out %return.param.loc6: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x.loc6
// CHECK:STDOUT:   %impl.elem0.loc6_34: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc6_34.1: <bound method> = bound_method %x.ref, %impl.elem0.loc6_34
// CHECK:STDOUT:   %specific_fn.loc6_34: <specific function> = specific_function %impl.elem0.loc6_34, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_34.2: <bound method> = bound_method %x.ref, %specific_fn.loc6_34
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_34.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- imported_repeated_defaults.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %.bb4: %pattern_type.d94 = default_value_pattern %x.patt, %int_0.753 [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.import_ref.e4b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Main//imported_repeated_defaults, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Main.import_ref.e4b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc4_13: %pattern_type.d94 = default_value_pattern %x.patt, %.loc4_15.2 [concrete = constants.%.bb4]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc4_21 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0.loc4_15: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc4_15.1: <bound method> = bound_method %int_0, %impl.elem0.loc4_15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc4_15: <specific function> = specific_function %impl.elem0.loc4_15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc4_15.2: <bound method> = bound_method %int_0, %specific_fn.loc4_15 [concrete = constants.%bound_method]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc4_15.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %.loc4_15.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %.loc4_15.2: %i32 = converted %int_0, %.loc4_15.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %i32.loc4_21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_21: Core.Form = init_form %i32.loc4_21 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc4_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %i32) -> out %return.param: %i32 [from "imported_repeated_defaults.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %impl.elem0.loc4_34: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc4_34.1: <bound method> = bound_method %x.ref, %impl.elem0.loc4_34
// CHECK:STDOUT:   %specific_fn.loc4_34: <specific function> = specific_function %impl.elem0.loc4_34, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_34.2: <bound method> = bound_method %x.ref, %specific_fn.loc4_34
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc4_34.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
