// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_field_name_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_field_name_mismatch.carbon

var a: {.a: i32} = {.a = 1 };
// CHECK:STDERR: fail_field_name_mismatch.carbon:[[@LINE+4]]:29: error: cannot convert from struct type `{.a: i32}` to `{.a: i32, .b: i32}`: missing field `b` in source type [StructInitMissingFieldInConversion]
// CHECK:STDERR: var b: {.a: i32, .b: i32} = a;
// CHECK:STDERR:                             ^
// CHECK:STDERR:
var b: {.a: i32, .b: i32} = a;

// CHECK:STDERR: fail_field_name_mismatch.carbon:[[@LINE+4]]:20: error: struct `{.a: i32}` has no field named `b` [StructInitUnexpectedFieldInLiteral]
// CHECK:STDERR: var x: {.a: i32} = {.b = 1};
// CHECK:STDERR:                    ^~~~~~~~
// CHECK:STDERR:
var x: {.a: i32} = {.b = 1};

// CHECK:STDERR: fail_field_name_mismatch.carbon:[[@LINE+4]]:20: error: cannot convert from struct type `{.a: i32}` to `{.b: i32}`: unexpected field `a` in source type [StructInitUnexpectedFieldInConversion]
// CHECK:STDERR: var y: {.b: i32} = x;
// CHECK:STDERR:                    ^
// CHECK:STDERR:
var y: {.b: i32} = x;

// CHECK:STDOUT: --- fail_field_name_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.1c5: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.d7c: type = pattern_type %struct_type.a.1c5 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.48c: %struct_type.a.a6c = struct_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %.8fa: ref %i32 = struct_access file.%a.var, element0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.754: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.754 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d9c: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.107: %struct_type.a.1c5 = struct_value (%int_1.d9c) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.941: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %.a23: ref %i32 = struct_access file.%b.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.b.a15: type = struct_type {.b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.62b: %struct_type.b.a15 = struct_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %struct_type.b.676: type = struct_type {.b: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %struct_type.b.676 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.754 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d7c = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d7c = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %struct_type.a.1c5 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc15: type = splice_block %struct_type.a.loc15 [concrete = constants.%struct_type.a.1c5] {
// CHECK:STDOUT:     %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.loc15: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a.1c5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %struct_type.a.1c5 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.941 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.941 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %struct_type.a.b = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc20: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b] {
// CHECK:STDOUT:     %i32.loc20_13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc20_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %struct_type.a.b = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d7c = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d7c = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a.1c5 = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc26: type = splice_block %struct_type.a.loc26 [concrete = constants.%struct_type.a.1c5] {
// CHECK:STDOUT:     %i32.loc26: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.loc26: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a.1c5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.1c5 = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.956 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.956 = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.b.676 = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc32: type = splice_block %struct_type.b [concrete = constants.%struct_type.b.676] {
// CHECK:STDOUT:     %i32.loc32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %i32} [concrete = constants.%struct_type.b.676]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.b.676 = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_28.1: %struct_type.a.a6c = struct_literal (%int_1.loc15) [concrete = constants.%struct.48c]
// CHECK:STDOUT:   %impl.elem0.loc15: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc15_28.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_28.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_28.2(%int_1.loc15) [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc15_28.2: init %i32 = converted %int_1.loc15, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc15_28.3: init %struct_type.a.1c5 to file.%a.var = struct_init (%.loc15_28.2) [concrete = constants.%struct.107]
// CHECK:STDOUT:   %.loc15_1: init %struct_type.a.1c5 = converted %.loc15_28.1, %.loc15_28.3 [concrete = constants.%struct.107]
// CHECK:STDOUT:   assign file.%a.var, %.loc15_1
// CHECK:STDOUT:   %a.ref: ref %struct_type.a.1c5 = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %.loc20_29.1: ref %i32 = struct_access %a.ref, element0 [concrete = constants.%.8fa]
// CHECK:STDOUT:   %.loc20_29.2: %i32 = acquire_value %.loc20_29.1
// CHECK:STDOUT:   %impl.elem0.loc20: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:   %bound_method.loc20_29.1: <bound method> = bound_method %.loc20_29.2, %impl.elem0.loc20
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_29.2: <bound method> = bound_method %.loc20_29.2, %specific_fn.loc20
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc20_29.2(%.loc20_29.2)
// CHECK:STDOUT:   %.loc20_29.3: ref %i32 = struct_access file.%b.var, element0 [concrete = constants.%.a23]
// CHECK:STDOUT:   %.loc20_29.4: init %i32 to %.loc20_29.3 = in_place_init %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc26: %struct_type.b.a15 = struct_literal (%int_1.loc26) [concrete = constants.%struct.62b]
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.1c5 = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
