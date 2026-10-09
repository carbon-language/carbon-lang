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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/one_entry.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/one_entry.carbon

var x: {.a: i32} = {.a = 4};
var y: {.a: i32} = x;

// CHECK:STDOUT: --- one_entry.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.8e6: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.9f8: type = pattern_type %struct_type.a.8e6 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.9f8 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.9f8 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.2cb: %struct_type.a.a6c = struct_value (%int_4.0c1) [concrete]
// CHECK:STDOUT:   %.219: ref %i32 = struct_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.09b = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f0f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e04, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.380: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %struct.f62: %struct_type.a.8e6 = struct_value (%int_4.380) [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.9f8 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.var_patt: %pattern_type.9f8 = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0ff: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a63: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.617: %Int.as.Copy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0ff) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bc1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.481: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bc1, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.617, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %x.var: ref %struct_type.a.8e6 = var_storage %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc15: type = splice_block %struct_type.a.loc15 [concrete = constants.%struct_type.a.8e6] {
// CHECK:STDOUT:     %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.loc15: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a.8e6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.8e6 = wrapper_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.9f8 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.9f8 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.a.8e6 = var_storage %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc16: type = splice_block %struct_type.a.loc16 [concrete = constants.%struct_type.a.8e6] {
// CHECK:STDOUT:     %i32.loc16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.loc16: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a.8e6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.a.8e6 = wrapper_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.9f8 = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.9f8 = var_pattern %y.patt [concrete = constants.%y.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc15_27.1: %struct_type.a.a6c = struct_literal (%int_4) [concrete = constants.%struct.2cb]
// CHECK:STDOUT:   %impl.elem0.loc15: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc15_27.1: <bound method> = bound_method %int_4, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_27.2: <bound method> = bound_method %int_4, %specific_fn.loc15 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_27.2(%int_4) [concrete = constants.%int_4.380]
// CHECK:STDOUT:   %.loc15_27.2: init %i32 = converted %int_4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.380]
// CHECK:STDOUT:   %.loc15_27.3: init %struct_type.a.8e6 to file.%x.var = struct_init (%.loc15_27.2) [concrete = constants.%struct.f62]
// CHECK:STDOUT:   %.loc15_1: init %struct_type.a.8e6 = converted %.loc15_27.1, %.loc15_27.3 [concrete = constants.%struct.f62]
// CHECK:STDOUT:   assign file.%x.var, %.loc15_1
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.8e6 = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc16_20.1: ref %i32 = struct_access %x.ref, element0 [concrete = constants.%.219]
// CHECK:STDOUT:   %.loc16_20.2: %i32 = acquire_value %.loc16_20.1
// CHECK:STDOUT:   %impl.elem0.loc16: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc16_20.1: <bound method> = bound_method %.loc16_20.2, %impl.elem0.loc16
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_20.2: <bound method> = bound_method %.loc16_20.2, %specific_fn.loc16
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc16_20.2(%.loc16_20.2)
// CHECK:STDOUT:   %.loc16_20.3: init %struct_type.a.8e6 to file.%y.var = struct_init (%Int.as.Copy.impl.Op.call)
// CHECK:STDOUT:   %.loc16_1: init %struct_type.a.8e6 = converted %x.ref, %.loc16_20.3
// CHECK:STDOUT:   assign file.%y.var, %.loc16_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
