// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/member_access.carbon

var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
var y: i32 = x.b;
var z: i32 = y;

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.09c: type = struct_type {.a: %f64.dc1, .b: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.0ed: type = pattern_type %struct_type.a.b.09c [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.0cc: type = struct_type {.a: Core.FloatLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.877: %struct_type.a.b.0cc = struct_value (%float.1f7, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %.c34: ref %f64.dc1 = struct_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.91e: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.dc1)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.43c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.393, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ef0: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7d2: %ImplicitAs.type.91e = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.43c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6c5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.dc1, %ImplicitAs.facet.7d2) [concrete]
// CHECK:STDOUT:   %.2d7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6c5, %ImplicitAs.facet.7d2 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.819: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.895: %f64.dc1 = float_value 0 [concrete]
// CHECK:STDOUT:   %.46a: ref %i32 = struct_access file.%x.var, element1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9f1: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.9f1) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet.9f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.77f: %struct_type.a.b.09c = struct_value (%float.895, %int_1.0c6) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/parts/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.c4c7: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.393 = impl_witness_table (%Core.import_ref.c4c7), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:     .z = %z
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.0ed = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.0ed = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.09c = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc15: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.09c] {
// CHECK:STDOUT:     %f64: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:     %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %f64.dc1, .b: %i32} [concrete = constants.%struct_type.a.b.09c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.b.09c = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.6b6 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.6b6 = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %i32 = var %y.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %y: ref %i32 = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %z.patt: %pattern_type.6b6 = ref_binding_pattern z [concrete]
// CHECK:STDOUT:     %z.var_patt: %pattern_type.6b6 = var_pattern %z.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %z.var: ref %i32 = var %z.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %z: ref %i32 = ref_binding z, %z.var [concrete = %z.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_46.1: %struct_type.a.b.0cc = struct_literal (%float, %int_1) [concrete = constants.%struct.877]
// CHECK:STDOUT:   %impl.elem0.loc15_46.1: %.2d7 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0]
// CHECK:STDOUT:   %bound_method.loc15_46.1: <bound method> = bound_method %float, %impl.elem0.loc15_46.1 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc15_46.1: <specific function> = specific_function %impl.elem0.loc15_46.1, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_46.2: <bound method> = bound_method %float, %specific_fn.loc15_46.1 [concrete = constants.%bound_method.819]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.dc1 = call %bound_method.loc15_46.2(%float) [concrete = constants.%float.895]
// CHECK:STDOUT:   %.loc15_46.2: init %f64.dc1 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.895]
// CHECK:STDOUT:   %.loc15_46.3: ref %f64.dc1 = struct_access file.%x.var, element0 [concrete = constants.%.c34]
// CHECK:STDOUT:   %.loc15_46.4: init %f64.dc1 to %.loc15_46.3 = in_place_init %.loc15_46.2 [concrete = constants.%float.895]
// CHECK:STDOUT:   %impl.elem0.loc15_46.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc15_46.3: <bound method> = bound_method %int_1, %impl.elem0.loc15_46.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc15_46.2: <specific function> = specific_function %impl.elem0.loc15_46.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_46.4: <bound method> = bound_method %int_1, %specific_fn.loc15_46.2 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_46.4(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc15_46.5: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc15_46.6: ref %i32 = struct_access file.%x.var, element1 [concrete = constants.%.46a]
// CHECK:STDOUT:   %.loc15_46.7: init %i32 to %.loc15_46.6 = in_place_init %.loc15_46.5 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc15_46.8: init %struct_type.a.b.09c to file.%x.var = struct_init (%.loc15_46.4, %.loc15_46.7) [concrete = constants.%struct.77f]
// CHECK:STDOUT:   %.loc15_1: init %struct_type.a.b.09c = converted %.loc15_46.1, %.loc15_46.8 [concrete = constants.%struct.77f]
// CHECK:STDOUT:   assign file.%x.var, %.loc15_1
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.b.09c = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc16_15.1: ref %i32 = struct_access %x.ref, element1 [concrete = constants.%.46a]
// CHECK:STDOUT:   %.loc16_15.2: %i32 = acquire_value %.loc16_15.1
// CHECK:STDOUT:   %impl.elem0.loc16: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc16_15.1: <bound method> = bound_method %.loc16_15.2, %impl.elem0.loc16
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_15.2: <bound method> = bound_method %.loc16_15.2, %specific_fn.loc16
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16: init %i32 = call %bound_method.loc16_15.2(%.loc16_15.2)
// CHECK:STDOUT:   assign file.%y.var, %Int.as.Copy.impl.Op.call.loc16
// CHECK:STDOUT:   %y.ref: ref %i32 = name_ref y, file.%y [concrete = file.%y.var]
// CHECK:STDOUT:   %.loc17: %i32 = acquire_value %y.ref
// CHECK:STDOUT:   %impl.elem0.loc17: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc17_14.1: <bound method> = bound_method %.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_14.2: <bound method> = bound_method %.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc17: init %i32 = call %bound_method.loc17_14.2(%.loc17)
// CHECK:STDOUT:   assign file.%z.var, %Int.as.Copy.impl.Op.call.loc17
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
