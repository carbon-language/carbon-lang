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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_member_access_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_member_access_type.carbon

var x: {.a: f64} = {.a = 4.0};
// CHECK:STDERR: fail_member_access_type.carbon:[[@LINE+4]]:14: error: type `{.a: f64}` does not have a member `b` [QualifiedExprNameNotFound]
// CHECK:STDERR: var y: i32 = x.b;
// CHECK:STDERR:              ^~~
// CHECK:STDERR:
var y: i32 = x.b;

// CHECK:STDOUT: --- fail_member_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f64.9a0: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %struct_type.a.316: type = struct_type {.a: %f64.9a0} [concrete]
// CHECK:STDOUT:   %pattern_type.941: type = pattern_type %struct_type.a.316 [concrete]
// CHECK:STDOUT:   %float.a4e: Core.FloatLiteral = float_literal_value 40e-1 [concrete]
// CHECK:STDOUT:   %struct_type.a.b4b: type = struct_type {.a: Core.FloatLiteral} [concrete]
// CHECK:STDOUT:   %struct.f91: %struct_type.a.b4b = struct_value (%float.a4e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.731: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.9a0)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.26a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.07b: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.731 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.26a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ed2: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.9a0, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.2f4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ed2, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.a4e, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.07b [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.07b, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.a4e, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.7a2: %f64.9a0 = float_value 4 [concrete]
// CHECK:STDOUT:   %struct.d38: %struct_type.a.316 = struct_value (%float.7a2) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/parts/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.9a6: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.9a6), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.941 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.941 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a.316 = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc15: type = splice_block %struct_type.a [concrete = constants.%struct_type.a.316] {
// CHECK:STDOUT:     %f64: type = type_literal constants.%f64.9a0 [concrete = constants.%f64.9a0]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %f64.9a0} [concrete = constants.%struct_type.a.316]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.316 = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.c6b = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.c6b = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %i32 = var %y.var_patt [concrete]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %y: ref %i32 = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 40e-1 [concrete = constants.%float.a4e]
// CHECK:STDOUT:   %.loc15_29.1: %struct_type.a.b4b = struct_literal (%float) [concrete = constants.%struct.f91]
// CHECK:STDOUT:   %impl.elem0: %.2f4 = impl_witness_access constants.%ImplicitAs.impl_witness.26a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.07b]
// CHECK:STDOUT:   %bound_method.loc15_29.1: <bound method> = bound_method %float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_29.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.9a0 = call %bound_method.loc15_29.2(%float) [concrete = constants.%float.7a2]
// CHECK:STDOUT:   %.loc15_29.2: init %f64.9a0 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.7a2]
// CHECK:STDOUT:   %.loc15_29.3: init %struct_type.a.316 to file.%x.var = struct_init (%.loc15_29.2) [concrete = constants.%struct.d38]
// CHECK:STDOUT:   %.loc15_1: init %struct_type.a.316 = converted %.loc15_29.1, %.loc15_29.3 [concrete = constants.%struct.d38]
// CHECK:STDOUT:   assign file.%x.var, %.loc15_1
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.316 = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
