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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/struct.carbon

fn G(s: {.a: i32, .b: i32});

fn F(cond: bool) {
  var a: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  G(if cond then a else a);
}

// CHECK:STDOUT: --- struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.b4c: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.20e3: type = pattern_type %struct_type.a.b.b4c [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.4aa: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.631: %struct_type.a.b.b4c = struct_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc18_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %s.param_patt: %pattern_type.20e3 = value_param_pattern [concrete]
// CHECK:STDOUT:     %s.patt: %pattern_type.20e3 = at_binding_pattern s, %s.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %s.param: %struct_type.a.b.b4c = value_param call_param0
// CHECK:STDOUT:     %.loc15: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.b4c] {
// CHECK:STDOUT:       %i32.loc15_14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc15_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.b4c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: %struct_type.a.b.b4c = value_binding s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %cond.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %cond.patt: %pattern_type.831 = at_binding_pattern cond, %cond.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %cond.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %cond: bool = value_binding cond, %cond.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s.param: %struct_type.a.b.b4c);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%cond.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.20e3 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.20e3 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %struct_type.a.b.b4c = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc18_46.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2) [concrete = constants.%struct.4aa]
// CHECK:STDOUT:   %impl.elem0.loc18_46.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc18_46.1: <bound method> = bound_method %int_1, %impl.elem0.loc18_46.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc18_46.1: <specific function> = specific_function %impl.elem0.loc18_46.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_46.2: <bound method> = bound_method %int_1, %specific_fn.loc18_46.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_46.1: init %i32 = call %bound_method.loc18_46.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc18_46.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_46.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc18_46.3: ref %i32 = struct_access %a.var, element0
// CHECK:STDOUT:   %.loc18_46.4: init %i32 to %.loc18_46.3 = in_place_init %.loc18_46.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc18_46.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc18_46.3: <bound method> = bound_method %int_2, %impl.elem0.loc18_46.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc18_46.2: <specific function> = specific_function %impl.elem0.loc18_46.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_46.4: <bound method> = bound_method %int_2, %specific_fn.loc18_46.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_46.2: init %i32 = call %bound_method.loc18_46.4(%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc18_46.5: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_46.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc18_46.6: ref %i32 = struct_access %a.var, element1
// CHECK:STDOUT:   %.loc18_46.7: init %i32 to %.loc18_46.6 = in_place_init %.loc18_46.5 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc18_46.8: init %struct_type.a.b.b4c to %a.var = struct_init (%.loc18_46.4, %.loc18_46.7) [concrete = constants.%struct.631]
// CHECK:STDOUT:   %.loc18_3: init %struct_type.a.b.b4c = converted %.loc18_46.1, %.loc18_46.8 [concrete = constants.%struct.631]
// CHECK:STDOUT:   assign %a.var, %.loc18_3
// CHECK:STDOUT:   %.loc18_27: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.b4c] {
// CHECK:STDOUT:     %i32.loc18_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.b4c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %struct_type.a.b.b4c = ref_binding a, %a.var
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %cond.ref: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %a.ref.loc19_18: ref %struct_type.a.b.b4c = name_ref a, %a
// CHECK:STDOUT:   %.loc19_18.1: ref %i32 = struct_access %a.ref.loc19_18, element0
// CHECK:STDOUT:   %.loc19_18.2: %i32 = acquire_value %.loc19_18.1
// CHECK:STDOUT:   %.loc19_18.3: ref %i32 = struct_access %a.ref.loc19_18, element1
// CHECK:STDOUT:   %.loc19_18.4: %i32 = acquire_value %.loc19_18.3
// CHECK:STDOUT:   %struct.loc19_18: %struct_type.a.b.b4c = struct_value (%.loc19_18.2, %.loc19_18.4)
// CHECK:STDOUT:   %.loc19_18.5: %struct_type.a.b.b4c = converted %a.ref.loc19_18, %struct.loc19_18
// CHECK:STDOUT:   br !if.expr.result(%.loc19_18.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc19_25: ref %struct_type.a.b.b4c = name_ref a, %a
// CHECK:STDOUT:   %.loc19_25.1: ref %i32 = struct_access %a.ref.loc19_25, element0
// CHECK:STDOUT:   %.loc19_25.2: %i32 = acquire_value %.loc19_25.1
// CHECK:STDOUT:   %.loc19_25.3: ref %i32 = struct_access %a.ref.loc19_25, element1
// CHECK:STDOUT:   %.loc19_25.4: %i32 = acquire_value %.loc19_25.3
// CHECK:STDOUT:   %struct.loc19_25: %struct_type.a.b.b4c = struct_value (%.loc19_25.2, %.loc19_25.4)
// CHECK:STDOUT:   %.loc19_20: %struct_type.a.b.b4c = converted %a.ref.loc19_25, %struct.loc19_25
// CHECK:STDOUT:   br !if.expr.result(%.loc19_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc19_5: %struct_type.a.b.b4c = block_arg !if.expr.result
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref(%.loc19_5)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_3.3(%self.param: ref %struct_type.a.b.b4c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
