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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/address_of_lvalue.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/address_of_lvalue.carbon

fn F() {
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};

  var unused p: {.a: i32, .b: i32}* = &s;
  var unused q: i32* = &s.a;
  var unused r: i32* = &s.b;

  var t: (i32, i32) = (1, 2);
  var unused t0: i32* = &t.0;
  var unused t1: i32* = &t.1;
}

// CHECK:STDOUT: --- address_of_lvalue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.577: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %ptr.03d: type = ptr_type %struct_type.a.b.577 [concrete]
// CHECK:STDOUT:   %pattern_type.272: type = pattern_type %struct_type.a.b.577 [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.272 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.272 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.4aa: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8f9: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.aa2: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.8e8: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8fb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.3e0: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.cb7: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.d7c: %struct_type.a.b.577 = struct_value (%int_1.8e8, %int_2.cb7) [concrete]
// CHECK:STDOUT:   %pattern_type.0f5: type = pattern_type %ptr.03d [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.0f5 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.0f5 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.20f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f12: %ptr.as.Copy.impl.Op.type.20f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.c1a: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%struct_type.a.b.577) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.dfd: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%struct_type.a.b.577) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.400: %ptr.as.Copy.impl.Op.type.dfd = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.464: %Copy.type = facet_value %ptr.03d, (%Copy.impl_witness.c1a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.9b4: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.464) [concrete]
// CHECK:STDOUT:   %.8ee: type = fn_type_with_self_type %Copy.WithSelf.Op.type.9b4, %Copy.facet.464 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.447: <specific function> = specific_function %ptr.as.Copy.impl.Op.400, @ptr.as.Copy.impl.Op(%struct_type.a.b.577) [concrete]
// CHECK:STDOUT:   %ptr.1fe: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.55f: type = pattern_type %ptr.1fe [concrete]
// CHECK:STDOUT:   %q.patt: %pattern_type.55f = ref_binding_pattern q [concrete]
// CHECK:STDOUT:   %q.var_patt: %pattern_type.55f = var_pattern %q.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.1fc: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.194: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.e42: %ptr.as.Copy.impl.Op.type.194 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.b57: %Copy.type = facet_value %ptr.1fe, (%Copy.impl_witness.1fc) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.4c0: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.b57) [concrete]
// CHECK:STDOUT:   %.b1d: type = fn_type_with_self_type %Copy.WithSelf.Op.type.4c0, %Copy.facet.b57 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.ef7: <specific function> = specific_function %ptr.as.Copy.impl.Op.e42, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %r.patt: %pattern_type.55f = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.var_patt: %pattern_type.55f = var_pattern %r.patt [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.f51: %tuple.type.693 = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.259: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.b4f: type = pattern_type %tuple.type.259 [concrete]
// CHECK:STDOUT:   %t.patt: %pattern_type.b4f = ref_binding_pattern t [concrete]
// CHECK:STDOUT:   %t.var_patt: %pattern_type.b4f = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %tuple.d3f: %tuple.type.259 = tuple_value (%int_1.8e8, %int_2.cb7) [concrete]
// CHECK:STDOUT:   %t0.patt: %pattern_type.55f = ref_binding_pattern t0 [concrete]
// CHECK:STDOUT:   %t0.var_patt: %pattern_type.55f = var_pattern %t0.patt [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %t1.patt: %pattern_type.55f = ref_binding_pattern t1 [concrete]
// CHECK:STDOUT:   %t1.var_patt: %pattern_type.55f = var_pattern %t1.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %self.param_patt.d0a: %pattern_type.55f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.8b1: %pattern_type.55f = wrapper_binding_pattern self, %self.param_patt.d0a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc24 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.6c0: %pattern_type.9d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1ea: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt.6c0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc22_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.04f: %pattern_type.b4f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.03b: %pattern_type.b4f = wrapper_binding_pattern self, %self.param_patt.04f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc22_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.ef5: %pattern_type.0f5 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.29b: %pattern_type.0f5 = wrapper_binding_pattern self, %self.param_patt.ef5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc18 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc18 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc18 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.d32: %pattern_type.272 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.fbc: %pattern_type.272 = wrapper_binding_pattern self, %self.param_patt.d32 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc16 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc16 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc16 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.06a: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.20f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.f12)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.06a), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.55f = ref_param_pattern [concrete = constants.%self.param_patt.d0a]
// CHECK:STDOUT:     %self.patt: %pattern_type.55f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.8b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.1fe = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.1fe = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc24 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.55f = ref_param_pattern [concrete = constants.%self.param_patt.d0a]
// CHECK:STDOUT:     %self.patt: %pattern_type.55f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.8b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.1fe = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.1fe = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc22_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b4f = ref_param_pattern [concrete = constants.%self.param_patt.04f]
// CHECK:STDOUT:     %self.patt: %pattern_type.b4f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.03b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.259 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.259 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc22_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b4f = ref_param_pattern [concrete = constants.%self.param_patt.04f]
// CHECK:STDOUT:     %self.patt: %pattern_type.b4f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.03b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.259 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.259 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc18 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0f5 = ref_param_pattern [concrete = constants.%self.param_patt.ef5]
// CHECK:STDOUT:     %self.patt: %pattern_type.0f5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.29b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.03d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.03d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc18 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0f5 = ref_param_pattern [concrete = constants.%self.param_patt.ef5]
// CHECK:STDOUT:     %self.patt: %pattern_type.0f5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.29b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.03d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.03d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc16 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.272 = ref_param_pattern [concrete = constants.%self.param_patt.d32]
// CHECK:STDOUT:     %self.patt: %pattern_type.272 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.fbc]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b.577 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b.577 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc16 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.272 = ref_param_pattern [concrete = constants.%self.param_patt.d32]
// CHECK:STDOUT:     %self.patt: %pattern_type.272 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.fbc]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b.577 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b.577 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %struct_type.a.b.577 = var_storage %s.var_patt
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc16: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc16_46.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc16, %int_2.loc16) [concrete = constants.%struct.4aa]
// CHECK:STDOUT:   %impl.elem0.loc16_46.1: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc16_46.1: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_46.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8f9]
// CHECK:STDOUT:   %specific_fn.loc16_46.1: <specific function> = specific_function %impl.elem0.loc16_46.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_46.2: <bound method> = bound_method %int_1.loc16, %specific_fn.loc16_46.1 [concrete = constants.%bound_method.aa2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.1: init %i32 = call %bound_method.loc16_46.2(%int_1.loc16) [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %.loc16_46.2: init %i32 = converted %int_1.loc16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.1 [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %.loc16_46.3: ref %i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc16_46.4: init %i32 to %.loc16_46.3 = in_place_init %.loc16_46.2 [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %impl.elem0.loc16_46.2: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc16_46.3: <bound method> = bound_method %int_2.loc16, %impl.elem0.loc16_46.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8fb]
// CHECK:STDOUT:   %specific_fn.loc16_46.2: <specific function> = specific_function %impl.elem0.loc16_46.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_46.4: <bound method> = bound_method %int_2.loc16, %specific_fn.loc16_46.2 [concrete = constants.%bound_method.3e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.2: init %i32 = call %bound_method.loc16_46.4(%int_2.loc16) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc16_46.5: init %i32 = converted %int_2.loc16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.2 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc16_46.6: ref %i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc16_46.7: init %i32 to %.loc16_46.6 = in_place_init %.loc16_46.5 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc16_46.8: init %struct_type.a.b.577 to %s.var = struct_init (%.loc16_46.4, %.loc16_46.7) [concrete = constants.%struct.d7c]
// CHECK:STDOUT:   %.loc16_3: init %struct_type.a.b.577 = converted %.loc16_46.1, %.loc16_46.8 [concrete = constants.%struct.d7c]
// CHECK:STDOUT:   assign %s.var, %.loc16_3
// CHECK:STDOUT:   %.loc16_27: type = splice_block %struct_type.a.b.loc16 [concrete = constants.%struct_type.a.b.577] {
// CHECK:STDOUT:     %i32.loc16_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc16_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc16: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.577]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %struct_type.a.b.577 = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.272 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.272 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.03d = var_storage %p.var_patt
// CHECK:STDOUT:   %s.ref.loc18: ref %struct_type.a.b.577 = name_ref s, %s
// CHECK:STDOUT:   %addr.loc18: %ptr.03d = addr_of %s.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18: %.8ee = impl_witness_access constants.%Copy.impl_witness.c1a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.400]
// CHECK:STDOUT:   %bound_method.loc18_39.1: <bound method> = bound_method %addr.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @ptr.as.Copy.impl.Op(constants.%struct_type.a.b.577) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.447]
// CHECK:STDOUT:   %bound_method.loc18_39.2: <bound method> = bound_method %addr.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc18: init %ptr.03d = call %bound_method.loc18_39.2(%addr.loc18)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call.loc18
// CHECK:STDOUT:   %.loc18: type = splice_block %ptr.loc18 [concrete = constants.%ptr.03d] {
// CHECK:STDOUT:     %i32.loc18_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc18: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.577]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %struct_type.a.b.loc18 [concrete = constants.%ptr.03d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.03d = wrapper_binding p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.0f5 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.0f5 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q.var: ref %ptr.1fe = var_storage %q.var_patt
// CHECK:STDOUT:   %s.ref.loc19: ref %struct_type.a.b.577 = name_ref s, %s
// CHECK:STDOUT:   %.loc19_26: ref %i32 = struct_access %s.ref.loc19, element0
// CHECK:STDOUT:   %addr.loc19: %ptr.1fe = addr_of %.loc19_26
// CHECK:STDOUT:   %impl.elem0.loc19: %.b1d = impl_witness_access constants.%Copy.impl_witness.1fc, element0 [concrete = constants.%ptr.as.Copy.impl.Op.e42]
// CHECK:STDOUT:   %bound_method.loc19_24.1: <bound method> = bound_method %addr.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.ef7]
// CHECK:STDOUT:   %bound_method.loc19_24.2: <bound method> = bound_method %addr.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc19: init %ptr.1fe = call %bound_method.loc19_24.2(%addr.loc19)
// CHECK:STDOUT:   assign %q.var, %ptr.as.Copy.impl.Op.call.loc19
// CHECK:STDOUT:   %.loc19_20: type = splice_block %ptr.loc19 [concrete = constants.%ptr.1fe] {
// CHECK:STDOUT:     %i32.loc19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc19: type = ptr_type %i32.loc19 [concrete = constants.%ptr.1fe]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q: ref %ptr.1fe = wrapper_binding q, %q.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %q.patt: %pattern_type.55f = ref_binding_pattern q [concrete = constants.%q.patt]
// CHECK:STDOUT:     %q.var_patt: %pattern_type.55f = var_pattern %q.patt [concrete = constants.%q.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r.var: ref %ptr.1fe = var_storage %r.var_patt
// CHECK:STDOUT:   %s.ref.loc20: ref %struct_type.a.b.577 = name_ref s, %s
// CHECK:STDOUT:   %.loc20_26: ref %i32 = struct_access %s.ref.loc20, element1
// CHECK:STDOUT:   %addr.loc20: %ptr.1fe = addr_of %.loc20_26
// CHECK:STDOUT:   %impl.elem0.loc20: %.b1d = impl_witness_access constants.%Copy.impl_witness.1fc, element0 [concrete = constants.%ptr.as.Copy.impl.Op.e42]
// CHECK:STDOUT:   %bound_method.loc20_24.1: <bound method> = bound_method %addr.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.ef7]
// CHECK:STDOUT:   %bound_method.loc20_24.2: <bound method> = bound_method %addr.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc20: init %ptr.1fe = call %bound_method.loc20_24.2(%addr.loc20)
// CHECK:STDOUT:   assign %r.var, %ptr.as.Copy.impl.Op.call.loc20
// CHECK:STDOUT:   %.loc20_20: type = splice_block %ptr.loc20 [concrete = constants.%ptr.1fe] {
// CHECK:STDOUT:     %i32.loc20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc20: type = ptr_type %i32.loc20 [concrete = constants.%ptr.1fe]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r: ref %ptr.1fe = wrapper_binding r, %r.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r.patt: %pattern_type.55f = ref_binding_pattern r [concrete = constants.%r.patt]
// CHECK:STDOUT:     %r.var_patt: %pattern_type.55f = var_pattern %r.patt [concrete = constants.%r.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %tuple.type.259 = var_storage %t.var_patt
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc22: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc22_28.1: %tuple.type.f94 = tuple_literal (%int_1.loc22, %int_2.loc22) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %impl.elem0.loc22_28.1: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc22_28.1: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_28.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8f9]
// CHECK:STDOUT:   %specific_fn.loc22_28.1: <specific function> = specific_function %impl.elem0.loc22_28.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22_28.1 [concrete = constants.%bound_method.aa2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.1: init %i32 = call %bound_method.loc22_28.2(%int_1.loc22) [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %.loc22_28.2: init %i32 = converted %int_1.loc22, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.1 [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %tuple.elem0.loc22: ref %i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc22_28.3: init %i32 to %tuple.elem0.loc22 = in_place_init %.loc22_28.2 [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %impl.elem0.loc22_28.2: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc22_28.3: <bound method> = bound_method %int_2.loc22, %impl.elem0.loc22_28.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8fb]
// CHECK:STDOUT:   %specific_fn.loc22_28.2: <specific function> = specific_function %impl.elem0.loc22_28.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.4: <bound method> = bound_method %int_2.loc22, %specific_fn.loc22_28.2 [concrete = constants.%bound_method.3e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.2: init %i32 = call %bound_method.loc22_28.4(%int_2.loc22) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc22_28.4: init %i32 = converted %int_2.loc22, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.2 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %tuple.elem1.loc22: ref %i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc22_28.5: init %i32 to %tuple.elem1.loc22 = in_place_init %.loc22_28.4 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc22_28.6: init %tuple.type.259 to %t.var = tuple_init (%.loc22_28.3, %.loc22_28.5) [concrete = constants.%tuple.d3f]
// CHECK:STDOUT:   %.loc22_3: init %tuple.type.259 = converted %.loc22_28.1, %.loc22_28.6 [concrete = constants.%tuple.d3f]
// CHECK:STDOUT:   assign %t.var, %.loc22_3
// CHECK:STDOUT:   %.loc22_19.1: type = splice_block %.loc22_19.3 [concrete = constants.%tuple.type.259] {
// CHECK:STDOUT:     %i32.loc22_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc22_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22_19.2: %tuple.type.693 = tuple_literal (%i32.loc22_11, %i32.loc22_16) [concrete = constants.%tuple.f51]
// CHECK:STDOUT:     %.loc22_19.3: type = converted %.loc22_19.2, constants.%tuple.type.259 [concrete = constants.%tuple.type.259]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %tuple.type.259 = wrapper_binding t, %t.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.b4f = ref_binding_pattern t [concrete = constants.%t.patt]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.b4f = var_pattern %t.patt [concrete = constants.%t.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0.var: ref %ptr.1fe = var_storage %t0.var_patt
// CHECK:STDOUT:   %t.ref.loc23: ref %tuple.type.259 = name_ref t, %t
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc23: ref %i32 = tuple_access %t.ref.loc23, element0
// CHECK:STDOUT:   %addr.loc23: %ptr.1fe = addr_of %tuple.elem0.loc23
// CHECK:STDOUT:   %impl.elem0.loc23: %.b1d = impl_witness_access constants.%Copy.impl_witness.1fc, element0 [concrete = constants.%ptr.as.Copy.impl.Op.e42]
// CHECK:STDOUT:   %bound_method.loc23_25.1: <bound method> = bound_method %addr.loc23, %impl.elem0.loc23
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.ef7]
// CHECK:STDOUT:   %bound_method.loc23_25.2: <bound method> = bound_method %addr.loc23, %specific_fn.loc23
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc23: init %ptr.1fe = call %bound_method.loc23_25.2(%addr.loc23)
// CHECK:STDOUT:   assign %t0.var, %ptr.as.Copy.impl.Op.call.loc23
// CHECK:STDOUT:   %.loc23: type = splice_block %ptr.loc23 [concrete = constants.%ptr.1fe] {
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc23: type = ptr_type %i32.loc23 [concrete = constants.%ptr.1fe]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0: ref %ptr.1fe = wrapper_binding t0, %t0.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t0.patt: %pattern_type.55f = ref_binding_pattern t0 [concrete = constants.%t0.patt]
// CHECK:STDOUT:     %t0.var_patt: %pattern_type.55f = var_pattern %t0.patt [concrete = constants.%t0.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1.var: ref %ptr.1fe = var_storage %t1.var_patt
// CHECK:STDOUT:   %t.ref.loc24: ref %tuple.type.259 = name_ref t, %t
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc24: ref %i32 = tuple_access %t.ref.loc24, element1
// CHECK:STDOUT:   %addr.loc24: %ptr.1fe = addr_of %tuple.elem1.loc24
// CHECK:STDOUT:   %impl.elem0.loc24: %.b1d = impl_witness_access constants.%Copy.impl_witness.1fc, element0 [concrete = constants.%ptr.as.Copy.impl.Op.e42]
// CHECK:STDOUT:   %bound_method.loc24_25.1: <bound method> = bound_method %addr.loc24, %impl.elem0.loc24
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.ef7]
// CHECK:STDOUT:   %bound_method.loc24_25.2: <bound method> = bound_method %addr.loc24, %specific_fn.loc24
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc24: init %ptr.1fe = call %bound_method.loc24_25.2(%addr.loc24)
// CHECK:STDOUT:   assign %t1.var, %ptr.as.Copy.impl.Op.call.loc24
// CHECK:STDOUT:   %.loc24: type = splice_block %ptr.loc24 [concrete = constants.%ptr.1fe] {
// CHECK:STDOUT:     %i32.loc24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc24: type = ptr_type %i32.loc24 [concrete = constants.%ptr.1fe]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1: ref %ptr.1fe = wrapper_binding t1, %t1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t1.patt: %pattern_type.55f = ref_binding_pattern t1 [concrete = constants.%t1.patt]
// CHECK:STDOUT:     %t1.var_patt: %pattern_type.55f = var_pattern %t1.patt [concrete = constants.%t1.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24: <bound method> = bound_method %t1.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24(%t1.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc23: <bound method> = bound_method %t0.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc23: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc23(%t0.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc22: <bound method> = bound_method %t.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc22: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc22(%t.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc20: <bound method> = bound_method %r.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc20: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc20(%r.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc19: <bound method> = bound_method %q.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc19: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc19(%q.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc18: <bound method> = bound_method %p.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc18: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc18(%p.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc16: <bound method> = bound_method %s.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc16: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc16(%s.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24(%self.param: ref %ptr.1fe) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24(%self.param: ref %ptr.1fe) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24(%self.param: ref %ptr.1fe) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_3.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_3.2(%self.param: ref %tuple.type.259) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc22_3.1: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc22_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc22_3.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc22_3.2: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc22_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc22_3.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_3.2(%self.param: ref %tuple.type.259) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_3.2(%self.param: ref %tuple.type.259) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc18(%self.param: ref %ptr.03d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc18(%self.param: ref %ptr.03d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc18(%self.param: ref %ptr.03d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc16(%self.param: ref %struct_type.a.b.577) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_3.1: ref %i32 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc16_3.1: <bound method> = bound_method %.loc16_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc16_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc16_3.1(%.loc16_3.1)
// CHECK:STDOUT:   %.loc16_3.2: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc16_3.2: <bound method> = bound_method %.loc16_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc16_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc16_3.2(%.loc16_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc16(%self.param: ref %struct_type.a.b.577) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc16(%self.param: ref %struct_type.a.b.577) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
