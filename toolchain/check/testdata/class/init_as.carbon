// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/init_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/init_as.carbon

class Class {
  var a: i32;
  var b: i32;
}

fn F() -> i32 {
  //@dump-sem-ir-begin
  return ({.a = 1, .b = 2} as Class).a;
  //@dump-sem-ir-end
}

fn G() -> i32 {
  //@dump-sem-ir-begin
  var v: Class = {.a = 1, .b = 2} as Class;
  //@dump-sem-ir-end
  return v.a;
}

// CHECK:STDOUT: --- init_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.1e8: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.67d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.874: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.ca1: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.92b: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%int_1.874, %int_2.92b) [concrete]
// CHECK:STDOUT:   %.cf6: ref %Class = temporary invalid, %Class.val [concrete]
// CHECK:STDOUT:   %.cbe: ref %i32 = class_element_access %.cf6, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b4b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.175: %Int.as.Copy.impl.Op.type.b4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.175, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %self.param_patt.59b: %pattern_type.df9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cc8: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt.59b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20_28.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc20_28.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc20_28.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.20b: type = pattern_type %struct_type.a.b.1e8 [concrete]
// CHECK:STDOUT:   %self.param_patt.b88: %pattern_type.20b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.25c: %pattern_type.20b = wrapper_binding_pattern self, %self.param_patt.b88 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20_28.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc20_28.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f15: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %self.param_patt.b2b: %pattern_type.f15 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.f19: %pattern_type.f15 = wrapper_binding_pattern self, %self.param_patt.b2b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20_28.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc20_28.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc20_28.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.cf6, %Destroy.WithSelf.SelfDestruct.db3fdb.3 [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.f15 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.f15 = var_pattern %v.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc20_28.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc20_28.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc20_28.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.20b = ref_param_pattern [concrete = constants.%self.param_patt.b88]
// CHECK:STDOUT:     %self.patt: %pattern_type.20b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.25c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b.1e8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b.1e8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc20_28.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.20b = ref_param_pattern [concrete = constants.%self.param_patt.b88]
// CHECK:STDOUT:     %self.patt: %pattern_type.20b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.25c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b.1e8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b.1e8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc20_28.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f15 = ref_param_pattern [concrete = constants.%self.param_patt.b2b]
// CHECK:STDOUT:     %self.patt: %pattern_type.f15 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f19]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Class = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc20_28.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f15 = ref_param_pattern [concrete = constants.%self.param_patt.b2b]
// CHECK:STDOUT:     %self.patt: %pattern_type.f15 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f19]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Class = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc20_26.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %impl.elem0.loc20_26.1: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc20_26.1: <bound method> = bound_method %int_1, %impl.elem0.loc20_26.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f]
// CHECK:STDOUT:   %specific_fn.loc20_26.1: <specific function> = specific_function %impl.elem0.loc20_26.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_26.2: <bound method> = bound_method %int_1, %specific_fn.loc20_26.1 [concrete = constants.%bound_method.67d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_26.1: init %i32 = call %bound_method.loc20_26.2(%int_1) [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc20_26.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_26.1 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc20_26.3: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc20_26.4: ref %i32 = class_element_access %.loc20_26.3, element0
// CHECK:STDOUT:   %.loc20_26.5: init %i32 to %.loc20_26.4 = in_place_init %.loc20_26.2 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %impl.elem0.loc20_26.2: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc20_26.3: <bound method> = bound_method %int_2, %impl.elem0.loc20_26.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc20_26.2: <specific function> = specific_function %impl.elem0.loc20_26.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_26.4: <bound method> = bound_method %int_2, %specific_fn.loc20_26.2 [concrete = constants.%bound_method.ca1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_26.2: init %i32 = call %bound_method.loc20_26.4(%int_2) [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc20_26.6: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_26.2 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc20_26.7: ref %i32 = class_element_access %.loc20_26.3, element1
// CHECK:STDOUT:   %.loc20_26.8: init %i32 to %.loc20_26.7 = in_place_init %.loc20_26.6 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc20_26.9: init %Class to %.loc20_26.3 = class_init (%.loc20_26.5, %.loc20_26.8) [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc20_28.1: init %Class = converted %.loc20_26.1, %.loc20_26.9 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc20_28.2: ref %Class = temporary %.loc20_26.3, %.loc20_28.1 [concrete = constants.%.cf6]
// CHECK:STDOUT:   %a.ref: %Class.elem = name_ref a, @Class.%field_decl.loc14 [concrete = @Class.%field_decl.loc14]
// CHECK:STDOUT:   %.loc20_37.1: ref %i32 = class_element_access %.loc20_28.2, element0 [concrete = constants.%.cbe]
// CHECK:STDOUT:   %.loc20_37.2: %i32 = acquire_value %.loc20_37.1
// CHECK:STDOUT:   %impl.elem0.loc20_37: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:   %bound_method.loc20_37.1: <bound method> = bound_method %.loc20_37.2, %impl.elem0.loc20_37
// CHECK:STDOUT:   %specific_fn.loc20_37: <specific function> = specific_function %impl.elem0.loc20_37, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_37.2: <bound method> = bound_method %.loc20_37.2, %specific_fn.loc20_37
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc20_37.2(%.loc20_37.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.cf6)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc20_28.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc20_28.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc20_28.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc20_28.2(%self.param: ref %struct_type.a.b.1e8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_28.1: ref %i32 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc20_28.1: <bound method> = bound_method %.loc20_28.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc20_28.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc20_28.1(%.loc20_28.1)
// CHECK:STDOUT:   %.loc20_28.2: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc20_28.2: <bound method> = bound_method %.loc20_28.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc20_28.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc20_28.2(%.loc20_28.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc20_28.2(%self.param: ref %struct_type.a.b.1e8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc20_28.2(%self.param: ref %struct_type.a.b.1e8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc20_28.3(%self.param: ref %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: %Class.elem = name_ref b, @Class.%field_decl.loc15 [concrete = @Class.%field_decl.loc15]
// CHECK:STDOUT:   %.loc20_28.1: ref %i32 = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc20_28.1: <bound method> = bound_method %.loc20_28.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc20_28.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc20_28.1(%.loc20_28.1)
// CHECK:STDOUT:   %a.ref: %Class.elem = name_ref a, @Class.%field_decl.loc14 [concrete = @Class.%field_decl.loc14]
// CHECK:STDOUT:   %.loc20_28.2: ref %i32 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc20_28.2: <bound method> = bound_method %.loc20_28.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc20_28.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc20_28.2(%.loc20_28.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc20_28.3(%self.param: ref %Class) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc20_28.3(%self.param: ref %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %Class = var_storage %v.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc26_33.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %Class.ref.loc26_38: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %impl.elem0.loc26_33.1: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc26_33.1: <bound method> = bound_method %int_1, %impl.elem0.loc26_33.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f]
// CHECK:STDOUT:   %specific_fn.loc26_33.1: <specific function> = specific_function %impl.elem0.loc26_33.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_33.2: <bound method> = bound_method %int_1, %specific_fn.loc26_33.1 [concrete = constants.%bound_method.67d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_33.1: init %i32 = call %bound_method.loc26_33.2(%int_1) [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc26_33.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_33.1 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc26_3: ref %Class = splice_block %v.var {}
// CHECK:STDOUT:   %.loc26_33.3: ref %i32 = class_element_access %.loc26_3, element0
// CHECK:STDOUT:   %.loc26_33.4: init %i32 to %.loc26_33.3 = in_place_init %.loc26_33.2 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %impl.elem0.loc26_33.2: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc26_33.3: <bound method> = bound_method %int_2, %impl.elem0.loc26_33.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc26_33.2: <specific function> = specific_function %impl.elem0.loc26_33.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_33.4: <bound method> = bound_method %int_2, %specific_fn.loc26_33.2 [concrete = constants.%bound_method.ca1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_33.2: init %i32 = call %bound_method.loc26_33.4(%int_2) [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc26_33.5: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_33.2 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc26_33.6: ref %i32 = class_element_access %.loc26_3, element1
// CHECK:STDOUT:   %.loc26_33.7: init %i32 to %.loc26_33.6 = in_place_init %.loc26_33.5 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc26_33.8: init %Class to %.loc26_3 = class_init (%.loc26_33.4, %.loc26_33.7) [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc26_35: init %Class = converted %.loc26_33.1, %.loc26_33.8 [concrete = constants.%Class.val]
// CHECK:STDOUT:   assign %v.var, %.loc26_35
// CHECK:STDOUT:   %Class.ref.loc26_10: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %v: ref %Class = wrapper_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.f15 = ref_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.f15 = var_pattern %v.patt [concrete = constants.%v.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %v.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%v.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
