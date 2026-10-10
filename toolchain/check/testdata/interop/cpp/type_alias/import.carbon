// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/type_alias/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/type_alias/import.carbon

// --- i32_typedef.h

typedef int foo;

using bar = int;

// --- use_i32_typedef.carbon

library "[[@TEST_NAME]]";

import Cpp library "i32_typedef.h";

fn F() {
  //@dump-sem-ir-begin
  var n: Cpp.foo = 42;
  // Types are equivalent.
  var unused p: Cpp.bar* = &n;
  //@dump-sem-ir-end
}

// --- class_typedef.h

class C {};
using D = C;

// --- use_class_typedef.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_typedef.h";

fn H(var c: Cpp.C, var d: Cpp.D) {
  //@dump-sem-ir-begin
  // Types are equivalent.
  var unused pd: Cpp.D* = &c;
  var unused pc: Cpp.C* = &d;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- use_i32_typedef.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.df9 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.df9 = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.02b: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %ptr.49e: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.50c: type = pattern_type %ptr.49e [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.50c = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.50c = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.dd4: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.d83: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.22e: %ptr.as.Copy.impl.Op.type.d83 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.49e, (%Copy.impl_witness.dd4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.037: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.29f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.037, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.22e, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %self.param_patt.58e: %pattern_type.50c = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5f2: %pattern_type.50c = wrapper_binding_pattern self, %self.param_patt.58e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.59b: %pattern_type.df9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cc8: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt.59b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = @F.%i32.4
// CHECK:STDOUT:     .bar = @F.%i32.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.50c = ref_param_pattern [concrete = constants.%self.param_patt.58e]
// CHECK:STDOUT:     %self.patt: %pattern_type.50c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5f2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.49e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.49e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc10 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.50c = ref_param_pattern [concrete = constants.%self.param_patt.58e]
// CHECK:STDOUT:     %self.patt: %pattern_type.50c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5f2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.49e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.49e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.var: ref %i32 = var_storage %n.var_patt
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0.loc8: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc8_3.1: <bound method> = bound_method %int_42, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_3.2: <bound method> = bound_method %int_42, %specific_fn.loc8 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_3.2(%int_42) [concrete = constants.%int_42.02b]
// CHECK:STDOUT:   %.loc8_3: init %i32 = converted %int_42, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_42.02b]
// CHECK:STDOUT:   assign %n.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %foo.ref [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %foo.ref: type = name_ref foo, %i32.4 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = wrapper_binding n, %n.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.df9 = ref_binding_pattern n [concrete = constants.%n.patt]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.df9 = var_pattern %n.patt [concrete = constants.%n.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.49e = var_storage %p.var_patt
// CHECK:STDOUT:   %n.ref: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %addr: %ptr.49e = addr_of %n.ref
// CHECK:STDOUT:   %impl.elem0.loc10: %.29f = impl_witness_access constants.%Copy.impl_witness.dd4, element0 [concrete = constants.%ptr.as.Copy.impl.Op.22e]
// CHECK:STDOUT:   %bound_method.loc10_28.1: <bound method> = bound_method %addr, %impl.elem0.loc10
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_28.2: <bound method> = bound_method %addr, %specific_fn.loc10
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.49e = call %bound_method.loc10_28.2(%addr)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr [concrete = constants.%ptr.49e] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bar.ref: type = name_ref bar, %i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %bar.ref [concrete = constants.%ptr.49e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.49e = wrapper_binding p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.50c = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.50c = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10: <bound method> = bound_method %p.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10(%p.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc8: <bound method> = bound_method %n.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc8(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10(%self.param: ref %ptr.49e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10(%self.param: ref %ptr.49e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10(%self.param: ref %ptr.49e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_class_typedef.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.0a2: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.8fa: type = pattern_type %ptr.0a2 [concrete]
// CHECK:STDOUT:   %pd.patt: %pattern_type.8fa = ref_binding_pattern pd [concrete]
// CHECK:STDOUT:   %pd.var_patt: %pattern_type.8fa = var_pattern %pd.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a1b: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.1e1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.4ad: %ptr.as.Copy.impl.Op.type.1e1 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.0a2, (%Copy.impl_witness.a1b) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e7e: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.ff1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e7e, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.4ad, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT:   %pc.patt: %pattern_type.8fa = ref_binding_pattern pc [concrete]
// CHECK:STDOUT:   %pc.var_patt: %pattern_type.8fa = var_pattern %pc.patt [concrete]
// CHECK:STDOUT:   %self.param_patt.7a2: %pattern_type.8fa = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.450: %pattern_type.8fa = wrapper_binding_pattern self, %self.param_patt.7a2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8fa = ref_param_pattern [concrete = constants.%self.param_patt.7a2]
// CHECK:STDOUT:     %self.patt: %pattern_type.8fa = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.450]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.0a2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.0a2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc10 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8fa = ref_param_pattern [concrete = constants.%self.param_patt.7a2]
// CHECK:STDOUT:     %self.patt: %pattern_type.8fa = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.450]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ptr.0a2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ptr.0a2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param: ref %C, %d.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %pd.var: ref %ptr.0a2 = var_storage %pd.var_patt
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr.loc9: %ptr.0a2 = addr_of %c.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.ff1 = impl_witness_access constants.%Copy.impl_witness.a1b, element0 [concrete = constants.%ptr.as.Copy.impl.Op.4ad]
// CHECK:STDOUT:   %bound_method.loc9_27.1: <bound method> = bound_method %addr.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @ptr.as.Copy.impl.Op(constants.%C) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_27.2: <bound method> = bound_method %addr.loc9, %specific_fn.loc9
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc9: init %ptr.0a2 = call %bound_method.loc9_27.2(%addr.loc9)
// CHECK:STDOUT:   assign %pd.var, %ptr.as.Copy.impl.Op.call.loc9
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9 [concrete = constants.%ptr.0a2] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %D.ref.loc9: type = name_ref D, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %D.ref.loc9 [concrete = constants.%ptr.0a2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pd: ref %ptr.0a2 = wrapper_binding pd, %pd.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pd.patt: %pattern_type.8fa = ref_binding_pattern pd [concrete = constants.%pd.patt]
// CHECK:STDOUT:     %pd.var_patt: %pattern_type.8fa = var_pattern %pd.patt [concrete = constants.%pd.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pc.var: ref %ptr.0a2 = var_storage %pc.var_patt
// CHECK:STDOUT:   %d.ref: ref %C = name_ref d, %d
// CHECK:STDOUT:   %addr.loc10: %ptr.0a2 = addr_of %d.ref
// CHECK:STDOUT:   %impl.elem0.loc10: %.ff1 = impl_witness_access constants.%Copy.impl_witness.a1b, element0 [concrete = constants.%ptr.as.Copy.impl.Op.4ad]
// CHECK:STDOUT:   %bound_method.loc10_27.1: <bound method> = bound_method %addr.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @ptr.as.Copy.impl.Op(constants.%C) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_27.2: <bound method> = bound_method %addr.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc10: init %ptr.0a2 = call %bound_method.loc10_27.2(%addr.loc10)
// CHECK:STDOUT:   assign %pc.var, %ptr.as.Copy.impl.Op.call.loc10
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr.loc10 [concrete = constants.%ptr.0a2] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %C.ref.loc10 [concrete = constants.%ptr.0a2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pc: ref %ptr.0a2 = wrapper_binding pc, %pc.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pc.patt: %pattern_type.8fa = ref_binding_pattern pc [concrete = constants.%pc.patt]
// CHECK:STDOUT:     %pc.var_patt: %pattern_type.8fa = var_pattern %pc.patt [concrete = constants.%pc.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10: <bound method> = bound_method %pc.var, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10(%pc.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9: <bound method> = bound_method %pd.var, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9(%pd.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10(%self.param: ref %ptr.0a2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10(%self.param: ref %ptr.0a2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10(%self.param: ref %ptr.0a2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
