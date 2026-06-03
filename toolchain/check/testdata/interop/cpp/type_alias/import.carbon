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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.ac4: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.f00: type = pattern_type %ptr.d08 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ce1: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c1c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.d1c: %ptr.as.Copy.impl.Op.type.c1c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.d08, (%Copy.impl_witness.ce1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f9d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.ac9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f9d, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.d1c, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = @F.%i32.4
// CHECK:STDOUT:     .bar = @F.%i32.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.6b6 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.6b6 = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var %n.var_patt
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0.loc8: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_3.1: <bound method> = bound_method %int_42, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_3.2: <bound method> = bound_method %int_42, %specific_fn.loc8 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_3.2(%int_42) [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc8_3: init %i32 = converted %int_42, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   assign %n.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %foo.ref [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %foo.ref: type = name_ref foo, %i32.4 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = ref_binding n, %n.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.f00 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.f00 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.d08 = var %p.var_patt
// CHECK:STDOUT:   %n.ref: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %addr: %ptr.d08 = addr_of %n.ref
// CHECK:STDOUT:   %impl.elem0.loc10: %.ac9 = impl_witness_access constants.%Copy.impl_witness.ce1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.d1c]
// CHECK:STDOUT:   %bound_method.loc10_28.1: <bound method> = bound_method %addr, %impl.elem0.loc10
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_28.2: <bound method> = bound_method %addr, %specific_fn.loc10
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.d08 = call %bound_method.loc10_28.2(%addr)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr [concrete = constants.%ptr.d08] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bar.ref: type = name_ref bar, %i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %bar.ref [concrete = constants.%ptr.d08]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.d08 = ref_binding p, %p.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %p.var, constants.%Destroy.Op.1a2547.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%p.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %n.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10(%self.param: ref %ptr.d08) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_class_typedef.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.0a2: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.8fa: type = pattern_type %ptr.0a2 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.00a: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.1e1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.4ad: %ptr.as.Copy.impl.Op.type.1e1 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.0a2, (%Copy.impl_witness.00a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.aaa: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.51b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.aaa, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.4ad, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param: ref %C, %d.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pd.patt: %pattern_type.8fa = ref_binding_pattern pd [concrete]
// CHECK:STDOUT:     %pd.var_patt: %pattern_type.8fa = var_pattern %pd.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pd.var: ref %ptr.0a2 = var %pd.var_patt
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr.loc9: %ptr.0a2 = addr_of %c.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.51b = impl_witness_access constants.%Copy.impl_witness.00a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.4ad]
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
// CHECK:STDOUT:   %pd: ref %ptr.0a2 = ref_binding pd, %pd.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pc.patt: %pattern_type.8fa = ref_binding_pattern pc [concrete]
// CHECK:STDOUT:     %pc.var_patt: %pattern_type.8fa = var_pattern %pc.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pc.var: ref %ptr.0a2 = var %pc.var_patt
// CHECK:STDOUT:   %d.ref: ref %C = name_ref d, %d
// CHECK:STDOUT:   %addr.loc10: %ptr.0a2 = addr_of %d.ref
// CHECK:STDOUT:   %impl.elem0.loc10: %.51b = impl_witness_access constants.%Copy.impl_witness.00a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.4ad]
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
// CHECK:STDOUT:   %pc: ref %ptr.0a2 = ref_binding pc, %pc.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %pc.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%pc.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %pd.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%pd.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %ptr.0a2) = "no_op";
// CHECK:STDOUT:
