// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/typedef.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/typedef.carbon

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
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.843: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c3c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.011: %ptr.as.Copy.impl.Op.type.c3c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.235, (%Copy.impl_witness.843) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e01: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a62: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e01, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.011, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc10 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc8 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = @F.%i32.4
// CHECK:STDOUT:     .bar = @F.%i32.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.7ce = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var %n.var_patt
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0.loc8: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc8_3.1: <bound method> = bound_method %int_42, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_3.2: <bound method> = bound_method %int_42, %specific_fn.loc8 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_3.2(%int_42) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc8_3: init %i32 = converted %int_42, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   assign %n.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %foo.ref [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %foo.ref: type = name_ref foo, %i32.4 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = ref_binding n, %n.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.fe8 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.fe8 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.235 = var %p.var_patt
// CHECK:STDOUT:   %n.ref: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %n.ref
// CHECK:STDOUT:   %impl.elem0.loc10: %.a62 = impl_witness_access constants.%Copy.impl_witness.843, element0 [concrete = constants.%ptr.as.Copy.impl.Op.011]
// CHECK:STDOUT:   %bound_method.loc10_28.1: <bound method> = bound_method %addr, %impl.elem0.loc10
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_28.2: <bound method> = bound_method %addr, %specific_fn.loc10
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.235 = call %bound_method.loc10_28.2(%addr)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bar.ref: type = name_ref bar, %i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %bar.ref [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.235 = ref_binding p, %p.var
// CHECK:STDOUT:   %DestroyOp.bound.loc10: <bound method> = bound_method %p.var, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc10: init %empty_tuple.type = call %DestroyOp.bound.loc10(%p.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc8: <bound method> = bound_method %n.var, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc8: init %empty_tuple.type = call %DestroyOp.bound.loc8(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10(%self.param: ref %ptr.235) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc8(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_class_typedef.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.a31: type = pattern_type %ptr.d9e [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.83a: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.e61: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f1d: %ptr.as.Copy.impl.Op.type.e61 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.d9e, (%Copy.impl_witness.83a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.5d0: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.798: type = fn_type_with_self_type %Copy.WithSelf.Op.type.5d0, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.f1d, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param: ref %C, %d.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pd.patt: %pattern_type.a31 = ref_binding_pattern pd [concrete]
// CHECK:STDOUT:     %pd.var_patt: %pattern_type.a31 = var_pattern %pd.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pd.var: ref %ptr.d9e = var %pd.var_patt
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr.loc9: %ptr.d9e = addr_of %c.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.798 = impl_witness_access constants.%Copy.impl_witness.83a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.f1d]
// CHECK:STDOUT:   %bound_method.loc9_27.1: <bound method> = bound_method %addr.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @ptr.as.Copy.impl.Op(constants.%C) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_27.2: <bound method> = bound_method %addr.loc9, %specific_fn.loc9
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc9: init %ptr.d9e = call %bound_method.loc9_27.2(%addr.loc9)
// CHECK:STDOUT:   assign %pd.var, %ptr.as.Copy.impl.Op.call.loc9
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9 [concrete = constants.%ptr.d9e] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %D.ref.loc9: type = name_ref D, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %D.ref.loc9 [concrete = constants.%ptr.d9e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pd: ref %ptr.d9e = ref_binding pd, %pd.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pc.patt: %pattern_type.a31 = ref_binding_pattern pc [concrete]
// CHECK:STDOUT:     %pc.var_patt: %pattern_type.a31 = var_pattern %pc.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pc.var: ref %ptr.d9e = var %pc.var_patt
// CHECK:STDOUT:   %d.ref: ref %C = name_ref d, %d
// CHECK:STDOUT:   %addr.loc10: %ptr.d9e = addr_of %d.ref
// CHECK:STDOUT:   %impl.elem0.loc10: %.798 = impl_witness_access constants.%Copy.impl_witness.83a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.f1d]
// CHECK:STDOUT:   %bound_method.loc10_27.1: <bound method> = bound_method %addr.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @ptr.as.Copy.impl.Op(constants.%C) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_27.2: <bound method> = bound_method %addr.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc10: init %ptr.d9e = call %bound_method.loc10_27.2(%addr.loc10)
// CHECK:STDOUT:   assign %pc.var, %ptr.as.Copy.impl.Op.call.loc10
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr.loc10 [concrete = constants.%ptr.d9e] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %C.ref.loc10 [concrete = constants.%ptr.d9e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pc: ref %ptr.d9e = ref_binding pc, %pc.var
// CHECK:STDOUT:   %DestroyOp.bound.loc10: <bound method> = bound_method %pc.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc10: init %empty_tuple.type = call %DestroyOp.bound.loc10(%pc.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc9: <bound method> = bound_method %pd.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc9: init %empty_tuple.type = call %DestroyOp.bound.loc9(%pd.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %ptr.d9e) = "no_op";
// CHECK:STDOUT:
