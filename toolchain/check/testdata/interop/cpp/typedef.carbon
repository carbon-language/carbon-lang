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
  var p: Cpp.bar* = &n;
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
  var pd: Cpp.D* = &c;
  var pc: Cpp.C* = &d;
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
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_42.20e, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Op.type.ba2: type = fn_type @Op.3, @Destroy.impl(%ptr.235) [concrete]
// CHECK:STDOUT:   %Op.649: %Op.type.ba2 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5d5: type = ptr_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Op.type.a17: type = fn_type @Op.3, @Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Op.e6a: %Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = @F.%i32.1
// CHECK:STDOUT:     .bar = @F.%i32.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.7ce = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var %n.var_patt
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc8_3.1: <bound method> = bound_method %int_42, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_3.2: <bound method> = bound_method %int_42, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc8_3.2(%int_42) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc8_3: init %i32 = converted %int_42, %int.convert_checked [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   assign %n.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %foo.ref [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %foo.ref: type = name_ref foo, %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %n.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.fe8 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.fe8 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.235 = var %p.var_patt
// CHECK:STDOUT:   %n.ref: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %addr.loc10_21: %ptr.235 = addr_of %n.ref
// CHECK:STDOUT:   assign %p.var, %addr.loc10_21
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bar.ref: type = name_ref bar, %i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %bar.ref [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.235 = bind_name p, %p.var
// CHECK:STDOUT:   %Op.bound.loc10: <bound method> = bound_method %p.var, constants.%Op.649
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %p.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_3: %ptr.5d5 = addr_of %p.var
// CHECK:STDOUT:   %no_op.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_3)
// CHECK:STDOUT:   %Op.bound.loc8: <bound method> = bound_method %n.var, constants.%Op.e6a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_3.3: <bound method> = bound_method %n.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8: %ptr.235 = addr_of %n.var
// CHECK:STDOUT:   %no_op.loc8: init %empty_tuple.type = call %bound_method.loc8_3.3(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_class_typedef.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.a31: type = pattern_type %ptr.d9e [concrete]
// CHECK:STDOUT:   %Op.type.cb8: type = fn_type @Op.2, @Destroy.impl(%ptr.d9e) [concrete]
// CHECK:STDOUT:   %Op.8e3: %Op.type.cb8 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.01d: type = ptr_type %ptr.d9e [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param: %C, %d.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pd.patt: %pattern_type.a31 = binding_pattern pd [concrete]
// CHECK:STDOUT:     %pd.var_patt: %pattern_type.a31 = var_pattern %pd.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pd.var: ref %ptr.d9e = var %pd.var_patt
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr.loc9_20: %ptr.d9e = addr_of %c.ref
// CHECK:STDOUT:   assign %pd.var, %addr.loc9_20
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9 [concrete = constants.%ptr.d9e] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %D.ref.loc9: type = name_ref D, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %D.ref.loc9 [concrete = constants.%ptr.d9e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pd: ref %ptr.d9e = bind_name pd, %pd.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pc.patt: %pattern_type.a31 = binding_pattern pc [concrete]
// CHECK:STDOUT:     %pc.var_patt: %pattern_type.a31 = var_pattern %pc.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pc.var: ref %ptr.d9e = var %pc.var_patt
// CHECK:STDOUT:   %d.ref: ref %C = name_ref d, %d
// CHECK:STDOUT:   %addr.loc10_20: %ptr.d9e = addr_of %d.ref
// CHECK:STDOUT:   assign %pc.var, %addr.loc10_20
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr.loc10 [concrete = constants.%ptr.d9e] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %C.ref.loc10 [concrete = constants.%ptr.d9e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pc: ref %ptr.d9e = bind_name pc, %pc.var
// CHECK:STDOUT:   %Op.bound.loc10: <bound method> = bound_method %pc.var, constants.%Op.8e3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %pc.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_3: %ptr.01d = addr_of %pc.var
// CHECK:STDOUT:   %no_op.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_3)
// CHECK:STDOUT:   %Op.bound.loc9: <bound method> = bound_method %pd.var, constants.%Op.8e3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %pd.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc9_3: %ptr.01d = addr_of %pd.var
// CHECK:STDOUT:   %no_op.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
