// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/call_combined_impl_witness.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/call_combined_impl_witness.carbon

interface Empty {
}
interface A {
  fn AA();
}
interface B {
  fn BB();
}

class C {}
impl C as Empty {}
impl C as A {
  fn AA() {}
}
impl C as B {
  fn BB() {}
}

fn G[T:! A & Empty & B](t: T) {
  t.AA();
  t.BB();

  T.AA();
  T.BB();

  T.(A.AA)();
  T.(B.BB)();
}

fn F() {
  G({} as C);
}

// CHECK:STDOUT: --- call_combined_impl_witness.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.193: %Empty.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.31d: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %AA.type.b97: type = fn_type @AA.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AA.c3a: %AA.type.b97 = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type %A.type [concrete]
// CHECK:STDOUT:   %assoc0.f33: %A.assoc_type = assoc_entity element0, @A.%AA.decl [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.783: %B.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BB.type.64d: type = fn_type @BB.1 [concrete]
// CHECK:STDOUT:   %BB.11b: %BB.type.64d = struct_value () [concrete]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type %B.type [concrete]
// CHECK:STDOUT:   %assoc0.019: %B.assoc_type = assoc_entity element0, @B.%BB.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %impl_witness.97a: <witness> = impl_witness (@impl.77d.%AA.decl) [concrete]
// CHECK:STDOUT:   %AA.type.c29: type = fn_type @AA.2 [concrete]
// CHECK:STDOUT:   %AA.95d: %AA.type.c29 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.213: %A.type = facet_value %C, (%impl_witness.97a) [concrete]
// CHECK:STDOUT:   %impl_witness.fe4: <witness> = impl_witness (@impl.c94.%BB.decl) [concrete]
// CHECK:STDOUT:   %BB.type.24d: type = fn_type @BB.2 [concrete]
// CHECK:STDOUT:   %BB.fe8: %BB.type.24d = struct_value () [concrete]
// CHECK:STDOUT:   %B.facet.f74: %B.type = facet_value %C, (%impl_witness.fe4) [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.25f: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0.a63: %BitAnd.assoc_type = assoc_entity element0, imports.%Core.import_ref.a93 [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.25f [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.db8: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.2, @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %impl_witness.3ea: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(type) [concrete]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.2, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.473: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%impl_witness.3ea) [concrete]
// CHECK:STDOUT:   %.2ac: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound.6b1: <bound method> = bound_method %A.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.444, @Op.2(type) [concrete]
// CHECK:STDOUT:   %bound_method.96c: <bound method> = bound_method %A.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type.d5f: type = facet_type <@Empty & @A> [concrete]
// CHECK:STDOUT:   %Op.bound.b5b: <bound method> = bound_method %facet_type.d5f, %Op.444 [concrete]
// CHECK:STDOUT:   %bound_method.cb6: <bound method> = bound_method %facet_type.d5f, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type.242: type = facet_type <@Empty & @A & @B> [concrete]
// CHECK:STDOUT:   %T.2df: %facet_type.242 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.fd9: %facet_type.242 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.2df [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.383: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface1: <witness> = facet_access_witness %T.2df, element1 [symbolic]
// CHECK:STDOUT:   %A.facet.5b4: %A.type = facet_value %T.as_type, (%T.as_wit.iface1) [symbolic]
// CHECK:STDOUT:   %.860: type = fn_type_with_self_type %AA.type.b97, %A.facet.5b4 [symbolic]
// CHECK:STDOUT:   %impl.elem0.656: %.860 = impl_witness_access %T.as_wit.iface1, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn.74e: <specific function> = specific_function %impl.elem0.656, @AA.1(%A.facet.5b4) [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface2: <witness> = facet_access_witness %T.2df, element2 [symbolic]
// CHECK:STDOUT:   %B.facet.abf: %B.type = facet_value %T.as_type, (%T.as_wit.iface2) [symbolic]
// CHECK:STDOUT:   %.a5b: type = fn_type_with_self_type %BB.type.64d, %B.facet.abf [symbolic]
// CHECK:STDOUT:   %impl.elem0.097: %.a5b = impl_witness_access %T.as_wit.iface2, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn.9fa: <specific function> = specific_function %impl.elem0.097, @BB.1(%B.facet.abf) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.242 = facet_value %C, (%impl_witness.1bc, %impl_witness.97a, %impl_witness.fe4) [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%facet_value) [concrete]
// CHECK:STDOUT:   %.4b2: type = fn_type_with_self_type %AA.type.b97, %A.facet.213 [concrete]
// CHECK:STDOUT:   %AA.specific_fn: <specific function> = specific_function %AA.95d, @AA.1(%A.facet.213) [concrete]
// CHECK:STDOUT:   %.2b4: type = fn_type_with_self_type %BB.type.64d, %B.facet.f74 [concrete]
// CHECK:STDOUT:   %BB.specific_fn: <specific function> = specific_function %BB.fe8, @BB.1(%B.facet.f74) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst100 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.08d: %BitAnd.assoc_type = import_ref Core//prelude, loc18_41, loaded [concrete = constants.%assoc0.a63]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.040: %BitAnd.type = import_ref Core//prelude, inst100 [no loc], loaded [symbolic = constants.%Self.25f]
// CHECK:STDOUT:   %Core.import_ref.51c: <witness> = import_ref Core//prelude, loc21_36, loaded [symbolic = @impl.f92.%impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.f92.%Op.type (%Op.type.f99) = import_ref Core//prelude, loc22_42, loaded [symbolic = @impl.f92.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [concrete = constants.%Empty.type] {} {}
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.d78 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc24: <witness> = impl_witness () [concrete = constants.%impl_witness.1bc]
// CHECK:STDOUT:   impl_decl @impl.77d [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc25: <witness> = impl_witness (@impl.77d.%AA.decl) [concrete = constants.%impl_witness.97a]
// CHECK:STDOUT:   impl_decl @impl.c94 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc28: <witness> = impl_witness (@impl.c94.%BB.decl) [concrete = constants.%impl_witness.fe4]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc32_6.1: %facet_type.242 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc32_6.2 (constants.%T.patt.fd9)]
// CHECK:STDOUT:     %t.patt: @G.%T.as_type.loc32_28.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @G.%T.as_type.loc32_28.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc32_20.1: type = splice_block %.loc32_20.3 [concrete = constants.%facet_type.242] {
// CHECK:STDOUT:       %A.ref.loc32: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:       %impl.elem0.loc32_12: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc32_12.1: <bound method> = bound_method %A.ref.loc32, %impl.elem0.loc32_12 [concrete = constants.%Op.bound.6b1]
// CHECK:STDOUT:       %specific_fn.loc32_12: <specific function> = specific_function %impl.elem0.loc32_12, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %bound_method.loc32_12.2: <bound method> = bound_method %A.ref.loc32, %specific_fn.loc32_12 [concrete = constants.%bound_method.96c]
// CHECK:STDOUT:       %type.and.loc32_12: init type = call %bound_method.loc32_12.2(%A.ref.loc32, %Empty.ref) [concrete = constants.%facet_type.d5f]
// CHECK:STDOUT:       %B.ref.loc32: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc32_20: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc32_20.1: <bound method> = bound_method %type.and.loc32_12, %impl.elem0.loc32_20 [concrete = constants.%Op.bound.b5b]
// CHECK:STDOUT:       %specific_fn.loc32_20: <specific function> = specific_function %impl.elem0.loc32_20, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %bound_method.loc32_20.2: <bound method> = bound_method %type.and.loc32_12, %specific_fn.loc32_20 [concrete = constants.%bound_method.cb6]
// CHECK:STDOUT:       %.loc32_12.1: type = value_of_initializer %type.and.loc32_12 [concrete = constants.%facet_type.d5f]
// CHECK:STDOUT:       %.loc32_12.2: type = converted %type.and.loc32_12, %.loc32_12.1 [concrete = constants.%facet_type.d5f]
// CHECK:STDOUT:       %type.and.loc32_20: init type = call %bound_method.loc32_20.2(%.loc32_12.2, %B.ref.loc32) [concrete = constants.%facet_type.242]
// CHECK:STDOUT:       %.loc32_20.2: type = value_of_initializer %type.and.loc32_20 [concrete = constants.%facet_type.242]
// CHECK:STDOUT:       %.loc32_20.3: type = converted %type.and.loc32_20, %.loc32_20.2 [concrete = constants.%facet_type.242]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc32_6.1: %facet_type.242 = bind_symbolic_name T, 0 [symbolic = %T.loc32_6.2 (constants.%T.2df)]
// CHECK:STDOUT:     %t.param: @G.%T.as_type.loc32_28.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc32_28.1: type = splice_block %.loc32_28.2 [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc32: %facet_type.242 = name_ref T, %T.loc32_6.1 [symbolic = %T.loc32_6.2 (constants.%T.2df)]
// CHECK:STDOUT:       %T.as_type.loc32_28.1: type = facet_access_type %T.ref.loc32 [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc32_28.2: type = converted %T.ref.loc32, %T.as_type.loc32_28.1 [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.as_type.loc32_28.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.193]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .AA = <poisoned>
// CHECK:STDOUT:   .BB = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.31d]
// CHECK:STDOUT:   %AA.decl: %AA.type.b97 = fn_decl @AA.1 [concrete = constants.%AA.c3a] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %AA.decl [concrete = constants.%assoc0.f33]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .AA = %assoc0
// CHECK:STDOUT:   .BB = <poisoned>
// CHECK:STDOUT:   witness = (%AA.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.783]
// CHECK:STDOUT:   %BB.decl: %BB.type.64d = fn_decl @BB.1 [concrete = constants.%BB.11b] {} {}
// CHECK:STDOUT:   %assoc0: %B.assoc_type = assoc_entity element0, %BB.decl [concrete = constants.%assoc0.019]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .BB = %assoc0
// CHECK:STDOUT:   .AA = <poisoned>
// CHECK:STDOUT:   witness = (%BB.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "toolchain/testing/min_prelude/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.08d
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.d78: %C.ref as %Empty.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.77d: %C.ref as %A.ref {
// CHECK:STDOUT:   %AA.decl: %AA.type.c29 = fn_decl @AA.2 [concrete = constants.%AA.95d] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .AA = %AA.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c94: %C.ref as %B.ref {
// CHECK:STDOUT:   %BB.decl: %BB.type.24d = fn_decl @BB.2 [concrete = constants.%BB.fe8] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .BB = %BB.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.1: type) [from "toolchain/testing/min_prelude/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.e01)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.db8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type.f99)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type.f99) = struct_value () [symbolic = %Op (constants.%Op.05a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T.8b3) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.51c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AA.1(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BB.1(@B.%Self: %B.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AA.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BB.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.040: %BitAnd.type) [from "toolchain/testing/min_prelude/facet_types.carbon"] {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.25f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type (%Self.as_type)](%other.param_patt: @Op.1.%Self.as_type (%Self.as_type)) -> @Op.1.%Self.as_type (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "toolchain/testing/min_prelude/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T.8b3)](%other.param_patt: @Op.2.%T (%T.8b3)) -> @Op.2.%T (%T.8b3) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc32_6.1: %facet_type.242) {
// CHECK:STDOUT:   %T.loc32_6.2: %facet_type.242 = bind_symbolic_name T, 0 [symbolic = %T.loc32_6.2 (constants.%T.2df)]
// CHECK:STDOUT:   %T.patt.loc32_6.2: %facet_type.242 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc32_6.2 (constants.%T.patt.fd9)]
// CHECK:STDOUT:   %T.as_type.loc32_28.2: type = facet_access_type %T.loc32_6.2 [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%T.as_type.loc32_28.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.383)]
// CHECK:STDOUT:   %T.as_wit.iface1.loc33_4.2: <witness> = facet_access_witness %T.loc32_6.2, element1 [symbolic = %T.as_wit.iface1.loc33_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:   %A.facet.loc33: %A.type = facet_value %T.as_type.loc32_28.2, (%T.as_wit.iface1.loc33_4.2) [symbolic = %A.facet.loc33 (constants.%A.facet.5b4)]
// CHECK:STDOUT:   %.loc33_4.2: type = fn_type_with_self_type constants.%AA.type.b97, %A.facet.loc33 [symbolic = %.loc33_4.2 (constants.%.860)]
// CHECK:STDOUT:   %impl.elem0.loc33_4.2: @G.%.loc33_4.2 (%.860) = impl_witness_access %T.as_wit.iface1.loc33_4.2, element0 [symbolic = %impl.elem0.loc33_4.2 (constants.%impl.elem0.656)]
// CHECK:STDOUT:   %specific_fn.loc33_4.2: <specific function> = specific_function %impl.elem0.loc33_4.2, @AA.1(%A.facet.loc33) [symbolic = %specific_fn.loc33_4.2 (constants.%specific_fn.74e)]
// CHECK:STDOUT:   %T.as_wit.iface2.loc34_4.2: <witness> = facet_access_witness %T.loc32_6.2, element2 [symbolic = %T.as_wit.iface2.loc34_4.2 (constants.%T.as_wit.iface2)]
// CHECK:STDOUT:   %B.facet.loc34: %B.type = facet_value %T.as_type.loc32_28.2, (%T.as_wit.iface2.loc34_4.2) [symbolic = %B.facet.loc34 (constants.%B.facet.abf)]
// CHECK:STDOUT:   %.loc34_4.2: type = fn_type_with_self_type constants.%BB.type.64d, %B.facet.loc34 [symbolic = %.loc34_4.2 (constants.%.a5b)]
// CHECK:STDOUT:   %impl.elem0.loc34_4.2: @G.%.loc34_4.2 (%.a5b) = impl_witness_access %T.as_wit.iface2.loc34_4.2, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.097)]
// CHECK:STDOUT:   %specific_fn.loc34_4.2: <specific function> = specific_function %impl.elem0.loc34_4.2, @BB.1(%B.facet.loc34) [symbolic = %specific_fn.loc34_4.2 (constants.%specific_fn.9fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc32_6.1: %facet_type.242](%t.param_patt: @G.%T.as_type.loc32_28.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref.loc33: @G.%T.as_type.loc32_28.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %AA.ref.loc33: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.f33]
// CHECK:STDOUT:     %T.as_type.loc33: type = facet_access_type constants.%T.2df [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc33_4.1: type = converted constants.%T.2df, %T.as_type.loc33 [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface1.loc33_4.1: <witness> = facet_access_witness constants.%T.2df, element1 [symbolic = %T.as_wit.iface1.loc33_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:     %impl.elem0.loc33_4.1: @G.%.loc33_4.2 (%.860) = impl_witness_access %T.as_wit.iface1.loc33_4.1, element0 [symbolic = %impl.elem0.loc33_4.2 (constants.%impl.elem0.656)]
// CHECK:STDOUT:     %specific_fn.loc33_4.1: <specific function> = specific_function %impl.elem0.loc33_4.1, @AA.1(constants.%A.facet.5b4) [symbolic = %specific_fn.loc33_4.2 (constants.%specific_fn.74e)]
// CHECK:STDOUT:     %AA.call.loc33: init %empty_tuple.type = call %specific_fn.loc33_4.1()
// CHECK:STDOUT:     %t.ref.loc34: @G.%T.as_type.loc32_28.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %BB.ref.loc34: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.019]
// CHECK:STDOUT:     %T.as_type.loc34: type = facet_access_type constants.%T.2df [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc34_4.1: type = converted constants.%T.2df, %T.as_type.loc34 [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface2.loc34_4.1: <witness> = facet_access_witness constants.%T.2df, element2 [symbolic = %T.as_wit.iface2.loc34_4.2 (constants.%T.as_wit.iface2)]
// CHECK:STDOUT:     %impl.elem0.loc34_4.1: @G.%.loc34_4.2 (%.a5b) = impl_witness_access %T.as_wit.iface2.loc34_4.1, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.097)]
// CHECK:STDOUT:     %specific_fn.loc34_4.1: <specific function> = specific_function %impl.elem0.loc34_4.1, @BB.1(constants.%B.facet.abf) [symbolic = %specific_fn.loc34_4.2 (constants.%specific_fn.9fa)]
// CHECK:STDOUT:     %BB.call.loc34: init %empty_tuple.type = call %specific_fn.loc34_4.1()
// CHECK:STDOUT:     %T.ref.loc36: %facet_type.242 = name_ref T, %T.loc32_6.1 [symbolic = %T.loc32_6.2 (constants.%T.2df)]
// CHECK:STDOUT:     %AA.ref.loc36: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.f33]
// CHECK:STDOUT:     %T.as_type.loc36: type = facet_access_type %T.ref.loc36 [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc36: type = converted %T.ref.loc36, %T.as_type.loc36 [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface1.loc36: <witness> = facet_access_witness %T.ref.loc36, element1 [symbolic = %T.as_wit.iface1.loc33_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:     %impl.elem0.loc36: @G.%.loc33_4.2 (%.860) = impl_witness_access %T.as_wit.iface1.loc36, element0 [symbolic = %impl.elem0.loc33_4.2 (constants.%impl.elem0.656)]
// CHECK:STDOUT:     %specific_fn.loc36: <specific function> = specific_function %impl.elem0.loc36, @AA.1(constants.%A.facet.5b4) [symbolic = %specific_fn.loc33_4.2 (constants.%specific_fn.74e)]
// CHECK:STDOUT:     %AA.call.loc36: init %empty_tuple.type = call %specific_fn.loc36()
// CHECK:STDOUT:     %T.ref.loc37: %facet_type.242 = name_ref T, %T.loc32_6.1 [symbolic = %T.loc32_6.2 (constants.%T.2df)]
// CHECK:STDOUT:     %BB.ref.loc37: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.019]
// CHECK:STDOUT:     %T.as_type.loc37: type = facet_access_type %T.ref.loc37 [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc37: type = converted %T.ref.loc37, %T.as_type.loc37 [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface2.loc37: <witness> = facet_access_witness %T.ref.loc37, element2 [symbolic = %T.as_wit.iface2.loc34_4.2 (constants.%T.as_wit.iface2)]
// CHECK:STDOUT:     %impl.elem0.loc37: @G.%.loc34_4.2 (%.a5b) = impl_witness_access %T.as_wit.iface2.loc37, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.097)]
// CHECK:STDOUT:     %specific_fn.loc37: <specific function> = specific_function %impl.elem0.loc37, @BB.1(constants.%B.facet.abf) [symbolic = %specific_fn.loc34_4.2 (constants.%specific_fn.9fa)]
// CHECK:STDOUT:     %BB.call.loc37: init %empty_tuple.type = call %specific_fn.loc37()
// CHECK:STDOUT:     %T.ref.loc39: %facet_type.242 = name_ref T, %T.loc32_6.1 [symbolic = %T.loc32_6.2 (constants.%T.2df)]
// CHECK:STDOUT:     %A.ref.loc39: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %AA.ref.loc39: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.f33]
// CHECK:STDOUT:     %T.as_wit.iface1.loc39: <witness> = facet_access_witness constants.%T.2df, element1 [symbolic = %T.as_wit.iface1.loc33_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:     %T.as_type.loc39: type = facet_access_type constants.%T.2df [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %A.facet.loc39: %A.type = facet_value %T.as_type.loc39, (%T.as_wit.iface1.loc39) [symbolic = %A.facet.loc33 (constants.%A.facet.5b4)]
// CHECK:STDOUT:     %.loc39: %A.type = converted %T.ref.loc39, %A.facet.loc39 [symbolic = %A.facet.loc33 (constants.%A.facet.5b4)]
// CHECK:STDOUT:     %as_wit.iface0.loc39: <witness> = facet_access_witness %.loc39, element0 [symbolic = %T.as_wit.iface1.loc33_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:     %impl.elem0.loc39: @G.%.loc33_4.2 (%.860) = impl_witness_access %as_wit.iface0.loc39, element0 [symbolic = %impl.elem0.loc33_4.2 (constants.%impl.elem0.656)]
// CHECK:STDOUT:     %specific_fn.loc39: <specific function> = specific_function %impl.elem0.loc39, @AA.1(constants.%A.facet.5b4) [symbolic = %specific_fn.loc33_4.2 (constants.%specific_fn.74e)]
// CHECK:STDOUT:     %AA.call.loc39: init %empty_tuple.type = call %specific_fn.loc39()
// CHECK:STDOUT:     %T.ref.loc40: %facet_type.242 = name_ref T, %T.loc32_6.1 [symbolic = %T.loc32_6.2 (constants.%T.2df)]
// CHECK:STDOUT:     %B.ref.loc40: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %BB.ref.loc40: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.019]
// CHECK:STDOUT:     %T.as_wit.iface2.loc40: <witness> = facet_access_witness constants.%T.2df, element2 [symbolic = %T.as_wit.iface2.loc34_4.2 (constants.%T.as_wit.iface2)]
// CHECK:STDOUT:     %T.as_type.loc40: type = facet_access_type constants.%T.2df [symbolic = %T.as_type.loc32_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %B.facet.loc40: %B.type = facet_value %T.as_type.loc40, (%T.as_wit.iface2.loc40) [symbolic = %B.facet.loc34 (constants.%B.facet.abf)]
// CHECK:STDOUT:     %.loc40: %B.type = converted %T.ref.loc40, %B.facet.loc40 [symbolic = %B.facet.loc34 (constants.%B.facet.abf)]
// CHECK:STDOUT:     %as_wit.iface0.loc40: <witness> = facet_access_witness %.loc40, element0 [symbolic = %T.as_wit.iface2.loc34_4.2 (constants.%T.as_wit.iface2)]
// CHECK:STDOUT:     %impl.elem0.loc40: @G.%.loc34_4.2 (%.a5b) = impl_witness_access %as_wit.iface0.loc40, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.097)]
// CHECK:STDOUT:     %specific_fn.loc40: <specific function> = specific_function %impl.elem0.loc40, @BB.1(constants.%B.facet.abf) [symbolic = %specific_fn.loc34_4.2 (constants.%specific_fn.9fa)]
// CHECK:STDOUT:     %BB.call.loc40: init %empty_tuple.type = call %specific_fn.loc40()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc44_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc44_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc44_6.3: init %C = class_init (), %.loc44_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc44_6.4: ref %C = temporary %.loc44_6.2, %.loc44_6.3
// CHECK:STDOUT:   %.loc44_8.1: ref %C = converted %.loc44_6.1, %.loc44_6.4
// CHECK:STDOUT:   %facet_value.loc44_12.1: %facet_type.242 = facet_value constants.%C, (constants.%impl_witness.1bc, constants.%impl_witness.97a, constants.%impl_witness.fe4) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc44_12.1: %facet_type.242 = converted constants.%C, %facet_value.loc44_12.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc44_12.2: %facet_type.242 = facet_value constants.%C, (constants.%impl_witness.1bc, constants.%impl_witness.97a, constants.%impl_witness.fe4) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc44_12.2: %facet_type.242 = converted constants.%C, %facet_value.loc44_12.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%facet_value) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %.loc44_8.2: %C = bind_value %.loc44_8.1
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn(%.loc44_8.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AA.1(constants.%Self.31d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.1(constants.%Self.783) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AA.1(constants.%A.facet.213) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.1(constants.%B.facet.f74) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.25f) {
// CHECK:STDOUT:   %Self => constants.%Self.25f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %T.patt => constants.%T.8b3
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.db8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %T.patt => type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3ea
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.eb8
// CHECK:STDOUT:   %Op => constants.%Op.444
// CHECK:STDOUT:   %require_complete => constants.%complete_type.473
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.2df) {
// CHECK:STDOUT:   %T.loc32_6.2 => constants.%T.2df
// CHECK:STDOUT:   %T.patt.loc32_6.2 => constants.%T.2df
// CHECK:STDOUT:   %T.as_type.loc32_28.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AA.1(constants.%A.facet.5b4) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.1(constants.%B.facet.abf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AA.1(@G.%A.facet.loc33) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.1(@G.%B.facet.loc34) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc32_6.2 => constants.%facet_value
// CHECK:STDOUT:   %T.patt.loc32_6.2 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc32_28.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %T.as_wit.iface1.loc33_4.2 => constants.%impl_witness.97a
// CHECK:STDOUT:   %A.facet.loc33 => constants.%A.facet.213
// CHECK:STDOUT:   %.loc33_4.2 => constants.%.4b2
// CHECK:STDOUT:   %impl.elem0.loc33_4.2 => constants.%AA.95d
// CHECK:STDOUT:   %specific_fn.loc33_4.2 => constants.%AA.specific_fn
// CHECK:STDOUT:   %T.as_wit.iface2.loc34_4.2 => constants.%impl_witness.fe4
// CHECK:STDOUT:   %B.facet.loc34 => constants.%B.facet.f74
// CHECK:STDOUT:   %.loc34_4.2 => constants.%.2b4
// CHECK:STDOUT:   %impl.elem0.loc34_4.2 => constants.%BB.fe8
// CHECK:STDOUT:   %specific_fn.loc34_4.2 => constants.%BB.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- toolchain/testing/min_prelude/facet_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type %As.type.8ba [symbolic]
// CHECK:STDOUT:   %assoc0.ac5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0.a50: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.e44: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.560: type = facet_access_type %Self.e44 [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0.121: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, (%impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt.loc9_14.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc9_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc9_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc13_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc13_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc13_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc21_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc21_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc9_14.1: type) {
// CHECK:STDOUT:   %Dest.loc9_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc9_14.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc9_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc9_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc9_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.8ba) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc10_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc9_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc10_20.1: type = splice_block %.loc10_20.3 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc10_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc10_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc10_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc10_20.3: type = converted %Self.ref, %Self.as_type.loc10_20.2 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc10_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc13_22.1: type) {
// CHECK:STDOUT:   %Dest.loc13_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc13_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc13_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc13_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc14_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc13_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.3 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc14_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc14_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc14_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc14_20.3: type = converted %Self.ref, %Self.as_type.loc14_20.2 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc14_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e44]
// CHECK:STDOUT:   %Op.decl: %Op.type.613 = fn_decl @Op.1 [concrete = constants.%Op.d98] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18_37: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:     %Self.as_type.loc18_37: type = facet_access_type %Self.ref.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %.loc18_37: type = converted %Self.ref.loc18_37, %Self.as_type.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param0
// CHECK:STDOUT:     %.loc18_15.1: type = splice_block %.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_15: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_15.2: type = facet_access_type %Self.ref.loc18_15 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_15.2: type = converted %Self.ref.loc18_15, %Self.as_type.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param1
// CHECK:STDOUT:     %.loc18_28.1: type = splice_block %.loc18_28.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_28: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_28: type = facet_access_type %Self.ref.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_28.2: type = converted %Self.ref.loc18_28, %Self.as_type.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.121]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T.loc21_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc21_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Op.decl), @impl(%T.loc21_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc21_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%T.loc21_14.2 (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%T (%T) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %other.patt: @Op.2.%T (%T) = binding_pattern other
// CHECK:STDOUT:       %other.param_patt: @Op.2.%T (%T) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:       %return.patt: @Op.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Op.2.%T (%T) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc22_37: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc22_15: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc22_28: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %other: @Op.2.%T (%T) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%T (%T) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc9_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc10_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc13_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419)]() -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:   %Self.as_type.loc18_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)](%other.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)) -> @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T)](%other.param_patt: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc9_14.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc9_14.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc10_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc13_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc13_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc13_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.e44) {
// CHECK:STDOUT:   %Self => constants.%Self.e44
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%Self.as_type.560
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc21_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
