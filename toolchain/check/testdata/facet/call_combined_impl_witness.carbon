// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/call_combined_impl_witness.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/call_combined_impl_witness.carbon

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
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.6e7: %A.assoc_type = assoc_entity element0, @A.%AA.decl [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.783: %B.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BB.type.64d: type = fn_type @BB.1 [concrete]
// CHECK:STDOUT:   %BB.11b: %BB.type.64d = struct_value () [concrete]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type @B [concrete]
// CHECK:STDOUT:   %assoc0.a29: %B.assoc_type = assoc_entity element0, @B.%BB.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness file.%Empty.impl_witness_table [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %AA.type.c29: type = fn_type @AA.2 [concrete]
// CHECK:STDOUT:   %AA.95d: %AA.type.c29 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.66c: %A.type = facet_value %C, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness file.%B.impl_witness_table [concrete]
// CHECK:STDOUT:   %BB.type.24d: type = fn_type @BB.2 [concrete]
// CHECK:STDOUT:   %BB.fe8: %BB.type.24d = struct_value () [concrete]
// CHECK:STDOUT:   %B.facet.82f: %B.type = facet_value %C, (%B.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness imports.%BitAnd.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%BitAnd.impl_witness) [concrete]
// CHECK:STDOUT:   %.6bf: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.type.c1b: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.0a6: %Op.type.c1b = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound.56f: <bound method> = bound_method %A.type, %Op.0a6 [concrete]
// CHECK:STDOUT:   %facet_type.d5f: type = facet_type <@Empty & @A> [concrete]
// CHECK:STDOUT:   %Op.bound.e1a: <bound method> = bound_method %facet_type.d5f, %Op.0a6 [concrete]
// CHECK:STDOUT:   %facet_type.242: type = facet_type <@Empty & @A & @B> [concrete]
// CHECK:STDOUT:   %T: %facet_type.242 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.8e7: type = pattern_type %facet_type.242 [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.9a0: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T, @A [symbolic]
// CHECK:STDOUT:   %A.facet.487: %A.type = facet_value %T.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.fde: type = fn_type_with_self_type %AA.type.b97, %A.facet.487 [symbolic]
// CHECK:STDOUT:   %impl.elem0.de2: %.fde = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.086: <specific function> = specific_impl_function %impl.elem0.de2, @AA.1(%A.facet.487) [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T, @B [symbolic]
// CHECK:STDOUT:   %B.facet.8d1: %B.type = facet_value %T.as_type, (%B.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.368: type = fn_type_with_self_type %BB.type.64d, %B.facet.8d1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.43b: %.368 = impl_witness_access %B.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.573: <specific function> = specific_impl_function %impl.elem0.43b, @BB.1(%B.facet.8d1) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.242 = facet_value %C, (%Empty.impl_witness, %A.impl_witness, %B.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%facet_value) [concrete]
// CHECK:STDOUT:   %.7ab: type = fn_type_with_self_type %AA.type.b97, %A.facet.66c [concrete]
// CHECK:STDOUT:   %.b43: type = fn_type_with_self_type %BB.type.64d, %B.facet.82f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAnd: type = import_ref Core//prelude, BitAnd, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.d90: %Op.type.c1b = import_ref Core//prelude, loc13_42, loaded [concrete = constants.%Op.0a6]
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (%Core.import_ref.d90), @impl.13c [concrete]
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
// CHECK:STDOUT:   %Empty.impl_witness_table = impl_witness_table (), @impl.d78 [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness %Empty.impl_witness_table [concrete = constants.%Empty.impl_witness]
// CHECK:STDOUT:   impl_decl @impl.77d [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@impl.77d.%AA.decl), @impl.77d [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   impl_decl @impl.c94 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.impl_witness_table = impl_witness_table (@impl.c94.%BB.decl), @impl.c94 [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness %B.impl_witness_table [concrete = constants.%B.impl_witness]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.8e7 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @G.%pattern_type (%pattern_type.9a0) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @G.%pattern_type (%pattern_type.9a0) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc33_20.1: type = splice_block %.loc33_20.3 [concrete = constants.%facet_type.242] {
// CHECK:STDOUT:       %A.ref.loc33: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:       %impl.elem0.loc33_12: %.6bf = impl_witness_access constants.%BitAnd.impl_witness, element0 [concrete = constants.%Op.0a6]
// CHECK:STDOUT:       %bound_method.loc33_12: <bound method> = bound_method %A.ref.loc33, %impl.elem0.loc33_12 [concrete = constants.%Op.bound.56f]
// CHECK:STDOUT:       %type.and.loc33_12: init type = call %bound_method.loc33_12(%A.ref.loc33, %Empty.ref) [concrete = constants.%facet_type.d5f]
// CHECK:STDOUT:       %B.ref.loc33: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc33_20: %.6bf = impl_witness_access constants.%BitAnd.impl_witness, element0 [concrete = constants.%Op.0a6]
// CHECK:STDOUT:       %bound_method.loc33_20: <bound method> = bound_method %type.and.loc33_12, %impl.elem0.loc33_20 [concrete = constants.%Op.bound.e1a]
// CHECK:STDOUT:       %.loc33_12.1: type = value_of_initializer %type.and.loc33_12 [concrete = constants.%facet_type.d5f]
// CHECK:STDOUT:       %.loc33_12.2: type = converted %type.and.loc33_12, %.loc33_12.1 [concrete = constants.%facet_type.d5f]
// CHECK:STDOUT:       %type.and.loc33_20: init type = call %bound_method.loc33_20(%.loc33_12.2, %B.ref.loc33) [concrete = constants.%facet_type.242]
// CHECK:STDOUT:       %.loc33_20.2: type = value_of_initializer %type.and.loc33_20 [concrete = constants.%facet_type.242]
// CHECK:STDOUT:       %.loc33_20.3: type = converted %type.and.loc33_20, %.loc33_20.2 [concrete = constants.%facet_type.242]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc33_6.1: %facet_type.242 = bind_symbolic_name T, 0 [symbolic = %T.loc33_6.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @G.%T.as_type.loc33_28.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc33_28.1: type = splice_block %.loc33_28.2 [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc33: %facet_type.242 = name_ref T, %T.loc33_6.1 [symbolic = %T.loc33_6.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc33_28.1: type = facet_access_type %T.ref.loc33 [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc33_28.2: type = converted %T.ref.loc33, %T.as_type.loc33_28.1 [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.as_type.loc33_28.2 (%T.as_type) = bind_name t, %t.param
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
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %AA.decl [concrete = constants.%assoc0.6e7]
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
// CHECK:STDOUT:   %assoc0: %B.assoc_type = assoc_entity element0, %BB.decl [concrete = constants.%assoc0.a29]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .BB = %assoc0
// CHECK:STDOUT:   .AA = <poisoned>
// CHECK:STDOUT:   witness = (%BB.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.d78: %C.ref as %Empty.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Empty.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.77d: %C.ref as %A.ref {
// CHECK:STDOUT:   %AA.decl: %AA.type.c29 = fn_decl @AA.2 [concrete = constants.%AA.95d] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .AA = %AA.decl
// CHECK:STDOUT:   witness = file.%A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c94: %C.ref as %B.ref {
// CHECK:STDOUT:   %BB.decl: %BB.type.24d = fn_decl @BB.2 [concrete = constants.%BB.fe8] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .BB = %BB.decl
// CHECK:STDOUT:   witness = file.%B.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT: generic fn @G(%T.loc33_6.1: %facet_type.242) {
// CHECK:STDOUT:   %T.loc33_6.2: %facet_type.242 = bind_symbolic_name T, 0 [symbolic = %T.loc33_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc33_28.2: type = facet_access_type %T.loc33_6.2 [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc33_28.2 [symbolic = %pattern_type (constants.%pattern_type.9a0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc33_28.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc33_6.2, @A [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet.loc34: %A.type = facet_value %T.as_type.loc33_28.2, (%A.lookup_impl_witness) [symbolic = %A.facet.loc34 (constants.%A.facet.487)]
// CHECK:STDOUT:   %.loc34_4.2: type = fn_type_with_self_type constants.%AA.type.b97, %A.facet.loc34 [symbolic = %.loc34_4.2 (constants.%.fde)]
// CHECK:STDOUT:   %impl.elem0.loc34_4.2: @G.%.loc34_4.2 (%.fde) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.de2)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem0.loc34_4.2, @AA.1(%A.facet.loc34) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.086)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc33_6.2, @B [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness)]
// CHECK:STDOUT:   %B.facet.loc35: %B.type = facet_value %T.as_type.loc33_28.2, (%B.lookup_impl_witness) [symbolic = %B.facet.loc35 (constants.%B.facet.8d1)]
// CHECK:STDOUT:   %.loc35_4.2: type = fn_type_with_self_type constants.%BB.type.64d, %B.facet.loc35 [symbolic = %.loc35_4.2 (constants.%.368)]
// CHECK:STDOUT:   %impl.elem0.loc35_4.2: @G.%.loc35_4.2 (%.368) = impl_witness_access %B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.43b)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2: <specific function> = specific_impl_function %impl.elem0.loc35_4.2, @BB.1(%B.facet.loc35) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.573)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @G.%T.as_type.loc33_28.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref.loc34: @G.%T.as_type.loc33_28.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %AA.ref.loc34: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.6e7]
// CHECK:STDOUT:     %T.as_type.loc34: type = facet_access_type constants.%T [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc34_4.1: type = converted constants.%T, %T.as_type.loc34 [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc34_4.1: @G.%.loc34_4.2 (%.fde) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.de2)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem0.loc34_4.1, @AA.1(constants.%A.facet.487) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.086)]
// CHECK:STDOUT:     %.loc34_8: init %empty_tuple.type = call %specific_impl_fn.loc34_4.1()
// CHECK:STDOUT:     %t.ref.loc35: @G.%T.as_type.loc33_28.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %BB.ref.loc35: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.a29]
// CHECK:STDOUT:     %T.as_type.loc35: type = facet_access_type constants.%T [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc35_4.1: type = converted constants.%T, %T.as_type.loc35 [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc35_4.1: @G.%.loc35_4.2 (%.368) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.43b)]
// CHECK:STDOUT:     %specific_impl_fn.loc35_4.1: <specific function> = specific_impl_function %impl.elem0.loc35_4.1, @BB.1(constants.%B.facet.8d1) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.573)]
// CHECK:STDOUT:     %.loc35_8: init %empty_tuple.type = call %specific_impl_fn.loc35_4.1()
// CHECK:STDOUT:     %T.ref.loc37: %facet_type.242 = name_ref T, %T.loc33_6.1 [symbolic = %T.loc33_6.2 (constants.%T)]
// CHECK:STDOUT:     %AA.ref.loc37: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.6e7]
// CHECK:STDOUT:     %T.as_type.loc37: type = facet_access_type %T.ref.loc37 [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc37_4: type = converted %T.ref.loc37, %T.as_type.loc37 [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc37: @G.%.loc34_4.2 (%.fde) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.de2)]
// CHECK:STDOUT:     %specific_impl_fn.loc37: <specific function> = specific_impl_function %impl.elem0.loc37, @AA.1(constants.%A.facet.487) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.086)]
// CHECK:STDOUT:     %.loc37_8: init %empty_tuple.type = call %specific_impl_fn.loc37()
// CHECK:STDOUT:     %T.ref.loc38: %facet_type.242 = name_ref T, %T.loc33_6.1 [symbolic = %T.loc33_6.2 (constants.%T)]
// CHECK:STDOUT:     %BB.ref.loc38: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.a29]
// CHECK:STDOUT:     %T.as_type.loc38: type = facet_access_type %T.ref.loc38 [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc38_4: type = converted %T.ref.loc38, %T.as_type.loc38 [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc38: @G.%.loc35_4.2 (%.368) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.43b)]
// CHECK:STDOUT:     %specific_impl_fn.loc38: <specific function> = specific_impl_function %impl.elem0.loc38, @BB.1(constants.%B.facet.8d1) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.573)]
// CHECK:STDOUT:     %.loc38_8: init %empty_tuple.type = call %specific_impl_fn.loc38()
// CHECK:STDOUT:     %T.ref.loc40: %facet_type.242 = name_ref T, %T.loc33_6.1 [symbolic = %T.loc33_6.2 (constants.%T)]
// CHECK:STDOUT:     %A.ref.loc40: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %AA.ref.loc40: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.6e7]
// CHECK:STDOUT:     %T.as_type.loc40: type = facet_access_type constants.%T [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %A.facet.loc40: %A.type = facet_value %T.as_type.loc40, (constants.%A.lookup_impl_witness) [symbolic = %A.facet.loc34 (constants.%A.facet.487)]
// CHECK:STDOUT:     %.loc40_4: %A.type = converted %T.ref.loc40, %A.facet.loc40 [symbolic = %A.facet.loc34 (constants.%A.facet.487)]
// CHECK:STDOUT:     %impl.elem0.loc40: @G.%.loc34_4.2 (%.fde) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.de2)]
// CHECK:STDOUT:     %specific_impl_fn.loc40: <specific function> = specific_impl_function %impl.elem0.loc40, @AA.1(constants.%A.facet.487) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.086)]
// CHECK:STDOUT:     %.loc40_12: init %empty_tuple.type = call %specific_impl_fn.loc40()
// CHECK:STDOUT:     %T.ref.loc41: %facet_type.242 = name_ref T, %T.loc33_6.1 [symbolic = %T.loc33_6.2 (constants.%T)]
// CHECK:STDOUT:     %B.ref.loc41: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %BB.ref.loc41: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.a29]
// CHECK:STDOUT:     %T.as_type.loc41: type = facet_access_type constants.%T [symbolic = %T.as_type.loc33_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %B.facet.loc41: %B.type = facet_value %T.as_type.loc41, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc35 (constants.%B.facet.8d1)]
// CHECK:STDOUT:     %.loc41_4: %B.type = converted %T.ref.loc41, %B.facet.loc41 [symbolic = %B.facet.loc35 (constants.%B.facet.8d1)]
// CHECK:STDOUT:     %impl.elem0.loc41: @G.%.loc35_4.2 (%.368) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.43b)]
// CHECK:STDOUT:     %specific_impl_fn.loc41: <specific function> = specific_impl_function %impl.elem0.loc41, @BB.1(constants.%B.facet.8d1) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.573)]
// CHECK:STDOUT:     %.loc41_12: init %empty_tuple.type = call %specific_impl_fn.loc41()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc45_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc45_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc45_6.3: init %C = class_init (), %.loc45_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc45_6.4: ref %C = temporary %.loc45_6.2, %.loc45_6.3
// CHECK:STDOUT:   %.loc45_8.1: ref %C = converted %.loc45_6.1, %.loc45_6.4
// CHECK:STDOUT:   %facet_value.loc45_12.1: %facet_type.242 = facet_value constants.%C, (constants.%Empty.impl_witness, constants.%A.impl_witness, constants.%B.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc45_12.1: %facet_type.242 = converted constants.%C, %facet_value.loc45_12.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc45_12.2: %facet_type.242 = facet_value constants.%C, (constants.%Empty.impl_witness, constants.%A.impl_witness, constants.%B.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc45_12.2: %facet_type.242 = converted constants.%C, %facet_value.loc45_12.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%facet_value) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %.loc45_8.2: %C = bind_value %.loc45_8.1
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn(%.loc45_8.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AA.1(constants.%Self.31d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.1(constants.%Self.783) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AA.1(constants.%A.facet.66c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.1(constants.%B.facet.82f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc33_6.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc33_28.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AA.1(constants.%A.facet.487) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.1(constants.%B.facet.8d1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc33_6.2 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc33_28.2 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet.loc34 => constants.%A.facet.66c
// CHECK:STDOUT:   %.loc34_4.2 => constants.%.7ab
// CHECK:STDOUT:   %impl.elem0.loc34_4.2 => constants.%AA.95d
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%AA.95d
// CHECK:STDOUT:   %B.lookup_impl_witness => constants.%B.impl_witness
// CHECK:STDOUT:   %B.facet.loc35 => constants.%B.facet.82f
// CHECK:STDOUT:   %.loc35_4.2 => constants.%.b43
// CHECK:STDOUT:   %impl.elem0.loc35_4.2 => constants.%BB.fe8
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%BB.fe8
// CHECK:STDOUT: }
// CHECK:STDOUT:
