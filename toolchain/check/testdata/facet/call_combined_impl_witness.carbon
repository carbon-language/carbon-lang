// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
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
// CHECK:STDOUT:   %Self.63a: %Empty.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c6e: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.AA.type.42c: type = fn_type @A.WithSelf.AA, @A.WithSelf(%Self.c6e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.WithSelf.AA.1b8: %A.WithSelf.AA.type.42c = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.0c8: %A.assoc_type = assoc_entity element0, @A.WithSelf.%A.WithSelf.AA.decl [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.775: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.type.490: type = fn_type @B.WithSelf.BB, @B.WithSelf(%Self.775) [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.336: %B.WithSelf.BB.type.490 = struct_value () [symbolic]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type @B [concrete]
// CHECK:STDOUT:   %assoc0.a5f: %B.assoc_type = assoc_entity element0, @B.WithSelf.%B.WithSelf.BB.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness @C.as.Empty.impl.%Empty.impl_witness_table [concrete]
// CHECK:STDOUT:   %Empty.facet: %Empty.type = facet_value %C, (%Empty.impl_witness) [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @C.as.A.impl.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.A.impl.AA.type: type = fn_type @C.as.A.impl.AA [concrete]
// CHECK:STDOUT:   %C.as.A.impl.AA: %C.as.A.impl.AA.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.a9c: %A.type = facet_value %C, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %A.WithSelf.AA.type.db5: type = fn_type @A.WithSelf.AA, @A.WithSelf(%A.facet.a9c) [concrete]
// CHECK:STDOUT:   %A.WithSelf.AA.9cf: %A.WithSelf.AA.type.db5 = struct_value () [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness @C.as.B.impl.%B.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.B.impl.BB.type: type = fn_type @C.as.B.impl.BB [concrete]
// CHECK:STDOUT:   %C.as.B.impl.BB: %C.as.B.impl.BB.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.facet.c28: %B.type = facet_value %C, (%B.impl_witness) [concrete]
// CHECK:STDOUT:   %B.WithSelf.BB.type.b44: type = fn_type @B.WithSelf.BB, @B.WithSelf(%B.facet.c28) [concrete]
// CHECK:STDOUT:   %B.WithSelf.BB.e64: %B.WithSelf.BB.type.b44 = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitAndWith.type.1ec: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.1ec = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.89e: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.89e = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.7b3: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.6c6: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.7b3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.dc4: <bound method> = bound_method %A.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.0bb: type = facet_type <@Empty & @A> [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.3c5: <bound method> = bound_method %facet_type.0bb, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.d65: type = facet_type <@Empty & @A & @B> [concrete]
// CHECK:STDOUT:   %pattern_type.0f2: type = pattern_type %facet_type.d65 [concrete]
// CHECK:STDOUT:   %T: %facet_type.d65 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.289: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %A.WithSelf.AA.type.127: type = fn_type @A.WithSelf.AA, @A.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.AA.17e: %A.WithSelf.AA.type.127 = struct_value () [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.type.d61: type = fn_type @B.WithSelf.BB, @B.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.13b: %B.WithSelf.BB.type.d61 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T, @A [symbolic]
// CHECK:STDOUT:   %A.facet.7ae: %A.type = facet_value %T.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.AA.type.719: type = fn_type @A.WithSelf.AA, @A.WithSelf(%A.facet.7ae) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.AA.9f3: %A.WithSelf.AA.type.719 = struct_value () [symbolic]
// CHECK:STDOUT:   %.d8d: type = fn_type_with_self_type %A.WithSelf.AA.type.719, %A.facet.7ae [symbolic]
// CHECK:STDOUT:   %impl.elem0.c57: %.d8d = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.19c: <specific function> = specific_impl_function %impl.elem0.c57, @A.WithSelf.AA(%A.facet.7ae) [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T, @B [symbolic]
// CHECK:STDOUT:   %B.facet.8d8: %B.type = facet_value %T.as_type, (%B.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.type.79f: type = fn_type @B.WithSelf.BB, @B.WithSelf(%B.facet.8d8) [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.bab: %B.WithSelf.BB.type.79f = struct_value () [symbolic]
// CHECK:STDOUT:   %.1d5: type = fn_type_with_self_type %B.WithSelf.BB.type.79f, %B.facet.8d8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.31a: %.1d5 = impl_witness_access %B.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.74b: <specific function> = specific_impl_function %impl.elem0.31a, @B.WithSelf.BB(%B.facet.8d8) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.d65 = facet_value %C, (%Empty.impl_witness, %A.impl_witness, %B.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.d7b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%facet_value) [concrete]
// CHECK:STDOUT:   %.b8f: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc45_8.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.b8f, %Destroy.Op.1dc86d.2 [concrete]
// CHECK:STDOUT:   %.415: type = fn_type_with_self_type %A.WithSelf.AA.type.db5, %A.facet.a9c [concrete]
// CHECK:STDOUT:   %.336: type = fn_type_with_self_type %B.WithSelf.BB.type.b44, %B.facet.c28 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.1ec = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.35b: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.35b), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:   impl_decl @C.as.Empty.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.B.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.0f2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.param_patt: @G.%pattern_type (%pattern_type.289) = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: @G.%pattern_type (%pattern_type.289) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc33_20.1: type = splice_block %.loc33_20.3 [concrete = constants.%facet_type.d65] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref.loc33: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:       %impl.elem0.loc33_12: %.6c6 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc33_12: <bound method> = bound_method %A.ref.loc33, %impl.elem0.loc33_12 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.dc4]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc33_12: init type = call %bound_method.loc33_12(%A.ref.loc33, %Empty.ref) [concrete = constants.%facet_type.0bb]
// CHECK:STDOUT:       %B.ref.loc33: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc33_20: %.6c6 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc33_20: <bound method> = bound_method %type.as.BitAndWith.impl.Op.call.loc33_12, %impl.elem0.loc33_20 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.3c5]
// CHECK:STDOUT:       %.loc33_12.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc33_12 [concrete = constants.%facet_type.0bb]
// CHECK:STDOUT:       %.loc33_12.2: type = converted %type.as.BitAndWith.impl.Op.call.loc33_12, %.loc33_12.1 [concrete = constants.%facet_type.0bb]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc33_20: init type = call %bound_method.loc33_20(%.loc33_12.2, %B.ref.loc33) [concrete = constants.%facet_type.d65]
// CHECK:STDOUT:       %.loc33_20.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc33_20 [concrete = constants.%facet_type.d65]
// CHECK:STDOUT:       %.loc33_20.3: type = converted %type.as.BitAndWith.impl.Op.call.loc33_20, %.loc33_20.2 [concrete = constants.%facet_type.d65]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc33_7.2: %facet_type.d65 = symbolic_binding T, 0 [symbolic = %T.loc33_7.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @G.%T.as_type.loc33_28.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc33_28.1: type = splice_block %.loc33_28.2 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc33: %facet_type.d65 = name_ref T, %T.loc33_7.2 [symbolic = %T.loc33_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc33_28.2: type = facet_access_type %T.ref.loc33 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc33_28.2: type = converted %T.ref.loc33, %T.as_type.loc33_28.2 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.as_type.loc33_28.1 (%T.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = symbolic_binding Self, 0 [symbolic = constants.%Self.63a]
// CHECK:STDOUT:   %Empty.WithSelf.decl = interface_with_self_decl @Empty [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .AA = <poisoned>
// CHECK:STDOUT:   .BB = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c6e]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %A.WithSelf.AA.decl: @A.WithSelf.%A.WithSelf.AA.type (%A.WithSelf.AA.type.42c) = fn_decl @A.WithSelf.AA [symbolic = @A.WithSelf.%A.WithSelf.AA (constants.%A.WithSelf.AA.1b8)] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.WithSelf.AA.decl [concrete = constants.%assoc0.0c8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .AA = @A.WithSelf.%assoc0
// CHECK:STDOUT:   .BB = <poisoned>
// CHECK:STDOUT:   witness = (@A.WithSelf.%A.WithSelf.AA.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.775]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %B.WithSelf.BB.decl: @B.WithSelf.%B.WithSelf.BB.type (%B.WithSelf.BB.type.490) = fn_decl @B.WithSelf.BB [symbolic = @B.WithSelf.%B.WithSelf.BB (constants.%B.WithSelf.BB.336)] {} {}
// CHECK:STDOUT:   %assoc0: %B.assoc_type = assoc_entity element0, %B.WithSelf.BB.decl [concrete = constants.%assoc0.a5f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .BB = @B.WithSelf.%assoc0
// CHECK:STDOUT:   .AA = <poisoned>
// CHECK:STDOUT:   witness = (@B.WithSelf.%B.WithSelf.BB.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Empty.impl: %C.ref as %Empty.ref {
// CHECK:STDOUT:   %Empty.impl_witness_table = impl_witness_table (), @C.as.Empty.impl [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness %Empty.impl_witness_table [concrete = constants.%Empty.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Empty.ref
// CHECK:STDOUT:   witness = %Empty.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.A.impl: %C.ref as %A.ref {
// CHECK:STDOUT:   %C.as.A.impl.AA.decl: %C.as.A.impl.AA.type = fn_decl @C.as.A.impl.AA [concrete = constants.%C.as.A.impl.AA] {} {}
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (%C.as.A.impl.AA.decl), @C.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .AA = %C.as.A.impl.AA.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT:   witness = %A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.B.impl: %C.ref as %B.ref {
// CHECK:STDOUT:   %C.as.B.impl.BB.decl: %C.as.B.impl.BB.type = fn_decl @C.as.B.impl.BB [concrete = constants.%C.as.B.impl.BB] {} {}
// CHECK:STDOUT:   %B.impl_witness_table = impl_witness_table (%C.as.B.impl.BB.decl), @C.as.B.impl [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness %B.impl_witness_table [concrete = constants.%B.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .BB = %C.as.B.impl.BB.decl
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT:   witness = %B.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.WithSelf.AA(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B.WithSelf.BB(@B.%Self: %B.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.A.impl.AA() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.B.impl.BB() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc33_7.2: %facet_type.d65) {
// CHECK:STDOUT:   %T.loc33_7.1: %facet_type.d65 = symbolic_binding T, 0 [symbolic = %T.loc33_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc33_28.1: type = facet_access_type %T.loc33_7.1 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc33_28.1 [symbolic = %pattern_type (constants.%pattern_type.289)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc33_28.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc33_7.1, @A [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet.loc34: %A.type = facet_value %T.as_type.loc33_28.1, (%A.lookup_impl_witness) [symbolic = %A.facet.loc34 (constants.%A.facet.7ae)]
// CHECK:STDOUT:   %A.WithSelf.AA.type: type = fn_type @A.WithSelf.AA, @A.WithSelf(%A.facet.loc34) [symbolic = %A.WithSelf.AA.type (constants.%A.WithSelf.AA.type.719)]
// CHECK:STDOUT:   %.loc34: type = fn_type_with_self_type %A.WithSelf.AA.type, %A.facet.loc34 [symbolic = %.loc34 (constants.%.d8d)]
// CHECK:STDOUT:   %impl.elem0.loc34_4.2: @G.%.loc34 (%.d8d) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.c57)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem0.loc34_4.2, @A.WithSelf.AA(%A.facet.loc34) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.19c)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc33_7.1, @B [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness)]
// CHECK:STDOUT:   %B.facet.loc35: %B.type = facet_value %T.as_type.loc33_28.1, (%B.lookup_impl_witness) [symbolic = %B.facet.loc35 (constants.%B.facet.8d8)]
// CHECK:STDOUT:   %B.WithSelf.BB.type: type = fn_type @B.WithSelf.BB, @B.WithSelf(%B.facet.loc35) [symbolic = %B.WithSelf.BB.type (constants.%B.WithSelf.BB.type.79f)]
// CHECK:STDOUT:   %.loc35: type = fn_type_with_self_type %B.WithSelf.BB.type, %B.facet.loc35 [symbolic = %.loc35 (constants.%.1d5)]
// CHECK:STDOUT:   %impl.elem0.loc35_4.2: @G.%.loc35 (%.1d5) = impl_witness_access %B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.31a)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2: <specific function> = specific_impl_function %impl.elem0.loc35_4.2, @B.WithSelf.BB(%B.facet.loc35) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.74b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @G.%T.as_type.loc33_28.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref.loc34: @G.%T.as_type.loc33_28.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %AA.ref.loc34: %A.assoc_type = name_ref AA, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.0c8]
// CHECK:STDOUT:     %impl.elem0.loc34_4.1: @G.%.loc34 (%.d8d) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.c57)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem0.loc34_4.1, @A.WithSelf.AA(constants.%A.facet.7ae) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.19c)]
// CHECK:STDOUT:     %A.WithSelf.AA.call.loc34: init %empty_tuple.type = call %specific_impl_fn.loc34_4.1()
// CHECK:STDOUT:     %t.ref.loc35: @G.%T.as_type.loc33_28.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %BB.ref.loc35: %B.assoc_type = name_ref BB, @B.WithSelf.%assoc0 [concrete = constants.%assoc0.a5f]
// CHECK:STDOUT:     %impl.elem0.loc35_4.1: @G.%.loc35 (%.1d5) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.31a)]
// CHECK:STDOUT:     %specific_impl_fn.loc35_4.1: <specific function> = specific_impl_function %impl.elem0.loc35_4.1, @B.WithSelf.BB(constants.%B.facet.8d8) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.74b)]
// CHECK:STDOUT:     %B.WithSelf.BB.call.loc35: init %empty_tuple.type = call %specific_impl_fn.loc35_4.1()
// CHECK:STDOUT:     %T.ref.loc37: %facet_type.d65 = name_ref T, %T.loc33_7.2 [symbolic = %T.loc33_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc37: type = facet_access_type %T.ref.loc37 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc37: type = converted %T.ref.loc37, %T.as_type.loc37 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %AA.ref.loc37: %A.assoc_type = name_ref AA, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.0c8]
// CHECK:STDOUT:     %impl.elem0.loc37: @G.%.loc34 (%.d8d) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.c57)]
// CHECK:STDOUT:     %specific_impl_fn.loc37: <specific function> = specific_impl_function %impl.elem0.loc37, @A.WithSelf.AA(constants.%A.facet.7ae) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.19c)]
// CHECK:STDOUT:     %A.WithSelf.AA.call.loc37: init %empty_tuple.type = call %specific_impl_fn.loc37()
// CHECK:STDOUT:     %T.ref.loc38: %facet_type.d65 = name_ref T, %T.loc33_7.2 [symbolic = %T.loc33_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc38: type = facet_access_type %T.ref.loc38 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc38: type = converted %T.ref.loc38, %T.as_type.loc38 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %BB.ref.loc38: %B.assoc_type = name_ref BB, @B.WithSelf.%assoc0 [concrete = constants.%assoc0.a5f]
// CHECK:STDOUT:     %impl.elem0.loc38: @G.%.loc35 (%.1d5) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.31a)]
// CHECK:STDOUT:     %specific_impl_fn.loc38: <specific function> = specific_impl_function %impl.elem0.loc38, @B.WithSelf.BB(constants.%B.facet.8d8) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.74b)]
// CHECK:STDOUT:     %B.WithSelf.BB.call.loc38: init %empty_tuple.type = call %specific_impl_fn.loc38()
// CHECK:STDOUT:     %T.ref.loc40: %facet_type.d65 = name_ref T, %T.loc33_7.2 [symbolic = %T.loc33_7.1 (constants.%T)]
// CHECK:STDOUT:     %A.ref.loc40: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %AA.ref.loc40: %A.assoc_type = name_ref AA, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.0c8]
// CHECK:STDOUT:     %T.as_type.loc40: type = facet_access_type %T.ref.loc40 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %A.facet.loc40: %A.type = facet_value %T.as_type.loc40, (constants.%A.lookup_impl_witness) [symbolic = %A.facet.loc34 (constants.%A.facet.7ae)]
// CHECK:STDOUT:     %.loc40: %A.type = converted %T.ref.loc40, %A.facet.loc40 [symbolic = %A.facet.loc34 (constants.%A.facet.7ae)]
// CHECK:STDOUT:     %impl.elem0.loc40: @G.%.loc34 (%.d8d) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.c57)]
// CHECK:STDOUT:     %specific_impl_fn.loc40: <specific function> = specific_impl_function %impl.elem0.loc40, @A.WithSelf.AA(constants.%A.facet.7ae) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.19c)]
// CHECK:STDOUT:     %A.WithSelf.AA.call.loc40: init %empty_tuple.type = call %specific_impl_fn.loc40()
// CHECK:STDOUT:     %T.ref.loc41: %facet_type.d65 = name_ref T, %T.loc33_7.2 [symbolic = %T.loc33_7.1 (constants.%T)]
// CHECK:STDOUT:     %B.ref.loc41: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %BB.ref.loc41: %B.assoc_type = name_ref BB, @B.WithSelf.%assoc0 [concrete = constants.%assoc0.a5f]
// CHECK:STDOUT:     %T.as_type.loc41: type = facet_access_type %T.ref.loc41 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %B.facet.loc41: %B.type = facet_value %T.as_type.loc41, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc35 (constants.%B.facet.8d8)]
// CHECK:STDOUT:     %.loc41: %B.type = converted %T.ref.loc41, %B.facet.loc41 [symbolic = %B.facet.loc35 (constants.%B.facet.8d8)]
// CHECK:STDOUT:     %impl.elem0.loc41: @G.%.loc35 (%.1d5) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.31a)]
// CHECK:STDOUT:     %specific_impl_fn.loc41: <specific function> = specific_impl_function %impl.elem0.loc41, @B.WithSelf.BB(constants.%B.facet.8d8) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.74b)]
// CHECK:STDOUT:     %B.WithSelf.BB.call.loc41: init %empty_tuple.type = call %specific_impl_fn.loc41()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc45_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc45_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc45_6.3: init %C to %.loc45_6.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc45_8.1: init %C = converted %.loc45_6.1, %.loc45_6.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %facet_value: %facet_type.d65 = facet_value constants.%C, (constants.%Empty.impl_witness, constants.%A.impl_witness, constants.%B.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc45_12: %facet_type.d65 = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%facet_value) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %.loc45_8.2: ref %C = temporary %.loc45_6.2, %.loc45_8.1 [concrete = constants.%.b8f]
// CHECK:STDOUT:   %.loc45_8.3: %C = acquire_value %.loc45_8.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn(%.loc45_8.3)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.b8f)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc45_8.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc45_8.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%Self.63a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c6e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.c6e
// CHECK:STDOUT:   %A.WithSelf.AA.type => constants.%A.WithSelf.AA.type.42c
// CHECK:STDOUT:   %A.WithSelf.AA => constants.%A.WithSelf.AA.1b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.AA(constants.%Self.c6e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.775) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.775
// CHECK:STDOUT:   %B.WithSelf.BB.type => constants.%B.WithSelf.BB.type.490
// CHECK:STDOUT:   %B.WithSelf.BB => constants.%B.WithSelf.BB.336
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf.BB(constants.%Self.775) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%Empty.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%A.facet.a9c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%A.facet.a9c
// CHECK:STDOUT:   %A.WithSelf.AA.type => constants.%A.WithSelf.AA.type.db5
// CHECK:STDOUT:   %A.WithSelf.AA => constants.%A.WithSelf.AA.9cf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.AA(constants.%A.facet.a9c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%B.facet.c28) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%B.facet.c28
// CHECK:STDOUT:   %B.WithSelf.BB.type => constants.%B.WithSelf.BB.type.b44
// CHECK:STDOUT:   %B.WithSelf.BB => constants.%B.WithSelf.BB.e64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf.BB(constants.%B.facet.c28) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc33_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc33_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.289
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %A.WithSelf.AA.type => constants.%A.WithSelf.AA.type.127
// CHECK:STDOUT:   %A.WithSelf.AA => constants.%A.WithSelf.AA.17e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %B.WithSelf.BB.type => constants.%B.WithSelf.BB.type.d61
// CHECK:STDOUT:   %B.WithSelf.BB => constants.%B.WithSelf.BB.13b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%A.facet.7ae) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%A.facet.7ae
// CHECK:STDOUT:   %A.WithSelf.AA.type => constants.%A.WithSelf.AA.type.719
// CHECK:STDOUT:   %A.WithSelf.AA => constants.%A.WithSelf.AA.9f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.AA(constants.%A.facet.7ae) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%B.facet.8d8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%B.facet.8d8
// CHECK:STDOUT:   %B.WithSelf.BB.type => constants.%B.WithSelf.BB.type.79f
// CHECK:STDOUT:   %B.WithSelf.BB => constants.%B.WithSelf.BB.bab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf.BB(constants.%B.facet.8d8) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc33_7.1 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc33_28.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d7b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet.loc34 => constants.%A.facet.a9c
// CHECK:STDOUT:   %A.WithSelf.AA.type => constants.%A.WithSelf.AA.type.db5
// CHECK:STDOUT:   %.loc34 => constants.%.415
// CHECK:STDOUT:   %impl.elem0.loc34_4.2 => constants.%C.as.A.impl.AA
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%C.as.A.impl.AA
// CHECK:STDOUT:   %B.lookup_impl_witness => constants.%B.impl_witness
// CHECK:STDOUT:   %B.facet.loc35 => constants.%B.facet.c28
// CHECK:STDOUT:   %B.WithSelf.BB.type => constants.%B.WithSelf.BB.type.b44
// CHECK:STDOUT:   %.loc35 => constants.%.336
// CHECK:STDOUT:   %impl.elem0.loc35_4.2 => constants.%C.as.B.impl.BB
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%C.as.B.impl.BB
// CHECK:STDOUT: }
// CHECK:STDOUT:
