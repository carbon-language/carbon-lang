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
// CHECK:STDOUT:   %Self.61e: %Empty.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.AA.type.236: type = fn_type @A.WithSelf.AA, @A.WithSelf(%Self.c51) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.WithSelf.AA.0c0: %A.WithSelf.AA.type.236 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.a62: %A.assoc_type = assoc_entity element0, @A.WithSelf.%A.WithSelf.AA.decl [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.d0b: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.type.17c: type = fn_type @B.WithSelf.BB, @B.WithSelf(%Self.d0b) [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.f00: %B.WithSelf.BB.type.17c = struct_value () [symbolic]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type @B [concrete]
// CHECK:STDOUT:   %assoc0.00a: %B.assoc_type = assoc_entity element0, @B.WithSelf.%B.WithSelf.BB.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness @C.as.Empty.impl.%Empty.impl_witness_table [concrete]
// CHECK:STDOUT:   %Empty.facet: %Empty.type = facet_value %C, (%Empty.impl_witness) [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @C.as.A.impl.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.A.impl.AA.type: type = fn_type @C.as.A.impl.AA [concrete]
// CHECK:STDOUT:   %C.as.A.impl.AA: %C.as.A.impl.AA.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.34f: %A.type = facet_value %C, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %A.WithSelf.AA.type.6ff: type = fn_type @A.WithSelf.AA, @A.WithSelf(%A.facet.34f) [concrete]
// CHECK:STDOUT:   %A.WithSelf.AA.ff5: %A.WithSelf.AA.type.6ff = struct_value () [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness @C.as.B.impl.%B.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.B.impl.BB.type: type = fn_type @C.as.B.impl.BB [concrete]
// CHECK:STDOUT:   %C.as.B.impl.BB: %C.as.B.impl.BB.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.facet.e93: %B.type = facet_value %C, (%B.impl_witness) [concrete]
// CHECK:STDOUT:   %B.WithSelf.BB.type.f5e: type = fn_type @B.WithSelf.BB, @B.WithSelf(%B.facet.e93) [concrete]
// CHECK:STDOUT:   %B.WithSelf.BB.09b: %B.WithSelf.BB.type.f5e = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4bd: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.4bd, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.05b: <bound method> = bound_method %A.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.902: type = facet_type <@Empty & @A> [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.ab2: <bound method> = bound_method %facet_type.902, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.dc7: type = facet_type <@Empty & @A & @B> [concrete]
// CHECK:STDOUT:   %pattern_type.773: type = pattern_type %facet_type.dc7 [concrete]
// CHECK:STDOUT:   %T: %facet_type.dc7 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.2ee: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %A.WithSelf.AA.type.ce6: type = fn_type @A.WithSelf.AA, @A.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.AA.020: %A.WithSelf.AA.type.ce6 = struct_value () [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.type.374: type = fn_type @B.WithSelf.BB, @B.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.1c2: %B.WithSelf.BB.type.374 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T, @A [symbolic]
// CHECK:STDOUT:   %A.facet.095: %A.type = facet_value %T.binding.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.AA.type.7b2: type = fn_type @A.WithSelf.AA, @A.WithSelf(%A.facet.095) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.AA.11e: %A.WithSelf.AA.type.7b2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.84b: type = fn_type_with_self_type %A.WithSelf.AA.type.7b2, %A.facet.095 [symbolic]
// CHECK:STDOUT:   %impl.elem0.26a: %.84b = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.567: <specific function> = specific_impl_function %impl.elem0.26a, @A.WithSelf.AA(%A.facet.095) [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T, @B [symbolic]
// CHECK:STDOUT:   %B.facet.0e7: %B.type = facet_value %T.binding.as_type, (%B.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.type.5a2: type = fn_type @B.WithSelf.BB, @B.WithSelf(%B.facet.0e7) [symbolic]
// CHECK:STDOUT:   %B.WithSelf.BB.b76: %B.WithSelf.BB.type.5a2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.dfd: type = fn_type_with_self_type %B.WithSelf.BB.type.5a2, %B.facet.0e7 [symbolic]
// CHECK:STDOUT:   %impl.elem0.557: %.dfd = impl_witness_access %B.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fc3: <specific function> = specific_impl_function %impl.elem0.557, @B.WithSelf.BB(%B.facet.0e7) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.dc7 = facet_value %C, (%Empty.impl_witness, %A.impl_witness, %B.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%facet_value) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %.171: type = fn_type_with_self_type %A.WithSelf.AA.type.6ff, %A.facet.34f [concrete]
// CHECK:STDOUT:   %.e04: type = fn_type_with_self_type %B.WithSelf.BB.type.f5e, %B.facet.e93 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.8d3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc130, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.8d3), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.773 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.param_patt: @G.%pattern_type (%pattern_type.2ee) = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: @G.%pattern_type (%pattern_type.2ee) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc65.1: type = splice_block %.loc65.3 [concrete = constants.%facet_type.dc7] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref.loc62: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:       %impl.elem0.loc63: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc63: <bound method> = bound_method %A.ref.loc62, %impl.elem0.loc63 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.05b]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc63: init type = call %bound_method.loc63(%A.ref.loc62, %Empty.ref) [concrete = constants.%facet_type.902]
// CHECK:STDOUT:       %B.ref.loc66: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc65: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc65: <bound method> = bound_method %type.as.BitAndWith.impl.Op.call.loc63, %impl.elem0.loc65 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.ab2]
// CHECK:STDOUT:       %.loc63.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc63 [concrete = constants.%facet_type.902]
// CHECK:STDOUT:       %.loc63.2: type = converted %type.as.BitAndWith.impl.Op.call.loc63, %.loc63.1 [concrete = constants.%facet_type.902]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc65: init type = call %bound_method.loc65(%.loc63.2, %B.ref.loc66) [concrete = constants.%facet_type.dc7]
// CHECK:STDOUT:       %.loc65.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc65 [concrete = constants.%facet_type.dc7]
// CHECK:STDOUT:       %.loc65.3: type = converted %type.as.BitAndWith.impl.Op.call.loc65, %.loc65.2 [concrete = constants.%facet_type.dc7]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc60.2: %facet_type.dc7 = symbolic_binding T, 0 [symbolic = %T.loc60.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @G.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc71.1: type = splice_block %.loc71.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc71: %facet_type.dc7 = name_ref T, %T.loc60.2 [symbolic = %T.loc60.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc71: type = facet_access_type %T.ref.loc71 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc71.2: type = converted %T.ref.loc71, %T.as_type.loc71 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = symbolic_binding Self, 0 [symbolic = constants.%Self.61e]
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
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %A.WithSelf.AA.decl: @A.WithSelf.%A.WithSelf.AA.type (%A.WithSelf.AA.type.236) = fn_decl @A.WithSelf.AA [symbolic = @A.WithSelf.%A.WithSelf.AA (constants.%A.WithSelf.AA.0c0)] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.WithSelf.AA.decl [concrete = constants.%assoc0.a62]
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
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0b]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %B.WithSelf.BB.decl: @B.WithSelf.%B.WithSelf.BB.type (%B.WithSelf.BB.type.17c) = fn_decl @B.WithSelf.BB [symbolic = @B.WithSelf.%B.WithSelf.BB (constants.%B.WithSelf.BB.f00)] {} {}
// CHECK:STDOUT:   %assoc0: %B.assoc_type = assoc_entity element0, %B.WithSelf.BB.decl [concrete = constants.%assoc0.00a]
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
// CHECK:STDOUT: generic fn @G(%T.loc60.2: %facet_type.dc7) {
// CHECK:STDOUT:   %T.loc60.1: %facet_type.dc7 = symbolic_binding T, 0 [symbolic = %T.loc60.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc60.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2ee)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc60.1, @A [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet.loc75: %A.type = facet_value %T.binding.as_type, (%A.lookup_impl_witness) [symbolic = %A.facet.loc75 (constants.%A.facet.095)]
// CHECK:STDOUT:   %A.WithSelf.AA.type: type = fn_type @A.WithSelf.AA, @A.WithSelf(%A.facet.loc75) [symbolic = %A.WithSelf.AA.type (constants.%A.WithSelf.AA.type.7b2)]
// CHECK:STDOUT:   %.loc75: type = fn_type_with_self_type %A.WithSelf.AA.type, %A.facet.loc75 [symbolic = %.loc75 (constants.%.84b)]
// CHECK:STDOUT:   %impl.elem0.loc75.2: @G.%.loc75 (%.84b) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc75.2 (constants.%impl.elem0.26a)]
// CHECK:STDOUT:   %specific_impl_fn.loc75.2: <specific function> = specific_impl_function %impl.elem0.loc75.2, @A.WithSelf.AA(%A.facet.loc75) [symbolic = %specific_impl_fn.loc75.2 (constants.%specific_impl_fn.567)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc60.1, @B [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness)]
// CHECK:STDOUT:   %B.facet.loc81: %B.type = facet_value %T.binding.as_type, (%B.lookup_impl_witness) [symbolic = %B.facet.loc81 (constants.%B.facet.0e7)]
// CHECK:STDOUT:   %B.WithSelf.BB.type: type = fn_type @B.WithSelf.BB, @B.WithSelf(%B.facet.loc81) [symbolic = %B.WithSelf.BB.type (constants.%B.WithSelf.BB.type.5a2)]
// CHECK:STDOUT:   %.loc81: type = fn_type_with_self_type %B.WithSelf.BB.type, %B.facet.loc81 [symbolic = %.loc81 (constants.%.dfd)]
// CHECK:STDOUT:   %impl.elem0.loc81.2: @G.%.loc81 (%.dfd) = impl_witness_access %B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc81.2 (constants.%impl.elem0.557)]
// CHECK:STDOUT:   %specific_impl_fn.loc81.2: <specific function> = specific_impl_function %impl.elem0.loc81.2, @B.WithSelf.BB(%B.facet.loc81) [symbolic = %specific_impl_fn.loc81.2 (constants.%specific_impl_fn.fc3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @G.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref.loc74: @G.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %AA.ref.loc75: %A.assoc_type = name_ref AA, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.a62]
// CHECK:STDOUT:     %impl.elem0.loc75.1: @G.%.loc75 (%.84b) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc75.2 (constants.%impl.elem0.26a)]
// CHECK:STDOUT:     %specific_impl_fn.loc75.1: <specific function> = specific_impl_function %impl.elem0.loc75.1, @A.WithSelf.AA(constants.%A.facet.095) [symbolic = %specific_impl_fn.loc75.2 (constants.%specific_impl_fn.567)]
// CHECK:STDOUT:     %A.WithSelf.AA.call.loc78: init %empty_tuple.type = call %specific_impl_fn.loc75.1()
// CHECK:STDOUT:     %t.ref.loc80: @G.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %BB.ref.loc81: %B.assoc_type = name_ref BB, @B.WithSelf.%assoc0 [concrete = constants.%assoc0.00a]
// CHECK:STDOUT:     %impl.elem0.loc81.1: @G.%.loc81 (%.dfd) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc81.2 (constants.%impl.elem0.557)]
// CHECK:STDOUT:     %specific_impl_fn.loc81.1: <specific function> = specific_impl_function %impl.elem0.loc81.1, @B.WithSelf.BB(constants.%B.facet.0e7) [symbolic = %specific_impl_fn.loc81.2 (constants.%specific_impl_fn.fc3)]
// CHECK:STDOUT:     %B.WithSelf.BB.call.loc84: init %empty_tuple.type = call %specific_impl_fn.loc81.1()
// CHECK:STDOUT:     %T.ref.loc86: %facet_type.dc7 = name_ref T, %T.loc60.2 [symbolic = %T.loc60.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc87: type = facet_access_type %T.ref.loc86 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc87: type = converted %T.ref.loc86, %T.as_type.loc87 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %AA.ref.loc87: %A.assoc_type = name_ref AA, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.a62]
// CHECK:STDOUT:     %impl.elem0.loc87: @G.%.loc75 (%.84b) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc75.2 (constants.%impl.elem0.26a)]
// CHECK:STDOUT:     %specific_impl_fn.loc87: <specific function> = specific_impl_function %impl.elem0.loc87, @A.WithSelf.AA(constants.%A.facet.095) [symbolic = %specific_impl_fn.loc75.2 (constants.%specific_impl_fn.567)]
// CHECK:STDOUT:     %A.WithSelf.AA.call.loc90: init %empty_tuple.type = call %specific_impl_fn.loc87()
// CHECK:STDOUT:     %T.ref.loc92: %facet_type.dc7 = name_ref T, %T.loc60.2 [symbolic = %T.loc60.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc93: type = facet_access_type %T.ref.loc92 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc93: type = converted %T.ref.loc92, %T.as_type.loc93 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %BB.ref.loc93: %B.assoc_type = name_ref BB, @B.WithSelf.%assoc0 [concrete = constants.%assoc0.00a]
// CHECK:STDOUT:     %impl.elem0.loc93: @G.%.loc81 (%.dfd) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc81.2 (constants.%impl.elem0.557)]
// CHECK:STDOUT:     %specific_impl_fn.loc93: <specific function> = specific_impl_function %impl.elem0.loc93, @B.WithSelf.BB(constants.%B.facet.0e7) [symbolic = %specific_impl_fn.loc81.2 (constants.%specific_impl_fn.fc3)]
// CHECK:STDOUT:     %B.WithSelf.BB.call.loc96: init %empty_tuple.type = call %specific_impl_fn.loc93()
// CHECK:STDOUT:     %T.ref.loc98: %facet_type.dc7 = name_ref T, %T.loc60.2 [symbolic = %T.loc60.1 (constants.%T)]
// CHECK:STDOUT:     %A.ref.loc101: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %AA.ref.loc102: %A.assoc_type = name_ref AA, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.a62]
// CHECK:STDOUT:     %T.as_type.loc99: type = facet_access_type %T.ref.loc98 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %A.facet.loc99: %A.type = facet_value %T.as_type.loc99, (constants.%A.lookup_impl_witness) [symbolic = %A.facet.loc75 (constants.%A.facet.095)]
// CHECK:STDOUT:     %.loc99: %A.type = converted %T.ref.loc98, %A.facet.loc99 [symbolic = %A.facet.loc75 (constants.%A.facet.095)]
// CHECK:STDOUT:     %impl.elem0.loc99: @G.%.loc75 (%.84b) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc75.2 (constants.%impl.elem0.26a)]
// CHECK:STDOUT:     %specific_impl_fn.loc99: <specific function> = specific_impl_function %impl.elem0.loc99, @A.WithSelf.AA(constants.%A.facet.095) [symbolic = %specific_impl_fn.loc75.2 (constants.%specific_impl_fn.567)]
// CHECK:STDOUT:     %A.WithSelf.AA.call.loc106: init %empty_tuple.type = call %specific_impl_fn.loc99()
// CHECK:STDOUT:     %T.ref.loc108: %facet_type.dc7 = name_ref T, %T.loc60.2 [symbolic = %T.loc60.1 (constants.%T)]
// CHECK:STDOUT:     %B.ref.loc111: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %BB.ref.loc112: %B.assoc_type = name_ref BB, @B.WithSelf.%assoc0 [concrete = constants.%assoc0.00a]
// CHECK:STDOUT:     %T.as_type.loc109: type = facet_access_type %T.ref.loc108 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %B.facet.loc109: %B.type = facet_value %T.as_type.loc109, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc81 (constants.%B.facet.0e7)]
// CHECK:STDOUT:     %.loc109: %B.type = converted %T.ref.loc108, %B.facet.loc109 [symbolic = %B.facet.loc81 (constants.%B.facet.0e7)]
// CHECK:STDOUT:     %impl.elem0.loc109: @G.%.loc81 (%.dfd) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc81.2 (constants.%impl.elem0.557)]
// CHECK:STDOUT:     %specific_impl_fn.loc109: <specific function> = specific_impl_function %impl.elem0.loc109, @B.WithSelf.BB(constants.%B.facet.0e7) [symbolic = %specific_impl_fn.loc81.2 (constants.%specific_impl_fn.fc3)]
// CHECK:STDOUT:     %B.WithSelf.BB.call.loc116: init %empty_tuple.type = call %specific_impl_fn.loc109()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc127.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc127.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc127.3: init %C to %.loc127.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc128.1: init %C = converted %.loc127.1, %.loc127.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %facet_value: %facet_type.dc7 = facet_value constants.%C, (constants.%Empty.impl_witness, constants.%A.impl_witness, constants.%B.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc130: %facet_type.dc7 = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%facet_value) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %.loc128.2: ref %C = temporary %.loc127.2, %.loc128.1
// CHECK:STDOUT:   %.loc128.3: %C = acquire_value %.loc128.2
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn(%.loc128.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc128.2, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc128.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%Self.61e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.c51
// CHECK:STDOUT:   %A.WithSelf.AA.type => constants.%A.WithSelf.AA.type.236
// CHECK:STDOUT:   %A.WithSelf.AA => constants.%A.WithSelf.AA.0c0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.AA(constants.%Self.c51) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.d0b
// CHECK:STDOUT:   %B.WithSelf.BB.type => constants.%B.WithSelf.BB.type.17c
// CHECK:STDOUT:   %B.WithSelf.BB => constants.%B.WithSelf.BB.f00
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf.BB(constants.%Self.d0b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%Empty.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%A.facet.34f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%A.facet.34f
// CHECK:STDOUT:   %A.WithSelf.AA.type => constants.%A.WithSelf.AA.type.6ff
// CHECK:STDOUT:   %A.WithSelf.AA => constants.%A.WithSelf.AA.ff5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.AA(constants.%A.facet.34f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%B.facet.e93) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%B.facet.e93
// CHECK:STDOUT:   %B.WithSelf.BB.type => constants.%B.WithSelf.BB.type.f5e
// CHECK:STDOUT:   %B.WithSelf.BB => constants.%B.WithSelf.BB.09b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf.BB(constants.%B.facet.e93) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc60.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %A.WithSelf.AA.type => constants.%A.WithSelf.AA.type.ce6
// CHECK:STDOUT:   %A.WithSelf.AA => constants.%A.WithSelf.AA.020
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %B.WithSelf.BB.type => constants.%B.WithSelf.BB.type.374
// CHECK:STDOUT:   %B.WithSelf.BB => constants.%B.WithSelf.BB.1c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%A.facet.095) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%A.facet.095
// CHECK:STDOUT:   %A.WithSelf.AA.type => constants.%A.WithSelf.AA.type.7b2
// CHECK:STDOUT:   %A.WithSelf.AA => constants.%A.WithSelf.AA.11e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.AA(constants.%A.facet.095) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%B.facet.0e7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%B.facet.0e7
// CHECK:STDOUT:   %B.WithSelf.BB.type => constants.%B.WithSelf.BB.type.5a2
// CHECK:STDOUT:   %B.WithSelf.BB => constants.%B.WithSelf.BB.b76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf.BB(constants.%B.facet.0e7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc60.1 => constants.%facet_value
// CHECK:STDOUT:   %T.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7c7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet.loc75 => constants.%A.facet.34f
// CHECK:STDOUT:   %A.WithSelf.AA.type => constants.%A.WithSelf.AA.type.6ff
// CHECK:STDOUT:   %.loc75 => constants.%.171
// CHECK:STDOUT:   %impl.elem0.loc75.2 => constants.%C.as.A.impl.AA
// CHECK:STDOUT:   %specific_impl_fn.loc75.2 => constants.%C.as.A.impl.AA
// CHECK:STDOUT:   %B.lookup_impl_witness => constants.%B.impl_witness
// CHECK:STDOUT:   %B.facet.loc81 => constants.%B.facet.e93
// CHECK:STDOUT:   %B.WithSelf.BB.type => constants.%B.WithSelf.BB.type.f5e
// CHECK:STDOUT:   %.loc81 => constants.%.e04
// CHECK:STDOUT:   %impl.elem0.loc81.2 => constants.%C.as.B.impl.BB
// CHECK:STDOUT:   %specific_impl_fn.loc81.2 => constants.%C.as.B.impl.BB
// CHECK:STDOUT: }
// CHECK:STDOUT:
