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
// CHECK:STDOUT:   %Self.b05: %Empty.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.907: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.AA.type: type = fn_type @A.AA [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.AA: %A.AA.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.6e7: %A.assoc_type = assoc_entity element0, @A.%A.AA.decl [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.284: %B.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %B.BB.type: type = fn_type @B.BB [concrete]
// CHECK:STDOUT:   %B.BB: %B.BB.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type @B [concrete]
// CHECK:STDOUT:   %assoc0.a29: %B.assoc_type = assoc_entity element0, @B.%B.BB.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.9d0: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness file.%Empty.impl_witness_table [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.A.impl.AA.type: type = fn_type @C.as.A.impl.AA [concrete]
// CHECK:STDOUT:   %C.as.A.impl.AA: %C.as.A.impl.AA.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.030: %A.type = facet_value %C, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness file.%B.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.B.impl.BB.type: type = fn_type @C.as.B.impl.BB [concrete]
// CHECK:STDOUT:   %C.as.B.impl.BB: %C.as.B.impl.BB.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.facet.e57: %B.type = facet_value %C, (%B.impl_witness) [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.9a3: type = fn_type @BitAndWith.Op, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.b22: type = fn_type_with_self_type %BitAndWith.Op.type.9a3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.bdd: <bound method> = bound_method %A.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.5da: type = facet_type <@Empty & @A> [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.f90: <bound method> = bound_method %facet_type.5da, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.4e3: type = facet_type <@Empty & @A & @B> [concrete]
// CHECK:STDOUT:   %T.0cd: %facet_type.4e3 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ae7: type = pattern_type %facet_type.4e3 [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.0cd [symbolic]
// CHECK:STDOUT:   %pattern_type.b68: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.f92: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.0cd, @A [symbolic]
// CHECK:STDOUT:   %A.facet.d79: %A.type = facet_value %T.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.b35: type = fn_type_with_self_type %A.AA.type, %A.facet.d79 [symbolic]
// CHECK:STDOUT:   %impl.elem0.2d3: %.b35 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c16: <specific function> = specific_impl_function %impl.elem0.2d3, @A.AA(%A.facet.d79) [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.0cd, @B [symbolic]
// CHECK:STDOUT:   %B.facet.01c: %B.type = facet_value %T.as_type, (%B.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.8c6: type = fn_type_with_self_type %B.BB.type, %B.facet.01c [symbolic]
// CHECK:STDOUT:   %impl.elem0.26a: %.8c6 = impl_witness_access %B.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ed7: <specific function> = specific_impl_function %impl.elem0.26a, @B.BB(%B.facet.01c) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.4e3 = facet_value %C, (%Empty.impl_witness, %A.impl_witness, %B.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%facet_value) [concrete]
// CHECK:STDOUT:   %.031: type = fn_type_with_self_type %A.AA.type, %A.facet.030 [concrete]
// CHECK:STDOUT:   %.0a1: type = fn_type_with_self_type %B.BB.type, %B.facet.e57 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.a4f: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude, loc14_42, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.a4f), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:   %Empty.impl_witness_table = impl_witness_table (), @C.as.Empty.impl [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness %Empty.impl_witness_table [concrete = constants.%Empty.impl_witness]
// CHECK:STDOUT:   impl_decl @C.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@C.as.A.impl.%C.as.A.impl.AA.decl), @C.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   impl_decl @C.as.B.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.impl_witness_table = impl_witness_table (@C.as.B.impl.%C.as.B.impl.BB.decl), @C.as.B.impl [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness %B.impl_witness_table [concrete = constants.%B.impl_witness]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ae7 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @G.%pattern_type (%pattern_type.b68) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @G.%pattern_type (%pattern_type.b68) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc33_20.1: type = splice_block %.loc33_20.3 [concrete = constants.%facet_type.4e3] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref.loc33: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:       %impl.elem0.loc33_12: %.b22 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc33_12: <bound method> = bound_method %A.ref.loc33, %impl.elem0.loc33_12 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.bdd]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc33_12: init type = call %bound_method.loc33_12(%A.ref.loc33, %Empty.ref) [concrete = constants.%facet_type.5da]
// CHECK:STDOUT:       %B.ref.loc33: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc33_20: %.b22 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc33_20: <bound method> = bound_method %type.as.BitAndWith.impl.Op.call.loc33_12, %impl.elem0.loc33_20 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.f90]
// CHECK:STDOUT:       %.loc33_12.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc33_12 [concrete = constants.%facet_type.5da]
// CHECK:STDOUT:       %.loc33_12.2: type = converted %type.as.BitAndWith.impl.Op.call.loc33_12, %.loc33_12.1 [concrete = constants.%facet_type.5da]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc33_20: init type = call %bound_method.loc33_20(%.loc33_12.2, %B.ref.loc33) [concrete = constants.%facet_type.4e3]
// CHECK:STDOUT:       %.loc33_20.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc33_20 [concrete = constants.%facet_type.4e3]
// CHECK:STDOUT:       %.loc33_20.3: type = converted %type.as.BitAndWith.impl.Op.call.loc33_20, %.loc33_20.2 [concrete = constants.%facet_type.4e3]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc33_6.2: %facet_type.4e3 = bind_symbolic_name T, 0 [symbolic = %T.loc33_6.1 (constants.%T.0cd)]
// CHECK:STDOUT:     %t.param: @G.%T.as_type.loc33_28.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc33_28.1: type = splice_block %.loc33_28.2 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc33: %facet_type.4e3 = name_ref T, %T.loc33_6.2 [symbolic = %T.loc33_6.1 (constants.%T.0cd)]
// CHECK:STDOUT:       %T.as_type.loc33_28.2: type = facet_access_type %T.ref.loc33 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc33_28.2: type = converted %T.ref.loc33, %T.as_type.loc33_28.2 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.as_type.loc33_28.1 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.b05]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .AA = <poisoned>
// CHECK:STDOUT:   .BB = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.907]
// CHECK:STDOUT:   %A.AA.decl: %A.AA.type = fn_decl @A.AA [concrete = constants.%A.AA] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.AA.decl [concrete = constants.%assoc0.6e7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .AA = %assoc0
// CHECK:STDOUT:   .BB = <poisoned>
// CHECK:STDOUT:   witness = (%A.AA.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.284]
// CHECK:STDOUT:   %B.BB.decl: %B.BB.type = fn_decl @B.BB [concrete = constants.%B.BB] {} {}
// CHECK:STDOUT:   %assoc0: %B.assoc_type = assoc_entity element0, %B.BB.decl [concrete = constants.%assoc0.a29]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .BB = %assoc0
// CHECK:STDOUT:   .AA = <poisoned>
// CHECK:STDOUT:   witness = (%B.BB.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc24: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Empty.impl: %C.ref as %Empty.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Empty.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.A.impl: %C.ref as %A.ref {
// CHECK:STDOUT:   %C.as.A.impl.AA.decl: %C.as.A.impl.AA.type = fn_decl @C.as.A.impl.AA [concrete = constants.%C.as.A.impl.AA] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .AA = %C.as.A.impl.AA.decl
// CHECK:STDOUT:   witness = file.%A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.B.impl: %C.ref as %B.ref {
// CHECK:STDOUT:   %C.as.B.impl.BB.decl: %C.as.B.impl.BB.type = fn_decl @C.as.B.impl.BB [concrete = constants.%C.as.B.impl.BB] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .BB = %C.as.B.impl.BB.decl
// CHECK:STDOUT:   witness = file.%B.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.9d0]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.AA(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B.BB(@B.%Self: %B.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
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
// CHECK:STDOUT: generic fn @G(%T.loc33_6.2: %facet_type.4e3) {
// CHECK:STDOUT:   %T.loc33_6.1: %facet_type.4e3 = bind_symbolic_name T, 0 [symbolic = %T.loc33_6.1 (constants.%T.0cd)]
// CHECK:STDOUT:   %T.as_type.loc33_28.1: type = facet_access_type %T.loc33_6.1 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc33_28.1 [symbolic = %pattern_type (constants.%pattern_type.b68)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc33_28.1 [symbolic = %require_complete (constants.%require_complete.f92)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc33_6.1, @A [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet.loc34: %A.type = facet_value %T.as_type.loc33_28.1, (%A.lookup_impl_witness) [symbolic = %A.facet.loc34 (constants.%A.facet.d79)]
// CHECK:STDOUT:   %.loc34_4.2: type = fn_type_with_self_type constants.%A.AA.type, %A.facet.loc34 [symbolic = %.loc34_4.2 (constants.%.b35)]
// CHECK:STDOUT:   %impl.elem0.loc34_4.2: @G.%.loc34_4.2 (%.b35) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.2d3)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem0.loc34_4.2, @A.AA(%A.facet.loc34) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.c16)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc33_6.1, @B [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness)]
// CHECK:STDOUT:   %B.facet.loc35: %B.type = facet_value %T.as_type.loc33_28.1, (%B.lookup_impl_witness) [symbolic = %B.facet.loc35 (constants.%B.facet.01c)]
// CHECK:STDOUT:   %.loc35_4.2: type = fn_type_with_self_type constants.%B.BB.type, %B.facet.loc35 [symbolic = %.loc35_4.2 (constants.%.8c6)]
// CHECK:STDOUT:   %impl.elem0.loc35_4.2: @G.%.loc35_4.2 (%.8c6) = impl_witness_access %B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.26a)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2: <specific function> = specific_impl_function %impl.elem0.loc35_4.2, @B.BB(%B.facet.loc35) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.ed7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @G.%T.as_type.loc33_28.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref.loc34: @G.%T.as_type.loc33_28.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %AA.ref.loc34: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.6e7]
// CHECK:STDOUT:     %T.as_type.loc34: type = facet_access_type constants.%T.0cd [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc34_4.1: type = converted constants.%T.0cd, %T.as_type.loc34 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc34_4.1: @G.%.loc34_4.2 (%.b35) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.2d3)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem0.loc34_4.1, @A.AA(constants.%A.facet.d79) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.c16)]
// CHECK:STDOUT:     %.loc34_8: init %empty_tuple.type = call %specific_impl_fn.loc34_4.1()
// CHECK:STDOUT:     %t.ref.loc35: @G.%T.as_type.loc33_28.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %BB.ref.loc35: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.a29]
// CHECK:STDOUT:     %T.as_type.loc35: type = facet_access_type constants.%T.0cd [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc35_4.1: type = converted constants.%T.0cd, %T.as_type.loc35 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc35_4.1: @G.%.loc35_4.2 (%.8c6) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.26a)]
// CHECK:STDOUT:     %specific_impl_fn.loc35_4.1: <specific function> = specific_impl_function %impl.elem0.loc35_4.1, @B.BB(constants.%B.facet.01c) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.ed7)]
// CHECK:STDOUT:     %.loc35_8: init %empty_tuple.type = call %specific_impl_fn.loc35_4.1()
// CHECK:STDOUT:     %T.ref.loc37: %facet_type.4e3 = name_ref T, %T.loc33_6.2 [symbolic = %T.loc33_6.1 (constants.%T.0cd)]
// CHECK:STDOUT:     %AA.ref.loc37: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.6e7]
// CHECK:STDOUT:     %T.as_type.loc37: type = facet_access_type %T.ref.loc37 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc37_4: type = converted %T.ref.loc37, %T.as_type.loc37 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc37: @G.%.loc34_4.2 (%.b35) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.2d3)]
// CHECK:STDOUT:     %specific_impl_fn.loc37: <specific function> = specific_impl_function %impl.elem0.loc37, @A.AA(constants.%A.facet.d79) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.c16)]
// CHECK:STDOUT:     %.loc37_8: init %empty_tuple.type = call %specific_impl_fn.loc37()
// CHECK:STDOUT:     %T.ref.loc38: %facet_type.4e3 = name_ref T, %T.loc33_6.2 [symbolic = %T.loc33_6.1 (constants.%T.0cd)]
// CHECK:STDOUT:     %BB.ref.loc38: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.a29]
// CHECK:STDOUT:     %T.as_type.loc38: type = facet_access_type %T.ref.loc38 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc38_4: type = converted %T.ref.loc38, %T.as_type.loc38 [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc38: @G.%.loc35_4.2 (%.8c6) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.26a)]
// CHECK:STDOUT:     %specific_impl_fn.loc38: <specific function> = specific_impl_function %impl.elem0.loc38, @B.BB(constants.%B.facet.01c) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.ed7)]
// CHECK:STDOUT:     %.loc38_8: init %empty_tuple.type = call %specific_impl_fn.loc38()
// CHECK:STDOUT:     %T.ref.loc40: %facet_type.4e3 = name_ref T, %T.loc33_6.2 [symbolic = %T.loc33_6.1 (constants.%T.0cd)]
// CHECK:STDOUT:     %A.ref.loc40: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %AA.ref.loc40: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.6e7]
// CHECK:STDOUT:     %T.as_type.loc40: type = facet_access_type constants.%T.0cd [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %A.facet.loc40: %A.type = facet_value %T.as_type.loc40, (constants.%A.lookup_impl_witness) [symbolic = %A.facet.loc34 (constants.%A.facet.d79)]
// CHECK:STDOUT:     %.loc40_4: %A.type = converted %T.ref.loc40, %A.facet.loc40 [symbolic = %A.facet.loc34 (constants.%A.facet.d79)]
// CHECK:STDOUT:     %impl.elem0.loc40: @G.%.loc34_4.2 (%.b35) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_4.2 (constants.%impl.elem0.2d3)]
// CHECK:STDOUT:     %specific_impl_fn.loc40: <specific function> = specific_impl_function %impl.elem0.loc40, @A.AA(constants.%A.facet.d79) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.c16)]
// CHECK:STDOUT:     %.loc40_12: init %empty_tuple.type = call %specific_impl_fn.loc40()
// CHECK:STDOUT:     %T.ref.loc41: %facet_type.4e3 = name_ref T, %T.loc33_6.2 [symbolic = %T.loc33_6.1 (constants.%T.0cd)]
// CHECK:STDOUT:     %B.ref.loc41: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %BB.ref.loc41: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.a29]
// CHECK:STDOUT:     %T.as_type.loc41: type = facet_access_type constants.%T.0cd [symbolic = %T.as_type.loc33_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %B.facet.loc41: %B.type = facet_value %T.as_type.loc41, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc35 (constants.%B.facet.01c)]
// CHECK:STDOUT:     %.loc41_4: %B.type = converted %T.ref.loc41, %B.facet.loc41 [symbolic = %B.facet.loc35 (constants.%B.facet.01c)]
// CHECK:STDOUT:     %impl.elem0.loc41: @G.%.loc35_4.2 (%.8c6) = impl_witness_access constants.%B.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_4.2 (constants.%impl.elem0.26a)]
// CHECK:STDOUT:     %specific_impl_fn.loc41: <specific function> = specific_impl_function %impl.elem0.loc41, @B.BB(constants.%B.facet.01c) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.ed7)]
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
// CHECK:STDOUT:   %facet_value.loc45_12.1: %facet_type.4e3 = facet_value constants.%C, (constants.%Empty.impl_witness, constants.%A.impl_witness, constants.%B.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc45_12.1: %facet_type.4e3 = converted constants.%C, %facet_value.loc45_12.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc45_12.2: %facet_type.4e3 = facet_value constants.%C, (constants.%Empty.impl_witness, constants.%A.impl_witness, constants.%B.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc45_12.2: %facet_type.4e3 = converted constants.%C, %facet_value.loc45_12.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%facet_value) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %.loc45_8.2: %C = bind_value %.loc45_8.1
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn(%.loc45_8.2)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc45_6.4, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc45_6.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.AA(constants.%Self.907) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.BB(constants.%Self.284) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.AA(constants.%A.facet.030) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.BB(constants.%B.facet.e57) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.0cd) {
// CHECK:STDOUT:   %T.loc33_6.1 => constants.%T.0cd
// CHECK:STDOUT:   %T.as_type.loc33_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.AA(constants.%A.facet.d79) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.BB(constants.%B.facet.01c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc33_6.1 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc33_28.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet.loc34 => constants.%A.facet.030
// CHECK:STDOUT:   %.loc34_4.2 => constants.%.031
// CHECK:STDOUT:   %impl.elem0.loc34_4.2 => constants.%C.as.A.impl.AA
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%C.as.A.impl.AA
// CHECK:STDOUT:   %B.lookup_impl_witness => constants.%B.impl_witness
// CHECK:STDOUT:   %B.facet.loc35 => constants.%B.facet.e57
// CHECK:STDOUT:   %.loc35_4.2 => constants.%.0a1
// CHECK:STDOUT:   %impl.elem0.loc35_4.2 => constants.%C.as.B.impl.BB
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%C.as.B.impl.BB
// CHECK:STDOUT: }
// CHECK:STDOUT:
