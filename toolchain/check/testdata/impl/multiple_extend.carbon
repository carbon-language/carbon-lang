// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/multiple_extend.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/multiple_extend.carbon

// --- different_impl_member_names.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F();
}

interface HasG {
  fn G();
}

class C {
  extend impl as HasF {
    fn F() {}
  }
  extend impl as HasG {
    fn G() {}
  }
}

fn H(c: C) {
  C.F();
  c.F();
  C.G();
  c.G();
}

// --- fail_ambiguous_impls.carbon

library "[[@TEST_NAME]]";

interface HasA1 {
  fn A();
}

interface HasA2 {
  fn A();
}

class D {
  extend impl as HasA1 {
    fn A() {}
  }
  extend impl as HasA2 {
    fn A() {}
  }
}

fn B(d: D) {
  // CHECK:STDERR: fail_ambiguous_impls.carbon:[[@LINE+4]]:3: error: ambiguous use of name `A` found in multiple extended scopes [NameAmbiguousDueToExtend]
  // CHECK:STDERR:   D.A();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  D.A();
  // CHECK:STDERR: fail_ambiguous_impls.carbon:[[@LINE+4]]:3: error: ambiguous use of name `A` found in multiple extended scopes [NameAmbiguousDueToExtend]
  // CHECK:STDERR:   d.A();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  d.A();
}

// --- different_impl_and_base.carbon

library "[[@TEST_NAME]]";

interface HasI {
  fn I();
}

base class B {
  fn J() {}
}

class E {
  extend base: B;
  extend impl as HasI {
    fn I() {}
  }
}

fn H(e: E) {
  E.I();
  e.I();
  E.J();
  e.J();
}

// --- fail_ambiguous_impl_and_base.carbon

library "[[@TEST_NAME]]";

base class Base {
  fn K() {}
}

interface HasK {
  fn K();
}

class L {
  extend base: Base;
  extend impl as HasK {
    fn K() {}
  }
}

fn M(l: L) {
   // CHECK:STDERR: fail_ambiguous_impl_and_base.carbon:[[@LINE+4]]:4: error: ambiguous use of name `K` found in multiple extended scopes [NameAmbiguousDueToExtend]
   // CHECK:STDERR:    L.K();
   // CHECK:STDERR:    ^~~
   // CHECK:STDERR:
   L.K();
   // CHECK:STDERR: fail_ambiguous_impl_and_base.carbon:[[@LINE+4]]:4: error: ambiguous use of name `K` found in multiple extended scopes [NameAmbiguousDueToExtend]
   // CHECK:STDERR:    l.K();
   // CHECK:STDERR:    ^~~
   // CHECK:STDERR:
   l.K();
}

// --- ambiguity_hidden.carbon

library "[[@TEST_NAME]]";

base class NBase {
  fn N() {}
}

interface HasN1 {
  fn N();
}

interface HasN2 {
  fn N();
}

class O {
  extend base: NBase;
  extend impl as HasN1 {
    fn N() {}
  }
  extend impl as HasN2 {
    fn N() {}
  }
  fn N();
}

fn P(o: O) {
   O.N();
   o.N();
}

// CHECK:STDOUT: --- different_impl_member_names.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self.c62: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.01b: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Self.c62) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.acd: %HasF.WithSelf.F.type.01b = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0.e62: %HasF.assoc_type = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self.ab8: %HasG.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasG.WithSelf.G.type.e8c: type = fn_type @HasG.WithSelf.G, @HasG.WithSelf(%Self.ab8) [symbolic]
// CHECK:STDOUT:   %HasG.WithSelf.G.ef7: %HasG.WithSelf.G.type.e8c = struct_value () [symbolic]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type @HasG [concrete]
// CHECK:STDOUT:   %assoc0.6fc: %HasG.assoc_type = assoc_entity element0, @HasG.WithSelf.%HasG.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.ba1: <witness> = impl_self_witness %C, @HasF [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @C.as.HasF.impl.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F: %C.as.HasF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %C, (%HasF.impl_witness) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.0a1: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%HasF.facet) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.a7e: %HasF.WithSelf.F.type.0a1 = struct_value () [concrete]
// CHECK:STDOUT:   %.e40: <witness> = impl_self_witness %C, @HasG [concrete]
// CHECK:STDOUT:   %HasG.impl_witness: <witness> = impl_witness @C.as.HasG.impl.%HasG.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.HasG.impl.G.type: type = fn_type @C.as.HasG.impl.G [concrete]
// CHECK:STDOUT:   %C.as.HasG.impl.G: %C.as.HasG.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.facet: %HasG.type = facet_value %C, (%HasG.impl_witness) [concrete]
// CHECK:STDOUT:   %HasG.WithSelf.G.type.c0e: type = fn_type @HasG.WithSelf.G, @HasG.WithSelf(%HasG.facet) [concrete]
// CHECK:STDOUT:   %HasG.WithSelf.G.5ae: %HasG.WithSelf.G.type.c0e = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type = wrapper_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.WithSelf.G.type.751: type = fn_type @HasG.WithSelf.G, @HasG.WithSelf(%C) [concrete]
// CHECK:STDOUT:   %HasG.WithSelf.G.568: %HasG.WithSelf.G.type.751 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.025: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%C) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.c72: %HasF.WithSelf.F.type.025 = struct_value () [concrete]
// CHECK:STDOUT:   %.837: type = fn_type_with_self_type %HasF.WithSelf.F.type.0a1, %HasF.facet [concrete]
// CHECK:STDOUT:   %.840: type = fn_type_with_self_type %HasG.WithSelf.G.type.c0e, %HasG.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .HasG = %HasG.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [concrete = constants.%HasF.type] {} {}
// CHECK:STDOUT:   %HasG.decl: type = interface_decl @HasG [concrete = constants.%HasG.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type = wrapper_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = wrapper_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c62]
// CHECK:STDOUT:   %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.01b) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.acd)] {} {}
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.WithSelf.F.decl [concrete = constants.%assoc0.e62]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @HasF.WithSelf.%assoc0
// CHECK:STDOUT:   .HasG = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   witness = (@HasF.WithSelf.%HasF.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT:   %Self: %HasG.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab8]
// CHECK:STDOUT:   %HasG.WithSelf.decl = interface_with_self_decl @HasG [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasG.WithSelf.G.decl: @HasG.WithSelf.%HasG.WithSelf.G.type (%HasG.WithSelf.G.type.e8c) = fn_decl @HasG.WithSelf.G [symbolic = @HasG.WithSelf.%HasG.WithSelf.G (constants.%HasG.WithSelf.G.ef7)] {} {}
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, %HasG.WithSelf.G.decl [concrete = constants.%assoc0.6fc]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = @HasG.WithSelf.%assoc0
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   witness = (@HasG.WithSelf.%HasG.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: %Self.ref as %HasF.ref {
// CHECK:STDOUT:   %C.as.HasF.impl.F.decl: %C.as.HasF.impl.F.type = fn_decl @C.as.HasF.impl.F [concrete = constants.%C.as.HasF.impl.F] {} {}
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.HasF.impl.F.decl
// CHECK:STDOUT:   extend %HasF.ref
// CHECK:STDOUT:   witness = %HasF.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasG.impl: %Self.ref as %HasG.ref {
// CHECK:STDOUT:   %C.as.HasG.impl.G.decl: %C.as.HasG.impl.G.type = fn_decl @C.as.HasG.impl.G [concrete = constants.%C.as.HasG.impl.G] {} {}
// CHECK:STDOUT:   %HasG.impl_witness_table = impl_witness_table (%C.as.HasG.impl.G.decl), @C.as.HasG.impl [concrete]
// CHECK:STDOUT:   %HasG.impl_witness: <witness> = impl_witness %HasG.impl_witness_table [concrete = constants.%HasG.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.HasG.impl.G.decl
// CHECK:STDOUT:   extend %HasG.ref
// CHECK:STDOUT:   witness = %HasG.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.HasF.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @C.as.HasF.impl.%Self.ref, @HasF [concrete = constants.%.ba1]
// CHECK:STDOUT:   impl_decl @C.as.HasG.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [concrete = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16: <witness> = impl_self_witness @C.as.HasG.impl.%Self.ref, @HasG [concrete = constants.%.e40]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .HasF = <poisoned>
// CHECK:STDOUT:   .HasG = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   extend @C.as.HasF.impl.%HasF.ref
// CHECK:STDOUT:   extend @C.as.HasG.impl.%HasG.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasG.WithSelf.G(@HasG.%Self: %HasG.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasF.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasG.impl.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref.loc22: %HasF.assoc_type = name_ref F, @HasF.WithSelf.%assoc0 [concrete = constants.%assoc0.e62]
// CHECK:STDOUT:   %impl.elem0.loc22: %.837 = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %C.as.HasF.impl.F.call.loc22: init %empty_tuple.type = call %impl.elem0.loc22()
// CHECK:STDOUT:   %c.ref.loc23: %C = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc23: %HasF.assoc_type = name_ref F, @HasF.WithSelf.%assoc0 [concrete = constants.%assoc0.e62]
// CHECK:STDOUT:   %impl.elem0.loc23: %.837 = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %C.as.HasF.impl.F.call.loc23: init %empty_tuple.type = call %impl.elem0.loc23()
// CHECK:STDOUT:   %C.ref.loc24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %G.ref.loc24: %HasG.assoc_type = name_ref G, @HasG.WithSelf.%assoc0 [concrete = constants.%assoc0.6fc]
// CHECK:STDOUT:   %impl.elem0.loc24: %.840 = impl_witness_access constants.%HasG.impl_witness, element0 [concrete = constants.%C.as.HasG.impl.G]
// CHECK:STDOUT:   %C.as.HasG.impl.G.call.loc24: init %empty_tuple.type = call %impl.elem0.loc24()
// CHECK:STDOUT:   %c.ref.loc25: %C = name_ref c, %c
// CHECK:STDOUT:   %G.ref.loc25: %HasG.assoc_type = name_ref G, @HasG.WithSelf.%assoc0 [concrete = constants.%assoc0.6fc]
// CHECK:STDOUT:   %impl.elem0.loc25: %.840 = impl_witness_access constants.%HasG.impl_witness, element0 [concrete = constants.%C.as.HasG.impl.G]
// CHECK:STDOUT:   %C.as.HasG.impl.G.call.loc25: init %empty_tuple.type = call %impl.elem0.loc25()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%Self.c62) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.c62
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.01b
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.acd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Self.c62) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.WithSelf(constants.%Self.ab8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.ab8
// CHECK:STDOUT:   %HasG.WithSelf.G.type => constants.%HasG.WithSelf.G.type.e8c
// CHECK:STDOUT:   %HasG.WithSelf.G => constants.%HasG.WithSelf.G.ef7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.WithSelf.G(constants.%Self.ab8) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%HasF.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.0a1
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.a7e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%HasF.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.WithSelf(constants.%HasG.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasG.facet
// CHECK:STDOUT:   %HasG.WithSelf.G.type => constants.%HasG.WithSelf.G.type.c0e
// CHECK:STDOUT:   %HasG.WithSelf.G => constants.%HasG.WithSelf.G.5ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.WithSelf.G(constants.%HasG.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.WithSelf(constants.%C) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT:   %HasG.WithSelf.G.type => constants.%HasG.WithSelf.G.type.751
// CHECK:STDOUT:   %HasG.WithSelf.G => constants.%HasG.WithSelf.G.568
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%C) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.025
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.c72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_ambiguous_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasA1.type: type = facet_type <@HasA1> [concrete]
// CHECK:STDOUT:   %Self.1db: %HasA1.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasA1.WithSelf.A.type.abc: type = fn_type @HasA1.WithSelf.A, @HasA1.WithSelf(%Self.1db) [symbolic]
// CHECK:STDOUT:   %HasA1.WithSelf.A.bb3: %HasA1.WithSelf.A.type.abc = struct_value () [symbolic]
// CHECK:STDOUT:   %HasA1.assoc_type: type = assoc_entity_type @HasA1 [concrete]
// CHECK:STDOUT:   %assoc0.54b: %HasA1.assoc_type = assoc_entity element0, @HasA1.WithSelf.%HasA1.WithSelf.A.decl [concrete]
// CHECK:STDOUT:   %HasA2.type: type = facet_type <@HasA2> [concrete]
// CHECK:STDOUT:   %Self.1d1: %HasA2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasA2.WithSelf.A.type.a4f: type = fn_type @HasA2.WithSelf.A, @HasA2.WithSelf(%Self.1d1) [symbolic]
// CHECK:STDOUT:   %HasA2.WithSelf.A.cfb: %HasA2.WithSelf.A.type.a4f = struct_value () [symbolic]
// CHECK:STDOUT:   %HasA2.assoc_type: type = assoc_entity_type @HasA2 [concrete]
// CHECK:STDOUT:   %assoc0.c71: %HasA2.assoc_type = assoc_entity element0, @HasA2.WithSelf.%HasA2.WithSelf.A.decl [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.869: <witness> = impl_self_witness %D, @HasA1 [concrete]
// CHECK:STDOUT:   %HasA1.impl_witness: <witness> = impl_witness @D.as.HasA1.impl.%HasA1.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.HasA1.impl.A.type: type = fn_type @D.as.HasA1.impl.A [concrete]
// CHECK:STDOUT:   %D.as.HasA1.impl.A: %D.as.HasA1.impl.A.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasA1.facet: %HasA1.type = facet_value %D, (%HasA1.impl_witness) [concrete]
// CHECK:STDOUT:   %HasA1.WithSelf.A.type.504: type = fn_type @HasA1.WithSelf.A, @HasA1.WithSelf(%HasA1.facet) [concrete]
// CHECK:STDOUT:   %HasA1.WithSelf.A.9ff: %HasA1.WithSelf.A.type.504 = struct_value () [concrete]
// CHECK:STDOUT:   %.b95: <witness> = impl_self_witness %D, @HasA2 [concrete]
// CHECK:STDOUT:   %HasA2.impl_witness: <witness> = impl_witness @D.as.HasA2.impl.%HasA2.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.HasA2.impl.A.type: type = fn_type @D.as.HasA2.impl.A [concrete]
// CHECK:STDOUT:   %D.as.HasA2.impl.A: %D.as.HasA2.impl.A.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasA2.facet: %HasA2.type = facet_value %D, (%HasA2.impl_witness) [concrete]
// CHECK:STDOUT:   %HasA2.WithSelf.A.type.cc2: type = fn_type @HasA2.WithSelf.A, @HasA2.WithSelf(%HasA2.facet) [concrete]
// CHECK:STDOUT:   %HasA2.WithSelf.A.628: %HasA2.WithSelf.A.type.cc2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %d.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type = wrapper_binding_pattern d, %d.param_patt [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasA2.WithSelf.A.type.063: type = fn_type @HasA2.WithSelf.A, @HasA2.WithSelf(%D) [concrete]
// CHECK:STDOUT:   %HasA2.WithSelf.A.1a9: %HasA2.WithSelf.A.type.063 = struct_value () [concrete]
// CHECK:STDOUT:   %HasA1.WithSelf.A.type.b1e: type = fn_type @HasA1.WithSelf.A, @HasA1.WithSelf(%D) [concrete]
// CHECK:STDOUT:   %HasA1.WithSelf.A.254: %HasA1.WithSelf.A.type.b1e = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .HasA1 = %HasA1.decl
// CHECK:STDOUT:     .HasA2 = %HasA2.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasA1.decl: type = interface_decl @HasA1 [concrete = constants.%HasA1.type] {} {}
// CHECK:STDOUT:   %HasA2.decl: type = interface_decl @HasA2 [concrete = constants.%HasA2.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %d.param_patt: %pattern_type = value_param_pattern [concrete = constants.%d.param_patt]
// CHECK:STDOUT:     %d.patt: %pattern_type = wrapper_binding_pattern d, %d.param_patt [concrete = constants.%d.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc21: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %d: %D = wrapper_binding d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasA1 {
// CHECK:STDOUT:   %Self: %HasA1.type = symbolic_binding Self, 0 [symbolic = constants.%Self.1db]
// CHECK:STDOUT:   %HasA1.WithSelf.decl = interface_with_self_decl @HasA1 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasA1.WithSelf.A.decl: @HasA1.WithSelf.%HasA1.WithSelf.A.type (%HasA1.WithSelf.A.type.abc) = fn_decl @HasA1.WithSelf.A [symbolic = @HasA1.WithSelf.%HasA1.WithSelf.A (constants.%HasA1.WithSelf.A.bb3)] {} {}
// CHECK:STDOUT:   %assoc0: %HasA1.assoc_type = assoc_entity element0, %HasA1.WithSelf.A.decl [concrete = constants.%assoc0.54b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = @HasA1.WithSelf.%assoc0
// CHECK:STDOUT:   .HasA2 = <poisoned>
// CHECK:STDOUT:   witness = (@HasA1.WithSelf.%HasA1.WithSelf.A.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasA2 {
// CHECK:STDOUT:   %Self: %HasA2.type = symbolic_binding Self, 0 [symbolic = constants.%Self.1d1]
// CHECK:STDOUT:   %HasA2.WithSelf.decl = interface_with_self_decl @HasA2 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasA2.WithSelf.A.decl: @HasA2.WithSelf.%HasA2.WithSelf.A.type (%HasA2.WithSelf.A.type.a4f) = fn_decl @HasA2.WithSelf.A [symbolic = @HasA2.WithSelf.%HasA2.WithSelf.A (constants.%HasA2.WithSelf.A.cfb)] {} {}
// CHECK:STDOUT:   %assoc0: %HasA2.assoc_type = assoc_entity element0, %HasA2.WithSelf.A.decl [concrete = constants.%assoc0.c71]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = @HasA2.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@HasA2.WithSelf.%HasA2.WithSelf.A.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasA1.impl: %Self.ref as %HasA1.ref {
// CHECK:STDOUT:   %D.as.HasA1.impl.A.decl: %D.as.HasA1.impl.A.type = fn_decl @D.as.HasA1.impl.A [concrete = constants.%D.as.HasA1.impl.A] {} {}
// CHECK:STDOUT:   %HasA1.impl_witness_table = impl_witness_table (%D.as.HasA1.impl.A.decl), @D.as.HasA1.impl [concrete]
// CHECK:STDOUT:   %HasA1.impl_witness: <witness> = impl_witness %HasA1.impl_witness_table [concrete = constants.%HasA1.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = %D.as.HasA1.impl.A.decl
// CHECK:STDOUT:   extend %HasA1.ref
// CHECK:STDOUT:   witness = %HasA1.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasA2.impl: %Self.ref as %HasA2.ref {
// CHECK:STDOUT:   %D.as.HasA2.impl.A.decl: %D.as.HasA2.impl.A.type = fn_decl @D.as.HasA2.impl.A [concrete = constants.%D.as.HasA2.impl.A] {} {}
// CHECK:STDOUT:   %HasA2.impl_witness_table = impl_witness_table (%D.as.HasA2.impl.A.decl), @D.as.HasA2.impl [concrete]
// CHECK:STDOUT:   %HasA2.impl_witness: <witness> = impl_witness %HasA2.impl_witness_table [concrete = constants.%HasA2.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = %D.as.HasA2.impl.A.decl
// CHECK:STDOUT:   extend %HasA2.ref
// CHECK:STDOUT:   witness = %HasA2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   impl_decl @D.as.HasA1.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %HasA1.ref: type = name_ref HasA1, file.%HasA1.decl [concrete = constants.%HasA1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @D.as.HasA1.impl.%Self.ref, @HasA1 [concrete = constants.%.869]
// CHECK:STDOUT:   impl_decl @D.as.HasA2.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %HasA2.ref: type = name_ref HasA2, file.%HasA2.decl [concrete = constants.%HasA2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16: <witness> = impl_self_witness @D.as.HasA2.impl.%Self.ref, @HasA2 [concrete = constants.%.b95]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .HasA1 = <poisoned>
// CHECK:STDOUT:   .HasA2 = <poisoned>
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   extend @D.as.HasA1.impl.%HasA1.ref
// CHECK:STDOUT:   extend @D.as.HasA2.impl.%HasA2.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasA1.WithSelf.A(@HasA1.%Self: %HasA1.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasA2.WithSelf.A(@HasA2.%Self: %HasA2.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.HasA1.impl.A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.HasA2.impl.A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%d.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref.loc26: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %A.ref.loc26: <error> = name_ref A, <error> [concrete = <error>]
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %A.ref.loc31: <error> = name_ref A, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA1.WithSelf(constants.%Self.1db) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.1db
// CHECK:STDOUT:   %HasA1.WithSelf.A.type => constants.%HasA1.WithSelf.A.type.abc
// CHECK:STDOUT:   %HasA1.WithSelf.A => constants.%HasA1.WithSelf.A.bb3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA1.WithSelf.A(constants.%Self.1db) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA2.WithSelf(constants.%Self.1d1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.1d1
// CHECK:STDOUT:   %HasA2.WithSelf.A.type => constants.%HasA2.WithSelf.A.type.a4f
// CHECK:STDOUT:   %HasA2.WithSelf.A => constants.%HasA2.WithSelf.A.cfb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA2.WithSelf.A(constants.%Self.1d1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA1.WithSelf(constants.%HasA1.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasA1.facet
// CHECK:STDOUT:   %HasA1.WithSelf.A.type => constants.%HasA1.WithSelf.A.type.504
// CHECK:STDOUT:   %HasA1.WithSelf.A => constants.%HasA1.WithSelf.A.9ff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA1.WithSelf.A(constants.%HasA1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA2.WithSelf(constants.%HasA2.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasA2.facet
// CHECK:STDOUT:   %HasA2.WithSelf.A.type => constants.%HasA2.WithSelf.A.type.cc2
// CHECK:STDOUT:   %HasA2.WithSelf.A => constants.%HasA2.WithSelf.A.628
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA2.WithSelf.A(constants.%HasA2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA2.WithSelf(constants.%D) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%D
// CHECK:STDOUT:   %HasA2.WithSelf.A.type => constants.%HasA2.WithSelf.A.type.063
// CHECK:STDOUT:   %HasA2.WithSelf.A => constants.%HasA2.WithSelf.A.1a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA1.WithSelf(constants.%D) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%D
// CHECK:STDOUT:   %HasA1.WithSelf.A.type => constants.%HasA1.WithSelf.A.type.b1e
// CHECK:STDOUT:   %HasA1.WithSelf.A => constants.%HasA1.WithSelf.A.254
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- different_impl_and_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasI.type: type = facet_type <@HasI> [concrete]
// CHECK:STDOUT:   %Self: %HasI.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasI.WithSelf.I.type.ad7: type = fn_type @HasI.WithSelf.I, @HasI.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasI.WithSelf.I.03b: %HasI.WithSelf.I.type.ad7 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasI.assoc_type: type = assoc_entity_type @HasI [concrete]
// CHECK:STDOUT:   %assoc0: %HasI.assoc_type = assoc_entity element0, @HasI.WithSelf.%HasI.WithSelf.I.decl [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.J.type: type = fn_type @B.J [concrete]
// CHECK:STDOUT:   %B.J: %B.J.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %E.elem: type = unbound_element_type %E, %B [concrete]
// CHECK:STDOUT:   %.aa2: <witness> = impl_self_witness %E, @HasI [concrete]
// CHECK:STDOUT:   %HasI.impl_witness: <witness> = impl_witness @E.as.HasI.impl.%HasI.impl_witness_table [concrete]
// CHECK:STDOUT:   %E.as.HasI.impl.I.type: type = fn_type @E.as.HasI.impl.I [concrete]
// CHECK:STDOUT:   %E.as.HasI.impl.I: %E.as.HasI.impl.I.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasI.facet: %HasI.type = facet_value %E, (%HasI.impl_witness) [concrete]
// CHECK:STDOUT:   %HasI.WithSelf.I.type.bea: type = fn_type @HasI.WithSelf.I, @HasI.WithSelf(%HasI.facet) [concrete]
// CHECK:STDOUT:   %HasI.WithSelf.I.31c: %HasI.WithSelf.I.type.bea = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.e05: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.c77: <witness> = complete_type_witness %struct_type.base.e05 [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %e.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type = wrapper_binding_pattern e, %e.param_patt [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasI.WithSelf.I.type.7f1: type = fn_type @HasI.WithSelf.I, @HasI.WithSelf(%E) [concrete]
// CHECK:STDOUT:   %HasI.WithSelf.I.f9b: %HasI.WithSelf.I.type.7f1 = struct_value () [concrete]
// CHECK:STDOUT:   %.4d0: type = fn_type_with_self_type %HasI.WithSelf.I.type.bea, %HasI.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .HasI = %HasI.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasI.decl: type = interface_decl @HasI [concrete = constants.%HasI.type] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %e.param_patt: %pattern_type = value_param_pattern [concrete = constants.%e.param_patt]
// CHECK:STDOUT:     %e.patt: %pattern_type = wrapper_binding_pattern e, %e.param_patt [concrete = constants.%e.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref.loc19: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = wrapper_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasI {
// CHECK:STDOUT:   %Self: %HasI.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasI.WithSelf.decl = interface_with_self_decl @HasI [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasI.WithSelf.I.decl: @HasI.WithSelf.%HasI.WithSelf.I.type (%HasI.WithSelf.I.type.ad7) = fn_decl @HasI.WithSelf.I [symbolic = @HasI.WithSelf.%HasI.WithSelf.I (constants.%HasI.WithSelf.I.03b)] {} {}
// CHECK:STDOUT:   %assoc0: %HasI.assoc_type = assoc_entity element0, %HasI.WithSelf.I.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = @HasI.WithSelf.%assoc0
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   witness = (@HasI.WithSelf.%HasI.WithSelf.I.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.HasI.impl: %Self.ref as %HasI.ref {
// CHECK:STDOUT:   %E.as.HasI.impl.I.decl: %E.as.HasI.impl.I.type = fn_decl @E.as.HasI.impl.I [concrete = constants.%E.as.HasI.impl.I] {} {}
// CHECK:STDOUT:   %HasI.impl_witness_table = impl_witness_table (%E.as.HasI.impl.I.decl), @E.as.HasI.impl [concrete]
// CHECK:STDOUT:   %HasI.impl_witness: <witness> = impl_witness %HasI.impl_witness_table [concrete = constants.%HasI.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .I = %E.as.HasI.impl.I.decl
// CHECK:STDOUT:   extend %HasI.ref
// CHECK:STDOUT:   witness = %HasI.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %B.J.decl: %B.J.type = fn_decl @B.J [concrete = constants.%B.J] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .J = %B.J.decl
// CHECK:STDOUT:   .HasI = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc13: %E.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   impl_decl @E.as.HasI.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %HasI.ref: type = name_ref HasI, file.%HasI.decl [concrete = constants.%HasI.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness @E.as.HasI.impl.%Self.ref, @HasI [concrete = constants.%.aa2]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.e05 [concrete = constants.%complete_type.c77]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .HasI = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT:   extend @E.as.HasI.impl.%HasI.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasI.WithSelf.I(@HasI.%Self: %HasI.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.J() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.HasI.impl.I() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%e.param: %E) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %E.ref.loc20: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %I.ref.loc20: %HasI.assoc_type = name_ref I, @HasI.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc20: %.4d0 = impl_witness_access constants.%HasI.impl_witness, element0 [concrete = constants.%E.as.HasI.impl.I]
// CHECK:STDOUT:   %E.as.HasI.impl.I.call.loc20: init %empty_tuple.type = call %impl.elem0.loc20()
// CHECK:STDOUT:   %e.ref.loc21: %E = name_ref e, %e
// CHECK:STDOUT:   %I.ref.loc21: %HasI.assoc_type = name_ref I, @HasI.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc21: %.4d0 = impl_witness_access constants.%HasI.impl_witness, element0 [concrete = constants.%E.as.HasI.impl.I]
// CHECK:STDOUT:   %E.as.HasI.impl.I.call.loc21: init %empty_tuple.type = call %impl.elem0.loc21()
// CHECK:STDOUT:   %E.ref.loc22: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %J.ref.loc22: %B.J.type = name_ref J, @B.%B.J.decl [concrete = constants.%B.J]
// CHECK:STDOUT:   %B.J.call.loc22: init %empty_tuple.type = call %J.ref.loc22()
// CHECK:STDOUT:   %e.ref.loc23: %E = name_ref e, %e
// CHECK:STDOUT:   %J.ref.loc23: %B.J.type = name_ref J, @B.%B.J.decl [concrete = constants.%B.J]
// CHECK:STDOUT:   %B.J.call.loc23: init %empty_tuple.type = call %J.ref.loc23()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasI.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %HasI.WithSelf.I.type => constants.%HasI.WithSelf.I.type.ad7
// CHECK:STDOUT:   %HasI.WithSelf.I => constants.%HasI.WithSelf.I.03b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasI.WithSelf.I(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasI.WithSelf(constants.%HasI.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasI.facet
// CHECK:STDOUT:   %HasI.WithSelf.I.type => constants.%HasI.WithSelf.I.type.bea
// CHECK:STDOUT:   %HasI.WithSelf.I => constants.%HasI.WithSelf.I.31c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasI.WithSelf.I(constants.%HasI.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasI.WithSelf(constants.%E) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%E
// CHECK:STDOUT:   %HasI.WithSelf.I.type => constants.%HasI.WithSelf.I.type.7f1
// CHECK:STDOUT:   %HasI.WithSelf.I => constants.%HasI.WithSelf.I.f9b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_ambiguous_impl_and_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Base.K.type: type = fn_type @Base.K [concrete]
// CHECK:STDOUT:   %Base.K: %Base.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasK.type: type = facet_type <@HasK> [concrete]
// CHECK:STDOUT:   %Self: %HasK.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasK.WithSelf.K.type.383: type = fn_type @HasK.WithSelf.K, @HasK.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %HasK.WithSelf.K.328: %HasK.WithSelf.K.type.383 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasK.assoc_type: type = assoc_entity_type @HasK [concrete]
// CHECK:STDOUT:   %assoc0: %HasK.assoc_type = assoc_entity element0, @HasK.WithSelf.%HasK.WithSelf.K.decl [concrete]
// CHECK:STDOUT:   %L: type = class_type @L [concrete]
// CHECK:STDOUT:   %L.elem: type = unbound_element_type %L, %Base [concrete]
// CHECK:STDOUT:   %.099: <witness> = impl_self_witness %L, @HasK [concrete]
// CHECK:STDOUT:   %HasK.impl_witness: <witness> = impl_witness @L.as.HasK.impl.%HasK.impl_witness_table [concrete]
// CHECK:STDOUT:   %L.as.HasK.impl.K.type: type = fn_type @L.as.HasK.impl.K [concrete]
// CHECK:STDOUT:   %L.as.HasK.impl.K: %L.as.HasK.impl.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasK.facet: %HasK.type = facet_value %L, (%HasK.impl_witness) [concrete]
// CHECK:STDOUT:   %HasK.WithSelf.K.type.581: type = fn_type @HasK.WithSelf.K, @HasK.WithSelf(%HasK.facet) [concrete]
// CHECK:STDOUT:   %HasK.WithSelf.K.543: %HasK.WithSelf.K.type.581 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.cae: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.c02: <witness> = complete_type_witness %struct_type.base.cae [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %L [concrete]
// CHECK:STDOUT:   %l.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %l.patt: %pattern_type = wrapper_binding_pattern l, %l.param_patt [concrete]
// CHECK:STDOUT:   %M.type: type = fn_type @M [concrete]
// CHECK:STDOUT:   %M: %M.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasK.WithSelf.K.type.4a7: type = fn_type @HasK.WithSelf.K, @HasK.WithSelf(%L) [concrete]
// CHECK:STDOUT:   %HasK.WithSelf.K.73a: %HasK.WithSelf.K.type.4a7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .HasK = %HasK.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %HasK.decl: type = interface_decl @HasK [concrete = constants.%HasK.type] {} {}
// CHECK:STDOUT:   %L.decl: type = class_decl @L [concrete = constants.%L] {} {}
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [concrete = constants.%M] {
// CHECK:STDOUT:     %l.param_patt: %pattern_type = value_param_pattern [concrete = constants.%l.param_patt]
// CHECK:STDOUT:     %l.patt: %pattern_type = wrapper_binding_pattern l, %l.param_patt [concrete = constants.%l.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %l.param: %L = value_param call_param0
// CHECK:STDOUT:     %L.ref.loc19: type = name_ref L, file.%L.decl [concrete = constants.%L]
// CHECK:STDOUT:     %l: %L = wrapper_binding l, %l.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasK {
// CHECK:STDOUT:   %Self: %HasK.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasK.WithSelf.decl = interface_with_self_decl @HasK [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasK.WithSelf.K.decl: @HasK.WithSelf.%HasK.WithSelf.K.type (%HasK.WithSelf.K.type.383) = fn_decl @HasK.WithSelf.K [symbolic = @HasK.WithSelf.%HasK.WithSelf.K (constants.%HasK.WithSelf.K.328)] {} {}
// CHECK:STDOUT:   %assoc0: %HasK.assoc_type = assoc_entity element0, %HasK.WithSelf.K.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = @HasK.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@HasK.WithSelf.%HasK.WithSelf.K.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @L.as.HasK.impl: %Self.ref as %HasK.ref {
// CHECK:STDOUT:   %L.as.HasK.impl.K.decl: %L.as.HasK.impl.K.type = fn_decl @L.as.HasK.impl.K [concrete = constants.%L.as.HasK.impl.K] {} {}
// CHECK:STDOUT:   %HasK.impl_witness_table = impl_witness_table (%L.as.HasK.impl.K.decl), @L.as.HasK.impl [concrete]
// CHECK:STDOUT:   %HasK.impl_witness: <witness> = impl_witness %HasK.impl_witness_table [concrete = constants.%HasK.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %L.as.HasK.impl.K.decl
// CHECK:STDOUT:   extend %HasK.ref
// CHECK:STDOUT:   witness = %HasK.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %Base.K.decl: %Base.K.type = fn_decl @Base.K [concrete = constants.%Base.K] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .K = %Base.K.decl
// CHECK:STDOUT:   .HasK = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @L {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc13: %L.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   impl_decl @L.as.HasK.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%L [concrete = constants.%L]
// CHECK:STDOUT:     %HasK.ref: type = name_ref HasK, file.%HasK.decl [concrete = constants.%HasK.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness @L.as.HasK.impl.%Self.ref, @HasK [concrete = constants.%.099]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.cae [concrete = constants.%complete_type.c02]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%L
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .HasK = <poisoned>
// CHECK:STDOUT:   .K = <poisoned>
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT:   extend @L.as.HasK.impl.%HasK.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.K() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasK.WithSelf.K(@HasK.%Self: %HasK.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L.as.HasK.impl.K() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @M(%l.param: %L) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %L.ref.loc24: type = name_ref L, file.%L.decl [concrete = constants.%L]
// CHECK:STDOUT:   %K.ref.loc24: <error> = name_ref K, <error> [concrete = <error>]
// CHECK:STDOUT:   %l.ref: %L = name_ref l, %l
// CHECK:STDOUT:   %K.ref.loc29: <error> = name_ref K, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasK.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %HasK.WithSelf.K.type => constants.%HasK.WithSelf.K.type.383
// CHECK:STDOUT:   %HasK.WithSelf.K => constants.%HasK.WithSelf.K.328
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasK.WithSelf.K(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasK.WithSelf(constants.%HasK.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasK.facet
// CHECK:STDOUT:   %HasK.WithSelf.K.type => constants.%HasK.WithSelf.K.type.581
// CHECK:STDOUT:   %HasK.WithSelf.K => constants.%HasK.WithSelf.K.543
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasK.WithSelf.K(constants.%HasK.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasK.WithSelf(constants.%L) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%L
// CHECK:STDOUT:   %HasK.WithSelf.K.type => constants.%HasK.WithSelf.K.type.4a7
// CHECK:STDOUT:   %HasK.WithSelf.K => constants.%HasK.WithSelf.K.73a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ambiguity_hidden.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NBase: type = class_type @NBase [concrete]
// CHECK:STDOUT:   %NBase.N.type: type = fn_type @NBase.N [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NBase.N: %NBase.N.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasN1.type: type = facet_type <@HasN1> [concrete]
// CHECK:STDOUT:   %Self.97e: %HasN1.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasN1.WithSelf.N.type.5e2: type = fn_type @HasN1.WithSelf.N, @HasN1.WithSelf(%Self.97e) [symbolic]
// CHECK:STDOUT:   %HasN1.WithSelf.N.9d7: %HasN1.WithSelf.N.type.5e2 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasN1.assoc_type: type = assoc_entity_type @HasN1 [concrete]
// CHECK:STDOUT:   %assoc0.b3a: %HasN1.assoc_type = assoc_entity element0, @HasN1.WithSelf.%HasN1.WithSelf.N.decl [concrete]
// CHECK:STDOUT:   %HasN2.type: type = facet_type <@HasN2> [concrete]
// CHECK:STDOUT:   %Self.e42: %HasN2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasN2.WithSelf.N.type.8b8: type = fn_type @HasN2.WithSelf.N, @HasN2.WithSelf(%Self.e42) [symbolic]
// CHECK:STDOUT:   %HasN2.WithSelf.N.2b2: %HasN2.WithSelf.N.type.8b8 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasN2.assoc_type: type = assoc_entity_type @HasN2 [concrete]
// CHECK:STDOUT:   %assoc0.f1b: %HasN2.assoc_type = assoc_entity element0, @HasN2.WithSelf.%HasN2.WithSelf.N.decl [concrete]
// CHECK:STDOUT:   %O: type = class_type @O [concrete]
// CHECK:STDOUT:   %O.elem: type = unbound_element_type %O, %NBase [concrete]
// CHECK:STDOUT:   %.665: <witness> = impl_self_witness %O, @HasN1 [concrete]
// CHECK:STDOUT:   %HasN1.impl_witness: <witness> = impl_witness @O.as.HasN1.impl.%HasN1.impl_witness_table [concrete]
// CHECK:STDOUT:   %O.as.HasN1.impl.N.type: type = fn_type @O.as.HasN1.impl.N [concrete]
// CHECK:STDOUT:   %O.as.HasN1.impl.N: %O.as.HasN1.impl.N.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasN1.facet: %HasN1.type = facet_value %O, (%HasN1.impl_witness) [concrete]
// CHECK:STDOUT:   %HasN1.WithSelf.N.type.296: type = fn_type @HasN1.WithSelf.N, @HasN1.WithSelf(%HasN1.facet) [concrete]
// CHECK:STDOUT:   %HasN1.WithSelf.N.b63: %HasN1.WithSelf.N.type.296 = struct_value () [concrete]
// CHECK:STDOUT:   %.81b: <witness> = impl_self_witness %O, @HasN2 [concrete]
// CHECK:STDOUT:   %HasN2.impl_witness: <witness> = impl_witness @O.as.HasN2.impl.%HasN2.impl_witness_table [concrete]
// CHECK:STDOUT:   %O.as.HasN2.impl.N.type: type = fn_type @O.as.HasN2.impl.N [concrete]
// CHECK:STDOUT:   %O.as.HasN2.impl.N: %O.as.HasN2.impl.N.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasN2.facet: %HasN2.type = facet_value %O, (%HasN2.impl_witness) [concrete]
// CHECK:STDOUT:   %HasN2.WithSelf.N.type.88c: type = fn_type @HasN2.WithSelf.N, @HasN2.WithSelf(%HasN2.facet) [concrete]
// CHECK:STDOUT:   %HasN2.WithSelf.N.306: %HasN2.WithSelf.N.type.88c = struct_value () [concrete]
// CHECK:STDOUT:   %O.N.type: type = fn_type @O.N [concrete]
// CHECK:STDOUT:   %O.N: %O.N.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.40f: type = struct_type {.base: %NBase} [concrete]
// CHECK:STDOUT:   %complete_type.3b3: <witness> = complete_type_witness %struct_type.base.40f [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %O [concrete]
// CHECK:STDOUT:   %o.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %o.patt: %pattern_type = wrapper_binding_pattern o, %o.param_patt [concrete]
// CHECK:STDOUT:   %P.type: type = fn_type @P [concrete]
// CHECK:STDOUT:   %P: %P.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NBase = %NBase.decl
// CHECK:STDOUT:     .HasN1 = %HasN1.decl
// CHECK:STDOUT:     .HasN2 = %HasN2.decl
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     .P = %P.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NBase.decl: type = class_decl @NBase [concrete = constants.%NBase] {} {}
// CHECK:STDOUT:   %HasN1.decl: type = interface_decl @HasN1 [concrete = constants.%HasN1.type] {} {}
// CHECK:STDOUT:   %HasN2.decl: type = interface_decl @HasN2 [concrete = constants.%HasN2.type] {} {}
// CHECK:STDOUT:   %O.decl: type = class_decl @O [concrete = constants.%O] {} {}
// CHECK:STDOUT:   %P.decl: %P.type = fn_decl @P [concrete = constants.%P] {
// CHECK:STDOUT:     %o.param_patt: %pattern_type = value_param_pattern [concrete = constants.%o.param_patt]
// CHECK:STDOUT:     %o.patt: %pattern_type = wrapper_binding_pattern o, %o.param_patt [concrete = constants.%o.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %o.param: %O = value_param call_param0
// CHECK:STDOUT:     %O.ref.loc27: type = name_ref O, file.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %o: %O = wrapper_binding o, %o.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasN1 {
// CHECK:STDOUT:   %Self: %HasN1.type = symbolic_binding Self, 0 [symbolic = constants.%Self.97e]
// CHECK:STDOUT:   %HasN1.WithSelf.decl = interface_with_self_decl @HasN1 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasN1.WithSelf.N.decl: @HasN1.WithSelf.%HasN1.WithSelf.N.type (%HasN1.WithSelf.N.type.5e2) = fn_decl @HasN1.WithSelf.N [symbolic = @HasN1.WithSelf.%HasN1.WithSelf.N (constants.%HasN1.WithSelf.N.9d7)] {} {}
// CHECK:STDOUT:   %assoc0: %HasN1.assoc_type = assoc_entity element0, %HasN1.WithSelf.N.decl [concrete = constants.%assoc0.b3a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = @HasN1.WithSelf.%assoc0
// CHECK:STDOUT:   .HasN2 = <poisoned>
// CHECK:STDOUT:   witness = (@HasN1.WithSelf.%HasN1.WithSelf.N.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasN2 {
// CHECK:STDOUT:   %Self: %HasN2.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e42]
// CHECK:STDOUT:   %HasN2.WithSelf.decl = interface_with_self_decl @HasN2 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %HasN2.WithSelf.N.decl: @HasN2.WithSelf.%HasN2.WithSelf.N.type (%HasN2.WithSelf.N.type.8b8) = fn_decl @HasN2.WithSelf.N [symbolic = @HasN2.WithSelf.%HasN2.WithSelf.N (constants.%HasN2.WithSelf.N.2b2)] {} {}
// CHECK:STDOUT:   %assoc0: %HasN2.assoc_type = assoc_entity element0, %HasN2.WithSelf.N.decl [concrete = constants.%assoc0.f1b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = @HasN2.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@HasN2.WithSelf.%HasN2.WithSelf.N.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @O.as.HasN1.impl: %Self.ref as %HasN1.ref {
// CHECK:STDOUT:   %O.as.HasN1.impl.N.decl: %O.as.HasN1.impl.N.type = fn_decl @O.as.HasN1.impl.N [concrete = constants.%O.as.HasN1.impl.N] {} {}
// CHECK:STDOUT:   %HasN1.impl_witness_table = impl_witness_table (%O.as.HasN1.impl.N.decl), @O.as.HasN1.impl [concrete]
// CHECK:STDOUT:   %HasN1.impl_witness: <witness> = impl_witness %HasN1.impl_witness_table [concrete = constants.%HasN1.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .N = %O.as.HasN1.impl.N.decl
// CHECK:STDOUT:   extend %HasN1.ref
// CHECK:STDOUT:   witness = %HasN1.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @O.as.HasN2.impl: %Self.ref as %HasN2.ref {
// CHECK:STDOUT:   %O.as.HasN2.impl.N.decl: %O.as.HasN2.impl.N.type = fn_decl @O.as.HasN2.impl.N [concrete = constants.%O.as.HasN2.impl.N] {} {}
// CHECK:STDOUT:   %HasN2.impl_witness_table = impl_witness_table (%O.as.HasN2.impl.N.decl), @O.as.HasN2.impl [concrete]
// CHECK:STDOUT:   %HasN2.impl_witness: <witness> = impl_witness %HasN2.impl_witness_table [concrete = constants.%HasN2.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .N = %O.as.HasN2.impl.N.decl
// CHECK:STDOUT:   extend %HasN2.ref
// CHECK:STDOUT:   witness = %HasN2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NBase {
// CHECK:STDOUT:   %NBase.N.decl: %NBase.N.type = fn_decl @NBase.N [concrete = constants.%NBase.N] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NBase
// CHECK:STDOUT:   .N = %NBase.N.decl
// CHECK:STDOUT:   .HasN1 = <poisoned>
// CHECK:STDOUT:   .HasN2 = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @O {
// CHECK:STDOUT:   %NBase.ref: type = name_ref NBase, file.%NBase.decl [concrete = constants.%NBase]
// CHECK:STDOUT:   %.loc17: %O.elem = base_decl %NBase.ref, element0 [concrete]
// CHECK:STDOUT:   impl_decl @O.as.HasN1.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%O [concrete = constants.%O]
// CHECK:STDOUT:     %HasN1.ref: type = name_ref HasN1, file.%HasN1.decl [concrete = constants.%HasN1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <witness> = impl_self_witness @O.as.HasN1.impl.%Self.ref, @HasN1 [concrete = constants.%.665]
// CHECK:STDOUT:   impl_decl @O.as.HasN2.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%O [concrete = constants.%O]
// CHECK:STDOUT:     %HasN2.ref: type = name_ref HasN2, file.%HasN2.decl [concrete = constants.%HasN2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21: <witness> = impl_self_witness @O.as.HasN2.impl.%Self.ref, @HasN2 [concrete = constants.%.81b]
// CHECK:STDOUT:   %O.N.decl: %O.N.type = fn_decl @O.N [concrete = constants.%O.N] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.40f [concrete = constants.%complete_type.3b3]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%O
// CHECK:STDOUT:   .NBase = <poisoned>
// CHECK:STDOUT:   .base = %.loc17
// CHECK:STDOUT:   .HasN1 = <poisoned>
// CHECK:STDOUT:   .HasN2 = <poisoned>
// CHECK:STDOUT:   .N = %O.N.decl
// CHECK:STDOUT:   extend %NBase.ref
// CHECK:STDOUT:   extend @O.as.HasN1.impl.%HasN1.ref
// CHECK:STDOUT:   extend @O.as.HasN2.impl.%HasN2.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NBase.N() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasN1.WithSelf.N(@HasN1.%Self: %HasN1.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasN2.WithSelf.N(@HasN2.%Self: %HasN2.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @O.as.HasN1.impl.N() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @O.as.HasN2.impl.N() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @O.N();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @P(%o.param: %O) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %O.ref.loc28: type = name_ref O, file.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %N.ref.loc28: %O.N.type = name_ref N, @O.%O.N.decl [concrete = constants.%O.N]
// CHECK:STDOUT:   %O.N.call.loc28: init %empty_tuple.type = call %N.ref.loc28()
// CHECK:STDOUT:   %o.ref: %O = name_ref o, %o
// CHECK:STDOUT:   %N.ref.loc29: %O.N.type = name_ref N, @O.%O.N.decl [concrete = constants.%O.N]
// CHECK:STDOUT:   %O.N.call.loc29: init %empty_tuple.type = call %N.ref.loc29()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN1.WithSelf(constants.%Self.97e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.97e
// CHECK:STDOUT:   %HasN1.WithSelf.N.type => constants.%HasN1.WithSelf.N.type.5e2
// CHECK:STDOUT:   %HasN1.WithSelf.N => constants.%HasN1.WithSelf.N.9d7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN1.WithSelf.N(constants.%Self.97e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN2.WithSelf(constants.%Self.e42) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.e42
// CHECK:STDOUT:   %HasN2.WithSelf.N.type => constants.%HasN2.WithSelf.N.type.8b8
// CHECK:STDOUT:   %HasN2.WithSelf.N => constants.%HasN2.WithSelf.N.2b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN2.WithSelf.N(constants.%Self.e42) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN1.WithSelf(constants.%HasN1.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasN1.facet
// CHECK:STDOUT:   %HasN1.WithSelf.N.type => constants.%HasN1.WithSelf.N.type.296
// CHECK:STDOUT:   %HasN1.WithSelf.N => constants.%HasN1.WithSelf.N.b63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN1.WithSelf.N(constants.%HasN1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN2.WithSelf(constants.%HasN2.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%HasN2.facet
// CHECK:STDOUT:   %HasN2.WithSelf.N.type => constants.%HasN2.WithSelf.N.type.88c
// CHECK:STDOUT:   %HasN2.WithSelf.N => constants.%HasN2.WithSelf.N.306
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN2.WithSelf.N(constants.%HasN2.facet) {}
// CHECK:STDOUT:
