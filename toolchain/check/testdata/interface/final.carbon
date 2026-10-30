// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/final.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/final.carbon

// --- require_specific_impl_definition.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";

interface I { fn F(); }
interface J {}

final impl forall [T: J] T as I { fn F() {} }

// This is creating an indirection where the `final impl` requires an indirect
// instantiation when the caller forms the specific `UseI(C)`.
fn UseI(generic T: I) { T.F(); }
fn UseJ(generic T: J) { UseI(T); }

class C {}
impl C as J {}
fn Use() { UseJ(C); }

// --- require_specific_impl_definition_from_symbolic_impl_in_class_definition.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";

interface I { fn F(); }
interface J {}

final impl forall [T: J] T as I { fn F() {} }

class B(U: I) {
  alias B1 = U;
}

class C(V: J) {
  // Does `V as I` which finds a symbolic ImplWitness. This adds a dependency
  // edge on the `T as I` impl's specific definition with a specific arg of `V`.
  alias C1 = B(V).B1;
}

class D {}
impl D as J {}
fn Use() {
  // Forms `C(D)` specific which also makes the ImplWitness concrete We now
  // have a dependency on the `T as I` impl with a specific arg of `D`. Forming
  // this specific must resolve the specific definition of the `T as I` impl
  // before we try to access `F` in it.
  alias C1 = C(D).C1;
  // Access `F` from the `T as I` impl with a specific arg of `D`. Compound member
  // access avoids performing a fresh lookup which would give an extra chance for
  // the specific definition to be resolved.
  C1.(I.F)();
}

// --- require_specific_impl_definition_from_lookup_impl_witness_in_class_definition.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";

interface I { fn F(); }
interface J {}

final impl forall [T: J] T as I { fn F() {} }

class C(U: I) {
  // This puts a LookupImplWitness for `U as I` in the eval block of C.
  alias C1 = U.F;
}

fn UseI(generic T: I) { C(T).C1(); }
fn UseJ(generic T: J) { UseI(T); }

class D {}
impl D as J {}
fn Use() {
  UseJ(D);
}

// --- require_specific_impl_definition_through_specific_instantiation.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";

interface L { let L1: type; }
interface M {}
final impl forall [T: M] T as L where .L1 = () {}

interface Y { let Y1: L; }
class C(U: Y) {
  // This puts a LookupImplWitness for `U.Y1 as L` in the eval block of C.
  alias C1 = U.Y1.L1;
}

fn F(generic T: Y where .Y1 impls M) {
  // Forms a specific of C whose definition must be resolved in order to
  // perform member access to C1. The resolved specific evaluates a modified
  // LookupImplWitness (`U` substed with `T`) and finds a `final impl` for L.
  //
  // A dependency on a specific impl is a dependency on the specific impl's
  // definition. We used to create a new instruction at the time we found the
  // `final impl` to encode that dependency. But that instruction would end up
  // in the eval block of `F` which is incorrect. The instruction should be
  // part of the eval block of `C`, so that it can be part of the specific
  // definition block of `C(T)`, where the relationship is actually
  // established.
  C(T).C1;
}

// --- require_specific_impl_definition_from_inside_impl_definition.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";

interface I;

class B(U: I) {
  alias B1 = U;
}

interface J {}
// Allow the decl of `B(V)` where `V impls J`.
final impl forall [T: J] T as I;

interface I {
  fn F();
  fn G(generic V: J) -> B(V);
}

final impl forall [T: J] T as I {
  fn F() {}
  // The return type `B(V)` forms an ImplWitness of `V as I` which refers to
  // the impl currently being defined. This requires a dependency edge from
  // the declaration of G to the specific `T as I` ImplDecl definition with a
  // symbolic argument of `V`.
  fn G(generic V: J) -> B(V) {
    return {};
  }
}

// Used to call `G` in order to form the specific `B(D)`.
class C {}
impl C as J {}

class D {}
impl D as J {}

fn Use() {
  // This calls `G(D)` which forms the specific `B(D)` for the return type.
  // That does a `D as I` lookup to form a FacetValue with an ImplWitness
  // referring to the specific `T as I` ImplDecl with argument `D`. The alias
  // is the FacetValue holding that ImplWitness.
  //
  // The decl of G should have declared a dependency edge on the specific
  // ImplDecl such that forming this ImplWitness resolves the `T as I`
  // specific definition with its argument `D`.
  alias D_As_I = C.(I.G)(D).B1;
  // This accesses `F` from the witness table using the `ImplWitness` from
  // the return type of `G(D)`. That needs the specific definition referred
  // to by the `ImplWitness` to be resolved.
  D_As_I.(I.F)();
}

// CHECK:STDOUT: --- require_specific_impl_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.de2: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.20b: %I.WithSelf.F.type.de2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.653: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.2c5: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.patt.ac1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.e40: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.d90: type = facet_access_type %T.e40 [symbolic]
// CHECK:STDOUT:   %.6ac: <witness> = impl_self_witness %T.e40, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness.e22: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%T.e40) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.898: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T.e40) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.fed: %T.as_type.as.I.impl.F.type.898 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet.297: %I.type = facet_value %T.as_type.d90, (%I.impl_witness.e22) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.492: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.297) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.df6: %I.WithSelf.F.type.492 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt.1df: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.dda: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %UseI.type: type = fn_type @UseI [concrete]
// CHECK:STDOUT:   %UseI: %UseI.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.eae: type = fn_type @I.WithSelf.F, @I.WithSelf(%T.dda) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.4b1: %I.WithSelf.F.type.eae = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.b38: type = facet_access_type %T.dda [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.dda, @I [symbolic]
// CHECK:STDOUT:   %.b94: type = fn_type_with_self_type %I.WithSelf.F.type.eae, %T.dda [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.b94 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.F(%T.dda) [symbolic]
// CHECK:STDOUT:   %UseJ.type: type = fn_type @UseJ [concrete]
// CHECK:STDOUT:   %UseJ: %UseJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseI.specific_fn.95d: <specific function> = specific_function %UseI, @UseI(%I.facet.297) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.ffd: <witness> = impl_self_witness %C, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %C, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseJ.specific_fn: <specific function> = specific_function %UseJ, @UseJ(%J.facet) [concrete]
// CHECK:STDOUT:   %.d7c: type = fn_type_with_self_type %I.WithSelf.F.type.492, %I.facet.297 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.specific_fn.2f7: <specific function> = specific_function %T.as_type.as.I.impl.F.fed, @T.as_type.as.I.impl.F(%T.e40) [symbolic]
// CHECK:STDOUT:   %.b7f: <witness> = impl_self_witness %C, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.2ce: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %I.facet.c8e: %I.type = facet_value %C, (%I.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %UseI.specific_fn.0ef: <specific function> = specific_function %UseI, @UseI(%I.facet.c8e) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.151: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.c8e) [concrete]
// CHECK:STDOUT:   %.e21: type = fn_type_with_self_type %I.WithSelf.F.type.151, %I.facet.c8e [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.ad4: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.9a9: %T.as_type.as.I.impl.F.type.ad4 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.specific_fn.8c5: <specific function> = specific_function %T.as_type.as.I.impl.F.9a9, @T.as_type.as.I.impl.F(%J.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .UseI = %UseI.decl
// CHECK:STDOUT:     .UseJ = %UseJ.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as_type.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_21.1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_21.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc7_21.2 [symbolic = %T.loc7_21.1 (constants.%T.e40)]
// CHECK:STDOUT:     %T.as_type.loc7_26.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %.loc7_26: type = converted %T.ref, %T.as_type.loc7_26.2 [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_23: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_21.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc7_21.1 (constants.%T.e40)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @T.as_type.as.I.impl.%.loc7_26, @I [symbolic = @T.as_type.as.I.impl.%.loc7_33 (constants.%.6ac)]
// CHECK:STDOUT:   %UseI.decl: %UseI.type = fn_decl @UseI [concrete = constants.%UseI] {
// CHECK:STDOUT:     %T.patt.loc11_18.1: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_18.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_20: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_18.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc11_18.1 (constants.%T.dda)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseJ.decl: %UseJ.type = fn_decl @UseJ [concrete = constants.%UseJ] {
// CHECK:STDOUT:     %T.patt.loc12_18.1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_18.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_20: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_18.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_18.1 (constants.%T.e40)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = impl_self_witness @C.as.J.impl.%C.ref, @J [concrete = constants.%.ffd]
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.de2) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.20b)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.653]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic final impl @T.as_type.as.I.impl(%T.loc7_21.2: %J.type) {
// CHECK:STDOUT:   %T.patt.loc7_21.2: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_21.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   %T.loc7_21.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc7_21.1 (constants.%T.e40)]
// CHECK:STDOUT:   %T.as_type.loc7_26.1: type = facet_access_type %T.loc7_21.1 [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type.d90)]
// CHECK:STDOUT:   %.loc7_33: <witness> = impl_self_witness %T.loc7_21.1, @I [symbolic = %.loc7_33 (constants.%.6ac)]
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(%T.loc7_21.1) [symbolic = %I.impl_witness.loc7_33.2 (constants.%I.impl_witness.e22)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T.loc7_21.1) [symbolic = %T.as_type.as.I.impl.F.type (constants.%T.as_type.as.I.impl.F.type.898)]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.898) = struct_value () [symbolic = %T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.fed)]
// CHECK:STDOUT:
// CHECK:STDOUT:   final impl: %.loc7_26 as %I.ref {
// CHECK:STDOUT:     %T.as_type.as.I.impl.F.decl: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.898) = fn_decl @T.as_type.as.I.impl.F [symbolic = @T.as_type.as.I.impl.%T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.fed)] {} {}
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%T.as_type.as.I.impl.F.decl), @T.as_type.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_33.1: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(constants.%T.e40) [symbolic = %I.impl_witness.loc7_33.2 (constants.%I.impl_witness.e22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as_type.as.I.impl.F.decl
// CHECK:STDOUT:     extend %I.ref
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_33.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref as %J.ref {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
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
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as_type.as.I.impl.F(@T.as_type.as.I.impl.%T.loc7_21.2: %J.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseI(%T.loc11_18.2: %I.type) {
// CHECK:STDOUT:   %T.patt.loc11_18.2: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_18.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:   %T.loc11_18.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc11_18.1 (constants.%T.dda)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc11_26.2: type = facet_access_type %T.loc11_18.1 [symbolic = %T.as_type.loc11_26.2 (constants.%T.as_type.b38)]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%T.loc11_18.1) [symbolic = %I.WithSelf.F.type (constants.%I.WithSelf.F.type.eae)]
// CHECK:STDOUT:   %.loc11_26.2: type = fn_type_with_self_type %I.WithSelf.F.type, %T.loc11_18.1 [symbolic = %.loc11_26.2 (constants.%.b94)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc11_18.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc11_26.2: @UseI.%.loc11_26.2 (%.b94) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_26.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_26.2: <specific function> = specific_impl_function %impl.elem0.loc11_26.2, @I.WithSelf.F(%T.loc11_18.1) [symbolic = %specific_impl_fn.loc11_26.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc11_18.2 [symbolic = %T.loc11_18.1 (constants.%T.dda)]
// CHECK:STDOUT:     %T.as_type.loc11_26.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_26.2 (constants.%T.as_type.b38)]
// CHECK:STDOUT:     %.loc11_26.1: type = converted %T.ref, %T.as_type.loc11_26.1 [symbolic = %T.as_type.loc11_26.2 (constants.%T.as_type.b38)]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc11_26.1: @UseI.%.loc11_26.2 (%.b94) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_26.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc11_26.1: <specific function> = specific_impl_function %impl.elem0.loc11_26.1, @I.WithSelf.F(constants.%T.dda) [symbolic = %specific_impl_fn.loc11_26.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %I.WithSelf.F.call: init %empty_tuple.type = call %specific_impl_fn.loc11_26.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseJ(%T.loc12_18.2: %J.type) {
// CHECK:STDOUT:   %T.patt.loc12_18.2: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_18.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   %T.loc12_18.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_18.1 (constants.%T.e40)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc12_31.2: type = facet_access_type %T.loc12_18.1 [symbolic = %T.as_type.loc12_31.2 (constants.%T.as_type.d90)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%T.loc12_18.1) [symbolic = %I.impl_witness (constants.%I.impl_witness.e22)]
// CHECK:STDOUT:   %I.facet.loc12_31.2: %I.type = facet_value %T.as_type.loc12_31.2, (%I.impl_witness) [symbolic = %I.facet.loc12_31.2 (constants.%I.facet.297)]
// CHECK:STDOUT:   %UseI.specific_fn.loc12_25.2: <specific function> = specific_function constants.%UseI, @UseI(%I.facet.loc12_31.2) [symbolic = %UseI.specific_fn.loc12_25.2 (constants.%UseI.specific_fn.95d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %UseI.ref: %UseI.type = name_ref UseI, file.%UseI.decl [concrete = constants.%UseI]
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc12_18.2 [symbolic = %T.loc12_18.1 (constants.%T.e40)]
// CHECK:STDOUT:     %T.as_type.loc12_31.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc12_31.2 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %I.facet.loc12_31.1: %I.type = facet_value %T.as_type.loc12_31.1, (constants.%I.impl_witness.e22) [symbolic = %I.facet.loc12_31.2 (constants.%I.facet.297)]
// CHECK:STDOUT:     %.loc12_31: %I.type = converted %T.ref, %I.facet.loc12_31.1 [symbolic = %I.facet.loc12_31.2 (constants.%I.facet.297)]
// CHECK:STDOUT:     %UseI.specific_fn.loc12_25.1: <specific function> = specific_function %UseI.ref, @UseI(constants.%I.facet.297) [symbolic = %UseI.specific_fn.loc12_25.2 (constants.%UseI.specific_fn.95d)]
// CHECK:STDOUT:     %UseI.call: init %empty_tuple.type = call %UseI.specific_fn.loc12_25.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %UseJ.ref: %UseJ.type = name_ref UseJ, file.%UseJ.decl [concrete = constants.%UseJ]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %C.ref, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc16: %J.type = converted %C.ref, %J.facet [concrete = constants.%J.facet]
// CHECK:STDOUT:   %UseJ.specific_fn: <specific function> = specific_function %UseJ.ref, @UseJ(constants.%J.facet) [concrete = constants.%UseJ.specific_fn]
// CHECK:STDOUT:   %UseJ.call: init %empty_tuple.type = call %UseJ.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.de2
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.20b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.653) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%T.e40) {
// CHECK:STDOUT:   %T.patt.loc7_21.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc7_21.1 => constants.%T.e40
// CHECK:STDOUT:   %T.as_type.loc7_26.1 => constants.%T.as_type.d90
// CHECK:STDOUT:   %.loc7_33 => constants.%.6ac
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2 => constants.%I.impl_witness.e22
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.898
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.fed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%T.e40) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.297) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.297
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.492
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.df6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.297) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%T.dda) {
// CHECK:STDOUT:   %T.patt.loc11_18.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc11_18.1 => constants.%T.dda
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.dda) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T.dda
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.eae
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.4b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T.dda) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseJ(constants.%T.e40) {
// CHECK:STDOUT:   %T.patt.loc12_18.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc12_18.1 => constants.%T.e40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%I.facet.297) {
// CHECK:STDOUT:   %T.patt.loc11_18.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc11_18.1 => constants.%I.facet.297
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc11_26.2 => constants.%T.as_type.d90
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.492
// CHECK:STDOUT:   %.loc11_26.2 => constants.%.d7c
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.e22
// CHECK:STDOUT:   %impl.elem0.loc11_26.2 => constants.%T.as_type.as.I.impl.F.fed
// CHECK:STDOUT:   %specific_impl_fn.loc11_26.2 => constants.%T.as_type.as.I.impl.F.specific_fn.2f7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseJ(constants.%J.facet) {
// CHECK:STDOUT:   %T.patt.loc12_18.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc12_18.1 => constants.%J.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc12_31.2 => constants.%C
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.2ce
// CHECK:STDOUT:   %I.facet.loc12_31.2 => constants.%I.facet.c8e
// CHECK:STDOUT:   %UseI.specific_fn.loc12_25.2 => constants.%UseI.specific_fn.0ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%J.facet) {
// CHECK:STDOUT:   %T.patt.loc7_21.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc7_21.1 => constants.%J.facet
// CHECK:STDOUT:   %T.as_type.loc7_26.1 => constants.%C
// CHECK:STDOUT:   %.loc7_33 => constants.%.b7f
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2 => constants.%I.impl_witness.2ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.ad4
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.9a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%I.facet.c8e) {
// CHECK:STDOUT:   %T.patt.loc11_18.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc11_18.1 => constants.%I.facet.c8e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc11_26.2 => constants.%C
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.151
// CHECK:STDOUT:   %.loc11_26.2 => constants.%.e21
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.2ce
// CHECK:STDOUT:   %impl.elem0.loc11_26.2 => constants.%T.as_type.as.I.impl.F.9a9
// CHECK:STDOUT:   %specific_impl_fn.loc11_26.2 => constants.%T.as_type.as.I.impl.F.specific_fn.8c5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.c8e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.c8e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_specific_impl_definition_from_symbolic_impl_in_class_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.de2: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.20b: %I.WithSelf.F.type.de2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.653: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.2c5: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %.6accd5.1: <witness> = impl_self_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness.e22ed8.1: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.898: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.fed: %T.as_type.as.I.impl.F.type.898 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet.297833.1: %I.type = facet_value %T.as_type, (%I.impl_witness.e22ed8.1) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.492: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.297833.1) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.df6: %I.WithSelf.F.type.492 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.6cb = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: %I.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = parameterized_class_name_type @B [concrete]
// CHECK:STDOUT:   %B.generic: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.348: type = class_type @B, @B(%U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.2c5 = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V: %J.type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.de8: type = class_type @C, @C(%V) [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %.6accd5.2: <witness> = impl_self_witness %V, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness.e22ed8.2: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%V) [symbolic]
// CHECK:STDOUT:   %I.facet.297833.2: %I.type = facet_value %V.as_type, (%I.impl_witness.e22ed8.2) [symbolic]
// CHECK:STDOUT:   %B.14e: type = class_type @B, @B(%I.facet.297833.2) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.b77: <witness> = impl_self_witness %D, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @D.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %D, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.a61: type = class_type @C, @C(%J.facet) [concrete]
// CHECK:STDOUT:   %.393: <witness> = impl_self_witness %D, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.6a0: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %I.facet.a62: %I.type = facet_value %D, (%I.impl_witness.6a0) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.1b8: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.a62) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.183: %I.WithSelf.F.type.1b8 = struct_value () [concrete]
// CHECK:STDOUT:   %.97a: type = fn_type_with_self_type %I.WithSelf.F.type.1b8, %I.facet.a62 [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.1de: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.ec8: %T.as_type.as.I.impl.F.type.1de = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.specific_fn: <specific function> = specific_function %T.as_type.as.I.impl.F.ec8, @T.as_type.as.I.impl.F(%J.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as_type.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_21.1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc7_21.2 [symbolic = %T.loc7_21.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc7_26.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc7_26: type = converted %T.ref, %T.as_type.loc7_26.2 [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_23: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_21.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc7_21.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @T.as_type.as.I.impl.%.loc7_26, @I [symbolic = @T.as_type.as.I.impl.%.loc7_33 (constants.%.6accd5.1)]
// CHECK:STDOUT:   %B.decl: %B.type = class_decl @B [concrete = constants.%B.generic] {
// CHECK:STDOUT:     %U.patt.loc9_10.1: %pattern_type.6cb = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc9_10.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc9_10.2: %I.type = symbolic_binding U, 0 [symbolic = %U.loc9_10.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %V.patt.loc13_10.1: %pattern_type.2c5 = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_10.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc13_10.2: %J.type = symbolic_binding V, 0 [symbolic = %V.loc13_10.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @D.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20: <witness> = impl_self_witness @D.as.J.impl.%D.ref, @J [concrete = constants.%.b77]
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.de2) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.20b)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.653]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic final impl @T.as_type.as.I.impl(%T.loc7_21.2: %J.type) {
// CHECK:STDOUT:   %T.patt.loc7_21.2: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_21.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc7_21.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc7_26.1: type = facet_access_type %T.loc7_21.1 [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc7_33: <witness> = impl_self_witness %T.loc7_21.1, @I [symbolic = %.loc7_33 (constants.%.6accd5.1)]
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(%T.loc7_21.1) [symbolic = %I.impl_witness.loc7_33.2 (constants.%I.impl_witness.e22ed8.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T.loc7_21.1) [symbolic = %T.as_type.as.I.impl.F.type (constants.%T.as_type.as.I.impl.F.type.898)]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.898) = struct_value () [symbolic = %T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.fed)]
// CHECK:STDOUT:
// CHECK:STDOUT:   final impl: %.loc7_26 as %I.ref {
// CHECK:STDOUT:     %T.as_type.as.I.impl.F.decl: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.898) = fn_decl @T.as_type.as.I.impl.F [symbolic = @T.as_type.as.I.impl.%T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.fed)] {} {}
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%T.as_type.as.I.impl.F.decl), @T.as_type.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_33.1: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(constants.%T) [symbolic = %I.impl_witness.loc7_33.2 (constants.%I.impl_witness.e22ed8.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as_type.as.I.impl.F.decl
// CHECK:STDOUT:     extend %I.ref
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_33.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.J.impl: %D.ref as %J.ref {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @D.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B(%U.loc9_10.2: %I.type) {
// CHECK:STDOUT:   %U.patt.loc9_10.2: %pattern_type.6cb = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc9_10.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc9_10.1: %I.type = symbolic_binding U, 0 [symbolic = %U.loc9_10.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc10: %I.type = specific_constant %U.loc9_10.2, @B(constants.%U) [symbolic = constants.%U]
// CHECK:STDOUT:     %U.ref: %I.type = name_ref U, %.loc10 [symbolic = constants.%U]
// CHECK:STDOUT:     %B1: %I.type = alias_binding B1, %U.ref [symbolic = %U.loc9_10.1 (constants.%U)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%B.348
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .B1 = %B1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%V.loc13_10.2: %J.type) {
// CHECK:STDOUT:   %V.patt.loc13_10.2: %pattern_type.2c5 = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_10.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc13_10.1: %J.type = symbolic_binding V, 0 [symbolic = %V.loc13_10.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %V.as_type.loc16_9: type = facet_access_type %V.loc13_10.1 [symbolic = %V.as_type.loc16_9 (constants.%V.as_type)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%V.loc13_10.1) [symbolic = %I.impl_witness (constants.%I.impl_witness.e22ed8.2)]
// CHECK:STDOUT:   %I.facet.loc16_9: %I.type = facet_value %V.as_type.loc16_9, (%I.impl_witness) [symbolic = %I.facet.loc16_9 (constants.%I.facet.297833.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %.loc16_16: %J.type = specific_constant %V.loc13_10.2, @C(constants.%V) [symbolic = constants.%V]
// CHECK:STDOUT:     %V.ref: %J.type = name_ref V, %.loc16_16 [symbolic = constants.%V]
// CHECK:STDOUT:     %V.as_type.loc16_17: type = facet_access_type %V.ref [symbolic = constants.%V.as_type]
// CHECK:STDOUT:     %I.facet.loc16_17: %I.type = facet_value %V.as_type.loc16_17, (constants.%I.impl_witness.e22ed8.2) [symbolic = constants.%I.facet.297833.2]
// CHECK:STDOUT:     %.loc16_17: %I.type = converted %V.ref, %I.facet.loc16_17 [symbolic = constants.%I.facet.297833.2]
// CHECK:STDOUT:     %B: type = class_type @B, @B(constants.%I.facet.297833.2) [symbolic = constants.%B.14e]
// CHECK:STDOUT:     %.loc16_18: %I.type = specific_constant @B.%B1, @B(constants.%I.facet.297833.2) [symbolic = constants.%I.facet.297833.2]
// CHECK:STDOUT:     %B1.ref: %I.type = name_ref B1, %.loc16_18 [symbolic = constants.%I.facet.297833.2]
// CHECK:STDOUT:     %C1: %I.type = alias_binding C1, %B1.ref [symbolic = %I.facet.loc16_9 (constants.%I.facet.297833.2)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.de8
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .V = <poisoned>
// CHECK:STDOUT:     .C1 = %C1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as_type.as.I.impl.F(@T.as_type.as.I.impl.%T.loc7_21.2: %J.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %D.ref, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc26_17: %J.type = converted %D.ref, %J.facet [concrete = constants.%J.facet]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%J.facet) [concrete = constants.%C.a61]
// CHECK:STDOUT:   %.loc26_18: %I.type = specific_constant @C.%C1, @C(constants.%J.facet) [concrete = constants.%I.facet.a62]
// CHECK:STDOUT:   %C1.ref.loc26: %I.type = name_ref C1, %.loc26_18 [concrete = constants.%I.facet.a62]
// CHECK:STDOUT:   %C1: %I.type = alias_binding C1, %C1.ref.loc26 [concrete = constants.%I.facet.a62]
// CHECK:STDOUT:   %C1.ref.loc30: %I.type = name_ref C1, %C1 [concrete = constants.%I.facet.a62]
// CHECK:STDOUT:   %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.97a = impl_witness_access constants.%I.impl_witness.6a0, element0 [concrete = constants.%T.as_type.as.I.impl.F.ec8]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as_type.as.I.impl.F(constants.%J.facet) [concrete = constants.%T.as_type.as.I.impl.F.specific_fn]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.call: init %empty_tuple.type = call %specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.de2
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.20b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.653) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_21.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_21.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc7_26.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc7_33 => constants.%.6accd5.1
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2 => constants.%I.impl_witness.e22ed8.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.898
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.fed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.297833.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.297833.1
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.492
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.df6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.297833.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc9_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc9_10.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V) {
// CHECK:STDOUT:   %V.patt.loc13_10.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc13_10.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%V) {
// CHECK:STDOUT:   %T.patt.loc7_21.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_21.1 => constants.%V
// CHECK:STDOUT:   %T.as_type.loc7_26.1 => constants.%V.as_type
// CHECK:STDOUT:   %.loc7_33 => constants.%.6accd5.2
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2 => constants.%I.impl_witness.e22ed8.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%I.facet.297833.2) {
// CHECK:STDOUT:   %U.patt.loc9_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc9_10.1 => constants.%I.facet.297833.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%J.facet) {
// CHECK:STDOUT:   %V.patt.loc13_10.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc13_10.1 => constants.%J.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %V.as_type.loc16_9 => constants.%D
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.6a0
// CHECK:STDOUT:   %I.facet.loc16_9 => constants.%I.facet.a62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%J.facet) {
// CHECK:STDOUT:   %T.patt.loc7_21.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_21.1 => constants.%J.facet
// CHECK:STDOUT:   %T.as_type.loc7_26.1 => constants.%D
// CHECK:STDOUT:   %.loc7_33 => constants.%.393
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2 => constants.%I.impl_witness.6a0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.1de
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.ec8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.a62) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.a62
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.1b8
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_specific_impl_definition_from_lookup_impl_witness_in_class_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.de2: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.20b: %I.WithSelf.F.type.de2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.653: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.2c5: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.patt.ac1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.e40: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.d90: type = facet_access_type %T.e40 [symbolic]
// CHECK:STDOUT:   %.6ac: <witness> = impl_self_witness %T.e40, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness.e22: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%T.e40) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.898: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T.e40) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.fed: %T.as_type.as.I.impl.F.type.898 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet.297: %I.type = facet_value %T.as_type.d90, (%I.impl_witness.e22) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.492: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.297) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.df6: %I.WithSelf.F.type.492 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.6cb = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: %I.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f92650.1: type = class_type @C, @C(%U) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.eae64e.1: type = fn_type @I.WithSelf.F, @I.WithSelf(%U) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.4b1: %I.WithSelf.F.type.eae64e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.03bc6d.1: <witness> = lookup_impl_witness %U, @I [symbolic]
// CHECK:STDOUT:   %.b94131.1: type = fn_type_with_self_type %I.WithSelf.F.type.eae64e.1, %U [symbolic]
// CHECK:STDOUT:   %impl.elem0.136579.1: %.b94131.1 = impl_witness_access %I.lookup_impl_witness.03bc6d.1, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.patt.1df: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.dda: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %UseI.type: type = fn_type @UseI [concrete]
// CHECK:STDOUT:   %UseI: %UseI.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f92650.2: type = class_type @C, @C(%T.dda) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.eae64e.2: type = fn_type @I.WithSelf.F, @I.WithSelf(%T.dda) [symbolic]
// CHECK:STDOUT:   %.b94131.2: type = fn_type_with_self_type %I.WithSelf.F.type.eae64e.2, %T.dda [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.03bc6d.2: <witness> = lookup_impl_witness %T.dda, @I [symbolic]
// CHECK:STDOUT:   %impl.elem0.136579.2: %.b94131.2 = impl_witness_access %I.lookup_impl_witness.03bc6d.2, element0 [symbolic]
// CHECK:STDOUT:   %require_complete.40f: <witness> = require_complete_type %C.f92650.2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0.136579.2, @I.WithSelf.F(%T.dda) [symbolic]
// CHECK:STDOUT:   %UseJ.type: type = fn_type @UseJ [concrete]
// CHECK:STDOUT:   %UseJ: %UseJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseI.specific_fn.95d: <specific function> = specific_function %UseI, @UseI(%I.facet.297) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.b77: <witness> = impl_self_witness %D, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @D.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %D, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseJ.specific_fn: <specific function> = specific_function %UseJ, @UseJ(%J.facet) [concrete]
// CHECK:STDOUT:   %C.ac5: type = class_type @C, @C(%I.facet.297) [symbolic]
// CHECK:STDOUT:   %require_complete.c9d: <witness> = require_complete_type %C.ac5 [symbolic]
// CHECK:STDOUT:   %.d7c: type = fn_type_with_self_type %I.WithSelf.F.type.492, %I.facet.297 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.specific_fn.2f7: <specific function> = specific_function %T.as_type.as.I.impl.F.fed, @T.as_type.as.I.impl.F(%T.e40) [symbolic]
// CHECK:STDOUT:   %.393: <witness> = impl_self_witness %D, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.6a0: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %I.facet.a62: %I.type = facet_value %D, (%I.impl_witness.6a0) [concrete]
// CHECK:STDOUT:   %UseI.specific_fn.033: <specific function> = specific_function %UseI, @UseI(%I.facet.a62) [concrete]
// CHECK:STDOUT:   %C.c11: type = class_type @C, @C(%I.facet.a62) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.1b8: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.a62) [concrete]
// CHECK:STDOUT:   %.97a: type = fn_type_with_self_type %I.WithSelf.F.type.1b8, %I.facet.a62 [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.1de: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.ec8: %T.as_type.as.I.impl.F.type.1de = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.specific_fn.c71: <specific function> = specific_function %T.as_type.as.I.impl.F.ec8, @T.as_type.as.I.impl.F(%J.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .UseI = %UseI.decl
// CHECK:STDOUT:     .UseJ = %UseJ.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as_type.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_21.1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_21.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc7_21.2 [symbolic = %T.loc7_21.1 (constants.%T.e40)]
// CHECK:STDOUT:     %T.as_type.loc7_26.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %.loc7_26: type = converted %T.ref, %T.as_type.loc7_26.2 [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_23: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_21.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc7_21.1 (constants.%T.e40)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @T.as_type.as.I.impl.%.loc7_26, @I [symbolic = @T.as_type.as.I.impl.%.loc7_33 (constants.%.6ac)]
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %U.patt.loc9_10.1: %pattern_type.6cb = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc9_10.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc9_10.2: %I.type = symbolic_binding U, 0 [symbolic = %U.loc9_10.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseI.decl: %UseI.type = fn_decl @UseI [concrete = constants.%UseI] {
// CHECK:STDOUT:     %T.patt.loc14_18.1: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_18.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_20: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_18.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc14_18.1 (constants.%T.dda)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseJ.decl: %UseJ.type = fn_decl @UseJ [concrete = constants.%UseJ] {
// CHECK:STDOUT:     %T.patt.loc15_18.1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_18.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_20: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_18.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc15_18.1 (constants.%T.e40)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @D.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <witness> = impl_self_witness @D.as.J.impl.%D.ref, @J [concrete = constants.%.b77]
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.de2) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.20b)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.653]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic final impl @T.as_type.as.I.impl(%T.loc7_21.2: %J.type) {
// CHECK:STDOUT:   %T.patt.loc7_21.2: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_21.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   %T.loc7_21.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc7_21.1 (constants.%T.e40)]
// CHECK:STDOUT:   %T.as_type.loc7_26.1: type = facet_access_type %T.loc7_21.1 [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type.d90)]
// CHECK:STDOUT:   %.loc7_33: <witness> = impl_self_witness %T.loc7_21.1, @I [symbolic = %.loc7_33 (constants.%.6ac)]
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(%T.loc7_21.1) [symbolic = %I.impl_witness.loc7_33.2 (constants.%I.impl_witness.e22)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T.loc7_21.1) [symbolic = %T.as_type.as.I.impl.F.type (constants.%T.as_type.as.I.impl.F.type.898)]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.898) = struct_value () [symbolic = %T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.fed)]
// CHECK:STDOUT:
// CHECK:STDOUT:   final impl: %.loc7_26 as %I.ref {
// CHECK:STDOUT:     %T.as_type.as.I.impl.F.decl: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.898) = fn_decl @T.as_type.as.I.impl.F [symbolic = @T.as_type.as.I.impl.%T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.fed)] {} {}
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%T.as_type.as.I.impl.F.decl), @T.as_type.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_33.1: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(constants.%T.e40) [symbolic = %I.impl_witness.loc7_33.2 (constants.%I.impl_witness.e22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as_type.as.I.impl.F.decl
// CHECK:STDOUT:     extend %I.ref
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_33.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.J.impl: %D.ref as %J.ref {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @D.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%U.loc9_10.2: %I.type) {
// CHECK:STDOUT:   %U.patt.loc9_10.2: %pattern_type.6cb = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc9_10.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc9_10.1: %I.type = symbolic_binding U, 0 [symbolic = %U.loc9_10.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%U.loc9_10.1) [symbolic = %I.WithSelf.F.type (constants.%I.WithSelf.F.type.eae64e.1)]
// CHECK:STDOUT:   %.loc11_9: type = fn_type_with_self_type %I.WithSelf.F.type, %U.loc9_10.1 [symbolic = %.loc11_9 (constants.%.b94131.1)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc9_10.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.03bc6d.1)]
// CHECK:STDOUT:   %impl.elem0.loc11_9: @C.%.loc11_9 (%.b94131.1) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_9 (constants.%impl.elem0.136579.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc11_14: %I.type = specific_constant %U.loc9_10.2, @C(constants.%U) [symbolic = constants.%U]
// CHECK:STDOUT:     %U.ref: %I.type = name_ref U, %.loc11_14 [symbolic = constants.%U]
// CHECK:STDOUT:     %U.as_type: type = facet_access_type %U.ref [symbolic = constants.%U.as_type]
// CHECK:STDOUT:     %.loc11_15: type = converted %U.ref, %U.as_type [symbolic = constants.%U.as_type]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc11_15: %.b94131.1 = impl_witness_access constants.%I.lookup_impl_witness.03bc6d.1, element0 [symbolic = constants.%impl.elem0.136579.1]
// CHECK:STDOUT:     %C1: @C.%.loc11_9 (%.b94131.1) = alias_binding C1, %impl.elem0.loc11_15 [symbolic = %impl.elem0.loc11_9 (constants.%impl.elem0.136579.1)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f92650.1
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .C1 = %C1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as_type.as.I.impl.F(@T.as_type.as.I.impl.%T.loc7_21.2: %J.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseI(%T.loc14_18.2: %I.type) {
// CHECK:STDOUT:   %T.patt.loc14_18.2: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_18.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:   %T.loc14_18.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc14_18.1 (constants.%T.dda)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc14_28.2: type = class_type @C, @C(%T.loc14_18.1) [symbolic = %C.loc14_28.2 (constants.%C.f92650.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc14_28.2 [symbolic = %require_complete (constants.%require_complete.40f)]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%T.loc14_18.1) [symbolic = %I.WithSelf.F.type (constants.%I.WithSelf.F.type.eae64e.2)]
// CHECK:STDOUT:   %.loc14_29.2: type = fn_type_with_self_type %I.WithSelf.F.type, %T.loc14_18.1 [symbolic = %.loc14_29.2 (constants.%.b94131.2)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc14_18.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.03bc6d.2)]
// CHECK:STDOUT:   %impl.elem0: @UseI.%.loc14_29.2 (%.b94131.2) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.136579.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_29.2: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.F(%T.loc14_18.1) [symbolic = %specific_impl_fn.loc14_29.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc14_18.2 [symbolic = %T.loc14_18.1 (constants.%T.dda)]
// CHECK:STDOUT:     %C.loc14_28.1: type = class_type @C, @C(constants.%T.dda) [symbolic = %C.loc14_28.2 (constants.%C.f92650.2)]
// CHECK:STDOUT:     %.loc14_29.1: @UseI.%.loc14_29.2 (%.b94131.2) = specific_constant @C.%C1, @C(constants.%T.dda) [symbolic = %impl.elem0 (constants.%impl.elem0.136579.2)]
// CHECK:STDOUT:     %C1.ref: @UseI.%.loc14_29.2 (%.b94131.2) = name_ref C1, %.loc14_29.1 [symbolic = %impl.elem0 (constants.%impl.elem0.136579.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_29.1: <specific function> = specific_impl_function %C1.ref, @I.WithSelf.F(constants.%T.dda) [symbolic = %specific_impl_fn.loc14_29.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %I.WithSelf.F.call: init %empty_tuple.type = call %specific_impl_fn.loc14_29.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseJ(%T.loc15_18.2: %J.type) {
// CHECK:STDOUT:   %T.patt.loc15_18.2: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_18.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   %T.loc15_18.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc15_18.1 (constants.%T.e40)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc15_31.2: type = facet_access_type %T.loc15_18.1 [symbolic = %T.as_type.loc15_31.2 (constants.%T.as_type.d90)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%T.loc15_18.1) [symbolic = %I.impl_witness (constants.%I.impl_witness.e22)]
// CHECK:STDOUT:   %I.facet.loc15_31.2: %I.type = facet_value %T.as_type.loc15_31.2, (%I.impl_witness) [symbolic = %I.facet.loc15_31.2 (constants.%I.facet.297)]
// CHECK:STDOUT:   %UseI.specific_fn.loc15_25.2: <specific function> = specific_function constants.%UseI, @UseI(%I.facet.loc15_31.2) [symbolic = %UseI.specific_fn.loc15_25.2 (constants.%UseI.specific_fn.95d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %UseI.ref: %UseI.type = name_ref UseI, file.%UseI.decl [concrete = constants.%UseI]
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc15_18.2 [symbolic = %T.loc15_18.1 (constants.%T.e40)]
// CHECK:STDOUT:     %T.as_type.loc15_31.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc15_31.2 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %I.facet.loc15_31.1: %I.type = facet_value %T.as_type.loc15_31.1, (constants.%I.impl_witness.e22) [symbolic = %I.facet.loc15_31.2 (constants.%I.facet.297)]
// CHECK:STDOUT:     %.loc15_31: %I.type = converted %T.ref, %I.facet.loc15_31.1 [symbolic = %I.facet.loc15_31.2 (constants.%I.facet.297)]
// CHECK:STDOUT:     %UseI.specific_fn.loc15_25.1: <specific function> = specific_function %UseI.ref, @UseI(constants.%I.facet.297) [symbolic = %UseI.specific_fn.loc15_25.2 (constants.%UseI.specific_fn.95d)]
// CHECK:STDOUT:     %UseI.call: init %empty_tuple.type = call %UseI.specific_fn.loc15_25.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %UseJ.ref: %UseJ.type = name_ref UseJ, file.%UseJ.decl [concrete = constants.%UseJ]
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %D.ref, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc20: %J.type = converted %D.ref, %J.facet [concrete = constants.%J.facet]
// CHECK:STDOUT:   %UseJ.specific_fn: <specific function> = specific_function %UseJ.ref, @UseJ(constants.%J.facet) [concrete = constants.%UseJ.specific_fn]
// CHECK:STDOUT:   %UseJ.call: init %empty_tuple.type = call %UseJ.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.de2
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.20b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.653) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%T.e40) {
// CHECK:STDOUT:   %T.patt.loc7_21.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc7_21.1 => constants.%T.e40
// CHECK:STDOUT:   %T.as_type.loc7_26.1 => constants.%T.as_type.d90
// CHECK:STDOUT:   %.loc7_33 => constants.%.6ac
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2 => constants.%I.impl_witness.e22
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.898
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.fed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%T.e40) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.297) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.297
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.492
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.df6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.297) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc9_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc9_10.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%U) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%U
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.eae64e.1
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.4b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%T.dda) {
// CHECK:STDOUT:   %T.patt.loc14_18.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc14_18.1 => constants.%T.dda
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.dda) {
// CHECK:STDOUT:   %U.patt.loc9_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc9_10.1 => constants.%T.dda
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.eae64e.2
// CHECK:STDOUT:   %.loc11_9 => constants.%.b94131.2
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.03bc6d.2
// CHECK:STDOUT:   %impl.elem0.loc11_9 => constants.%impl.elem0.136579.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.dda) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T.dda) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseJ(constants.%T.e40) {
// CHECK:STDOUT:   %T.patt.loc15_18.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc15_18.1 => constants.%T.e40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%I.facet.297) {
// CHECK:STDOUT:   %T.patt.loc14_18.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc14_18.1 => constants.%I.facet.297
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc14_28.2 => constants.%C.ac5
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c9d
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.492
// CHECK:STDOUT:   %.loc14_29.2 => constants.%.d7c
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.e22
// CHECK:STDOUT:   %impl.elem0 => constants.%T.as_type.as.I.impl.F.fed
// CHECK:STDOUT:   %specific_impl_fn.loc14_29.2 => constants.%T.as_type.as.I.impl.F.specific_fn.2f7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseJ(constants.%J.facet) {
// CHECK:STDOUT:   %T.patt.loc15_18.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc15_18.1 => constants.%J.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc15_31.2 => constants.%D
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.6a0
// CHECK:STDOUT:   %I.facet.loc15_31.2 => constants.%I.facet.a62
// CHECK:STDOUT:   %UseI.specific_fn.loc15_25.2 => constants.%UseI.specific_fn.033
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.facet.297) {
// CHECK:STDOUT:   %U.patt.loc9_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc9_10.1 => constants.%I.facet.297
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%J.facet) {
// CHECK:STDOUT:   %T.patt.loc7_21.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc7_21.1 => constants.%J.facet
// CHECK:STDOUT:   %T.as_type.loc7_26.1 => constants.%D
// CHECK:STDOUT:   %.loc7_33 => constants.%.393
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2 => constants.%I.impl_witness.6a0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.1de
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.ec8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%I.facet.a62) {
// CHECK:STDOUT:   %T.patt.loc14_18.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc14_18.1 => constants.%I.facet.a62
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc14_28.2 => constants.%C.c11
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.1b8
// CHECK:STDOUT:   %.loc14_29.2 => constants.%.97a
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.6a0
// CHECK:STDOUT:   %impl.elem0 => constants.%T.as_type.as.I.impl.F.ec8
// CHECK:STDOUT:   %specific_impl_fn.loc14_29.2 => constants.%T.as_type.as.I.impl.F.specific_fn.c71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.facet.a62) {
// CHECK:STDOUT:   %U.patt.loc9_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc9_10.1 => constants.%I.facet.a62
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.1b8
// CHECK:STDOUT:   %.loc11_9 => constants.%.97a
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.6a0
// CHECK:STDOUT:   %impl.elem0.loc11_9 => constants.%T.as_type.as.I.impl.F.ec8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.a62) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.a62) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_specific_impl_definition_through_specific_instantiation.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %Self.480: %L.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %L.assoc_type: type = assoc_entity_type @L [concrete]
// CHECK:STDOUT:   %assoc0.246: %L.assoc_type = assoc_entity element0, @L.WithSelf.%L1 [concrete]
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self.c16: %M.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.1af: type = pattern_type %M.type [concrete]
// CHECK:STDOUT:   %T.patt.3fe: %pattern_type.1af = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.5fb: %M.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.af8: type = facet_access_type %T.5fb [symbolic]
// CHECK:STDOUT:   %.Self.frozen.f7f: %L.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type.8c7: type = facet_access_type %.Self.frozen.f7f [symbolic_self]
// CHECK:STDOUT:   %.63f3: <witness> = impl_self_witness %.Self.frozen.f7f, @L [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2e9: type = impl_witness_access %.63f3, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.866: %L.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.300: <witness> = impl_self_witness %.Self.866, @L [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.865: type = impl_witness_access %.300, element0 [symbolic_self]
// CHECK:STDOUT:   %L_where.type.94d: type = facet_type <@L where %impl.elem0.865 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %.63f6: <witness> = impl_self_witness %T.5fb, @L [symbolic]
// CHECK:STDOUT:   %L.impl_witness.4ca: <witness> = impl_witness @T.as_type.as.L.impl.%L.impl_witness_table, @T.as_type.as.L.impl(%T.5fb) [symbolic]
// CHECK:STDOUT:   %L.facet.d07: %L.type = facet_value %T.as_type.af8, (%L.impl_witness.4ca) [symbolic]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.765: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0.509: %Y.assoc_type = assoc_entity element0, @Y.WithSelf.%Y1 [concrete]
// CHECK:STDOUT:   %pattern_type.cfb22c.1: type = pattern_type %Y.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.cfb22c.1 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: %Y.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.410: type = class_type @C, @C(%U) [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.6a9071.1: <witness> = lookup_impl_witness %U, @Y [symbolic]
// CHECK:STDOUT:   %impl.elem0.0788a1.1: %L.type = impl_witness_access %Y.lookup_impl_witness.6a9071.1, element0 [symbolic]
// CHECK:STDOUT:   %as_type.529665.1: type = facet_access_type %impl.elem0.0788a1.1 [symbolic]
// CHECK:STDOUT:   %L.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.0788a1.1, @L [symbolic]
// CHECK:STDOUT:   %impl.elem0.226: type = impl_witness_access %L.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.frozen.3a0: %Y.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type.449: type = facet_access_type %.Self.frozen.3a0 [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.94c: <witness> = lookup_impl_witness %.Self.frozen.3a0, @Y [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f59: %L.type = impl_witness_access %Y.lookup_impl_witness.94c, element0 [symbolic_self]
// CHECK:STDOUT:   %as_type.81d: type = facet_access_type %impl.elem0.f59 [symbolic_self]
// CHECK:STDOUT:   %.Self.61d: %Y.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.9c7: <witness> = lookup_impl_witness %.Self.61d, @Y [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e05: %L.type = impl_witness_access %Y.lookup_impl_witness.9c7, element0 [symbolic_self]
// CHECK:STDOUT:   %as_type.702: type = facet_access_type %impl.elem0.e05 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type.bd02cd.1: type = facet_type <@Y where %impl.elem0.e05 impls @M> [concrete]
// CHECK:STDOUT:   %pattern_type.cfb22c.2: type = pattern_type %Y_where.type.bd02cd.1 [concrete]
// CHECK:STDOUT:   %T.patt.dc7: %pattern_type.cfb22c.2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.95b: %Y_where.type.bd02cd.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.917: type = facet_access_type %T.95b [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.6a9071.2: <witness> = lookup_impl_witness %T.95b, @Y [symbolic]
// CHECK:STDOUT:   %impl.elem0.0788a1.2: %L.type = impl_witness_access %Y.lookup_impl_witness.6a9071.2, element0 [symbolic]
// CHECK:STDOUT:   %Y.facet.c69: %Y.type = facet_value %T.as_type.917, (%Y.lookup_impl_witness.6a9071.2) [symbolic]
// CHECK:STDOUT:   %C.cbf: type = class_type @C, @C(%Y.facet.c69) [symbolic]
// CHECK:STDOUT:   %as_type.529665.2: type = facet_access_type %impl.elem0.0788a1.2 [symbolic]
// CHECK:STDOUT:   %M.lookup_impl_witness.07c63d.2: <witness> = lookup_impl_witness %impl.elem0.0788a1.2, @M [symbolic]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %as_type.529665.2, (%M.lookup_impl_witness.07c63d.2) [symbolic]
// CHECK:STDOUT:   %.34b: <witness> = impl_self_witness %impl.elem0.0788a1.2, @L [symbolic]
// CHECK:STDOUT:   %L.impl_witness.d68: <witness> = impl_witness @T.as_type.as.L.impl.%L.impl_witness_table, @T.as_type.as.L.impl(%M.facet) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.cbf [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %L.decl: type = interface_decl @L [concrete = constants.%L.type] {} {}
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as_type.as.L.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc6_21.1: %pattern_type.1af = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_21.2 (constants.%T.patt.3fe)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %M.type = name_ref T, %T.loc6_21.1 [symbolic = %T.loc6_21.2 (constants.%T.5fb)]
// CHECK:STDOUT:     %T.as_type.loc6_26.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_26.2 (constants.%T.as_type.af8)]
// CHECK:STDOUT:     %.loc6_26: type = converted %T.ref, %T.as_type.loc6_26.1 [symbolic = %T.as_type.loc6_26.2 (constants.%T.as_type.af8)]
// CHECK:STDOUT:     %L.ref: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     %.Self.frozen.loc6_33: %L.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.f7f]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %L.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %L.type = name_ref .Self, %.Self.frozen.loc6_33 [symbolic_self = constants.%.Self.frozen.f7f]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type.8c7]
// CHECK:STDOUT:     %.loc6_39: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type.8c7]
// CHECK:STDOUT:     %L1.ref: %L.assoc_type = name_ref L1, @L1.%assoc0 [concrete = constants.%assoc0.246]
// CHECK:STDOUT:     %impl.elem0.loc6_39.1: type = impl_witness_access constants.%.63f3, element0 [symbolic_self = constants.%impl.elem0.2e9]
// CHECK:STDOUT:     %.loc6_46.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_46.2: type = converted %.loc6_46.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %rewrite.loc6_43.1 = requirement_rewrite %impl.elem0.loc6_39.1, %.loc6_46.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc6_39.2: type = impl_witness_access constants.%.300, element0 [symbolic_self = constants.%impl.elem0.865]
// CHECK:STDOUT:     %rewrite.loc6_43.2 = requirement_rewrite %impl.elem0.loc6_39.2, %.loc6_46.2 [concrete]
// CHECK:STDOUT:     %.loc6_33: type = where_expr [concrete = constants.%L_where.type.94d] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %L.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc6_43.2 = requirement_rewrite %impl.elem0.loc6_39.2, %.loc6_46.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc6_23: type = splice_block %M.ref [concrete = constants.%M.type] {
// CHECK:STDOUT:       %.Self.frozen.loc6_21: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_21.1: %M.type = symbolic_binding T, 0 [symbolic = %T.loc6_21.2 (constants.%T.5fb)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = impl_self_witness @T.as_type.as.L.impl.%.loc6_26, @L [symbolic = @T.as_type.as.L.impl.%.loc6_48 (constants.%.63f6)]
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %U.patt.loc9_10.1: %pattern_type.cfb22c.1 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc9_10.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9: type = splice_block %Y.ref [concrete = constants.%Y.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc9_10.2: %Y.type = symbolic_binding U, 0 [symbolic = %U.loc9_10.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc14_15.1: %pattern_type.cfb22c.2 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt.dc7)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [concrete = constants.%Y_where.type.bd02cd.1] {
// CHECK:STDOUT:       %.Self.frozen.loc14_15: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:       %.Self.frozen.loc14_19: %Y.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Y.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %Y.type = name_ref .Self, %.Self.frozen.loc14_19 [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type.449]
// CHECK:STDOUT:       %.loc14_25.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type.449]
// CHECK:STDOUT:       %Y1.ref: %Y.assoc_type = name_ref Y1, @Y1.%assoc0 [concrete = constants.%assoc0.509]
// CHECK:STDOUT:       %impl.elem0.loc14_25.1: %L.type = impl_witness_access constants.%Y.lookup_impl_witness.94c, element0 [symbolic_self = constants.%impl.elem0.f59]
// CHECK:STDOUT:       %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:       %as_type.loc14_25.1: type = facet_access_type %impl.elem0.loc14_25.1 [symbolic_self = constants.%as_type.81d]
// CHECK:STDOUT:       %.loc14_25.2: type = converted %impl.elem0.loc14_25.1, %as_type.loc14_25.1 [symbolic_self = constants.%as_type.81d]
// CHECK:STDOUT:       %impls.loc14_29.1 = requirement_impls %.loc14_25.2, %M.ref [concrete]
// CHECK:STDOUT:       %impl.elem0.loc14_25.2: %L.type = impl_witness_access constants.%Y.lookup_impl_witness.9c7, element0 [symbolic_self = constants.%impl.elem0.e05]
// CHECK:STDOUT:       %as_type.loc14_25.2: type = facet_access_type %impl.elem0.loc14_25.2 [symbolic_self = constants.%as_type.702]
// CHECK:STDOUT:       %.loc14_25.3: type = converted %impl.elem0.loc14_25.1, %as_type.loc14_25.2 [symbolic_self = constants.%as_type.702]
// CHECK:STDOUT:       %impls.loc14_29.2 = requirement_impls %.loc14_25.3, %M.ref [concrete]
// CHECK:STDOUT:       %.loc14_19.2: type = where_expr [concrete = constants.%Y_where.type.bd02cd.1] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %Y.ref [concrete]
// CHECK:STDOUT:         %impls.loc14_29.2 = requirement_impls %.loc14_25.3, %M.ref [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_15.2: %Y_where.type.bd02cd.1 = symbolic_binding T, 0 [symbolic = %T.loc14_15.1 (constants.%T.95b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L {
// CHECK:STDOUT:   %Self: %L.type = symbolic_binding Self, 0 [symbolic = constants.%Self.480]
// CHECK:STDOUT:   %L.WithSelf.decl = interface_with_self_decl @L [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %L1: type = assoc_const_decl @L1 [concrete] {
// CHECK:STDOUT:     %assoc0: %L.assoc_type = assoc_entity element0, @L.WithSelf.%L1 [concrete = constants.%assoc0.246]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .L1 = @L1.%assoc0
// CHECK:STDOUT:   witness = (@L.WithSelf.%L1)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c16]
// CHECK:STDOUT:   %M.WithSelf.decl = interface_with_self_decl @M [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.765]
// CHECK:STDOUT:   %Y.WithSelf.decl = interface_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Y1: %L.type = assoc_const_decl @Y1 [concrete] {
// CHECK:STDOUT:     %assoc0: %Y.assoc_type = assoc_entity element0, @Y.WithSelf.%Y1 [concrete = constants.%assoc0.509]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .L = <poisoned>
// CHECK:STDOUT:   .L = <poisoned>
// CHECK:STDOUT:   .Y1 = @Y1.%assoc0
// CHECK:STDOUT:   witness = (@Y.WithSelf.%Y1)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic final impl @T.as_type.as.L.impl(%T.loc6_21.1: %M.type) {
// CHECK:STDOUT:   %T.patt.loc6_21.2: %pattern_type.1af = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_21.2 (constants.%T.patt.3fe)]
// CHECK:STDOUT:   %T.loc6_21.2: %M.type = symbolic_binding T, 0 [symbolic = %T.loc6_21.2 (constants.%T.5fb)]
// CHECK:STDOUT:   %T.as_type.loc6_26.2: type = facet_access_type %T.loc6_21.2 [symbolic = %T.as_type.loc6_26.2 (constants.%T.as_type.af8)]
// CHECK:STDOUT:   %.loc6_48: <witness> = impl_self_witness %T.loc6_21.2, @L [symbolic = %.loc6_48 (constants.%.63f6)]
// CHECK:STDOUT:   %L.impl_witness.loc6_48.2: <witness> = impl_witness %L.impl_witness_table, @T.as_type.as.L.impl(%T.loc6_21.2) [symbolic = %L.impl_witness.loc6_48.2 (constants.%L.impl_witness.4ca)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   final impl: %.loc6_26 as %L.ref {
// CHECK:STDOUT:     %L.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as_type.as.L.impl [concrete]
// CHECK:STDOUT:     %L.impl_witness.loc6_48.1: <witness> = impl_witness %L.impl_witness_table, @T.as_type.as.L.impl(constants.%T.5fb) [symbolic = %L.impl_witness.loc6_48.2 (constants.%L.impl_witness.4ca)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %L.ref
// CHECK:STDOUT:     witness = %L.impl_witness.loc6_48.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%U.loc9_10.2: %Y.type) {
// CHECK:STDOUT:   %U.patt.loc9_10.2: %pattern_type.cfb22c.1 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc9_10.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc9_10.1: %Y.type = symbolic_binding U, 0 [symbolic = %U.loc9_10.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc9_10.1, @Y [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.6a9071.1)]
// CHECK:STDOUT:   %impl.elem0.loc11_9.1: %L.type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_9.1 (constants.%impl.elem0.0788a1.1)]
// CHECK:STDOUT:   %L.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc11_9.1, @L [symbolic = %L.lookup_impl_witness (constants.%L.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc11_9.2: type = impl_witness_access %L.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_9.2 (constants.%impl.elem0.226)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc11_14: %Y.type = specific_constant %U.loc9_10.2, @C(constants.%U) [symbolic = constants.%U]
// CHECK:STDOUT:     %U.ref: %Y.type = name_ref U, %.loc11_14 [symbolic = constants.%U]
// CHECK:STDOUT:     %U.as_type: type = facet_access_type %U.ref [symbolic = constants.%U.as_type]
// CHECK:STDOUT:     %.loc11_15: type = converted %U.ref, %U.as_type [symbolic = constants.%U.as_type]
// CHECK:STDOUT:     %Y1.ref: %Y.assoc_type = name_ref Y1, @Y1.%assoc0 [concrete = constants.%assoc0.509]
// CHECK:STDOUT:     %impl.elem0.loc11_15: %L.type = impl_witness_access constants.%Y.lookup_impl_witness.6a9071.1, element0 [symbolic = constants.%impl.elem0.0788a1.1]
// CHECK:STDOUT:     %as_type: type = facet_access_type %impl.elem0.loc11_15 [symbolic = constants.%as_type.529665.1]
// CHECK:STDOUT:     %.loc11_18: type = converted %impl.elem0.loc11_15, %as_type [symbolic = constants.%as_type.529665.1]
// CHECK:STDOUT:     %L1.ref: %L.assoc_type = name_ref L1, @L1.%assoc0 [concrete = constants.%assoc0.246]
// CHECK:STDOUT:     %impl.elem0.loc11_18: type = impl_witness_access constants.%L.lookup_impl_witness, element0 [symbolic = constants.%impl.elem0.226]
// CHECK:STDOUT:     %C1: type = alias_binding C1, %impl.elem0.loc11_18 [symbolic = %impl.elem0.loc11_9.2 (constants.%impl.elem0.226)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.410
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .C1 = %C1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc14_15.2: %Y_where.type.bd02cd.1) {
// CHECK:STDOUT:   %T.patt.loc14_15.2: %pattern_type.cfb22c.2 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt.dc7)]
// CHECK:STDOUT:   %T.loc14_15.1: %Y_where.type.bd02cd.1 = symbolic_binding T, 0 [symbolic = %T.loc14_15.1 (constants.%T.95b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc26_6.2: type = facet_access_type %T.loc14_15.1 [symbolic = %T.as_type.loc26_6.2 (constants.%T.as_type.917)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc14_15.1, @Y [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.6a9071.2)]
// CHECK:STDOUT:   %Y.facet.loc26_6.2: %Y.type = facet_value %T.as_type.loc26_6.2, (%Y.lookup_impl_witness) [symbolic = %Y.facet.loc26_6.2 (constants.%Y.facet.c69)]
// CHECK:STDOUT:   %C.loc26_6.2: type = class_type @C, @C(%Y.facet.loc26_6.2) [symbolic = %C.loc26_6.2 (constants.%C.cbf)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc26_6.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: %Y_where.type.bd02cd.1 = name_ref T, %T.loc14_15.2 [symbolic = %T.loc14_15.1 (constants.%T.95b)]
// CHECK:STDOUT:     %T.as_type.loc26_6.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc26_6.2 (constants.%T.as_type.917)]
// CHECK:STDOUT:     %Y.facet.loc26_6.1: %Y.type = facet_value %T.as_type.loc26_6.1, (constants.%Y.lookup_impl_witness.6a9071.2) [symbolic = %Y.facet.loc26_6.2 (constants.%Y.facet.c69)]
// CHECK:STDOUT:     %.loc26_6: %Y.type = converted %T.ref, %Y.facet.loc26_6.1 [symbolic = %Y.facet.loc26_6.2 (constants.%Y.facet.c69)]
// CHECK:STDOUT:     %C.loc26_6.1: type = class_type @C, @C(constants.%Y.facet.c69) [symbolic = %C.loc26_6.2 (constants.%C.cbf)]
// CHECK:STDOUT:     %.loc26_7: type = specific_constant @C.%C1, @C(constants.%Y.facet.c69) [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C1.ref: type = name_ref C1, %.loc26_7 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L.WithSelf(constants.%Self.480) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M.WithSelf(constants.%Self.c16) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L.WithSelf(constants.%.Self.frozen.f7f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.L.impl(constants.%T.5fb) {
// CHECK:STDOUT:   %T.patt.loc6_21.2 => constants.%T.patt.3fe
// CHECK:STDOUT:   %T.loc6_21.2 => constants.%T.5fb
// CHECK:STDOUT:   %T.as_type.loc6_26.2 => constants.%T.as_type.af8
// CHECK:STDOUT:   %.loc6_48 => constants.%.63f6
// CHECK:STDOUT:   %L.impl_witness.loc6_48.2 => constants.%L.impl_witness.4ca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L.WithSelf(constants.%L.facet.d07) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Self.765) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc9_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc9_10.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%U) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L.WithSelf(constants.%impl.elem0.0788a1.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%.Self.frozen.3a0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.95b) {
// CHECK:STDOUT:   %T.patt.loc14_15.2 => constants.%T.patt.dc7
// CHECK:STDOUT:   %T.loc14_15.1 => constants.%T.95b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Y.facet.c69) {
// CHECK:STDOUT:   %U.patt.loc9_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc9_10.1 => constants.%Y.facet.c69
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.6a9071.2
// CHECK:STDOUT:   %impl.elem0.loc11_9.1 => constants.%impl.elem0.0788a1.2
// CHECK:STDOUT:   %L.lookup_impl_witness => constants.%L.impl_witness.d68
// CHECK:STDOUT:   %impl.elem0.loc11_9.2 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.L.impl(constants.%M.facet) {
// CHECK:STDOUT:   %T.patt.loc6_21.2 => constants.%T.patt.3fe
// CHECK:STDOUT:   %T.loc6_21.2 => constants.%M.facet
// CHECK:STDOUT:   %T.as_type.loc6_26.2 => constants.%as_type.529665.2
// CHECK:STDOUT:   %.loc6_48 => constants.%.34b
// CHECK:STDOUT:   %L.impl_witness.loc6_48.2 => constants.%L.impl_witness.d68
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_specific_impl_definition_from_inside_impl_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.6cb = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: %I.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = parameterized_class_name_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %B.generic: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.348: type = class_type @B, @B(%U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.653: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c5: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %.6ac: <witness> = impl_self_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness.e22: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.de2: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.20b: %I.WithSelf.F.type.de2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.2c5 = symbolic_binding_pattern V, 1 [symbolic]
// CHECK:STDOUT:   %V: %J.type = symbolic_binding V, 1 [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %.5be: <witness> = impl_self_witness %V, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness.7ba: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%V) [symbolic]
// CHECK:STDOUT:   %I.facet.4fb: %I.type = facet_value %V.as_type, (%I.impl_witness.7ba) [symbolic]
// CHECK:STDOUT:   %B.e4c: type = class_type @B, @B(%I.facet.4fb) [symbolic]
// CHECK:STDOUT:   %.0c7: Core.Form = init_form %B.e4c [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %B.e4c [symbolic]
// CHECK:STDOUT:   %return.param_patt.468: %pattern_type.6de = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.e7f: %pattern_type.6de = return_slot_pattern %return.param_patt.468, %B.e4c [symbolic]
// CHECK:STDOUT:   %I.WithSelf.G.type.ace: type = fn_type @I.WithSelf.G, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.G.179: %I.WithSelf.G.type.ace = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.WithSelf.%I.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.898: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.fed: %T.as_type.as.I.impl.F.type.898 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.type.f6f: type = fn_type @T.as_type.as.I.impl.G, @T.as_type.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.884: %T.as_type.as.I.impl.G.type.f6f = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet.297: %I.type = facet_value %T.as_type, (%I.impl_witness.e22) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.492: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.297) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.df6: %I.WithSelf.F.type.492 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.WithSelf.G.type.8df: type = fn_type @I.WithSelf.G, @I.WithSelf(%I.facet.297) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.G.b54: %I.WithSelf.G.type.8df = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %B.e4c [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %B.val.2d3: %B.e4c = struct_value () [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.ffd: <witness> = impl_self_witness %C, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness.90c: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %J.facet.0da: %J.type = facet_value %C, (%J.impl_witness.90c) [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.b77: <witness> = impl_self_witness %D, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness.73f: <witness> = impl_witness @D.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %J.facet.28d: %J.type = facet_value %D, (%J.impl_witness.73f) [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b7f: <witness> = impl_self_witness %C, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.2ce: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%J.facet.0da) [concrete]
// CHECK:STDOUT:   %I.facet.c8e: %I.type = facet_value %C, (%I.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.151: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.c8e) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.b4b: %I.WithSelf.F.type.151 = struct_value () [concrete]
// CHECK:STDOUT:   %I.WithSelf.G.type.278: type = fn_type @I.WithSelf.G, @I.WithSelf(%I.facet.c8e) [concrete]
// CHECK:STDOUT:   %I.WithSelf.G.b9d: %I.WithSelf.G.type.278 = struct_value () [concrete]
// CHECK:STDOUT:   %.d0a: type = fn_type_with_self_type %I.WithSelf.G.type.278, %I.facet.c8e [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.ad4: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%J.facet.0da) [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.9a9: %T.as_type.as.I.impl.F.type.ad4 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.type.f54: type = fn_type @T.as_type.as.I.impl.G, @T.as_type.as.I.impl(%J.facet.0da) [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.772: %T.as_type.as.I.impl.G.type.f54 = struct_value () [concrete]
// CHECK:STDOUT:   %.393: <witness> = impl_self_witness %D, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.6a0: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%J.facet.28d) [concrete]
// CHECK:STDOUT:   %I.facet.a62: %I.type = facet_value %D, (%I.impl_witness.6a0) [concrete]
// CHECK:STDOUT:   %B.51d: type = class_type @B, @B(%I.facet.a62) [concrete]
// CHECK:STDOUT:   %.907: Core.Form = init_form %B.51d [concrete]
// CHECK:STDOUT:   %pattern_type.711: type = pattern_type %B.51d [concrete]
// CHECK:STDOUT:   %return.param_patt.75f: %pattern_type.711 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.121: %pattern_type.711 = return_slot_pattern %return.param_patt.75f, %B.51d [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.specific_fn: <specific function> = specific_function %T.as_type.as.I.impl.G.772, @T.as_type.as.I.impl.G(%J.facet.0da, %J.facet.28d) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.1b8: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.a62) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.183: %I.WithSelf.F.type.1b8 = struct_value () [concrete]
// CHECK:STDOUT:   %I.WithSelf.G.type.ec1: type = fn_type @I.WithSelf.G, @I.WithSelf(%I.facet.a62) [concrete]
// CHECK:STDOUT:   %I.WithSelf.G.bb6: %I.WithSelf.G.type.ec1 = struct_value () [concrete]
// CHECK:STDOUT:   %.97a: type = fn_type_with_self_type %I.WithSelf.F.type.1b8, %I.facet.a62 [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.1de: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%J.facet.28d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.ec8: %T.as_type.as.I.impl.F.type.1de = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.type.8be: type = fn_type @T.as_type.as.I.impl.G, @T.as_type.as.I.impl(%J.facet.28d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.e57: %T.as_type.as.I.impl.G.type.8be = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.specific_fn: <specific function> = specific_function %T.as_type.as.I.impl.F.ec8, @T.as_type.as.I.impl.F(%J.facet.28d) [concrete]
// CHECK:STDOUT:   %B.val.052: %B.51d = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl.loc4
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl.loc4: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = class_decl @B [concrete = constants.%B.generic] {
// CHECK:STDOUT:     %U.patt.loc6_10.1: %pattern_type.6cb = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_10.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl.loc4 [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc6_10.2: %I.type = symbolic_binding U, 0 [symbolic = %U.loc6_10.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as_type.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc12_21.1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12: %J.type = name_ref T, %T.loc12_21.2 [symbolic = %T.loc12_21.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc12_26.2: type = facet_access_type %T.ref.loc12 [symbolic = %T.as_type.loc12_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc12_26: type = converted %T.ref.loc12, %T.as_type.loc12_26.2 [symbolic = %T.as_type.loc12_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.ref.loc12: type = name_ref I, file.%I.decl.loc4 [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc12_23: type = splice_block %J.ref.loc12 [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen.loc12: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref.loc12: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_21.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_21.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <witness> = impl_self_witness @T.as_type.as.I.impl.%.loc12_26, @I [symbolic = @T.as_type.as.I.impl.%.loc12_32 (constants.%.6ac)]
// CHECK:STDOUT:   %I.decl.loc14: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as_type.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc12_21.1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc19: %J.type = name_ref T, %T.loc19 [symbolic = %T.loc12_21.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc19: type = facet_access_type %T.ref.loc19 [symbolic = %T.as_type.loc12_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc19_26: type = converted %T.ref.loc19, %T.as_type.loc19 [symbolic = %T.as_type.loc12_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.ref.loc19: type = name_ref I, file.%I.decl.loc4 [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc19_23: type = splice_block %J.ref.loc19 [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen.loc19: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref.loc19: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc19: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_21.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: <witness> = impl_self_witness @T.as_type.as.I.impl.%.loc19_26, @I [symbolic = @T.as_type.as.I.impl.%.loc12_32 (constants.%.6ac)]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc32: <witness> = impl_self_witness @C.as.J.impl.%C.ref, @J [concrete = constants.%.ffd]
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @D.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc35: <witness> = impl_self_witness @D.as.J.impl.%D.ref, @J [concrete = constants.%.b77]
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.de2) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.20b)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %I.WithSelf.G.decl: @I.WithSelf.%I.WithSelf.G.type (%I.WithSelf.G.type.ace) = fn_decl @I.WithSelf.G [symbolic = @I.WithSelf.%I.WithSelf.G (constants.%I.WithSelf.G.179)] {
// CHECK:STDOUT:     %V.patt.loc16_17.1: %pattern_type.2c5 = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc16_17.2 (constants.%V.patt)]
// CHECK:STDOUT:     %return.param_patt.loc16_28.1: @I.WithSelf.G.%pattern_type (%pattern_type.6de) = out_param_pattern [symbolic = %return.param_patt.loc16_28.2 (constants.%return.param_patt.468)]
// CHECK:STDOUT:     %return.patt.loc16_22.1: @I.WithSelf.G.%pattern_type (%pattern_type.6de) = return_slot_pattern %return.param_patt.loc16_28.1, %B.loc16_28.2 [symbolic = %return.patt.loc16_22.2 (constants.%return.patt.e7f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %V.ref: %J.type = name_ref V, %V.loc16_17.2 [symbolic = %V.loc16_17.1 (constants.%V)]
// CHECK:STDOUT:     %V.as_type.loc16_28.2: type = facet_access_type %V.ref [symbolic = %V.as_type.loc16_28.1 (constants.%V.as_type)]
// CHECK:STDOUT:     %I.facet.loc16_28.2: %I.type = facet_value %V.as_type.loc16_28.2, (constants.%I.impl_witness.7ba) [symbolic = %I.facet.loc16_28.1 (constants.%I.facet.4fb)]
// CHECK:STDOUT:     %.loc16_28.2: %I.type = converted %V.ref, %I.facet.loc16_28.2 [symbolic = %I.facet.loc16_28.1 (constants.%I.facet.4fb)]
// CHECK:STDOUT:     %B.loc16_28.2: type = class_type @B, @B(constants.%I.facet.4fb) [symbolic = %B.loc16_28.1 (constants.%B.e4c)]
// CHECK:STDOUT:     %.loc16_28.3: Core.Form = init_form %B.loc16_28.2 [symbolic = %.loc16_28.1 (constants.%.0c7)]
// CHECK:STDOUT:     %.loc16_19: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc16_17.2: %J.type = symbolic_binding V, 1 [symbolic = %V.loc16_17.1 (constants.%V)]
// CHECK:STDOUT:     %return.param: ref @I.WithSelf.G.%B.loc16_28.1 (%B.e4c) = out_param call_param0
// CHECK:STDOUT:     %return: ref @I.WithSelf.G.%B.loc16_28.1 (%B.e4c) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %I.WithSelf.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .G = @I.WithSelf.%assoc1
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.%I.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.653]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic final impl @T.as_type.as.I.impl(%T.loc12_21.2: %J.type) {
// CHECK:STDOUT:   %T.patt.loc12_21.2: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc12_21.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_21.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc12_26.1: type = facet_access_type %T.loc12_21.1 [symbolic = %T.as_type.loc12_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc12_32: <witness> = impl_self_witness %T.loc12_21.1, @I [symbolic = %.loc12_32 (constants.%.6ac)]
// CHECK:STDOUT:   %I.impl_witness.loc12_32.2: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(%T.loc12_21.1) [symbolic = %I.impl_witness.loc12_32.2 (constants.%I.impl_witness.e22)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T.loc12_21.1) [symbolic = %T.as_type.as.I.impl.F.type (constants.%T.as_type.as.I.impl.F.type.898)]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.898) = struct_value () [symbolic = %T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.fed)]
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.type: type = fn_type @T.as_type.as.I.impl.G, @T.as_type.as.I.impl(%T.loc12_21.1) [symbolic = %T.as_type.as.I.impl.G.type (constants.%T.as_type.as.I.impl.G.type.f6f)]
// CHECK:STDOUT:   %T.as_type.as.I.impl.G: @T.as_type.as.I.impl.%T.as_type.as.I.impl.G.type (%T.as_type.as.I.impl.G.type.f6f) = struct_value () [symbolic = %T.as_type.as.I.impl.G (constants.%T.as_type.as.I.impl.G.884)]
// CHECK:STDOUT:
// CHECK:STDOUT:   final impl: %.loc12_26 as %I.ref.loc12 {
// CHECK:STDOUT:     %T.as_type.as.I.impl.F.decl: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.898) = fn_decl @T.as_type.as.I.impl.F [symbolic = @T.as_type.as.I.impl.%T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.fed)] {} {}
// CHECK:STDOUT:     %T.as_type.as.I.impl.G.decl: @T.as_type.as.I.impl.%T.as_type.as.I.impl.G.type (%T.as_type.as.I.impl.G.type.f6f) = fn_decl @T.as_type.as.I.impl.G [symbolic = @T.as_type.as.I.impl.%T.as_type.as.I.impl.G (constants.%T.as_type.as.I.impl.G.884)] {
// CHECK:STDOUT:       %V.patt.loc25_17.1: %pattern_type.2c5 = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc25_17.2 (constants.%V.patt)]
// CHECK:STDOUT:       %return.param_patt.loc25_28.1: @T.as_type.as.I.impl.G.%pattern_type (%pattern_type.6de) = out_param_pattern [symbolic = %return.param_patt.loc25_28.2 (constants.%return.param_patt.468)]
// CHECK:STDOUT:       %return.patt.loc25_22.1: @T.as_type.as.I.impl.G.%pattern_type (%pattern_type.6de) = return_slot_pattern %return.param_patt.loc25_28.1, %B.loc25_28.2 [symbolic = %return.patt.loc25_22.2 (constants.%return.patt.e7f)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:       %V.ref: %J.type = name_ref V, %V.loc25_17.2 [symbolic = %V.loc25_17.1 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc25_28.2: type = facet_access_type %V.ref [symbolic = %V.as_type.loc25_28.1 (constants.%V.as_type)]
// CHECK:STDOUT:       %I.facet.loc25_28.2: %I.type = facet_value %V.as_type.loc25_28.2, (constants.%I.impl_witness.7ba) [symbolic = %I.facet.loc25_28.1 (constants.%I.facet.4fb)]
// CHECK:STDOUT:       %.loc25_28.2: %I.type = converted %V.ref, %I.facet.loc25_28.2 [symbolic = %I.facet.loc25_28.1 (constants.%I.facet.4fb)]
// CHECK:STDOUT:       %B.loc25_28.2: type = class_type @B, @B(constants.%I.facet.4fb) [symbolic = %B.loc25_28.1 (constants.%B.e4c)]
// CHECK:STDOUT:       %.loc25_28.3: Core.Form = init_form %B.loc25_28.2 [symbolic = %.loc25_28.1 (constants.%.0c7)]
// CHECK:STDOUT:       %.loc25_19: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:         %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:         %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %V.loc25_17.2: %J.type = symbolic_binding V, 1 [symbolic = %V.loc25_17.1 (constants.%V)]
// CHECK:STDOUT:       %return.param: ref @T.as_type.as.I.impl.G.%B.loc25_28.1 (%B.e4c) = out_param call_param0
// CHECK:STDOUT:       %return: ref @T.as_type.as.I.impl.G.%B.loc25_28.1 (%B.e4c) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%T.as_type.as.I.impl.F.decl, %T.as_type.as.I.impl.G.decl), @T.as_type.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc12_32.1: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(constants.%T) [symbolic = %I.impl_witness.loc12_32.2 (constants.%I.impl_witness.e22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as_type.as.I.impl.F.decl
// CHECK:STDOUT:     .J = <poisoned>
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .G = %T.as_type.as.I.impl.G.decl
// CHECK:STDOUT:     extend %I.ref.loc12
// CHECK:STDOUT:     witness = %I.impl_witness.loc12_32.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref as %J.ref {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.90c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.J.impl: %D.ref as %J.ref {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @D.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.73f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B(%U.loc6_10.2: %I.type) {
// CHECK:STDOUT:   %U.patt.loc6_10.2: %pattern_type.6cb = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_10.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc6_10.1: %I.type = symbolic_binding U, 0 [symbolic = %U.loc6_10.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc7: %I.type = specific_constant %U.loc6_10.2, @B(constants.%U) [symbolic = constants.%U]
// CHECK:STDOUT:     %U.ref: %I.type = name_ref U, %.loc7 [symbolic = constants.%U]
// CHECK:STDOUT:     %B1: %I.type = alias_binding B1, %U.ref [symbolic = %U.loc6_10.1 (constants.%U)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%B.348
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .B1 = %B1
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.G(@I.%Self: %I.type, %V.loc16_17.2: %J.type) {
// CHECK:STDOUT:   %V.patt.loc16_17.2: %pattern_type.2c5 = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc16_17.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc16_17.1: %J.type = symbolic_binding V, 1 [symbolic = %V.loc16_17.1 (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc16_28.1: type = facet_access_type %V.loc16_17.1 [symbolic = %V.as_type.loc16_28.1 (constants.%V.as_type)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%V.loc16_17.1) [symbolic = %I.impl_witness (constants.%I.impl_witness.7ba)]
// CHECK:STDOUT:   %I.facet.loc16_28.1: %I.type = facet_value %V.as_type.loc16_28.1, (%I.impl_witness) [symbolic = %I.facet.loc16_28.1 (constants.%I.facet.4fb)]
// CHECK:STDOUT:   %B.loc16_28.1: type = class_type @B, @B(%I.facet.loc16_28.1) [symbolic = %B.loc16_28.1 (constants.%B.e4c)]
// CHECK:STDOUT:   %.loc16_28.1: Core.Form = init_form %B.loc16_28.1 [symbolic = %.loc16_28.1 (constants.%.0c7)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %B.loc16_28.1 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:   %return.param_patt.loc16_28.2: @I.WithSelf.G.%pattern_type (%pattern_type.6de) = out_param_pattern [symbolic = %return.param_patt.loc16_28.2 (constants.%return.param_patt.468)]
// CHECK:STDOUT:   %return.patt.loc16_22.2: @I.WithSelf.G.%pattern_type (%pattern_type.6de) = return_slot_pattern %return.param_patt.loc16_28.2, %B.loc16_28.1 [symbolic = %return.patt.loc16_22.2 (constants.%return.patt.e7f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @I.WithSelf.G.%B.loc16_28.1 (%B.e4c);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as_type.as.I.impl.F(@T.as_type.as.I.impl.%T.loc19: %J.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as_type.as.I.impl.G(@T.as_type.as.I.impl.%T.loc19: %J.type, %V.loc25_17.2: %J.type) {
// CHECK:STDOUT:   %V.patt.loc25_17.2: %pattern_type.2c5 = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc25_17.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc25_17.1: %J.type = symbolic_binding V, 1 [symbolic = %V.loc25_17.1 (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc25_28.1: type = facet_access_type %V.loc25_17.1 [symbolic = %V.as_type.loc25_28.1 (constants.%V.as_type)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%V.loc25_17.1) [symbolic = %I.impl_witness (constants.%I.impl_witness.7ba)]
// CHECK:STDOUT:   %I.facet.loc25_28.1: %I.type = facet_value %V.as_type.loc25_28.1, (%I.impl_witness) [symbolic = %I.facet.loc25_28.1 (constants.%I.facet.4fb)]
// CHECK:STDOUT:   %B.loc25_28.1: type = class_type @B, @B(%I.facet.loc25_28.1) [symbolic = %B.loc25_28.1 (constants.%B.e4c)]
// CHECK:STDOUT:   %.loc25_28.1: Core.Form = init_form %B.loc25_28.1 [symbolic = %.loc25_28.1 (constants.%.0c7)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %B.loc25_28.1 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:   %return.param_patt.loc25_28.2: @T.as_type.as.I.impl.G.%pattern_type (%pattern_type.6de) = out_param_pattern [symbolic = %return.param_patt.loc25_28.2 (constants.%return.param_patt.468)]
// CHECK:STDOUT:   %return.patt.loc25_22.2: @T.as_type.as.I.impl.G.%pattern_type (%pattern_type.6de) = return_slot_pattern %return.param_patt.loc25_28.2, %B.loc25_28.1 [symbolic = %return.patt.loc25_22.2 (constants.%return.patt.e7f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %B.loc25_28.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %B.val: @T.as_type.as.I.impl.G.%B.loc25_28.1 (%B.e4c) = struct_value () [symbolic = %B.val (constants.%B.val.2d3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @T.as_type.as.I.impl.G.%B.loc25_28.1 (%B.e4c) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc26_13.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc26_13.2: init @T.as_type.as.I.impl.G.%B.loc25_28.1 (%B.e4c) to %return.param = class_init () [symbolic = %B.val (constants.%B.val.2d3)]
// CHECK:STDOUT:     %.loc26_14: init @T.as_type.as.I.impl.G.%B.loc25_28.1 (%B.e4c) = converted %.loc26_13.1, %.loc26_13.2 [symbolic = %B.val (constants.%B.val.2d3)]
// CHECK:STDOUT:     return %.loc26_14 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref.loc46: type = name_ref I, file.%I.decl.loc4 [concrete = constants.%I.type]
// CHECK:STDOUT:   %G.ref: %I.assoc_type = name_ref G, @I.WithSelf.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness.2ce) [concrete = constants.%I.facet.c8e]
// CHECK:STDOUT:   %.loc46_19: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet.c8e]
// CHECK:STDOUT:   %impl.elem1: %.d0a = impl_witness_access constants.%I.impl_witness.2ce, element1 [concrete = constants.%T.as_type.as.I.impl.G.772]
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %D.ref, (constants.%J.impl_witness.73f) [concrete = constants.%J.facet.28d]
// CHECK:STDOUT:   %.loc46_27.1: %J.type = converted %D.ref, %J.facet [concrete = constants.%J.facet.28d]
// CHECK:STDOUT:   %specific_fn.loc46: <specific function> = specific_function %impl.elem1, @T.as_type.as.I.impl.G(constants.%J.facet.0da, constants.%J.facet.28d) [concrete = constants.%T.as_type.as.I.impl.G.specific_fn]
// CHECK:STDOUT:   %.loc46_27.2: ref %B.51d = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.call: init %B.51d to %.loc46_27.2 = call %specific_fn.loc46()
// CHECK:STDOUT:   %.loc46_27.3: ref %B.51d = temporary %.loc46_27.2, %T.as_type.as.I.impl.G.call
// CHECK:STDOUT:   %.loc46_28: %I.type = specific_constant @B.%B1, @B(constants.%I.facet.a62) [concrete = constants.%I.facet.a62]
// CHECK:STDOUT:   %B1.ref: %I.type = name_ref B1, %.loc46_28 [concrete = constants.%I.facet.a62]
// CHECK:STDOUT:   %D_As_I: %I.type = alias_binding D_As_I, %B1.ref [concrete = constants.%I.facet.a62]
// CHECK:STDOUT:   %D_As_I.ref: %I.type = name_ref D_As_I, %D_As_I [concrete = constants.%I.facet.a62]
// CHECK:STDOUT:   %I.ref.loc50: type = name_ref I, file.%I.decl.loc4 [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.97a = impl_witness_access constants.%I.impl_witness.6a0, element0 [concrete = constants.%T.as_type.as.I.impl.F.ec8]
// CHECK:STDOUT:   %specific_fn.loc50: <specific function> = specific_function %impl.elem0, @T.as_type.as.I.impl.F(constants.%J.facet.28d) [concrete = constants.%T.as_type.as.I.impl.F.specific_fn]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.call: init %empty_tuple.type = call %specific_fn.loc50()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc6_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc6_10.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.653) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc12_21.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc12_21.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc12_26.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc12_32 => constants.%.6ac
// CHECK:STDOUT:   %I.impl_witness.loc12_32.2 => constants.%I.impl_witness.e22
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.898
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.fed
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.type => constants.%T.as_type.as.I.impl.G.type.f6f
// CHECK:STDOUT:   %T.as_type.as.I.impl.G => constants.%T.as_type.as.I.impl.G.884
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.de2
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.20b
// CHECK:STDOUT:   %I.WithSelf.G.type => constants.%I.WithSelf.G.type.ace
// CHECK:STDOUT:   %I.WithSelf.G => constants.%I.WithSelf.G.179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%V) {
// CHECK:STDOUT:   %T.patt.loc12_21.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc12_21.1 => constants.%V
// CHECK:STDOUT:   %T.as_type.loc12_26.1 => constants.%V.as_type
// CHECK:STDOUT:   %.loc12_32 => constants.%.5be
// CHECK:STDOUT:   %I.impl_witness.loc12_32.2 => constants.%I.impl_witness.7ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%I.facet.4fb) {
// CHECK:STDOUT:   %U.patt.loc6_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc6_10.1 => constants.%I.facet.4fb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.G(constants.%Self.37e, constants.%V) {
// CHECK:STDOUT:   %V.patt.loc16_17.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc16_17.1 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc16_28.1 => constants.%V.as_type
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.7ba
// CHECK:STDOUT:   %I.facet.loc16_28.1 => constants.%I.facet.4fb
// CHECK:STDOUT:   %B.loc16_28.1 => constants.%B.e4c
// CHECK:STDOUT:   %.loc16_28.1 => constants.%.0c7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT:   %return.param_patt.loc16_28.2 => constants.%return.param_patt.468
// CHECK:STDOUT:   %return.patt.loc16_22.2 => constants.%return.patt.e7f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.G(constants.%T, constants.%V) {
// CHECK:STDOUT:   %V.patt.loc25_17.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc25_17.1 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc25_28.1 => constants.%V.as_type
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.7ba
// CHECK:STDOUT:   %I.facet.loc25_28.1 => constants.%I.facet.4fb
// CHECK:STDOUT:   %B.loc25_28.1 => constants.%B.e4c
// CHECK:STDOUT:   %.loc25_28.1 => constants.%.0c7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT:   %return.param_patt.loc25_28.2 => constants.%return.param_patt.468
// CHECK:STDOUT:   %return.patt.loc25_22.2 => constants.%return.patt.e7f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.297) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.297
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.492
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.df6
// CHECK:STDOUT:   %I.WithSelf.G.type => constants.%I.WithSelf.G.type.8df
// CHECK:STDOUT:   %I.WithSelf.G => constants.%I.WithSelf.G.b54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.297) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.G(constants.%I.facet.297, constants.%V) {
// CHECK:STDOUT:   %V.patt.loc16_17.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc16_17.1 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc16_28.1 => constants.%V.as_type
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.7ba
// CHECK:STDOUT:   %I.facet.loc16_28.1 => constants.%I.facet.4fb
// CHECK:STDOUT:   %B.loc16_28.1 => constants.%B.e4c
// CHECK:STDOUT:   %.loc16_28.1 => constants.%.0c7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT:   %return.param_patt.loc16_28.2 => constants.%return.param_patt.468
// CHECK:STDOUT:   %return.patt.loc16_22.2 => constants.%return.patt.e7f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.0da) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.28d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%J.facet.0da) {
// CHECK:STDOUT:   %T.patt.loc12_21.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc12_21.1 => constants.%J.facet.0da
// CHECK:STDOUT:   %T.as_type.loc12_26.1 => constants.%C
// CHECK:STDOUT:   %.loc12_32 => constants.%.b7f
// CHECK:STDOUT:   %I.impl_witness.loc12_32.2 => constants.%I.impl_witness.2ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.ad4
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.9a9
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.type => constants.%T.as_type.as.I.impl.G.type.f54
// CHECK:STDOUT:   %T.as_type.as.I.impl.G => constants.%T.as_type.as.I.impl.G.772
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.c8e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.c8e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.151
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.b4b
// CHECK:STDOUT:   %I.WithSelf.G.type => constants.%I.WithSelf.G.type.278
// CHECK:STDOUT:   %I.WithSelf.G => constants.%I.WithSelf.G.b9d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.G(constants.%J.facet.0da, constants.%J.facet.28d) {
// CHECK:STDOUT:   %V.patt.loc25_17.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc25_17.1 => constants.%J.facet.28d
// CHECK:STDOUT:   %V.as_type.loc25_28.1 => constants.%D
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.6a0
// CHECK:STDOUT:   %I.facet.loc25_28.1 => constants.%I.facet.a62
// CHECK:STDOUT:   %B.loc25_28.1 => constants.%B.51d
// CHECK:STDOUT:   %.loc25_28.1 => constants.%.907
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.711
// CHECK:STDOUT:   %return.param_patt.loc25_28.2 => constants.%return.param_patt.75f
// CHECK:STDOUT:   %return.patt.loc25_22.2 => constants.%return.patt.121
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %B.val => constants.%B.val.052
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%J.facet.28d) {
// CHECK:STDOUT:   %T.patt.loc12_21.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc12_21.1 => constants.%J.facet.28d
// CHECK:STDOUT:   %T.as_type.loc12_26.1 => constants.%D
// CHECK:STDOUT:   %.loc12_32 => constants.%.393
// CHECK:STDOUT:   %I.impl_witness.loc12_32.2 => constants.%I.impl_witness.6a0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.1de
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.ec8
// CHECK:STDOUT:   %T.as_type.as.I.impl.G.type => constants.%T.as_type.as.I.impl.G.type.8be
// CHECK:STDOUT:   %T.as_type.as.I.impl.G => constants.%T.as_type.as.I.impl.G.e57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%I.facet.a62) {
// CHECK:STDOUT:   %U.patt.loc6_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc6_10.1 => constants.%I.facet.a62
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.a62) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.a62
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.1b8
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.183
// CHECK:STDOUT:   %I.WithSelf.G.type => constants.%I.WithSelf.G.type.ec1
// CHECK:STDOUT:   %I.WithSelf.G => constants.%I.WithSelf.G.bb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%J.facet.28d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
