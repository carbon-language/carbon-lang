// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/use_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/use_assoc_const.carbon

// --- associated_type_in_method_signature.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F[self: Self](u: U) -> U;
}

impl () as J where .U = i32 {
  fn F[self: Self](u: i32) -> i32 { return -u; }
}

fn CallMethod(x: ()) -> i32 {
  return x.(J.F)(40);
}

class C {
  // This allows the type to be copyable so it can be returned.
  adapt {};
}

impl C as J where .U = C {
  fn F[self: Self](u: C) -> C { return self; }
}

// --- associated_type_in_function_signature.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F(u: U) -> U;
}

class D { }

impl D as J where .U = i32 {
  fn F(u: i32) -> i32 { return -u; }
}

fn CallFunction() -> i32 {
  return D.(J.F)(4);
}

// --- extend_impl_with_associated_type_in_signature.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F(u: U) -> U;
  fn G[self: Self](v: U) -> U;
}

class E {
  extend impl as J where .U = i32 {
    fn F(u: i32) -> i32 {
      return -u;
    }
    fn G[self: Self](v: i32) -> i32 {
      return -v;
    }
  }
}

fn CallBoth(e: E) {
  let e1: i32 = e.F(2);
  let e2: i32 = e.G(3);
  let e3: i32 = E.F(4);
  let e4: i32 = e.(E.G)(5);
  let e5: i32 = e.(J.G)(6);
}

fn GenericCallF[T:! J](t: T, u: T.U) -> T.U {
  return t.F(u);
}

fn CallGeneric(e: E) -> i32 {
  return GenericCallF(e, 2);
}

// --- use_constraint_on_associated_type.carbon
library "[[@TEST_NAME]]";

interface I {
  fn Op[self: Self](b: Self) -> Self;
}

interface J {
  let U:! I & Core.Destroy;
  fn F(u: U) -> U;
}

fn GenericResult[T:! J](t: T, u: T.U) -> T.U {
  return t.F(u).(I.Op)(T.F(u));
}

// --- interface_qualified.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn G[self: Self](v: U) -> U;
}

fn GenericCallInterfaceQualified[T:! J](t: T, u: T.U) -> T.U {
  return t.(J.G)(u);
}

// --- use_where.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F(u: U) -> U;
}

fn GenericCallFI32[T:! J where .U = i32](t: T) -> i32 {
  return t.F(2);
}

// --- fail_todo_use_associated_constant_in_impl.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F(u: U) -> U;
}

class E {
  extend impl as J where .U = i32 {
    // CHECK:STDERR: fail_todo_use_associated_constant_in_impl.carbon:[[@LINE+14]]:13: error: cannot implicitly convert non-type value of type `<associated entity in J>` to `type` [ConversionFailureNonTypeToFacet]
    // CHECK:STDERR:     fn F(u: U) -> U {
    // CHECK:STDERR:             ^
    // CHECK:STDERR: fail_todo_use_associated_constant_in_impl.carbon:[[@LINE+11]]:13: note: type `<associated entity in J>` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
    // CHECK:STDERR:     fn F(u: U) -> U {
    // CHECK:STDERR:             ^
    // CHECK:STDERR:
    // CHECK:STDERR: fail_todo_use_associated_constant_in_impl.carbon:[[@LINE+7]]:19: error: cannot implicitly convert non-type value of type `<associated entity in J>` to `type` [ConversionFailureNonTypeToFacet]
    // CHECK:STDERR:     fn F(u: U) -> U {
    // CHECK:STDERR:                   ^
    // CHECK:STDERR: fail_todo_use_associated_constant_in_impl.carbon:[[@LINE+4]]:19: note: type `<associated entity in J>` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
    // CHECK:STDERR:     fn F(u: U) -> U {
    // CHECK:STDERR:                   ^
    // CHECK:STDERR:
    fn F(u: U) -> U {
      return u;
    }
  }
}

// --- fail_todo_self_period_associated_type.carbon
library "[[@TEST_NAME]]";

interface J2 {
  let U2:! type;
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE+14]]:23: error: member access into object of incomplete type `J2` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:   fn F[self: Self](z: Self.U2) -> Self.U2;
  // CHECK:STDERR:                       ^~~~
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE-5]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface J2 {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE+7]]:35: error: member access into object of incomplete type `J2` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:   fn F[self: Self](z: Self.U2) -> Self.U2;
  // CHECK:STDERR:                                   ^~~~
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE-12]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface J2 {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F[self: Self](z: Self.U2) -> Self.U2;
}

impl () as J2 where .U2 = {} {
  fn F[self: Self](z: {}) -> {} { return z; }
}

class C2 {
  // This allows the type to be copyable so it can be returned.
  adapt {};
}

impl C2 as J2 where .U2 = C2 {
  fn F[self: Self](z: C2) -> C2 { return self; }
}

// --- fail_associated_type_in_signature_mismatch.carbon
library "[[@TEST_NAME]]";

interface K {
  let V:! type;
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE+3]]:20: error: cannot convert from struct type `{.a: ()}` to `{.x: ()}`: missing field `x` in source type [StructInitMissingFieldInConversion]
  // CHECK:STDERR:   fn F[self: Self](v: V) -> V;
  // CHECK:STDERR:                    ^~~~
  fn F[self: Self](v: V) -> V;
}

impl () as K where .V = {.a: ()} {
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE+14]]:20: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[self: Self](v: {.x: ()}) -> {.x: ()} { return v; }
  // CHECK:STDERR:                    ^~~~~~~~~~~
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE-7]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F[self: Self](v: V) -> V;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE+7]]:3: error: cannot convert from struct type `{.x: ()}` to `{.a: ()}`: missing field `a` in source type [StructInitMissingFieldInConversion]
  // CHECK:STDERR:   fn F[self: Self](v: {.x: ()}) -> {.x: ()} { return v; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE-14]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F[self: Self](v: V) -> V;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F[self: Self](v: {.x: ()}) -> {.x: ()} { return v; }
}

// --- use_non-type_in_function.carbon
library "[[@TEST_NAME]]";

interface M {
  let Z:! {.b: {}};
  fn G() -> {};
}

impl () as M where .Z = {.b = {}} {
  fn G() -> {} {
    return (Self as M).Z.b;
  }
}

// --- self_as_uses_correct_rewrite_constraint.carbon
library "[[@TEST_NAME]]";

class C(T:! type) {}

interface M {
  let Z:! {.b: type};
  fn G() -> type;
}

impl C({}) as M where .Z = {.b = {}} {
  fn G() -> type {
    return (Self as M).Z.b;
  }
}

impl C(()) as M where .Z = {.b = ()} {
  fn G() -> type {
    return (Self as M).Z.b;
  }
}

// --- associated_int_in_array.carbon
library "[[@TEST_NAME]]";

interface I {
  let N:! i32;
  fn F[self: Self]() -> array(bool, N);
}

impl () as I where .N = 2 {
  fn F[self: Self]() -> array(bool, 2) { return (true, false); }
}

// --- symbolic_associated_type_in_concrete_context.carbon

interface Z {
  let X:! type;
}

class C(T:! type) {}
class D {}

impl forall [T:! type] T as Z where .X = C(T) {}

fn F() {
  let a: D.(Z.X) = {} as C(D);
}

// CHECK:STDOUT: --- associated_type_in_method_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.dc6: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.1fa: type = symbolic_binding_type Self, 0, %Self.dc6 [symbolic]
// CHECK:STDOUT:   %pattern_type.4c4: type = pattern_type %Self.binding.as_type.1fa [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.cd2: <witness> = lookup_impl_witness %Self.dc6, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.fa9: type = impl_witness_access %J.lookup_impl_witness.cd2, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.345: type = pattern_type %impl.elem0.fa9 [symbolic]
// CHECK:STDOUT:   %J.F.type: type = fn_type @J.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %J.F: %J.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.572: %J.assoc_type = assoc_entity element1, @J.%J.F.decl [concrete]
// CHECK:STDOUT:   %.Self.c58: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.c58 [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.ab8: <witness> = lookup_impl_witness %.Self.c58, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ab2: type = impl_witness_access %J.lookup_impl_witness.ab8, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type.847: type = facet_type <@J where %impl.elem0.ab2 = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness.c0b: <witness> = impl_witness file.%J.impl_witness_table.loc8 [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J.impl.F.type: type = fn_type @empty_tuple.type.as.J.impl.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J.impl.F: %empty_tuple.type.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.2a7: %J.type = facet_value %empty_tuple.type, (%J.impl_witness.c0b) [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.Op.type: type = fn_type @Negate.Op [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.1d3: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.5d2: %Int.as.Negate.impl.Op.type.1d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Negate.impl_witness.bc7: <witness> = impl_witness imports.%Negate.impl_witness_table, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.6b1: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.4f9: %Int.as.Negate.impl.Op.type.6b1 = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %i32, (%Negate.impl_witness.bc7) [concrete]
// CHECK:STDOUT:   %.f7e: type = fn_type_with_self_type %Negate.Op.type, %Negate.facet [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Negate.impl.Op.4f9, @Int.as.Negate.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %CallMethod.type: type = fn_type @CallMethod [concrete]
// CHECK:STDOUT:   %CallMethod: %CallMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %.813: type = fn_type_with_self_type %J.F.type, %J.facet.2a7 [concrete]
// CHECK:STDOUT:   %int_40.f80: Core.IntLiteral = int_value 40 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_40.f80, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_40.f80, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_40.518: %i32 = int_value 40 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J_where.type.9e8: type = facet_type <@J where %impl.elem0.ab2 = %C> [concrete]
// CHECK:STDOUT:   %J.impl_witness.2a8: <witness> = impl_witness file.%J.impl_witness_table.loc21 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.J.impl.F.type: type = fn_type @C.as.J.impl.F [concrete]
// CHECK:STDOUT:   %C.as.J.impl.F: %C.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.226: %J.type = facet_value %C, (%J.impl_witness.2a8) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Negate = %Core.Negate
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/parts/int_literal, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.3f0 = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ca8: @Int.as.Negate.impl.%Int.as.Negate.impl.Op.type (%Int.as.Negate.impl.Op.type.1d3) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Negate.impl.%Int.as.Negate.impl.Op (constants.%Int.as.Negate.impl.Op.5d2)]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.3f0, %Core.import_ref.ca8), @Int.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .CallMethod = %CallMethod.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.c58]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.c58]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.ab8, element0 [symbolic_self = constants.%impl.elem0.ab2]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%J_where.type.847] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table.loc8 = impl_witness_table (%impl_witness_assoc_constant.loc8, @empty_tuple.type.as.J.impl.%empty_tuple.type.as.J.impl.F.decl), @empty_tuple.type.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness.loc8: <witness> = impl_witness %J.impl_witness_table.loc8 [concrete = constants.%J.impl_witness.c0b]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc8: type = impl_witness_assoc_constant constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %CallMethod.decl: %CallMethod.type = fn_decl @CallMethod [concrete = constants.%CallMethod] {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc12_19.1: type = splice_block %.loc12_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc12_19.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc12_19.3: type = converted %.loc12_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc21_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.c58]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.c58]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc21_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.ab8, element0 [symbolic_self = constants.%impl.elem0.ab2]
// CHECK:STDOUT:     %C.ref.loc21_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc21_13: type = where_expr %.Self [concrete = constants.%J_where.type.9e8] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref.loc21_24
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table.loc21 = impl_witness_table (%impl_witness_assoc_constant.loc21, @C.as.J.impl.%C.as.J.impl.F.decl), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness.loc21: <witness> = impl_witness %J.impl_witness_table.loc21 [concrete = constants.%J.impl_witness.2a8]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc21: type = impl_witness_assoc_constant constants.%C [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.dc6]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.F.decl: %J.F.type = fn_decl @J.F [concrete = constants.%J.F] {
// CHECK:STDOUT:     %self.patt: @J.F.%pattern_type.loc5_8 (%pattern_type.4c4) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @J.F.%pattern_type.loc5_8 (%pattern_type.4c4) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @J.F.%pattern_type.loc5_20 (%pattern_type.345) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.F.%pattern_type.loc5_20 (%pattern_type.345) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @J.F.%pattern_type.loc5_20 (%pattern_type.345) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.F.%pattern_type.loc5_20 (%pattern_type.345) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access constants.%J.lookup_impl_witness.cd2, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     %U.ref.loc5_29: type = name_ref U, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     %self.param: @J.F.%Self.binding.as_type (%Self.binding.as_type.1fa) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1fa)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.dc6)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1fa)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1fa)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J.F.%Self.binding.as_type (%Self.binding.as_type.1fa) = value_binding self, %self.param
// CHECK:STDOUT:     %u.param: @J.F.%impl.elem0.loc5_23.1 (%impl.elem0.fa9) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %U.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9)] {
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access constants.%J.lookup_impl_witness.cd2, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:       %U.ref.loc5_23: type = name_ref U, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.F.%impl.elem0.loc5_23.1 (%impl.elem0.fa9) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.F.%impl.elem0.loc5_23.1 (%impl.elem0.fa9) = out_param call_param2
// CHECK:STDOUT:     %return: ref @J.F.%impl.elem0.loc5_23.1 (%impl.elem0.fa9) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.F.decl [concrete = constants.%assoc1.572]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.J.impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %empty_tuple.type.as.J.impl.F.decl: %empty_tuple.type.as.J.impl.F.type = fn_decl @empty_tuple.type.as.J.impl.F [concrete = constants.%empty_tuple.type.as.J.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: %pattern_type.7ce = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.7ce = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc9_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.J.impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc9: type = splice_block %i32.loc9_23 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc9_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.J.impl.F.decl
// CHECK:STDOUT:   witness = file.%J.impl_witness.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref.loc21_6 as %.loc21_13 {
// CHECK:STDOUT:   %C.as.J.impl.F.decl: %C.as.J.impl.F.type = fn_decl @C.as.J.impl.F [concrete = constants.%C.as.J.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: %pattern_type.c48 = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.c48 = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc22_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.J.impl.%C.ref.loc21_6 [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:     %u.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc22_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %u: %C = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %C.as.J.impl.F.decl
// CHECK:STDOUT:   witness = file.%J.impl_witness.loc21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc18_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc18_11: type = converted %.loc18_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc18_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.dc6)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1fa)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.4c4)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.cd2)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:   %pattern_type.loc5_20: type = pattern_type %impl.elem0.loc5_23.1 [symbolic = %pattern_type.loc5_20 (constants.%pattern_type.345)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J.F.%Self.binding.as_type (%Self.binding.as_type.1fa), %u.param: @J.F.%impl.elem0.loc5_23.1 (%impl.elem0.fa9)) -> @J.F.%impl.elem0.loc5_23.1 (%impl.elem0.fa9);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.J.impl.F(%self.param: %empty_tuple.type, %u.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %impl.elem1: %.f7e = impl_witness_access constants.%Negate.impl_witness.bc7, element1 [concrete = constants.%Int.as.Negate.impl.Op.4f9]
// CHECK:STDOUT:   %bound_method.loc9_44.1: <bound method> = bound_method %u.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Int.as.Negate.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Negate.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_44.2: <bound method> = bound_method %u.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call: init %i32 = call %bound_method.loc9_44.2(%u.ref)
// CHECK:STDOUT:   return %Int.as.Negate.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallMethod(%x.param: %empty_tuple.type) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.572]
// CHECK:STDOUT:   %impl.elem1: %.813 = impl_witness_access constants.%J.impl_witness.c0b, element1 [concrete = constants.%empty_tuple.type.as.J.impl.F]
// CHECK:STDOUT:   %bound_method.loc13_11: <bound method> = bound_method %x.ref, %impl.elem1
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40.f80]
// CHECK:STDOUT:   %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc13_18.1: <bound method> = bound_method %int_40, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_18.2: <bound method> = bound_method %int_40, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_18.2(%int_40) [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %.loc13_18.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %.loc13_18.2: %i32 = converted %int_40, %.loc13_18.1 [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %empty_tuple.type.as.J.impl.F.call: init %i32 = call %bound_method.loc13_11(%x.ref, %.loc13_18.2)
// CHECK:STDOUT:   return %empty_tuple.type.as.J.impl.F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.J.impl.F(%self.param: %C, %u.param: %C) -> %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   %.loc22_44.1: %empty_struct_type = as_compatible %self.ref
// CHECK:STDOUT:   %.loc22_44.2: ref %empty_struct_type = as_compatible %return
// CHECK:STDOUT:   %.loc22_44.3: init %empty_struct_type = struct_init () to %.loc22_44.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22_44.4: init %C = as_compatible %.loc22_44.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc22_44.5: init %C = converted %self.ref, %.loc22_44.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc22_44.5 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.dc6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%Self.dc6) {
// CHECK:STDOUT:   %Self => constants.%Self.dc6
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.1fa
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.4c4
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.fa9
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.345
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%.Self.c58) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet.2a7) {
// CHECK:STDOUT:   %Self => constants.%J.facet.2a7
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness.c0b
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet.226) {
// CHECK:STDOUT:   %Self => constants.%J.facet.226
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.c48
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness.2a8
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.c48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_type_in_function_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.dc6: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %J.lookup_impl_witness.cd2: <witness> = lookup_impl_witness %Self.dc6, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.fa9: type = impl_witness_access %J.lookup_impl_witness.cd2, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.345: type = pattern_type %impl.elem0.fa9 [symbolic]
// CHECK:STDOUT:   %J.F.type: type = fn_type @J.F [concrete]
// CHECK:STDOUT:   %J.F: %J.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.572: %J.assoc_type = assoc_entity element1, @J.%J.F.decl [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.c58: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.c58 [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.ab8: <witness> = lookup_impl_witness %.Self.c58, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ab2: type = impl_witness_access %J.lookup_impl_witness.ab8, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.ab2 = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness file.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %D.as.J.impl.F.type: type = fn_type @D.as.J.impl.F [concrete]
// CHECK:STDOUT:   %D.as.J.impl.F: %D.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %D, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.Op.type: type = fn_type @Negate.Op [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.1d3: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.5d2: %Int.as.Negate.impl.Op.type.1d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Negate.impl_witness.bc7: <witness> = impl_witness imports.%Negate.impl_witness_table, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.6b1: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.4f9: %Int.as.Negate.impl.Op.type.6b1 = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %i32, (%Negate.impl_witness.bc7) [concrete]
// CHECK:STDOUT:   %.f7e: type = fn_type_with_self_type %Negate.Op.type, %Negate.facet [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Negate.impl.Op.4f9, @Int.as.Negate.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %CallFunction.type: type = fn_type @CallFunction [concrete]
// CHECK:STDOUT:   %CallFunction: %CallFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %.7bd: type = fn_type_with_self_type %J.F.type, %J.facet [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Negate = %Core.Negate
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/parts/int_literal, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.3f0 = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ca8: @Int.as.Negate.impl.%Int.as.Negate.impl.Op.type (%Int.as.Negate.impl.Op.type.1d3) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Negate.impl.%Int.as.Negate.impl.Op (constants.%Int.as.Negate.impl.Op.5d2)]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.3f0, %Core.import_ref.ca8), @Int.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .CallFunction = %CallFunction.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @D.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.c58]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.c58]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.ab8, element0 [symbolic_self = constants.%impl.elem0.ab2]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_13: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @D.as.J.impl.%D.as.J.impl.F.decl), @D.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %CallFunction.decl: %CallFunction.type = fn_decl @CallFunction [concrete = constants.%CallFunction] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.dc6]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.F.decl: %J.F.type = fn_decl @J.F [concrete = constants.%J.F] {
// CHECK:STDOUT:     %u.patt: @J.F.%pattern_type (%pattern_type.345) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.F.%pattern_type (%pattern_type.345) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @J.F.%pattern_type (%pattern_type.345) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.F.%pattern_type (%pattern_type.345) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.cd2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     %u.param: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.cd2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = out_param call_param1
// CHECK:STDOUT:     %return: ref @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.F.decl [concrete = constants.%assoc1.572]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.J.impl: %D.ref as %.loc10_13 {
// CHECK:STDOUT:   %D.as.J.impl.F.decl: %D.as.J.impl.F.type = fn_decl @D.as.J.impl.F [concrete = constants.%D.as.J.impl.F] {
// CHECK:STDOUT:     %u.patt: %pattern_type.7ce = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.7ce = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %i32.loc11_11 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %D.as.J.impl.F.decl
// CHECK:STDOUT:   witness = file.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.dc6)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.cd2)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.345)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9)) -> @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.J.impl.F(%u.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %impl.elem1: %.f7e = impl_witness_access constants.%Negate.impl_witness.bc7, element1 [concrete = constants.%Int.as.Negate.impl.Op.4f9]
// CHECK:STDOUT:   %bound_method.loc11_32.1: <bound method> = bound_method %u.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Int.as.Negate.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Negate.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_32.2: <bound method> = bound_method %u.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call: init %i32 = call %bound_method.loc11_32.2(%u.ref)
// CHECK:STDOUT:   return %Int.as.Negate.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFunction() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.572]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %D.ref, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc15_11: %J.type = converted %D.ref, %J.facet [concrete = constants.%J.facet]
// CHECK:STDOUT:   %impl.elem1: %.7bd = impl_witness_access constants.%J.impl_witness, element1 [concrete = constants.%D.as.J.impl.F]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc15_18.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_18.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_18.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc15_18.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc15_18.2: %i32 = converted %int_4, %.loc15_18.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %D.as.J.impl.F.call: init %i32 = call %impl.elem1(%.loc15_18.2)
// CHECK:STDOUT:   return %D.as.J.impl.F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.dc6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%Self.dc6) {
// CHECK:STDOUT:   %Self => constants.%Self.dc6
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.fa9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.345
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%.Self.c58) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_with_associated_type_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.dc6: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.1fa: type = symbolic_binding_type Self, 0, %Self.dc6 [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.cd2552.1: <witness> = lookup_impl_witness %Self.dc6, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.fa9e71.1: type = impl_witness_access %J.lookup_impl_witness.cd2552.1, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.345548.1: type = pattern_type %impl.elem0.fa9e71.1 [symbolic]
// CHECK:STDOUT:   %J.F.type: type = fn_type @J.F [concrete]
// CHECK:STDOUT:   %J.F: %J.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.572: %J.assoc_type = assoc_entity element1, @J.%J.F.decl [concrete]
// CHECK:STDOUT:   %pattern_type.4c421b.1: type = pattern_type %Self.binding.as_type.1fa [symbolic]
// CHECK:STDOUT:   %J.G.type: type = fn_type @J.G [concrete]
// CHECK:STDOUT:   %J.G: %J.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc2: %J.assoc_type = assoc_entity element2, @J.%J.G.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %.Self.c58: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.c58 [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.ab8: <witness> = lookup_impl_witness %.Self.c58, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ab2: type = impl_witness_access %J.lookup_impl_witness.ab8, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.ab2 = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type: type = fn_type @E.as.J.impl.F [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F: %E.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a4a: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %E.as.J.impl.G.type: type = fn_type @E.as.J.impl.G [concrete]
// CHECK:STDOUT:   %E.as.J.impl.G: %E.as.J.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.Op.type: type = fn_type @Negate.Op [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.1d3: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.5d2: %Int.as.Negate.impl.Op.type.1d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Negate.impl_witness.bc7: <witness> = impl_witness imports.%Negate.impl_witness_table, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.6b1: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.4f9: %Int.as.Negate.impl.Op.type.6b1 = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %i32, (%Negate.impl_witness.bc7) [concrete]
// CHECK:STDOUT:   %.f7e: type = fn_type_with_self_type %Negate.Op.type, %Negate.facet [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Negate.impl.Op.4f9, @Int.as.Negate.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %CallBoth.type: type = fn_type @CallBoth [concrete]
// CHECK:STDOUT:   %CallBoth: %CallBoth.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ed8: type = fn_type_with_self_type %J.F.type, %J.facet [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9f7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %.553: type = fn_type_with_self_type %J.G.type, %J.facet [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.def: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b5: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.5dd: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.451: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.c57: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a4a: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.b96: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.84b: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4c421b.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.cd2552.2: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.fa9e71.2: type = impl_witness_access %J.lookup_impl_witness.cd2552.2, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.345548.2: type = pattern_type %impl.elem0.fa9e71.2 [symbolic]
// CHECK:STDOUT:   %GenericCallF.type: type = fn_type @GenericCallF [concrete]
// CHECK:STDOUT:   %GenericCallF: %GenericCallF.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.157: <witness> = require_complete_type %impl.elem0.fa9e71.2 [symbolic]
// CHECK:STDOUT:   %require_complete.d01: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.fe6: type = fn_type_with_self_type %J.F.type, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.fe6 = impl_witness_access %J.lookup_impl_witness.cd2552.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem1, @J.F(%T) [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericCallF.specific_fn: <specific function> = specific_function %GenericCallF, @GenericCallF(%J.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Negate = %Core.Negate
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/parts/int_literal, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.3f0 = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ca8: @Int.as.Negate.impl.%Int.as.Negate.impl.Op.type (%Int.as.Negate.impl.Op.type.1d3) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Negate.impl.%Int.as.Negate.impl.Op (constants.%Int.as.Negate.impl.Op.5d2)]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.3f0, %Core.import_ref.ca8), @Int.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .CallBoth = %CallBoth.decl
// CHECK:STDOUT:     .GenericCallF = %GenericCallF.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   %CallBoth.decl: %CallBoth.type = fn_decl @CallBoth [concrete = constants.%CallBoth] {
// CHECK:STDOUT:     %e.patt: %pattern_type.a4a = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: %pattern_type.a4a = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref.loc20: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericCallF.decl: %GenericCallF.type = fn_decl @GenericCallF [concrete = constants.%GenericCallF] {
// CHECK:STDOUT:     %T.patt: %pattern_type.84b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericCallF.%pattern_type.loc28_24 (%pattern_type.4c421b.2) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericCallF.%pattern_type.loc28_24 (%pattern_type.4c421b.2) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.345548.2) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.345548.2) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.345548.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.345548.2) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc28_41: %J.type = name_ref T, %T.loc28_17.2 [symbolic = %T.loc28_17.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc28_42: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %T.as_type.loc28_42: type = facet_access_type %T.ref.loc28_41 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc28_42: type = converted %T.ref.loc28_41, %T.as_type.loc28_42 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc28_42: type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.2, element0 [symbolic = %impl.elem0.loc28_34.1 (constants.%impl.elem0.fa9e71.2)]
// CHECK:STDOUT:     %.loc28_21: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc28_17.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc28_17.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericCallF.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc28_27.1: type = splice_block %.loc28_27.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc28_27: %J.type = name_ref T, %T.loc28_17.2 [symbolic = %T.loc28_17.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc28_27: type = facet_access_type %T.ref.loc28_27 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc28_27.2: type = converted %T.ref.loc28_27, %T.as_type.loc28_27 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericCallF.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:     %u.param: @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.fa9e71.2) = value_param call_param1
// CHECK:STDOUT:     %.loc28_34.1: type = splice_block %impl.elem0.loc28_34.2 [symbolic = %impl.elem0.loc28_34.1 (constants.%impl.elem0.fa9e71.2)] {
// CHECK:STDOUT:       %T.ref.loc28_33: %J.type = name_ref T, %T.loc28_17.2 [symbolic = %T.loc28_17.1 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc28_34: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:       %T.as_type.loc28_34: type = facet_access_type %T.ref.loc28_33 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc28_34.2: type = converted %T.ref.loc28_33, %T.as_type.loc28_34 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc28_34.2: type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.2, element0 [symbolic = %impl.elem0.loc28_34.1 (constants.%impl.elem0.fa9e71.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.fa9e71.2) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.fa9e71.2) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.fa9e71.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %e.patt: %pattern_type.a4a = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: %pattern_type.a4a = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = value_binding e, %e.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.dc6]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.F.decl: %J.F.type = fn_decl @J.F [concrete = constants.%J.F] {
// CHECK:STDOUT:     %u.patt: @J.F.%pattern_type (%pattern_type.345548.1) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.F.%pattern_type (%pattern_type.345548.1) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @J.F.%pattern_type (%pattern_type.345548.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.F.%pattern_type (%pattern_type.345548.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:     %u.param: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9e71.1) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9e71.1)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9e71.1) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9e71.1) = out_param call_param1
// CHECK:STDOUT:     %return: ref @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9e71.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.F.decl [concrete = constants.%assoc1.572]
// CHECK:STDOUT:   %J.G.decl: %J.G.type = fn_decl @J.G [concrete = constants.%J.G] {
// CHECK:STDOUT:     %self.patt: @J.G.%pattern_type.loc6_8 (%pattern_type.4c421b.1) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @J.G.%pattern_type.loc6_8 (%pattern_type.4c421b.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: @J.G.%pattern_type.loc6_20 (%pattern_type.345548.1) = value_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: @J.G.%pattern_type.loc6_20 (%pattern_type.345548.1) = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @J.G.%pattern_type.loc6_20 (%pattern_type.345548.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.G.%pattern_type.loc6_20 (%pattern_type.345548.1) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc6_29: type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.1, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:     %U.ref.loc6_29: type = name_ref U, %impl.elem0.loc6_29 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:     %self.param: @J.G.%Self.binding.as_type (%Self.binding.as_type.1fa) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1fa)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.dc6)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1fa)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1fa)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J.G.%Self.binding.as_type (%Self.binding.as_type.1fa) = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: @J.G.%impl.elem0.loc6_23.1 (%impl.elem0.fa9e71.1) = value_param call_param1
// CHECK:STDOUT:     %.loc6_23: type = splice_block %U.ref.loc6_23 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fa9e71.1)] {
// CHECK:STDOUT:       %impl.elem0.loc6_23.2: type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.1, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:       %U.ref.loc6_23: type = name_ref U, %impl.elem0.loc6_23.2 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @J.G.%impl.elem0.loc6_23.1 (%impl.elem0.fa9e71.1) = value_binding v, %v.param
// CHECK:STDOUT:     %return.param: ref @J.G.%impl.elem0.loc6_23.1 (%impl.elem0.fa9e71.1) = out_param call_param2
// CHECK:STDOUT:     %return: ref @J.G.%impl.elem0.loc6_23.1 (%impl.elem0.fa9e71.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc2: %J.assoc_type = assoc_entity element2, %J.G.decl [concrete = constants.%assoc2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   .G = %assoc2
// CHECK:STDOUT:   witness = (%U, %J.F.decl, %J.G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.J.impl: %Self.ref as %.loc10_20 {
// CHECK:STDOUT:   %E.as.J.impl.F.decl: %E.as.J.impl.F.type = fn_decl @E.as.J.impl.F [concrete = constants.%E.as.J.impl.F] {
// CHECK:STDOUT:     %u.patt: %pattern_type.7ce = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.7ce = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %i32.loc11_13 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.as.J.impl.G.decl: %E.as.J.impl.G.type = fn_decl @E.as.J.impl.G [concrete = constants.%E.as.J.impl.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a4a = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a4a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = value_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: %pattern_type.7ce = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc14_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %E = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %self: %E = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc14: type = splice_block %i32.loc14_25 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc14_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: %i32 = value_binding v, %v.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %E.as.J.impl.F.decl
// CHECK:STDOUT:   .G = %E.as.J.impl.G.decl
// CHECK:STDOUT:   witness = @E.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @E.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.c58]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.c58]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_26: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.ab8, element0 [symbolic_self = constants.%impl.elem0.ab2]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_20: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @E.as.J.impl.%E.as.J.impl.F.decl, @E.as.J.impl.%E.as.J.impl.G.decl), @E.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   extend @E.as.J.impl.%.loc10_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.dc6)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.cd2552.1)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.345548.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9e71.1)) -> @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9e71.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.G(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.dc6)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1fa)]
// CHECK:STDOUT:   %pattern_type.loc6_8: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc6_8 (constants.%pattern_type.4c421b.1)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.cd2552.1)]
// CHECK:STDOUT:   %impl.elem0.loc6_23.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:   %pattern_type.loc6_20: type = pattern_type %impl.elem0.loc6_23.1 [symbolic = %pattern_type.loc6_20 (constants.%pattern_type.345548.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J.G.%Self.binding.as_type (%Self.binding.as_type.1fa), %v.param: @J.G.%impl.elem0.loc6_23.1 (%impl.elem0.fa9e71.1)) -> @J.G.%impl.elem0.loc6_23.1 (%impl.elem0.fa9e71.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.F(%u.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %impl.elem1: %.f7e = impl_witness_access constants.%Negate.impl_witness.bc7, element1 [concrete = constants.%Int.as.Negate.impl.Op.4f9]
// CHECK:STDOUT:   %bound_method.loc12_14.1: <bound method> = bound_method %u.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Int.as.Negate.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Negate.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_14.2: <bound method> = bound_method %u.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call: init %i32 = call %bound_method.loc12_14.2(%u.ref)
// CHECK:STDOUT:   return %Int.as.Negate.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.G(%self.param: %E, %v.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: %i32 = name_ref v, %v
// CHECK:STDOUT:   %impl.elem1: %.f7e = impl_witness_access constants.%Negate.impl_witness.bc7, element1 [concrete = constants.%Int.as.Negate.impl.Op.4f9]
// CHECK:STDOUT:   %bound_method.loc15_14.1: <bound method> = bound_method %v.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Int.as.Negate.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Negate.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_14.2: <bound method> = bound_method %v.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call: init %i32 = call %bound_method.loc15_14.2(%v.ref)
// CHECK:STDOUT:   return %Int.as.Negate.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallBoth(%e.param: %E) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e1.patt: %pattern_type.7ce = value_binding_pattern e1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc21: %E = name_ref e, %e
// CHECK:STDOUT:   %F.ref.loc21: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.572]
// CHECK:STDOUT:   %impl.elem1.loc21: %.ed8 = impl_witness_access constants.%J.impl_witness, element1 [concrete = constants.%E.as.J.impl.F]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc21: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc21_21.1: <bound method> = bound_method %int_2, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_21.2: <bound method> = bound_method %int_2, %specific_fn.loc21 [concrete = constants.%bound_method.9f7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %i32 = call %bound_method.loc21_21.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc21_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc21_21.2: %i32 = converted %int_2, %.loc21_21.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %E.as.J.impl.F.call.loc21: init %i32 = call %impl.elem1.loc21(%.loc21_21.2)
// CHECK:STDOUT:   %.loc21_11: type = splice_block %i32.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_22.1: %i32 = value_of_initializer %E.as.J.impl.F.call.loc21
// CHECK:STDOUT:   %.loc21_22.2: %i32 = converted %E.as.J.impl.F.call.loc21, %.loc21_22.1
// CHECK:STDOUT:   %e1: %i32 = value_binding e1, %.loc21_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e2.patt: %pattern_type.7ce = value_binding_pattern e2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc22: %E = name_ref e, %e
// CHECK:STDOUT:   %G.ref.loc22: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc22: %.553 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%E.as.J.impl.G]
// CHECK:STDOUT:   %bound_method.loc22_18: <bound method> = bound_method %e.ref.loc22, %impl.elem2.loc22
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc22: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc22_21.1: <bound method> = bound_method %int_3, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_21.2: <bound method> = bound_method %int_3, %specific_fn.loc22 [concrete = constants.%bound_method.def]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %i32 = call %bound_method.loc22_21.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22_21.2: %i32 = converted %int_3, %.loc22_21.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %E.as.J.impl.G.call.loc22: init %i32 = call %bound_method.loc22_18(%e.ref.loc22, %.loc22_21.2)
// CHECK:STDOUT:   %.loc22_11: type = splice_block %i32.loc22 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_22.1: %i32 = value_of_initializer %E.as.J.impl.G.call.loc22
// CHECK:STDOUT:   %.loc22_22.2: %i32 = converted %E.as.J.impl.G.call.loc22, %.loc22_22.1
// CHECK:STDOUT:   %e2: %i32 = value_binding e2, %.loc22_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e3.patt: %pattern_type.7ce = value_binding_pattern e3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.ref.loc23: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %F.ref.loc23: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.572]
// CHECK:STDOUT:   %impl.elem1.loc23: %.ed8 = impl_witness_access constants.%J.impl_witness, element1 [concrete = constants.%E.as.J.impl.F]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc23: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc23_21.1: <bound method> = bound_method %int_4, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b5]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_21.2: <bound method> = bound_method %int_4, %specific_fn.loc23 [concrete = constants.%bound_method.5dd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %i32 = call %bound_method.loc23_21.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc23_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc23_21.2: %i32 = converted %int_4, %.loc23_21.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %E.as.J.impl.F.call.loc23: init %i32 = call %impl.elem1.loc23(%.loc23_21.2)
// CHECK:STDOUT:   %.loc23_11: type = splice_block %i32.loc23 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_22.1: %i32 = value_of_initializer %E.as.J.impl.F.call.loc23
// CHECK:STDOUT:   %.loc23_22.2: %i32 = converted %E.as.J.impl.F.call.loc23, %.loc23_22.1
// CHECK:STDOUT:   %e3: %i32 = value_binding e3, %.loc23_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e4.patt: %pattern_type.7ce = value_binding_pattern e4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc24: %E = name_ref e, %e
// CHECK:STDOUT:   %E.ref.loc24: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %G.ref.loc24: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc24: %.553 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%E.as.J.impl.G]
// CHECK:STDOUT:   %bound_method.loc24_18: <bound method> = bound_method %e.ref.loc24, %impl.elem2.loc24
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc24: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc24_25.1: <bound method> = bound_method %int_5, %impl.elem0.loc24 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.451]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_25.2: <bound method> = bound_method %int_5, %specific_fn.loc24 [concrete = constants.%bound_method.c57]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24: init %i32 = call %bound_method.loc24_25.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc24_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc24_25.2: %i32 = converted %int_5, %.loc24_25.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %E.as.J.impl.G.call.loc24: init %i32 = call %bound_method.loc24_18(%e.ref.loc24, %.loc24_25.2)
// CHECK:STDOUT:   %.loc24_11: type = splice_block %i32.loc24 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24_26.1: %i32 = value_of_initializer %E.as.J.impl.G.call.loc24
// CHECK:STDOUT:   %.loc24_26.2: %i32 = converted %E.as.J.impl.G.call.loc24, %.loc24_26.1
// CHECK:STDOUT:   %e4: %i32 = value_binding e4, %.loc24_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e5.patt: %pattern_type.7ce = value_binding_pattern e5 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc25: %E = name_ref e, %e
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %G.ref.loc25: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc25: %.553 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%E.as.J.impl.G]
// CHECK:STDOUT:   %bound_method.loc25_18: <bound method> = bound_method %e.ref.loc25, %impl.elem2.loc25
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %impl.elem0.loc25: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc25_25.1: <bound method> = bound_method %int_6, %impl.elem0.loc25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a4a]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_25.2: <bound method> = bound_method %int_6, %specific_fn.loc25 [concrete = constants.%bound_method.b96]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25: init %i32 = call %bound_method.loc25_25.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc25_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc25_25.2: %i32 = converted %int_6, %.loc25_25.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %E.as.J.impl.G.call.loc25: init %i32 = call %bound_method.loc25_18(%e.ref.loc25, %.loc25_25.2)
// CHECK:STDOUT:   %.loc25_11: type = splice_block %i32.loc25 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25_26.1: %i32 = value_of_initializer %E.as.J.impl.G.call.loc25
// CHECK:STDOUT:   %.loc25_26.2: %i32 = converted %E.as.J.impl.G.call.loc25, %.loc25_26.1
// CHECK:STDOUT:   %e5: %i32 = value_binding e5, %.loc25_26.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallF(%T.loc28_17.2: %J.type) {
// CHECK:STDOUT:   %T.loc28_17.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc28_17.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc28_17.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc28_24: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc28_24 (constants.%pattern_type.4c421b.2)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc28_17.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.cd2552.2)]
// CHECK:STDOUT:   %impl.elem0.loc28_34.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_34.1 (constants.%impl.elem0.fa9e71.2)]
// CHECK:STDOUT:   %pattern_type.loc28_30: type = pattern_type %impl.elem0.loc28_34.1 [symbolic = %pattern_type.loc28_30 (constants.%pattern_type.345548.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28_38: <witness> = require_complete_type %impl.elem0.loc28_34.1 [symbolic = %require_complete.loc28_38 (constants.%require_complete.157)]
// CHECK:STDOUT:   %require_complete.loc28_25: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc28_25 (constants.%require_complete.d01)]
// CHECK:STDOUT:   %.loc29_11: type = fn_type_with_self_type constants.%J.F.type, %T.loc28_17.1 [symbolic = %.loc29_11 (constants.%.fe6)]
// CHECK:STDOUT:   %impl.elem1.loc29_11.2: @GenericCallF.%.loc29_11 (%.fe6) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc29_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc29_11.2: <specific function> = specific_impl_function %impl.elem1.loc29_11.2, @J.F(%T.loc28_17.1) [symbolic = %specific_impl_fn.loc29_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericCallF.%T.binding.as_type (%T.binding.as_type), %u.param: @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.fa9e71.2)) -> %return.param: @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.fa9e71.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallF.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.572]
// CHECK:STDOUT:     %impl.elem1.loc29_11.1: @GenericCallF.%.loc29_11 (%.fe6) = impl_witness_access constants.%J.lookup_impl_witness.cd2552.2, element1 [symbolic = %impl.elem1.loc29_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref: @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.fa9e71.2) = name_ref u, %u
// CHECK:STDOUT:     %specific_impl_fn.loc29_11.1: <specific function> = specific_impl_function %impl.elem1.loc29_11.1, @J.F(constants.%T) [symbolic = %specific_impl_fn.loc29_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc28_38: ref @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.fa9e71.2) = splice_block %return {}
// CHECK:STDOUT:     %.loc29_15: init @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.fa9e71.2) = call %specific_impl_fn.loc29_11.1(%u.ref) to %.loc28_38
// CHECK:STDOUT:     return %.loc29_15 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGeneric(%e.param: %E) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %GenericCallF.ref: %GenericCallF.type = name_ref GenericCallF, file.%GenericCallF.decl [concrete = constants.%GenericCallF]
// CHECK:STDOUT:   %e.ref: %E = name_ref e, %e
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %J.facet.loc33_27.1: %J.type = facet_value constants.%E, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc33_27.1: %J.type = converted constants.%E, %J.facet.loc33_27.1 [concrete = constants.%J.facet]
// CHECK:STDOUT:   %J.facet.loc33_27.2: %J.type = facet_value constants.%E, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc33_27.2: %J.type = converted constants.%E, %J.facet.loc33_27.2 [concrete = constants.%J.facet]
// CHECK:STDOUT:   %GenericCallF.specific_fn: <specific function> = specific_function %GenericCallF.ref, @GenericCallF(constants.%J.facet) [concrete = constants.%GenericCallF.specific_fn]
// CHECK:STDOUT:   %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc33_26.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_26.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.9f7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc33_26.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc33_26.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc33_26.2: %i32 = converted %int_2, %.loc33_26.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %GenericCallF.call: init %i32 = call %GenericCallF.specific_fn(%e.ref, %.loc33_26.2)
// CHECK:STDOUT:   return %GenericCallF.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.dc6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%Self.dc6) {
// CHECK:STDOUT:   %Self => constants.%Self.dc6
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2552.1
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.fa9e71.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.345548.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.G(constants.%Self.dc6) {
// CHECK:STDOUT:   %Self => constants.%Self.dc6
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.1fa
// CHECK:STDOUT:   %pattern_type.loc6_8 => constants.%pattern_type.4c421b.1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2552.1
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%impl.elem0.fa9e71.1
// CHECK:STDOUT:   %pattern_type.loc6_20 => constants.%pattern_type.345548.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%.Self.c58) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.G(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%E
// CHECK:STDOUT:   %pattern_type.loc6_8 => constants.%pattern_type.a4a
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc6_20 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallF(constants.%T) {
// CHECK:STDOUT:   %T.loc28_17.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc28_24 => constants.%pattern_type.4c421b.2
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2552.2
// CHECK:STDOUT:   %impl.elem0.loc28_34.1 => constants.%impl.elem0.fa9e71.2
// CHECK:STDOUT:   %pattern_type.loc28_30 => constants.%pattern_type.345548.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2552.2
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.fa9e71.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.345548.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallF(constants.%J.facet) {
// CHECK:STDOUT:   %T.loc28_17.1 => constants.%J.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%E
// CHECK:STDOUT:   %pattern_type.loc28_24 => constants.%pattern_type.a4a
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc28_34.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc28_30 => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28_38 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc28_25 => constants.%complete_type.357
// CHECK:STDOUT:   %.loc29_11 => constants.%.ed8
// CHECK:STDOUT:   %impl.elem1.loc29_11.2 => constants.%E.as.J.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc29_11.2 => constants.%E.as.J.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_constraint_on_associated_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.9f2: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.bcb: type = symbolic_binding_type Self, 0, %Self.9f2 [symbolic]
// CHECK:STDOUT:   %pattern_type.89c: type = pattern_type %Self.binding.as_type.bcb [symbolic]
// CHECK:STDOUT:   %I.Op.type: type = fn_type @I.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.Op: %I.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.ea8: %I.assoc_type = assoc_entity element0, @I.%I.Op.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.dc6: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @Destroy> [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.d3b: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %J.lookup_impl_witness.cd2552.1: <witness> = lookup_impl_witness %Self.dc6, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.63dbfc.1: %facet_type = impl_witness_access %J.lookup_impl_witness.cd2552.1, element0 [symbolic]
// CHECK:STDOUT:   %as_type.9e6a88.1: type = facet_access_type %impl.elem0.63dbfc.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.49a6ab.1: type = pattern_type %as_type.9e6a88.1 [symbolic]
// CHECK:STDOUT:   %J.F.type: type = fn_type @J.F [concrete]
// CHECK:STDOUT:   %J.F: %J.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%J.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.84b: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4c4: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.cd2552.2: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.63dbfc.2: %facet_type = impl_witness_access %J.lookup_impl_witness.cd2552.2, element0 [symbolic]
// CHECK:STDOUT:   %as_type.9e6a88.2: type = facet_access_type %impl.elem0.63dbfc.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.49a6ab.2: type = pattern_type %as_type.9e6a88.2 [symbolic]
// CHECK:STDOUT:   %GenericResult.type: type = fn_type @GenericResult [concrete]
// CHECK:STDOUT:   %GenericResult: %GenericResult.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.565: <witness> = require_complete_type %as_type.9e6a88.2 [symbolic]
// CHECK:STDOUT:   %require_complete.d01: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.fe6: type = fn_type_with_self_type %J.F.type, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.fe6 = impl_witness_access %J.lookup_impl_witness.cd2552.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2a8: <specific function> = specific_impl_function %impl.elem1, @J.F(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.63dbfc.2, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %as_type.9e6a88.2, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.e95: type = fn_type_with_self_type %I.Op.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.4d2: %.e95 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4cf: <specific function> = specific_impl_function %impl.elem0.4d2, @I.Op(%I.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.63dbfc.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.9e6a88.2, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.42c: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.6c4: %.42c = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %ptr.ade: type = ptr_type %as_type.9e6a88.2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6c6: <specific function> = specific_impl_function %impl.elem0.6c4, @Destroy.Op(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %require_complete.009: <witness> = require_complete_type %ptr.ade [symbolic]
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
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .GenericResult = %GenericResult.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %GenericResult.decl: %GenericResult.type = fn_decl @GenericResult [concrete = constants.%GenericResult] {
// CHECK:STDOUT:     %T.patt: %pattern_type.84b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericResult.%pattern_type.loc12_25 (%pattern_type.4c4) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericResult.%pattern_type.loc12_25 (%pattern_type.4c4) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.49a6ab.2) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.49a6ab.2) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.49a6ab.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.49a6ab.2) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_42: %J.type = name_ref T, %T.loc12_18.2 [symbolic = %T.loc12_18.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc12_43: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.d3b]
// CHECK:STDOUT:     %T.as_type.loc12_43: type = facet_access_type %T.ref.loc12_42 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc12_43.1: type = converted %T.ref.loc12_42, %T.as_type.loc12_43 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc12_43: %facet_type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.2, element0 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.63dbfc.2)]
// CHECK:STDOUT:     %as_type.loc12_43: type = facet_access_type %impl.elem0.loc12_43 [symbolic = %as_type.loc12_35.1 (constants.%as_type.9e6a88.2)]
// CHECK:STDOUT:     %.loc12_43.2: type = converted %impl.elem0.loc12_43, %as_type.loc12_43 [symbolic = %as_type.loc12_35.1 (constants.%as_type.9e6a88.2)]
// CHECK:STDOUT:     %.loc12_22: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_18.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_18.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericResult.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc12_28.1: type = splice_block %.loc12_28.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc12_28: %J.type = name_ref T, %T.loc12_18.2 [symbolic = %T.loc12_18.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc12_28: type = facet_access_type %T.ref.loc12_28 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc12_28.2: type = converted %T.ref.loc12_28, %T.as_type.loc12_28 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericResult.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:     %u.param: @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = value_param call_param1
// CHECK:STDOUT:     %.loc12_35.1: type = splice_block %.loc12_35.3 [symbolic = %as_type.loc12_35.1 (constants.%as_type.9e6a88.2)] {
// CHECK:STDOUT:       %T.ref.loc12_34: %J.type = name_ref T, %T.loc12_18.2 [symbolic = %T.loc12_18.1 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc12_35: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.d3b]
// CHECK:STDOUT:       %T.as_type.loc12_35: type = facet_access_type %T.ref.loc12_34 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc12_35.2: type = converted %T.ref.loc12_34, %T.as_type.loc12_35 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc12_35.2: %facet_type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.2, element0 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.63dbfc.2)]
// CHECK:STDOUT:       %as_type.loc12_35.2: type = facet_access_type %impl.elem0.loc12_35.2 [symbolic = %as_type.loc12_35.1 (constants.%as_type.9e6a88.2)]
// CHECK:STDOUT:       %.loc12_35.3: type = converted %impl.elem0.loc12_35.2, %as_type.loc12_35.2 [symbolic = %as_type.loc12_35.1 (constants.%as_type.9e6a88.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.9f2]
// CHECK:STDOUT:   %I.Op.decl: %I.Op.type = fn_decl @I.Op [concrete = constants.%I.Op] {
// CHECK:STDOUT:     %self.patt: @I.Op.%pattern_type (%pattern_type.89c) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @I.Op.%pattern_type (%pattern_type.89c) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: @I.Op.%pattern_type (%pattern_type.89c) = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: @I.Op.%pattern_type (%pattern_type.89c) = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @I.Op.%pattern_type (%pattern_type.89c) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @I.Op.%pattern_type (%pattern_type.89c) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc4_33: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:     %Self.as_type.loc4_33: type = facet_access_type %Self.ref.loc4_33 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:     %.loc4_33: type = converted %Self.ref.loc4_33, %Self.as_type.loc4_33 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:     %self.param: @I.Op.%Self.binding.as_type (%Self.binding.as_type.bcb) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)] {
// CHECK:STDOUT:       %Self.ref.loc4_15: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:       %Self.as_type.loc4_15: type = facet_access_type %Self.ref.loc4_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref.loc4_15, %Self.as_type.loc4_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.Op.%Self.binding.as_type (%Self.binding.as_type.bcb) = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: @I.Op.%Self.binding.as_type (%Self.binding.as_type.bcb) = value_param call_param1
// CHECK:STDOUT:     %.loc4_24.1: type = splice_block %.loc4_24.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)] {
// CHECK:STDOUT:       %Self.ref.loc4_24: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:       %Self.as_type.loc4_24: type = facet_access_type %Self.ref.loc4_24 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:       %.loc4_24.2: type = converted %Self.ref.loc4_24, %Self.as_type.loc4_24 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: @I.Op.%Self.binding.as_type (%Self.binding.as_type.bcb) = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref @I.Op.%Self.binding.as_type (%Self.binding.as_type.bcb) = out_param call_param2
// CHECK:STDOUT:     %return: ref @I.Op.%Self.binding.as_type (%Self.binding.as_type.bcb) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.Op.decl [concrete = constants.%assoc0.ea8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%I.Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.dc6]
// CHECK:STDOUT:   %U: %facet_type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.d3b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.F.decl: %J.F.type = fn_decl @J.F [concrete = constants.%J.F] {
// CHECK:STDOUT:     %u.patt: @J.F.%pattern_type (%pattern_type.49a6ab.1) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.F.%pattern_type (%pattern_type.49a6ab.1) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @J.F.%pattern_type (%pattern_type.49a6ab.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.F.%pattern_type (%pattern_type.49a6ab.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc9_17: %facet_type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.1, element0 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.63dbfc.1)]
// CHECK:STDOUT:     %U.ref.loc9_17: %facet_type = name_ref U, %impl.elem0.loc9_17 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.63dbfc.1)]
// CHECK:STDOUT:     %U.as_type.loc9_17: type = facet_access_type %U.ref.loc9_17 [symbolic = %as_type (constants.%as_type.9e6a88.1)]
// CHECK:STDOUT:     %.loc9_17: type = converted %U.ref.loc9_17, %U.as_type.loc9_17 [symbolic = %as_type (constants.%as_type.9e6a88.1)]
// CHECK:STDOUT:     %u.param: @J.F.%as_type (%as_type.9e6a88.1) = value_param call_param0
// CHECK:STDOUT:     %.loc9_11.1: type = splice_block %.loc9_11.2 [symbolic = %as_type (constants.%as_type.9e6a88.1)] {
// CHECK:STDOUT:       %impl.elem0.loc9_11.2: %facet_type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.1, element0 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.63dbfc.1)]
// CHECK:STDOUT:       %U.ref.loc9_11: %facet_type = name_ref U, %impl.elem0.loc9_11.2 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.63dbfc.1)]
// CHECK:STDOUT:       %U.as_type.loc9_11: type = facet_access_type %U.ref.loc9_11 [symbolic = %as_type (constants.%as_type.9e6a88.1)]
// CHECK:STDOUT:       %.loc9_11.2: type = converted %U.ref.loc9_11, %U.as_type.loc9_11 [symbolic = %as_type (constants.%as_type.9e6a88.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.F.%as_type (%as_type.9e6a88.1) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.F.%as_type (%as_type.9e6a88.1) = out_param call_param1
// CHECK:STDOUT:     %return: ref @J.F.%as_type (%as_type.9e6a88.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! %facet_type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.Op(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.89c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.Op.%Self.binding.as_type (%Self.binding.as_type.bcb), %b.param: @I.Op.%Self.binding.as_type (%Self.binding.as_type.bcb)) -> @I.Op.%Self.binding.as_type (%Self.binding.as_type.bcb);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.dc6)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.cd2552.1)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.1: %facet_type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.63dbfc.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc9_11.1 [symbolic = %as_type (constants.%as_type.9e6a88.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %as_type [symbolic = %pattern_type (constants.%pattern_type.49a6ab.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @J.F.%as_type (%as_type.9e6a88.1)) -> @J.F.%as_type (%as_type.9e6a88.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericResult(%T.loc12_18.2: %J.type) {
// CHECK:STDOUT:   %T.loc12_18.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_18.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc12_18.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc12_25: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc12_25 (constants.%pattern_type.4c4)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc12_18.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.cd2552.2)]
// CHECK:STDOUT:   %impl.elem0.loc12_35.1: %facet_type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.63dbfc.2)]
// CHECK:STDOUT:   %as_type.loc12_35.1: type = facet_access_type %impl.elem0.loc12_35.1 [symbolic = %as_type.loc12_35.1 (constants.%as_type.9e6a88.2)]
// CHECK:STDOUT:   %pattern_type.loc12_31: type = pattern_type %as_type.loc12_35.1 [symbolic = %pattern_type.loc12_31 (constants.%pattern_type.49a6ab.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_39: <witness> = require_complete_type %as_type.loc12_35.1 [symbolic = %require_complete.loc12_39 (constants.%require_complete.565)]
// CHECK:STDOUT:   %require_complete.loc12_26: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc12_26 (constants.%require_complete.d01)]
// CHECK:STDOUT:   %.loc13_11: type = fn_type_with_self_type constants.%J.F.type, %T.loc12_18.1 [symbolic = %.loc13_11 (constants.%.fe6)]
// CHECK:STDOUT:   %impl.elem1.loc13_11.2: @GenericResult.%.loc13_11 (%.fe6) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc13_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_11.2: <specific function> = specific_impl_function %impl.elem1.loc13_11.2, @J.F(%T.loc12_18.1) [symbolic = %specific_impl_fn.loc13_11.2 (constants.%specific_impl_fn.2a8)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc12_35.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet.loc13_16: %I.type = facet_value %as_type.loc12_35.1, (%I.lookup_impl_witness) [symbolic = %I.facet.loc13_16 (constants.%I.facet)]
// CHECK:STDOUT:   %.loc13_16: type = fn_type_with_self_type constants.%I.Op.type, %I.facet.loc13_16 [symbolic = %.loc13_16 (constants.%.e95)]
// CHECK:STDOUT:   %impl.elem0.loc13_16.2: @GenericResult.%.loc13_16 (%.e95) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_16.2 (constants.%impl.elem0.4d2)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_16.2: <specific function> = specific_impl_function %impl.elem0.loc13_16.2, @I.Op(%I.facet.loc13_16) [symbolic = %specific_impl_fn.loc13_16.2 (constants.%specific_impl_fn.4cf)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc12_35.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.loc12_35.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %.loc13_29.8: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc13_29.8 (constants.%.42c)]
// CHECK:STDOUT:   %impl.elem0.loc13_29.2: @GenericResult.%.loc13_29.8 (%.42c) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_29.2 (constants.%impl.elem0.6c4)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_29.2: <specific function> = specific_impl_function %impl.elem0.loc13_29.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc13_29.2 (constants.%specific_impl_fn.6c6)]
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type.loc12_35.1 [symbolic = %ptr (constants.%ptr.ade)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc13 (constants.%require_complete.009)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericResult.%T.binding.as_type (%T.binding.as_type), %u.param: @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2)) -> %return.param: @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericResult.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref.loc13_11: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc13_11.1: @GenericResult.%.loc13_11 (%.fe6) = impl_witness_access constants.%J.lookup_impl_witness.cd2552.2, element1 [symbolic = %impl.elem1.loc13_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref.loc13_14: @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = name_ref u, %u
// CHECK:STDOUT:     %.loc13_15.1: %facet_type = converted constants.%as_type.9e6a88.2, constants.%impl.elem0.63dbfc.2 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.63dbfc.2)]
// CHECK:STDOUT:     %.loc13_15.2: %facet_type = converted constants.%as_type.9e6a88.2, constants.%impl.elem0.63dbfc.2 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.63dbfc.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc13_11.1: <specific function> = specific_impl_function %impl.elem1.loc13_11.1, @J.F(constants.%T) [symbolic = %specific_impl_fn.loc13_11.2 (constants.%specific_impl_fn.2a8)]
// CHECK:STDOUT:     %.loc13_15.3: ref @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = temporary_storage
// CHECK:STDOUT:     %.loc13_15.4: init @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = call %specific_impl_fn.loc13_11.1(%u.ref.loc13_14) to %.loc13_15.3
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %Op.ref: %I.assoc_type = name_ref Op, @I.%assoc0 [concrete = constants.%assoc0.ea8]
// CHECK:STDOUT:     %impl.elem0.loc13_16.1: @GenericResult.%.loc13_16 (%.e95) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_16.2 (constants.%impl.elem0.4d2)]
// CHECK:STDOUT:     %bound_method.loc13_16: <bound method> = bound_method %.loc13_15.4, %impl.elem0.loc13_16.1
// CHECK:STDOUT:     %T.ref.loc13: %J.type = name_ref T, %T.loc12_18.2 [symbolic = %T.loc12_18.1 (constants.%T)]
// CHECK:STDOUT:     %F.ref.loc13_25: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc13: type = facet_access_type %T.ref.loc13 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc13_25: type = converted %T.ref.loc13, %T.as_type.loc13 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %impl.elem1.loc13_25: @GenericResult.%.loc13_11 (%.fe6) = impl_witness_access constants.%J.lookup_impl_witness.cd2552.2, element1 [symbolic = %impl.elem1.loc13_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref.loc13_28: @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = name_ref u, %u
// CHECK:STDOUT:     %.loc13_29.1: %facet_type = converted constants.%as_type.9e6a88.2, constants.%impl.elem0.63dbfc.2 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.63dbfc.2)]
// CHECK:STDOUT:     %.loc13_29.2: %facet_type = converted constants.%as_type.9e6a88.2, constants.%impl.elem0.63dbfc.2 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.63dbfc.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc13_25: <specific function> = specific_impl_function %impl.elem1.loc13_25, @J.F(constants.%T) [symbolic = %specific_impl_fn.loc13_11.2 (constants.%specific_impl_fn.2a8)]
// CHECK:STDOUT:     %.loc13_29.3: ref @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = temporary_storage
// CHECK:STDOUT:     %.loc13_29.4: init @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = call %specific_impl_fn.loc13_25(%u.ref.loc13_28) to %.loc13_29.3
// CHECK:STDOUT:     %I.facet.loc13_30.1: %I.type = facet_value constants.%as_type.9e6a88.2, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc13_16 (constants.%I.facet)]
// CHECK:STDOUT:     %.loc13_30.1: %I.type = converted constants.%as_type.9e6a88.2, %I.facet.loc13_30.1 [symbolic = %I.facet.loc13_16 (constants.%I.facet)]
// CHECK:STDOUT:     %I.facet.loc13_30.2: %I.type = facet_value constants.%as_type.9e6a88.2, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc13_16 (constants.%I.facet)]
// CHECK:STDOUT:     %.loc13_30.2: %I.type = converted constants.%as_type.9e6a88.2, %I.facet.loc13_30.2 [symbolic = %I.facet.loc13_16 (constants.%I.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc13_16.1: <specific function> = specific_impl_function %impl.elem0.loc13_16.1, @I.Op(constants.%I.facet) [symbolic = %specific_impl_fn.loc13_16.2 (constants.%specific_impl_fn.4cf)]
// CHECK:STDOUT:     %bound_method.loc13_30: <bound method> = bound_method %.loc13_15.4, %specific_impl_fn.loc13_16.1
// CHECK:STDOUT:     %.loc12_39: ref @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = splice_block %return {}
// CHECK:STDOUT:     %.loc13_15.5: ref @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = temporary %.loc13_15.3, %.loc13_15.4
// CHECK:STDOUT:     %.loc13_15.6: @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = acquire_value %.loc13_15.5
// CHECK:STDOUT:     %.loc13_29.5: ref @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = temporary %.loc13_29.3, %.loc13_29.4
// CHECK:STDOUT:     %.loc13_29.6: @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = acquire_value %.loc13_29.5
// CHECK:STDOUT:     %.loc13_30.3: init @GenericResult.%as_type.loc12_35.1 (%as_type.9e6a88.2) = call %bound_method.loc13_30(%.loc13_15.6, %.loc13_29.6) to %.loc12_39
// CHECK:STDOUT:     %impl.elem0.loc13_29.1: @GenericResult.%.loc13_29.8 (%.42c) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_29.2 (constants.%impl.elem0.6c4)]
// CHECK:STDOUT:     %bound_method.loc13_29.1: <bound method> = bound_method %.loc13_29.5, %impl.elem0.loc13_29.1
// CHECK:STDOUT:     %specific_impl_fn.loc13_29.1: <specific function> = specific_impl_function %impl.elem0.loc13_29.1, @Destroy.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc13_29.2 (constants.%specific_impl_fn.6c6)]
// CHECK:STDOUT:     %bound_method.loc13_29.2: <bound method> = bound_method %.loc13_29.5, %specific_impl_fn.loc13_29.1
// CHECK:STDOUT:     %addr.loc13_29: @GenericResult.%ptr (%ptr.ade) = addr_of %.loc13_29.5
// CHECK:STDOUT:     %.loc13_29.7: init %empty_tuple.type = call %bound_method.loc13_29.2(%addr.loc13_29)
// CHECK:STDOUT:     %impl.elem0.loc13_15: @GenericResult.%.loc13_29.8 (%.42c) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_29.2 (constants.%impl.elem0.6c4)]
// CHECK:STDOUT:     %bound_method.loc13_15.1: <bound method> = bound_method %.loc13_15.5, %impl.elem0.loc13_15
// CHECK:STDOUT:     %specific_impl_fn.loc13_15: <specific function> = specific_impl_function %impl.elem0.loc13_15, @Destroy.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc13_29.2 (constants.%specific_impl_fn.6c6)]
// CHECK:STDOUT:     %bound_method.loc13_15.2: <bound method> = bound_method %.loc13_15.5, %specific_impl_fn.loc13_15
// CHECK:STDOUT:     %addr.loc13_15: @GenericResult.%ptr (%ptr.ade) = addr_of %.loc13_15.5
// CHECK:STDOUT:     %.loc13_15.7: init %empty_tuple.type = call %bound_method.loc13_15.2(%addr.loc13_15)
// CHECK:STDOUT:     return %.loc13_30.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.Op(constants.%Self.9f2) {
// CHECK:STDOUT:   %Self => constants.%Self.9f2
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.bcb
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.89c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.dc6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%Self.dc6) {
// CHECK:STDOUT:   %Self => constants.%Self.dc6
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2552.1
// CHECK:STDOUT:   %impl.elem0.loc9_11.1 => constants.%impl.elem0.63dbfc.1
// CHECK:STDOUT:   %as_type => constants.%as_type.9e6a88.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.49a6ab.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericResult(constants.%T) {
// CHECK:STDOUT:   %T.loc12_18.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc12_25 => constants.%pattern_type.4c4
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2552.2
// CHECK:STDOUT:   %impl.elem0.loc12_35.1 => constants.%impl.elem0.63dbfc.2
// CHECK:STDOUT:   %as_type.loc12_35.1 => constants.%as_type.9e6a88.2
// CHECK:STDOUT:   %pattern_type.loc12_31 => constants.%pattern_type.49a6ab.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2552.2
// CHECK:STDOUT:   %impl.elem0.loc9_11.1 => constants.%impl.elem0.63dbfc.2
// CHECK:STDOUT:   %as_type => constants.%as_type.9e6a88.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.49a6ab.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.Op(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%as_type.9e6a88.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.49a6ab.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface_qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.4c421b.1: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.cd2552.1: <witness> = lookup_impl_witness %Self, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.fa9e71.1: type = impl_witness_access %J.lookup_impl_witness.cd2552.1, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.345548.1: type = pattern_type %impl.elem0.fa9e71.1 [symbolic]
// CHECK:STDOUT:   %J.G.type: type = fn_type @J.G [concrete]
// CHECK:STDOUT:   %J.G: %J.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%J.G.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.84b: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4c421b.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.cd2552.2: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.fa9e71.2: type = impl_witness_access %J.lookup_impl_witness.cd2552.2, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.345548.2: type = pattern_type %impl.elem0.fa9e71.2 [symbolic]
// CHECK:STDOUT:   %GenericCallInterfaceQualified.type: type = fn_type @GenericCallInterfaceQualified [concrete]
// CHECK:STDOUT:   %GenericCallInterfaceQualified: %GenericCallInterfaceQualified.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.157: <witness> = require_complete_type %impl.elem0.fa9e71.2 [symbolic]
// CHECK:STDOUT:   %require_complete.d01: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.a8d: type = fn_type_with_self_type %J.G.type, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.a8d = impl_witness_access %J.lookup_impl_witness.cd2552.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem1, @J.G(%T) [symbolic]
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
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .GenericCallInterfaceQualified = %GenericCallInterfaceQualified.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %GenericCallInterfaceQualified.decl: %GenericCallInterfaceQualified.type = fn_decl @GenericCallInterfaceQualified [concrete = constants.%GenericCallInterfaceQualified] {
// CHECK:STDOUT:     %T.patt: %pattern_type.84b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericCallInterfaceQualified.%pattern_type.loc8_41 (%pattern_type.4c421b.2) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericCallInterfaceQualified.%pattern_type.loc8_41 (%pattern_type.4c421b.2) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.345548.2) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.345548.2) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.345548.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.345548.2) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_58: %J.type = name_ref T, %T.loc8_34.2 [symbolic = %T.loc8_34.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc8_59: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc8_59: type = facet_access_type %T.ref.loc8_58 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc8_59: type = converted %T.ref.loc8_58, %T.as_type.loc8_59 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc8_59: type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.2, element0 [symbolic = %impl.elem0.loc8_51.1 (constants.%impl.elem0.fa9e71.2)]
// CHECK:STDOUT:     %.loc8_38: type = splice_block %J.ref.loc8 [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref.loc8: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_34.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc8_34.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericCallInterfaceQualified.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_44.1: type = splice_block %.loc8_44.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc8_44: %J.type = name_ref T, %T.loc8_34.2 [symbolic = %T.loc8_34.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8_44: type = facet_access_type %T.ref.loc8_44 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc8_44.2: type = converted %T.ref.loc8_44, %T.as_type.loc8_44 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericCallInterfaceQualified.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:     %u.param: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.fa9e71.2) = value_param call_param1
// CHECK:STDOUT:     %.loc8_51.1: type = splice_block %impl.elem0.loc8_51.2 [symbolic = %impl.elem0.loc8_51.1 (constants.%impl.elem0.fa9e71.2)] {
// CHECK:STDOUT:       %T.ref.loc8_50: %J.type = name_ref T, %T.loc8_34.2 [symbolic = %T.loc8_34.1 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc8_51: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %T.as_type.loc8_51: type = facet_access_type %T.ref.loc8_50 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc8_51.2: type = converted %T.ref.loc8_50, %T.as_type.loc8_51 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc8_51.2: type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.2, element0 [symbolic = %impl.elem0.loc8_51.1 (constants.%impl.elem0.fa9e71.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.fa9e71.2) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.fa9e71.2) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.fa9e71.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.G.decl: %J.G.type = fn_decl @J.G [concrete = constants.%J.G] {
// CHECK:STDOUT:     %self.patt: @J.G.%pattern_type.loc5_8 (%pattern_type.4c421b.1) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @J.G.%pattern_type.loc5_8 (%pattern_type.4c421b.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: @J.G.%pattern_type.loc5_20 (%pattern_type.345548.1) = value_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: @J.G.%pattern_type.loc5_20 (%pattern_type.345548.1) = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @J.G.%pattern_type.loc5_20 (%pattern_type.345548.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.G.%pattern_type.loc5_20 (%pattern_type.345548.1) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.1, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:     %U.ref.loc5_29: type = name_ref U, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:     %self.param: @J.G.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J.G.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: @J.G.%impl.elem0.loc5_23.1 (%impl.elem0.fa9e71.1) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %U.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9e71.1)] {
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access constants.%J.lookup_impl_witness.cd2552.1, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:       %U.ref.loc5_23: type = name_ref U, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @J.G.%impl.elem0.loc5_23.1 (%impl.elem0.fa9e71.1) = value_binding v, %v.param
// CHECK:STDOUT:     %return.param: ref @J.G.%impl.elem0.loc5_23.1 (%impl.elem0.fa9e71.1) = out_param call_param2
// CHECK:STDOUT:     %return: ref @J.G.%impl.elem0.loc5_23.1 (%impl.elem0.fa9e71.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.G(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.4c421b.1)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.cd2552.1)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.fa9e71.1)]
// CHECK:STDOUT:   %pattern_type.loc5_20: type = pattern_type %impl.elem0.loc5_23.1 [symbolic = %pattern_type.loc5_20 (constants.%pattern_type.345548.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J.G.%Self.binding.as_type (%Self.binding.as_type), %v.param: @J.G.%impl.elem0.loc5_23.1 (%impl.elem0.fa9e71.1)) -> @J.G.%impl.elem0.loc5_23.1 (%impl.elem0.fa9e71.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallInterfaceQualified(%T.loc8_34.2: %J.type) {
// CHECK:STDOUT:   %T.loc8_34.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc8_34.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_34.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_41: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc8_41 (constants.%pattern_type.4c421b.2)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_34.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.cd2552.2)]
// CHECK:STDOUT:   %impl.elem0.loc8_51.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_51.1 (constants.%impl.elem0.fa9e71.2)]
// CHECK:STDOUT:   %pattern_type.loc8_47: type = pattern_type %impl.elem0.loc8_51.1 [symbolic = %pattern_type.loc8_47 (constants.%pattern_type.345548.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_55: <witness> = require_complete_type %impl.elem0.loc8_51.1 [symbolic = %require_complete.loc8_55 (constants.%require_complete.157)]
// CHECK:STDOUT:   %require_complete.loc8_42: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc8_42 (constants.%require_complete.d01)]
// CHECK:STDOUT:   %.loc9_11: type = fn_type_with_self_type constants.%J.G.type, %T.loc8_34.1 [symbolic = %.loc9_11 (constants.%.a8d)]
// CHECK:STDOUT:   %impl.elem1.loc9_11.2: @GenericCallInterfaceQualified.%.loc9_11 (%.a8d) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem1.loc9_11.2, @J.G(%T.loc8_34.1) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericCallInterfaceQualified.%T.binding.as_type (%T.binding.as_type), %u.param: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.fa9e71.2)) -> %return.param: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.fa9e71.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallInterfaceQualified.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %J.ref.loc9: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %G.ref: %J.assoc_type = name_ref G, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc9_11.1: @GenericCallInterfaceQualified.%.loc9_11 (%.a8d) = impl_witness_access constants.%J.lookup_impl_witness.cd2552.2, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc9_11: <bound method> = bound_method %t.ref, %impl.elem1.loc9_11.1
// CHECK:STDOUT:     %u.ref: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.fa9e71.2) = name_ref u, %u
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem1.loc9_11.1, @J.G(constants.%T) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc9_19: <bound method> = bound_method %t.ref, %specific_impl_fn.loc9_11.1
// CHECK:STDOUT:     %.loc8_55: ref @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.fa9e71.2) = splice_block %return {}
// CHECK:STDOUT:     %.loc9_19: init @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.fa9e71.2) = call %bound_method.loc9_19(%t.ref, %u.ref) to %.loc8_55
// CHECK:STDOUT:     return %.loc9_19 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.4c421b.1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2552.1
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.fa9e71.1
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.345548.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallInterfaceQualified(constants.%T) {
// CHECK:STDOUT:   %T.loc8_34.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_41 => constants.%pattern_type.4c421b.2
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2552.2
// CHECK:STDOUT:   %impl.elem0.loc8_51.1 => constants.%impl.elem0.fa9e71.2
// CHECK:STDOUT:   %pattern_type.loc8_47 => constants.%pattern_type.345548.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.G(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.4c421b.2
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2552.2
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.fa9e71.2
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.345548.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.dc6: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %J.lookup_impl_witness.cd2: <witness> = lookup_impl_witness %Self.dc6, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.fa9: type = impl_witness_access %J.lookup_impl_witness.cd2, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.345: type = pattern_type %impl.elem0.fa9 [symbolic]
// CHECK:STDOUT:   %J.F.type: type = fn_type @J.F [concrete]
// CHECK:STDOUT:   %J.F: %J.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%J.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.c58: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.c58 [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.ab8: <witness> = lookup_impl_witness %.Self.c58, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ab2: type = impl_witness_access %J.lookup_impl_witness.ab8, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.ab2 = %i32> [concrete]
// CHECK:STDOUT:   %T: %J_where.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %J_where.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.2ce: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %GenericCallFI32.type: type = fn_type @GenericCallFI32 [concrete]
// CHECK:STDOUT:   %GenericCallFI32: %GenericCallFI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.440: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.873: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.binding.as_type, (%J.lookup_impl_witness.873) [symbolic]
// CHECK:STDOUT:   %.e93: type = fn_type_with_self_type %J.F.type, %J.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.e93 = impl_witness_access %J.lookup_impl_witness.873, element1 [symbolic]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem1, @J.F(%J.facet) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .GenericCallFI32 = %GenericCallFI32.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %GenericCallFI32.decl: %GenericCallFI32.type = fn_decl @GenericCallFI32 [concrete = constants.%GenericCallFI32] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9f6 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericCallFI32.%pattern_type (%pattern_type.2ce) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericCallFI32.%pattern_type (%pattern_type.2ce) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc8_51: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_51: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_26.1: type = splice_block %.loc8_26.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.2: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.c58]
// CHECK:STDOUT:       %.Self.ref: %J.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.c58]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc8_32: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %impl.elem0.loc8: type = impl_witness_access constants.%J.lookup_impl_witness.ab8, element0 [symbolic_self = constants.%impl.elem0.ab2]
// CHECK:STDOUT:       %int_32.loc8_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc8_26.2: type = where_expr %.Self.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc8, %i32.loc8_37
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_20.2: %J_where.type = symbolic_binding T, 0 [symbolic = %T.loc8_20.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericCallFI32.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_45.1: type = splice_block %.loc8_45.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %J_where.type = name_ref T, %T.loc8_20.2 [symbolic = %T.loc8_20.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc8_45.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericCallFI32.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.dc6]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.F.decl: %J.F.type = fn_decl @J.F [concrete = constants.%J.F] {
// CHECK:STDOUT:     %u.patt: @J.F.%pattern_type (%pattern_type.345) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.F.%pattern_type (%pattern_type.345) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @J.F.%pattern_type (%pattern_type.345) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.F.%pattern_type (%pattern_type.345) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.cd2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     %u.param: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.cd2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = out_param call_param1
// CHECK:STDOUT:     %return: ref @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.dc6)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.cd2)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.345)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9)) -> @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallFI32(%T.loc8_20.2: %J_where.type) {
// CHECK:STDOUT:   %T.loc8_20.1: %J_where.type = symbolic_binding T, 0 [symbolic = %T.loc8_20.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_20.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2ce)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.440)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_20.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.873)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.binding.as_type, (%J.lookup_impl_witness) [symbolic = %J.facet (constants.%J.facet)]
// CHECK:STDOUT:   %.loc9_11: type = fn_type_with_self_type constants.%J.F.type, %J.facet [symbolic = %.loc9_11 (constants.%.e93)]
// CHECK:STDOUT:   %impl.elem1.loc9_11.2: @GenericCallFI32.%.loc9_11 (%.e93) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem1.loc9_11.2, @J.F(%J.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericCallFI32.%T.binding.as_type (%T.binding.as_type)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallFI32.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc9_11.1: @GenericCallFI32.%.loc9_11 (%.e93) = impl_witness_access constants.%J.lookup_impl_witness.873, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem1.loc9_11.1, @J.F(constants.%J.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %impl.elem0.loc9: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:     %bound_method.loc9_14.1: <bound method> = bound_method %int_2, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_14.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_14.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_14.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_14.2: %i32 = converted %int_2, %.loc9_14.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_15: init %i32 = call %specific_impl_fn.loc9_11.1(%.loc9_14.2)
// CHECK:STDOUT:     return %.loc9_15 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.dc6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%Self.dc6) {
// CHECK:STDOUT:   %Self => constants.%Self.dc6
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.fa9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.345
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%.Self.c58) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallFI32(constants.%T) {
// CHECK:STDOUT:   %T.loc8_20.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.873
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_associated_constant_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.dc6: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %J.lookup_impl_witness.cd2: <witness> = lookup_impl_witness %Self.dc6, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.fa9: type = impl_witness_access %J.lookup_impl_witness.cd2, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.345: type = pattern_type %impl.elem0.fa9 [symbolic]
// CHECK:STDOUT:   %J.F.type: type = fn_type @J.F [concrete]
// CHECK:STDOUT:   %J.F: %J.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%J.F.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %.Self: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.ab8: <witness> = lookup_impl_witness %.Self, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ab2: type = impl_witness_access %J.lookup_impl_witness.ab8, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.ab2 = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type.3487c0.1: type = fn_type @E.as.J.impl.F.loc24_21.1 [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.1487eb.1: %E.as.J.impl.F.type.3487c0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type.3487c0.2: type = fn_type @E.as.J.impl.F.loc24_21.2 [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.1487eb.2: %E.as.J.impl.F.type.3487c0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.dc6]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.F.decl: %J.F.type = fn_decl @J.F [concrete = constants.%J.F] {
// CHECK:STDOUT:     %u.patt: @J.F.%pattern_type (%pattern_type.345) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.F.%pattern_type (%pattern_type.345) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @J.F.%pattern_type (%pattern_type.345) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.F.%pattern_type (%pattern_type.345) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.cd2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     %u.param: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.cd2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = out_param call_param1
// CHECK:STDOUT:     %return: ref @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.J.impl: %Self.ref as %.loc9_20 {
// CHECK:STDOUT:   %E.as.J.impl.F.decl.loc24_21.1: %E.as.J.impl.F.type.3487c0.1 = fn_decl @E.as.J.impl.F.loc24_21.1 [concrete = constants.%E.as.J.impl.F.1487eb.1] {
// CHECK:STDOUT:     %u.patt: <error> = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: <error> = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc24_19: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.loc24_19: type = converted %U.ref.loc24_19, <error> [concrete = <error>]
// CHECK:STDOUT:     %u.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %U.ref.loc24_13: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:       %.loc24_13: type = converted %U.ref.loc24_13, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: <error> = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param1
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.as.J.impl.F.decl.loc24_21.2: %E.as.J.impl.F.type.3487c0.2 = fn_decl @E.as.J.impl.F.loc24_21.2 [concrete = constants.%E.as.J.impl.F.1487eb.2] {
// CHECK:STDOUT:     %u.patt: %pattern_type.7ce = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.7ce = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %u: %i32 = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .U = <poisoned>
// CHECK:STDOUT:   .F = %E.as.J.impl.F.decl.loc24_21.1
// CHECK:STDOUT:   witness = @E.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @E.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc9_26: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.ab8, element0 [symbolic_self = constants.%impl.elem0.ab2]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc9_20: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @E.as.J.impl.%E.as.J.impl.F.decl.loc24_21.2), @E.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .U = <poisoned>
// CHECK:STDOUT:   extend @E.as.J.impl.%.loc9_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.dc6)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.cd2)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.fa9)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.345)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9)) -> @J.F.%impl.elem0.loc5_11.1 (%impl.elem0.fa9);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.F.loc24_21.1(%u.param: <error>) -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: <error> = name_ref u, %u
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.F.loc24_21.2(%u.param: %i32) -> %i32 [thunk @E.as.J.impl.%E.as.J.impl.F.decl.loc24_21.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %E.as.J.impl.F.type.3487c0.1 = name_ref F, @E.as.J.impl.%E.as.J.impl.F.decl.loc24_21.1 [concrete = constants.%E.as.J.impl.F.1487eb.1]
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u.param
// CHECK:STDOUT:   %return.ref: ref %i32 = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %E.as.J.impl.F.call: init <error> = call %F.ref(<error>)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.dc6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%Self.dc6) {
// CHECK:STDOUT:   %Self => constants.%Self.dc6
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.cd2
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.fa9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.345
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%.Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_self_period_associated_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J2.type: type = facet_type <@J2> [concrete]
// CHECK:STDOUT:   %Self: %J2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J2.assoc_type: type = assoc_entity_type @J2 [concrete]
// CHECK:STDOUT:   %assoc0: %J2.assoc_type = assoc_entity element0, @J2.%U2 [concrete]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.0a6: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %J2.F.type: type = fn_type @J2.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %J2.F: %J2.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J2.assoc_type = assoc_entity element1, @J2.%J2.F.decl [concrete]
// CHECK:STDOUT:   %.Self: %J2.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %J2.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @J2 [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %J2.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %J2_where.type.744: type = facet_type <@J2 where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %J2.impl_witness.b89: <witness> = impl_witness file.%J2.impl_witness_table.loc22 [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.type.d9b215.1: type = fn_type @empty_tuple.type.as.J2.impl.F.loc23_33.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.b047a1.1: %empty_tuple.type.as.J2.impl.F.type.d9b215.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J2.facet.02c: %J2.type = facet_value %empty_tuple.type, (%J2.impl_witness.b89) [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.type.d9b215.2: type = fn_type @empty_tuple.type.as.J2.impl.F.loc23_33.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.b047a1.2: %empty_tuple.type.as.J2.impl.F.type.d9b215.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J2_where.type.76d: type = facet_type <@J2 where %impl.elem0 = %C2> [concrete]
// CHECK:STDOUT:   %J2.impl_witness.b77: <witness> = impl_witness file.%J2.impl_witness_table.loc31 [concrete]
// CHECK:STDOUT:   %pattern_type.838: type = pattern_type %C2 [concrete]
// CHECK:STDOUT:   %C2.as.J2.impl.F.type.5b1c29.1: type = fn_type @C2.as.J2.impl.F.loc32_33.1 [concrete]
// CHECK:STDOUT:   %C2.as.J2.impl.F.8b36c6.1: %C2.as.J2.impl.F.type.5b1c29.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J2.facet.73a: %J2.type = facet_value %C2, (%J2.impl_witness.b77) [concrete]
// CHECK:STDOUT:   %C2.as.J2.impl.F.type.5b1c29.2: type = fn_type @C2.as.J2.impl.F.loc32_33.2 [concrete]
// CHECK:STDOUT:   %C2.as.J2.impl.F.8b36c6.2: %C2.as.J2.impl.F.type.5b1c29.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C2.val: %C2 = struct_value () [concrete]
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
// CHECK:STDOUT:     .J2 = %J2.decl
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J2.decl: type = interface_decl @J2 [concrete = constants.%J2.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.J2.impl [concrete] {} {
// CHECK:STDOUT:     %.loc22_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_7.2: type = converted %.loc22_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J2.ref: type = name_ref J2, file.%J2.decl [concrete = constants.%J2.type]
// CHECK:STDOUT:     %.Self: %J2.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U2.ref: %J2.assoc_type = name_ref U2, @U2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc22_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J2.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc22_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc22_28.2: type = converted %.loc22_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc22_15: type = where_expr %.Self [concrete = constants.%J2_where.type.744] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J2.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc22_28.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J2.impl_witness_table.loc22 = impl_witness_table (%impl_witness_assoc_constant.loc22, @empty_tuple.type.as.J2.impl.%empty_tuple.type.as.J2.impl.F.decl.loc23_33.2), @empty_tuple.type.as.J2.impl [concrete]
// CHECK:STDOUT:   %J2.impl_witness.loc22: <witness> = impl_witness %J2.impl_witness_table.loc22 [concrete = constants.%J2.impl_witness.b89]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc22: type = impl_witness_assoc_constant constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [concrete = constants.%C2] {} {}
// CHECK:STDOUT:   impl_decl @C2.as.J2.impl [concrete] {} {
// CHECK:STDOUT:     %C2.ref.loc31_6: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %J2.ref: type = name_ref J2, file.%J2.decl [concrete = constants.%J2.type]
// CHECK:STDOUT:     %.Self: %J2.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U2.ref: %J2.assoc_type = name_ref U2, @U2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc31_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J2.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C2.ref.loc31_27: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %.loc31_15: type = where_expr %.Self [concrete = constants.%J2_where.type.76d] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J2.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C2.ref.loc31_27
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J2.impl_witness_table.loc31 = impl_witness_table (%impl_witness_assoc_constant.loc31, @C2.as.J2.impl.%C2.as.J2.impl.F.decl.loc32_33.2), @C2.as.J2.impl [concrete]
// CHECK:STDOUT:   %J2.impl_witness.loc31: <witness> = impl_witness %J2.impl_witness_table.loc31 [concrete = constants.%J2.impl_witness.b77]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc31: type = impl_witness_assoc_constant constants.%C2 [concrete = constants.%C2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J2 {
// CHECK:STDOUT:   %Self: %J2.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U2: type = assoc_const_decl @U2 [concrete] {
// CHECK:STDOUT:     %assoc0: %J2.assoc_type = assoc_entity element0, @J2.%U2 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J2.F.decl: %J2.F.type = fn_decl @J2.F [concrete = constants.%J2.F] {
// CHECK:STDOUT:     %self.patt: @J2.F.%pattern_type (%pattern_type.0a6) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @J2.F.%pattern_type (%pattern_type.0a6) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %z.patt: <error> = value_binding_pattern z [concrete]
// CHECK:STDOUT:     %z.param_patt: <error> = value_param_pattern %z.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc19_35: %J2.type = name_ref Self, @J2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %self.param: @J2.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc19_14.1: type = splice_block %.loc19_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc19_14: %J2.type = name_ref Self, @J2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref.loc19_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc19_14.2: type = converted %Self.ref.loc19_14, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J2.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     %z.param: <error> = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc19_23: %J2.type = name_ref Self, @J2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %z: <error> = value_binding z, %z.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J2.assoc_type = assoc_entity element1, %J2.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U2 = @U2.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U2, %J2.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U2(@J2.%Self: %J2.type) {
// CHECK:STDOUT:   assoc_const U2:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.J2.impl: %.loc22_7.2 as %.loc22_15 {
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.decl.loc23_33.1: %empty_tuple.type.as.J2.impl.F.type.d9b215.1 = fn_decl @empty_tuple.type.as.J2.impl.F.loc23_33.1 [concrete = constants.%empty_tuple.type.as.J2.impl.F.b047a1.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %z.patt: %pattern_type.a96 = value_binding_pattern z [concrete]
// CHECK:STDOUT:     %z.param_patt: %pattern_type.a96 = value_param_pattern %z.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc23_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc23_31.2: type = converted %.loc23_31.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.J2.impl.%.loc22_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %z.param: %empty_struct_type = value_param call_param1
// CHECK:STDOUT:     %.loc23_24.1: type = splice_block %.loc23_24.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc23_24.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc23_24.3: type = converted %.loc23_24.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %z: %empty_struct_type = value_binding z, %z.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param2
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.decl.loc23_33.2: %empty_tuple.type.as.J2.impl.F.type.d9b215.2 = fn_decl @empty_tuple.type.as.J2.impl.F.loc23_33.2 [concrete = constants.%empty_tuple.type.as.J2.impl.F.b047a1.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.param_patt: <error> = value_param_pattern <error>, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %.param: <error> = value_param call_param1
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.J2.impl.F.decl.loc23_33.1
// CHECK:STDOUT:   witness = file.%J2.impl_witness.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C2.as.J2.impl: %C2.ref.loc31_6 as %.loc31_15 {
// CHECK:STDOUT:   %C2.as.J2.impl.F.decl.loc32_33.1: %C2.as.J2.impl.F.type.5b1c29.1 = fn_decl @C2.as.J2.impl.F.loc32_33.1 [concrete = constants.%C2.as.J2.impl.F.8b36c6.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.838 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.838 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %z.patt: %pattern_type.838 = value_binding_pattern z [concrete]
// CHECK:STDOUT:     %z.param_patt: %pattern_type.838 = value_param_pattern %z.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.838 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.838 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C2.ref.loc32_30: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %self.param: %C2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C2.as.J2.impl.%C2.ref.loc31_6 [concrete = constants.%C2]
// CHECK:STDOUT:     %self: %C2 = value_binding self, %self.param
// CHECK:STDOUT:     %z.param: %C2 = value_param call_param1
// CHECK:STDOUT:     %C2.ref.loc32_23: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %z: %C2 = value_binding z, %z.param
// CHECK:STDOUT:     %return.param: ref %C2 = out_param call_param2
// CHECK:STDOUT:     %return: ref %C2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2.as.J2.impl.F.decl.loc32_33.2: %C2.as.J2.impl.F.type.5b1c29.2 = fn_decl @C2.as.J2.impl.F.loc32_33.2 [concrete = constants.%C2.as.J2.impl.F.8b36c6.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.838 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.838 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.param_patt: <error> = value_param_pattern <error>, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C2 = value_param call_param0
// CHECK:STDOUT:     %self: %C2 = value_binding self, %self.param
// CHECK:STDOUT:     %.param: <error> = value_param call_param1
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C2 = <poisoned>
// CHECK:STDOUT:   .F = %C2.as.J2.impl.F.decl.loc32_33.1
// CHECK:STDOUT:   witness = file.%J2.impl_witness.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %.loc28_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc28_11: type = converted %.loc28_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc28_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J2.F(@J2.%Self: %J2.type) {
// CHECK:STDOUT:   %Self: %J2.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.0a6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J2.F.%Self.binding.as_type (%Self.binding.as_type), %z.param: <error>) -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.J2.impl.F.loc23_33.1(%self.param: %empty_tuple.type, %z.param: %empty_struct_type) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %z.ref: %empty_struct_type = name_ref z, %z
// CHECK:STDOUT:   %.loc23_42: init %empty_struct_type = struct_init () to %return [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc23_43: init %empty_struct_type = converted %z.ref, %.loc23_42 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc23_43 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.J2.impl.F.loc23_33.2(%self.param: %empty_tuple.type, %.param: <error>) -> <error> [thunk @empty_tuple.type.as.J2.impl.%empty_tuple.type.as.J2.impl.F.decl.loc23_33.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.J2.impl.F.type.d9b215.1 = name_ref F, @empty_tuple.type.as.J2.impl.%empty_tuple.type.as.J2.impl.F.decl.loc23_33.1 [concrete = constants.%empty_tuple.type.as.J2.impl.F.b047a1.1]
// CHECK:STDOUT:   %self.ref: %empty_tuple.type = name_ref self, %self.param
// CHECK:STDOUT:   %.ref: <error> = name_ref <none>, %.param
// CHECK:STDOUT:   %return.ref: ref <error> = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.call: init %empty_struct_type = call %empty_tuple.type.as.J2.impl.F.bound(%self.ref, <error>)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C2.as.J2.impl.F.loc32_33.1(%self.param: %C2, %z.param: %C2) -> %C2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C2 = name_ref self, %self
// CHECK:STDOUT:   %.loc32_46.1: %empty_struct_type = as_compatible %self.ref
// CHECK:STDOUT:   %.loc32_46.2: ref %empty_struct_type = as_compatible %return
// CHECK:STDOUT:   %.loc32_46.3: init %empty_struct_type = struct_init () to %.loc32_46.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_46.4: init %C2 = as_compatible %.loc32_46.3 [concrete = constants.%C2.val]
// CHECK:STDOUT:   %.loc32_46.5: init %C2 = converted %self.ref, %.loc32_46.4 [concrete = constants.%C2.val]
// CHECK:STDOUT:   return %.loc32_46.5 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C2.as.J2.impl.F.loc32_33.2(%self.param: %C2, %.param: <error>) -> <error> [thunk @C2.as.J2.impl.%C2.as.J2.impl.F.decl.loc32_33.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %C2.as.J2.impl.F.type.5b1c29.1 = name_ref F, @C2.as.J2.impl.%C2.as.J2.impl.F.decl.loc32_33.1 [concrete = constants.%C2.as.J2.impl.F.8b36c6.1]
// CHECK:STDOUT:   %self.ref: %C2 = name_ref self, %self.param
// CHECK:STDOUT:   %.ref: <error> = name_ref <none>, %.param
// CHECK:STDOUT:   %return.ref: ref <error> = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %C2.as.J2.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   %C2.as.J2.impl.F.call: init %C2 = call %C2.as.J2.impl.F.bound(%self.ref, <error>)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U2(constants.%.Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2.F(constants.%J2.facet.02c) {
// CHECK:STDOUT:   %Self => constants.%J2.facet.02c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2.F(constants.%J2.facet.73a) {
// CHECK:STDOUT:   %Self => constants.%J2.facet.73a
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.838
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_associated_type_in_signature_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self.999: %K.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type @K [concrete]
// CHECK:STDOUT:   %assoc0.29d: %K.assoc_type = assoc_entity element0, @K.%V [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.1c0: type = symbolic_binding_type Self, 0, %Self.999 [symbolic]
// CHECK:STDOUT:   %pattern_type.dfd: type = pattern_type %Self.binding.as_type.1c0 [symbolic]
// CHECK:STDOUT:   %K.lookup_impl_witness.0dc: <witness> = lookup_impl_witness %Self.999, @K [symbolic]
// CHECK:STDOUT:   %impl.elem0.747: type = impl_witness_access %K.lookup_impl_witness.0dc, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.d73: type = pattern_type %impl.elem0.747 [symbolic]
// CHECK:STDOUT:   %K.F.type: type = fn_type @K.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %K.F: %K.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %K.assoc_type = assoc_entity element1, @K.%K.F.decl [concrete]
// CHECK:STDOUT:   %.Self: %K.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %K.lookup_impl_witness.e69: <witness> = lookup_impl_witness %.Self, @K [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e40: type = impl_witness_access %K.lookup_impl_witness.e69, element0 [symbolic_self]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %K_where.type: type = facet_type <@K where %impl.elem0.e40 = %struct_type.a> [concrete]
// CHECK:STDOUT:   %K.impl_witness: <witness> = impl_witness file.%K.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.414: type = pattern_type %struct_type.x [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.type.662f2e.1: type = fn_type @empty_tuple.type.as.K.impl.F.loc26_45.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.ec3e8a.1: %empty_tuple.type.as.K.impl.F.type.662f2e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %K.facet: %K.type = facet_value %empty_tuple.type, (%K.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.e85: type = pattern_type %struct_type.a [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.type.662f2e.2: type = fn_type @empty_tuple.type.as.K.impl.F.loc26_45.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.ec3e8a.2: %empty_tuple.type.as.K.impl.F.type.662f2e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %struct_type.x, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.a2f: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.4e1: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.a2f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.2f8: type = ptr_type %struct_type.x [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.4e1, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [concrete = constants.%K.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.K.impl [concrete] {} {
// CHECK:STDOUT:     %.loc11_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_7.2: type = converted %.loc11_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     %.Self: %K.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %K.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %K.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.29d]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc11_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%K.lookup_impl_witness.e69, element0 [symbolic_self = constants.%impl.elem0.e40]
// CHECK:STDOUT:     %.loc11_31.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_31.2: type = converted %.loc11_31.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %.loc11_14: type = where_expr %.Self [concrete = constants.%K_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%K.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %struct_type.a
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @empty_tuple.type.as.K.impl.%empty_tuple.type.as.K.impl.F.decl.loc26_45.2), @empty_tuple.type.as.K.impl [concrete]
// CHECK:STDOUT:   %K.impl_witness: <witness> = impl_witness %K.impl_witness_table [concrete = constants.%K.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%struct_type.a [concrete = constants.%struct_type.a]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = symbolic_binding Self, 0 [symbolic = constants.%Self.999]
// CHECK:STDOUT:   %V: type = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %K.assoc_type = assoc_entity element0, @K.%V [concrete = constants.%assoc0.29d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.F.decl: %K.F.type = fn_decl @K.F [concrete = constants.%K.F] {
// CHECK:STDOUT:     %self.patt: @K.F.%pattern_type.loc8_8 (%pattern_type.dfd) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @K.F.%pattern_type.loc8_8 (%pattern_type.dfd) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: @K.F.%pattern_type.loc8_20 (%pattern_type.d73) = value_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: @K.F.%pattern_type.loc8_20 (%pattern_type.d73) = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @K.F.%pattern_type.loc8_20 (%pattern_type.d73) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @K.F.%pattern_type.loc8_20 (%pattern_type.d73) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc8_29: type = impl_witness_access constants.%K.lookup_impl_witness.0dc, element0 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.747)]
// CHECK:STDOUT:     %V.ref.loc8_29: type = name_ref V, %impl.elem0.loc8_29 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.747)]
// CHECK:STDOUT:     %self.param: @K.F.%Self.binding.as_type (%Self.binding.as_type.1c0) = value_param call_param0
// CHECK:STDOUT:     %.loc8_14.1: type = splice_block %.loc8_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1c0)] {
// CHECK:STDOUT:       %Self.ref: %K.type = name_ref Self, @K.%Self [symbolic = %Self (constants.%Self.999)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1c0)]
// CHECK:STDOUT:       %.loc8_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1c0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @K.F.%Self.binding.as_type (%Self.binding.as_type.1c0) = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: @K.F.%impl.elem0.loc8_23.1 (%impl.elem0.747) = value_param call_param1
// CHECK:STDOUT:     %.loc8_23: type = splice_block %V.ref.loc8_23 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.747)] {
// CHECK:STDOUT:       %impl.elem0.loc8_23.2: type = impl_witness_access constants.%K.lookup_impl_witness.0dc, element0 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.747)]
// CHECK:STDOUT:       %V.ref.loc8_23: type = name_ref V, %impl.elem0.loc8_23.2 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.747)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @K.F.%impl.elem0.loc8_23.1 (%impl.elem0.747) = value_binding v, %v.param
// CHECK:STDOUT:     %return.param: ref @K.F.%impl.elem0.loc8_23.1 (%impl.elem0.747) = out_param call_param2
// CHECK:STDOUT:     %return: ref @K.F.%impl.elem0.loc8_23.1 (%impl.elem0.747) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %K.assoc_type = assoc_entity element1, %K.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%V, %K.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@K.%Self: %K.type) {
// CHECK:STDOUT:   assoc_const V:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.K.impl: %.loc11_7.2 as %.loc11_14 {
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.decl.loc26_45.1: %empty_tuple.type.as.K.impl.F.type.662f2e.1 = fn_decl @empty_tuple.type.as.K.impl.F.loc26_45.1 [concrete = constants.%empty_tuple.type.as.K.impl.F.ec3e8a.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: %pattern_type.414 = value_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: %pattern_type.414 = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.414 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.414 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_42.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc26_42.2: type = converted %.loc26_42.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.x.loc26_43: type = struct_type {.x: %empty_tuple.type} [concrete = constants.%struct_type.x]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.K.impl.%.loc11_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: %struct_type.x = value_param call_param1
// CHECK:STDOUT:     %.loc26_30: type = splice_block %struct_type.x.loc26_30 [concrete = constants.%struct_type.x] {
// CHECK:STDOUT:       %.loc26_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc26_29.2: type = converted %.loc26_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.loc26_30: type = struct_type {.x: %empty_tuple.type} [concrete = constants.%struct_type.x]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: %struct_type.x = value_binding v, %v.param
// CHECK:STDOUT:     %return.param: ref %struct_type.x = out_param call_param2
// CHECK:STDOUT:     %return: ref %struct_type.x = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.decl.loc26_45.2: %empty_tuple.type.as.K.impl.F.type.662f2e.2 = fn_decl @empty_tuple.type.as.K.impl.F.loc26_45.2 [concrete = constants.%empty_tuple.type.as.K.impl.F.ec3e8a.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: %pattern_type.e85 = value_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: %pattern_type.e85 = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.e85 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e85 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: %struct_type.a = value_param call_param1
// CHECK:STDOUT:     %v: %struct_type.a = value_binding v, %v.param
// CHECK:STDOUT:     %return.param: ref %struct_type.a = out_param call_param2
// CHECK:STDOUT:     %return: ref %struct_type.a = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.K.impl.F.decl.loc26_45.1
// CHECK:STDOUT:   witness = file.%K.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K.F(@K.%Self: %K.type) {
// CHECK:STDOUT:   %Self: %K.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.999)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1c0)]
// CHECK:STDOUT:   %pattern_type.loc8_8: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc8_8 (constants.%pattern_type.dfd)]
// CHECK:STDOUT:   %K.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @K [symbolic = %K.lookup_impl_witness (constants.%K.lookup_impl_witness.0dc)]
// CHECK:STDOUT:   %impl.elem0.loc8_23.1: type = impl_witness_access %K.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.747)]
// CHECK:STDOUT:   %pattern_type.loc8_20: type = pattern_type %impl.elem0.loc8_23.1 [symbolic = %pattern_type.loc8_20 (constants.%pattern_type.d73)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @K.F.%Self.binding.as_type (%Self.binding.as_type.1c0), %v.param: @K.F.%impl.elem0.loc8_23.1 (%impl.elem0.747)) -> @K.F.%impl.elem0.loc8_23.1 (%impl.elem0.747);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.K.impl.F.loc26_45.1(%self.param: %empty_tuple.type, %v.param: %struct_type.x) -> %struct_type.x {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: %struct_type.x = name_ref v, %v
// CHECK:STDOUT:   %.loc26_54.1: %empty_tuple.type = struct_access %v.ref, element0
// CHECK:STDOUT:   %.loc26_54.2: ref %empty_tuple.type = struct_access %return, element0
// CHECK:STDOUT:   %.loc26_54.3: init %empty_tuple.type = tuple_init () to %.loc26_54.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc26_54.4: init %empty_tuple.type = converted %.loc26_54.1, %.loc26_54.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc26_54.5: init %struct_type.x = struct_init (%.loc26_54.4) to %return [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc26_55: init %struct_type.x = converted %v.ref, %.loc26_54.5 [concrete = constants.%struct]
// CHECK:STDOUT:   return %.loc26_55 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.K.impl.F.loc26_45.2(%self.param: %empty_tuple.type, %v.param: %struct_type.a) -> %struct_type.a [thunk @empty_tuple.type.as.K.impl.%empty_tuple.type.as.K.impl.F.decl.loc26_45.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.K.impl.F.type.662f2e.1 = name_ref F, @empty_tuple.type.as.K.impl.%empty_tuple.type.as.K.impl.F.decl.loc26_45.1 [concrete = constants.%empty_tuple.type.as.K.impl.F.ec3e8a.1]
// CHECK:STDOUT:   %self.ref: %empty_tuple.type = name_ref self, %self.param
// CHECK:STDOUT:   %v.ref: %struct_type.a = name_ref v, %v.param
// CHECK:STDOUT:   %return.ref: ref %struct_type.a = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.call: init %struct_type.x = call %empty_tuple.type.as.K.impl.F.bound(%self.ref, <error>)
// CHECK:STDOUT:   %.loc26_45.1: ref %struct_type.x = temporary_storage
// CHECK:STDOUT:   %.loc26_45.2: ref %struct_type.x = temporary %.loc26_45.1, %empty_tuple.type.as.K.impl.F.call
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc26_45.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.4e1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.4e1, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc26_45.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.2f8 = addr_of %.loc26_45.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self.999) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.F(constants.%Self.999) {
// CHECK:STDOUT:   %Self => constants.%Self.999
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.1c0
// CHECK:STDOUT:   %pattern_type.loc8_8 => constants.%pattern_type.dfd
// CHECK:STDOUT:   %K.lookup_impl_witness => constants.%K.lookup_impl_witness.0dc
// CHECK:STDOUT:   %impl.elem0.loc8_23.1 => constants.%impl.elem0.747
// CHECK:STDOUT:   %pattern_type.loc8_20 => constants.%pattern_type.d73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%.Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.F(constants.%K.facet) {
// CHECK:STDOUT:   %Self => constants.%K.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type.loc8_8 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %K.lookup_impl_witness => constants.%K.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc8_23.1 => constants.%struct_type.a
// CHECK:STDOUT:   %pattern_type.loc8_20 => constants.%pattern_type.e85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_non-type_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self: %M.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type @M [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %M.G.type: type = fn_type @M.G [concrete]
// CHECK:STDOUT:   %M.G: %M.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%M.G.decl [concrete]
// CHECK:STDOUT:   %.Self: %M.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %M.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @M [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access %M.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.b.347 = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %M_where.type: type = facet_type <@M where %impl.elem0 = %struct> [concrete]
// CHECK:STDOUT:   %M.impl_witness: <witness> = impl_witness file.%M.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.M.impl.G.type: type = fn_type @empty_tuple.type.as.M.impl.G [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.M.impl.G: %empty_tuple.type.as.M.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %empty_tuple.type, (%M.impl_witness) [concrete]
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
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.M.impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.347 = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_32: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_33.1: %struct_type.b.347 = struct_literal (%.loc8_32)
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_33.2: %empty_struct_type = converted %.loc8_32, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %struct: %struct_type.b.347 = struct_value (%.loc8_33.2) [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc8_33.3: %struct_type.b.347 = converted %.loc8_33.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%M_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%M.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_33.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @empty_tuple.type.as.M.impl.%empty_tuple.type.as.M.impl.G.decl), @empty_tuple.type.as.M.impl [concrete]
// CHECK:STDOUT:   %M.impl_witness: <witness> = impl_witness %M.impl_witness_table [concrete = constants.%M.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %struct_type.b.347 = impl_witness_assoc_constant constants.%struct [concrete = constants.%struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Z: %struct_type.b.347 = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.G.decl: %M.G.type = fn_decl @M.G [concrete = constants.%M.G] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, %M.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = @Z.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Z, %M.G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Z(@M.%Self: %M.type) {
// CHECK:STDOUT:   assoc_const Z:! %struct_type.b.347;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.M.impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %empty_tuple.type.as.M.impl.G.decl: %empty_tuple.type.as.M.impl.G.type = fn_decl @empty_tuple.type.as.M.impl.G [concrete = constants.%empty_tuple.type.as.M.impl.G] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_14.2: type = converted %.loc9_14.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %empty_tuple.type.as.M.impl.G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = file.%M.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @M.G(@M.%Self: %M.type) {
// CHECK:STDOUT:   fn() -> %empty_struct_type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.M.impl.G() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @empty_tuple.type.as.M.impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %Self.ref, (constants.%M.impl_witness) [concrete = constants.%M.facet]
// CHECK:STDOUT:   %.loc10_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_18 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc10_23: type = converted %.loc10_18, %as_type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access constants.%M.impl_witness, element0 [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_25.1: %empty_struct_type = struct_access %impl.elem0, element0 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_25.2: init %empty_struct_type = struct_init () to %return [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_27: init %empty_struct_type = converted %.loc10_25.1, %.loc10_25.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc10_27 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M.G(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%.Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M.G(constants.%M.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_as_uses_correct_rewrite_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.3f0: type = class_type @C, @C(%T.d9f) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self.70e: %M.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.b.86f: type = struct_type {.b: type} [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type @M [concrete]
// CHECK:STDOUT:   %assoc0.105: %M.assoc_type = assoc_entity element0, @M.%Z [concrete]
// CHECK:STDOUT:   %M.G.type: type = fn_type @M.G [concrete]
// CHECK:STDOUT:   %M.G: %M.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%M.G.decl [concrete]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %.Self.5f5: %M.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.5f5 [symbolic_self]
// CHECK:STDOUT:   %M.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.5f5, @M [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.228: %struct_type.b.86f = impl_witness_access %M.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.b.161: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct.0f3: %struct_type.b.86f = struct_value (%empty_struct_type) [concrete]
// CHECK:STDOUT:   %M_where.type.a69: type = facet_type <@M where %impl.elem0.228 = %struct.0f3> [concrete]
// CHECK:STDOUT:   %M.impl_witness.2c0: <witness> = impl_witness file.%M.impl_witness_table.loc10 [concrete]
// CHECK:STDOUT:   %C.as.M.impl.G.type.062: type = fn_type @C.as.M.impl.G.loc11 [concrete]
// CHECK:STDOUT:   %C.as.M.impl.G.2c5: %C.as.M.impl.G.type.062 = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.9bf: %M.type = facet_value %C.7a7, (%M.impl_witness.2c0) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.de9: <witness> = impl_witness imports.%Copy.impl_witness_table.40f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.de9) [concrete]
// CHECK:STDOUT:   %.98f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound.583: <bound method> = bound_method %empty_struct_type, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.3a0: type = class_type @C, @C(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %struct.c94: %struct_type.b.86f = struct_value (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %M_where.type.87f: type = facet_type <@M where %impl.elem0.228 = %struct.c94> [concrete]
// CHECK:STDOUT:   %M.impl_witness.1f1: <witness> = impl_witness file.%M.impl_witness_table.loc16 [concrete]
// CHECK:STDOUT:   %C.as.M.impl.G.type.54d: type = fn_type @C.as.M.impl.G.loc17 [concrete]
// CHECK:STDOUT:   %C.as.M.impl.G.a42: %C.as.M.impl.G.type.54d = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.fbc: %M.type = facet_value %C.3a0, (%M.impl_witness.1f1) [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound.a1c: <bound method> = bound_method %empty_tuple.type, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.f97: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.40f = impl_witness_table (%Core.import_ref.f97), @type.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %T.loc3_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_9.1 (constants.%T.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.M.impl.cd4 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc10_9: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_10: type = converted %.loc10_9, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.7a7]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = symbolic_binding .Self [symbolic_self = constants.%.Self.5f5]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.5f5]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0.105]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_23: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.86f = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.228]
// CHECK:STDOUT:     %.loc10_35: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_36.1: %struct_type.b.347 = struct_literal (%.loc10_35)
// CHECK:STDOUT:     %.loc10_36.2: type = converted %.loc10_35, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct: %struct_type.b.86f = struct_value (%.loc10_36.2) [concrete = constants.%struct.0f3]
// CHECK:STDOUT:     %.loc10_36.3: %struct_type.b.86f = converted %.loc10_36.1, %struct [concrete = constants.%struct.0f3]
// CHECK:STDOUT:     %.loc10_17: type = where_expr %.Self [concrete = constants.%M_where.type.a69] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%M.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc10_36.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.impl_witness_table.loc10 = impl_witness_table (%impl_witness_assoc_constant.loc10, @C.as.M.impl.cd4.%C.as.M.impl.G.decl), @C.as.M.impl.cd4 [concrete]
// CHECK:STDOUT:   %M.impl_witness.loc10: <witness> = impl_witness %M.impl_witness_table.loc10 [concrete = constants.%M.impl_witness.2c0]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc10: %struct_type.b.86f = impl_witness_assoc_constant constants.%struct.0f3 [concrete = constants.%struct.0f3]
// CHECK:STDOUT:   impl_decl @C.as.M.impl.3ca [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc16_9: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_10: type = converted %.loc16_9, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_tuple.type) [concrete = constants.%C.3a0]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = symbolic_binding .Self [symbolic_self = constants.%.Self.5f5]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.5f5]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0.105]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc16_23: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.86f = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.228]
// CHECK:STDOUT:     %.loc16_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_36.1: %struct_type.b.161 = struct_literal (%.loc16_35)
// CHECK:STDOUT:     %.loc16_36.2: type = converted %.loc16_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct: %struct_type.b.86f = struct_value (%.loc16_36.2) [concrete = constants.%struct.c94]
// CHECK:STDOUT:     %.loc16_36.3: %struct_type.b.86f = converted %.loc16_36.1, %struct [concrete = constants.%struct.c94]
// CHECK:STDOUT:     %.loc16_17: type = where_expr %.Self [concrete = constants.%M_where.type.87f] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%M.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc16_36.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.impl_witness_table.loc16 = impl_witness_table (%impl_witness_assoc_constant.loc16, @C.as.M.impl.3ca.%C.as.M.impl.G.decl), @C.as.M.impl.3ca [concrete]
// CHECK:STDOUT:   %M.impl_witness.loc16: <witness> = impl_witness %M.impl_witness_table.loc16 [concrete = constants.%M.impl_witness.1f1]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc16: %struct_type.b.86f = impl_witness_assoc_constant constants.%struct.c94 [concrete = constants.%struct.c94]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = symbolic_binding Self, 0 [symbolic = constants.%Self.70e]
// CHECK:STDOUT:   %Z: %struct_type.b.86f = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0.105]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.G.decl: %M.G.type = fn_decl @M.G [concrete = constants.%M.G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, %M.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = @Z.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Z, %M.G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Z(@M.%Self: %M.type) {
// CHECK:STDOUT:   assoc_const Z:! %struct_type.b.86f;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.M.impl.cd4: %C as %.loc10_17 {
// CHECK:STDOUT:   %C.as.M.impl.G.decl: %C.as.M.impl.G.type.062 = fn_decl @C.as.M.impl.G.loc11 [concrete = constants.%C.as.M.impl.G.2c5] {
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.M.impl.G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = file.%M.impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.M.impl.3ca: %C as %.loc16_17 {
// CHECK:STDOUT:   %C.as.M.impl.G.decl: %C.as.M.impl.G.type.54d = fn_decl @C.as.M.impl.G.loc17 [concrete = constants.%C.as.M.impl.G.a42] {
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.M.impl.G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = file.%M.impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc3_9.2: type) {
// CHECK:STDOUT:   %T.loc3_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_9.1 (constants.%T.d9f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.3f0
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @M.G(@M.%Self: %M.type) {
// CHECK:STDOUT:   fn() -> type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.M.impl.G.loc11() -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @C.as.M.impl.cd4.%C [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %Self.ref, (constants.%M.impl_witness.2c0) [concrete = constants.%M.facet.9bf]
// CHECK:STDOUT:   %.loc12_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.9bf]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0.105]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc12_18 [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %.loc12_23: type = converted %.loc12_18, %as_type [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %impl.elem0.loc12_23: %struct_type.b.86f = impl_witness_access constants.%M.impl_witness.2c0, element0 [concrete = constants.%struct.0f3]
// CHECK:STDOUT:   %.loc12_25: type = struct_access %impl.elem0.loc12_23, element0 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %impl.elem0.loc12_25: %.98f = impl_witness_access constants.%Copy.impl_witness.de9, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc12_25, %impl.elem0.loc12_25 [concrete = constants.%type.as.Copy.impl.Op.bound.583]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%.loc12_25) [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   return %type.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.M.impl.G.loc17() -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @C.as.M.impl.3ca.%C [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %Self.ref, (constants.%M.impl_witness.1f1) [concrete = constants.%M.facet.fbc]
// CHECK:STDOUT:   %.loc18_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.fbc]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0.105]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc18_18 [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %.loc18_23: type = converted %.loc18_18, %as_type [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %impl.elem0.loc18_23: %struct_type.b.86f = impl_witness_access constants.%M.impl_witness.1f1, element0 [concrete = constants.%struct.c94]
// CHECK:STDOUT:   %.loc18_25: type = struct_access %impl.elem0.loc18_23, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %impl.elem0.loc18_25: %.98f = impl_witness_access constants.%Copy.impl_witness.de9, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc18_25, %impl.elem0.loc18_25 [concrete = constants.%type.as.Copy.impl.Op.bound.a1c]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%.loc18_25) [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   return %type.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc3_9.1 => constants.%T.d9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.70e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M.G(constants.%Self.70e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc3_9.1 => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%.Self.5f5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M.G(constants.%M.facet.9bf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.9bf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.loc3_9.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M.G(constants.%M.facet.fbc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.fbc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_int_in_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.9f2: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.995: %I.assoc_type = assoc_entity element0, @I.%N [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.bcb: type = symbolic_binding_type Self, 0, %Self.9f2 [symbolic]
// CHECK:STDOUT:   %pattern_type.89c: type = pattern_type %Self.binding.as_type.bcb [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.lookup_impl_witness.9b0: <witness> = lookup_impl_witness %Self.9f2, @I [symbolic]
// CHECK:STDOUT:   %impl.elem0.b81: %i32 = impl_witness_access %I.lookup_impl_witness.9b0, element0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7a9: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2c2: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.cdf: %Int.as.ImplicitAs.impl.Convert.type.2c2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.972: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.783, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.592: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.047: %Int.as.ImplicitAs.impl.Convert.type.592 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.64a: %ImplicitAs.type.7a9 = facet_value %i32, (%ImplicitAs.impl_witness.972) [concrete]
// CHECK:STDOUT:   %.63b: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.64a [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.fe5: <bound method> = bound_method %impl.elem0.b81, %Int.as.ImplicitAs.impl.Convert.047 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.047, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.51d: <bound method> = bound_method %impl.elem0.b81, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.51d(%impl.elem0.b81) [symbolic]
// CHECK:STDOUT:   %array_type.e0f: type = array_type %Int.as.ImplicitAs.impl.Convert.call, bool [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %array_type.e0f [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%I.F.decl [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.fd8: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b22: %i32 = impl_witness_access %I.lookup_impl_witness.fd8, element0 [symbolic_self]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3ad: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.3ad [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9f7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.b22 = %int_2.ef8> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %array_type.c9b: type = array_type %int_2.ecc, bool [concrete]
// CHECK:STDOUT:   %pattern_type.5d5: type = pattern_type %array_type.c9b [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type: type = fn_type @empty_tuple.type.as.I.impl.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F: %empty_tuple.type.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_tuple.type, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.6fa: <bound method> = bound_method %int_2.ef8, %Int.as.ImplicitAs.impl.Convert.047 [concrete]
// CHECK:STDOUT:   %bound_method.33f: <bound method> = bound_method %int_2.ef8, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %tuple.type.784: type = tuple_type (bool, bool) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a56: <witness> = impl_witness imports.%Copy.impl_witness_table.189 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.a56) [concrete]
// CHECK:STDOUT:   %.05c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound.5ee: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound.4eb: <bound method> = bound_method %false, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %array: %array_type.c9b = tuple_value (%true, %false) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.a86: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2c2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.cdf)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.783 = impl_witness_table (%Core.import_ref.a86), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cc5: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.189 = impl_witness_table (%Core.import_ref.cc5), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %N.ref: %I.assoc_type = name_ref N, @N.%assoc0 [concrete = constants.%assoc0.995]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0.loc8_20: %i32 = impl_witness_access constants.%I.lookup_impl_witness.fd8, element0 [symbolic_self = constants.%impl.elem0.b22]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc8_25: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:     %bound_method.loc8_25.1: <bound method> = bound_method %int_2, %impl.elem0.loc8_25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc8_25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_25.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.9f7]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_25.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_25.2: %i32 = converted %int_2, %.loc8_25.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8_20, %.loc8_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl), @empty_tuple.type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %i32 = impl_witness_assoc_constant constants.%int_2.ef8 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.9f2]
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl @N [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%N [concrete = constants.%assoc0.995]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %self.patt: @I.F.%pattern_type.loc5_8 (%pattern_type.89c) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @I.F.%pattern_type.loc5_8 (%pattern_type.89c) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @I.F.%pattern_type.loc5_22 (%pattern_type.a96) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @I.F.%pattern_type.loc5_22 (%pattern_type.a96) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %impl.elem0.loc5_37.2: %i32 = impl_witness_access constants.%I.lookup_impl_witness.9b0, element0 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.b81)]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %impl.elem0.loc5_37.2 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.b81)]
// CHECK:STDOUT:     %.loc5_31.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc5_31.2: type = converted %Bool.call, %.loc5_31.1 [concrete = bool]
// CHECK:STDOUT:     %impl.elem0.loc5_37.3: %.63b = impl_witness_access constants.%ImplicitAs.impl_witness.972, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.047]
// CHECK:STDOUT:     %bound_method.loc5_37.2: <bound method> = bound_method %N.ref, %impl.elem0.loc5_37.3 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.fe5)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc5_37.3, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc5_37.3: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc5_37.1 (constants.%bound_method.51d)]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call.loc5_37.2: init Core.IntLiteral = call %bound_method.loc5_37.3(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_37.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc5_37.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_37.2: Core.IntLiteral = converted %N.ref, %.loc5_37.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %array_type.loc5_38.2: type = array_type %.loc5_37.2, %.loc5_31.2 [symbolic = %array_type.loc5_38.1 (constants.%array_type.e0f)]
// CHECK:STDOUT:     %self.param: @I.F.%Self.binding.as_type (%Self.binding.as_type.bcb) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.F.%Self.binding.as_type (%Self.binding.as_type.bcb) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @I.F.%array_type.loc5_38.1 (%array_type.e0f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @I.F.%array_type.loc5_38.1 (%array_type.e0f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %I.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = @N.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%N, %I.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @N(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const N:! %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl: %empty_tuple.type.as.I.impl.F.type = fn_decl @empty_tuple.type.as.I.impl.F [concrete = constants.%empty_tuple.type.as.I.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5d5 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5d5 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc9_31.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc9_31.2: type = converted %Bool.call, %.loc9_31.1 [concrete = bool]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, %.loc9_31.2 [concrete = constants.%array_type.c9b]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.I.impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %array_type.c9b = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.c9b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.89c)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.9b0)]
// CHECK:STDOUT:   %impl.elem0.loc5_37.1: %i32 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.b81)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %impl.elem0.loc5_37.1, constants.%Int.as.ImplicitAs.impl.Convert.047 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.fe5)]
// CHECK:STDOUT:   %bound_method.loc5_37.1: <bound method> = bound_method %impl.elem0.loc5_37.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc5_37.1 (constants.%bound_method.51d)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1: init Core.IntLiteral = call %bound_method.loc5_37.1(%impl.elem0.loc5_37.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc5_38.1: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1, bool [symbolic = %array_type.loc5_38.1 (constants.%array_type.e0f)]
// CHECK:STDOUT:   %pattern_type.loc5_22: type = pattern_type %array_type.loc5_38.1 [symbolic = %pattern_type.loc5_22 (constants.%pattern_type.a96)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.F.%Self.binding.as_type (%Self.binding.as_type.bcb)) -> @I.F.%array_type.loc5_38.1 (%array_type.e0f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F(%self.param: %empty_tuple.type) -> %return.param: %array_type.c9b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_61.1: %tuple.type.784 = tuple_literal (%true, %false)
// CHECK:STDOUT:   %impl.elem0.loc9_50: %.05c = impl_witness_access constants.%Copy.impl_witness.a56, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9_50: <bound method> = bound_method %true, %impl.elem0.loc9_50 [concrete = constants.%bool.as.Copy.impl.Op.bound.5ee]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call.loc9_50: init bool = call %bound_method.loc9_50(%true) [concrete = constants.%true]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_61.2: ref bool = array_index %return, %int_0
// CHECK:STDOUT:   %.loc9_61.3: init bool = initialize_from %bool.as.Copy.impl.Op.call.loc9_50 to %.loc9_61.2 [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0.loc9_56: %.05c = impl_witness_access constants.%Copy.impl_witness.a56, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9_56: <bound method> = bound_method %false, %impl.elem0.loc9_56 [concrete = constants.%bool.as.Copy.impl.Op.bound.4eb]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call.loc9_56: init bool = call %bound_method.loc9_56(%false) [concrete = constants.%false]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_61.4: ref bool = array_index %return, %int_1
// CHECK:STDOUT:   %.loc9_61.5: init bool = initialize_from %bool.as.Copy.impl.Op.call.loc9_56 to %.loc9_61.4 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_61.6: init %array_type.c9b = array_init (%.loc9_61.3, %.loc9_61.5) to %return [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_62: init %array_type.c9b = converted %.loc9_61.1, %.loc9_61.6 [concrete = constants.%array]
// CHECK:STDOUT:   return %.loc9_62 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%Self.9f2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.9f2) {
// CHECK:STDOUT:   %Self => constants.%Self.9f2
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.bcb
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.89c
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.9b0
// CHECK:STDOUT:   %impl.elem0.loc5_37.1 => constants.%impl.elem0.b81
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound.fe5
// CHECK:STDOUT:   %bound_method.loc5_37.1 => constants.%bound_method.51d
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %array_type.loc5_38.1 => constants.%array_type.e0f
// CHECK:STDOUT:   %pattern_type.loc5_22 => constants.%pattern_type.a96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%.Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_37.1 => constants.%int_2.ef8
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound.6fa
// CHECK:STDOUT:   %bound_method.loc5_37.1 => constants.%bound_method.33f
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 => constants.%int_2.ecc
// CHECK:STDOUT:   %array_type.loc5_38.1 => constants.%array_type.c9b
// CHECK:STDOUT:   %pattern_type.loc5_22 => constants.%pattern_type.5d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- symbolic_associated_type_in_concrete_context.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.3b3: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.659: %Z.assoc_type = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.3f0: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.Self.f2a: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.f2a [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.f2a, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.b44: type = facet_type <@Z where %impl.elem0 = %C.3f0> [symbolic]
// CHECK:STDOUT:   %require_complete.5d8: <witness> = require_complete_type %Z_where.type.b44 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.269: <witness> = impl_witness file.%Z.impl_witness_table, @T.as.Z.impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.131: type = class_type @C, @C(%D) [concrete]
// CHECK:STDOUT:   %Z_where.type.88e: type = facet_type <@Z where %impl.elem0 = %C.131> [concrete]
// CHECK:STDOUT:   %complete_type.6ef: <witness> = complete_type_witness %Z_where.type.88e [concrete]
// CHECK:STDOUT:   %Z.impl_witness.490: <witness> = impl_witness file.%Z.impl_witness_table, @T.as.Z.impl(%D) [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %D, (%Z.impl_witness.490) [concrete]
// CHECK:STDOUT:   %pattern_type.7f8: type = pattern_type %C.131 [concrete]
// CHECK:STDOUT:   %C.val: %C.131 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C.131, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.951: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.372: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.951 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.aaa: type = ptr_type %C.131 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.372, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %T.loc6_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_24: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self.1: %Z.type = symbolic_binding .Self [symbolic_self = constants.%.Self.f2a]
// CHECK:STDOUT:     %.Self.ref: %Z.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.f2a]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.659]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc9_37: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref.loc9_44: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:     %C.loc9_45.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc9_45.2 (constants.%C.3f0)]
// CHECK:STDOUT:     %.loc9_31: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.b44)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.loc9_45.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %T.loc9_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl(constants.%T) [symbolic = @T.as.Z.impl.%Z.impl_witness (constants.%Z.impl_witness.269)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C.3f0 [symbolic = @T.as.Z.impl.%C.loc9_45.2 (constants.%C.3f0)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3b3]
// CHECK:STDOUT:   %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%X [concrete = constants.%assoc0.659]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (%X)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl(%T.loc9_14.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:   %C.loc9_45.2: type = class_type @C, @C(%T.loc9_14.2) [symbolic = %C.loc9_45.2 (constants.%C.3f0)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z where constants.%impl.elem0 = %C.loc9_45.2> [symbolic = %Z_where.type (constants.%Z_where.type.b44)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.5d8)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table, @T.as.Z.impl(%T.loc9_14.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.269)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc9_24 as %.loc9_31 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc6_9.2: type) {
// CHECK:STDOUT:   %T.loc6_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.3f0
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7f8 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %D.ref.loc12_28: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%D) [concrete = constants.%C.131]
// CHECK:STDOUT:   %.loc12_21.2: ref %C.131 = temporary_storage
// CHECK:STDOUT:   %.loc12_21.3: init %C.131 = class_init (), %.loc12_21.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc12_21.4: ref %C.131 = temporary %.loc12_21.2, %.loc12_21.3
// CHECK:STDOUT:   %.loc12_23.1: ref %C.131 = converted %.loc12_21.1, %.loc12_21.4
// CHECK:STDOUT:   %.loc12_11.1: type = splice_block %impl.elem0 [concrete = constants.%C.131] {
// CHECK:STDOUT:     %D.ref.loc12_10: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.659]
// CHECK:STDOUT:     %Z.facet: %Z.type = facet_value %D.ref.loc12_10, (constants.%Z.impl_witness.490) [concrete = constants.%Z.facet]
// CHECK:STDOUT:     %.loc12_11.2: %Z.type = converted %D.ref.loc12_10, %Z.facet [concrete = constants.%Z.facet]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.impl_witness.490, element0 [concrete = constants.%C.131]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_23.2: %C.131 = acquire_value %.loc12_23.1
// CHECK:STDOUT:   %a: %C.131 = value_binding a, %.loc12_23.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_21.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.372
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.372, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc12_21.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.aaa = addr_of %.loc12_21.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Self.3b3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%.Self.f2a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T
// CHECK:STDOUT:   %C.loc9_45.2 => constants.%C.3f0
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.b44
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5d8
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl(constants.%D) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%D
// CHECK:STDOUT:   %C.loc9_45.2 => constants.%C.131
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.88e
// CHECK:STDOUT:   %require_complete => constants.%complete_type.6ef
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.490
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%D) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Z.facet) {}
// CHECK:STDOUT:
