// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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
  let U:! I;
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
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE+7]]:3: error: expression cannot be used as a value [UseOfNonExprAsValue]
  // CHECK:STDERR:   fn F[self: Self](z: {}) -> {} { return z; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE-7]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F[self: Self](z: Self.U2) -> Self.U2;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
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
  fn F[self: Self](v: V) -> V;
}

impl () as K where .V = {.a: ()} {
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE+7]]:3: error: expression cannot be used as a value [UseOfNonExprAsValue]
  // CHECK:STDERR:   fn F[self: Self](v: {.x: ()}) -> {.x: ()} { return v; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE-7]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
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
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %pattern_type.4ca: type = pattern_type %Self.as_type.3df [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.646: <witness> = lookup_impl_witness %Self.ccd, @J [symbolic]
// CHECK:STDOUT:   %J.facet.521: %J.type = facet_value %Self.as_type.3df, (%J.lookup_impl_witness.646) [symbolic]
// CHECK:STDOUT:   %impl.elem0.11f: type = impl_witness_access %J.lookup_impl_witness.646, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b98: type = pattern_type %impl.elem0.11f [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.572: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %.Self.968: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.78d: type = facet_access_type %.Self.968 [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.25c: <witness> = lookup_impl_witness %.Self.968, @J [symbolic_self]
// CHECK:STDOUT:   %J.facet.330: %J.type = facet_value %.Self.as_type.78d, (%J.lookup_impl_witness.25c) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.83f: type = impl_witness_access %J.lookup_impl_witness.25c, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type.b64: type = facet_type <@J where %impl.elem0.83f = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness.e42: <witness> = impl_witness file.%J.impl_witness_table.loc8 [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type.7d4: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.46e: %F.type.7d4 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.ecc: %J.type = facet_value %empty_tuple.type, (%J.impl_witness.e42) [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Op.type.e42: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.type.fc5: type = fn_type @Op.11, @impl.47e(%N) [symbolic]
// CHECK:STDOUT:   %Op.8d0: %Op.type.fc5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Negate.impl_witness.943: <witness> = impl_witness imports.%Negate.impl_witness_table, @impl.47e(%int_32) [concrete]
// CHECK:STDOUT:   %Op.type.66f: type = fn_type @Op.11, @impl.47e(%int_32) [concrete]
// CHECK:STDOUT:   %Op.5c0: %Op.type.66f = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %i32, (%Negate.impl_witness.943) [concrete]
// CHECK:STDOUT:   %.6f0: type = fn_type_with_self_type %Op.type.e42, %Negate.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn.f32: <specific function> = specific_function %Op.5c0, @Op.11(%int_32) [concrete]
// CHECK:STDOUT:   %CallMethod.type: type = fn_type @CallMethod [concrete]
// CHECK:STDOUT:   %CallMethod: %CallMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %.597: type = fn_type_with_self_type %F.type.c14, %J.facet.ecc [concrete]
// CHECK:STDOUT:   %int_40.f80: Core.IntLiteral = int_value 40 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_40.f80, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_40.f80, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_40.518: %i32 = int_value 40 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J_where.type.0bc: type = facet_type <@J where %impl.elem0.83f = %C> [concrete]
// CHECK:STDOUT:   %J.impl_witness.12d: <witness> = impl_witness file.%J.impl_witness_table.loc21 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type.2a7: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.fb5: %F.type.2a7 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.30e: %J.type = facet_value %C, (%J.impl_witness.12d) [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/operators/arithmetic, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.5f5145.5 = import_ref Core//prelude/types/int, loc112_43, unloaded
// CHECK:STDOUT:   %Core.import_ref.ff2: @impl.47e.%Op.type (%Op.type.fc5) = import_ref Core//prelude/types/int, loc113_31, loaded [symbolic = @impl.47e.%Op (constants.%Op.8d0)]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.5f5145.5, %Core.import_ref.ff2), @impl.47e [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:   impl_decl @impl.32a [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.25c, element0 [symbolic_self = constants.%impl.elem0.83f]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%J_where.type.b64] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table.loc8 = impl_witness_table (%impl_witness_assoc_constant.loc8, @impl.32a.%F.decl), @impl.32a [concrete]
// CHECK:STDOUT:   %J.impl_witness.loc8: <witness> = impl_witness %J.impl_witness_table.loc8 [concrete = constants.%J.impl_witness.e42]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc8: type = impl_witness_assoc_constant constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %CallMethod.decl: %CallMethod.type = fn_decl @CallMethod [concrete = constants.%CallMethod] {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = binding_pattern x [concrete]
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
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.043 [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc21_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:     %.loc21_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.25c, element0 [symbolic_self = constants.%impl.elem0.83f]
// CHECK:STDOUT:     %C.ref.loc21_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc21_13: type = where_expr %.Self [concrete = constants.%J_where.type.0bc] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref.loc21_24
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table.loc21 = impl_witness_table (%impl_witness_assoc_constant.loc21, @impl.043.%F.decl), @impl.043 [concrete]
// CHECK:STDOUT:   %J.impl_witness.loc21: <witness> = impl_witness %J.impl_witness_table.loc21 [concrete = constants.%J.impl_witness.12d]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc21: type = impl_witness_assoc_constant constants.%C [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %self.patt: @F.1.%pattern_type.loc5_8 (%pattern_type.4ca) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @F.1.%pattern_type.loc5_8 (%pattern_type.4ca) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @F.1.%pattern_type.loc5_20 (%pattern_type.b98) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @F.1.%pattern_type.loc5_20 (%pattern_type.b98) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @F.1.%pattern_type.loc5_20 (%pattern_type.b98) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.1.%pattern_type.loc5_20 (%pattern_type.b98) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access constants.%J.lookup_impl_witness.646, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     %U.ref.loc5_29: type = name_ref U, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.11f) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %U.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11f)] {
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access constants.%J.lookup_impl_witness.646, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:       %U.ref.loc5_23: type = name_ref U, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.11f) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.11f) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.11f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1.572]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.32a: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %F.decl: %F.type.7d4 = fn_decl @F.2 [concrete = constants.%F.46e] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: %pattern_type.7ce = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.7ce = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc9_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.32a.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc9_23: type = splice_block %i32.loc9_23 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc9_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%J.impl_witness.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.043: %C.ref.loc21_6 as %.loc21_13 {
// CHECK:STDOUT:   %F.decl: %F.type.2a7 = fn_decl @F.3 [concrete = constants.%F.fb5] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: %pattern_type.c48 = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.c48 = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc22_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.043.%C.ref.loc21_6 [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc22_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %u: %C = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
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
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.4ca)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.646)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:   %pattern_type.loc5_20: type = pattern_type %impl.elem0.loc5_23.1 [symbolic = %pattern_type.loc5_20 (constants.%pattern_type.b98)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.3df), %u.param: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.11f)) -> @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.11f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%self.param: %empty_tuple.type, %u.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %impl.elem1: %.6f0 = impl_witness_access constants.%Negate.impl_witness.943, element1 [concrete = constants.%Op.5c0]
// CHECK:STDOUT:   %bound_method.loc9_44.1: <bound method> = bound_method %u.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Op.11(constants.%int_32) [concrete = constants.%Op.specific_fn.f32]
// CHECK:STDOUT:   %bound_method.loc9_44.2: <bound method> = bound_method %u.ref, %specific_fn
// CHECK:STDOUT:   %int.snegate: init %i32 = call %bound_method.loc9_44.2(%u.ref)
// CHECK:STDOUT:   %.loc9_46.1: %i32 = value_of_initializer %int.snegate
// CHECK:STDOUT:   %.loc9_46.2: %i32 = converted %int.snegate, %.loc9_46.1
// CHECK:STDOUT:   return %.loc9_46.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallMethod(%x.param: %empty_tuple.type) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.572]
// CHECK:STDOUT:   %impl.elem1: %.597 = impl_witness_access constants.%J.impl_witness.e42, element1 [concrete = constants.%F.46e]
// CHECK:STDOUT:   %bound_method.loc13_11: <bound method> = bound_method %x.ref, %impl.elem1
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40.f80]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_18.1: <bound method> = bound_method %int_40, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_18.2: <bound method> = bound_method %int_40, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc13_18.2(%int_40) [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %.loc13_18.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %.loc13_18.2: %i32 = converted %int_40, %.loc13_18.1 [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %F.call: init %i32 = call %bound_method.loc13_11(%x.ref, %.loc13_18.2)
// CHECK:STDOUT:   %.loc13_21.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc13_21.2: %i32 = converted %F.call, %.loc13_21.1
// CHECK:STDOUT:   return %.loc13_21.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%self.param: %C, %u.param: %C) -> %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   return %self.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.521) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type.3df
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.4ca
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.646
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.11f
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.b98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.330) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.ecc) {
// CHECK:STDOUT:   %Self => constants.%J.facet.ecc
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness.e42
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.30e) {
// CHECK:STDOUT:   %Self => constants.%J.facet.30e
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.c48
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness.12d
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.c48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_type_in_function_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.646: <witness> = lookup_impl_witness %Self.ccd, @J [symbolic]
// CHECK:STDOUT:   %J.facet.521: %J.type = facet_value %Self.as_type.3df, (%J.lookup_impl_witness.646) [symbolic]
// CHECK:STDOUT:   %impl.elem0.11f: type = impl_witness_access %J.lookup_impl_witness.646, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b98: type = pattern_type %impl.elem0.11f [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.572: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.968: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.78d: type = facet_access_type %.Self.968 [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.25c: <witness> = lookup_impl_witness %.Self.968, @J [symbolic_self]
// CHECK:STDOUT:   %J.facet.330: %J.type = facet_value %.Self.as_type.78d, (%J.lookup_impl_witness.25c) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.83f: type = impl_witness_access %J.lookup_impl_witness.25c, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.83f = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness file.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type.5d3: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.b9c: %F.type.5d3 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.183: %J.type = facet_value %D, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Op.type.e42: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.type.fc5: type = fn_type @Op.11, @impl.47e(%N) [symbolic]
// CHECK:STDOUT:   %Op.8d0: %Op.type.fc5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Negate.impl_witness.943: <witness> = impl_witness imports.%Negate.impl_witness_table, @impl.47e(%int_32) [concrete]
// CHECK:STDOUT:   %Op.type.66f: type = fn_type @Op.11, @impl.47e(%int_32) [concrete]
// CHECK:STDOUT:   %Op.5c0: %Op.type.66f = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %i32, (%Negate.impl_witness.943) [concrete]
// CHECK:STDOUT:   %.6f0: type = fn_type_with_self_type %Op.type.e42, %Negate.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn.f32: <specific function> = specific_function %Op.5c0, @Op.11(%int_32) [concrete]
// CHECK:STDOUT:   %CallFunction.type: type = fn_type @CallFunction [concrete]
// CHECK:STDOUT:   %CallFunction: %CallFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %.490: type = fn_type_with_self_type %F.type.c14, %J.facet.183 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/operators/arithmetic, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.5f5145.5 = import_ref Core//prelude/types/int, loc112_43, unloaded
// CHECK:STDOUT:   %Core.import_ref.ff2: @impl.47e.%Op.type (%Op.type.fc5) = import_ref Core//prelude/types/int, loc113_31, loaded [symbolic = @impl.47e.%Op (constants.%Op.8d0)]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.5f5145.5, %Core.import_ref.ff2), @impl.47e [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:   impl_decl @impl.fc1 [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:     %.loc10_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.25c, element0 [symbolic_self = constants.%impl.elem0.83f]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_13: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @impl.fc1.%F.decl), @impl.fc1 [concrete]
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
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %u.patt: @F.1.%pattern_type (%pattern_type.b98) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @F.1.%pattern_type (%pattern_type.b98) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.1.%pattern_type (%pattern_type.b98) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.1.%pattern_type (%pattern_type.b98) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.646, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.646, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1.572]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.fc1: %D.ref as %.loc10_13 {
// CHECK:STDOUT:   %F.decl: %F.type.5d3 = fn_decl @F.2 [concrete = constants.%F.b9c] {
// CHECK:STDOUT:     %u.patt: %pattern_type.7ce = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.7ce = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc11_11: type = splice_block %i32.loc11_11 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.646)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.b98)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f)) -> @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%u.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %impl.elem1: %.6f0 = impl_witness_access constants.%Negate.impl_witness.943, element1 [concrete = constants.%Op.5c0]
// CHECK:STDOUT:   %bound_method.loc11_32.1: <bound method> = bound_method %u.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Op.11(constants.%int_32) [concrete = constants.%Op.specific_fn.f32]
// CHECK:STDOUT:   %bound_method.loc11_32.2: <bound method> = bound_method %u.ref, %specific_fn
// CHECK:STDOUT:   %int.snegate: init %i32 = call %bound_method.loc11_32.2(%u.ref)
// CHECK:STDOUT:   %.loc11_34.1: %i32 = value_of_initializer %int.snegate
// CHECK:STDOUT:   %.loc11_34.2: %i32 = converted %int.snegate, %.loc11_34.1
// CHECK:STDOUT:   return %.loc11_34.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFunction() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.572]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value constants.%D, (constants.%J.impl_witness) [concrete = constants.%J.facet.183]
// CHECK:STDOUT:   %.loc15_11: %J.type = converted %D.ref, %J.facet [concrete = constants.%J.facet.183]
// CHECK:STDOUT:   %impl.elem1: %.490 = impl_witness_access constants.%J.impl_witness, element1 [concrete = constants.%F.b9c]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_18.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_18.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc15_18.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc15_18.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc15_18.2: %i32 = converted %int_4, %.loc15_18.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %F.call: init %i32 = call %impl.elem1(%.loc15_18.2)
// CHECK:STDOUT:   %.loc15_20.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc15_20.2: %i32 = converted %F.call, %.loc15_20.1
// CHECK:STDOUT:   return %.loc15_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.521) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.646
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.11f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.330) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.183) {
// CHECK:STDOUT:   %Self => constants.%J.facet.183
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_with_associated_type_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.64628e.1: <witness> = lookup_impl_witness %Self.ccd, @J [symbolic]
// CHECK:STDOUT:   %J.facet.5210a7.1: %J.type = facet_value %Self.as_type.3df, (%J.lookup_impl_witness.64628e.1) [symbolic]
// CHECK:STDOUT:   %impl.elem0.11fef0.1: type = impl_witness_access %J.lookup_impl_witness.64628e.1, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b984e7.1: type = pattern_type %impl.elem0.11fef0.1 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.572: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %pattern_type.4ca604.1: type = pattern_type %Self.as_type.3df [symbolic]
// CHECK:STDOUT:   %G.type.285: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.e4b: %G.type.285 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc2.d75: %J.assoc_type = assoc_entity element2, @J.%G.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %.Self.968: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.78d: type = facet_access_type %.Self.968 [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.25c: <witness> = lookup_impl_witness %.Self.968, @J [symbolic_self]
// CHECK:STDOUT:   %J.facet.330: %J.type = facet_value %.Self.as_type.78d, (%J.lookup_impl_witness.25c) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.83f: type = impl_witness_access %J.lookup_impl_witness.25c, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.83f = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type.b84: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.b07: %F.type.b84 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a4a: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %G.type.4b6: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.553: %G.type.4b6 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.ee5: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Op.type.e42: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.type.fc5: type = fn_type @Op.11, @impl.47e(%N) [symbolic]
// CHECK:STDOUT:   %Op.8d0: %Op.type.fc5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Negate.impl_witness.943: <witness> = impl_witness imports.%Negate.impl_witness_table, @impl.47e(%int_32) [concrete]
// CHECK:STDOUT:   %Op.type.66f: type = fn_type @Op.11, @impl.47e(%int_32) [concrete]
// CHECK:STDOUT:   %Op.5c0: %Op.type.66f = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %i32, (%Negate.impl_witness.943) [concrete]
// CHECK:STDOUT:   %.6f0: type = fn_type_with_self_type %Op.type.e42, %Negate.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn.f32: <specific function> = specific_function %Op.5c0, @Op.11(%int_32) [concrete]
// CHECK:STDOUT:   %CallBoth.type: type = fn_type @CallBoth [concrete]
// CHECK:STDOUT:   %CallBoth: %CallBoth.type = struct_value () [concrete]
// CHECK:STDOUT:   %.29b: type = fn_type_with_self_type %F.type.c14, %J.facet.ee5 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %.ff5: type = fn_type_with_self_type %G.type.285, %J.facet.ee5 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.a25: <bound method> = bound_method %int_5.64b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Convert.bound.ce9: <bound method> = bound_method %int_6.462, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.efa: <bound method> = bound_method %int_6.462, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %T.ccd: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.28e: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.as_type.3df: type = facet_access_type %T.ccd [symbolic]
// CHECK:STDOUT:   %pattern_type.4ca604.2: type = pattern_type %T.as_type.3df [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.64628e.2: <witness> = lookup_impl_witness %T.ccd, @J [symbolic]
// CHECK:STDOUT:   %J.facet.5210a7.2: %J.type = facet_value %T.as_type.3df, (%J.lookup_impl_witness.64628e.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.11fef0.2: type = impl_witness_access %J.lookup_impl_witness.64628e.2, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b984e7.2: type = pattern_type %impl.elem0.11fef0.2 [symbolic]
// CHECK:STDOUT:   %GenericCallF.type: type = fn_type @GenericCallF [concrete]
// CHECK:STDOUT:   %GenericCallF: %GenericCallF.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.58f: <witness> = require_complete_type %impl.elem0.11fef0.2 [symbolic]
// CHECK:STDOUT:   %require_complete.2a5: <witness> = require_complete_type %T.as_type.3df [symbolic]
// CHECK:STDOUT:   %.d43: type = fn_type_with_self_type %F.type.c14, %J.facet.5210a7.2 [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.d43 = impl_witness_access %J.lookup_impl_witness.64628e.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fe8: <specific function> = specific_impl_function %impl.elem1, @F.1(%J.facet.5210a7.2) [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericCallF.specific_fn: <specific function> = specific_function %GenericCallF, @GenericCallF(%J.facet.ee5) [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/operators/arithmetic, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.5f5145.5 = import_ref Core//prelude/types/int, loc112_43, unloaded
// CHECK:STDOUT:   %Core.import_ref.ff2: @impl.47e.%Op.type (%Op.type.fc5) = import_ref Core//prelude/types/int, loc113_31, loaded [symbolic = @impl.47e.%Op (constants.%Op.8d0)]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.5f5145.5, %Core.import_ref.ff2), @impl.47e [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %e.patt: %pattern_type.a4a = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: %pattern_type.a4a = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref.loc20: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericCallF.decl: %GenericCallF.type = fn_decl @GenericCallF [concrete = constants.%GenericCallF] {
// CHECK:STDOUT:     %T.patt: %pattern_type.28e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericCallF.%pattern_type.loc28_24 (%pattern_type.4ca604.2) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericCallF.%pattern_type.loc28_24 (%pattern_type.4ca604.2) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.b984e7.2) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.b984e7.2) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.b984e7.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.b984e7.2) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc28_41: %J.type = name_ref T, %T.loc28_17.1 [symbolic = %T.loc28_17.2 (constants.%T.ccd)]
// CHECK:STDOUT:     %U.ref.loc28_42: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %T.as_type.loc28_42: type = facet_access_type %T.ref.loc28_41 [symbolic = %T.as_type.loc28_27.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     %.loc28_42: type = converted %T.ref.loc28_41, %T.as_type.loc28_42 [symbolic = %T.as_type.loc28_27.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     %impl.elem0.loc28_42: type = impl_witness_access constants.%J.lookup_impl_witness.64628e.2, element0 [symbolic = %impl.elem0.loc28_34.2 (constants.%impl.elem0.11fef0.2)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc28_17.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc28_17.2 (constants.%T.ccd)]
// CHECK:STDOUT:     %t.param: @GenericCallF.%T.as_type.loc28_27.2 (%T.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc28_27.1: type = splice_block %.loc28_27.2 [symbolic = %T.as_type.loc28_27.2 (constants.%T.as_type.3df)] {
// CHECK:STDOUT:       %T.ref.loc28_27: %J.type = name_ref T, %T.loc28_17.1 [symbolic = %T.loc28_17.2 (constants.%T.ccd)]
// CHECK:STDOUT:       %T.as_type.loc28_27.1: type = facet_access_type %T.ref.loc28_27 [symbolic = %T.as_type.loc28_27.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:       %.loc28_27.2: type = converted %T.ref.loc28_27, %T.as_type.loc28_27.1 [symbolic = %T.as_type.loc28_27.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericCallF.%T.as_type.loc28_27.2 (%T.as_type.3df) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: @GenericCallF.%impl.elem0.loc28_34.2 (%impl.elem0.11fef0.2) = value_param call_param1
// CHECK:STDOUT:     %.loc28_34.1: type = splice_block %impl.elem0.loc28_34.1 [symbolic = %impl.elem0.loc28_34.2 (constants.%impl.elem0.11fef0.2)] {
// CHECK:STDOUT:       %T.ref.loc28_33: %J.type = name_ref T, %T.loc28_17.1 [symbolic = %T.loc28_17.2 (constants.%T.ccd)]
// CHECK:STDOUT:       %U.ref.loc28_34: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:       %T.as_type.loc28_34: type = facet_access_type %T.ref.loc28_33 [symbolic = %T.as_type.loc28_27.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:       %.loc28_34.2: type = converted %T.ref.loc28_33, %T.as_type.loc28_34 [symbolic = %T.as_type.loc28_27.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:       %impl.elem0.loc28_34.1: type = impl_witness_access constants.%J.lookup_impl_witness.64628e.2, element0 [symbolic = %impl.elem0.loc28_34.2 (constants.%impl.elem0.11fef0.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericCallF.%impl.elem0.loc28_34.2 (%impl.elem0.11fef0.2) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericCallF.%impl.elem0.loc28_34.2 (%impl.elem0.11fef0.2) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericCallF.%impl.elem0.loc28_34.2 (%impl.elem0.11fef0.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %e.patt: %pattern_type.a4a = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: %pattern_type.a4a = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = bind_name e, %e.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %u.patt: @F.1.%pattern_type (%pattern_type.b984e7.1) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @F.1.%pattern_type (%pattern_type.b984e7.1) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.1.%pattern_type (%pattern_type.b984e7.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.1.%pattern_type (%pattern_type.b984e7.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.64628e.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11fef0.1) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11fef0.1)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.64628e.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11fef0.1) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11fef0.1) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11fef0.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1.572]
// CHECK:STDOUT:   %G.decl: %G.type.285 = fn_decl @G.1 [concrete = constants.%G.e4b] {
// CHECK:STDOUT:     %self.patt: @G.1.%pattern_type.loc6_8 (%pattern_type.4ca604.1) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @G.1.%pattern_type.loc6_8 (%pattern_type.4ca604.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: @G.1.%pattern_type.loc6_20 (%pattern_type.b984e7.1) = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: @G.1.%pattern_type.loc6_20 (%pattern_type.b984e7.1) = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @G.1.%pattern_type.loc6_20 (%pattern_type.b984e7.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @G.1.%pattern_type.loc6_20 (%pattern_type.b984e7.1) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc6_29: type = impl_witness_access constants.%J.lookup_impl_witness.64628e.1, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:     %U.ref.loc6_29: type = name_ref U, %impl.elem0.loc6_29 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:     %self.param: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc6_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.11fef0.1) = value_param call_param1
// CHECK:STDOUT:     %.loc6_23: type = splice_block %U.ref.loc6_23 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.11fef0.1)] {
// CHECK:STDOUT:       %impl.elem0.loc6_23.2: type = impl_witness_access constants.%J.lookup_impl_witness.64628e.1, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:       %U.ref.loc6_23: type = name_ref U, %impl.elem0.loc6_23.2 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.11fef0.1) = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.11fef0.1) = out_param call_param2
// CHECK:STDOUT:     %return: ref @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.11fef0.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc2: %J.assoc_type = assoc_entity element2, %G.decl [concrete = constants.%assoc2.d75]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   .G = %assoc2
// CHECK:STDOUT:   witness = (%U, %F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.939: %Self.ref as %.loc10_20 {
// CHECK:STDOUT:   %F.decl: %F.type.b84 = fn_decl @F.2 [concrete = constants.%F.b07] {
// CHECK:STDOUT:     %u.patt: %pattern_type.7ce = binding_pattern u [concrete]
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
// CHECK:STDOUT:     %u: %i32 = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.4b6 = fn_decl @G.2 [concrete = constants.%G.553] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a4a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a4a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: %pattern_type.7ce = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc14_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %E = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %self: %E = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc14: type = splice_block %i32.loc14_25 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc14_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: %i32 = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = @E.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @impl.939 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:     %.loc10_26: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.25c, element0 [symbolic_self = constants.%impl.elem0.83f]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_20: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @impl.939.%F.decl, @impl.939.%G.decl), @impl.939 [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   extend @impl.939.%.loc10_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.64628e.1)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.b984e7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11fef0.1)) -> @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11fef0.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc6_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:   %pattern_type.loc6_8: type = pattern_type %Self.as_type.loc6_14.1 [symbolic = %pattern_type.loc6_8 (constants.%pattern_type.4ca604.1)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.64628e.1)]
// CHECK:STDOUT:   %impl.elem0.loc6_23.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:   %pattern_type.loc6_20: type = pattern_type %impl.elem0.loc6_23.1 [symbolic = %pattern_type.loc6_20 (constants.%pattern_type.b984e7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df), %v.param: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.11fef0.1)) -> @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.11fef0.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%u.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %impl.elem1: %.6f0 = impl_witness_access constants.%Negate.impl_witness.943, element1 [concrete = constants.%Op.5c0]
// CHECK:STDOUT:   %bound_method.loc12_14.1: <bound method> = bound_method %u.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Op.11(constants.%int_32) [concrete = constants.%Op.specific_fn.f32]
// CHECK:STDOUT:   %bound_method.loc12_14.2: <bound method> = bound_method %u.ref, %specific_fn
// CHECK:STDOUT:   %int.snegate: init %i32 = call %bound_method.loc12_14.2(%u.ref)
// CHECK:STDOUT:   %.loc12_16.1: %i32 = value_of_initializer %int.snegate
// CHECK:STDOUT:   %.loc12_16.2: %i32 = converted %int.snegate, %.loc12_16.1
// CHECK:STDOUT:   return %.loc12_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2(%self.param: %E, %v.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: %i32 = name_ref v, %v
// CHECK:STDOUT:   %impl.elem1: %.6f0 = impl_witness_access constants.%Negate.impl_witness.943, element1 [concrete = constants.%Op.5c0]
// CHECK:STDOUT:   %bound_method.loc15_14.1: <bound method> = bound_method %v.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Op.11(constants.%int_32) [concrete = constants.%Op.specific_fn.f32]
// CHECK:STDOUT:   %bound_method.loc15_14.2: <bound method> = bound_method %v.ref, %specific_fn
// CHECK:STDOUT:   %int.snegate: init %i32 = call %bound_method.loc15_14.2(%v.ref)
// CHECK:STDOUT:   %.loc15_16.1: %i32 = value_of_initializer %int.snegate
// CHECK:STDOUT:   %.loc15_16.2: %i32 = converted %int.snegate, %.loc15_16.1
// CHECK:STDOUT:   return %.loc15_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallBoth(%e.param: %E) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e1.patt: %pattern_type.7ce = binding_pattern e1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc21: %E = name_ref e, %e
// CHECK:STDOUT:   %F.ref.loc21: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.572]
// CHECK:STDOUT:   %impl.elem1.loc21: %.29b = impl_witness_access constants.%J.impl_witness, element1 [concrete = constants.%F.b07]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc21: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_21.1: <bound method> = bound_method %int_2, %impl.elem0.loc21 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_21.2: <bound method> = bound_method %int_2, %specific_fn.loc21 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %bound_method.loc21_21.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc21_21.1: %i32 = value_of_initializer %int.convert_checked.loc21 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc21_21.2: %i32 = converted %int_2, %.loc21_21.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %F.call.loc21: init %i32 = call %impl.elem1.loc21(%.loc21_21.2)
// CHECK:STDOUT:   %.loc21_11: type = splice_block %i32.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_22.1: %i32 = value_of_initializer %F.call.loc21
// CHECK:STDOUT:   %.loc21_22.2: %i32 = converted %F.call.loc21, %.loc21_22.1
// CHECK:STDOUT:   %e1: %i32 = bind_name e1, %.loc21_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e2.patt: %pattern_type.7ce = binding_pattern e2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc22: %E = name_ref e, %e
// CHECK:STDOUT:   %G.ref.loc22: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2.d75]
// CHECK:STDOUT:   %impl.elem2.loc22: %.ff5 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%G.553]
// CHECK:STDOUT:   %bound_method.loc22_18: <bound method> = bound_method %e.ref.loc22, %impl.elem2.loc22
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc22: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_21.1: <bound method> = bound_method %int_3, %impl.elem0.loc22 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_21.2: <bound method> = bound_method %int_3, %specific_fn.loc22 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc22: init %i32 = call %bound_method.loc22_21.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22_21.1: %i32 = value_of_initializer %int.convert_checked.loc22 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22_21.2: %i32 = converted %int_3, %.loc22_21.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %G.call.loc22: init %i32 = call %bound_method.loc22_18(%e.ref.loc22, %.loc22_21.2)
// CHECK:STDOUT:   %.loc22_11: type = splice_block %i32.loc22 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_22.1: %i32 = value_of_initializer %G.call.loc22
// CHECK:STDOUT:   %.loc22_22.2: %i32 = converted %G.call.loc22, %.loc22_22.1
// CHECK:STDOUT:   %e2: %i32 = bind_name e2, %.loc22_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e3.patt: %pattern_type.7ce = binding_pattern e3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.ref.loc23: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %F.ref.loc23: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.572]
// CHECK:STDOUT:   %impl.elem1.loc23: %.29b = impl_witness_access constants.%J.impl_witness, element1 [concrete = constants.%F.b07]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc23: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc23_21.1: <bound method> = bound_method %int_4, %impl.elem0.loc23 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_21.2: <bound method> = bound_method %int_4, %specific_fn.loc23 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc23: init %i32 = call %bound_method.loc23_21.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc23_21.1: %i32 = value_of_initializer %int.convert_checked.loc23 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc23_21.2: %i32 = converted %int_4, %.loc23_21.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %F.call.loc23: init %i32 = call %impl.elem1.loc23(%.loc23_21.2)
// CHECK:STDOUT:   %.loc23_11: type = splice_block %i32.loc23 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_22.1: %i32 = value_of_initializer %F.call.loc23
// CHECK:STDOUT:   %.loc23_22.2: %i32 = converted %F.call.loc23, %.loc23_22.1
// CHECK:STDOUT:   %e3: %i32 = bind_name e3, %.loc23_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e4.patt: %pattern_type.7ce = binding_pattern e4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc24: %E = name_ref e, %e
// CHECK:STDOUT:   %E.ref.loc24: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %G.ref.loc24: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2.d75]
// CHECK:STDOUT:   %impl.elem2.loc24: %.ff5 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%G.553]
// CHECK:STDOUT:   %bound_method.loc24_18: <bound method> = bound_method %e.ref.loc24, %impl.elem2.loc24
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc24: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc24_25.1: <bound method> = bound_method %int_5, %impl.elem0.loc24 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_25.2: <bound method> = bound_method %int_5, %specific_fn.loc24 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %int.convert_checked.loc24: init %i32 = call %bound_method.loc24_25.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc24_25.1: %i32 = value_of_initializer %int.convert_checked.loc24 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc24_25.2: %i32 = converted %int_5, %.loc24_25.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %G.call.loc24: init %i32 = call %bound_method.loc24_18(%e.ref.loc24, %.loc24_25.2)
// CHECK:STDOUT:   %.loc24_11: type = splice_block %i32.loc24 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24_26.1: %i32 = value_of_initializer %G.call.loc24
// CHECK:STDOUT:   %.loc24_26.2: %i32 = converted %G.call.loc24, %.loc24_26.1
// CHECK:STDOUT:   %e4: %i32 = bind_name e4, %.loc24_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e5.patt: %pattern_type.7ce = binding_pattern e5 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc25: %E = name_ref e, %e
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %G.ref.loc25: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2.d75]
// CHECK:STDOUT:   %impl.elem2.loc25: %.ff5 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%G.553]
// CHECK:STDOUT:   %bound_method.loc25_18: <bound method> = bound_method %e.ref.loc25, %impl.elem2.loc25
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %impl.elem0.loc25: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc25_25.1: <bound method> = bound_method %int_6, %impl.elem0.loc25 [concrete = constants.%Convert.bound.ce9]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_25.2: <bound method> = bound_method %int_6, %specific_fn.loc25 [concrete = constants.%bound_method.efa]
// CHECK:STDOUT:   %int.convert_checked.loc25: init %i32 = call %bound_method.loc25_25.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc25_25.1: %i32 = value_of_initializer %int.convert_checked.loc25 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc25_25.2: %i32 = converted %int_6, %.loc25_25.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %G.call.loc25: init %i32 = call %bound_method.loc25_18(%e.ref.loc25, %.loc25_25.2)
// CHECK:STDOUT:   %.loc25_11: type = splice_block %i32.loc25 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25_26.1: %i32 = value_of_initializer %G.call.loc25
// CHECK:STDOUT:   %.loc25_26.2: %i32 = converted %G.call.loc25, %.loc25_26.1
// CHECK:STDOUT:   %e5: %i32 = bind_name e5, %.loc25_26.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallF(%T.loc28_17.1: %J.type) {
// CHECK:STDOUT:   %T.loc28_17.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc28_17.2 (constants.%T.ccd)]
// CHECK:STDOUT:   %T.as_type.loc28_27.2: type = facet_access_type %T.loc28_17.2 [symbolic = %T.as_type.loc28_27.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:   %pattern_type.loc28_24: type = pattern_type %T.as_type.loc28_27.2 [symbolic = %pattern_type.loc28_24 (constants.%pattern_type.4ca604.2)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc28_17.2, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.64628e.2)]
// CHECK:STDOUT:   %impl.elem0.loc28_34.2: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_34.2 (constants.%impl.elem0.11fef0.2)]
// CHECK:STDOUT:   %pattern_type.loc28_30: type = pattern_type %impl.elem0.loc28_34.2 [symbolic = %pattern_type.loc28_30 (constants.%pattern_type.b984e7.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28_38: <witness> = require_complete_type %impl.elem0.loc28_34.2 [symbolic = %require_complete.loc28_38 (constants.%require_complete.58f)]
// CHECK:STDOUT:   %require_complete.loc28_25: <witness> = require_complete_type %T.as_type.loc28_27.2 [symbolic = %require_complete.loc28_25 (constants.%require_complete.2a5)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type.loc28_27.2, (%J.lookup_impl_witness) [symbolic = %J.facet (constants.%J.facet.5210a7.2)]
// CHECK:STDOUT:   %.loc29_11.2: type = fn_type_with_self_type constants.%F.type.c14, %J.facet [symbolic = %.loc29_11.2 (constants.%.d43)]
// CHECK:STDOUT:   %impl.elem1.loc29_11.2: @GenericCallF.%.loc29_11.2 (%.d43) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc29_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc29_11.2: <specific function> = specific_impl_function %impl.elem1.loc29_11.2, @F.1(%J.facet) [symbolic = %specific_impl_fn.loc29_11.2 (constants.%specific_impl_fn.fe8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericCallF.%T.as_type.loc28_27.2 (%T.as_type.3df), %u.param: @GenericCallF.%impl.elem0.loc28_34.2 (%impl.elem0.11fef0.2)) -> @GenericCallF.%impl.elem0.loc28_34.2 (%impl.elem0.11fef0.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallF.%T.as_type.loc28_27.2 (%T.as_type.3df) = name_ref t, %t
// CHECK:STDOUT:     %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.572]
// CHECK:STDOUT:     %T.as_type.loc29: type = facet_access_type constants.%T.ccd [symbolic = %T.as_type.loc28_27.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     %.loc29_11.1: type = converted constants.%T.ccd, %T.as_type.loc29 [symbolic = %T.as_type.loc28_27.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     %impl.elem1.loc29_11.1: @GenericCallF.%.loc29_11.2 (%.d43) = impl_witness_access constants.%J.lookup_impl_witness.64628e.2, element1 [symbolic = %impl.elem1.loc29_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref: @GenericCallF.%impl.elem0.loc28_34.2 (%impl.elem0.11fef0.2) = name_ref u, %u
// CHECK:STDOUT:     %specific_impl_fn.loc29_11.1: <specific function> = specific_impl_function %impl.elem1.loc29_11.1, @F.1(constants.%J.facet.5210a7.2) [symbolic = %specific_impl_fn.loc29_11.2 (constants.%specific_impl_fn.fe8)]
// CHECK:STDOUT:     %.loc29_15: init @GenericCallF.%impl.elem0.loc28_34.2 (%impl.elem0.11fef0.2) = call %specific_impl_fn.loc29_11.1(%u.ref)
// CHECK:STDOUT:     %.loc29_16.1: @GenericCallF.%impl.elem0.loc28_34.2 (%impl.elem0.11fef0.2) = value_of_initializer %.loc29_15
// CHECK:STDOUT:     %.loc29_16.2: @GenericCallF.%impl.elem0.loc28_34.2 (%impl.elem0.11fef0.2) = converted %.loc29_15, %.loc29_16.1
// CHECK:STDOUT:     return %.loc29_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGeneric(%e.param: %E) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %GenericCallF.ref: %GenericCallF.type = name_ref GenericCallF, file.%GenericCallF.decl [concrete = constants.%GenericCallF]
// CHECK:STDOUT:   %e.ref: %E = name_ref e, %e
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %J.facet.loc33_27.1: %J.type = facet_value constants.%E, (constants.%J.impl_witness) [concrete = constants.%J.facet.ee5]
// CHECK:STDOUT:   %.loc33_27.1: %J.type = converted constants.%E, %J.facet.loc33_27.1 [concrete = constants.%J.facet.ee5]
// CHECK:STDOUT:   %J.facet.loc33_27.2: %J.type = facet_value constants.%E, (constants.%J.impl_witness) [concrete = constants.%J.facet.ee5]
// CHECK:STDOUT:   %.loc33_27.2: %J.type = converted constants.%E, %J.facet.loc33_27.2 [concrete = constants.%J.facet.ee5]
// CHECK:STDOUT:   %GenericCallF.specific_fn: <specific function> = specific_function %GenericCallF.ref, @GenericCallF(constants.%J.facet.ee5) [concrete = constants.%GenericCallF.specific_fn]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc33_26.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_26.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc33_26.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc33_26.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc33_26.2: %i32 = converted %int_2, %.loc33_26.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %GenericCallF.call: init %i32 = call %GenericCallF.specific_fn(%e.ref, %.loc33_26.2)
// CHECK:STDOUT:   %.loc33_28.1: %i32 = value_of_initializer %GenericCallF.call
// CHECK:STDOUT:   %.loc33_28.2: %i32 = converted %GenericCallF.call, %.loc33_28.1
// CHECK:STDOUT:   return %.loc33_28.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.5210a7.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.64628e.1
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.11fef0.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b984e7.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc6_14.1 => constants.%Self.as_type.3df
// CHECK:STDOUT:   %pattern_type.loc6_8 => constants.%pattern_type.4ca604.1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.64628e.1
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%impl.elem0.11fef0.1
// CHECK:STDOUT:   %pattern_type.loc6_20 => constants.%pattern_type.b984e7.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.330) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.ee5) {
// CHECK:STDOUT:   %Self => constants.%J.facet.ee5
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%J.facet.ee5) {
// CHECK:STDOUT:   %Self => constants.%J.facet.ee5
// CHECK:STDOUT:   %Self.as_type.loc6_14.1 => constants.%E
// CHECK:STDOUT:   %pattern_type.loc6_8 => constants.%pattern_type.a4a
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc6_20 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.5210a7.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallF(constants.%T.ccd) {
// CHECK:STDOUT:   %T.loc28_17.2 => constants.%T.ccd
// CHECK:STDOUT:   %T.as_type.loc28_27.2 => constants.%T.as_type.3df
// CHECK:STDOUT:   %pattern_type.loc28_24 => constants.%pattern_type.4ca604.2
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.64628e.2
// CHECK:STDOUT:   %impl.elem0.loc28_34.2 => constants.%impl.elem0.11fef0.2
// CHECK:STDOUT:   %pattern_type.loc28_30 => constants.%pattern_type.b984e7.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.5210a7.2) {
// CHECK:STDOUT:   %Self => constants.%J.facet.5210a7.2
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.64628e.2
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.11fef0.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b984e7.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallF(constants.%J.facet.ee5) {
// CHECK:STDOUT:   %T.loc28_17.2 => constants.%J.facet.ee5
// CHECK:STDOUT:   %T.as_type.loc28_27.2 => constants.%E
// CHECK:STDOUT:   %pattern_type.loc28_24 => constants.%pattern_type.a4a
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc28_34.2 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc28_30 => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28_38 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc28_25 => constants.%complete_type.357
// CHECK:STDOUT:   %J.facet => constants.%J.facet.ee5
// CHECK:STDOUT:   %.loc29_11.2 => constants.%.29b
// CHECK:STDOUT:   %impl.elem1.loc29_11.2 => constants.%F.b07
// CHECK:STDOUT:   %specific_impl_fn.loc29_11.2 => constants.%F.b07
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_constraint_on_associated_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type.b70 [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op [concrete]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.ea8: %I.assoc_type = assoc_entity element0, @I.%Op.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.5be: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.64628e.1: <witness> = lookup_impl_witness %Self.ccd, @J [symbolic]
// CHECK:STDOUT:   %J.facet.5210a7.1: %J.type = facet_value %Self.as_type.3df, (%J.lookup_impl_witness.64628e.1) [symbolic]
// CHECK:STDOUT:   %impl.elem0.5205fd.1: %I.type = impl_witness_access %J.lookup_impl_witness.64628e.1, element0 [symbolic]
// CHECK:STDOUT:   %as_type.f5d8f3.1: type = facet_access_type %impl.elem0.5205fd.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.74afd4.1: type = pattern_type %as_type.f5d8f3.1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %T: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.28e: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4ca: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.64628e.2: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %J.facet.5210a7.2: %J.type = facet_value %T.as_type, (%J.lookup_impl_witness.64628e.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.5205fd.2: %I.type = impl_witness_access %J.lookup_impl_witness.64628e.2, element0 [symbolic]
// CHECK:STDOUT:   %as_type.f5d8f3.2: type = facet_access_type %impl.elem0.5205fd.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.74afd4.2: type = pattern_type %as_type.f5d8f3.2 [symbolic]
// CHECK:STDOUT:   %GenericResult.type: type = fn_type @GenericResult [concrete]
// CHECK:STDOUT:   %GenericResult: %GenericResult.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.b33: <witness> = require_complete_type %as_type.f5d8f3.2 [symbolic]
// CHECK:STDOUT:   %require_complete.2a5: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.d43: type = fn_type_with_self_type %F.type, %J.facet.5210a7.2 [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.d43 = impl_witness_access %J.lookup_impl_witness.64628e.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fe8: <specific function> = specific_impl_function %impl.elem1, @F(%J.facet.5210a7.2) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.5205fd.2, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %as_type.f5d8f3.2, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.c9a: type = fn_type_with_self_type %Op.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.45a: %.c9a = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ec: <specific function> = specific_impl_function %impl.elem0.45a, @Op(%I.facet) [symbolic]
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
// CHECK:STDOUT:     .GenericResult = %GenericResult.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %GenericResult.decl: %GenericResult.type = fn_decl @GenericResult [concrete = constants.%GenericResult] {
// CHECK:STDOUT:     %T.patt: %pattern_type.28e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericResult.%pattern_type.loc12_25 (%pattern_type.4ca) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericResult.%pattern_type.loc12_25 (%pattern_type.4ca) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.74afd4.2) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.74afd4.2) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.74afd4.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.74afd4.2) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_42: %J.type = name_ref T, %T.loc12_18.1 [symbolic = %T.loc12_18.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc12_43: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.5be]
// CHECK:STDOUT:     %T.as_type.loc12_43: type = facet_access_type %T.ref.loc12_42 [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc12_43.1: type = converted %T.ref.loc12_42, %T.as_type.loc12_43 [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc12_43: %I.type = impl_witness_access constants.%J.lookup_impl_witness.64628e.2, element0 [symbolic = %impl.elem0.loc12_35.2 (constants.%impl.elem0.5205fd.2)]
// CHECK:STDOUT:     %as_type.loc12_43: type = facet_access_type %impl.elem0.loc12_43 [symbolic = %as_type.loc12_35.2 (constants.%as_type.f5d8f3.2)]
// CHECK:STDOUT:     %.loc12_43.2: type = converted %impl.elem0.loc12_43, %as_type.loc12_43 [symbolic = %as_type.loc12_35.2 (constants.%as_type.f5d8f3.2)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc12_18.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_18.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericResult.%T.as_type.loc12_28.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc12_28.1: type = splice_block %.loc12_28.2 [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc12_28: %J.type = name_ref T, %T.loc12_18.1 [symbolic = %T.loc12_18.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc12_28.1: type = facet_access_type %T.ref.loc12_28 [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc12_28.2: type = converted %T.ref.loc12_28, %T.as_type.loc12_28.1 [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericResult.%T.as_type.loc12_28.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = value_param call_param1
// CHECK:STDOUT:     %.loc12_35.1: type = splice_block %.loc12_35.3 [symbolic = %as_type.loc12_35.2 (constants.%as_type.f5d8f3.2)] {
// CHECK:STDOUT:       %T.ref.loc12_34: %J.type = name_ref T, %T.loc12_18.1 [symbolic = %T.loc12_18.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc12_35: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.5be]
// CHECK:STDOUT:       %T.as_type.loc12_35: type = facet_access_type %T.ref.loc12_34 [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc12_35.2: type = converted %T.ref.loc12_34, %T.as_type.loc12_35 [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc12_35.1: %I.type = impl_witness_access constants.%J.lookup_impl_witness.64628e.2, element0 [symbolic = %impl.elem0.loc12_35.2 (constants.%impl.elem0.5205fd.2)]
// CHECK:STDOUT:       %as_type.loc12_35.1: type = facet_access_type %impl.elem0.loc12_35.1 [symbolic = %as_type.loc12_35.2 (constants.%as_type.f5d8f3.2)]
// CHECK:STDOUT:       %.loc12_35.3: type = converted %impl.elem0.loc12_35.1, %as_type.loc12_35.1 [symbolic = %as_type.loc12_35.2 (constants.%as_type.f5d8f3.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %Op.decl: %Op.type = fn_decl @Op [concrete = constants.%Op] {
// CHECK:STDOUT:     %self.patt: @Op.%pattern_type (%pattern_type.6de) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Op.%pattern_type (%pattern_type.6de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: @Op.%pattern_type (%pattern_type.6de) = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: @Op.%pattern_type (%pattern_type.6de) = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @Op.%pattern_type (%pattern_type.6de) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Op.%pattern_type (%pattern_type.6de) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc4_33: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:     %Self.as_type.loc4_33: type = facet_access_type %Self.ref.loc4_33 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     %.loc4_33: type = converted %Self.ref.loc4_33, %Self.as_type.loc4_33 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     %self.param: @Op.%Self.as_type.loc4_15.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref.loc4_15: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc4_15.2: type = facet_access_type %Self.ref.loc4_15 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref.loc4_15, %Self.as_type.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.%Self.as_type.loc4_15.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: @Op.%Self.as_type.loc4_15.1 (%Self.as_type.b70) = value_param call_param1
// CHECK:STDOUT:     %.loc4_24.1: type = splice_block %.loc4_24.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref.loc4_24: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc4_24: type = facet_access_type %Self.ref.loc4_24 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc4_24.2: type = converted %Self.ref.loc4_24, %Self.as_type.loc4_24 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: @Op.%Self.as_type.loc4_15.1 (%Self.as_type.b70) = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref @Op.%Self.as_type.loc4_15.1 (%Self.as_type.b70) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.%Self.as_type.loc4_15.1 (%Self.as_type.b70) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.ea8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: %I.type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.5be]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %u.patt: @F.%pattern_type (%pattern_type.74afd4.1) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @F.%pattern_type (%pattern_type.74afd4.1) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.74afd4.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.74afd4.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc9_17: %I.type = impl_witness_access constants.%J.lookup_impl_witness.64628e.1, element0 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.5205fd.1)]
// CHECK:STDOUT:     %U.ref.loc9_17: %I.type = name_ref U, %impl.elem0.loc9_17 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.5205fd.1)]
// CHECK:STDOUT:     %U.as_type.loc9_17: type = facet_access_type %U.ref.loc9_17 [symbolic = %as_type (constants.%as_type.f5d8f3.1)]
// CHECK:STDOUT:     %.loc9_17: type = converted %U.ref.loc9_17, %U.as_type.loc9_17 [symbolic = %as_type (constants.%as_type.f5d8f3.1)]
// CHECK:STDOUT:     %u.param: @F.%as_type (%as_type.f5d8f3.1) = value_param call_param0
// CHECK:STDOUT:     %.loc9_11.1: type = splice_block %.loc9_11.2 [symbolic = %as_type (constants.%as_type.f5d8f3.1)] {
// CHECK:STDOUT:       %impl.elem0.loc9_11.2: %I.type = impl_witness_access constants.%J.lookup_impl_witness.64628e.1, element0 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.5205fd.1)]
// CHECK:STDOUT:       %U.ref.loc9_11: %I.type = name_ref U, %impl.elem0.loc9_11.2 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.5205fd.1)]
// CHECK:STDOUT:       %U.as_type.loc9_11: type = facet_access_type %U.ref.loc9_11 [symbolic = %as_type (constants.%as_type.f5d8f3.1)]
// CHECK:STDOUT:       %.loc9_11.2: type = converted %U.ref.loc9_11, %U.as_type.loc9_11 [symbolic = %as_type (constants.%as_type.f5d8f3.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.%as_type (%as_type.f5d8f3.1) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.%as_type (%as_type.f5d8f3.1) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%as_type (%as_type.f5d8f3.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! %I.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc4_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc4_15.1 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.%Self.as_type.loc4_15.1 (%Self.as_type.b70), %b.param: @Op.%Self.as_type.loc4_15.1 (%Self.as_type.b70)) -> @Op.%Self.as_type.loc4_15.1 (%Self.as_type.b70);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.64628e.1)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.1: %I.type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.5205fd.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc9_11.1 [symbolic = %as_type (constants.%as_type.f5d8f3.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %as_type [symbolic = %pattern_type (constants.%pattern_type.74afd4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @F.%as_type (%as_type.f5d8f3.1)) -> @F.%as_type (%as_type.f5d8f3.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericResult(%T.loc12_18.1: %J.type) {
// CHECK:STDOUT:   %T.loc12_18.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_18.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc12_28.2: type = facet_access_type %T.loc12_18.2 [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc12_25: type = pattern_type %T.as_type.loc12_28.2 [symbolic = %pattern_type.loc12_25 (constants.%pattern_type.4ca)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc12_18.2, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.64628e.2)]
// CHECK:STDOUT:   %impl.elem0.loc12_35.2: %I.type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_35.2 (constants.%impl.elem0.5205fd.2)]
// CHECK:STDOUT:   %as_type.loc12_35.2: type = facet_access_type %impl.elem0.loc12_35.2 [symbolic = %as_type.loc12_35.2 (constants.%as_type.f5d8f3.2)]
// CHECK:STDOUT:   %pattern_type.loc12_31: type = pattern_type %as_type.loc12_35.2 [symbolic = %pattern_type.loc12_31 (constants.%pattern_type.74afd4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_39: <witness> = require_complete_type %as_type.loc12_35.2 [symbolic = %require_complete.loc12_39 (constants.%require_complete.b33)]
// CHECK:STDOUT:   %require_complete.loc12_26: <witness> = require_complete_type %T.as_type.loc12_28.2 [symbolic = %require_complete.loc12_26 (constants.%require_complete.2a5)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type.loc12_28.2, (%J.lookup_impl_witness) [symbolic = %J.facet (constants.%J.facet.5210a7.2)]
// CHECK:STDOUT:   %.loc13_11.2: type = fn_type_with_self_type constants.%F.type, %J.facet [symbolic = %.loc13_11.2 (constants.%.d43)]
// CHECK:STDOUT:   %impl.elem1.loc13_11.2: @GenericResult.%.loc13_11.2 (%.d43) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc13_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_11.2: <specific function> = specific_impl_function %impl.elem1.loc13_11.2, @F(%J.facet) [symbolic = %specific_impl_fn.loc13_11.2 (constants.%specific_impl_fn.fe8)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc12_35.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %as_type.loc12_35.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc13_16: type = fn_type_with_self_type constants.%Op.type, %I.facet [symbolic = %.loc13_16 (constants.%.c9a)]
// CHECK:STDOUT:   %impl.elem0.loc13_16.2: @GenericResult.%.loc13_16 (%.c9a) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_16.2 (constants.%impl.elem0.45a)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_16.2: <specific function> = specific_impl_function %impl.elem0.loc13_16.2, @Op(%I.facet) [symbolic = %specific_impl_fn.loc13_16.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericResult.%T.as_type.loc12_28.2 (%T.as_type), %u.param: @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2)) -> @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericResult.%T.as_type.loc12_28.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref.loc13_11: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc13_11: type = facet_access_type constants.%T [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc13_11.1: type = converted constants.%T, %T.as_type.loc13_11 [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem1.loc13_11.1: @GenericResult.%.loc13_11.2 (%.d43) = impl_witness_access constants.%J.lookup_impl_witness.64628e.2, element1 [symbolic = %impl.elem1.loc13_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref.loc13_14: @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = name_ref u, %u
// CHECK:STDOUT:     %.loc13_15.1: %I.type = converted constants.%as_type.f5d8f3.2, constants.%impl.elem0.5205fd.2 [symbolic = %impl.elem0.loc12_35.2 (constants.%impl.elem0.5205fd.2)]
// CHECK:STDOUT:     %.loc13_15.2: %I.type = converted constants.%as_type.f5d8f3.2, constants.%impl.elem0.5205fd.2 [symbolic = %impl.elem0.loc12_35.2 (constants.%impl.elem0.5205fd.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc13_11.1: <specific function> = specific_impl_function %impl.elem1.loc13_11.1, @F(constants.%J.facet.5210a7.2) [symbolic = %specific_impl_fn.loc13_11.2 (constants.%specific_impl_fn.fe8)]
// CHECK:STDOUT:     %.loc13_15.3: init @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = call %specific_impl_fn.loc13_11.1(%u.ref.loc13_14)
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %Op.ref: %I.assoc_type = name_ref Op, @I.%assoc0 [concrete = constants.%assoc0.ea8]
// CHECK:STDOUT:     %impl.elem0.loc13_16.1: @GenericResult.%.loc13_16 (%.c9a) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_16.2 (constants.%impl.elem0.45a)]
// CHECK:STDOUT:     %bound_method.loc13_16: <bound method> = bound_method %.loc13_15.3, %impl.elem0.loc13_16.1
// CHECK:STDOUT:     %T.ref.loc13: %J.type = name_ref T, %T.loc12_18.1 [symbolic = %T.loc12_18.2 (constants.%T)]
// CHECK:STDOUT:     %F.ref.loc13_25: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc13_25: type = facet_access_type %T.ref.loc13 [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc13_25: type = converted %T.ref.loc13, %T.as_type.loc13_25 [symbolic = %T.as_type.loc12_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem1.loc13_25: @GenericResult.%.loc13_11.2 (%.d43) = impl_witness_access constants.%J.lookup_impl_witness.64628e.2, element1 [symbolic = %impl.elem1.loc13_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref.loc13_28: @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = name_ref u, %u
// CHECK:STDOUT:     %.loc13_29.1: %I.type = converted constants.%as_type.f5d8f3.2, constants.%impl.elem0.5205fd.2 [symbolic = %impl.elem0.loc12_35.2 (constants.%impl.elem0.5205fd.2)]
// CHECK:STDOUT:     %.loc13_29.2: %I.type = converted constants.%as_type.f5d8f3.2, constants.%impl.elem0.5205fd.2 [symbolic = %impl.elem0.loc12_35.2 (constants.%impl.elem0.5205fd.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc13_25: <specific function> = specific_impl_function %impl.elem1.loc13_25, @F(constants.%J.facet.5210a7.2) [symbolic = %specific_impl_fn.loc13_11.2 (constants.%specific_impl_fn.fe8)]
// CHECK:STDOUT:     %.loc13_29.3: init @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = call %specific_impl_fn.loc13_25(%u.ref.loc13_28)
// CHECK:STDOUT:     %.loc13_30.1: %I.type = converted constants.%as_type.f5d8f3.2, constants.%impl.elem0.5205fd.2 [symbolic = %impl.elem0.loc12_35.2 (constants.%impl.elem0.5205fd.2)]
// CHECK:STDOUT:     %.loc13_30.2: %I.type = converted constants.%as_type.f5d8f3.2, constants.%impl.elem0.5205fd.2 [symbolic = %impl.elem0.loc12_35.2 (constants.%impl.elem0.5205fd.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc13_16.1: <specific function> = specific_impl_function %impl.elem0.loc13_16.1, @Op(constants.%I.facet) [symbolic = %specific_impl_fn.loc13_16.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:     %bound_method.loc13_30: <bound method> = bound_method %.loc13_15.3, %specific_impl_fn.loc13_16.1
// CHECK:STDOUT:     %.loc13_15.4: @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = value_of_initializer %.loc13_15.3
// CHECK:STDOUT:     %.loc13_15.5: @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = converted %.loc13_15.3, %.loc13_15.4
// CHECK:STDOUT:     %.loc13_29.4: @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = value_of_initializer %.loc13_29.3
// CHECK:STDOUT:     %.loc13_29.5: @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = converted %.loc13_29.3, %.loc13_29.4
// CHECK:STDOUT:     %.loc13_30.3: init @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = call %bound_method.loc13_30(%.loc13_15.5, %.loc13_29.5)
// CHECK:STDOUT:     %.loc13_31.1: @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = value_of_initializer %.loc13_30.3
// CHECK:STDOUT:     %.loc13_31.2: @GenericResult.%as_type.loc12_35.2 (%as_type.f5d8f3.2) = converted %.loc13_30.3, %.loc13_31.1
// CHECK:STDOUT:     return %.loc13_31.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%Self.as_type.b70
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.5210a7.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.64628e.1
// CHECK:STDOUT:   %impl.elem0.loc9_11.1 => constants.%impl.elem0.5205fd.1
// CHECK:STDOUT:   %as_type => constants.%as_type.f5d8f3.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.74afd4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.5210a7.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericResult(constants.%T) {
// CHECK:STDOUT:   %T.loc12_18.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc12_28.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc12_25 => constants.%pattern_type.4ca
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.64628e.2
// CHECK:STDOUT:   %impl.elem0.loc12_35.2 => constants.%impl.elem0.5205fd.2
// CHECK:STDOUT:   %as_type.loc12_35.2 => constants.%as_type.f5d8f3.2
// CHECK:STDOUT:   %pattern_type.loc12_31 => constants.%pattern_type.74afd4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%J.facet.5210a7.2) {
// CHECK:STDOUT:   %Self => constants.%J.facet.5210a7.2
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.64628e.2
// CHECK:STDOUT:   %impl.elem0.loc9_11.1 => constants.%impl.elem0.5205fd.2
// CHECK:STDOUT:   %as_type => constants.%as_type.f5d8f3.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.74afd4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%as_type.f5d8f3.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.74afd4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface_qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.4ca604.1: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.64628e.1: <witness> = lookup_impl_witness %Self, @J [symbolic]
// CHECK:STDOUT:   %J.facet.5210a7.1: %J.type = facet_value %Self.as_type, (%J.lookup_impl_witness.64628e.1) [symbolic]
// CHECK:STDOUT:   %impl.elem0.11fef0.1: type = impl_witness_access %J.lookup_impl_witness.64628e.1, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b984e7.1: type = pattern_type %impl.elem0.11fef0.1 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%G.decl [concrete]
// CHECK:STDOUT:   %T: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.28e: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4ca604.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.64628e.2: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %J.facet.5210a7.2: %J.type = facet_value %T.as_type, (%J.lookup_impl_witness.64628e.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.11fef0.2: type = impl_witness_access %J.lookup_impl_witness.64628e.2, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b984e7.2: type = pattern_type %impl.elem0.11fef0.2 [symbolic]
// CHECK:STDOUT:   %GenericCallInterfaceQualified.type: type = fn_type @GenericCallInterfaceQualified [concrete]
// CHECK:STDOUT:   %GenericCallInterfaceQualified: %GenericCallInterfaceQualified.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.58f: <witness> = require_complete_type %impl.elem0.11fef0.2 [symbolic]
// CHECK:STDOUT:   %require_complete.2a5: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.442: type = fn_type_with_self_type %G.type, %J.facet.5210a7.2 [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.442 = impl_witness_access %J.lookup_impl_witness.64628e.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem1, @G(%J.facet.5210a7.2) [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.28e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericCallInterfaceQualified.%pattern_type.loc8_41 (%pattern_type.4ca604.2) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericCallInterfaceQualified.%pattern_type.loc8_41 (%pattern_type.4ca604.2) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.b984e7.2) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.b984e7.2) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.b984e7.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.b984e7.2) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_58: %J.type = name_ref T, %T.loc8_34.1 [symbolic = %T.loc8_34.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc8_59: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc8_59: type = facet_access_type %T.ref.loc8_58 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8_59: type = converted %T.ref.loc8_58, %T.as_type.loc8_59 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc8_59: type = impl_witness_access constants.%J.lookup_impl_witness.64628e.2, element0 [symbolic = %impl.elem0.loc8_51.2 (constants.%impl.elem0.11fef0.2)]
// CHECK:STDOUT:     %J.ref.loc8: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc8_34.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_34.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericCallInterfaceQualified.%T.as_type.loc8_44.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_44.1: type = splice_block %.loc8_44.2 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc8_44: %J.type = name_ref T, %T.loc8_34.1 [symbolic = %T.loc8_34.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8_44.1: type = facet_access_type %T.ref.loc8_44 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_44.2: type = converted %T.ref.loc8_44, %T.as_type.loc8_44.1 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericCallInterfaceQualified.%T.as_type.loc8_44.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.11fef0.2) = value_param call_param1
// CHECK:STDOUT:     %.loc8_51.1: type = splice_block %impl.elem0.loc8_51.1 [symbolic = %impl.elem0.loc8_51.2 (constants.%impl.elem0.11fef0.2)] {
// CHECK:STDOUT:       %T.ref.loc8_50: %J.type = name_ref T, %T.loc8_34.1 [symbolic = %T.loc8_34.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc8_51: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %T.as_type.loc8_51: type = facet_access_type %T.ref.loc8_50 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_51.2: type = converted %T.ref.loc8_50, %T.as_type.loc8_51 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc8_51.1: type = impl_witness_access constants.%J.lookup_impl_witness.64628e.2, element0 [symbolic = %impl.elem0.loc8_51.2 (constants.%impl.elem0.11fef0.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.11fef0.2) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.11fef0.2) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.11fef0.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %self.patt: @G.%pattern_type.loc5_8 (%pattern_type.4ca604.1) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @G.%pattern_type.loc5_8 (%pattern_type.4ca604.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: @G.%pattern_type.loc5_20 (%pattern_type.b984e7.1) = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: @G.%pattern_type.loc5_20 (%pattern_type.b984e7.1) = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @G.%pattern_type.loc5_20 (%pattern_type.b984e7.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @G.%pattern_type.loc5_20 (%pattern_type.b984e7.1) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access constants.%J.lookup_impl_witness.64628e.1, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:     %U.ref.loc5_29: type = name_ref U, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:     %self.param: @G.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: @G.%impl.elem0.loc5_23.1 (%impl.elem0.11fef0.1) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %U.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11fef0.1)] {
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access constants.%J.lookup_impl_witness.64628e.1, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:       %U.ref.loc5_23: type = name_ref U, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @G.%impl.elem0.loc5_23.1 (%impl.elem0.11fef0.1) = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref @G.%impl.elem0.loc5_23.1 (%impl.elem0.11fef0.1) = out_param call_param2
// CHECK:STDOUT:     %return: ref @G.%impl.elem0.loc5_23.1 (%impl.elem0.11fef0.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%U, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.4ca604.1)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.64628e.1)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.11fef0.1)]
// CHECK:STDOUT:   %pattern_type.loc5_20: type = pattern_type %impl.elem0.loc5_23.1 [symbolic = %pattern_type.loc5_20 (constants.%pattern_type.b984e7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @G.%Self.as_type.loc5_14.1 (%Self.as_type), %v.param: @G.%impl.elem0.loc5_23.1 (%impl.elem0.11fef0.1)) -> @G.%impl.elem0.loc5_23.1 (%impl.elem0.11fef0.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallInterfaceQualified(%T.loc8_34.1: %J.type) {
// CHECK:STDOUT:   %T.loc8_34.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_34.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc8_44.2: type = facet_access_type %T.loc8_34.2 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_41: type = pattern_type %T.as_type.loc8_44.2 [symbolic = %pattern_type.loc8_41 (constants.%pattern_type.4ca604.2)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_34.2, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.64628e.2)]
// CHECK:STDOUT:   %impl.elem0.loc8_51.2: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_51.2 (constants.%impl.elem0.11fef0.2)]
// CHECK:STDOUT:   %pattern_type.loc8_47: type = pattern_type %impl.elem0.loc8_51.2 [symbolic = %pattern_type.loc8_47 (constants.%pattern_type.b984e7.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_55: <witness> = require_complete_type %impl.elem0.loc8_51.2 [symbolic = %require_complete.loc8_55 (constants.%require_complete.58f)]
// CHECK:STDOUT:   %require_complete.loc8_42: <witness> = require_complete_type %T.as_type.loc8_44.2 [symbolic = %require_complete.loc8_42 (constants.%require_complete.2a5)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type.loc8_44.2, (%J.lookup_impl_witness) [symbolic = %J.facet (constants.%J.facet.5210a7.2)]
// CHECK:STDOUT:   %.loc9_11: type = fn_type_with_self_type constants.%G.type, %J.facet [symbolic = %.loc9_11 (constants.%.442)]
// CHECK:STDOUT:   %impl.elem1.loc9_11.2: @GenericCallInterfaceQualified.%.loc9_11 (%.442) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem1.loc9_11.2, @G(%J.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericCallInterfaceQualified.%T.as_type.loc8_44.2 (%T.as_type), %u.param: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.11fef0.2)) -> @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.11fef0.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallInterfaceQualified.%T.as_type.loc8_44.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %J.ref.loc9: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %G.ref: %J.assoc_type = name_ref G, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc9_11.1: @GenericCallInterfaceQualified.%.loc9_11 (%.442) = impl_witness_access constants.%J.lookup_impl_witness.64628e.2, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc9_11: <bound method> = bound_method %t.ref, %impl.elem1.loc9_11.1
// CHECK:STDOUT:     %u.ref: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.11fef0.2) = name_ref u, %u
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem1.loc9_11.1, @G(constants.%J.facet.5210a7.2) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc9_19: <bound method> = bound_method %t.ref, %specific_impl_fn.loc9_11.1
// CHECK:STDOUT:     %.loc9_19: init @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.11fef0.2) = call %bound_method.loc9_19(%t.ref, %u.ref)
// CHECK:STDOUT:     %.loc9_20.1: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.11fef0.2) = value_of_initializer %.loc9_19
// CHECK:STDOUT:     %.loc9_20.2: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.11fef0.2) = converted %.loc9_19, %.loc9_20.1
// CHECK:STDOUT:     return %.loc9_20.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.5210a7.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.4ca604.1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.64628e.1
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.11fef0.1
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.b984e7.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.5210a7.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallInterfaceQualified(constants.%T) {
// CHECK:STDOUT:   %T.loc8_34.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_44.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc8_41 => constants.%pattern_type.4ca604.2
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.64628e.2
// CHECK:STDOUT:   %impl.elem0.loc8_51.2 => constants.%impl.elem0.11fef0.2
// CHECK:STDOUT:   %pattern_type.loc8_47 => constants.%pattern_type.b984e7.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%J.facet.5210a7.2) {
// CHECK:STDOUT:   %Self => constants.%J.facet.5210a7.2
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.4ca604.2
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.64628e.2
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.11fef0.2
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.b984e7.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.646: <witness> = lookup_impl_witness %Self.ccd, @J [symbolic]
// CHECK:STDOUT:   %J.facet.521: %J.type = facet_value %Self.as_type.3df, (%J.lookup_impl_witness.646) [symbolic]
// CHECK:STDOUT:   %impl.elem0.11f: type = impl_witness_access %J.lookup_impl_witness.646, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b98: type = pattern_type %impl.elem0.11f [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.572: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %.Self.968: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.78d: type = facet_access_type %.Self.968 [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.25c: <witness> = lookup_impl_witness %.Self.968, @J [symbolic_self]
// CHECK:STDOUT:   %J.facet.330: %J.type = facet_value %.Self.as_type.78d, (%J.lookup_impl_witness.25c) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.83f: type = impl_witness_access %J.lookup_impl_witness.25c, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.83f = %i32> [concrete]
// CHECK:STDOUT:   %T.c74: %J_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a22: type = pattern_type %J_where.type [concrete]
// CHECK:STDOUT:   %T.as_type.f20: type = facet_access_type %T.c74 [symbolic]
// CHECK:STDOUT:   %pattern_type.b32: type = pattern_type %T.as_type.f20 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %GenericCallFI32.type: type = fn_type @GenericCallFI32 [concrete]
// CHECK:STDOUT:   %GenericCallFI32: %GenericCallFI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.5de: <witness> = require_complete_type %T.as_type.f20 [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.bee: <witness> = lookup_impl_witness %T.c74, @J [symbolic]
// CHECK:STDOUT:   %J.facet.487: %J.type = facet_value %T.as_type.f20, (%J.lookup_impl_witness.bee) [symbolic]
// CHECK:STDOUT:   %.9b0: type = fn_type_with_self_type %F.type, %J.facet.487 [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.9b0 = impl_witness_access %J.lookup_impl_witness.bee, element1 [symbolic]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %specific_impl_fn.30b: <specific function> = specific_impl_function %impl.elem1, @F(%J.facet.487) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.3, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.3, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.a22 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericCallFI32.%pattern_type (%pattern_type.b32) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericCallFI32.%pattern_type (%pattern_type.b32) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc8_51: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_51: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_26.1: type = splice_block %.loc8_26.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:       %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:       %.loc8_32: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:       %impl.elem0.loc8: type = impl_witness_access constants.%J.lookup_impl_witness.25c, element0 [symbolic_self = constants.%impl.elem0.83f]
// CHECK:STDOUT:       %int_32.loc8_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc8_26.2: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc8, %i32.loc8_37
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_20.1: %J_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_20.2 (constants.%T.c74)]
// CHECK:STDOUT:     %t.param: @GenericCallFI32.%T.as_type.loc8_45.2 (%T.as_type.f20) = value_param call_param0
// CHECK:STDOUT:     %.loc8_45.1: type = splice_block %.loc8_45.2 [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type.f20)] {
// CHECK:STDOUT:       %T.ref: %J_where.type = name_ref T, %T.loc8_20.1 [symbolic = %T.loc8_20.2 (constants.%T.c74)]
// CHECK:STDOUT:       %T.as_type.loc8_45.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type.f20)]
// CHECK:STDOUT:       %.loc8_45.2: type = converted %T.ref, %T.as_type.loc8_45.1 [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type.f20)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericCallFI32.%T.as_type.loc8_45.2 (%T.as_type.f20) = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %u.patt: @F.%pattern_type (%pattern_type.b98) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @F.%pattern_type (%pattern_type.b98) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.b98) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.b98) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.646, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     %u.param: @F.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.646, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1.572]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.646)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.b98)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @F.%impl.elem0.loc5_11.1 (%impl.elem0.11f)) -> @F.%impl.elem0.loc5_11.1 (%impl.elem0.11f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallFI32(%T.loc8_20.1: %J_where.type) {
// CHECK:STDOUT:   %T.loc8_20.2: %J_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_20.2 (constants.%T.c74)]
// CHECK:STDOUT:   %T.as_type.loc8_45.2: type = facet_access_type %T.loc8_20.2 [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type.f20)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc8_45.2 [symbolic = %pattern_type (constants.%pattern_type.b32)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc8_45.2 [symbolic = %require_complete (constants.%require_complete.5de)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_20.2, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.bee)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type.loc8_45.2, (%J.lookup_impl_witness) [symbolic = %J.facet (constants.%J.facet.487)]
// CHECK:STDOUT:   %.loc9_11.2: type = fn_type_with_self_type constants.%F.type, %J.facet [symbolic = %.loc9_11.2 (constants.%.9b0)]
// CHECK:STDOUT:   %impl.elem1.loc9_11.2: @GenericCallFI32.%.loc9_11.2 (%.9b0) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem1.loc9_11.2, @F(%J.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.30b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericCallFI32.%T.as_type.loc8_45.2 (%T.as_type.f20)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallFI32.%T.as_type.loc8_45.2 (%T.as_type.f20) = name_ref t, %t
// CHECK:STDOUT:     %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.572]
// CHECK:STDOUT:     %T.as_type.loc9: type = facet_access_type constants.%T.c74 [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type.f20)]
// CHECK:STDOUT:     %.loc9_11.1: type = converted constants.%T.c74, %T.as_type.loc9 [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type.f20)]
// CHECK:STDOUT:     %impl.elem1.loc9_11.1: @GenericCallFI32.%.loc9_11.2 (%.9b0) = impl_witness_access constants.%J.lookup_impl_witness.bee, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem1.loc9_11.1, @F(constants.%J.facet.487) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.30b)]
// CHECK:STDOUT:     %impl.elem0.loc9: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc9_14.1: <bound method> = bound_method %int_2, %impl.elem0.loc9 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc9, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_14.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc9_14.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_14.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_14.2: %i32 = converted %int_2, %.loc9_14.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_15: init %i32 = call %specific_impl_fn.loc9_11.1(%.loc9_14.2)
// CHECK:STDOUT:     %.loc9_16.1: %i32 = value_of_initializer %.loc9_15
// CHECK:STDOUT:     %.loc9_16.2: %i32 = converted %.loc9_15, %.loc9_16.1
// CHECK:STDOUT:     return %.loc9_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.521) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.646
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.11f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.330) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallFI32(constants.%T.c74) {
// CHECK:STDOUT:   %T.loc8_20.2 => constants.%T.c74
// CHECK:STDOUT:   %T.as_type.loc8_45.2 => constants.%T.as_type.f20
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%J.facet.487) {
// CHECK:STDOUT:   %Self => constants.%J.facet.487
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.bee
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_associated_constant_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.646: <witness> = lookup_impl_witness %Self.ccd, @J [symbolic]
// CHECK:STDOUT:   %J.facet.521: %J.type = facet_value %Self.as_type.3df, (%J.lookup_impl_witness.646) [symbolic]
// CHECK:STDOUT:   %impl.elem0.11f: type = impl_witness_access %J.lookup_impl_witness.646, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b98: type = pattern_type %impl.elem0.11f [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.25c: <witness> = lookup_impl_witness %.Self, @J [symbolic_self]
// CHECK:STDOUT:   %J.facet.330: %J.type = facet_value %.Self.as_type, (%J.lookup_impl_witness.25c) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.83f: type = impl_witness_access %J.lookup_impl_witness.25c, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.83f = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type.b842fd.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.b07d12.1: %F.type.b842fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.f43: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type.b842fd.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.b07d12.2: %F.type.b842fd.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %u.patt: @F.1.%pattern_type (%pattern_type.b98) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @F.1.%pattern_type (%pattern_type.b98) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.1.%pattern_type (%pattern_type.b98) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.1.%pattern_type (%pattern_type.b98) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.646, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.646, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.939: %Self.ref as %.loc9_20 {
// CHECK:STDOUT:   %F.decl.loc24_21.1: %F.type.b842fd.1 = fn_decl @F.2 [concrete = constants.%F.b07d12.1] {
// CHECK:STDOUT:     %u.patt: <error> = binding_pattern u [concrete]
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
// CHECK:STDOUT:     %u: <error> = bind_name u, %u.param [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param1
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc24_21.2: %F.type.b842fd.2 = fn_decl @F.3 [concrete = constants.%F.b07d12.2] {
// CHECK:STDOUT:     %u.patt: %pattern_type.7ce = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.7ce = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %u: %i32 = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .U = <poisoned>
// CHECK:STDOUT:   .F = %F.decl.loc24_21.1
// CHECK:STDOUT:   witness = @E.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @impl.939 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc9_26: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.25c, element0 [symbolic_self = constants.%impl.elem0.83f]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc9_20: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @impl.939.%F.decl.loc24_21.2), @impl.939 [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .U = <poisoned>
// CHECK:STDOUT:   extend @impl.939.%.loc9_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.646)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.11f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.b98)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f)) -> @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.11f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%u.param: <error>) -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: <error> = name_ref u, %u [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%u.param: %i32) -> %i32 [thunk @impl.939.%F.decl.loc24_21.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type.b842fd.1 = name_ref F, @impl.939.%F.decl.loc24_21.1 [concrete = constants.%F.b07d12.1]
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u.param
// CHECK:STDOUT:   %return.ref: ref %i32 = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %F.call: init <error> = call %F.ref(<error>)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.521) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.646
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.11f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.330) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.f43) {
// CHECK:STDOUT:   %Self => constants.%J.facet.f43
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_self_period_associated_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J2.type: type = facet_type <@J2> [concrete]
// CHECK:STDOUT:   %Self: %J2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J2.assoc_type: type = assoc_entity_type @J2 [concrete]
// CHECK:STDOUT:   %assoc0: %J2.assoc_type = assoc_entity element0, @J2.%U2 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.969: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %F.type.c41: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.932: %F.type.c41 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J2.assoc_type = assoc_entity element1, @J2.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %J2.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %J2.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @J2 [symbolic_self]
// CHECK:STDOUT:   %J2.facet.e30: %J2.type = facet_value %.Self.as_type, (%J2.lookup_impl_witness) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %J2.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %J2_where.type.9e1: type = facet_type <@J2 where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %J2.impl_witness.6a9: <witness> = impl_witness file.%J2.impl_witness_table.loc22 [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type.f405c5.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.214a71.1: %F.type.f405c5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J2.facet.86f: %J2.type = facet_value %empty_tuple.type, (%J2.impl_witness.6a9) [concrete]
// CHECK:STDOUT:   %F.type.f405c5.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.214a71.2: %F.type.f405c5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J2_where.type.df9: type = facet_type <@J2 where %impl.elem0 = %C2> [concrete]
// CHECK:STDOUT:   %J2.impl_witness.267: <witness> = impl_witness file.%J2.impl_witness_table.loc38 [concrete]
// CHECK:STDOUT:   %pattern_type.838: type = pattern_type %C2 [concrete]
// CHECK:STDOUT:   %F.type.05d6a0.1: type = fn_type @F.4 [concrete]
// CHECK:STDOUT:   %F.bfa759.1: %F.type.05d6a0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J2.facet.c3f: %J2.type = facet_value %C2, (%J2.impl_witness.267) [concrete]
// CHECK:STDOUT:   %F.type.05d6a0.2: type = fn_type @F.5 [concrete]
// CHECK:STDOUT:   %F.bfa759.2: %F.type.05d6a0.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   impl_decl @impl.4e9 [concrete] {} {
// CHECK:STDOUT:     %.loc22_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_7.2: type = converted %.loc22_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J2.ref: type = name_ref J2, file.%J2.decl [concrete = constants.%J2.type]
// CHECK:STDOUT:     %.Self: %J2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U2.ref: %J2.assoc_type = name_ref U2, @U2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc22_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J2.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc22_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc22_28.2: type = converted %.loc22_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc22_15: type = where_expr %.Self [concrete = constants.%J2_where.type.9e1] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc22_28.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J2.impl_witness_table.loc22 = impl_witness_table (%impl_witness_assoc_constant.loc22, @impl.4e9.%F.decl.loc30_33.2), @impl.4e9 [concrete]
// CHECK:STDOUT:   %J2.impl_witness.loc22: <witness> = impl_witness %J2.impl_witness_table.loc22 [concrete = constants.%J2.impl_witness.6a9]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc22: type = impl_witness_assoc_constant constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [concrete = constants.%C2] {} {}
// CHECK:STDOUT:   impl_decl @impl.8b3 [concrete] {} {
// CHECK:STDOUT:     %C2.ref.loc38_6: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %J2.ref: type = name_ref J2, file.%J2.decl [concrete = constants.%J2.type]
// CHECK:STDOUT:     %.Self: %J2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U2.ref: %J2.assoc_type = name_ref U2, @U2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc38_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J2.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C2.ref.loc38_27: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %.loc38_15: type = where_expr %.Self [concrete = constants.%J2_where.type.df9] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C2.ref.loc38_27
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J2.impl_witness_table.loc38 = impl_witness_table (%impl_witness_assoc_constant.loc38, @impl.8b3.%F.decl.loc39_33.2), @impl.8b3 [concrete]
// CHECK:STDOUT:   %J2.impl_witness.loc38: <witness> = impl_witness %J2.impl_witness_table.loc38 [concrete = constants.%J2.impl_witness.267]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc38: type = impl_witness_assoc_constant constants.%C2 [concrete = constants.%C2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J2 {
// CHECK:STDOUT:   %Self: %J2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U2: type = assoc_const_decl @U2 [concrete] {
// CHECK:STDOUT:     %assoc0: %J2.assoc_type = assoc_entity element0, @J2.%U2 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c41 = fn_decl @F.1 [concrete = constants.%F.932] {
// CHECK:STDOUT:     %self.patt: @F.1.%pattern_type (%pattern_type.969) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @F.1.%pattern_type (%pattern_type.969) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %z.patt: <error> = binding_pattern z [concrete]
// CHECK:STDOUT:     %z.param_patt: <error> = value_param_pattern %z.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc19_35: %J2.type = name_ref Self, @J2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc19_14.1: type = splice_block %.loc19_14.2 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc19_14: %J2.type = name_ref Self, @J2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc19_14.2: type = facet_access_type %Self.ref.loc19_14 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc19_14.2: type = converted %Self.ref.loc19_14, %Self.as_type.loc19_14.2 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %z.param: <error> = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc19_23: %J2.type = name_ref Self, @J2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %z: <error> = bind_name z, %z.param [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J2.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U2 = @U2.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U2, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U2(@J2.%Self: %J2.type) {
// CHECK:STDOUT:   assoc_const U2:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4e9: %.loc22_7.2 as %.loc22_15 {
// CHECK:STDOUT:   %F.decl.loc30_33.1: %F.type.f405c5.1 = fn_decl @F.2 [concrete = constants.%F.214a71.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %z.patt: %pattern_type.a96 = binding_pattern z [concrete]
// CHECK:STDOUT:     %z.param_patt: %pattern_type.a96 = value_param_pattern %z.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc30_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc30_31.2: type = converted %.loc30_31.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.4e9.%.loc22_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %z.param: %empty_struct_type = value_param call_param1
// CHECK:STDOUT:     %.loc30_24.1: type = splice_block %.loc30_24.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc30_24.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc30_24.3: type = converted %.loc30_24.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %z: %empty_struct_type = bind_name z, %z.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param2
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc30_33.2: %F.type.f405c5.2 = fn_decl @F.3 [concrete = constants.%F.214a71.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.param_patt: <error> = value_param_pattern <error>, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %.param: <error> = value_param call_param1
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl.loc30_33.1
// CHECK:STDOUT:   witness = file.%J2.impl_witness.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8b3: %C2.ref.loc38_6 as %.loc38_15 {
// CHECK:STDOUT:   %F.decl.loc39_33.1: %F.type.05d6a0.1 = fn_decl @F.4 [concrete = constants.%F.bfa759.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.838 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.838 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %z.patt: %pattern_type.838 = binding_pattern z [concrete]
// CHECK:STDOUT:     %z.param_patt: %pattern_type.838 = value_param_pattern %z.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.838 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.838 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C2.ref.loc39_30: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %self.param: %C2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.8b3.%C2.ref.loc38_6 [concrete = constants.%C2]
// CHECK:STDOUT:     %self: %C2 = bind_name self, %self.param
// CHECK:STDOUT:     %z.param: %C2 = value_param call_param1
// CHECK:STDOUT:     %C2.ref.loc39_23: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %z: %C2 = bind_name z, %z.param
// CHECK:STDOUT:     %return.param: ref %C2 = out_param call_param2
// CHECK:STDOUT:     %return: ref %C2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc39_33.2: %F.type.05d6a0.2 = fn_decl @F.5 [concrete = constants.%F.bfa759.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.838 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.838 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.param_patt: <error> = value_param_pattern <error>, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C2 = value_param call_param0
// CHECK:STDOUT:     %self: %C2 = bind_name self, %self.param
// CHECK:STDOUT:     %.param: <error> = value_param call_param1
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C2 = <poisoned>
// CHECK:STDOUT:   .F = %F.decl.loc39_33.1
// CHECK:STDOUT:   witness = file.%J2.impl_witness.loc38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %.loc35_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc35_11: type = converted %.loc35_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc35_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J2.%Self: %J2.type) {
// CHECK:STDOUT:   %Self: %J2.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc19_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc19_14.1 [symbolic = %pattern_type (constants.%pattern_type.969)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type), %z.param: <error>) -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%self.param: %empty_tuple.type, %z.param: %empty_struct_type) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %z.ref: %empty_struct_type = name_ref z, %z
// CHECK:STDOUT:   return %z.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%self.param: %empty_tuple.type, %.param: <error>) -> <error> [thunk @impl.4e9.%F.decl.loc30_33.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type.f405c5.1 = name_ref F, @impl.4e9.%F.decl.loc30_33.1 [concrete = constants.%F.214a71.1]
// CHECK:STDOUT:   %self.ref: %empty_tuple.type = name_ref self, %self.param
// CHECK:STDOUT:   %.ref: <error> = name_ref <none>, %.param [concrete = <error>]
// CHECK:STDOUT:   %return.ref: ref <error> = name_ref <return slot>, %return.param [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.4(%self.param: %C2, %z.param: %C2) -> %C2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C2 = name_ref self, %self
// CHECK:STDOUT:   return %self.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.5(%self.param: %C2, %.param: <error>) -> <error> [thunk @impl.8b3.%F.decl.loc39_33.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type.05d6a0.1 = name_ref F, @impl.8b3.%F.decl.loc39_33.1 [concrete = constants.%F.bfa759.1]
// CHECK:STDOUT:   %self.ref: %C2 = name_ref self, %self.param
// CHECK:STDOUT:   %.ref: <error> = name_ref <none>, %.param [concrete = <error>]
// CHECK:STDOUT:   %return.ref: ref <error> = name_ref <return slot>, %return.param [concrete = <error>]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   %F.call: init %C2 = call %F.bound(%self.ref, <error>)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.969
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U2(constants.%J2.facet.e30) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J2.facet.86f) {
// CHECK:STDOUT:   %Self => constants.%J2.facet.86f
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J2.facet.c3f) {
// CHECK:STDOUT:   %Self => constants.%J2.facet.c3f
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%C2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.838
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_associated_type_in_signature_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type @K [concrete]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, @K.%V [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.ca1: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %K.lookup_impl_witness.da8: <witness> = lookup_impl_witness %Self, @K [symbolic]
// CHECK:STDOUT:   %K.facet.589: %K.type = facet_value %Self.as_type, (%K.lookup_impl_witness.da8) [symbolic]
// CHECK:STDOUT:   %impl.elem0.75d: type = impl_witness_access %K.lookup_impl_witness.da8, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6c7: type = pattern_type %impl.elem0.75d [symbolic]
// CHECK:STDOUT:   %F.type.cf5: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.06b: %F.type.cf5 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %K.assoc_type = assoc_entity element1, @K.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %K.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %K.lookup_impl_witness.eb2: <witness> = lookup_impl_witness %.Self, @K [symbolic_self]
// CHECK:STDOUT:   %K.facet.b09: %K.type = facet_value %.Self.as_type, (%K.lookup_impl_witness.eb2) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.52c: type = impl_witness_access %K.lookup_impl_witness.eb2, element0 [symbolic_self]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %K_where.type: type = facet_type <@K where %impl.elem0.52c = %struct_type.a> [concrete]
// CHECK:STDOUT:   %K.impl_witness: <witness> = impl_witness file.%K.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.414: type = pattern_type %struct_type.x [concrete]
// CHECK:STDOUT:   %F.type.c5249c.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.5eb4dd.1: %F.type.c5249c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %K.facet.e32: %K.type = facet_value %empty_tuple.type, (%K.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.e85: type = pattern_type %struct_type.a [concrete]
// CHECK:STDOUT:   %F.type.c5249c.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.5eb4dd.2: %F.type.c5249c.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [concrete = constants.%K.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     %.Self: %K.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %K.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %K.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%K.lookup_impl_witness.eb2, element0 [symbolic_self = constants.%impl.elem0.52c]
// CHECK:STDOUT:     %.loc8_31.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_31.2: type = converted %.loc8_31.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%K_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %struct_type.a
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @impl.%F.decl.loc16_45.2), @impl [concrete]
// CHECK:STDOUT:   %K.impl_witness: <witness> = impl_witness %K.impl_witness_table [concrete = constants.%K.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%struct_type.a [concrete = constants.%struct_type.a]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %V: type = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %K.assoc_type = assoc_entity element0, @K.%V [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.cf5 = fn_decl @F.1 [concrete = constants.%F.06b] {
// CHECK:STDOUT:     %self.patt: @F.1.%pattern_type.loc5_8 (%pattern_type.ca1) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @F.1.%pattern_type.loc5_8 (%pattern_type.ca1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: @F.1.%pattern_type.loc5_20 (%pattern_type.6c7) = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: @F.1.%pattern_type.loc5_20 (%pattern_type.6c7) = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @F.1.%pattern_type.loc5_20 (%pattern_type.6c7) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.1.%pattern_type.loc5_20 (%pattern_type.6c7) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access constants.%K.lookup_impl_witness.da8, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.75d)]
// CHECK:STDOUT:     %V.ref.loc5_29: type = name_ref V, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.75d)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %K.type = name_ref Self, @K.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.75d) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %V.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.75d)] {
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access constants.%K.lookup_impl_witness.da8, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.75d)]
// CHECK:STDOUT:       %V.ref.loc5_23: type = name_ref V, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.75d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.75d) = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.75d) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.75d) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %K.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%V, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@K.%Self: %K.type) {
// CHECK:STDOUT:   assoc_const V:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %F.decl.loc16_45.1: %F.type.c5249c.1 = fn_decl @F.2 [concrete = constants.%F.5eb4dd.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: %pattern_type.414 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: %pattern_type.414 = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.414 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.414 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_42.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_42.2: type = converted %.loc16_42.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.x.loc16_43: type = struct_type {.x: %empty_tuple.type} [concrete = constants.%struct_type.x]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: %struct_type.x = value_param call_param1
// CHECK:STDOUT:     %.loc16_30: type = splice_block %struct_type.x.loc16_30 [concrete = constants.%struct_type.x] {
// CHECK:STDOUT:       %.loc16_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc16_29.2: type = converted %.loc16_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.loc16_30: type = struct_type {.x: %empty_tuple.type} [concrete = constants.%struct_type.x]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: %struct_type.x = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref %struct_type.x = out_param call_param2
// CHECK:STDOUT:     %return: ref %struct_type.x = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc16_45.2: %F.type.c5249c.2 = fn_decl @F.3 [concrete = constants.%F.5eb4dd.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: %pattern_type.e85 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: %pattern_type.e85 = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.e85 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e85 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: %struct_type.a = value_param call_param1
// CHECK:STDOUT:     %v: %struct_type.a = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref %struct_type.a = out_param call_param2
// CHECK:STDOUT:     %return: ref %struct_type.a = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl.loc16_45.1
// CHECK:STDOUT:   witness = file.%K.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@K.%Self: %K.type) {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.ca1)]
// CHECK:STDOUT:   %K.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @K [symbolic = %K.lookup_impl_witness (constants.%K.lookup_impl_witness.da8)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %K.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.75d)]
// CHECK:STDOUT:   %pattern_type.loc5_20: type = pattern_type %impl.elem0.loc5_23.1 [symbolic = %pattern_type.loc5_20 (constants.%pattern_type.6c7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type), %v.param: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.75d)) -> @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.75d);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%self.param: %empty_tuple.type, %v.param: %struct_type.x) -> %struct_type.x {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: %struct_type.x = name_ref v, %v
// CHECK:STDOUT:   return %v.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%self.param: %empty_tuple.type, %v.param: %struct_type.a) -> %struct_type.a [thunk @impl.%F.decl.loc16_45.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type.c5249c.1 = name_ref F, @impl.%F.decl.loc16_45.1 [concrete = constants.%F.5eb4dd.1]
// CHECK:STDOUT:   %self.ref: %empty_tuple.type = name_ref self, %self.param
// CHECK:STDOUT:   %v.ref: %struct_type.a = name_ref v, %v.param
// CHECK:STDOUT:   %return.ref: ref %struct_type.a = name_ref <return slot>, %return.param
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%K.facet.589) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.ca1
// CHECK:STDOUT:   %K.lookup_impl_witness => constants.%K.lookup_impl_witness.da8
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.75d
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.6c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%K.facet.b09) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%K.facet.e32) {
// CHECK:STDOUT:   %Self => constants.%K.facet.e32
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %K.lookup_impl_witness => constants.%K.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%struct_type.a
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.e85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_non-type_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type @M [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type.020: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.91c: %G.type.020 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%G.decl [concrete]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %M.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @M [symbolic_self]
// CHECK:STDOUT:   %M.facet.caa: %M.type = facet_value %.Self.as_type, (%M.lookup_impl_witness) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access %M.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.b.347 = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %M_where.type: type = facet_type <@M where %impl.elem0 = %struct> [concrete]
// CHECK:STDOUT:   %M.impl_witness: <witness> = impl_witness file.%M.impl_witness_table [concrete]
// CHECK:STDOUT:   %G.type.8d1: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.bb3: %G.type.8d1 = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.56b: %M.type = facet_value %empty_tuple.type, (%M.impl_witness) [concrete]
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.347 = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_32: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_33.1: %struct_type.b.347 = struct_literal (%.loc8_32)
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_33.2: %empty_struct_type = converted %.loc8_32, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %struct: %struct_type.b.347 = struct_value (%.loc8_33.2) [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc8_33.3: %struct_type.b.347 = converted %.loc8_33.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%M_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_33.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @impl.%G.decl), @impl [concrete]
// CHECK:STDOUT:   %M.impl_witness: <witness> = impl_witness %M.impl_witness_table [concrete = constants.%M.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %struct_type.b.347 = impl_witness_assoc_constant constants.%struct [concrete = constants.%struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Z: %struct_type.b.347 = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.020 = fn_decl @G.1 [concrete = constants.%G.91c] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, %G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = @Z.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Z, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Z(@M.%Self: %M.type) {
// CHECK:STDOUT:   assoc_const Z:! %struct_type.b.347;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %G.decl: %G.type.8d1 = fn_decl @G.2 [concrete = constants.%G.bb3] {
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
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = file.%M.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@M.%Self: %M.type) {
// CHECK:STDOUT:   fn() -> %empty_struct_type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value constants.%empty_tuple.type, (constants.%M.impl_witness) [concrete = constants.%M.facet.56b]
// CHECK:STDOUT:   %.loc10_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.56b]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_18 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc10_23: type = converted %.loc10_18, %as_type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access constants.%M.impl_witness, element0 [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_25: %empty_struct_type = struct_access %impl.elem0, element0 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc10_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.caa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%M.facet.56b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.56b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_as_uses_correct_rewrite_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.b.86f: type = struct_type {.b: type} [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type @M [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete]
// CHECK:STDOUT:   %G.type.020: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.91c: %G.type.020 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%G.decl [concrete]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %M.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @M [symbolic_self]
// CHECK:STDOUT:   %M.facet.caa: %M.type = facet_value %.Self.as_type, (%M.lookup_impl_witness) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.86f = impl_witness_access %M.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.0f3: %struct_type.b.86f = struct_value (%empty_struct_type) [concrete]
// CHECK:STDOUT:   %M_where.type.d04: type = facet_type <@M where %impl.elem0 = %struct.0f3> [concrete]
// CHECK:STDOUT:   %M.impl_witness.baa: <witness> = impl_witness file.%M.impl_witness_table.loc10 [concrete]
// CHECK:STDOUT:   %G.type.d92: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.eac: %G.type.d92 = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.1dc: %M.type = facet_value %C.7a7, (%M.impl_witness.baa) [concrete]
// CHECK:STDOUT:   %C.3a0: type = class_type @C, @C(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %struct_type.b.161: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct.c94: %struct_type.b.86f = struct_value (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %M_where.type.eda: type = facet_type <@M where %impl.elem0 = %struct.c94> [concrete]
// CHECK:STDOUT:   %M.impl_witness.feb: <witness> = impl_witness file.%M.impl_witness_table.loc16 [concrete]
// CHECK:STDOUT:   %G.type.b29: type = fn_type @G.3 [concrete]
// CHECK:STDOUT:   %G.f3b: %G.type.b29 = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.80d: %M.type = facet_value %C.3a0, (%M.impl_witness.feb) [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.8b4 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc10_9: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_10: type = converted %.loc10_9, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.7a7]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_23: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.86f = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc10_35: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_36.1: %struct_type.b.347 = struct_literal (%.loc10_35)
// CHECK:STDOUT:     %.loc10_36.2: type = converted %.loc10_35, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct: %struct_type.b.86f = struct_value (%.loc10_36.2) [concrete = constants.%struct.0f3]
// CHECK:STDOUT:     %.loc10_36.3: %struct_type.b.86f = converted %.loc10_36.1, %struct [concrete = constants.%struct.0f3]
// CHECK:STDOUT:     %.loc10_17: type = where_expr %.Self [concrete = constants.%M_where.type.d04] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc10_36.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.impl_witness_table.loc10 = impl_witness_table (%impl_witness_assoc_constant.loc10, @impl.8b4.%G.decl), @impl.8b4 [concrete]
// CHECK:STDOUT:   %M.impl_witness.loc10: <witness> = impl_witness %M.impl_witness_table.loc10 [concrete = constants.%M.impl_witness.baa]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc10: %struct_type.b.86f = impl_witness_assoc_constant constants.%struct.0f3 [concrete = constants.%struct.0f3]
// CHECK:STDOUT:   impl_decl @impl.d97 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc16_9: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_10: type = converted %.loc16_9, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_tuple.type) [concrete = constants.%C.3a0]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc16_23: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.86f = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc16_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_36.1: %struct_type.b.161 = struct_literal (%.loc16_35)
// CHECK:STDOUT:     %.loc16_36.2: type = converted %.loc16_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct: %struct_type.b.86f = struct_value (%.loc16_36.2) [concrete = constants.%struct.c94]
// CHECK:STDOUT:     %.loc16_36.3: %struct_type.b.86f = converted %.loc16_36.1, %struct [concrete = constants.%struct.c94]
// CHECK:STDOUT:     %.loc16_17: type = where_expr %.Self [concrete = constants.%M_where.type.eda] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc16_36.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.impl_witness_table.loc16 = impl_witness_table (%impl_witness_assoc_constant.loc16, @impl.d97.%G.decl), @impl.d97 [concrete]
// CHECK:STDOUT:   %M.impl_witness.loc16: <witness> = impl_witness %M.impl_witness_table.loc16 [concrete = constants.%M.impl_witness.feb]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc16: %struct_type.b.86f = impl_witness_assoc_constant constants.%struct.c94 [concrete = constants.%struct.c94]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Z: %struct_type.b.86f = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.020 = fn_decl @G.1 [concrete = constants.%G.91c] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, %G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = @Z.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Z, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Z(@M.%Self: %M.type) {
// CHECK:STDOUT:   assoc_const Z:! %struct_type.b.86f;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8b4: %C as %.loc10_17 {
// CHECK:STDOUT:   %G.decl: %G.type.d92 = fn_decl @G.2 [concrete = constants.%G.eac] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = file.%M.impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.d97: %C as %.loc16_17 {
// CHECK:STDOUT:   %G.decl: %G.type.b29 = fn_decl @G.3 [concrete = constants.%G.f3b] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = file.%M.impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc3_9.1: type) {
// CHECK:STDOUT:   %T.loc3_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_9.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@M.%Self: %M.type) {
// CHECK:STDOUT:   fn() -> type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.8b4.%C [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value constants.%C.7a7, (constants.%M.impl_witness.baa) [concrete = constants.%M.facet.1dc]
// CHECK:STDOUT:   %.loc12_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.1dc]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc12_18 [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %.loc12_23: type = converted %.loc12_18, %as_type [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.86f = impl_witness_access constants.%M.impl_witness.baa, element0 [concrete = constants.%struct.0f3]
// CHECK:STDOUT:   %.loc12_25: type = struct_access %impl.elem0, element0 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   return %.loc12_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.3() -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.d97.%C [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value constants.%C.3a0, (constants.%M.impl_witness.feb) [concrete = constants.%M.facet.80d]
// CHECK:STDOUT:   %.loc18_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.80d]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc18_18 [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %.loc18_23: type = converted %.loc18_18, %as_type [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.86f = impl_witness_access constants.%M.impl_witness.feb, element0 [concrete = constants.%struct.c94]
// CHECK:STDOUT:   %.loc18_25: type = struct_access %impl.elem0, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   return %.loc18_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.caa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%M.facet.1dc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.1dc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%M.facet.80d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.80d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_int_in_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.8268: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.995: %I.assoc_type = assoc_entity element0, @I.%N [concrete]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.8268 [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type.b70 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.lookup_impl_witness.5d6: <witness> = lookup_impl_witness %Self.8268, @I [symbolic]
// CHECK:STDOUT:   %I.facet.93d: %I.type = facet_value %Self.as_type.b70, (%I.lookup_impl_witness.5d6) [symbolic]
// CHECK:STDOUT:   %impl.elem0.fab: %i32 = impl_witness_access %I.lookup_impl_witness.5d6, element0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.3, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.708: type = fn_type @Convert.4, @impl.971(%From) [symbolic]
// CHECK:STDOUT:   %Convert.c68: %Convert.type.708 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a11: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1d9, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.4, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f49: %ImplicitAs.type.2fd = facet_value %i32, (%ImplicitAs.impl_witness.a11) [concrete]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.f49 [concrete]
// CHECK:STDOUT:   %Convert.bound.44d: <bound method> = bound_method %impl.elem0.fab, %Convert.960 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.8a8: <specific function> = specific_function %Convert.960, @Convert.4(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.81c: <bound method> = bound_method %impl.elem0.fab, %Convert.specific_fn.8a8 [symbolic]
// CHECK:STDOUT:   %int.convert_checked.acf: init Core.IntLiteral = call %bound_method.81c(%impl.elem0.fab) [symbolic]
// CHECK:STDOUT:   %array_type.377: type = array_type %int.convert_checked.acf, bool [symbolic]
// CHECK:STDOUT:   %pattern_type.bc6: type = pattern_type %array_type.377 [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.dfe: %I.assoc_type = assoc_entity element1, @I.%F.decl [concrete]
// CHECK:STDOUT:   %.Self.258: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.541: type = facet_access_type %.Self.258 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.5f5: <witness> = lookup_impl_witness %.Self.258, @I [symbolic_self]
// CHECK:STDOUT:   %I.facet.74c: %I.type = facet_value %.Self.as_type.541, (%I.lookup_impl_witness.5f5) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d4d: %i32 = impl_witness_access %I.lookup_impl_witness.5f5, element0 [symbolic_self]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.3, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b6f: <specific function> = specific_function %Convert.956, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn.b6f [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.d4d = %int_2.ef8> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %array_type.c9b: type = array_type %int_2.ecc, bool [concrete]
// CHECK:STDOUT:   %pattern_type.5d5: type = pattern_type %array_type.c9b [concrete]
// CHECK:STDOUT:   %F.type.2c7: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.0b4: %F.type.2c7 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.85d: %I.type = facet_value %empty_tuple.type, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %Convert.bound.fbf: <bound method> = bound_method %int_2.ef8, %Convert.960 [concrete]
// CHECK:STDOUT:   %bound_method.992: <bound method> = bound_method %int_2.ef8, %Convert.specific_fn.8a8 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (bool, bool) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %array: %array_type.c9b = tuple_value (%true, %false) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.85c: @impl.971.%Convert.type (%Convert.type.708) = import_ref Core//prelude/types/int, loc23_44, loaded [symbolic = @impl.971.%Convert (constants.%Convert.c68)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1d9 = impl_witness_table (%Core.import_ref.85c), @impl.971 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.9a6 [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:     %N.ref: %I.assoc_type = name_ref N, @N.%assoc0 [concrete = constants.%assoc0.995]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:     %impl.elem0.loc8_20: %i32 = impl_witness_access constants.%I.lookup_impl_witness.5f5, element0 [symbolic_self = constants.%impl.elem0.d4d]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc8_25: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc8_25.1: <bound method> = bound_method %int_2, %impl.elem0.loc8_25 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc8_25, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:     %bound_method.loc8_25.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc8_25.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_25.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_25.2: %i32 = converted %int_2, %.loc8_25.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8_20, %.loc8_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @impl.9a6.%F.decl), @impl.9a6 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %i32 = impl_witness_assoc_constant constants.%int_2.ef8 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.8268]
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl @N [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%N [concrete = constants.%assoc0.995]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %self.patt: @F.1.%pattern_type.loc5_8 (%pattern_type.6de) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @F.1.%pattern_type.loc5_8 (%pattern_type.6de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.1.%pattern_type.loc5_22 (%pattern_type.bc6) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.1.%pattern_type.loc5_22 (%pattern_type.bc6) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %impl.elem0.loc5_37.2: %i32 = impl_witness_access constants.%I.lookup_impl_witness.5d6, element0 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.fab)]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %impl.elem0.loc5_37.2 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.fab)]
// CHECK:STDOUT:     %.loc5_31.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc5_31.2: type = converted %bool.make_type, %.loc5_31.1 [concrete = bool]
// CHECK:STDOUT:     %impl.elem0.loc5_37.3: %.0ea = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:     %bound_method.loc5_37.2: <bound method> = bound_method %N.ref, %impl.elem0.loc5_37.3 [symbolic = %Convert.bound (constants.%Convert.bound.44d)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc5_37.3, @Convert.4(constants.%int_32) [concrete = constants.%Convert.specific_fn.8a8]
// CHECK:STDOUT:     %bound_method.loc5_37.3: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc5_37.1 (constants.%bound_method.81c)]
// CHECK:STDOUT:     %int.convert_checked.loc5_37.2: init Core.IntLiteral = call %bound_method.loc5_37.3(%N.ref) [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked.acf)]
// CHECK:STDOUT:     %.loc5_37.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc5_37.2 [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked.acf)]
// CHECK:STDOUT:     %.loc5_37.2: Core.IntLiteral = converted %N.ref, %.loc5_37.1 [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked.acf)]
// CHECK:STDOUT:     %array_type.loc5_38.2: type = array_type %.loc5_37.2, %.loc5_31.2 [symbolic = %array_type.loc5_38.1 (constants.%array_type.377)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.8268)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref @F.1.%array_type.loc5_38.1 (%array_type.377) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%array_type.loc5_38.1 (%array_type.377) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1.dfe]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = @N.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%N, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @N(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const N:! %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9a6: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %F.decl: %F.type.2c7 = fn_decl @F.2 [concrete = constants.%F.0b4] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5d5 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5d5 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc9_31.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc9_31.2: type = converted %bool.make_type, %.loc9_31.1 [concrete = bool]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, %.loc9_31.2 [concrete = constants.%array_type.c9b]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.9a6.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %array_type.c9b = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.c9b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.8268)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.6de)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.5d6)]
// CHECK:STDOUT:   %impl.elem0.loc5_37.1: %i32 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.fab)]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %impl.elem0.loc5_37.1, constants.%Convert.960 [symbolic = %Convert.bound (constants.%Convert.bound.44d)]
// CHECK:STDOUT:   %bound_method.loc5_37.1: <bound method> = bound_method %impl.elem0.loc5_37.1, constants.%Convert.specific_fn.8a8 [symbolic = %bound_method.loc5_37.1 (constants.%bound_method.81c)]
// CHECK:STDOUT:   %int.convert_checked.loc5_37.1: init Core.IntLiteral = call %bound_method.loc5_37.1(%impl.elem0.loc5_37.1) [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked.acf)]
// CHECK:STDOUT:   %array_type.loc5_38.1: type = array_type %int.convert_checked.loc5_37.1, bool [symbolic = %array_type.loc5_38.1 (constants.%array_type.377)]
// CHECK:STDOUT:   %pattern_type.loc5_22: type = pattern_type %array_type.loc5_38.1 [symbolic = %pattern_type.loc5_22 (constants.%pattern_type.bc6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70)) -> @F.1.%array_type.loc5_38.1 (%array_type.377);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%self.param: %empty_tuple.type) -> %return.param: %array_type.c9b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_61.1: %tuple.type = tuple_literal (%true, %false)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_61.2: ref bool = array_index %return, %int_0
// CHECK:STDOUT:   %.loc9_61.3: init bool = initialize_from %true to %.loc9_61.2 [concrete = constants.%true]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_61.4: ref bool = array_index %return, %int_1
// CHECK:STDOUT:   %.loc9_61.5: init bool = initialize_from %false to %.loc9_61.4 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_61.6: init %array_type.c9b = array_init (%.loc9_61.3, %.loc9_61.5) to %return [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_62: init %array_type.c9b = converted %.loc9_61.1, %.loc9_61.6 [concrete = constants.%array]
// CHECK:STDOUT:   return %.loc9_62 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%Self.8268) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%I.facet.93d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.8268) {
// CHECK:STDOUT:   %Self => constants.%Self.8268
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type.b70
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.6de
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.5d6
// CHECK:STDOUT:   %impl.elem0.loc5_37.1 => constants.%impl.elem0.fab
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound.44d
// CHECK:STDOUT:   %bound_method.loc5_37.1 => constants.%bound_method.81c
// CHECK:STDOUT:   %int.convert_checked.loc5_37.1 => constants.%int.convert_checked.acf
// CHECK:STDOUT:   %array_type.loc5_38.1 => constants.%array_type.377
// CHECK:STDOUT:   %pattern_type.loc5_22 => constants.%pattern_type.bc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%I.facet.74c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.85d) {
// CHECK:STDOUT:   %Self => constants.%I.facet.85d
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_37.1 => constants.%int_2.ef8
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound.fbf
// CHECK:STDOUT:   %bound_method.loc5_37.1 => constants.%bound_method.992
// CHECK:STDOUT:   %int.convert_checked.loc5_37.1 => constants.%int_2.ecc
// CHECK:STDOUT:   %array_type.loc5_38.1 => constants.%array_type.c9b
// CHECK:STDOUT:   %pattern_type.loc5_22 => constants.%pattern_type.5d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- symbolic_associated_type_in_concrete_context.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.Self: %Z.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @Z [symbolic_self]
// CHECK:STDOUT:   %Z.facet.2a2: %Z.type = facet_value %.Self.as_type, (%Z.lookup_impl_witness) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.585: type = facet_type <@Z where %impl.elem0 = %C.f2e> [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type.585 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.8bc: <witness> = impl_witness file.%Z.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.131: type = class_type @C, @C(%D) [concrete]
// CHECK:STDOUT:   %Z_where.type.1a6: type = facet_type <@Z where %impl.elem0 = %C.131> [concrete]
// CHECK:STDOUT:   %complete_type.00b: <witness> = complete_type_witness %Z_where.type.1a6 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.19e: <witness> = impl_witness file.%Z.impl_witness_table, @impl(%D) [concrete]
// CHECK:STDOUT:   %Z.facet.f2a: %Z.type = facet_value %D, (%Z.impl_witness.19e) [concrete]
// CHECK:STDOUT:   %pattern_type.7f8: type = pattern_type %C.131 [concrete]
// CHECK:STDOUT:   %C.val: %C.131 = struct_value () [concrete]
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
// CHECK:STDOUT:     %T.loc6_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_24: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self: %Z.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc9_37: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref.loc9_44: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:     %C.loc9_45.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc9_45.2 (constants.%C.f2e)]
// CHECK:STDOUT:     %.loc9_31: type = where_expr %.Self [symbolic = %Z_where.type (constants.%Z_where.type.585)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.loc9_45.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table, @impl(constants.%T) [symbolic = @impl.%Z.impl_witness (constants.%Z.impl_witness.8bc)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C.f2e [symbolic = @impl.%C.loc9_45.2 (constants.%C.f2e)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%X [concrete = constants.%assoc0]
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
// CHECK:STDOUT: generic impl @impl(%T.loc9_14.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:   %C.loc9_45.2: type = class_type @C, @C(%T.loc9_14.2) [symbolic = %C.loc9_45.2 (constants.%C.f2e)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z where constants.%impl.elem0 = %C.loc9_45.2> [symbolic = %Z_where.type (constants.%Z_where.type.585)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table, @impl(%T.loc9_14.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.8bc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc9_24 as %.loc9_31 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc6_9.1: type) {
// CHECK:STDOUT:   %T.loc6_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7f8 = binding_pattern a [concrete]
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
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %Z.facet: %Z.type = facet_value constants.%D, (constants.%Z.impl_witness.19e) [concrete = constants.%Z.facet.f2a]
// CHECK:STDOUT:     %.loc12_11.2: %Z.type = converted %D.ref.loc12_10, %Z.facet [concrete = constants.%Z.facet.f2a]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.impl_witness.19e, element0 [concrete = constants.%C.131]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_23.2: %C.131 = bind_value %.loc12_23.1
// CHECK:STDOUT:   %a: %C.131 = bind_name a, %.loc12_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc6_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Z.facet.2a2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T
// CHECK:STDOUT:   %C.loc9_45.2 => constants.%C.f2e
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.585
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.8bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%D) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%D
// CHECK:STDOUT:   %C.loc9_45.2 => constants.%C.131
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.1a6
// CHECK:STDOUT:   %require_complete => constants.%complete_type.00b
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.19e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%D) {
// CHECK:STDOUT:   %T.loc6_9.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Z.facet.f2a) {}
// CHECK:STDOUT:
