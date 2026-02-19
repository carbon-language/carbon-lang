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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/use_assoc_entity.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/use_assoc_entity.carbon

// --- associated_type_in_method_signature.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F[self: Self](u: U) -> U;
}

impl () as J where .U = i32 {
  fn F[unused self: Self](u: i32) -> i32 { return -u; }
}

fn CallMethod(x: ()) -> i32 {
  return x.(J.F)(40);
}

class C {
  // This allows the type to be copyable so it can be returned.
  adapt {};
}

impl C as J where .U = C {
  fn F[self: Self](unused u: C) -> C { return self; }
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
    fn G[unused self: Self](v: i32) -> i32 {
      return -v;
    }
  }
}

fn CallBoth(e: E) {
  let unused e1: i32 = e.F(2);
  let unused e2: i32 = e.G(3);
  let unused e3: i32 = E.F(4);
  let unused e4: i32 = e.(E.G)(5);
  let unused e5: i32 = e.(J.G)(6);
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
    // TODO: U should be in scope and usable as soon as we finish the
    // declaration / enter the definition.
    //
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

// --- fail_todo_call_extend_fn.carbon
library "[[@TEST_NAME]]";

interface J {
  fn F() {}
}

class E {
  extend impl as J {
    fn F() {}
  }

  // CHECK:STDERR: fail_todo_call_extend_fn.carbon:[[@LINE+4]]:12: error: value of type `<associated entity in J>` is not callable [CallToNonCallable]
  // CHECK:STDERR:   fn G() { F(); }
  // CHECK:STDERR:            ^~~
  // CHECK:STDERR:
  fn G() { F(); }
}

// --- call_extend_method.carbon
library "[[@TEST_NAME]]";

interface J {
  fn F[unused self: Self]() {}
}

class E {
  extend impl as J {
    fn F[unused self: Self]() {}
  }

  fn G[self: Self]() { self.F(); }
}

// --- fail_todo_use_extend_constant.carbon
library "[[@TEST_NAME]]";

interface J {
  let X:! type;
}

class E {
  extend impl as J where .X = () {}

  // TODO: `X` should be in scope and directly usable.
  // CHECK:STDERR: fail_todo_use_extend_constant.carbon:[[@LINE+7]]:13: error: cannot implicitly convert non-type value of type `<associated entity in J>` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   fn G() -> X { return (); }
  // CHECK:STDERR:             ^
  // CHECK:STDERR: fail_todo_use_extend_constant.carbon:[[@LINE+4]]:13: note: type `<associated entity in J>` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn G() -> X { return (); }
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn G() -> X { return (); }
}

// --- fail_todo_self_period_associated_type.carbon
library "[[@TEST_NAME]]";

interface J2 {
  let U2:! type;
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE+14]]:23: error: member access into facet of incomplete type `J2` [IncompleteTypeInMemberAccessOfFacet]
  // CHECK:STDERR:   fn F[self: Self](z: Self.U2) -> Self.U2;
  // CHECK:STDERR:                       ^~~~
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE-5]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface J2 {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE+7]]:35: error: member access into facet of incomplete type `J2` [IncompleteTypeInMemberAccessOfFacet]
  // CHECK:STDERR:   fn F[self: Self](z: Self.U2) -> Self.U2;
  // CHECK:STDERR:                                   ^~~~
  // CHECK:STDERR: fail_todo_self_period_associated_type.carbon:[[@LINE-12]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface J2 {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F[self: Self](z: Self.U2) -> Self.U2;
}

impl () as J2 where .U2 = {} {
  fn F[unused self: Self](z: {}) -> {} { return z; }
}

class C2 {
  // This allows the type to be copyable so it can be returned.
  adapt {};
}

impl C2 as J2 where .U2 = C2 {
  fn F[self: Self](unused z: C2) -> C2 { return self; }
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
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE+14]]:27: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[unused self: Self](v: {.x: ()}) -> {.x: ()} { return v; }
  // CHECK:STDERR:                           ^~~~~~~~~~~
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE-7]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F[self: Self](v: V) -> V;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE+7]]:3: error: cannot convert from struct type `{.x: ()}` to `{.a: ()}`: missing field `a` in source type [StructInitMissingFieldInConversion]
  // CHECK:STDERR:   fn F[unused self: Self](v: {.x: ()}) -> {.x: ()} { return v; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE-14]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F[self: Self](v: V) -> V;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F[unused self: Self](v: {.x: ()}) -> {.x: ()} { return v; }
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
  fn F[unused self: Self]() -> array(bool, 2) { return (true, false); }
}

// --- symbolic_associated_type_in_concrete_context.carbon

interface Z {
  let X:! type;
}

class C(T:! type) {}
class D {}

impl forall [T:! type] T as Z where .X = C(T) {}

fn F() {
  let unused a: D.(Z.X) = {} as C(D);
}

// CHECK:STDOUT: --- associated_type_in_method_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.ebd: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.ed8: type = symbolic_binding_type Self, 0, %Self.8a1 [symbolic]
// CHECK:STDOUT:   %pattern_type.832: type = pattern_type %Self.binding.as_type.ed8 [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.89e: <witness> = lookup_impl_witness %Self.8a1, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.335: type = impl_witness_access %J.lookup_impl_witness.89e, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c1a: type = pattern_type %impl.elem0.335 [symbolic]
// CHECK:STDOUT:   %.147: Core.Form = init_form %impl.elem0.335, call_param2 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.adb: type = fn_type @J.WithSelf.F, @J(%Self.8a1) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.569: %J.WithSelf.F.type.adb = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.eeb: %J.assoc_type = assoc_entity element1, @J.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.2fa: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.type.e19: type = fn_type @J.WithSelf.F, @J(%.Self.2fa) [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.659: %J.WithSelf.F.type.e19 = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.2fa [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.46a: <witness> = lookup_impl_witness %.Self.2fa, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a58: type = impl_witness_access %J.lookup_impl_witness.46a, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type.928: type = facet_type <@J where %impl.elem0.a58 = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness.5bf: <witness> = impl_witness @empty_tuple.type.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.8ef: Core.Form = init_form %i32, call_param2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J.impl.F.type: type = fn_type @empty_tuple.type.as.J.impl.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J.impl.F: %empty_tuple.type.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.a51: %J.type = facet_value %empty_tuple.type, (%J.impl_witness.5bf) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.fd2: type = fn_type @J.WithSelf.F, @J(%J.facet.a51) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.497: %J.WithSelf.F.type.fd2 = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.5e6: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.7f5: %Int.as.Negate.impl.Op.type.5e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Negate.impl_witness.11c: <witness> = impl_witness imports.%Negate.impl_witness_table, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.e60: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.e7f: %Int.as.Negate.impl.Op.type.e60 = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %i32, (%Negate.impl_witness.11c) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.140: type = fn_type @Negate.WithSelf.Op, @Negate(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.5dd: type = fn_type_with_self_type %Negate.WithSelf.Op.type.140, %Negate.facet [concrete]
// CHECK:STDOUT:   %.e54: Core.Form = init_form %i32, call_param1 [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Negate.impl.Op.e7f, @Int.as.Negate.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %CallMethod.type: type = fn_type @CallMethod [concrete]
// CHECK:STDOUT:   %CallMethod: %CallMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b39: type = fn_type_with_self_type %J.WithSelf.F.type.fd2, %J.facet.a51 [concrete]
// CHECK:STDOUT:   %int_40.f80: Core.IntLiteral = int_value 40 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_40.f80, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_40.f80, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_40.518: %i32 = int_value 40 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J_where.type.d0e: type = facet_type <@J where %impl.elem0.a58 = %C> [concrete]
// CHECK:STDOUT:   %J.impl_witness.00b: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.92f: Core.Form = init_form %C, call_param2 [concrete]
// CHECK:STDOUT:   %C.as.J.impl.F.type: type = fn_type @C.as.J.impl.F [concrete]
// CHECK:STDOUT:   %C.as.J.impl.F: %C.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.05d: %J.type = facet_value %C, (%J.impl_witness.00b) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.24e: type = fn_type @J.WithSelf.F, @J(%J.facet.05d) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.bd5: %J.WithSelf.F.type.24e = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.475 = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d40: @Int.as.Negate.impl.%Int.as.Negate.impl.Op.type (%Int.as.Negate.impl.Op.type.5e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Negate.impl.%Int.as.Negate.impl.Op (constants.%Int.as.Negate.impl.Op.7f5)]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.475, %Core.import_ref.d40), @Int.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.2fa]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.2fa]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.46a, element0 [symbolic_self = constants.%impl.elem0.a58]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%J_where.type.928] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallMethod.decl: %CallMethod.type = fn_decl @CallMethod [concrete = constants.%CallMethod] {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12_25: Core.Form = init_form %i32, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc12_19.1: type = splice_block %.loc12_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc12_19.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
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
// CHECK:STDOUT:     %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.2fa]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.2fa]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc21_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.46a, element0 [symbolic_self = constants.%impl.elem0.a58]
// CHECK:STDOUT:     %C.ref.loc21_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc21_13: type = where_expr %.Self [concrete = constants.%J_where.type.d0e] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref.loc21_24
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self.loc3: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.%J.WithSelf.F.type (%J.WithSelf.F.type.adb) = fn_decl @J.WithSelf.F [symbolic = @J.%J.WithSelf.F (constants.%J.WithSelf.F.569)] {
// CHECK:STDOUT:     %self.patt: @J.WithSelf.F.%pattern_type.loc5_8 (%pattern_type.832) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @J.WithSelf.F.%pattern_type.loc5_8 (%pattern_type.832) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @J.WithSelf.F.%pattern_type.loc5_20 (%pattern_type.c1a) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.WithSelf.F.%pattern_type.loc5_20 (%pattern_type.c1a) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @J.WithSelf.F.%pattern_type.loc5_20 (%pattern_type.c1a) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.WithSelf.F.%pattern_type.loc5_20 (%pattern_type.c1a) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %U.ref.loc5_29: type = name_ref U, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %.loc5_29.2: Core.Form = init_form %U.ref.loc5_29, call_param2 [symbolic = %.loc5_29.1 (constants.%.147)]
// CHECK:STDOUT:     %self.param: @J.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.ed8) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ed8)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self.loc3 [symbolic = %Self (constants.%Self.8a1)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ed8)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ed8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.ed8) = value_binding self, %self.param
// CHECK:STDOUT:     %u.param: @J.WithSelf.F.%impl.elem0.loc5_23.1 (%impl.elem0.335) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %U.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)] {
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:       %U.ref.loc5_23: type = name_ref U, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.WithSelf.F.%impl.elem0.loc5_23.1 (%impl.elem0.335) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.WithSelf.F.%impl.elem0.loc5_23.1 (%impl.elem0.335) = out_param call_param2
// CHECK:STDOUT:     %return: ref @J.WithSelf.F.%impl.elem0.loc5_23.1 (%impl.elem0.335) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.WithSelf.F.decl [concrete = constants.%assoc1.eeb]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
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
// CHECK:STDOUT:     %int_32.loc9_38: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_38: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc9_38: Core.Form = init_form %i32.loc9_38, call_param2 [concrete = constants.%.8ef]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.J.impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc9_30: type = splice_block %i32.loc9_30 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc9_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9_30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %empty_tuple.type.as.J.impl.F.decl), @empty_tuple.type.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.5bf]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.J.impl.F.decl
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref.loc21_6 as %.loc21_13 {
// CHECK:STDOUT:   %C.as.J.impl.F.decl: %C.as.J.impl.F.type = fn_decl @C.as.J.impl.F [concrete = constants.%C.as.J.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.7c7 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c7 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: %pattern_type.7c7 = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.7c7 = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7c7 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7c7 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc22_36: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc22_36: Core.Form = init_form %C.ref.loc22_36, call_param2 [concrete = constants.%.92f]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.J.impl.%C.ref.loc21_6 [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:     %u.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc22_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %u: %C = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %C.as.J.impl.F.decl), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.00b]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C [concrete = constants.%C]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %C.as.J.impl.F.decl
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc18_10: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc18_11: type = converted %.loc18_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc18_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self.loc3: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.8a1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ed8)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.832)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.89e)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:   %pattern_type.loc5_20: type = pattern_type %impl.elem0.loc5_23.1 [symbolic = %pattern_type.loc5_20 (constants.%pattern_type.c1a)]
// CHECK:STDOUT:   %.loc5_29.1: Core.Form = init_form %impl.elem0.loc5_23.1, call_param2 [symbolic = %.loc5_29.1 (constants.%.147)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.ed8), %u.param: @J.WithSelf.F.%impl.elem0.loc5_23.1 (%impl.elem0.335)) -> out %return.param: @J.WithSelf.F.%impl.elem0.loc5_23.1 (%impl.elem0.335);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.J.impl.F(%self.param: %empty_tuple.type, %u.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %impl.elem1: %.5dd = impl_witness_access constants.%Negate.impl_witness.11c, element1 [concrete = constants.%Int.as.Negate.impl.Op.e7f]
// CHECK:STDOUT:   %bound_method.loc9_51.1: <bound method> = bound_method %u.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Int.as.Negate.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Negate.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_51.2: <bound method> = bound_method %u.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call: init %i32 = call %bound_method.loc9_51.2(%u.ref)
// CHECK:STDOUT:   return %Int.as.Negate.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallMethod(%x.param: %empty_tuple.type) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.eeb]
// CHECK:STDOUT:   %impl.elem1: %.b39 = impl_witness_access constants.%J.impl_witness.5bf, element1 [concrete = constants.%empty_tuple.type.as.J.impl.F]
// CHECK:STDOUT:   %bound_method.loc13_11: <bound method> = bound_method %x.ref, %impl.elem1
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40.f80]
// CHECK:STDOUT:   %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc13_18.1: <bound method> = bound_method %int_40, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_18.2: <bound method> = bound_method %int_40, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_18.2(%int_40) [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %.loc13_18.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %.loc13_18.2: %i32 = converted %int_40, %.loc13_18.1 [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %empty_tuple.type.as.J.impl.F.call: init %i32 = call %bound_method.loc13_11(%x.ref, %.loc13_18.2)
// CHECK:STDOUT:   return %empty_tuple.type.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.J.impl.F(%self.param: %C, %u.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   %.loc22_51.1: %empty_struct_type = as_compatible %self.ref
// CHECK:STDOUT:   %.loc22_51.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22_51.3: init %C = as_compatible %.loc22_51.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc22_51.4: init %C = converted %self.ref, %.loc22_51.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc22_51.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%Self.8a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%Self.8a1
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.adb
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.569
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.8a1) {
// CHECK:STDOUT:   %Self => constants.%Self.8a1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.ed8
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.832
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.89e
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.335
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.c1a
// CHECK:STDOUT:   %.loc5_29.1 => constants.%.147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%.Self.2fa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%.Self.2fa
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.e19
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.659
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%J.facet.a51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%J.facet.a51
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.fd2
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.497
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.a51) {
// CHECK:STDOUT:   %Self => constants.%J.facet.a51
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness.5bf
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.7ce
// CHECK:STDOUT:   %.loc5_29.1 => constants.%.8ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%J.facet.05d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%J.facet.05d
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.24e
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.bd5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.05d) {
// CHECK:STDOUT:   %Self => constants.%J.facet.05d
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.7c7
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness.00b
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.7c7
// CHECK:STDOUT:   %.loc5_29.1 => constants.%.92f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_type_in_function_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.ebd: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %J.lookup_impl_witness.89e: <witness> = lookup_impl_witness %Self.8a1, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.335: type = impl_witness_access %J.lookup_impl_witness.89e, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c1a: type = pattern_type %impl.elem0.335 [symbolic]
// CHECK:STDOUT:   %.a6a: Core.Form = init_form %impl.elem0.335, call_param1 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.adb: type = fn_type @J.WithSelf.F, @J(%Self.8a1) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.569: %J.WithSelf.F.type.adb = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.eeb: %J.assoc_type = assoc_entity element1, @J.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.2fa: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.type.e19: type = fn_type @J.WithSelf.F, @J(%.Self.2fa) [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.659: %J.WithSelf.F.type.e19 = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.2fa [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.46a: <witness> = lookup_impl_witness %.Self.2fa, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a58: type = impl_witness_access %J.lookup_impl_witness.46a, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.a58 = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @D.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.e54: Core.Form = init_form %i32, call_param1 [concrete]
// CHECK:STDOUT:   %D.as.J.impl.F.type: type = fn_type @D.as.J.impl.F [concrete]
// CHECK:STDOUT:   %D.as.J.impl.F: %D.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %D, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.7f3: type = fn_type @J.WithSelf.F, @J(%J.facet) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.973: %J.WithSelf.F.type.7f3 = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.5e6: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.7f5: %Int.as.Negate.impl.Op.type.5e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Negate.impl_witness.11c: <witness> = impl_witness imports.%Negate.impl_witness_table, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.e60: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.e7f: %Int.as.Negate.impl.Op.type.e60 = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %i32, (%Negate.impl_witness.11c) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.140: type = fn_type @Negate.WithSelf.Op, @Negate(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.5dd: type = fn_type_with_self_type %Negate.WithSelf.Op.type.140, %Negate.facet [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Negate.impl.Op.e7f, @Int.as.Negate.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %.437: Core.Form = init_form %i32, call_param0 [concrete]
// CHECK:STDOUT:   %CallFunction.type: type = fn_type @CallFunction [concrete]
// CHECK:STDOUT:   %CallFunction: %CallFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %.291: type = fn_type_with_self_type %J.WithSelf.F.type.7f3, %J.facet [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.475 = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d40: @Int.as.Negate.impl.%Int.as.Negate.impl.Op.type (%Int.as.Negate.impl.Op.type.5e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Negate.impl.%Int.as.Negate.impl.Op (constants.%Int.as.Negate.impl.Op.7f5)]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.475, %Core.import_ref.d40), @Int.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.2fa]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.2fa]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.46a, element0 [symbolic_self = constants.%impl.elem0.a58]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_13: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFunction.decl: %CallFunction.type = fn_decl @CallFunction [concrete = constants.%CallFunction] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14: Core.Form = init_form %i32, call_param0 [concrete = constants.%.437]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self.loc3: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.%J.WithSelf.F.type (%J.WithSelf.F.type.adb) = fn_decl @J.WithSelf.F [symbolic = @J.%J.WithSelf.F (constants.%J.WithSelf.F.569)] {
// CHECK:STDOUT:     %u.patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %.loc5_17.2: Core.Form = init_form %U.ref.loc5_17, call_param1 [symbolic = %.loc5_17.1 (constants.%.a6a)]
// CHECK:STDOUT:     %u.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = value_param call_param0
// CHECK:STDOUT:     %.loc5_11: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = out_param call_param1
// CHECK:STDOUT:     %return: ref @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.WithSelf.F.decl [concrete = constants.%assoc1.eeb]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
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
// CHECK:STDOUT:     %.loc11_19: Core.Form = init_form %i32.loc11_19, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc11_11: type = splice_block %i32.loc11_11 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %D.as.J.impl.F.decl), @D.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %D.as.J.impl.F.decl
// CHECK:STDOUT:   witness = %J.impl_witness
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
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self.loc3: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.8a1)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.89e)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.c1a)]
// CHECK:STDOUT:   %.loc5_17.1: Core.Form = init_form %impl.elem0.loc5_11.1, call_param1 [symbolic = %.loc5_17.1 (constants.%.a6a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335)) -> out %return.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.J.impl.F(%u.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %impl.elem1: %.5dd = impl_witness_access constants.%Negate.impl_witness.11c, element1 [concrete = constants.%Int.as.Negate.impl.Op.e7f]
// CHECK:STDOUT:   %bound_method.loc11_32.1: <bound method> = bound_method %u.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Int.as.Negate.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Negate.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_32.2: <bound method> = bound_method %u.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call: init %i32 = call %bound_method.loc11_32.2(%u.ref)
// CHECK:STDOUT:   return %Int.as.Negate.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFunction() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.eeb]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %D.ref, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc15_11: %J.type = converted %D.ref, %J.facet [concrete = constants.%J.facet]
// CHECK:STDOUT:   %impl.elem1: %.291 = impl_witness_access constants.%J.impl_witness, element1 [concrete = constants.%D.as.J.impl.F]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc15_18.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_18.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_18.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc15_18.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc15_18.2: %i32 = converted %int_4, %.loc15_18.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %D.as.J.impl.F.call: init %i32 = call %impl.elem1(%.loc15_18.2)
// CHECK:STDOUT:   return %D.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%Self.8a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%Self.8a1
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.adb
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.569
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.8a1) {
// CHECK:STDOUT:   %Self => constants.%Self.8a1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.89e
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.335
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1a
// CHECK:STDOUT:   %.loc5_17.1 => constants.%.a6a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%.Self.2fa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%.Self.2fa
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.e19
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.659
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%J.facet
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.7f3
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.973
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:   %.loc5_17.1 => constants.%.e54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_with_associated_type_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.ebd: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %J.lookup_impl_witness.89e: <witness> = lookup_impl_witness %Self.8a1, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.335: type = impl_witness_access %J.lookup_impl_witness.89e, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c1a: type = pattern_type %impl.elem0.335 [symbolic]
// CHECK:STDOUT:   %.a6a: Core.Form = init_form %impl.elem0.335, call_param1 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.adb: type = fn_type @J.WithSelf.F, @J(%Self.8a1) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.569: %J.WithSelf.F.type.adb = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.eeb: %J.assoc_type = assoc_entity element1, @J.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.ed8: type = symbolic_binding_type Self, 0, %Self.8a1 [symbolic]
// CHECK:STDOUT:   %pattern_type.832: type = pattern_type %Self.binding.as_type.ed8 [symbolic]
// CHECK:STDOUT:   %.147: Core.Form = init_form %impl.elem0.335, call_param2 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.G.type.ace: type = fn_type @J.WithSelf.G, @J(%Self.8a1) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.G.6e7: %J.WithSelf.G.type.ace = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc2: %J.assoc_type = assoc_entity element2, @J.%J.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %.Self.2fa: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.type.e19: type = fn_type @J.WithSelf.F, @J(%.Self.2fa) [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.659: %J.WithSelf.F.type.e19 = struct_value () [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.G.type.2cf: type = fn_type @J.WithSelf.G, @J(%.Self.2fa) [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.G.4f5: %J.WithSelf.G.type.2cf = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.2fa [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.46a: <witness> = lookup_impl_witness %.Self.2fa, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a58: type = impl_witness_access %J.lookup_impl_witness.46a, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.a58 = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.e54: Core.Form = init_form %i32, call_param1 [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type: type = fn_type @E.as.J.impl.F [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F: %E.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.99f: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %.8ef: Core.Form = init_form %i32, call_param2 [concrete]
// CHECK:STDOUT:   %E.as.J.impl.G.type: type = fn_type @E.as.J.impl.G [concrete]
// CHECK:STDOUT:   %E.as.J.impl.G: %E.as.J.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.7ab: type = fn_type @J.WithSelf.F, @J(%J.facet) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.ef0: %J.WithSelf.F.type.7ab = struct_value () [concrete]
// CHECK:STDOUT:   %J.WithSelf.G.type.caf: type = fn_type @J.WithSelf.G, @J(%J.facet) [concrete]
// CHECK:STDOUT:   %J.WithSelf.G.f88: %J.WithSelf.G.type.caf = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.5e6: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.7f5: %Int.as.Negate.impl.Op.type.5e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Negate.impl_witness.11c: <witness> = impl_witness imports.%Negate.impl_witness_table, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.e60: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.e7f: %Int.as.Negate.impl.Op.type.e60 = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %i32, (%Negate.impl_witness.11c) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.140: type = fn_type @Negate.WithSelf.Op, @Negate(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.5dd: type = fn_type_with_self_type %Negate.WithSelf.Op.type.140, %Negate.facet [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Negate.impl.Op.e7f, @Int.as.Negate.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %CallBoth.type: type = fn_type @CallBoth [concrete]
// CHECK:STDOUT:   %CallBoth: %CallBoth.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %E.type.facet: %type = facet_value %E, () [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.d1c: type = fn_type @J.WithSelf.F, @J(%E.type.facet) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.1a6: %J.WithSelf.F.type.d1c = struct_value () [concrete]
// CHECK:STDOUT:   %J.WithSelf.G.type.d39: type = fn_type @J.WithSelf.G, @J(%E.type.facet) [concrete]
// CHECK:STDOUT:   %J.WithSelf.G.a50: %J.WithSelf.G.type.d39 = struct_value () [concrete]
// CHECK:STDOUT:   %.ff2: type = fn_type_with_self_type %J.WithSelf.F.type.7ab, %J.facet [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %.825: type = fn_type_with_self_type %J.WithSelf.G.type.caf, %J.facet [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.6d7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.e9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.502: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.bc2: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.14f: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.41f: type = fn_type @J.WithSelf.F, @J(%T) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.f01: %J.WithSelf.F.type.41f = struct_value () [symbolic]
// CHECK:STDOUT:   %J.WithSelf.G.type.b2e: type = fn_type @J.WithSelf.G, @J(%T) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.G.310: %J.WithSelf.G.type.b2e = struct_value () [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.264: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.560: type = impl_witness_access %J.lookup_impl_witness.264, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ad6: type = pattern_type %impl.elem0.560 [symbolic]
// CHECK:STDOUT:   %.0b0: Core.Form = init_form %impl.elem0.560, call_param2 [symbolic]
// CHECK:STDOUT:   %GenericCallF.type: type = fn_type @GenericCallF [concrete]
// CHECK:STDOUT:   %GenericCallF: %GenericCallF.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.19a: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.06a: <witness> = require_complete_type %impl.elem0.560 [symbolic]
// CHECK:STDOUT:   %.43f: type = fn_type_with_self_type %J.WithSelf.F.type.41f, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.43f = impl_witness_access %J.lookup_impl_witness.264, element1 [symbolic]
// CHECK:STDOUT:   %.a84: Core.Form = init_form %impl.elem0.560, call_param1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem1, @J.WithSelf.F(%T) [symbolic]
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
// CHECK:STDOUT:   %Core.import_ref.475 = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d40: @Int.as.Negate.impl.%Int.as.Negate.impl.Op.type (%Int.as.Negate.impl.Op.type.5e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Negate.impl.%Int.as.Negate.impl.Op (constants.%Int.as.Negate.impl.Op.7f5)]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.475, %Core.import_ref.d40), @Int.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %e.patt: %pattern_type.99f = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: %pattern_type.99f = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref.loc20: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericCallF.decl: %GenericCallF.type = fn_decl @GenericCallF [concrete = constants.%GenericCallF] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f76 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericCallF.%pattern_type.loc28_24 (%pattern_type.14f) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericCallF.%pattern_type.loc28_24 (%pattern_type.14f) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.ad6) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.ad6) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.ad6) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @GenericCallF.%pattern_type.loc28_30 (%pattern_type.ad6) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc28_41: %J.type = name_ref T, %T.loc28_17.2 [symbolic = %T.loc28_17.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc28_42: type = facet_access_type %T.ref.loc28_41 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc28_42.3: type = converted %T.ref.loc28_41, %T.as_type.loc28_42 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %U.ref.loc28_42: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:     %impl.elem0.loc28_42: type = impl_witness_access constants.%J.lookup_impl_witness.264, element0 [symbolic = %impl.elem0.loc28_34.1 (constants.%impl.elem0.560)]
// CHECK:STDOUT:     %.loc28_42.4: Core.Form = init_form %impl.elem0.loc28_42, call_param2 [symbolic = %.loc28_42.2 (constants.%.0b0)]
// CHECK:STDOUT:     %.loc28_21: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
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
// CHECK:STDOUT:     %u.param: @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.560) = value_param call_param1
// CHECK:STDOUT:     %.loc28_34.1: type = splice_block %impl.elem0.loc28_34.2 [symbolic = %impl.elem0.loc28_34.1 (constants.%impl.elem0.560)] {
// CHECK:STDOUT:       %T.ref.loc28_33: %J.type = name_ref T, %T.loc28_17.2 [symbolic = %T.loc28_17.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc28_34: type = facet_access_type %T.ref.loc28_33 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc28_34.2: type = converted %T.ref.loc28_33, %T.as_type.loc28_34 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %U.ref.loc28_34: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:       %impl.elem0.loc28_34.2: type = impl_witness_access constants.%J.lookup_impl_witness.264, element0 [symbolic = %impl.elem0.loc28_34.1 (constants.%impl.elem0.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.560) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %e.patt: %pattern_type.99f = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: %pattern_type.99f = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc32: Core.Form = init_form %i32, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = value_binding e, %e.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self.loc3: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.%J.WithSelf.F.type (%J.WithSelf.F.type.adb) = fn_decl @J.WithSelf.F [symbolic = @J.%J.WithSelf.F (constants.%J.WithSelf.F.569)] {
// CHECK:STDOUT:     %u.patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %.loc5_17.2: Core.Form = init_form %U.ref.loc5_17, call_param1 [symbolic = %.loc5_17.1 (constants.%.a6a)]
// CHECK:STDOUT:     %u.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = value_param call_param0
// CHECK:STDOUT:     %.loc5_11: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = out_param call_param1
// CHECK:STDOUT:     %return: ref @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.WithSelf.F.decl [concrete = constants.%assoc1.eeb]
// CHECK:STDOUT:   %J.WithSelf.G.decl: @J.%J.WithSelf.G.type (%J.WithSelf.G.type.ace) = fn_decl @J.WithSelf.G [symbolic = @J.%J.WithSelf.G (constants.%J.WithSelf.G.6e7)] {
// CHECK:STDOUT:     %self.patt: @J.WithSelf.G.%pattern_type.loc6_8 (%pattern_type.832) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @J.WithSelf.G.%pattern_type.loc6_8 (%pattern_type.832) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: @J.WithSelf.G.%pattern_type.loc6_20 (%pattern_type.c1a) = value_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: @J.WithSelf.G.%pattern_type.loc6_20 (%pattern_type.c1a) = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @J.WithSelf.G.%pattern_type.loc6_20 (%pattern_type.c1a) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.WithSelf.G.%pattern_type.loc6_20 (%pattern_type.c1a) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc6_29: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %U.ref.loc6_29: type = name_ref U, %impl.elem0.loc6_29 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %.loc6_29.2: Core.Form = init_form %U.ref.loc6_29, call_param2 [symbolic = %.loc6_29.1 (constants.%.147)]
// CHECK:STDOUT:     %self.param: @J.WithSelf.G.%Self.binding.as_type (%Self.binding.as_type.ed8) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ed8)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self.loc3 [symbolic = %Self (constants.%Self.8a1)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ed8)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ed8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J.WithSelf.G.%Self.binding.as_type (%Self.binding.as_type.ed8) = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: @J.WithSelf.G.%impl.elem0.loc6_23.1 (%impl.elem0.335) = value_param call_param1
// CHECK:STDOUT:     %.loc6_23: type = splice_block %U.ref.loc6_23 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.335)] {
// CHECK:STDOUT:       %impl.elem0.loc6_23.2: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:       %U.ref.loc6_23: type = name_ref U, %impl.elem0.loc6_23.2 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @J.WithSelf.G.%impl.elem0.loc6_23.1 (%impl.elem0.335) = value_binding v, %v.param
// CHECK:STDOUT:     %return.param: ref @J.WithSelf.G.%impl.elem0.loc6_23.1 (%impl.elem0.335) = out_param call_param2
// CHECK:STDOUT:     %return: ref @J.WithSelf.G.%impl.elem0.loc6_23.1 (%impl.elem0.335) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc2: %J.assoc_type = assoc_entity element2, %J.WithSelf.G.decl [concrete = constants.%assoc2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   .G = %assoc2
// CHECK:STDOUT:   witness = (%U, %J.WithSelf.F.decl, %J.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
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
// CHECK:STDOUT:     %.loc11_21: Core.Form = init_form %i32.loc11_21, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc11_13: type = splice_block %i32.loc11_13 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.as.J.impl.G.decl: %E.as.J.impl.G.type = fn_decl @E.as.J.impl.G [concrete = constants.%E.as.J.impl.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.99f = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.99f = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = value_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: %pattern_type.7ce = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc14_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_40: Core.Form = init_form %i32.loc14_40, call_param2 [concrete = constants.%.8ef]
// CHECK:STDOUT:     %self.param: %E = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %self: %E = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc14_32: type = splice_block %i32.loc14_32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc14_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: %i32 = value_binding v, %v.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %E.as.J.impl.F.decl, %E.as.J.impl.G.decl), @E.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %E.as.J.impl.F.decl
// CHECK:STDOUT:   .G = %E.as.J.impl.G.decl
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @E.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.2fa]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.2fa]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_26: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.46a, element0 [symbolic_self = constants.%impl.elem0.a58]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_20: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self.loc3: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.8a1)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.89e)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.c1a)]
// CHECK:STDOUT:   %.loc5_17.1: Core.Form = init_form %impl.elem0.loc5_11.1, call_param1 [symbolic = %.loc5_17.1 (constants.%.a6a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335)) -> out %return.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.G(@J.%Self.loc3: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.8a1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ed8)]
// CHECK:STDOUT:   %pattern_type.loc6_8: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc6_8 (constants.%pattern_type.832)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.89e)]
// CHECK:STDOUT:   %impl.elem0.loc6_23.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:   %pattern_type.loc6_20: type = pattern_type %impl.elem0.loc6_23.1 [symbolic = %pattern_type.loc6_20 (constants.%pattern_type.c1a)]
// CHECK:STDOUT:   %.loc6_29.1: Core.Form = init_form %impl.elem0.loc6_23.1, call_param2 [symbolic = %.loc6_29.1 (constants.%.147)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J.WithSelf.G.%Self.binding.as_type (%Self.binding.as_type.ed8), %v.param: @J.WithSelf.G.%impl.elem0.loc6_23.1 (%impl.elem0.335)) -> out %return.param: @J.WithSelf.G.%impl.elem0.loc6_23.1 (%impl.elem0.335);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.F(%u.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %impl.elem1: %.5dd = impl_witness_access constants.%Negate.impl_witness.11c, element1 [concrete = constants.%Int.as.Negate.impl.Op.e7f]
// CHECK:STDOUT:   %bound_method.loc12_14.1: <bound method> = bound_method %u.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Int.as.Negate.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Negate.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_14.2: <bound method> = bound_method %u.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call: init %i32 = call %bound_method.loc12_14.2(%u.ref)
// CHECK:STDOUT:   return %Int.as.Negate.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.G(%self.param: %E, %v.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: %i32 = name_ref v, %v
// CHECK:STDOUT:   %impl.elem1: %.5dd = impl_witness_access constants.%Negate.impl_witness.11c, element1 [concrete = constants.%Int.as.Negate.impl.Op.e7f]
// CHECK:STDOUT:   %bound_method.loc15_14.1: <bound method> = bound_method %v.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Int.as.Negate.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Negate.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_14.2: <bound method> = bound_method %v.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call: init %i32 = call %bound_method.loc15_14.2(%v.ref)
// CHECK:STDOUT:   return %Int.as.Negate.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallBoth(%e.param: %E) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e1.patt: %pattern_type.7ce = value_binding_pattern e1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc21: %E = name_ref e, %e
// CHECK:STDOUT:   %F.ref.loc21: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.eeb]
// CHECK:STDOUT:   %impl.elem1.loc21: %.ff2 = impl_witness_access constants.%J.impl_witness, element1 [concrete = constants.%E.as.J.impl.F]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc21: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc21_28.1: <bound method> = bound_method %int_2, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_28.2: <bound method> = bound_method %int_2, %specific_fn.loc21 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %i32 = call %bound_method.loc21_28.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc21_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc21_28.2: %i32 = converted %int_2, %.loc21_28.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %E.as.J.impl.F.call.loc21: init %i32 = call %impl.elem1.loc21(%.loc21_28.2)
// CHECK:STDOUT:   %.loc21_18: type = splice_block %i32.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_29.1: %i32 = value_of_initializer %E.as.J.impl.F.call.loc21
// CHECK:STDOUT:   %.loc21_29.2: %i32 = converted %E.as.J.impl.F.call.loc21, %.loc21_29.1
// CHECK:STDOUT:   %e1: %i32 = value_binding e1, %.loc21_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e2.patt: %pattern_type.7ce = value_binding_pattern e2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc22: %E = name_ref e, %e
// CHECK:STDOUT:   %G.ref.loc22: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc22: %.825 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%E.as.J.impl.G]
// CHECK:STDOUT:   %bound_method.loc22_25: <bound method> = bound_method %e.ref.loc22, %impl.elem2.loc22
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc22: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc22_28.1: <bound method> = bound_method %int_3, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.2: <bound method> = bound_method %int_3, %specific_fn.loc22 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %i32 = call %bound_method.loc22_28.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22_28.2: %i32 = converted %int_3, %.loc22_28.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %E.as.J.impl.G.call.loc22: init %i32 = call %bound_method.loc22_25(%e.ref.loc22, %.loc22_28.2)
// CHECK:STDOUT:   %.loc22_18: type = splice_block %i32.loc22 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_29.1: %i32 = value_of_initializer %E.as.J.impl.G.call.loc22
// CHECK:STDOUT:   %.loc22_29.2: %i32 = converted %E.as.J.impl.G.call.loc22, %.loc22_29.1
// CHECK:STDOUT:   %e2: %i32 = value_binding e2, %.loc22_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e3.patt: %pattern_type.7ce = value_binding_pattern e3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.ref.loc23: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %F.ref.loc23: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.eeb]
// CHECK:STDOUT:   %impl.elem1.loc23: %.ff2 = impl_witness_access constants.%J.impl_witness, element1 [concrete = constants.%E.as.J.impl.F]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc23: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc23_28.1: <bound method> = bound_method %int_4, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_28.2: <bound method> = bound_method %int_4, %specific_fn.loc23 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %i32 = call %bound_method.loc23_28.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc23_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc23_28.2: %i32 = converted %int_4, %.loc23_28.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %E.as.J.impl.F.call.loc23: init %i32 = call %impl.elem1.loc23(%.loc23_28.2)
// CHECK:STDOUT:   %.loc23_18: type = splice_block %i32.loc23 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_29.1: %i32 = value_of_initializer %E.as.J.impl.F.call.loc23
// CHECK:STDOUT:   %.loc23_29.2: %i32 = converted %E.as.J.impl.F.call.loc23, %.loc23_29.1
// CHECK:STDOUT:   %e3: %i32 = value_binding e3, %.loc23_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e4.patt: %pattern_type.7ce = value_binding_pattern e4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc24: %E = name_ref e, %e
// CHECK:STDOUT:   %E.ref.loc24: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %G.ref.loc24: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc24: %.825 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%E.as.J.impl.G]
// CHECK:STDOUT:   %bound_method.loc24_25: <bound method> = bound_method %e.ref.loc24, %impl.elem2.loc24
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc24: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc24_32.1: <bound method> = bound_method %int_5, %impl.elem0.loc24 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_32.2: <bound method> = bound_method %int_5, %specific_fn.loc24 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24: init %i32 = call %bound_method.loc24_32.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc24_32.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc24_32.2: %i32 = converted %int_5, %.loc24_32.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %E.as.J.impl.G.call.loc24: init %i32 = call %bound_method.loc24_25(%e.ref.loc24, %.loc24_32.2)
// CHECK:STDOUT:   %.loc24_18: type = splice_block %i32.loc24 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24_33.1: %i32 = value_of_initializer %E.as.J.impl.G.call.loc24
// CHECK:STDOUT:   %.loc24_33.2: %i32 = converted %E.as.J.impl.G.call.loc24, %.loc24_33.1
// CHECK:STDOUT:   %e4: %i32 = value_binding e4, %.loc24_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e5.patt: %pattern_type.7ce = value_binding_pattern e5 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc25: %E = name_ref e, %e
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %G.ref.loc25: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc25: %.825 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%E.as.J.impl.G]
// CHECK:STDOUT:   %bound_method.loc25_25: <bound method> = bound_method %e.ref.loc25, %impl.elem2.loc25
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %impl.elem0.loc25: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc25_32.1: <bound method> = bound_method %int_6, %impl.elem0.loc25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.502]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_32.2: <bound method> = bound_method %int_6, %specific_fn.loc25 [concrete = constants.%bound_method.bc2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25: init %i32 = call %bound_method.loc25_32.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc25_32.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc25_32.2: %i32 = converted %int_6, %.loc25_32.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %E.as.J.impl.G.call.loc25: init %i32 = call %bound_method.loc25_25(%e.ref.loc25, %.loc25_32.2)
// CHECK:STDOUT:   %.loc25_18: type = splice_block %i32.loc25 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25_33.1: %i32 = value_of_initializer %E.as.J.impl.G.call.loc25
// CHECK:STDOUT:   %.loc25_33.2: %i32 = converted %E.as.J.impl.G.call.loc25, %.loc25_33.1
// CHECK:STDOUT:   %e5: %i32 = value_binding e5, %.loc25_33.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallF(%T.loc28_17.2: %J.type) {
// CHECK:STDOUT:   %T.loc28_17.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc28_17.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc28_17.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc28_24: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc28_24 (constants.%pattern_type.14f)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc28_17.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.264)]
// CHECK:STDOUT:   %impl.elem0.loc28_34.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_34.1 (constants.%impl.elem0.560)]
// CHECK:STDOUT:   %pattern_type.loc28_30: type = pattern_type %impl.elem0.loc28_34.1 [symbolic = %pattern_type.loc28_30 (constants.%pattern_type.ad6)]
// CHECK:STDOUT:   %.loc28_42.2: Core.Form = init_form %impl.elem0.loc28_34.1, call_param2 [symbolic = %.loc28_42.2 (constants.%.0b0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28_25: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc28_25 (constants.%require_complete.19a)]
// CHECK:STDOUT:   %require_complete.loc28_31: <witness> = require_complete_type %impl.elem0.loc28_34.1 [symbolic = %require_complete.loc28_31 (constants.%require_complete.06a)]
// CHECK:STDOUT:   %J.WithSelf.F.type: type = fn_type @J.WithSelf.F, @J(%T.loc28_17.1) [symbolic = %J.WithSelf.F.type (constants.%J.WithSelf.F.type.41f)]
// CHECK:STDOUT:   %.loc29: type = fn_type_with_self_type %J.WithSelf.F.type, %T.loc28_17.1 [symbolic = %.loc29 (constants.%.43f)]
// CHECK:STDOUT:   %impl.elem1.loc29_11.2: @GenericCallF.%.loc29 (%.43f) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc29_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc29_11.2: <specific function> = specific_impl_function %impl.elem1.loc29_11.2, @J.WithSelf.F(%T.loc28_17.1) [symbolic = %specific_impl_fn.loc29_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericCallF.%T.binding.as_type (%T.binding.as_type), %u.param: @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.560)) -> out %return.param: @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.560) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallF.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1.eeb]
// CHECK:STDOUT:     %impl.elem1.loc29_11.1: @GenericCallF.%.loc29 (%.43f) = impl_witness_access constants.%J.lookup_impl_witness.264, element1 [symbolic = %impl.elem1.loc29_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref: @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.560) = name_ref u, %u
// CHECK:STDOUT:     %specific_impl_fn.loc29_11.1: <specific function> = specific_impl_function %impl.elem1.loc29_11.1, @J.WithSelf.F(constants.%T) [symbolic = %specific_impl_fn.loc29_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc28_42.1: ref @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.560) = splice_block %return.param {}
// CHECK:STDOUT:     %J.WithSelf.F.call: init @GenericCallF.%impl.elem0.loc28_34.1 (%impl.elem0.560) to %.loc28_42.1 = call %specific_impl_fn.loc29_11.1(%u.ref)
// CHECK:STDOUT:     return %J.WithSelf.F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGeneric(%e.param: %E) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %GenericCallF.ref: %GenericCallF.type = name_ref GenericCallF, file.%GenericCallF.decl [concrete = constants.%GenericCallF]
// CHECK:STDOUT:   %e.ref: %E = name_ref e, %e
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %J.facet.loc33_27.1: %J.type = facet_value constants.%E, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc33_27.1: %J.type = converted constants.%E, %J.facet.loc33_27.1 [concrete = constants.%J.facet]
// CHECK:STDOUT:   %J.facet.loc33_27.2: %J.type = facet_value constants.%E, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc33_27.2: %J.type = converted constants.%E, %J.facet.loc33_27.2 [concrete = constants.%J.facet]
// CHECK:STDOUT:   %GenericCallF.specific_fn: <specific function> = specific_function %GenericCallF.ref, @GenericCallF(constants.%J.facet) [concrete = constants.%GenericCallF.specific_fn]
// CHECK:STDOUT:   %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc33_26.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_26.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc33_26.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc33_26.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc33_26.2: %i32 = converted %int_2, %.loc33_26.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %GenericCallF.call: init %i32 = call %GenericCallF.specific_fn(%e.ref, %.loc33_26.2)
// CHECK:STDOUT:   return %GenericCallF.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%Self.8a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%Self.8a1
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.adb
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.569
// CHECK:STDOUT:   %J.WithSelf.G.type => constants.%J.WithSelf.G.type.ace
// CHECK:STDOUT:   %J.WithSelf.G => constants.%J.WithSelf.G.6e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.8a1) {
// CHECK:STDOUT:   %Self => constants.%Self.8a1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.89e
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.335
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1a
// CHECK:STDOUT:   %.loc5_17.1 => constants.%.a6a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.G(constants.%Self.8a1) {
// CHECK:STDOUT:   %Self => constants.%Self.8a1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.ed8
// CHECK:STDOUT:   %pattern_type.loc6_8 => constants.%pattern_type.832
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.89e
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%impl.elem0.335
// CHECK:STDOUT:   %pattern_type.loc6_20 => constants.%pattern_type.c1a
// CHECK:STDOUT:   %.loc6_29.1 => constants.%.147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%.Self.2fa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%.Self.2fa
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.e19
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.659
// CHECK:STDOUT:   %J.WithSelf.G.type => constants.%J.WithSelf.G.type.2cf
// CHECK:STDOUT:   %J.WithSelf.G => constants.%J.WithSelf.G.4f5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%J.facet
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.7ab
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.ef0
// CHECK:STDOUT:   %J.WithSelf.G.type => constants.%J.WithSelf.G.type.caf
// CHECK:STDOUT:   %J.WithSelf.G => constants.%J.WithSelf.G.f88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:   %.loc5_17.1 => constants.%.e54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.G(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%E
// CHECK:STDOUT:   %pattern_type.loc6_8 => constants.%pattern_type.99f
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc6_20 => constants.%pattern_type.7ce
// CHECK:STDOUT:   %.loc6_29.1 => constants.%.8ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%E.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%E.type.facet
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.d1c
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.1a6
// CHECK:STDOUT:   %J.WithSelf.G.type => constants.%J.WithSelf.G.type.d39
// CHECK:STDOUT:   %J.WithSelf.G => constants.%J.WithSelf.G.a50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%T
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.41f
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.f01
// CHECK:STDOUT:   %J.WithSelf.G.type => constants.%J.WithSelf.G.type.b2e
// CHECK:STDOUT:   %J.WithSelf.G => constants.%J.WithSelf.G.310
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallF(constants.%T) {
// CHECK:STDOUT:   %T.loc28_17.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc28_24 => constants.%pattern_type.14f
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.264
// CHECK:STDOUT:   %impl.elem0.loc28_34.1 => constants.%impl.elem0.560
// CHECK:STDOUT:   %pattern_type.loc28_30 => constants.%pattern_type.ad6
// CHECK:STDOUT:   %.loc28_42.2 => constants.%.0b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.264
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.560
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ad6
// CHECK:STDOUT:   %.loc5_17.1 => constants.%.a84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallF(constants.%J.facet) {
// CHECK:STDOUT:   %T.loc28_17.1 => constants.%J.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%E
// CHECK:STDOUT:   %pattern_type.loc28_24 => constants.%pattern_type.99f
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc28_34.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc28_30 => constants.%pattern_type.7ce
// CHECK:STDOUT:   %.loc28_42.2 => constants.%.8ef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28_25 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc28_31 => constants.%complete_type.f8a
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.7ab
// CHECK:STDOUT:   %.loc29 => constants.%.ff2
// CHECK:STDOUT:   %impl.elem1.loc29_11.2 => constants.%E.as.J.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc29_11.2 => constants.%E.as.J.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_constraint_on_associated_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.d31: type = symbolic_binding_type Self, 0, %Self.ab9 [symbolic]
// CHECK:STDOUT:   %pattern_type.fa0: type = pattern_type %Self.binding.as_type.d31 [symbolic]
// CHECK:STDOUT:   %.389: Core.Form = init_form %Self.binding.as_type.d31, call_param2 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Op.type.71c: type = fn_type @I.WithSelf.Op, @I(%Self.ab9) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.Op.ae1: %I.WithSelf.Op.type.71c = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.aa8: %I.assoc_type = assoc_entity element0, @I.%I.WithSelf.Op.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @Destroy> [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.445: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %J.lookup_impl_witness.89e: <witness> = lookup_impl_witness %Self.8a1, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.193: %facet_type = impl_witness_access %J.lookup_impl_witness.89e, element0 [symbolic]
// CHECK:STDOUT:   %as_type.d83: type = facet_access_type %impl.elem0.193 [symbolic]
// CHECK:STDOUT:   %pattern_type.d76: type = pattern_type %as_type.d83 [symbolic]
// CHECK:STDOUT:   %.3f3: Core.Form = init_form %as_type.d83, call_param1 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.adb: type = fn_type @J.WithSelf.F, @J(%Self.8a1) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.569: %J.WithSelf.F.type.adb = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.14f: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.41f: type = fn_type @J.WithSelf.F, @J(%T) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.f01: %J.WithSelf.F.type.41f = struct_value () [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.264: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.397: %facet_type = impl_witness_access %J.lookup_impl_witness.264, element0 [symbolic]
// CHECK:STDOUT:   %as_type.baf: type = facet_access_type %impl.elem0.397 [symbolic]
// CHECK:STDOUT:   %pattern_type.bba: type = pattern_type %as_type.baf [symbolic]
// CHECK:STDOUT:   %.2cc: Core.Form = init_form %as_type.baf, call_param2 [symbolic]
// CHECK:STDOUT:   %GenericResult.type: type = fn_type @GenericResult [concrete]
// CHECK:STDOUT:   %GenericResult: %GenericResult.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.19a: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.9b3: <witness> = require_complete_type %as_type.baf [symbolic]
// CHECK:STDOUT:   %.43f: type = fn_type_with_self_type %J.WithSelf.F.type.41f, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.43f = impl_witness_access %J.lookup_impl_witness.264, element1 [symbolic]
// CHECK:STDOUT:   %.c35: Core.Form = init_form %as_type.baf, call_param1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a64: <specific function> = specific_impl_function %impl.elem1, @J.WithSelf.F(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.397, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %as_type.baf, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Op.type.25c: type = fn_type @I.WithSelf.Op, @I(%I.facet) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Op.fdc: %I.WithSelf.Op.type.25c = struct_value () [symbolic]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %I.WithSelf.Op.type.25c, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.d74: %.73b = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b45: <specific function> = specific_impl_function %impl.elem0.d74, @I.WithSelf.Op(%I.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.397, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.baf, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.077: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %.0e3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.077, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.4a3: %.0e3 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.86e: <specific function> = specific_impl_function %impl.elem0.4a3, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.f76 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericResult.%pattern_type.loc12_25 (%pattern_type.14f) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericResult.%pattern_type.loc12_25 (%pattern_type.14f) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.bba) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.bba) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.bba) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @GenericResult.%pattern_type.loc12_31 (%pattern_type.bba) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_42: %J.type = name_ref T, %T.loc12_18.2 [symbolic = %T.loc12_18.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc12_43: type = facet_access_type %T.ref.loc12_42 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc12_43.3: type = converted %T.ref.loc12_42, %T.as_type.loc12_43 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %U.ref.loc12_43: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.445]
// CHECK:STDOUT:     %impl.elem0.loc12_43: %facet_type = impl_witness_access constants.%J.lookup_impl_witness.264, element0 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.397)]
// CHECK:STDOUT:     %as_type.loc12_43: type = facet_access_type %impl.elem0.loc12_43 [symbolic = %as_type.loc12_35.1 (constants.%as_type.baf)]
// CHECK:STDOUT:     %.loc12_43.4: type = converted %impl.elem0.loc12_43, %as_type.loc12_43 [symbolic = %as_type.loc12_35.1 (constants.%as_type.baf)]
// CHECK:STDOUT:     %.loc12_43.5: Core.Form = init_form %.loc12_43.4, call_param2 [symbolic = %.loc12_43.2 (constants.%.2cc)]
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
// CHECK:STDOUT:     %u.param: @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = value_param call_param1
// CHECK:STDOUT:     %.loc12_35.1: type = splice_block %.loc12_35.3 [symbolic = %as_type.loc12_35.1 (constants.%as_type.baf)] {
// CHECK:STDOUT:       %T.ref.loc12_34: %J.type = name_ref T, %T.loc12_18.2 [symbolic = %T.loc12_18.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc12_35: type = facet_access_type %T.ref.loc12_34 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc12_35.2: type = converted %T.ref.loc12_34, %T.as_type.loc12_35 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %U.ref.loc12_35: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.445]
// CHECK:STDOUT:       %impl.elem0.loc12_35.2: %facet_type = impl_witness_access constants.%J.lookup_impl_witness.264, element0 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.397)]
// CHECK:STDOUT:       %as_type.loc12_35.2: type = facet_access_type %impl.elem0.loc12_35.2 [symbolic = %as_type.loc12_35.1 (constants.%as_type.baf)]
// CHECK:STDOUT:       %.loc12_35.3: type = converted %impl.elem0.loc12_35.2, %as_type.loc12_35.2 [symbolic = %as_type.loc12_35.1 (constants.%as_type.baf)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self.loc3: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.Op.decl: @I.%I.WithSelf.Op.type (%I.WithSelf.Op.type.71c) = fn_decl @I.WithSelf.Op [symbolic = @I.%I.WithSelf.Op (constants.%I.WithSelf.Op.ae1)] {
// CHECK:STDOUT:     %self.patt: @I.WithSelf.Op.%pattern_type (%pattern_type.fa0) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @I.WithSelf.Op.%pattern_type (%pattern_type.fa0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: @I.WithSelf.Op.%pattern_type (%pattern_type.fa0) = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: @I.WithSelf.Op.%pattern_type (%pattern_type.fa0) = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @I.WithSelf.Op.%pattern_type (%pattern_type.fa0) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @I.WithSelf.Op.%pattern_type (%pattern_type.fa0) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc4_33: %I.type = name_ref Self, @I.%Self.loc3 [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:     %Self.as_type.loc4_33: type = facet_access_type %Self.ref.loc4_33 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:     %.loc4_33.2: type = converted %Self.ref.loc4_33, %Self.as_type.loc4_33 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:     %.loc4_33.3: Core.Form = init_form %.loc4_33.2, call_param2 [symbolic = %.loc4_33.1 (constants.%.389)]
// CHECK:STDOUT:     %self.param: @I.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type.d31) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)] {
// CHECK:STDOUT:       %Self.ref.loc4_15: %I.type = name_ref Self, @I.%Self.loc3 [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:       %Self.as_type.loc4_15: type = facet_access_type %Self.ref.loc4_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref.loc4_15, %Self.as_type.loc4_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type.d31) = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: @I.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type.d31) = value_param call_param1
// CHECK:STDOUT:     %.loc4_24.1: type = splice_block %.loc4_24.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)] {
// CHECK:STDOUT:       %Self.ref.loc4_24: %I.type = name_ref Self, @I.%Self.loc3 [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:       %Self.as_type.loc4_24: type = facet_access_type %Self.ref.loc4_24 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:       %.loc4_24.2: type = converted %Self.ref.loc4_24, %Self.as_type.loc4_24 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: @I.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type.d31) = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref @I.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type.d31) = out_param call_param2
// CHECK:STDOUT:     %return: ref @I.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type.d31) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.Op.decl [concrete = constants.%assoc0.aa8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%I.WithSelf.Op.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self.loc7: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %U: %facet_type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.445]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.%J.WithSelf.F.type (%J.WithSelf.F.type.adb) = fn_decl @J.WithSelf.F [symbolic = @J.%J.WithSelf.F (constants.%J.WithSelf.F.569)] {
// CHECK:STDOUT:     %u.patt: @J.WithSelf.F.%pattern_type (%pattern_type.d76) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.WithSelf.F.%pattern_type (%pattern_type.d76) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @J.WithSelf.F.%pattern_type (%pattern_type.d76) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.WithSelf.F.%pattern_type (%pattern_type.d76) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc9_17: %facet_type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.193)]
// CHECK:STDOUT:     %U.ref.loc9_17: %facet_type = name_ref U, %impl.elem0.loc9_17 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.193)]
// CHECK:STDOUT:     %U.as_type.loc9_17: type = facet_access_type %U.ref.loc9_17 [symbolic = %as_type (constants.%as_type.d83)]
// CHECK:STDOUT:     %.loc9_17.2: type = converted %U.ref.loc9_17, %U.as_type.loc9_17 [symbolic = %as_type (constants.%as_type.d83)]
// CHECK:STDOUT:     %.loc9_17.3: Core.Form = init_form %.loc9_17.2, call_param1 [symbolic = %.loc9_17.1 (constants.%.3f3)]
// CHECK:STDOUT:     %u.param: @J.WithSelf.F.%as_type (%as_type.d83) = value_param call_param0
// CHECK:STDOUT:     %.loc9_11.1: type = splice_block %.loc9_11.2 [symbolic = %as_type (constants.%as_type.d83)] {
// CHECK:STDOUT:       %impl.elem0.loc9_11.2: %facet_type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.193)]
// CHECK:STDOUT:       %U.ref.loc9_11: %facet_type = name_ref U, %impl.elem0.loc9_11.2 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.193)]
// CHECK:STDOUT:       %U.as_type.loc9_11: type = facet_access_type %U.ref.loc9_11 [symbolic = %as_type (constants.%as_type.d83)]
// CHECK:STDOUT:       %.loc9_11.2: type = converted %U.ref.loc9_11, %U.as_type.loc9_11 [symbolic = %as_type (constants.%as_type.d83)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.WithSelf.F.%as_type (%as_type.d83) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.WithSelf.F.%as_type (%as_type.d83) = out_param call_param1
// CHECK:STDOUT:     %return: ref @J.WithSelf.F.%as_type (%as_type.d83) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.WithSelf.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc7
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.Op(@I.%Self.loc3: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.fa0)]
// CHECK:STDOUT:   %.loc4_33.1: Core.Form = init_form %Self.binding.as_type, call_param2 [symbolic = %.loc4_33.1 (constants.%.389)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type.d31), %b.param: @I.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type.d31)) -> out %return.param: @I.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type.d31);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self.loc7: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.8a1)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.89e)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.1: %facet_type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.1 (constants.%impl.elem0.193)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc9_11.1 [symbolic = %as_type (constants.%as_type.d83)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %as_type [symbolic = %pattern_type (constants.%pattern_type.d76)]
// CHECK:STDOUT:   %.loc9_17.1: Core.Form = init_form %as_type, call_param1 [symbolic = %.loc9_17.1 (constants.%.3f3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @J.WithSelf.F.%as_type (%as_type.d83)) -> out %return.param: @J.WithSelf.F.%as_type (%as_type.d83);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericResult(%T.loc12_18.2: %J.type) {
// CHECK:STDOUT:   %T.loc12_18.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_18.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc12_18.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc12_25: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc12_25 (constants.%pattern_type.14f)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc12_18.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.264)]
// CHECK:STDOUT:   %impl.elem0.loc12_35.1: %facet_type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.397)]
// CHECK:STDOUT:   %as_type.loc12_35.1: type = facet_access_type %impl.elem0.loc12_35.1 [symbolic = %as_type.loc12_35.1 (constants.%as_type.baf)]
// CHECK:STDOUT:   %pattern_type.loc12_31: type = pattern_type %as_type.loc12_35.1 [symbolic = %pattern_type.loc12_31 (constants.%pattern_type.bba)]
// CHECK:STDOUT:   %.loc12_43.2: Core.Form = init_form %as_type.loc12_35.1, call_param2 [symbolic = %.loc12_43.2 (constants.%.2cc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_26: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc12_26 (constants.%require_complete.19a)]
// CHECK:STDOUT:   %require_complete.loc12_32: <witness> = require_complete_type %as_type.loc12_35.1 [symbolic = %require_complete.loc12_32 (constants.%require_complete.9b3)]
// CHECK:STDOUT:   %J.WithSelf.F.type: type = fn_type @J.WithSelf.F, @J(%T.loc12_18.1) [symbolic = %J.WithSelf.F.type (constants.%J.WithSelf.F.type.41f)]
// CHECK:STDOUT:   %.loc13_11: type = fn_type_with_self_type %J.WithSelf.F.type, %T.loc12_18.1 [symbolic = %.loc13_11 (constants.%.43f)]
// CHECK:STDOUT:   %impl.elem1.loc13_11.2: @GenericResult.%.loc13_11 (%.43f) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc13_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_11.2: <specific function> = specific_impl_function %impl.elem1.loc13_11.2, @J.WithSelf.F(%T.loc12_18.1) [symbolic = %specific_impl_fn.loc13_11.2 (constants.%specific_impl_fn.a64)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc12_35.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet.loc13_16: %I.type = facet_value %as_type.loc12_35.1, (%I.lookup_impl_witness) [symbolic = %I.facet.loc13_16 (constants.%I.facet)]
// CHECK:STDOUT:   %I.WithSelf.Op.type: type = fn_type @I.WithSelf.Op, @I(%I.facet.loc13_16) [symbolic = %I.WithSelf.Op.type (constants.%I.WithSelf.Op.type.25c)]
// CHECK:STDOUT:   %.loc13_16: type = fn_type_with_self_type %I.WithSelf.Op.type, %I.facet.loc13_16 [symbolic = %.loc13_16 (constants.%.73b)]
// CHECK:STDOUT:   %impl.elem0.loc13_16.2: @GenericResult.%.loc13_16 (%.73b) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_16.2 (constants.%impl.elem0.d74)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_16.2: <specific function> = specific_impl_function %impl.elem0.loc13_16.2, @I.WithSelf.Op(%I.facet.loc13_16) [symbolic = %specific_impl_fn.loc13_16.2 (constants.%specific_impl_fn.b45)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc12_35.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.loc12_35.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.077)]
// CHECK:STDOUT:   %.loc13_29.6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc13_29.6 (constants.%.0e3)]
// CHECK:STDOUT:   %impl.elem0.loc13_29.2: @GenericResult.%.loc13_29.6 (%.0e3) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_29.2 (constants.%impl.elem0.4a3)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_29.2: <specific function> = specific_impl_function %impl.elem0.loc13_29.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc13_29.2 (constants.%specific_impl_fn.86e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericResult.%T.binding.as_type (%T.binding.as_type), %u.param: @GenericResult.%as_type.loc12_35.1 (%as_type.baf)) -> out %return.param: @GenericResult.%as_type.loc12_35.1 (%as_type.baf) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericResult.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref.loc13_11: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc13_11.1: @GenericResult.%.loc13_11 (%.43f) = impl_witness_access constants.%J.lookup_impl_witness.264, element1 [symbolic = %impl.elem1.loc13_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref.loc13_14: @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = name_ref u, %u
// CHECK:STDOUT:     %.loc13_15.1: %facet_type = converted constants.%as_type.baf, constants.%impl.elem0.397 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.397)]
// CHECK:STDOUT:     %.loc13_15.2: %facet_type = converted constants.%as_type.baf, constants.%impl.elem0.397 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.397)]
// CHECK:STDOUT:     %specific_impl_fn.loc13_11.1: <specific function> = specific_impl_function %impl.elem1.loc13_11.1, @J.WithSelf.F(constants.%T) [symbolic = %specific_impl_fn.loc13_11.2 (constants.%specific_impl_fn.a64)]
// CHECK:STDOUT:     %.loc13_15.3: ref @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = temporary_storage
// CHECK:STDOUT:     %J.WithSelf.F.call.loc13_15: init @GenericResult.%as_type.loc12_35.1 (%as_type.baf) to %.loc13_15.3 = call %specific_impl_fn.loc13_11.1(%u.ref.loc13_14)
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %Op.ref: %I.assoc_type = name_ref Op, @I.%assoc0 [concrete = constants.%assoc0.aa8]
// CHECK:STDOUT:     %impl.elem0.loc13_16.1: @GenericResult.%.loc13_16 (%.73b) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_16.2 (constants.%impl.elem0.d74)]
// CHECK:STDOUT:     %bound_method.loc13_16: <bound method> = bound_method %J.WithSelf.F.call.loc13_15, %impl.elem0.loc13_16.1
// CHECK:STDOUT:     %T.ref.loc13: %J.type = name_ref T, %T.loc12_18.2 [symbolic = %T.loc12_18.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc13: type = facet_access_type %T.ref.loc13 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc13_25: type = converted %T.ref.loc13, %T.as_type.loc13 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %F.ref.loc13_25: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc13_25: @GenericResult.%.loc13_11 (%.43f) = impl_witness_access constants.%J.lookup_impl_witness.264, element1 [symbolic = %impl.elem1.loc13_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref.loc13_28: @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = name_ref u, %u
// CHECK:STDOUT:     %.loc13_29.1: %facet_type = converted constants.%as_type.baf, constants.%impl.elem0.397 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.397)]
// CHECK:STDOUT:     %.loc13_29.2: %facet_type = converted constants.%as_type.baf, constants.%impl.elem0.397 [symbolic = %impl.elem0.loc12_35.1 (constants.%impl.elem0.397)]
// CHECK:STDOUT:     %specific_impl_fn.loc13_25: <specific function> = specific_impl_function %impl.elem1.loc13_25, @J.WithSelf.F(constants.%T) [symbolic = %specific_impl_fn.loc13_11.2 (constants.%specific_impl_fn.a64)]
// CHECK:STDOUT:     %.loc13_29.3: ref @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = temporary_storage
// CHECK:STDOUT:     %J.WithSelf.F.call.loc13_29: init @GenericResult.%as_type.loc12_35.1 (%as_type.baf) to %.loc13_29.3 = call %specific_impl_fn.loc13_25(%u.ref.loc13_28)
// CHECK:STDOUT:     %I.facet.loc13_30.1: %I.type = facet_value constants.%as_type.baf, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc13_16 (constants.%I.facet)]
// CHECK:STDOUT:     %.loc13_30.1: %I.type = converted constants.%as_type.baf, %I.facet.loc13_30.1 [symbolic = %I.facet.loc13_16 (constants.%I.facet)]
// CHECK:STDOUT:     %I.facet.loc13_30.2: %I.type = facet_value constants.%as_type.baf, (constants.%I.lookup_impl_witness) [symbolic = %I.facet.loc13_16 (constants.%I.facet)]
// CHECK:STDOUT:     %.loc13_30.2: %I.type = converted constants.%as_type.baf, %I.facet.loc13_30.2 [symbolic = %I.facet.loc13_16 (constants.%I.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc13_16.1: <specific function> = specific_impl_function %impl.elem0.loc13_16.1, @I.WithSelf.Op(constants.%I.facet) [symbolic = %specific_impl_fn.loc13_16.2 (constants.%specific_impl_fn.b45)]
// CHECK:STDOUT:     %bound_method.loc13_30: <bound method> = bound_method %J.WithSelf.F.call.loc13_15, %specific_impl_fn.loc13_16.1
// CHECK:STDOUT:     %.loc12_43.1: ref @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = splice_block %return.param {}
// CHECK:STDOUT:     %.loc13_15.4: ref @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = temporary %.loc13_15.3, %J.WithSelf.F.call.loc13_15
// CHECK:STDOUT:     %.loc13_15.5: @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = acquire_value %.loc13_15.4
// CHECK:STDOUT:     %.loc13_29.4: ref @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = temporary %.loc13_29.3, %J.WithSelf.F.call.loc13_29
// CHECK:STDOUT:     %.loc13_29.5: @GenericResult.%as_type.loc12_35.1 (%as_type.baf) = acquire_value %.loc13_29.4
// CHECK:STDOUT:     %I.WithSelf.Op.call: init @GenericResult.%as_type.loc12_35.1 (%as_type.baf) to %.loc12_43.1 = call %bound_method.loc13_30(%.loc13_15.5, %.loc13_29.5)
// CHECK:STDOUT:     %impl.elem0.loc13_29.1: @GenericResult.%.loc13_29.6 (%.0e3) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_29.2 (constants.%impl.elem0.4a3)]
// CHECK:STDOUT:     %bound_method.loc13_29.1: <bound method> = bound_method %.loc13_29.4, %impl.elem0.loc13_29.1
// CHECK:STDOUT:     %specific_impl_fn.loc13_29.1: <specific function> = specific_impl_function %impl.elem0.loc13_29.1, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc13_29.2 (constants.%specific_impl_fn.86e)]
// CHECK:STDOUT:     %bound_method.loc13_29.2: <bound method> = bound_method %.loc13_29.4, %specific_impl_fn.loc13_29.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc13_29: init %empty_tuple.type = call %bound_method.loc13_29.2(%.loc13_29.4)
// CHECK:STDOUT:     %impl.elem0.loc13_15: @GenericResult.%.loc13_29.6 (%.0e3) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_29.2 (constants.%impl.elem0.4a3)]
// CHECK:STDOUT:     %bound_method.loc13_15.1: <bound method> = bound_method %.loc13_15.4, %impl.elem0.loc13_15
// CHECK:STDOUT:     %specific_impl_fn.loc13_15: <specific function> = specific_impl_function %impl.elem0.loc13_15, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc13_29.2 (constants.%specific_impl_fn.86e)]
// CHECK:STDOUT:     %bound_method.loc13_15.2: <bound method> = bound_method %.loc13_15.4, %specific_impl_fn.loc13_15
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc13_15: init %empty_tuple.type = call %bound_method.loc13_15.2(%.loc13_15.4)
// CHECK:STDOUT:     return %I.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc4 => constants.%Self.ab9
// CHECK:STDOUT:   %I.WithSelf.Op.type => constants.%I.WithSelf.Op.type.71c
// CHECK:STDOUT:   %I.WithSelf.Op => constants.%I.WithSelf.Op.ae1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Op(constants.%Self.ab9) {
// CHECK:STDOUT:   %Self => constants.%Self.ab9
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.d31
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fa0
// CHECK:STDOUT:   %.loc4_33.1 => constants.%.389
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%Self.8a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc9 => constants.%Self.8a1
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.adb
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.569
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.8a1) {
// CHECK:STDOUT:   %Self => constants.%Self.8a1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.89e
// CHECK:STDOUT:   %impl.elem0.loc9_11.1 => constants.%impl.elem0.193
// CHECK:STDOUT:   %as_type => constants.%as_type.d83
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d76
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.3f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc9 => constants.%T
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.41f
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.f01
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericResult(constants.%T) {
// CHECK:STDOUT:   %T.loc12_18.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc12_25 => constants.%pattern_type.14f
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.264
// CHECK:STDOUT:   %impl.elem0.loc12_35.1 => constants.%impl.elem0.397
// CHECK:STDOUT:   %as_type.loc12_35.1 => constants.%as_type.baf
// CHECK:STDOUT:   %pattern_type.loc12_31 => constants.%pattern_type.bba
// CHECK:STDOUT:   %.loc12_43.2 => constants.%.2cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.264
// CHECK:STDOUT:   %impl.elem0.loc9_11.1 => constants.%impl.elem0.397
// CHECK:STDOUT:   %as_type => constants.%as_type.baf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bba
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.c35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc4 => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.Op.type => constants.%I.WithSelf.Op.type.25c
// CHECK:STDOUT:   %I.WithSelf.Op => constants.%I.WithSelf.Op.fdc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Op(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%as_type.baf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bba
// CHECK:STDOUT:   %.loc4_33.1 => constants.%.2cc
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
// CHECK:STDOUT:   %pattern_type.832: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.89e: <witness> = lookup_impl_witness %Self, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.335: type = impl_witness_access %J.lookup_impl_witness.89e, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c1a: type = pattern_type %impl.elem0.335 [symbolic]
// CHECK:STDOUT:   %.147: Core.Form = init_form %impl.elem0.335, call_param2 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.G.type.ace: type = fn_type @J.WithSelf.G, @J(%Self) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.G.6e7: %J.WithSelf.G.type.ace = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%J.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.14f: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %J.WithSelf.G.type.b2e: type = fn_type @J.WithSelf.G, @J(%T) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.G.310: %J.WithSelf.G.type.b2e = struct_value () [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.264: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.560: type = impl_witness_access %J.lookup_impl_witness.264, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ad6: type = pattern_type %impl.elem0.560 [symbolic]
// CHECK:STDOUT:   %.0b0: Core.Form = init_form %impl.elem0.560, call_param2 [symbolic]
// CHECK:STDOUT:   %GenericCallInterfaceQualified.type: type = fn_type @GenericCallInterfaceQualified [concrete]
// CHECK:STDOUT:   %GenericCallInterfaceQualified: %GenericCallInterfaceQualified.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.19a: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.06a: <witness> = require_complete_type %impl.elem0.560 [symbolic]
// CHECK:STDOUT:   %.298: type = fn_type_with_self_type %J.WithSelf.G.type.b2e, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.298 = impl_witness_access %J.lookup_impl_witness.264, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem1, @J.WithSelf.G(%T) [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.f76 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericCallInterfaceQualified.%pattern_type.loc8_41 (%pattern_type.14f) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericCallInterfaceQualified.%pattern_type.loc8_41 (%pattern_type.14f) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %u.patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.ad6) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.ad6) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.ad6) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @GenericCallInterfaceQualified.%pattern_type.loc8_47 (%pattern_type.ad6) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_58: %J.type = name_ref T, %T.loc8_34.2 [symbolic = %T.loc8_34.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc8_59: type = facet_access_type %T.ref.loc8_58 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc8_59.3: type = converted %T.ref.loc8_58, %T.as_type.loc8_59 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %U.ref.loc8_59: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_59: type = impl_witness_access constants.%J.lookup_impl_witness.264, element0 [symbolic = %impl.elem0.loc8_51.1 (constants.%impl.elem0.560)]
// CHECK:STDOUT:     %.loc8_59.4: Core.Form = init_form %impl.elem0.loc8_59, call_param2 [symbolic = %.loc8_59.2 (constants.%.0b0)]
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
// CHECK:STDOUT:     %u.param: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.560) = value_param call_param1
// CHECK:STDOUT:     %.loc8_51.1: type = splice_block %impl.elem0.loc8_51.2 [symbolic = %impl.elem0.loc8_51.1 (constants.%impl.elem0.560)] {
// CHECK:STDOUT:       %T.ref.loc8_50: %J.type = name_ref T, %T.loc8_34.2 [symbolic = %T.loc8_34.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8_51: type = facet_access_type %T.ref.loc8_50 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc8_51.2: type = converted %T.ref.loc8_50, %T.as_type.loc8_51 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %U.ref.loc8_51: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc8_51.2: type = impl_witness_access constants.%J.lookup_impl_witness.264, element0 [symbolic = %impl.elem0.loc8_51.1 (constants.%impl.elem0.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.560) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self.loc3: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.WithSelf.G.decl: @J.%J.WithSelf.G.type (%J.WithSelf.G.type.ace) = fn_decl @J.WithSelf.G [symbolic = @J.%J.WithSelf.G (constants.%J.WithSelf.G.6e7)] {
// CHECK:STDOUT:     %self.patt: @J.WithSelf.G.%pattern_type.loc5_8 (%pattern_type.832) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @J.WithSelf.G.%pattern_type.loc5_8 (%pattern_type.832) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: @J.WithSelf.G.%pattern_type.loc5_20 (%pattern_type.c1a) = value_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: @J.WithSelf.G.%pattern_type.loc5_20 (%pattern_type.c1a) = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @J.WithSelf.G.%pattern_type.loc5_20 (%pattern_type.c1a) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.WithSelf.G.%pattern_type.loc5_20 (%pattern_type.c1a) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %U.ref.loc5_29: type = name_ref U, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %.loc5_29.2: Core.Form = init_form %U.ref.loc5_29, call_param2 [symbolic = %.loc5_29.1 (constants.%.147)]
// CHECK:STDOUT:     %self.param: @J.WithSelf.G.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self.loc3 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J.WithSelf.G.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: @J.WithSelf.G.%impl.elem0.loc5_23.1 (%impl.elem0.335) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %U.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)] {
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:       %U.ref.loc5_23: type = name_ref U, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @J.WithSelf.G.%impl.elem0.loc5_23.1 (%impl.elem0.335) = value_binding v, %v.param
// CHECK:STDOUT:     %return.param: ref @J.WithSelf.G.%impl.elem0.loc5_23.1 (%impl.elem0.335) = out_param call_param2
// CHECK:STDOUT:     %return: ref @J.WithSelf.G.%impl.elem0.loc5_23.1 (%impl.elem0.335) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.WithSelf.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.G(@J.%Self.loc3: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.832)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.89e)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:   %pattern_type.loc5_20: type = pattern_type %impl.elem0.loc5_23.1 [symbolic = %pattern_type.loc5_20 (constants.%pattern_type.c1a)]
// CHECK:STDOUT:   %.loc5_29.1: Core.Form = init_form %impl.elem0.loc5_23.1, call_param2 [symbolic = %.loc5_29.1 (constants.%.147)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J.WithSelf.G.%Self.binding.as_type (%Self.binding.as_type), %v.param: @J.WithSelf.G.%impl.elem0.loc5_23.1 (%impl.elem0.335)) -> out %return.param: @J.WithSelf.G.%impl.elem0.loc5_23.1 (%impl.elem0.335);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallInterfaceQualified(%T.loc8_34.2: %J.type) {
// CHECK:STDOUT:   %T.loc8_34.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc8_34.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_34.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_41: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc8_41 (constants.%pattern_type.14f)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_34.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.264)]
// CHECK:STDOUT:   %impl.elem0.loc8_51.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_51.1 (constants.%impl.elem0.560)]
// CHECK:STDOUT:   %pattern_type.loc8_47: type = pattern_type %impl.elem0.loc8_51.1 [symbolic = %pattern_type.loc8_47 (constants.%pattern_type.ad6)]
// CHECK:STDOUT:   %.loc8_59.2: Core.Form = init_form %impl.elem0.loc8_51.1, call_param2 [symbolic = %.loc8_59.2 (constants.%.0b0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_42: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc8_42 (constants.%require_complete.19a)]
// CHECK:STDOUT:   %require_complete.loc8_48: <witness> = require_complete_type %impl.elem0.loc8_51.1 [symbolic = %require_complete.loc8_48 (constants.%require_complete.06a)]
// CHECK:STDOUT:   %J.WithSelf.G.type: type = fn_type @J.WithSelf.G, @J(%T.loc8_34.1) [symbolic = %J.WithSelf.G.type (constants.%J.WithSelf.G.type.b2e)]
// CHECK:STDOUT:   %.loc9: type = fn_type_with_self_type %J.WithSelf.G.type, %T.loc8_34.1 [symbolic = %.loc9 (constants.%.298)]
// CHECK:STDOUT:   %impl.elem1.loc9_11.2: @GenericCallInterfaceQualified.%.loc9 (%.298) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem1.loc9_11.2, @J.WithSelf.G(%T.loc8_34.1) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericCallInterfaceQualified.%T.binding.as_type (%T.binding.as_type), %u.param: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.560)) -> out %return.param: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.560) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallInterfaceQualified.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %J.ref.loc9: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %G.ref: %J.assoc_type = name_ref G, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc9_11.1: @GenericCallInterfaceQualified.%.loc9 (%.298) = impl_witness_access constants.%J.lookup_impl_witness.264, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc9_11: <bound method> = bound_method %t.ref, %impl.elem1.loc9_11.1
// CHECK:STDOUT:     %u.ref: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.560) = name_ref u, %u
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem1.loc9_11.1, @J.WithSelf.G(constants.%T) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc9_19: <bound method> = bound_method %t.ref, %specific_impl_fn.loc9_11.1
// CHECK:STDOUT:     %.loc8_59.1: ref @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.560) = splice_block %return.param {}
// CHECK:STDOUT:     %J.WithSelf.G.call: init @GenericCallInterfaceQualified.%impl.elem0.loc8_51.1 (%impl.elem0.560) to %.loc8_59.1 = call %bound_method.loc9_19(%t.ref, %u.ref)
// CHECK:STDOUT:     return %J.WithSelf.G.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%Self
// CHECK:STDOUT:   %J.WithSelf.G.type => constants.%J.WithSelf.G.type.ace
// CHECK:STDOUT:   %J.WithSelf.G => constants.%J.WithSelf.G.6e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.832
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.89e
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.335
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.c1a
// CHECK:STDOUT:   %.loc5_29.1 => constants.%.147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%T
// CHECK:STDOUT:   %J.WithSelf.G.type => constants.%J.WithSelf.G.type.b2e
// CHECK:STDOUT:   %J.WithSelf.G => constants.%J.WithSelf.G.310
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallInterfaceQualified(constants.%T) {
// CHECK:STDOUT:   %T.loc8_34.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_41 => constants.%pattern_type.14f
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.264
// CHECK:STDOUT:   %impl.elem0.loc8_51.1 => constants.%impl.elem0.560
// CHECK:STDOUT:   %pattern_type.loc8_47 => constants.%pattern_type.ad6
// CHECK:STDOUT:   %.loc8_59.2 => constants.%.0b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.G(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.14f
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.264
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.560
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.ad6
// CHECK:STDOUT:   %.loc5_29.1 => constants.%.0b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.ebd: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %J.lookup_impl_witness.89e: <witness> = lookup_impl_witness %Self.8a1, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.335: type = impl_witness_access %J.lookup_impl_witness.89e, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c1a: type = pattern_type %impl.elem0.335 [symbolic]
// CHECK:STDOUT:   %.a6a: Core.Form = init_form %impl.elem0.335, call_param1 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.adb: type = fn_type @J.WithSelf.F, @J(%Self.8a1) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.569: %J.WithSelf.F.type.adb = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.2fa: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.type.e19: type = fn_type @J.WithSelf.F, @J(%.Self.2fa) [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.659: %J.WithSelf.F.type.e19 = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.2fa [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.46a: <witness> = lookup_impl_witness %.Self.2fa, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a58: type = impl_witness_access %J.lookup_impl_witness.46a, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.a58 = %i32> [concrete]
// CHECK:STDOUT:   %T: %J_where.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6f0: type = pattern_type %J_where.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.12a: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.e54: Core.Form = init_form %i32, call_param1 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %GenericCallFI32.type: type = fn_type @GenericCallFI32 [concrete]
// CHECK:STDOUT:   %GenericCallFI32: %GenericCallFI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.226: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.4bb: type = fn_type @J.WithSelf.F, @J(%T) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.3c9: %J.WithSelf.F.type.4bb = struct_value () [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.74d: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.binding.as_type, (%J.lookup_impl_witness.74d) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.606: type = fn_type @J.WithSelf.F, @J(%J.facet) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.459: %J.WithSelf.F.type.606 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3ee: type = fn_type_with_self_type %J.WithSelf.F.type.606, %J.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.3ee = impl_witness_access %J.lookup_impl_witness.74d, element1 [symbolic]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem1, @J.WithSelf.F(%J.facet) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.6f0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @GenericCallFI32.%pattern_type (%pattern_type.12a) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @GenericCallFI32.%pattern_type (%pattern_type.12a) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc8_51: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_51: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_51: Core.Form = init_form %i32.loc8_51, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %.loc8_26.1: type = splice_block %.loc8_26.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.2: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.2fa]
// CHECK:STDOUT:       %.Self.ref: %J.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.2fa]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc8_32: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:       %impl.elem0.loc8: type = impl_witness_access constants.%J.lookup_impl_witness.46a, element0 [symbolic_self = constants.%impl.elem0.a58]
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
// CHECK:STDOUT:   %Self.loc3: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.%J.WithSelf.F.type (%J.WithSelf.F.type.adb) = fn_decl @J.WithSelf.F [symbolic = @J.%J.WithSelf.F (constants.%J.WithSelf.F.569)] {
// CHECK:STDOUT:     %u.patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %.loc5_17.2: Core.Form = init_form %U.ref.loc5_17, call_param1 [symbolic = %.loc5_17.1 (constants.%.a6a)]
// CHECK:STDOUT:     %u.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = value_param call_param0
// CHECK:STDOUT:     %.loc5_11: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = out_param call_param1
// CHECK:STDOUT:     %return: ref @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.WithSelf.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self.loc3: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.8a1)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.89e)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.c1a)]
// CHECK:STDOUT:   %.loc5_17.1: Core.Form = init_form %impl.elem0.loc5_11.1, call_param1 [symbolic = %.loc5_17.1 (constants.%.a6a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335)) -> out %return.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallFI32(%T.loc8_20.2: %J_where.type) {
// CHECK:STDOUT:   %T.loc8_20.1: %J_where.type = symbolic_binding T, 0 [symbolic = %T.loc8_20.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_20.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.12a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.226)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_20.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.74d)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.binding.as_type, (%J.lookup_impl_witness) [symbolic = %J.facet (constants.%J.facet)]
// CHECK:STDOUT:   %J.WithSelf.F.type: type = fn_type @J.WithSelf.F, @J(%J.facet) [symbolic = %J.WithSelf.F.type (constants.%J.WithSelf.F.type.606)]
// CHECK:STDOUT:   %.loc9_11: type = fn_type_with_self_type %J.WithSelf.F.type, %J.facet [symbolic = %.loc9_11 (constants.%.3ee)]
// CHECK:STDOUT:   %impl.elem1.loc9_11.2: @GenericCallFI32.%.loc9_11 (%.3ee) = impl_witness_access %J.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem1.loc9_11.2, @J.WithSelf.F(%J.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @GenericCallFI32.%T.binding.as_type (%T.binding.as_type)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallFI32.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc9_11.1: @GenericCallFI32.%.loc9_11 (%.3ee) = impl_witness_access constants.%J.lookup_impl_witness.74d, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem1.loc9_11.1, @J.WithSelf.F(constants.%J.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %impl.elem0.loc9: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc9_14.1: <bound method> = bound_method %int_2, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_14.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_14.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_14.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_14.2: %i32 = converted %int_2, %.loc9_14.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %J.WithSelf.F.call: init %i32 = call %specific_impl_fn.loc9_11.1(%.loc9_14.2)
// CHECK:STDOUT:     return %J.WithSelf.F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%Self.8a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%Self.8a1
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.adb
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.569
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.8a1) {
// CHECK:STDOUT:   %Self => constants.%Self.8a1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.89e
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.335
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1a
// CHECK:STDOUT:   %.loc5_17.1 => constants.%.a6a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%.Self.2fa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%.Self.2fa
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.e19
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.659
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallFI32(constants.%T) {
// CHECK:STDOUT:   %T.loc8_20.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.12a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%T
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.4bb
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.3c9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%J.facet
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.606
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.459
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.74d
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:   %.loc5_17.1 => constants.%.e54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_associated_constant_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.ebd: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %J.lookup_impl_witness.89e: <witness> = lookup_impl_witness %Self.8a1, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.335: type = impl_witness_access %J.lookup_impl_witness.89e, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c1a: type = pattern_type %impl.elem0.335 [symbolic]
// CHECK:STDOUT:   %.a6a: Core.Form = init_form %impl.elem0.335, call_param1 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.adb: type = fn_type @J.WithSelf.F, @J(%Self.8a1) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.569: %J.WithSelf.F.type.adb = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %.Self: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.type.e19: type = fn_type @J.WithSelf.F, @J(%.Self) [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.659: %J.WithSelf.F.type.e19 = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.46a: <witness> = lookup_impl_witness %.Self, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a58: type = impl_witness_access %J.lookup_impl_witness.46a, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.a58 = %i32> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type.4b5c2a.1: type = fn_type @E.as.J.impl.F.loc27_21.1 [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.b4e6ab.1: %E.as.J.impl.F.type.4b5c2a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.a23: type = fn_type @J.WithSelf.F, @J(%J.facet) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.4a4: %J.WithSelf.F.type.a23 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.e54: Core.Form = init_form %i32, call_param1 [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type.4b5c2a.2: type = fn_type @E.as.J.impl.F.loc27_21.2 [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.b4e6ab.2: %E.as.J.impl.F.type.4b5c2a.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Self.loc3: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.%J.WithSelf.F.type (%J.WithSelf.F.type.adb) = fn_decl @J.WithSelf.F [symbolic = @J.%J.WithSelf.F (constants.%J.WithSelf.F.569)] {
// CHECK:STDOUT:     %u.patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.WithSelf.F.%pattern_type (%pattern_type.c1a) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     %.loc5_17.2: Core.Form = init_form %U.ref.loc5_17, call_param1 [symbolic = %.loc5_17.1 (constants.%.a6a)]
// CHECK:STDOUT:     %u.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = value_param call_param0
// CHECK:STDOUT:     %.loc5_11: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)] {
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access constants.%J.lookup_impl_witness.89e, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = out_param call_param1
// CHECK:STDOUT:     %return: ref @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.WithSelf.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %J.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.J.impl: %Self.ref as %.loc9_20 {
// CHECK:STDOUT:   %E.as.J.impl.F.decl.loc27_21.1: %E.as.J.impl.F.type.4b5c2a.1 = fn_decl @E.as.J.impl.F.loc27_21.1 [concrete = constants.%E.as.J.impl.F.b4e6ab.1] {
// CHECK:STDOUT:     %u.patt: <error> = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: <error> = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc27_19: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:     %.loc27_19: type = converted %U.ref.loc27_19, <error> [concrete = <error>]
// CHECK:STDOUT:     %u.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %U.ref.loc27_13: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:       %.loc27_13: type = converted %U.ref.loc27_13, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: <error> = value_binding u, %u.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_17.1: type = specific_constant @J.WithSelf.F.%U.ref.loc5_17, @J.WithSelf.F(constants.%J.facet) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5_17.2: type = specific_constant @J.WithSelf.F.%.loc5_17.2, @J.WithSelf.F(constants.%J.facet) [concrete = constants.%.e54]
// CHECK:STDOUT:   %E.as.J.impl.F.decl.loc27_21.2: %E.as.J.impl.F.type.4b5c2a.2 = fn_decl @E.as.J.impl.F.loc27_21.2 [concrete = constants.%E.as.J.impl.F.b4e6ab.2] {
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
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %E.as.J.impl.F.decl.loc27_21.2), @E.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .U = <poisoned>
// CHECK:STDOUT:   .F = %E.as.J.impl.F.decl.loc27_21.1
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @E.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc9_26: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.ebd]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness.46a, element0 [symbolic_self = constants.%impl.elem0.a58]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc9_20: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self.loc3: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.8a1)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.89e)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.335)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc5_11.1 [symbolic = %pattern_type (constants.%pattern_type.c1a)]
// CHECK:STDOUT:   %.loc5_17.1: Core.Form = init_form %impl.elem0.loc5_11.1, call_param1 [symbolic = %.loc5_17.1 (constants.%.a6a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335)) -> out %return.param: @J.WithSelf.F.%impl.elem0.loc5_11.1 (%impl.elem0.335);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.F.loc27_21.1(%u.param: <error>) -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: <error> = name_ref u, %u
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.F.loc27_21.2(%u.param: %i32) -> out %return.param: %i32 [thunk @E.as.J.impl.%E.as.J.impl.F.decl.loc27_21.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %E.as.J.impl.F.type.4b5c2a.1 = name_ref F, @E.as.J.impl.%E.as.J.impl.F.decl.loc27_21.1 [concrete = constants.%E.as.J.impl.F.b4e6ab.1]
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u.param
// CHECK:STDOUT:   %return.ref: ref %i32 = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %E.as.J.impl.F.call: <error> = call %F.ref(<error>)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%Self.8a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%Self.8a1
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.adb
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.569
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.8a1) {
// CHECK:STDOUT:   %Self => constants.%Self.8a1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.89e
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.335
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1a
// CHECK:STDOUT:   %.loc5_17.1 => constants.%.a6a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%.Self
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.e19
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.659
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%J.facet
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.a23
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.4a4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:   %.loc5_17.1 => constants.%.e54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_call_extend_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.adb: type = fn_type @J.WithSelf.F, @J(%Self) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.569: %J.WithSelf.F.type.adb = struct_value () [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type: type = fn_type @E.as.J.impl.F [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F: %E.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.59e: type = fn_type @J.WithSelf.F, @J(%J.facet) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.d37: %J.WithSelf.F.type.59e = struct_value () [concrete]
// CHECK:STDOUT:   %E.G.type: type = fn_type @E.G [concrete]
// CHECK:STDOUT:   %E.G: %E.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self.loc3: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.%J.WithSelf.F.type (%J.WithSelf.F.type.adb) = fn_decl @J.WithSelf.F [symbolic = @J.%J.WithSelf.F (constants.%J.WithSelf.F.569)] {} {}
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%J.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %E.as.J.impl.F.decl: %E.as.J.impl.F.type = fn_decl @E.as.J.impl.F [concrete = constants.%E.as.J.impl.F] {} {}
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%E.as.J.impl.F.decl), @E.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %E.as.J.impl.F.decl
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @E.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.G.decl: %E.G.type = fn_decl @E.G [concrete = constants.%E.G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .G = %E.G.decl
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend @E.as.J.impl.%J.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self.loc3: %J.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %J.assoc_type = name_ref F, @J.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc4 => constants.%Self
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.adb
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.569
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc4 => constants.%J.facet
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.59e
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.d37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_extend_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.832: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.adb: type = fn_type @J.WithSelf.F, @J(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.569: %J.WithSelf.F.type.adb = struct_value () [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.99f: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type: type = fn_type @E.as.J.impl.F [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F: %E.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.59e: type = fn_type @J.WithSelf.F, @J(%J.facet) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.d37: %J.WithSelf.F.type.59e = struct_value () [concrete]
// CHECK:STDOUT:   %E.G.type: type = fn_type @E.G [concrete]
// CHECK:STDOUT:   %E.G: %E.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %E.type.facet: %type = facet_value %E, () [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.d1c: type = fn_type @J.WithSelf.F, @J(%E.type.facet) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.1a6: %J.WithSelf.F.type.d1c = struct_value () [concrete]
// CHECK:STDOUT:   %.8de: type = fn_type_with_self_type %J.WithSelf.F.type.59e, %J.facet [concrete]
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
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self.loc3: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.%J.WithSelf.F.type (%J.WithSelf.F.type.adb) = fn_decl @J.WithSelf.F [symbolic = @J.%J.WithSelf.F (constants.%J.WithSelf.F.569)] {
// CHECK:STDOUT:     %self.patt: @J.WithSelf.F.%pattern_type (%pattern_type.832) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @J.WithSelf.F.%pattern_type (%pattern_type.832) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @J.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_21.1: type = splice_block %.loc4_21.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self.loc3 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc4_21.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%J.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %E.as.J.impl.F.decl: %E.as.J.impl.F.type = fn_decl @E.as.J.impl.F [concrete = constants.%E.as.J.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.99f = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.99f = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %E = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %self: %E = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%E.as.J.impl.F.decl), @E.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %E.as.J.impl.F.decl
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @E.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.G.decl: %E.G.type = fn_decl @E.G [concrete = constants.%E.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.99f = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.99f = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %E = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %self: %E = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .G = %E.G.decl
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend @E.as.J.impl.%J.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self.loc3: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.832)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.F(%self.param: %E) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.G(%self.param: %E) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %E = name_ref self, %self
// CHECK:STDOUT:   %F.ref: %J.assoc_type = name_ref F, @J.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.8de = impl_witness_access constants.%J.impl_witness, element0 [concrete = constants.%E.as.J.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %self.ref, %impl.elem0
// CHECK:STDOUT:   %E.as.J.impl.F.call: init %empty_tuple.type = call %bound_method(%self.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc4 => constants.%Self
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.adb
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.569
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.832
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc4 => constants.%J.facet
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.59e
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.d37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%E
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.99f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%E.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc4 => constants.%E.type.facet
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.d1c
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.1a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_extend_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.df0: %J.assoc_type = assoc_entity element0, @J.%X [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %.Self: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %E.G.type: type = fn_type @E.G [concrete]
// CHECK:STDOUT:   %E.G: %E.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%X [concrete = constants.%assoc0.df0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (%X)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.J.impl: %Self.ref as %.loc8_20 {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @E.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @E.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_26: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %X.ref: %J.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.df0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_32.2: type = converted %.loc8_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_20: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.G.decl: %E.G.type = fn_decl @E.G [concrete = constants.%E.G] {} {
// CHECK:STDOUT:     %X.ref: %J.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.df0]
// CHECK:STDOUT:     %.loc18_13: type = converted %X.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .G = %E.G.decl
// CHECK:STDOUT:   extend @E.as.J.impl.%.loc8_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.G() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_25: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%Self.8a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%J.facet) {
// CHECK:STDOUT: !definition:
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
// CHECK:STDOUT:   %pattern_type.21d: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %J2.WithSelf.F.type.68c: type = fn_type @J2.WithSelf.F, @J2(%Self) [symbolic]
// CHECK:STDOUT:   %J2.WithSelf.F.643: %J2.WithSelf.F.type.68c = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %J2.assoc_type = assoc_entity element1, @J2.%J2.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self: %J2.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %J2.WithSelf.F.type.30c: type = fn_type @J2.WithSelf.F, @J2(%.Self) [symbolic_self]
// CHECK:STDOUT:   %J2.WithSelf.F.99a: %J2.WithSelf.F.type.30c = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %J2.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @J2 [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %J2.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %J2_where.type.1ad: type = facet_type <@J2 where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %J2.impl_witness.941: <witness> = impl_witness @empty_tuple.type.as.J2.impl.%J2.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %.0df: Core.Form = init_form %empty_struct_type, call_param2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.type.56bdaf.1: type = fn_type @empty_tuple.type.as.J2.impl.F.loc23_40.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.7b8679.1: %empty_tuple.type.as.J2.impl.F.type.56bdaf.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J2.facet.6c9: %J2.type = facet_value %empty_tuple.type, (%J2.impl_witness.941) [concrete]
// CHECK:STDOUT:   %J2.WithSelf.F.type.43a: type = fn_type @J2.WithSelf.F, @J2(%J2.facet.6c9) [concrete]
// CHECK:STDOUT:   %J2.WithSelf.F.2bb: %J2.WithSelf.F.type.43a = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.type.56bdaf.2: type = fn_type @empty_tuple.type.as.J2.impl.F.loc23_40.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.7b8679.2: %empty_tuple.type.as.J2.impl.F.type.56bdaf.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J2_where.type.de0: type = facet_type <@J2 where %impl.elem0 = %C2> [concrete]
// CHECK:STDOUT:   %J2.impl_witness.30f: <witness> = impl_witness @C2.as.J2.impl.%J2.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.8df: type = pattern_type %C2 [concrete]
// CHECK:STDOUT:   %.704: Core.Form = init_form %C2, call_param2 [concrete]
// CHECK:STDOUT:   %C2.as.J2.impl.F.type.d2a210.1: type = fn_type @C2.as.J2.impl.F.loc32_40.1 [concrete]
// CHECK:STDOUT:   %C2.as.J2.impl.F.720bb2.1: %C2.as.J2.impl.F.type.d2a210.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J2.facet.f19: %J2.type = facet_value %C2, (%J2.impl_witness.30f) [concrete]
// CHECK:STDOUT:   %J2.WithSelf.F.type.3cd: type = fn_type @J2.WithSelf.F, @J2(%J2.facet.f19) [concrete]
// CHECK:STDOUT:   %J2.WithSelf.F.06e: %J2.WithSelf.F.type.3cd = struct_value () [concrete]
// CHECK:STDOUT:   %C2.as.J2.impl.F.type.d2a210.2: type = fn_type @C2.as.J2.impl.F.loc32_40.2 [concrete]
// CHECK:STDOUT:   %C2.as.J2.impl.F.720bb2.2: %C2.as.J2.impl.F.type.d2a210.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %.loc22_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc22_7.2: type = converted %.loc22_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J2.ref: type = name_ref J2, file.%J2.decl [concrete = constants.%J2.type]
// CHECK:STDOUT:     %.Self: %J2.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc22_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %U2.ref: %J2.assoc_type = name_ref U2, @U2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J2.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc22_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc22_28.2: type = converted %.loc22_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc22_15: type = where_expr %.Self [concrete = constants.%J2_where.type.1ad] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J2.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc22_28.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [concrete = constants.%C2] {} {}
// CHECK:STDOUT:   impl_decl @C2.as.J2.impl [concrete] {} {
// CHECK:STDOUT:     %C2.ref.loc31_6: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %J2.ref: type = name_ref J2, file.%J2.decl [concrete = constants.%J2.type]
// CHECK:STDOUT:     %.Self: %J2.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc31_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %U2.ref: %J2.assoc_type = name_ref U2, @U2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J2.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C2.ref.loc31_27: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %.loc31_15: type = where_expr %.Self [concrete = constants.%J2_where.type.de0] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J2.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C2.ref.loc31_27
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J2 {
// CHECK:STDOUT:   %Self.loc3: %J2.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @J2 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %U2: type = assoc_const_decl @U2 [concrete] {
// CHECK:STDOUT:     %assoc0: %J2.assoc_type = assoc_entity element0, @J2.%U2 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J2.WithSelf.F.decl: @J2.%J2.WithSelf.F.type (%J2.WithSelf.F.type.68c) = fn_decl @J2.WithSelf.F [symbolic = @J2.%J2.WithSelf.F (constants.%J2.WithSelf.F.643)] {
// CHECK:STDOUT:     %self.patt: @J2.WithSelf.F.%pattern_type (%pattern_type.21d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @J2.WithSelf.F.%pattern_type (%pattern_type.21d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %z.patt: <error> = value_binding_pattern z [concrete]
// CHECK:STDOUT:     %z.param_patt: <error> = value_param_pattern %z.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc19_35: %J2.type = name_ref Self, @J2.%Self.loc3 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc19_39: type = facet_access_type %Self.ref.loc19_35 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc19_39: type = converted %Self.ref.loc19_35, %Self.as_type.loc19_39 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %U2.ref.loc19_39: <error> = name_ref U2, <error> [concrete = <error>]
// CHECK:STDOUT:     %self.param: @J2.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc19_14.1: type = splice_block %.loc19_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc19_14: %J2.type = name_ref Self, @J2.%Self.loc3 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc19_14: type = facet_access_type %Self.ref.loc19_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc19_14.2: type = converted %Self.ref.loc19_14, %Self.as_type.loc19_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J2.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     %z.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %Self.ref.loc19_23: %J2.type = name_ref Self, @J2.%Self.loc3 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc19_27: type = facet_access_type %Self.ref.loc19_23 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc19_27: type = converted %Self.ref.loc19_23, %Self.as_type.loc19_27 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %U2.ref.loc19_27: <error> = name_ref U2, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %z: <error> = value_binding z, %z.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J2.assoc_type = assoc_entity element1, %J2.WithSelf.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .U2 = @U2.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U2, %J2.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.J2.impl: %.loc22_7.2 as %.loc22_15 {
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.decl.loc23_40.1: %empty_tuple.type.as.J2.impl.F.type.56bdaf.1 = fn_decl @empty_tuple.type.as.J2.impl.F.loc23_40.1 [concrete = constants.%empty_tuple.type.as.J2.impl.F.7b8679.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %z.patt: %pattern_type.a96 = value_binding_pattern z [concrete]
// CHECK:STDOUT:     %z.param_patt: %pattern_type.a96 = value_param_pattern %z.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc23_38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc23_38.2: type = converted %.loc23_38.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc23_38.3: Core.Form = init_form %.loc23_38.2, call_param2 [concrete = constants.%.0df]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.J2.impl.%.loc22_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %z.param: %empty_struct_type = value_param call_param1
// CHECK:STDOUT:     %.loc23_31.1: type = splice_block %.loc23_31.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc23_31.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc23_31.3: type = converted %.loc23_31.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %z: %empty_struct_type = value_binding z, %z.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param2
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: type = specific_constant <error>, @J2.WithSelf.F(constants.%J2.facet.6c9) [concrete = <error>]
// CHECK:STDOUT:   %.2: type = specific_constant <error>, @J2.WithSelf.F(constants.%J2.facet.6c9) [concrete = <error>]
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.decl.loc23_40.2: %empty_tuple.type.as.J2.impl.F.type.56bdaf.2 = fn_decl @empty_tuple.type.as.J2.impl.F.loc23_40.2 [concrete = constants.%empty_tuple.type.as.J2.impl.F.7b8679.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.param_patt: <error> = value_param_pattern <error>, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %.param: <error> = value_param call_param1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J2.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %empty_tuple.type.as.J2.impl.F.decl.loc23_40.2), @empty_tuple.type.as.J2.impl [concrete]
// CHECK:STDOUT:   %J2.impl_witness: <witness> = impl_witness %J2.impl_witness_table [concrete = constants.%J2.impl_witness.941]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.J2.impl.F.decl.loc23_40.1
// CHECK:STDOUT:   witness = %J2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C2.as.J2.impl: %C2.ref.loc31_6 as %.loc31_15 {
// CHECK:STDOUT:   %C2.as.J2.impl.F.decl.loc32_40.1: %C2.as.J2.impl.F.type.d2a210.1 = fn_decl @C2.as.J2.impl.F.loc32_40.1 [concrete = constants.%C2.as.J2.impl.F.720bb2.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.8df = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8df = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %z.patt: %pattern_type.8df = value_binding_pattern z [concrete]
// CHECK:STDOUT:     %z.param_patt: %pattern_type.8df = value_param_pattern %z.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.8df = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8df = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C2.ref.loc32_37: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %.loc32_37: Core.Form = init_form %C2.ref.loc32_37, call_param2 [concrete = constants.%.704]
// CHECK:STDOUT:     %self.param: %C2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C2.as.J2.impl.%C2.ref.loc31_6 [concrete = constants.%C2]
// CHECK:STDOUT:     %self: %C2 = value_binding self, %self.param
// CHECK:STDOUT:     %z.param: %C2 = value_param call_param1
// CHECK:STDOUT:     %C2.ref.loc32_30: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %z: %C2 = value_binding z, %z.param
// CHECK:STDOUT:     %return.param: ref %C2 = out_param call_param2
// CHECK:STDOUT:     %return: ref %C2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: type = specific_constant <error>, @J2.WithSelf.F(constants.%J2.facet.f19) [concrete = <error>]
// CHECK:STDOUT:   %.2: type = specific_constant <error>, @J2.WithSelf.F(constants.%J2.facet.f19) [concrete = <error>]
// CHECK:STDOUT:   %C2.as.J2.impl.F.decl.loc32_40.2: %C2.as.J2.impl.F.type.d2a210.2 = fn_decl @C2.as.J2.impl.F.loc32_40.2 [concrete = constants.%C2.as.J2.impl.F.720bb2.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.8df = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8df = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.param_patt: <error> = value_param_pattern <error>, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C2 = value_param call_param0
// CHECK:STDOUT:     %self: %C2 = value_binding self, %self.param
// CHECK:STDOUT:     %.param: <error> = value_param call_param1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J2.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %C2.as.J2.impl.F.decl.loc32_40.2), @C2.as.J2.impl [concrete]
// CHECK:STDOUT:   %J2.impl_witness: <witness> = impl_witness %J2.impl_witness_table [concrete = constants.%J2.impl_witness.30f]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C2 [concrete = constants.%C2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C2 = <poisoned>
// CHECK:STDOUT:   .F = %C2.as.J2.impl.F.decl.loc32_40.1
// CHECK:STDOUT:   witness = %J2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %.loc28_10: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc28_11: type = converted %.loc28_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc28_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J2.WithSelf.F(@J2.%Self.loc3: %J2.type) {
// CHECK:STDOUT:   %Self: %J2.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.21d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J2.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type), %z.param: <error>) -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.J2.impl.F.loc23_40.1(%self.param: %empty_tuple.type, %z.param: %empty_struct_type) -> out %return.param: %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %z.ref: %empty_struct_type = name_ref z, %z
// CHECK:STDOUT:   %.loc23_49: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc23_50: init %empty_struct_type = converted %z.ref, %.loc23_49 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc23_50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.J2.impl.F.loc23_40.2(%self.param: %empty_tuple.type, %.param: <error>) -> <error> [thunk @empty_tuple.type.as.J2.impl.%empty_tuple.type.as.J2.impl.F.decl.loc23_40.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.J2.impl.F.type.56bdaf.1 = name_ref F, @empty_tuple.type.as.J2.impl.%empty_tuple.type.as.J2.impl.F.decl.loc23_40.1 [concrete = constants.%empty_tuple.type.as.J2.impl.F.7b8679.1]
// CHECK:STDOUT:   %self.ref: %empty_tuple.type = name_ref self, %self.param
// CHECK:STDOUT:   %.ref: <error> = name_ref <none>, %.param
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   %empty_tuple.type.as.J2.impl.F.call: init %empty_struct_type = call %empty_tuple.type.as.J2.impl.F.bound(%self.ref, <error>)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C2.as.J2.impl.F.loc32_40.1(%self.param: %C2, %z.param: %C2) -> out %return.param: %C2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C2 = name_ref self, %self
// CHECK:STDOUT:   %.loc32_53.1: %empty_struct_type = as_compatible %self.ref
// CHECK:STDOUT:   %.loc32_53.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_53.3: init %C2 = as_compatible %.loc32_53.2 [concrete = constants.%C2.val]
// CHECK:STDOUT:   %.loc32_53.4: init %C2 = converted %self.ref, %.loc32_53.3 [concrete = constants.%C2.val]
// CHECK:STDOUT:   return %.loc32_53.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C2.as.J2.impl.F.loc32_40.2(%self.param: %C2, %.param: <error>) -> <error> [thunk @C2.as.J2.impl.%C2.as.J2.impl.F.decl.loc32_40.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %C2.as.J2.impl.F.type.d2a210.1 = name_ref F, @C2.as.J2.impl.%C2.as.J2.impl.F.decl.loc32_40.1 [concrete = constants.%C2.as.J2.impl.F.720bb2.1]
// CHECK:STDOUT:   %self.ref: %C2 = name_ref self, %self.param
// CHECK:STDOUT:   %.ref: <error> = name_ref <none>, %.param
// CHECK:STDOUT:   %C2.as.J2.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   %C2.as.J2.impl.F.call: init %C2 = call %C2.as.J2.impl.F.bound(%self.ref, <error>)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc19 => constants.%Self
// CHECK:STDOUT:   %J2.WithSelf.F.type => constants.%J2.WithSelf.F.type.68c
// CHECK:STDOUT:   %J2.WithSelf.F => constants.%J2.WithSelf.F.643
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.21d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc19 => constants.%.Self
// CHECK:STDOUT:   %J2.WithSelf.F.type => constants.%J2.WithSelf.F.type.30c
// CHECK:STDOUT:   %J2.WithSelf.F => constants.%J2.WithSelf.F.99a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2(constants.%J2.facet.6c9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc19 => constants.%J2.facet.6c9
// CHECK:STDOUT:   %J2.WithSelf.F.type => constants.%J2.WithSelf.F.type.43a
// CHECK:STDOUT:   %J2.WithSelf.F => constants.%J2.WithSelf.F.2bb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2.WithSelf.F(constants.%J2.facet.6c9) {
// CHECK:STDOUT:   %Self => constants.%J2.facet.6c9
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2(constants.%J2.facet.f19) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc19 => constants.%J2.facet.f19
// CHECK:STDOUT:   %J2.WithSelf.F.type => constants.%J2.WithSelf.F.type.3cd
// CHECK:STDOUT:   %J2.WithSelf.F => constants.%J2.WithSelf.F.06e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2.WithSelf.F(constants.%J2.facet.f19) {
// CHECK:STDOUT:   %Self => constants.%J2.facet.f19
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_associated_type_in_signature_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self.44d: %K.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type @K [concrete]
// CHECK:STDOUT:   %assoc0.6cb: %K.assoc_type = assoc_entity element0, @K.%V [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.87a: type = symbolic_binding_type Self, 0, %Self.44d [symbolic]
// CHECK:STDOUT:   %pattern_type.bcd: type = pattern_type %Self.binding.as_type.87a [symbolic]
// CHECK:STDOUT:   %K.lookup_impl_witness.a24: <witness> = lookup_impl_witness %Self.44d, @K [symbolic]
// CHECK:STDOUT:   %impl.elem0.c23: type = impl_witness_access %K.lookup_impl_witness.a24, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9fa: type = pattern_type %impl.elem0.c23 [symbolic]
// CHECK:STDOUT:   %.bf0: Core.Form = init_form %impl.elem0.c23, call_param2 [symbolic]
// CHECK:STDOUT:   %K.WithSelf.F.type.d74: type = fn_type @K.WithSelf.F, @K(%Self.44d) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %K.WithSelf.F.f53: %K.WithSelf.F.type.d74 = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %K.assoc_type = assoc_entity element1, @K.%K.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self: %K.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %K.WithSelf.F.type.262: type = fn_type @K.WithSelf.F, @K(%.Self) [symbolic_self]
// CHECK:STDOUT:   %K.WithSelf.F.40c: %K.WithSelf.F.type.262 = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %K.lookup_impl_witness.28c: <witness> = lookup_impl_witness %.Self, @K [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.afd: type = impl_witness_access %K.lookup_impl_witness.28c, element0 [symbolic_self]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %K_where.type: type = facet_type <@K where %impl.elem0.afd = %struct_type.a> [concrete]
// CHECK:STDOUT:   %K.impl_witness: <witness> = impl_witness @empty_tuple.type.as.K.impl.%K.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.414: type = pattern_type %struct_type.x [concrete]
// CHECK:STDOUT:   %.bdc: Core.Form = init_form %struct_type.x, call_param2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.type.1710d5.1: type = fn_type @empty_tuple.type.as.K.impl.F.loc26_52.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.907e52.1: %empty_tuple.type.as.K.impl.F.type.1710d5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %K.facet: %K.type = facet_value %empty_tuple.type, (%K.impl_witness) [concrete]
// CHECK:STDOUT:   %K.WithSelf.F.type.aac: type = fn_type @K.WithSelf.F, @K(%K.facet) [concrete]
// CHECK:STDOUT:   %K.WithSelf.F.e22: %K.WithSelf.F.type.aac = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e85: type = pattern_type %struct_type.a [concrete]
// CHECK:STDOUT:   %.074: Core.Form = init_form %struct_type.a, call_param2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.type.1710d5.2: type = fn_type @empty_tuple.type.as.K.impl.F.loc26_52.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.907e52.2: %empty_tuple.type.as.K.impl.F.type.1710d5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %.loc11_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11_7.2: type = converted %.loc11_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     %.Self: %K.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %K.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc11_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %V.ref: %K.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.6cb]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%K.lookup_impl_witness.28c, element0 [symbolic_self = constants.%impl.elem0.afd]
// CHECK:STDOUT:     %.loc11_31.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11_31.2: type = converted %.loc11_31.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %.loc11_14: type = where_expr %.Self [concrete = constants.%K_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%K.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %struct_type.a
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self.loc3: %K.type = symbolic_binding Self, 0 [symbolic = constants.%Self.44d]
// CHECK:STDOUT:   interface_with_self_decl @K [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %V: type = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %K.assoc_type = assoc_entity element0, @K.%V [concrete = constants.%assoc0.6cb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.WithSelf.F.decl: @K.%K.WithSelf.F.type (%K.WithSelf.F.type.d74) = fn_decl @K.WithSelf.F [symbolic = @K.%K.WithSelf.F (constants.%K.WithSelf.F.f53)] {
// CHECK:STDOUT:     %self.patt: @K.WithSelf.F.%pattern_type.loc8_8 (%pattern_type.bcd) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @K.WithSelf.F.%pattern_type.loc8_8 (%pattern_type.bcd) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: @K.WithSelf.F.%pattern_type.loc8_20 (%pattern_type.9fa) = value_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: @K.WithSelf.F.%pattern_type.loc8_20 (%pattern_type.9fa) = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @K.WithSelf.F.%pattern_type.loc8_20 (%pattern_type.9fa) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @K.WithSelf.F.%pattern_type.loc8_20 (%pattern_type.9fa) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc8_29: type = impl_witness_access constants.%K.lookup_impl_witness.a24, element0 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.c23)]
// CHECK:STDOUT:     %V.ref.loc8_29: type = name_ref V, %impl.elem0.loc8_29 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.c23)]
// CHECK:STDOUT:     %.loc8_29.2: Core.Form = init_form %V.ref.loc8_29, call_param2 [symbolic = %.loc8_29.1 (constants.%.bf0)]
// CHECK:STDOUT:     %self.param: @K.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.87a) = value_param call_param0
// CHECK:STDOUT:     %.loc8_14.1: type = splice_block %.loc8_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.87a)] {
// CHECK:STDOUT:       %Self.ref: %K.type = name_ref Self, @K.%Self.loc3 [symbolic = %Self (constants.%Self.44d)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.87a)]
// CHECK:STDOUT:       %.loc8_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.87a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @K.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.87a) = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: @K.WithSelf.F.%impl.elem0.loc8_23.1 (%impl.elem0.c23) = value_param call_param1
// CHECK:STDOUT:     %.loc8_23: type = splice_block %V.ref.loc8_23 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.c23)] {
// CHECK:STDOUT:       %impl.elem0.loc8_23.2: type = impl_witness_access constants.%K.lookup_impl_witness.a24, element0 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.c23)]
// CHECK:STDOUT:       %V.ref.loc8_23: type = name_ref V, %impl.elem0.loc8_23.2 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.c23)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @K.WithSelf.F.%impl.elem0.loc8_23.1 (%impl.elem0.c23) = value_binding v, %v.param
// CHECK:STDOUT:     %return.param: ref @K.WithSelf.F.%impl.elem0.loc8_23.1 (%impl.elem0.c23) = out_param call_param2
// CHECK:STDOUT:     %return: ref @K.WithSelf.F.%impl.elem0.loc8_23.1 (%impl.elem0.c23) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %K.assoc_type = assoc_entity element1, %K.WithSelf.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%V, %K.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.K.impl: %.loc11_7.2 as %.loc11_14 {
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.decl.loc26_52.1: %empty_tuple.type.as.K.impl.F.type.1710d5.1 = fn_decl @empty_tuple.type.as.K.impl.F.loc26_52.1 [concrete = constants.%empty_tuple.type.as.K.impl.F.907e52.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: %pattern_type.414 = value_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: %pattern_type.414 = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.414 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.414 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_49.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc26_49.2: type = converted %.loc26_49.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.x.loc26_50: type = struct_type {.x: %empty_tuple.type} [concrete = constants.%struct_type.x]
// CHECK:STDOUT:     %.loc26_50: Core.Form = init_form %struct_type.x.loc26_50, call_param2 [concrete = constants.%.bdc]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.K.impl.%.loc11_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: %struct_type.x = value_param call_param1
// CHECK:STDOUT:     %.loc26_37: type = splice_block %struct_type.x.loc26_37 [concrete = constants.%struct_type.x] {
// CHECK:STDOUT:       %.loc26_36.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc26_36.2: type = converted %.loc26_36.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.x.loc26_37: type = struct_type {.x: %empty_tuple.type} [concrete = constants.%struct_type.x]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: %struct_type.x = value_binding v, %v.param
// CHECK:STDOUT:     %return.param: ref %struct_type.x = out_param call_param2
// CHECK:STDOUT:     %return: ref %struct_type.x = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_29.1: type = specific_constant @K.WithSelf.F.%V.ref.loc8_29, @K.WithSelf.F(constants.%K.facet) [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   %.loc8_29.2: type = specific_constant @K.WithSelf.F.%.loc8_29.2, @K.WithSelf.F(constants.%K.facet) [concrete = constants.%.074]
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.decl.loc26_52.2: %empty_tuple.type.as.K.impl.F.type.1710d5.2 = fn_decl @empty_tuple.type.as.K.impl.F.loc26_52.2 [concrete = constants.%empty_tuple.type.as.K.impl.F.907e52.2] {
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
// CHECK:STDOUT:   %K.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %empty_tuple.type.as.K.impl.F.decl.loc26_52.2), @empty_tuple.type.as.K.impl [concrete]
// CHECK:STDOUT:   %K.impl_witness: <witness> = impl_witness %K.impl_witness_table [concrete = constants.%K.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%struct_type.a [concrete = constants.%struct_type.a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.K.impl.F.decl.loc26_52.1
// CHECK:STDOUT:   witness = %K.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K.WithSelf.F(@K.%Self.loc3: %K.type) {
// CHECK:STDOUT:   %Self: %K.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.44d)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.87a)]
// CHECK:STDOUT:   %pattern_type.loc8_8: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc8_8 (constants.%pattern_type.bcd)]
// CHECK:STDOUT:   %K.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @K [symbolic = %K.lookup_impl_witness (constants.%K.lookup_impl_witness.a24)]
// CHECK:STDOUT:   %impl.elem0.loc8_23.1: type = impl_witness_access %K.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_23.1 (constants.%impl.elem0.c23)]
// CHECK:STDOUT:   %pattern_type.loc8_20: type = pattern_type %impl.elem0.loc8_23.1 [symbolic = %pattern_type.loc8_20 (constants.%pattern_type.9fa)]
// CHECK:STDOUT:   %.loc8_29.1: Core.Form = init_form %impl.elem0.loc8_23.1, call_param2 [symbolic = %.loc8_29.1 (constants.%.bf0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @K.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.87a), %v.param: @K.WithSelf.F.%impl.elem0.loc8_23.1 (%impl.elem0.c23)) -> out %return.param: @K.WithSelf.F.%impl.elem0.loc8_23.1 (%impl.elem0.c23);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.K.impl.F.loc26_52.1(%self.param: %empty_tuple.type, %v.param: %struct_type.x) -> out %return.param: %struct_type.x {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: %struct_type.x = name_ref v, %v
// CHECK:STDOUT:   %.loc26_61.1: %empty_tuple.type = struct_access %v.ref, element0
// CHECK:STDOUT:   %.loc26_61.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc26_61.3: init %empty_tuple.type = converted %.loc26_61.1, %.loc26_61.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc26_61.4: init %struct_type.x = struct_init (%.loc26_61.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc26_62: init %struct_type.x = converted %v.ref, %.loc26_61.4 [concrete = constants.%struct]
// CHECK:STDOUT:   return %.loc26_62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.K.impl.F.loc26_52.2(%self.param: %empty_tuple.type, %v.param: %struct_type.a) -> out %return.param: %struct_type.a [thunk @empty_tuple.type.as.K.impl.%empty_tuple.type.as.K.impl.F.decl.loc26_52.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.K.impl.F.type.1710d5.1 = name_ref F, @empty_tuple.type.as.K.impl.%empty_tuple.type.as.K.impl.F.decl.loc26_52.1 [concrete = constants.%empty_tuple.type.as.K.impl.F.907e52.1]
// CHECK:STDOUT:   %self.ref: %empty_tuple.type = name_ref self, %self.param
// CHECK:STDOUT:   %v.ref: %struct_type.a = name_ref v, %v.param
// CHECK:STDOUT:   %return.ref: ref %struct_type.a = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   %empty_tuple.type.as.K.impl.F.call: init %struct_type.x = call %empty_tuple.type.as.K.impl.F.bound(%self.ref, <error>)
// CHECK:STDOUT:   %.loc26_52.1: ref %struct_type.x = temporary_storage
// CHECK:STDOUT:   %.loc26_52.2: ref %struct_type.x = temporary %.loc26_52.1, %empty_tuple.type.as.K.impl.F.call
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc26_52.2, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc26_52.2)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %struct_type.x) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%Self.44d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc8 => constants.%Self.44d
// CHECK:STDOUT:   %K.WithSelf.F.type => constants.%K.WithSelf.F.type.d74
// CHECK:STDOUT:   %K.WithSelf.F => constants.%K.WithSelf.F.f53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.WithSelf.F(constants.%Self.44d) {
// CHECK:STDOUT:   %Self => constants.%Self.44d
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.87a
// CHECK:STDOUT:   %pattern_type.loc8_8 => constants.%pattern_type.bcd
// CHECK:STDOUT:   %K.lookup_impl_witness => constants.%K.lookup_impl_witness.a24
// CHECK:STDOUT:   %impl.elem0.loc8_23.1 => constants.%impl.elem0.c23
// CHECK:STDOUT:   %pattern_type.loc8_20 => constants.%pattern_type.9fa
// CHECK:STDOUT:   %.loc8_29.1 => constants.%.bf0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc8 => constants.%.Self
// CHECK:STDOUT:   %K.WithSelf.F.type => constants.%K.WithSelf.F.type.262
// CHECK:STDOUT:   %K.WithSelf.F => constants.%K.WithSelf.F.40c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%K.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc8 => constants.%K.facet
// CHECK:STDOUT:   %K.WithSelf.F.type => constants.%K.WithSelf.F.type.aac
// CHECK:STDOUT:   %K.WithSelf.F => constants.%K.WithSelf.F.e22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.WithSelf.F(constants.%K.facet) {
// CHECK:STDOUT:   %Self => constants.%K.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type.loc8_8 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %K.lookup_impl_witness => constants.%K.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc8_23.1 => constants.%struct_type.a
// CHECK:STDOUT:   %pattern_type.loc8_20 => constants.%pattern_type.e85
// CHECK:STDOUT:   %.loc8_29.1 => constants.%.074
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_non-type_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self: %M.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type @M [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete]
// CHECK:STDOUT:   %.dde: Core.Form = init_form %empty_struct_type, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %M.WithSelf.G.type.e2f: type = fn_type @M.WithSelf.G, @M(%Self) [symbolic]
// CHECK:STDOUT:   %M.WithSelf.G.020: %M.WithSelf.G.type.e2f = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%M.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self: %M.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %M.WithSelf.G.type.a84: type = fn_type @M.WithSelf.G, @M(%.Self) [symbolic_self]
// CHECK:STDOUT:   %M.WithSelf.G.6c7: %M.WithSelf.G.type.a84 = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %M.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @M [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access %M.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %struct: %struct_type.b.347 = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %M_where.type: type = facet_type <@M where %impl.elem0 = %struct> [concrete]
// CHECK:STDOUT:   %M.impl_witness: <witness> = impl_witness @empty_tuple.type.as.M.impl.%M.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.M.impl.G.type: type = fn_type @empty_tuple.type.as.M.impl.G [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.M.impl.G: %empty_tuple.type.as.M.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %empty_tuple.type, (%M.impl_witness) [concrete]
// CHECK:STDOUT:   %M.WithSelf.G.type.5a0: type = fn_type @M.WithSelf.G, @M(%M.facet) [concrete]
// CHECK:STDOUT:   %M.WithSelf.G.986: %M.WithSelf.G.type.5a0 = struct_value () [concrete]
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
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.347 = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_32: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_33.1: %struct_type.b.347 = struct_literal (%.loc8_32) [concrete = constants.%struct]
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_33.2: %empty_struct_type = converted %.loc8_32, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %struct: %struct_type.b.347 = struct_value (%.loc8_33.2) [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc8_33.3: %struct_type.b.347 = converted %.loc8_33.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%M_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%M.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_33.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self.loc3: %M.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @M [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z: %struct_type.b.347 = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.WithSelf.G.decl: @M.%M.WithSelf.G.type (%M.WithSelf.G.type.e2f) = fn_decl @M.WithSelf.G [symbolic = @M.%M.WithSelf.G (constants.%M.WithSelf.G.020)] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_14.3: Core.Form = init_form %.loc5_14.2, call_param0 [concrete = constants.%.dde]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, %M.WithSelf.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .Z = @Z.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Z, %M.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.M.impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %empty_tuple.type.as.M.impl.G.decl: %empty_tuple.type.as.M.impl.G.type = fn_decl @empty_tuple.type.as.M.impl.G [concrete = constants.%empty_tuple.type.as.M.impl.G] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_14.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc9_14.2: type = converted %.loc9_14.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc9_14.3: Core.Form = init_form %.loc9_14.2, call_param0 [concrete = constants.%.dde]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %empty_tuple.type.as.M.impl.G.decl), @empty_tuple.type.as.M.impl [concrete]
// CHECK:STDOUT:   %M.impl_witness: <witness> = impl_witness %M.impl_witness_table [concrete = constants.%M.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %struct_type.b.347 = impl_witness_assoc_constant constants.%struct [concrete = constants.%struct]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %empty_tuple.type.as.M.impl.G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = %M.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @M.WithSelf.G(@M.%Self.loc3: %M.type) {
// CHECK:STDOUT:   fn() -> out %return.param: %empty_struct_type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.M.impl.G() -> out %return.param: %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @empty_tuple.type.as.M.impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %Self.ref, (constants.%M.impl_witness) [concrete = constants.%M.facet]
// CHECK:STDOUT:   %.loc10_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_18 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc10_23: type = converted %.loc10_18, %as_type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access constants.%M.impl_witness, element0 [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_25.1: %empty_struct_type = struct_access %impl.elem0, element0 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_25.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_27: init %empty_struct_type = converted %.loc10_25.1, %.loc10_25.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc10_27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%Self
// CHECK:STDOUT:   %M.WithSelf.G.type => constants.%M.WithSelf.G.type.e2f
// CHECK:STDOUT:   %M.WithSelf.G => constants.%M.WithSelf.G.020
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M.WithSelf.G(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%.Self
// CHECK:STDOUT:   %M.WithSelf.G.type => constants.%M.WithSelf.G.type.a84
// CHECK:STDOUT:   %M.WithSelf.G => constants.%M.WithSelf.G.6c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M(constants.%M.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%M.facet
// CHECK:STDOUT:   %M.WithSelf.G.type => constants.%M.WithSelf.G.type.5a0
// CHECK:STDOUT:   %M.WithSelf.G => constants.%M.WithSelf.G.986
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M.WithSelf.G(constants.%M.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_as_uses_correct_rewrite_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.5a3: type = class_type @C, @C(%T.67d) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self.0c7: %M.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.b.86f: type = struct_type {.b: type} [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type @M [concrete]
// CHECK:STDOUT:   %assoc0.ed2: %M.assoc_type = assoc_entity element0, @M.%Z [concrete]
// CHECK:STDOUT:   %.824: Core.Form = init_form type, call_param0 [concrete]
// CHECK:STDOUT:   %M.WithSelf.G.type.e2f: type = fn_type @M.WithSelf.G, @M(%Self.0c7) [symbolic]
// CHECK:STDOUT:   %M.WithSelf.G.020: %M.WithSelf.G.type.e2f = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%M.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.850: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %.Self.b4d: %M.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %M.WithSelf.G.type.a84: type = fn_type @M.WithSelf.G, @M(%.Self.b4d) [symbolic_self]
// CHECK:STDOUT:   %M.WithSelf.G.6c7: %M.WithSelf.G.type.a84 = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.b4d [symbolic_self]
// CHECK:STDOUT:   %M.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.b4d, @M [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.47b: %struct_type.b.86f = impl_witness_access %M.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.692: %struct_type.b.347 = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %struct_type.b.161: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct.0f3: %struct_type.b.86f = struct_value (%empty_struct_type) [concrete]
// CHECK:STDOUT:   %M_where.type.2ad: type = facet_type <@M where %impl.elem0.47b = %struct.0f3> [concrete]
// CHECK:STDOUT:   %M.impl_witness.9de: <witness> = impl_witness @C.as.M.impl.20d.%M.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.M.impl.G.type.107: type = fn_type @C.as.M.impl.G.loc11 [concrete]
// CHECK:STDOUT:   %C.as.M.impl.G.eb6: %C.as.M.impl.G.type.107 = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.9e0: %M.type = facet_value %C.850, (%M.impl_witness.9de) [concrete]
// CHECK:STDOUT:   %M.WithSelf.G.type.563: type = fn_type @M.WithSelf.G, @M(%M.facet.9e0) [concrete]
// CHECK:STDOUT:   %M.WithSelf.G.897: %M.WithSelf.G.type.563 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.b47: <witness> = impl_witness imports.%Copy.impl_witness_table.b1c [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.b47) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a4f: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.070: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a4f, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound.bca: <bound method> = bound_method %empty_struct_type, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.e8f: type = class_type @C, @C(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %struct.63c: %struct_type.b.161 = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %struct.c94: %struct_type.b.86f = struct_value (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %M_where.type.bf5: type = facet_type <@M where %impl.elem0.47b = %struct.c94> [concrete]
// CHECK:STDOUT:   %M.impl_witness.a5c: <witness> = impl_witness @C.as.M.impl.243.%M.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.M.impl.G.type.52d: type = fn_type @C.as.M.impl.G.loc17 [concrete]
// CHECK:STDOUT:   %C.as.M.impl.G.973: %C.as.M.impl.G.type.52d = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.586: %M.type = facet_value %C.e8f, (%M.impl_witness.a5c) [concrete]
// CHECK:STDOUT:   %M.WithSelf.G.type.5ab: type = fn_type @M.WithSelf.G, @M(%M.facet.586) [concrete]
// CHECK:STDOUT:   %M.WithSelf.G.bdc: %M.WithSelf.G.type.5ab = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound.f05: <bound method> = bound_method %empty_tuple.type, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.1a7: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b1c = impl_witness_table (%Core.import_ref.1a7), @type.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %T.loc3_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_9.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.M.impl.20d [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc10_9: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_10: type = converted %.loc10_9, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.850]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = symbolic_binding .Self [symbolic_self = constants.%.Self.b4d]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.b4d]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_23: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0.ed2]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.86f = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.47b]
// CHECK:STDOUT:     %.loc10_35: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_36.1: %struct_type.b.347 = struct_literal (%.loc10_35) [concrete = constants.%struct.692]
// CHECK:STDOUT:     %.loc10_36.2: type = converted %.loc10_35, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct: %struct_type.b.86f = struct_value (%.loc10_36.2) [concrete = constants.%struct.0f3]
// CHECK:STDOUT:     %.loc10_36.3: %struct_type.b.86f = converted %.loc10_36.1, %struct [concrete = constants.%struct.0f3]
// CHECK:STDOUT:     %.loc10_17: type = where_expr %.Self [concrete = constants.%M_where.type.2ad] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%M.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc10_36.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.M.impl.243 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc16_9: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_10: type = converted %.loc16_9, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_tuple.type) [concrete = constants.%C.e8f]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = symbolic_binding .Self [symbolic_self = constants.%.Self.b4d]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.b4d]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc16_23: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0.ed2]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.86f = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.47b]
// CHECK:STDOUT:     %.loc16_35: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_36.1: %struct_type.b.161 = struct_literal (%.loc16_35) [concrete = constants.%struct.63c]
// CHECK:STDOUT:     %.loc16_36.2: type = converted %.loc16_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct: %struct_type.b.86f = struct_value (%.loc16_36.2) [concrete = constants.%struct.c94]
// CHECK:STDOUT:     %.loc16_36.3: %struct_type.b.86f = converted %.loc16_36.1, %struct [concrete = constants.%struct.c94]
// CHECK:STDOUT:     %.loc16_17: type = where_expr %.Self [concrete = constants.%M_where.type.bf5] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%M.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc16_36.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self.loc5: %M.type = symbolic_binding Self, 0 [symbolic = constants.%Self.0c7]
// CHECK:STDOUT:   interface_with_self_decl @M [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z: %struct_type.b.86f = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0.ed2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.WithSelf.G.decl: @M.%M.WithSelf.G.type (%M.WithSelf.G.type.e2f) = fn_decl @M.WithSelf.G [symbolic = @M.%M.WithSelf.G (constants.%M.WithSelf.G.020)] {
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: Core.Form = init_form type, call_param0 [concrete = constants.%.824]
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, %M.WithSelf.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc5
// CHECK:STDOUT:   .Z = @Z.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Z, %M.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.M.impl.20d: %C as %.loc10_17 {
// CHECK:STDOUT:   %C.as.M.impl.G.decl: %C.as.M.impl.G.type.107 = fn_decl @C.as.M.impl.G.loc11 [concrete = constants.%C.as.M.impl.G.eb6] {
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11: Core.Form = init_form type, call_param0 [concrete = constants.%.824]
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %C.as.M.impl.G.decl), @C.as.M.impl.20d [concrete]
// CHECK:STDOUT:   %M.impl_witness: <witness> = impl_witness %M.impl_witness_table [concrete = constants.%M.impl_witness.9de]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %struct_type.b.86f = impl_witness_assoc_constant constants.%struct.0f3 [concrete = constants.%struct.0f3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.M.impl.G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = %M.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.M.impl.243: %C as %.loc16_17 {
// CHECK:STDOUT:   %C.as.M.impl.G.decl: %C.as.M.impl.G.type.52d = fn_decl @C.as.M.impl.G.loc17 [concrete = constants.%C.as.M.impl.G.973] {
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17: Core.Form = init_form type, call_param0 [concrete = constants.%.824]
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %C.as.M.impl.G.decl), @C.as.M.impl.243 [concrete]
// CHECK:STDOUT:   %M.impl_witness: <witness> = impl_witness %M.impl_witness_table [concrete = constants.%M.impl_witness.a5c]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %struct_type.b.86f = impl_witness_assoc_constant constants.%struct.c94 [concrete = constants.%struct.c94]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.M.impl.G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = %M.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc3_9.2: type) {
// CHECK:STDOUT:   %T.loc3_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_9.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.5a3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @M.WithSelf.G(@M.%Self.loc5: %M.type) {
// CHECK:STDOUT:   fn() -> out %return.param: type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.M.impl.G.loc11() -> out %return.param: type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @C.as.M.impl.20d.%C [concrete = constants.%C.850]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %Self.ref, (constants.%M.impl_witness.9de) [concrete = constants.%M.facet.9e0]
// CHECK:STDOUT:   %.loc12_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.9e0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc12_18 [concrete = constants.%C.850]
// CHECK:STDOUT:   %.loc12_23: type = converted %.loc12_18, %as_type [concrete = constants.%C.850]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0.ed2]
// CHECK:STDOUT:   %impl.elem0.loc12_23: %struct_type.b.86f = impl_witness_access constants.%M.impl_witness.9de, element0 [concrete = constants.%struct.0f3]
// CHECK:STDOUT:   %.loc12_25: type = struct_access %impl.elem0.loc12_23, element0 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %impl.elem0.loc12_25: %.070 = impl_witness_access constants.%Copy.impl_witness.b47, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc12_25, %impl.elem0.loc12_25 [concrete = constants.%type.as.Copy.impl.Op.bound.bca]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%.loc12_25) [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   return %type.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.M.impl.G.loc17() -> out %return.param: type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @C.as.M.impl.243.%C [concrete = constants.%C.e8f]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %Self.ref, (constants.%M.impl_witness.a5c) [concrete = constants.%M.facet.586]
// CHECK:STDOUT:   %.loc18_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.586]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc18_18 [concrete = constants.%C.e8f]
// CHECK:STDOUT:   %.loc18_23: type = converted %.loc18_18, %as_type [concrete = constants.%C.e8f]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0.ed2]
// CHECK:STDOUT:   %impl.elem0.loc18_23: %struct_type.b.86f = impl_witness_access constants.%M.impl_witness.a5c, element0 [concrete = constants.%struct.c94]
// CHECK:STDOUT:   %.loc18_25: type = struct_access %impl.elem0.loc18_23, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %impl.elem0.loc18_25: %.070 = impl_witness_access constants.%Copy.impl_witness.b47, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc18_25, %impl.elem0.loc18_25 [concrete = constants.%type.as.Copy.impl.Op.bound.f05]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%.loc18_25) [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   return %type.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc3_9.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M(constants.%Self.0c7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc7 => constants.%Self.0c7
// CHECK:STDOUT:   %M.WithSelf.G.type => constants.%M.WithSelf.G.type.e2f
// CHECK:STDOUT:   %M.WithSelf.G => constants.%M.WithSelf.G.020
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M.WithSelf.G(constants.%Self.0c7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc3_9.1 => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M(constants.%.Self.b4d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc7 => constants.%.Self.b4d
// CHECK:STDOUT:   %M.WithSelf.G.type => constants.%M.WithSelf.G.type.a84
// CHECK:STDOUT:   %M.WithSelf.G => constants.%M.WithSelf.G.6c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M(constants.%M.facet.9e0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc7 => constants.%M.facet.9e0
// CHECK:STDOUT:   %M.WithSelf.G.type => constants.%M.WithSelf.G.type.563
// CHECK:STDOUT:   %M.WithSelf.G => constants.%M.WithSelf.G.897
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M.WithSelf.G(constants.%M.facet.9e0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.loc3_9.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M(constants.%M.facet.586) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc7 => constants.%M.facet.586
// CHECK:STDOUT:   %M.WithSelf.G.type => constants.%M.WithSelf.G.type.5ab
// CHECK:STDOUT:   %M.WithSelf.G => constants.%M.WithSelf.G.bdc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @M.WithSelf.G(constants.%M.facet.586) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_int_in_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.d72: %I.assoc_type = assoc_entity element0, @I.%N [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.d31: type = symbolic_binding_type Self, 0, %Self.ab9 [symbolic]
// CHECK:STDOUT:   %pattern_type.fa0: type = pattern_type %Self.binding.as_type.d31 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.lookup_impl_witness.3ae: <witness> = lookup_impl_witness %Self.ab9, @I [symbolic]
// CHECK:STDOUT:   %impl.elem0.f83: %i32 = impl_witness_access %I.lookup_impl_witness.3ae, element0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2ed: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.d29: %Int.as.ImplicitAs.impl.Convert.type.2ed = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.640: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ea2, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.240: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.dd4: %Int.as.ImplicitAs.impl.Convert.type.240 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.290: %ImplicitAs.type.139 = facet_value %i32, (%ImplicitAs.impl_witness.640) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.462: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(Core.IntLiteral, %ImplicitAs.facet.290) [concrete]
// CHECK:STDOUT:   %.0a7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.462, %ImplicitAs.facet.290 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.0e7: <bound method> = bound_method %impl.elem0.f83, %Int.as.ImplicitAs.impl.Convert.dd4 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.dd4, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.2a2: <bound method> = bound_method %impl.elem0.f83, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.2a2(%impl.elem0.f83) [symbolic]
// CHECK:STDOUT:   %array_type.d35: type = array_type %Int.as.ImplicitAs.impl.Convert.call, bool [symbolic]
// CHECK:STDOUT:   %.d61: Core.Form = init_form %array_type.d35, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.3e2: type = pattern_type %array_type.d35 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.08c: type = fn_type @I.WithSelf.F, @I(%Self.ab9) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.705: %I.WithSelf.F.type.08c = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.WithSelf.F.type.29d: type = fn_type @I.WithSelf.F, @I(%.Self) [symbolic_self]
// CHECK:STDOUT:   %I.WithSelf.F.8ec: %I.WithSelf.F.type.29d = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.c4b: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.667: %i32 = impl_witness_access %I.lookup_impl_witness.c4b, element0 [symbolic_self]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet.b94) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.667 = %int_2.ef8> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_tuple.type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %array_type.c9b: type = array_type %int_2.ecc, bool [concrete]
// CHECK:STDOUT:   %.d10: Core.Form = init_form %array_type.c9b, call_param1 [concrete]
// CHECK:STDOUT:   %pattern_type.5d5: type = pattern_type %array_type.c9b [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type: type = fn_type @empty_tuple.type.as.I.impl.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F: %empty_tuple.type.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_tuple.type, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.7ea: type = fn_type @I.WithSelf.F, @I(%I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.755: %I.WithSelf.F.type.7ea = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.075: <bound method> = bound_method %int_2.ef8, %Int.as.ImplicitAs.impl.Convert.dd4 [concrete]
// CHECK:STDOUT:   %bound_method.bea: <bound method> = bound_method %int_2.ef8, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %tuple.type.784: type = tuple_type (bool, bool) [concrete]
// CHECK:STDOUT:   %tuple.a9a: %tuple.type.784 = tuple_value (%true, %false) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.348: <witness> = impl_witness imports.%Copy.impl_witness_table.3cc [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.348) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6dd: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.86d: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6dd, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound.12b: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound.082: <bound method> = bound_method %false, %bool.as.Copy.impl.Op [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.0bc: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2ed) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.d29)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ea2 = impl_witness_table (%Core.import_ref.0bc), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.595: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.3cc = impl_witness_table (%Core.import_ref.595), @bool.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %N.ref: %I.assoc_type = name_ref N, @N.%assoc0 [concrete = constants.%assoc0.d72]
// CHECK:STDOUT:     %impl.elem0.loc8_20: %i32 = impl_witness_access constants.%I.lookup_impl_witness.c4b, element0 [symbolic_self = constants.%impl.elem0.667]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc8_25: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc8_25.1: <bound method> = bound_method %int_2, %impl.elem0.loc8_25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc8_25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_25.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.646]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_25.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_25.2: %i32 = converted %int_2, %.loc8_25.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8_20, %.loc8_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self.loc3: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl @N [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%N [concrete = constants.%assoc0.d72]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.%I.WithSelf.F.type (%I.WithSelf.F.type.08c) = fn_decl @I.WithSelf.F [symbolic = @I.%I.WithSelf.F (constants.%I.WithSelf.F.705)] {
// CHECK:STDOUT:     %self.patt: @I.WithSelf.F.%pattern_type.loc5_8 (%pattern_type.fa0) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @I.WithSelf.F.%pattern_type.loc5_8 (%pattern_type.fa0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @I.WithSelf.F.%pattern_type.loc5_22 (%pattern_type.3e2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @I.WithSelf.F.%pattern_type.loc5_22 (%pattern_type.3e2) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %impl.elem0.loc5_37.2: %i32 = impl_witness_access constants.%I.lookup_impl_witness.3ae, element0 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.f83)]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %impl.elem0.loc5_37.2 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.f83)]
// CHECK:STDOUT:     %.loc5_31.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc5_31.2: type = converted %Bool.call, %.loc5_31.1 [concrete = bool]
// CHECK:STDOUT:     %impl.elem0.loc5_37.3: %.0a7 = impl_witness_access constants.%ImplicitAs.impl_witness.640, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.dd4]
// CHECK:STDOUT:     %bound_method.loc5_37.2: <bound method> = bound_method %N.ref, %impl.elem0.loc5_37.3 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.0e7)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc5_37.3, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc5_37.3: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc5_37.1 (constants.%bound_method.2a2)]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call.loc5_37.2: init Core.IntLiteral = call %bound_method.loc5_37.3(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_37.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc5_37.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc5_37.2: Core.IntLiteral = converted %N.ref, %.loc5_37.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %array_type.loc5_38.2: type = array_type %.loc5_37.2, %.loc5_31.2 [symbolic = %array_type.loc5_38.1 (constants.%array_type.d35)]
// CHECK:STDOUT:     %.loc5_38.2: Core.Form = init_form %array_type.loc5_38.2, call_param1 [symbolic = %.loc5_38.1 (constants.%.d61)]
// CHECK:STDOUT:     %self.param: @I.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.d31) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self.loc3 [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.d31) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @I.WithSelf.F.%array_type.loc5_38.1 (%array_type.d35) = out_param call_param1
// CHECK:STDOUT:     %return: ref @I.WithSelf.F.%array_type.loc5_38.1 (%array_type.d35) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %I.WithSelf.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc3
// CHECK:STDOUT:   .N = @N.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%N, %I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
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
// CHECK:STDOUT:     %.loc9_38.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc9_38.2: type = converted %Bool.call, %.loc9_38.1 [concrete = bool]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, %.loc9_38.2 [concrete = constants.%array_type.c9b]
// CHECK:STDOUT:     %.loc9_45: Core.Form = init_form %array_type, call_param1 [concrete = constants.%.d10]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.I.impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %array_type.c9b = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.c9b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %empty_tuple.type.as.I.impl.F.decl), @empty_tuple.type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %i32 = impl_witness_assoc_constant constants.%int_2.ef8 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self.loc3: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.fa0)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.3ae)]
// CHECK:STDOUT:   %impl.elem0.loc5_37.1: %i32 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.f83)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %impl.elem0.loc5_37.1, constants.%Int.as.ImplicitAs.impl.Convert.dd4 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.0e7)]
// CHECK:STDOUT:   %bound_method.loc5_37.1: <bound method> = bound_method %impl.elem0.loc5_37.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc5_37.1 (constants.%bound_method.2a2)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1: init Core.IntLiteral = call %bound_method.loc5_37.1(%impl.elem0.loc5_37.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc5_38.1: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1, bool [symbolic = %array_type.loc5_38.1 (constants.%array_type.d35)]
// CHECK:STDOUT:   %.loc5_38.1: Core.Form = init_form %array_type.loc5_38.1, call_param1 [symbolic = %.loc5_38.1 (constants.%.d61)]
// CHECK:STDOUT:   %pattern_type.loc5_22: type = pattern_type %array_type.loc5_38.1 [symbolic = %pattern_type.loc5_22 (constants.%pattern_type.3e2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.d31)) -> out %return.param: @I.WithSelf.F.%array_type.loc5_38.1 (%array_type.d35);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F(%self.param: %empty_tuple.type) -> out %return.param: %array_type.c9b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_68.1: %tuple.type.784 = tuple_literal (%true, %false) [concrete = constants.%tuple.a9a]
// CHECK:STDOUT:   %impl.elem0.loc9_57: %.86d = impl_witness_access constants.%Copy.impl_witness.348, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9_57: <bound method> = bound_method %true, %impl.elem0.loc9_57 [concrete = constants.%bool.as.Copy.impl.Op.bound.12b]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call.loc9_57: init bool = call %bound_method.loc9_57(%true) [concrete = constants.%true]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_68.2: ref bool = array_index %return.param, %int_0
// CHECK:STDOUT:   %.loc9_68.3: init bool to %.loc9_68.2 = in_place_init %bool.as.Copy.impl.Op.call.loc9_57 [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0.loc9_63: %.86d = impl_witness_access constants.%Copy.impl_witness.348, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9_63: <bound method> = bound_method %false, %impl.elem0.loc9_63 [concrete = constants.%bool.as.Copy.impl.Op.bound.082]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call.loc9_63: init bool = call %bound_method.loc9_63(%false) [concrete = constants.%false]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_68.4: ref bool = array_index %return.param, %int_1
// CHECK:STDOUT:   %.loc9_68.5: init bool to %.loc9_68.4 = in_place_init %bool.as.Copy.impl.Op.call.loc9_63 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_68.6: init %array_type.c9b to %return.param = array_init (%.loc9_68.3, %.loc9_68.5) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_69: init %array_type.c9b = converted %.loc9_68.1, %.loc9_68.6 [concrete = constants.%array]
// CHECK:STDOUT:   return %.loc9_69 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%Self.ab9
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.08c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.705
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.ab9) {
// CHECK:STDOUT:   %Self => constants.%Self.ab9
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.d31
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.fa0
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.3ae
// CHECK:STDOUT:   %impl.elem0.loc5_37.1 => constants.%impl.elem0.f83
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound.0e7
// CHECK:STDOUT:   %bound_method.loc5_37.1 => constants.%bound_method.2a2
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %array_type.loc5_38.1 => constants.%array_type.d35
// CHECK:STDOUT:   %.loc5_38.1 => constants.%.d61
// CHECK:STDOUT:   %pattern_type.loc5_22 => constants.%pattern_type.3e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%.Self
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.29d
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.8ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5 => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.7ea
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.755
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_37.1 => constants.%int_2.ef8
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound.075
// CHECK:STDOUT:   %bound_method.loc5_37.1 => constants.%bound_method.bea
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc5_37.1 => constants.%int_2.ecc
// CHECK:STDOUT:   %array_type.loc5_38.1 => constants.%array_type.c9b
// CHECK:STDOUT:   %.loc5_38.1 => constants.%.d10
// CHECK:STDOUT:   %pattern_type.loc5_22 => constants.%pattern_type.5d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- symbolic_associated_type_in_concrete_context.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.c59: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.cdf: %Z.assoc_type = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.5a3: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.Self.aac: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.aac [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.aac, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.ea1: type = facet_type <@Z where %impl.elem0 = %C.5a3> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.8d2: <witness> = impl_witness @T.as.Z.impl.%Z.impl_witness_table, @T.as.Z.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type.ea1 [symbolic]
// CHECK:STDOUT:   %Z.facet.49c: %Z.type = facet_value %T, (%Z.impl_witness.8d2) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.302: type = class_type @C, @C(%D) [concrete]
// CHECK:STDOUT:   %Z_where.type.4e0: type = facet_type <@Z where %impl.elem0 = %C.302> [concrete]
// CHECK:STDOUT:   %Z.impl_witness.63b: <witness> = impl_witness @T.as.Z.impl.%Z.impl_witness_table, @T.as.Z.impl(%D) [concrete]
// CHECK:STDOUT:   %complete_type.9a2: <witness> = complete_type_witness %Z_where.type.4e0 [concrete]
// CHECK:STDOUT:   %Z.facet.59e: %Z.type = facet_value %D, (%Z.impl_witness.63b) [concrete]
// CHECK:STDOUT:   %pattern_type.a7e: type = pattern_type %C.302 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C.302 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %T.loc6_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_24: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self.1: %Z.type = symbolic_binding .Self [symbolic_self = constants.%.Self.aac]
// CHECK:STDOUT:     %.Self.ref: %Z.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.aac]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc9_37: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.cdf]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref.loc9_44: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:     %C.loc9_45.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc9_45.2 (constants.%C.5a3)]
// CHECK:STDOUT:     %.loc9_31: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.ea1)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.loc9_45.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %T.loc9_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c59]
// CHECK:STDOUT:   interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%X [concrete = constants.%assoc0.cdf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (%X)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl(%T.loc9_14.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:   %C.loc9_45.2: type = class_type @C, @C(%T.loc9_14.2) [symbolic = %C.loc9_45.2 (constants.%C.5a3)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z where constants.%impl.elem0 = %C.loc9_45.2> [symbolic = %Z_where.type (constants.%Z_where.type.ea1)]
// CHECK:STDOUT:   %Z.impl_witness.loc9_47.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl(%T.loc9_14.2) [symbolic = %Z.impl_witness.loc9_47.2 (constants.%Z.impl_witness.8d2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc9_24 as %.loc9_31 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc9_47.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl(constants.%T) [symbolic = %Z.impl_witness.loc9_47.2 (constants.%Z.impl_witness.8d2)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C.5a3 [symbolic = %C.loc9_45.2 (constants.%C.5a3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Z.impl_witness.loc9_47.1
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
// CHECK:STDOUT:     .Self = constants.%C.5a3
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
// CHECK:STDOUT:     %a.patt: %pattern_type.a7e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %D.ref.loc12_35: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%D) [concrete = constants.%C.302]
// CHECK:STDOUT:   %.loc12_28.2: ref %C.302 = temporary_storage
// CHECK:STDOUT:   %.loc12_28.3: init %C.302 to %.loc12_28.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc12_28.4: ref %C.302 = temporary %.loc12_28.2, %.loc12_28.3
// CHECK:STDOUT:   %.loc12_30.1: ref %C.302 = converted %.loc12_28.1, %.loc12_28.4
// CHECK:STDOUT:   %.loc12_18.1: type = splice_block %impl.elem0 [concrete = constants.%C.302] {
// CHECK:STDOUT:     %D.ref.loc12_17: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.cdf]
// CHECK:STDOUT:     %Z.facet: %Z.type = facet_value %D.ref.loc12_17, (constants.%Z.impl_witness.63b) [concrete = constants.%Z.facet.59e]
// CHECK:STDOUT:     %.loc12_18.2: %Z.type = converted %D.ref.loc12_17, %Z.facet [concrete = constants.%Z.facet.59e]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.impl_witness.63b, element0 [concrete = constants.%C.302]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_30.2: %C.302 = acquire_value %.loc12_30.1
// CHECK:STDOUT:   %a: %C.302 = value_binding a, %.loc12_30.2
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc12_28.4, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc12_28.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %C.302) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.c59) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%.Self.aac) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T
// CHECK:STDOUT:   %C.loc9_45.2 => constants.%C.5a3
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.ea1
// CHECK:STDOUT:   %Z.impl_witness.loc9_47.2 => constants.%Z.impl_witness.8d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Z.facet.49c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl(constants.%D) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%D
// CHECK:STDOUT:   %C.loc9_45.2 => constants.%C.302
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.4e0
// CHECK:STDOUT:   %Z.impl_witness.loc9_47.2 => constants.%Z.impl_witness.63b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.9a2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%D) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Z.facet.59e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
