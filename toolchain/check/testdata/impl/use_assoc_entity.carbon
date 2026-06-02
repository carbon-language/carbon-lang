// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
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
  //@dump-sem-ir-begin
  fn F[self: Self](u: U) -> U;
  //@dump-sem-ir-end
}

impl () as J where .U = i32 {
  //@dump-sem-ir-begin
  fn F[unused self: Self](u: i32) -> i32 { return -u; }
  //@dump-sem-ir-end
}

fn CallMethod(x: ()) -> i32 {
  //@dump-sem-ir-begin
  return x.(J.F)(40);
  //@dump-sem-ir-end
}

class C {
  // This allows the type to be copyable so it can be returned.
  adapt {};
}

impl C as J where .U = C {
  //@dump-sem-ir-begin
  fn F[self: Self](unused u: C) -> C { return self; }
  //@dump-sem-ir-end
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
  //@dump-sem-ir-begin
  let unused e1: i32 = e.F(2);
  let unused e2: i32 = e.G(3);
  let unused e3: i32 = E.F(4);
  let unused e4: i32 = e.(E.G)(5);
  let unused e5: i32 = e.(J.G)(6);
  //@dump-sem-ir-end
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

// --- use_associated_fn_in_impl.carbon
library "[[@TEST_NAME]]";

interface J {
  fn F();
  fn G();
}

class E {}

impl E as J {
  fn F() {}
  //@dump-sem-ir-begin
  fn G() { F(); }
  //@dump-sem-ir-end
}

// --- use_associated_constant_in_impl.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F(u: U) -> U;
}

class E {
  extend impl as J where .U = i32 {
    //@dump-sem-ir-begin
    fn F(u: U) -> U {
      return u;
    }
    //@dump-sem-ir-end
  }
}

// --- call_extend_fn.carbon
library "[[@TEST_NAME]]";

interface J {
  fn F();
}

class E {
  extend impl as J {
    fn F() {}
  }

  fn G() {
    //@dump-sem-ir-begin
    F();
    //@dump-sem-ir-end
  }
}

// --- call_extend_method.carbon
library "[[@TEST_NAME]]";

interface J {
  fn F[self: Self]();
}

class E {
  extend impl as J {
    fn F[unused self: Self]() {}
  }

  fn G[self: Self]() { self.F(); }
}

// --- use_extend_constant.carbon
library "[[@TEST_NAME]]";

interface J {
  let X:! type;
}

class E {
  extend impl as J where .X = () {}

  //@dump-sem-ir-begin
  fn G() -> X { return (); }
  //@dump-sem-ir-end
}

// --- associated_entity_scope.carbon
library "[[@TEST_NAME]]";

interface J {
  let T:! type;
  fn F() -> T;
}

class A {}

class E(T:! type) {
  extend impl as J where .T = A {
    // This `T` names the interface's `T` (extended by the `impl`), not the
    // class's `T` parameter.
    fn F() -> T {
      return {} as A;
    }
  }
}

impl forall [T:! type] T as J where .T = () {
  // This `T` names the interface's `T` (extended by the `impl`), not the
  // class's `T` parameter.
  //
  // TODO: Should it? Arguably here the `forall T` is part of the impl, so we
  // should look there before checking for scopes the `impl` extends. But also
  // arguably we're naming the `T` inside the impl itself, which is within the
  // `forall`.
  fn F() -> T {
    return ();
  }
}

// --- fail_todo_associated_constant_use_before_value_known.carbon
library "[[@TEST_NAME]]";

interface J {
  let T:! type;
  let U:! type;
}

class A {}

// TODO: Make sure this behaves reasonably once we support `where` declarations.

// CHECK:STDERR: fail_todo_associated_constant_use_before_value_known.carbon:[[@LINE+14]]:1: error: associated constant T not given a value in impl of interface J [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl A as J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_associated_constant_use_before_value_known.carbon:[[@LINE-11]]:7: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:   let T:! type;
// CHECK:STDERR:       ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_associated_constant_use_before_value_known.carbon:[[@LINE+7]]:1: error: associated constant U not given a value in impl of interface J [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl A as J {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_associated_constant_use_before_value_known.carbon:[[@LINE-17]]:7: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:   let U:! type;
// CHECK:STDERR:       ^~~~~~~~
// CHECK:STDERR:
impl A as J {
  // CHECK:STDERR: fail_todo_associated_constant_use_before_value_known.carbon:[[@LINE+8]]:3: error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:   where .U = T;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_associated_constant_use_before_value_known.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   where .U = T;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  where .U = T;
  // CHECK:STDERR: fail_todo_associated_constant_use_before_value_known.carbon:[[@LINE+4]]:3: error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:   where .T = ();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  where .T = ();
}

// --- associated_fn_use_in_def_before_decl.carbon
library "[[@TEST_NAME]]";

interface J {
  fn F();
  fn G();
}

class A {}

impl A as J {
  // OK, we handle both function declarations before either function body.
  //@dump-sem-ir-begin
  fn F() { G(); }
  //@dump-sem-ir-end
  fn G() {}
}

// --- fail_associated_fn_use_in_decl_after_decl.carbon
library "[[@TEST_NAME]]";

interface J {
  fn F() -> Self;
  fn G() -> type;
}

class A {}

impl A as J {
  // Evaluation of the call to `G()` below fails because we don't parse the
  // definition of `G()` until the enclosing `impl` is complete.
  // TODO: Diagnose this better.
  eval fn G() -> type { return A; }
  // CHECK:STDERR: fail_associated_fn_use_in_decl_after_decl.carbon:[[@LINE+4]]:13: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   fn F() -> G();
  // CHECK:STDERR:             ^~~
  // CHECK:STDERR:
  fn F() -> G();
}

// --- fail_associated_fn_use_in_decl_before_decl.carbon
library "[[@TEST_NAME]]";

interface J {
  fn F() -> Self;
  fn G() -> type;
}

class A {}

impl A as J {
  // CHECK:STDERR: fail_associated_fn_use_in_decl_before_decl.carbon:[[@LINE+4]]:13: error: accessing member from impl before it has a defined value [ImplAccessMemberBeforeSet]
  // CHECK:STDERR:   fn F() -> G();
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn F() -> G();
  eval fn G() -> type { return A; }
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
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE+3]]:20: error: cannot convert from struct type `{.a: ()}` to `{.x: ()}`: unexpected field `a` in source type [StructInitUnexpectedFieldInConversion]
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
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE+7]]:3: error: cannot convert from struct type `{.x: ()}` to `{.a: ()}`: unexpected field `x` in source type [StructInitUnexpectedFieldInConversion]
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
    //@dump-sem-ir-begin
    return (Self as M).Z.b;
    //@dump-sem-ir-end
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
library "[[@TEST_NAME]]";

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
// CHECK:STDOUT:   %Self.fd8: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %Self.as_type.e52: type = facet_access_type %Self.fd8 [symbolic]
// CHECK:STDOUT:   %pattern_type.9d0: type = pattern_type %Self.as_type.e52 [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.89d: <witness> = lookup_impl_witness %Self.fd8, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.fbe: type = impl_witness_access %J.lookup_impl_witness.89d, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.d8a: type = pattern_type %impl.elem0.fbe [symbolic]
// CHECK:STDOUT:   %.f0e: Core.Form = init_form %impl.elem0.fbe [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.37c: type = fn_type @J.WithSelf.F, @J.WithSelf(%Self.fd8) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.acf: %J.WithSelf.F.type.37c = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.24c: %J.assoc_type = assoc_entity element1, @J.WithSelf.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %.Self.944: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.type.131: type = fn_type @J.WithSelf.F, @J.WithSelf(%.Self.944) [symbolic_self]
// CHECK:STDOUT:   %J.WithSelf.F.51c: %J.WithSelf.F.type.131 = struct_value () [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J.impl_witness.133: <witness> = impl_witness @empty_tuple.type.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.7ea: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J.impl.F.type: type = fn_type @empty_tuple.type.as.J.impl.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.J.impl.F: %empty_tuple.type.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.180: %J.type = facet_value %empty_tuple.type, (%J.impl_witness.133) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.096: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.180) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.cd8: %J.WithSelf.F.type.096 = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.8c1: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.f1a: %Int.as.Negate.impl.Op.type.8c1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Negate.impl_witness.a16: <witness> = impl_witness imports.%Negate.impl_witness_table, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.type.c30: type = fn_type @Int.as.Negate.impl.Op, @Int.as.Negate.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.aa9: %Int.as.Negate.impl.Op.type.c30 = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %i32, (%Negate.impl_witness.a16) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.b32: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.7fb: type = fn_type_with_self_type %Negate.WithSelf.Op.type.b32, %Negate.facet [concrete]
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Negate.impl.Op.aa9, @Int.as.Negate.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %J.WithSelf.F.type.096, %J.facet.180 [concrete]
// CHECK:STDOUT:   %int_40.f80: Core.IntLiteral = int_value 40 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_40.f80, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_40.f80, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_40.c3d: %i32 = int_value 40 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %J.impl_witness.62f: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.0d5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.8f0: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %C.as.J.impl.F.type: type = fn_type @C.as.J.impl.F [concrete]
// CHECK:STDOUT:   %C.as.J.impl.F: %C.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.bf8: %J.type = facet_value %C, (%J.impl_witness.62f) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.78a: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.bf8) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.a5c: %J.WithSelf.F.type.78a = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.d77 = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b1a: @Int.as.Negate.impl.%Int.as.Negate.impl.Op.type (%Int.as.Negate.impl.Op.type.8c1) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Negate.impl.%Int.as.Negate.impl.Op (constants.%Int.as.Negate.impl.Op.f1a)]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.d77, %Core.import_ref.b1a), @Int.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.WithSelf.%J.WithSelf.F.type (%J.WithSelf.F.type.37c) = fn_decl @J.WithSelf.F [symbolic = @J.WithSelf.%J.WithSelf.F (constants.%J.WithSelf.F.acf)] {
// CHECK:STDOUT:     %self.param_patt: @J.WithSelf.F.%pattern_type.loc6_12 (%pattern_type.9d0) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @J.WithSelf.F.%pattern_type.loc6_12 (%pattern_type.9d0) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %u.param_patt: @J.WithSelf.F.%pattern_type.loc6_21 (%pattern_type.d8a) = value_param_pattern [concrete]
// CHECK:STDOUT:     %u.patt: @J.WithSelf.F.%pattern_type.loc6_21 (%pattern_type.d8a) = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @J.WithSelf.F.%pattern_type.loc6_21 (%pattern_type.d8a) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @J.WithSelf.F.%pattern_type.loc6_21 (%pattern_type.d8a) = return_slot_pattern %return.param_patt, %U.ref.loc6_29 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc6_29: type = impl_witness_access constants.%J.lookup_impl_witness.89d, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fbe)]
// CHECK:STDOUT:     %U.ref.loc6_29: type = name_ref U, %impl.elem0.loc6_29 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fbe)]
// CHECK:STDOUT:     %.loc6_29.2: Core.Form = init_form %U.ref.loc6_29 [symbolic = %.loc6_29.1 (constants.%.f0e)]
// CHECK:STDOUT:     %self.param: @J.WithSelf.F.%Self.as_type.loc6_14.1 (%Self.as_type.e52) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.e52)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.fd8)]
// CHECK:STDOUT:       %Self.as_type.loc6_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.e52)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.e52)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @J.WithSelf.F.%Self.as_type.loc6_14.1 (%Self.as_type.e52) = value_binding self, %self.param
// CHECK:STDOUT:     %u.param: @J.WithSelf.F.%impl.elem0.loc6_23.1 (%impl.elem0.fbe) = value_param call_param1
// CHECK:STDOUT:     %.loc6_23: type = splice_block %U.ref.loc6_23 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fbe)] {
// CHECK:STDOUT:       %impl.elem0.loc6_23.2: type = impl_witness_access constants.%J.lookup_impl_witness.89d, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fbe)]
// CHECK:STDOUT:       %U.ref.loc6_23: type = name_ref U, %impl.elem0.loc6_23.2 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fbe)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @J.WithSelf.F.%impl.elem0.loc6_23.1 (%impl.elem0.fbe) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @J.WithSelf.F.%impl.elem0.loc6_23.1 (%impl.elem0.fbe) = out_param call_param2
// CHECK:STDOUT:     %return: ref @J.WithSelf.F.%impl.elem0.loc6_23.1 (%impl.elem0.fbe) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %J.WithSelf.F.decl [concrete = constants.%assoc1.24c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = @J.WithSelf.%assoc1
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   witness = (@J.WithSelf.%U, @J.WithSelf.%J.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.J.impl: %.loc10_7.2 as %.loc10_14 {
// CHECK:STDOUT:   %empty_tuple.type.as.J.impl.F.decl: %empty_tuple.type.as.J.impl.F.type = fn_decl @empty_tuple.type.as.J.impl.F [concrete = constants.%empty_tuple.type.as.J.impl.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.104 = value_param_pattern [concrete]
// CHECK:STDOUT:     %u.patt: %pattern_type.104 = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.104 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.104 = return_slot_pattern %return.param_patt, %i32.loc12_38 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc12_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %i32.loc12_38 [concrete = constants.%.7ea]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.J.impl.%.loc10_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc12_30: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %u: %i32 = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.J.impl.F.decl
// CHECK:STDOUT:   extend %.loc10_14
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref.loc27_6 as %.loc27_13 {
// CHECK:STDOUT:   %C.as.J.impl.F.decl: %C.as.J.impl.F.type = fn_decl @C.as.J.impl.F [concrete = constants.%C.as.J.impl.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0d5 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.0d5 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.0d5 = value_param_pattern [concrete]
// CHECK:STDOUT:     %u.patt: %pattern_type.0d5 = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.0d5 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.0d5 = return_slot_pattern %return.param_patt, %C.ref.loc29_36 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc29_36: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc29_36: Core.Form = init_form %C.ref.loc29_36 [concrete = constants.%.8f0]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.J.impl.%C.ref.loc27_6 [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:     %u.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc29_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %u: %C = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %C.as.J.impl.F.decl
// CHECK:STDOUT:   extend %.loc27_13
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.fd8)]
// CHECK:STDOUT:   %Self.as_type.loc6_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.e52)]
// CHECK:STDOUT:   %pattern_type.loc6_12: type = pattern_type %Self.as_type.loc6_14.1 [symbolic = %pattern_type.loc6_12 (constants.%pattern_type.9d0)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.89d)]
// CHECK:STDOUT:   %impl.elem0.loc6_23.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.fbe)]
// CHECK:STDOUT:   %pattern_type.loc6_21: type = pattern_type %impl.elem0.loc6_23.1 [symbolic = %pattern_type.loc6_21 (constants.%pattern_type.d8a)]
// CHECK:STDOUT:   %.loc6_29.1: Core.Form = init_form %impl.elem0.loc6_23.1 [symbolic = %.loc6_29.1 (constants.%.f0e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @J.WithSelf.F.%Self.as_type.loc6_14.1 (%Self.as_type.e52), %u.param: @J.WithSelf.F.%impl.elem0.loc6_23.1 (%impl.elem0.fbe)) -> out %return.param: @J.WithSelf.F.%impl.elem0.loc6_23.1 (%impl.elem0.fbe);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.J.impl.F(%self.param: %empty_tuple.type, %u.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %impl.elem1: %.7fb = impl_witness_access constants.%Negate.impl_witness.a16, element1 [concrete = constants.%Int.as.Negate.impl.Op.aa9]
// CHECK:STDOUT:   %bound_method.loc12_51.1: <bound method> = bound_method %u.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @Int.as.Negate.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Negate.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_51.2: <bound method> = bound_method %u.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call: init %i32 = call %bound_method.loc12_51.2(%u.ref)
// CHECK:STDOUT:   return %Int.as.Negate.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallMethod(%x.param: %empty_tuple.type) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %F.ref: %J.assoc_type = name_ref F, @J.WithSelf.%assoc1 [concrete = constants.%assoc1.24c]
// CHECK:STDOUT:   %impl.elem1: %.205 = impl_witness_access constants.%J.impl_witness.133, element1 [concrete = constants.%empty_tuple.type.as.J.impl.F]
// CHECK:STDOUT:   %bound_method.loc18_11: <bound method> = bound_method %x.ref, %impl.elem1
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40.f80]
// CHECK:STDOUT:   %impl.elem0: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc18_18.1: <bound method> = bound_method %int_40, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_18.2: <bound method> = bound_method %int_40, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc18_18.2(%int_40) [concrete = constants.%int_40.c3d]
// CHECK:STDOUT:   %.loc18_18.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_40.c3d]
// CHECK:STDOUT:   %.loc18_18.2: %i32 = converted %int_40, %.loc18_18.1 [concrete = constants.%int_40.c3d]
// CHECK:STDOUT:   %empty_tuple.type.as.J.impl.F.call: init %i32 = call %bound_method.loc18_11(%x.ref, %.loc18_18.2)
// CHECK:STDOUT:   return %empty_tuple.type.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.J.impl.F(%self.param: %C, %u.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   %.loc29_51.1: %empty_struct_type = as_compatible %self.ref
// CHECK:STDOUT:   %.loc29_51.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc29_51.3: init %C = as_compatible %.loc29_51.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc29_51.4: init %C = converted %self.ref, %.loc29_51.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc29_51.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.fd8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.fd8
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.37c
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.acf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.fd8) {
// CHECK:STDOUT:   %Self => constants.%Self.fd8
// CHECK:STDOUT:   %Self.as_type.loc6_14.1 => constants.%Self.as_type.e52
// CHECK:STDOUT:   %pattern_type.loc6_12 => constants.%pattern_type.9d0
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.89d
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%impl.elem0.fbe
// CHECK:STDOUT:   %pattern_type.loc6_21 => constants.%pattern_type.d8a
// CHECK:STDOUT:   %.loc6_29.1 => constants.%.f0e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%.Self.944) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%.Self.944
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.131
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.51c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.180) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.180
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.096
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.cd8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.180) {
// CHECK:STDOUT:   %Self => constants.%J.facet.180
// CHECK:STDOUT:   %Self.as_type.loc6_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type.loc6_12 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness.133
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc6_21 => constants.%pattern_type.104
// CHECK:STDOUT:   %.loc6_29.1 => constants.%.7ea
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.bf8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.bf8
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.78a
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.a5c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.bf8) {
// CHECK:STDOUT:   %Self => constants.%J.facet.bf8
// CHECK:STDOUT:   %Self.as_type.loc6_14.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc6_12 => constants.%pattern_type.0d5
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.impl_witness.62f
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc6_21 => constants.%pattern_type.0d5
// CHECK:STDOUT:   %.loc6_29.1 => constants.%.8f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_with_associated_type_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc1.24c: %J.assoc_type = assoc_entity element1, @J.WithSelf.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %assoc2: %J.assoc_type = assoc_entity element2, @J.WithSelf.%J.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type: type = fn_type @E.as.J.impl.F [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F: %E.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.as.J.impl.G.type: type = fn_type @E.as.J.impl.G [concrete]
// CHECK:STDOUT:   %E.as.J.impl.G: %E.as.J.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.399: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet) [concrete]
// CHECK:STDOUT:   %J.WithSelf.G.type.73c: type = fn_type @J.WithSelf.G, @J.WithSelf(%J.facet) [concrete]
// CHECK:STDOUT:   %.e20: type = fn_type_with_self_type %J.WithSelf.F.type.399, %J.facet [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.7d5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.228: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %.4b1: type = fn_type_with_self_type %J.WithSelf.G.type.73c, %J.facet [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8c0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.655: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.6cc: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.afe: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.9f5: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.a4d: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.55c: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.e3e: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.354: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6e4: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.50f: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.77d: %i32 = int_value 6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallBoth(%e.param: %E) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e1.patt: %pattern_type.104 = value_binding_pattern e1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc22: %E = name_ref e, %e
// CHECK:STDOUT:   %F.ref.loc22: %J.assoc_type = name_ref F, @J.WithSelf.%assoc1 [concrete = constants.%assoc1.24c]
// CHECK:STDOUT:   %impl.elem1.loc22: %.e20 = impl_witness_access constants.%J.impl_witness, element1 [concrete = constants.%E.as.J.impl.F]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc22: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc22_28.1: <bound method> = bound_method %int_2, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.2: <bound method> = bound_method %int_2, %specific_fn.loc22 [concrete = constants.%bound_method.7d5]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %i32 = call %bound_method.loc22_28.2(%int_2) [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc22_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc22_28.2: %i32 = converted %int_2, %.loc22_28.1 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %E.as.J.impl.F.call.loc22: init %i32 = call %impl.elem1.loc22(%.loc22_28.2)
// CHECK:STDOUT:   %i32.loc22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc22_29.1: %i32 = value_of_initializer %E.as.J.impl.F.call.loc22
// CHECK:STDOUT:   %.loc22_29.2: %i32 = converted %E.as.J.impl.F.call.loc22, %.loc22_29.1
// CHECK:STDOUT:   %e1: %i32 = value_binding e1, %.loc22_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e2.patt: %pattern_type.104 = value_binding_pattern e2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc23: %E = name_ref e, %e
// CHECK:STDOUT:   %G.ref.loc23: %J.assoc_type = name_ref G, @J.WithSelf.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc23: %.4b1 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%E.as.J.impl.G]
// CHECK:STDOUT:   %bound_method.loc23_25: <bound method> = bound_method %e.ref.loc23, %impl.elem2.loc23
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc23: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc23_28.1: <bound method> = bound_method %int_3, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8c0]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_28.2: <bound method> = bound_method %int_3, %specific_fn.loc23 [concrete = constants.%bound_method.655]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %i32 = call %bound_method.loc23_28.2(%int_3) [concrete = constants.%int_3.6cc]
// CHECK:STDOUT:   %.loc23_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_3.6cc]
// CHECK:STDOUT:   %.loc23_28.2: %i32 = converted %int_3, %.loc23_28.1 [concrete = constants.%int_3.6cc]
// CHECK:STDOUT:   %E.as.J.impl.G.call.loc23: init %i32 = call %bound_method.loc23_25(%e.ref.loc23, %.loc23_28.2)
// CHECK:STDOUT:   %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc23_29.1: %i32 = value_of_initializer %E.as.J.impl.G.call.loc23
// CHECK:STDOUT:   %.loc23_29.2: %i32 = converted %E.as.J.impl.G.call.loc23, %.loc23_29.1
// CHECK:STDOUT:   %e2: %i32 = value_binding e2, %.loc23_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e3.patt: %pattern_type.104 = value_binding_pattern e3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.ref.loc24: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %F.ref.loc24: %J.assoc_type = name_ref F, @J.WithSelf.%assoc1 [concrete = constants.%assoc1.24c]
// CHECK:STDOUT:   %impl.elem1.loc24: %.e20 = impl_witness_access constants.%J.impl_witness, element1 [concrete = constants.%E.as.J.impl.F]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc24: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc24_28.1: <bound method> = bound_method %int_4, %impl.elem0.loc24 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.afe]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_28.2: <bound method> = bound_method %int_4, %specific_fn.loc24 [concrete = constants.%bound_method.9f5]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24: init %i32 = call %bound_method.loc24_28.2(%int_4) [concrete = constants.%int_4.a4d]
// CHECK:STDOUT:   %.loc24_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24 [concrete = constants.%int_4.a4d]
// CHECK:STDOUT:   %.loc24_28.2: %i32 = converted %int_4, %.loc24_28.1 [concrete = constants.%int_4.a4d]
// CHECK:STDOUT:   %E.as.J.impl.F.call.loc24: init %i32 = call %impl.elem1.loc24(%.loc24_28.2)
// CHECK:STDOUT:   %i32.loc24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc24_29.1: %i32 = value_of_initializer %E.as.J.impl.F.call.loc24
// CHECK:STDOUT:   %.loc24_29.2: %i32 = converted %E.as.J.impl.F.call.loc24, %.loc24_29.1
// CHECK:STDOUT:   %e3: %i32 = value_binding e3, %.loc24_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e4.patt: %pattern_type.104 = value_binding_pattern e4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc25: %E = name_ref e, %e
// CHECK:STDOUT:   %E.ref.loc25: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %G.ref.loc25: %J.assoc_type = name_ref G, @J.WithSelf.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc25: %.4b1 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%E.as.J.impl.G]
// CHECK:STDOUT:   %bound_method.loc25_25: <bound method> = bound_method %e.ref.loc25, %impl.elem2.loc25
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc25: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc25_32.1: <bound method> = bound_method %int_5, %impl.elem0.loc25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.55c]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_32.2: <bound method> = bound_method %int_5, %specific_fn.loc25 [concrete = constants.%bound_method.e3e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25: init %i32 = call %bound_method.loc25_32.2(%int_5) [concrete = constants.%int_5.354]
// CHECK:STDOUT:   %.loc25_32.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25 [concrete = constants.%int_5.354]
// CHECK:STDOUT:   %.loc25_32.2: %i32 = converted %int_5, %.loc25_32.1 [concrete = constants.%int_5.354]
// CHECK:STDOUT:   %E.as.J.impl.G.call.loc25: init %i32 = call %bound_method.loc25_25(%e.ref.loc25, %.loc25_32.2)
// CHECK:STDOUT:   %i32.loc25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc25_33.1: %i32 = value_of_initializer %E.as.J.impl.G.call.loc25
// CHECK:STDOUT:   %.loc25_33.2: %i32 = converted %E.as.J.impl.G.call.loc25, %.loc25_33.1
// CHECK:STDOUT:   %e4: %i32 = value_binding e4, %.loc25_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e5.patt: %pattern_type.104 = value_binding_pattern e5 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.ref.loc26: %E = name_ref e, %e
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %G.ref.loc26: %J.assoc_type = name_ref G, @J.WithSelf.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc26: %.4b1 = impl_witness_access constants.%J.impl_witness, element2 [concrete = constants.%E.as.J.impl.G]
// CHECK:STDOUT:   %bound_method.loc26_25: <bound method> = bound_method %e.ref.loc26, %impl.elem2.loc26
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %impl.elem0.loc26: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc26_32.1: <bound method> = bound_method %int_6, %impl.elem0.loc26 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6e4]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem0.loc26, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_32.2: <bound method> = bound_method %int_6, %specific_fn.loc26 [concrete = constants.%bound_method.50f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26: init %i32 = call %bound_method.loc26_32.2(%int_6) [concrete = constants.%int_6.77d]
// CHECK:STDOUT:   %.loc26_32.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26 [concrete = constants.%int_6.77d]
// CHECK:STDOUT:   %.loc26_32.2: %i32 = converted %int_6, %.loc26_32.1 [concrete = constants.%int_6.77d]
// CHECK:STDOUT:   %E.as.J.impl.G.call.loc26: init %i32 = call %bound_method.loc26_25(%e.ref.loc26, %.loc26_32.2)
// CHECK:STDOUT:   %i32.loc26: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc26_33.1: %i32 = value_of_initializer %E.as.J.impl.G.call.loc26
// CHECK:STDOUT:   %.loc26_33.2: %i32 = converted %E.as.J.impl.G.call.loc26, %.loc26_33.1
// CHECK:STDOUT:   %e5: %i32 = value_binding e5, %.loc26_33.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_associated_fn_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type: type = fn_type @E.as.J.impl.F [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F: %E.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.as.J.impl.G.type: type = fn_type @E.as.J.impl.G [concrete]
// CHECK:STDOUT:   %E.as.J.impl.G: %E.as.J.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.J.impl: %E.ref as %J.ref {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %E.as.J.impl.G.decl: %E.as.J.impl.G.type = fn_decl @E.as.J.impl.G [concrete = constants.%E.as.J.impl.G] {} {}
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %E.as.J.impl.F.decl
// CHECK:STDOUT:   .G = %E.as.J.impl.G.decl
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %E.as.J.impl.F.type = name_ref F, @E.as.J.impl.%E.as.J.impl.F.decl [concrete = constants.%E.as.J.impl.F]
// CHECK:STDOUT:   %E.as.J.impl.F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_associated_constant_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.7ea: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type: type = fn_type @E.as.J.impl.F [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F: %E.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.J.impl: %Self.ref as %.loc9_20 {
// CHECK:STDOUT:   %E.as.J.impl.F.decl: %E.as.J.impl.F.type = fn_decl @E.as.J.impl.F [concrete = constants.%E.as.J.impl.F] {
// CHECK:STDOUT:     %u.param_patt: %pattern_type.104 = value_param_pattern [concrete]
// CHECK:STDOUT:     %u.patt: %pattern_type.104 = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.104 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.104 = return_slot_pattern %return.param_patt, %U.ref.loc11_19 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.facet.loc11_19: %J.type = facet_value @E.as.J.impl.%Self.ref, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:     %.loc11_19.1: %J.type = converted @E.as.J.impl.%Self.ref, %J.facet.loc11_19 [concrete = constants.%J.facet]
// CHECK:STDOUT:     %impl.elem0.loc11_19: type = impl_witness_access constants.%J.impl_witness, element0 [concrete = constants.%i32]
// CHECK:STDOUT:     %U.ref.loc11_19: type = name_ref U, %impl.elem0.loc11_19 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.2: Core.Form = init_form %U.ref.loc11_19 [concrete = constants.%.7ea]
// CHECK:STDOUT:     %u.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc11_13.1: type = splice_block %U.ref.loc11_13 [concrete = constants.%i32] {
// CHECK:STDOUT:       %J.facet.loc11_13: %J.type = facet_value @E.as.J.impl.%Self.ref, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:       %.loc11_13.2: %J.type = converted @E.as.J.impl.%Self.ref, %J.facet.loc11_13 [concrete = constants.%J.facet]
// CHECK:STDOUT:       %impl.elem0.loc11_13: type = impl_witness_access constants.%J.impl_witness, element0 [concrete = constants.%i32]
// CHECK:STDOUT:       %U.ref.loc11_13: type = name_ref U, %impl.elem0.loc11_13 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %i32 = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .U = <poisoned>
// CHECK:STDOUT:   .F = %E.as.J.impl.F.decl
// CHECK:STDOUT:   extend %.loc9_20
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   extend @E.as.J.impl.%.loc9_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.J.impl.F(%u.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %impl.elem0.loc12: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:   %bound_method.loc12_14.1: <bound method> = bound_method %u.ref, %impl.elem0.loc12
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc12, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_14.2: <bound method> = bound_method %u.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc12_14.2(%u.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_extend_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F.type: type = fn_type @E.as.J.impl.F [concrete]
// CHECK:STDOUT:   %E.as.J.impl.F: %E.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.903: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet) [concrete]
// CHECK:STDOUT:   %.451: type = fn_type_with_self_type %J.WithSelf.F.type.903, %J.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT: fn @E.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %J.facet: %J.type = facet_value constants.%E, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc14: %J.type = converted constants.%E, %J.facet [concrete = constants.%J.facet]
// CHECK:STDOUT:   %impl.elem0: %.451 = impl_witness_access constants.%J.impl_witness, element0 [concrete = constants.%E.as.J.impl.F]
// CHECK:STDOUT:   %F.ref: %.451 = name_ref F, %impl.elem0 [concrete = constants.%E.as.J.impl.F]
// CHECK:STDOUT:   %E.as.J.impl.F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_extend_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @E.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %E, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %E.G.type: type = fn_type @E.G [concrete]
// CHECK:STDOUT:   %E.G: %E.G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %E.G.decl: %E.G.type = fn_decl @E.G [concrete = constants.%E.G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern %return.param_patt, %X.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.facet: %J.type = facet_value constants.%E, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:     %.loc11_13.1: %J.type = converted constants.%E, %J.facet [concrete = constants.%J.facet]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%J.impl_witness, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %X.ref: type = name_ref X, %impl.elem0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc11_13.2: Core.Form = init_form %X.ref [concrete = constants.%.262]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT: fn @E.G() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_25.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_26: init %empty_tuple.type = converted %.loc11_25.1, %.loc11_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc11_26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_fn_use_in_def_before_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.as.J.impl.F.type: type = fn_type @A.as.J.impl.F [concrete]
// CHECK:STDOUT:   %A.as.J.impl.F: %A.as.J.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.as.J.impl.G.type: type = fn_type @A.as.J.impl.G [concrete]
// CHECK:STDOUT:   %A.as.J.impl.G: %A.as.J.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.J.impl: %A.ref as %J.ref {
// CHECK:STDOUT:   %A.as.J.impl.F.decl: %A.as.J.impl.F.type = fn_decl @A.as.J.impl.F [concrete = constants.%A.as.J.impl.F] {} {}
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %A.as.J.impl.F.decl
// CHECK:STDOUT:   .G = %A.as.J.impl.G.decl
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.J.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %A.as.J.impl.G.type = name_ref G, @A.as.J.impl.%A.as.J.impl.G.decl [concrete = constants.%A.as.J.impl.G]
// CHECK:STDOUT:   %A.as.J.impl.G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_non-type_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type @M [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.WithSelf.%Z [concrete]
// CHECK:STDOUT:   %struct: %struct_type.b.347 = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %M.impl_witness: <witness> = impl_witness @empty_tuple.type.as.M.impl.%M.impl_witness_table [concrete]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %empty_tuple.type, (%M.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.M.impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %empty_tuple.type.as.M.impl.G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   extend %.loc8_14
// CHECK:STDOUT:   witness = %M.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.M.impl.G() -> out %return.param: %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @empty_tuple.type.as.M.impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %Self.ref, (constants.%M.impl_witness) [concrete = constants.%M.facet]
// CHECK:STDOUT:   %.loc11_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc11_18 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc11_23: type = converted %.loc11_18, %as_type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access constants.%M.impl_witness, element0 [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc11_25.1: %empty_struct_type = struct_access %impl.elem0, element0 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_25.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_27: init %empty_struct_type = converted %.loc11_25.1, %.loc11_25.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc11_27
// CHECK:STDOUT: }
// CHECK:STDOUT:
