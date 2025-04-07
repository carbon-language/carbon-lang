// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  fn F[self: Self](u: i32) -> i32 { return u + 2; }
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
  fn F(u: i32) -> i32 { return u + 3; }
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
      return u + 1;
    }
    fn G[self: Self](v: i32) -> i32 {
      return v + 2;
    }
  }
}

fn CallBoth(e: E) -> i32 {
  return e.F(2) + e.G(3) + E.F(4) + e.(E.G)(5) + e.(J.G)(6);
}

fn GenericCallF[T:! J](t: T, u: T.U) -> T.U {
  return t.F(u);
}

fn CallGeneric(e: E) -> i32 {
  return GenericCallF(e, 2);
}

// --- use_constraint_on_associated_type.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! Core.Add;
  fn F(u: U) -> U;
}

fn GenericAddResult[T:! J](t: T, u: T.U) -> T.U {
  return t.F(u) + T.F(u);
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

// --- fail_todo_use_where.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F(u: U) -> U;
}

fn GenericCallFI32[T:! J where .U = i32](t: T) -> i32 {
  // CHECK:STDERR: fail_todo_use_where.carbon:[[@LINE+17]]:14: error: cannot implicitly convert expression of type `Core.IntLiteral` to `T.(J.U)` [ConversionFailure]
  // CHECK:STDERR:   return t.F(2);
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_todo_use_where.carbon:[[@LINE+14]]:14: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(T.(J.U))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return t.F(2);
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_todo_use_where.carbon:[[@LINE-10]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F(u: U) -> U;
  // CHECK:STDERR:        ^~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use_where.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `T.(J.U)` to `i32` [ConversionFailure]
  // CHECK:STDERR:   return t.F(2);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_use_where.carbon:[[@LINE+4]]:3: note: type `T.(J.U)` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return t.F(2);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
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
      return u + 1;
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
  fn F[self: Self](v: V) -> V;
}

impl () as K where .V = {.a: ()} {
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE+7]]:20: error: type `{.x: ()}` of parameter 1 in redeclaration differs from previous parameter type `{.a: ()}` [RedeclParamDiffersType]
  // CHECK:STDERR:   fn F[self: Self](v: {.x: ()}) -> {.x: ()} { return v; }
  // CHECK:STDERR:                    ^~~~~~~~~~~
  // CHECK:STDERR: fail_associated_type_in_signature_mismatch.carbon:[[@LINE-7]]:20: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
  // CHECK:STDERR:   fn F[self: Self](v: V) -> V;
  // CHECK:STDERR:                    ^~~~
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

// --- fail_todo_symbolic_associated_type_in_concrete_context.carbon

interface Z {
  let X:! type;
}

class C(T:! type) {}
class D {}

impl forall [T:! type] T as Z where .X = C(T) {}

fn F() {
  // TODO: `D.(Z.X)` is `C(D)`, but is incorrecty treated only as the symbolic
  // `C(T)` here which fails to convert.

  // CHECK:STDERR: fail_todo_symbolic_associated_type_in_concrete_context.carbon:[[@LINE+7]]:20: error: cannot implicitly convert expression of type `C(D)` to `C(T)` [ConversionFailure]
  // CHECK:STDERR:   let a: D.(Z.X) = {} as C(D);
  // CHECK:STDERR:                    ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_symbolic_associated_type_in_concrete_context.carbon:[[@LINE+4]]:20: note: type `C(D)` does not implement interface `Core.ImplicitAs(C(T))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let a: D.(Z.X) = {} as C(D);
  // CHECK:STDERR:                    ^~~~~~~~~~
  // CHECK:STDERR:
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
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.085: %J.type = facet_value %Self.as_type.3df, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d9c: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %J.facet.28c: %J.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.53a: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type.b78: type = facet_type <@J where %impl.elem0.53a = %i32> [concrete]
// CHECK:STDOUT:   %impl_witness.bcc: <witness> = impl_witness (%i32, @impl.05c.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.ea9: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.405: %F.type.ea9 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.473: %J.type = facet_value %empty_tuple.type, (%impl_witness.bcc) [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Op.type.545: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %impl_witness.01d: <witness> = impl_witness (imports.%Core.import_ref.344), @impl.2c2(%int_32) [concrete]
// CHECK:STDOUT:   %Op.type.210: type = fn_type @Op.2, @impl.2c2(%int_32) [concrete]
// CHECK:STDOUT:   %Op.c82: %Op.type.210 = struct_value () [concrete]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %i32, (%impl_witness.01d) [concrete]
// CHECK:STDOUT:   %.f1a: type = fn_type_with_self_type %Op.type.545, %Add.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.c82, @Op.2(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %CallMethod.type: type = fn_type @CallMethod [concrete]
// CHECK:STDOUT:   %CallMethod: %CallMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %.351: type = fn_type_with_self_type %F.type.c14, %J.facet.473 [concrete]
// CHECK:STDOUT:   %int_40.f80: Core.IntLiteral = int_value 40 [concrete]
// CHECK:STDOUT:   %Convert.bound.e52: <bound method> = bound_method %int_40.f80, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.7b8: <bound method> = bound_method %int_40.f80, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_40.518: %i32 = int_value 40 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J_where.type.6f2: type = facet_type <@J where %impl.elem0.53a = %C> [concrete]
// CHECK:STDOUT:   %impl_witness.192: <witness> = impl_witness (%C, @impl.f57.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.69d: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.451: %F.type.69d = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.e0b: %J.type = facet_value %C, (%impl_witness.192) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   impl_decl @impl.05c [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.53a]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%J_where.type.b78] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc8: <witness> = impl_witness (constants.%i32, @impl.05c.%F.decl) [concrete = constants.%impl_witness.bcc]
// CHECK:STDOUT:   %CallMethod.decl: %CallMethod.type = fn_decl @CallMethod [concrete = constants.%CallMethod] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:   impl_decl @impl.f57 [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc21_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc21_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.53a]
// CHECK:STDOUT:     %C.ref.loc21_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc21_13: type = where_expr %.Self [concrete = constants.%J_where.type.6f2] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref.loc21_24
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc21: <witness> = impl_witness (constants.%C, @impl.f57.%F.decl) [concrete = constants.%impl_witness.192]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.3df) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.3df) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %u.patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.d9c) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.d9c) = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.d9c) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.d9c) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_29: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access %Self.as_wit.iface0.loc5_29, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     %U.ref.loc5_29: type = name_ref U, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.d9c) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %U.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_23.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access %Self.as_wit.iface0.loc5_23.2, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:       %U.ref.loc5_23: type = name_ref U, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.d9c) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.d9c) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.d9c) = return_slot %return.param
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
// CHECK:STDOUT: impl @impl.05c: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %F.decl: %F.type.ea9 = fn_decl @F.2 [concrete = constants.%F.405] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %u.patt: %i32 = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %i32 = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc9_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.05c.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
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
// CHECK:STDOUT:   witness = file.%impl_witness.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.f57: %C.ref.loc21_6 as %.loc21_13 {
// CHECK:STDOUT:   %F.decl: %F.type.69d = fn_decl @F.3 [concrete = constants.%F.451] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %u.patt: %C = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %C = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc22_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.f57.%C.ref.loc21_6 [concrete = constants.%C]
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
// CHECK:STDOUT:   witness = file.%impl_witness.loc21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc18_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc18_11: type = converted %.loc18_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc18_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %Self.as_wit.iface0.loc5_23.1, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.3df)](%u.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.d9c)) -> @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.d9c);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type](%u.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc9_46: %.f1a = impl_witness_access constants.%impl_witness.01d, element0 [concrete = constants.%Op.c82]
// CHECK:STDOUT:   %bound_method.loc9_46.1: <bound method> = bound_method %u.ref, %impl.elem0.loc9_46
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %specific_fn.loc9_46: <specific function> = specific_function %impl.elem0.loc9_46, @Op.2(constants.%int_32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_46.2: <bound method> = bound_method %u.ref, %specific_fn.loc9_46
// CHECK:STDOUT:   %impl.elem0.loc9_48: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_48.1: <bound method> = bound_method %int_2, %impl.elem0.loc9_48 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc9_48: <specific function> = specific_function %impl.elem0.loc9_48, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_48.2: <bound method> = bound_method %int_2, %specific_fn.loc9_48 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc9_48.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_48.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_48.2: %i32 = converted %int_2, %.loc9_48.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int.sadd: init %i32 = call %bound_method.loc9_46.2(%u.ref, %.loc9_48.2)
// CHECK:STDOUT:   %.loc9_49.1: %i32 = value_of_initializer %int.sadd
// CHECK:STDOUT:   %.loc9_49.2: %i32 = converted %int.sadd, %.loc9_49.1
// CHECK:STDOUT:   return %.loc9_49.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallMethod(%x.param_patt: %empty_tuple.type) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl.elem1: %.351 = impl_witness_access constants.%impl_witness.bcc, element1 [concrete = constants.%F.405]
// CHECK:STDOUT:   %bound_method.loc13_11: <bound method> = bound_method %x.ref, %impl.elem1
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40.f80]
// CHECK:STDOUT:   %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_18.1: <bound method> = bound_method %int_40, %impl.elem0 [concrete = constants.%Convert.bound.e52]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_18.2: <bound method> = bound_method %int_40, %specific_fn [concrete = constants.%bound_method.7b8]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc13_18.2(%int_40) [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %.loc13_18.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %.loc13_18.2: %i32 = converted %int_40, %.loc13_18.1 [concrete = constants.%int_40.518]
// CHECK:STDOUT:   %F.call: init %i32 = call %bound_method.loc13_11(%x.ref, %.loc13_18.2)
// CHECK:STDOUT:   %.loc13_21.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc13_21.2: %i32 = converted %F.call, %.loc13_21.1
// CHECK:STDOUT:   return %.loc13_21.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.param_patt: %C](%u.param_patt: %C) -> %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   return %self.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.085) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type.3df
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.d9c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.28c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.473) {
// CHECK:STDOUT:   %Self => constants.%J.facet.473
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1 => constants.%impl_witness.bcc
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.e0b) {
// CHECK:STDOUT:   %Self => constants.%J.facet.e0b
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%C
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1 => constants.%impl_witness.192
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%C
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
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.085: %J.type = facet_value %Self.as_type.3df, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d9c: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %J.facet.28c: %J.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.53a: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.53a = %i32> [concrete]
// CHECK:STDOUT:   %impl_witness.56c: <witness> = impl_witness (%i32, @impl.847.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.d45: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.fdb: %F.type.d45 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.5af: %J.type = facet_value %D, (%impl_witness.56c) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Op.type.545: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %impl_witness.01d: <witness> = impl_witness (imports.%Core.import_ref.344), @impl.2c2(%int_32) [concrete]
// CHECK:STDOUT:   %Op.type.210: type = fn_type @Op.2, @impl.2c2(%int_32) [concrete]
// CHECK:STDOUT:   %Op.c82: %Op.type.210 = struct_value () [concrete]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %i32, (%impl_witness.01d) [concrete]
// CHECK:STDOUT:   %.f1a: type = fn_type_with_self_type %Op.type.545, %Add.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.c82, @Op.2(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %CallFunction.type: type = fn_type @CallFunction [concrete]
// CHECK:STDOUT:   %CallFunction: %CallFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %.1ca: type = fn_type_with_self_type %F.type.c14, %J.facet.5af [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   impl_decl @impl.847 [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.53a]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_13: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%i32, @impl.847.%F.decl) [concrete = constants.%impl_witness.56c]
// CHECK:STDOUT:   %CallFunction.decl: %CallFunction.type = fn_decl @CallFunction [concrete = constants.%CallFunction] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
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
// CHECK:STDOUT:     %u.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_17: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access %Self.as_wit.iface0.loc5_17, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_11.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access %Self.as_wit.iface0.loc5_11.2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = return_slot %return.param
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
// CHECK:STDOUT: impl @impl.847: %D.ref as %.loc10_13 {
// CHECK:STDOUT:   %F.decl: %F.type.d45 = fn_decl @F.2 [concrete = constants.%F.fdb] {
// CHECK:STDOUT:     %u.patt: %i32 = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %i32 = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %Self.as_wit.iface0.loc5_11.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c)) -> @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%u.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc11_34: %.f1a = impl_witness_access constants.%impl_witness.01d, element0 [concrete = constants.%Op.c82]
// CHECK:STDOUT:   %bound_method.loc11_34.1: <bound method> = bound_method %u.ref, %impl.elem0.loc11_34
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %specific_fn.loc11_34: <specific function> = specific_function %impl.elem0.loc11_34, @Op.2(constants.%int_32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_34.2: <bound method> = bound_method %u.ref, %specific_fn.loc11_34
// CHECK:STDOUT:   %impl.elem0.loc11_36: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_36.1: <bound method> = bound_method %int_3, %impl.elem0.loc11_36 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc11_36: <specific function> = specific_function %impl.elem0.loc11_36, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_36.2: <bound method> = bound_method %int_3, %specific_fn.loc11_36 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc11_36.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc11_36.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc11_36.2: %i32 = converted %int_3, %.loc11_36.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int.sadd: init %i32 = call %bound_method.loc11_34.2(%u.ref, %.loc11_36.2)
// CHECK:STDOUT:   %.loc11_37.1: %i32 = value_of_initializer %int.sadd
// CHECK:STDOUT:   %.loc11_37.2: %i32 = converted %int.sadd, %.loc11_37.1
// CHECK:STDOUT:   return %.loc11_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFunction() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value constants.%D, (constants.%impl_witness.56c) [concrete = constants.%J.facet.5af]
// CHECK:STDOUT:   %.loc15_11: %J.type = converted %D.ref, %J.facet [concrete = constants.%J.facet.5af]
// CHECK:STDOUT:   %as_wit.iface0: <witness> = facet_access_witness %.loc15_11, element0 [concrete = constants.%impl_witness.56c]
// CHECK:STDOUT:   %impl.elem1: %.1ca = impl_witness_access %as_wit.iface0, element1 [concrete = constants.%F.fdb]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_18.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_18.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method.1da]
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
// CHECK:STDOUT: specific @U(constants.%J.facet.085) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.d9c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.28c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.5af) {
// CHECK:STDOUT:   %Self => constants.%J.facet.5af
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%impl_witness.56c
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
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
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.085e97.1: %J.type = facet_value %Self.as_type.3df, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d9c3a6.1: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %G.type.285: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.e4b: %G.type.285 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc2: %J.assoc_type = assoc_entity element2, @J.%G.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %J.facet.28c: %J.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.53a: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.53a = %i32> [concrete]
// CHECK:STDOUT:   %impl_witness.71e: <witness> = impl_witness (%i32, @impl.2dd.%F.decl, @impl.2dd.%G.decl) [concrete]
// CHECK:STDOUT:   %F.type.e5e: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c81: %F.type.e5e = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.89e: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.105: %G.type.89e = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.ccb: %J.type = facet_value %E, (%impl_witness.71e) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Op.type.545: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %impl_witness.01d: <witness> = impl_witness (imports.%Core.import_ref.344), @impl.2c2(%int_32) [concrete]
// CHECK:STDOUT:   %Op.type.210: type = fn_type @Op.2, @impl.2c2(%int_32) [concrete]
// CHECK:STDOUT:   %Op.c82: %Op.type.210 = struct_value () [concrete]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %i32, (%impl_witness.01d) [concrete]
// CHECK:STDOUT:   %.f1a: type = fn_type_with_self_type %Op.type.545, %Add.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.c82, @Op.2(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %CallBoth.type: type = fn_type @CallBoth [concrete]
// CHECK:STDOUT:   %CallBoth: %CallBoth.type = struct_value () [concrete]
// CHECK:STDOUT:   %.dea: type = fn_type_with_self_type %F.type.c14, %J.facet.ccb [concrete]
// CHECK:STDOUT:   %.4bc: type = fn_type_with_self_type %G.type.285, %J.facet.ccb [concrete]
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
// CHECK:STDOUT:   %T: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0: <witness> = facet_access_witness %T, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.085e97.2: %J.type = facet_value %T.as_type, (%T.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d9c3a6.2: type = impl_witness_access %T.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %GenericCallF.type: type = fn_type @GenericCallF [concrete]
// CHECK:STDOUT:   %GenericCallF: %GenericCallF.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d9a: <witness> = require_complete_type %impl.elem0.d9c3a6.2 [symbolic]
// CHECK:STDOUT:   %require_complete.2a5: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.4d2: type = fn_type_with_self_type %F.type.c14, %J.facet.085e97.2 [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.4d2 = impl_witness_access %T.as_wit.iface0, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem1, @F.1(%J.facet.085e97.2) [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericCallF.specific_fn: <specific function> = specific_function %GenericCallF, @GenericCallF(%J.facet.ccb) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
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
// CHECK:STDOUT:     .CallBoth = %CallBoth.decl
// CHECK:STDOUT:     .GenericCallF = %GenericCallF.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   %CallBoth.decl: %CallBoth.type = fn_decl @CallBoth [concrete = constants.%CallBoth] {
// CHECK:STDOUT:     %e.patt: %E = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: %E = value_param_pattern %e.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref.loc20: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = bind_name e, %e.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericCallF.decl: %GenericCallF.type = fn_decl @GenericCallF [concrete = constants.%GenericCallF] {
// CHECK:STDOUT:     %T.patt.loc24_17.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_17.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: @GenericCallF.%T.as_type.loc24_27.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @GenericCallF.%T.as_type.loc24_27.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %u.patt: @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc24_41: %J.type = name_ref T, %T.loc24_17.1 [symbolic = %T.loc24_17.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc24_42: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %T.as_type.loc24_42: type = facet_access_type %T.ref.loc24_41 [symbolic = %T.as_type.loc24_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc24_42: type = converted %T.ref.loc24_41, %T.as_type.loc24_42 [symbolic = %T.as_type.loc24_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc24_42: <witness> = facet_access_witness %T.ref.loc24_41, element0 [symbolic = %T.as_wit.iface0.loc24_34.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc24_42: type = impl_witness_access %T.as_wit.iface0.loc24_42, element0 [symbolic = %impl.elem0.loc24_34.2 (constants.%impl.elem0.d9c3a6.2)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc24_17.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc24_17.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericCallF.%T.as_type.loc24_27.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc24_27.1: type = splice_block %.loc24_27.2 [symbolic = %T.as_type.loc24_27.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc24_27: %J.type = name_ref T, %T.loc24_17.1 [symbolic = %T.loc24_17.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc24_27.1: type = facet_access_type %T.ref.loc24_27 [symbolic = %T.as_type.loc24_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc24_27.2: type = converted %T.ref.loc24_27, %T.as_type.loc24_27.1 [symbolic = %T.as_type.loc24_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericCallF.%T.as_type.loc24_27.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = value_param call_param1
// CHECK:STDOUT:     %.loc24_34.1: type = splice_block %impl.elem0.loc24_34.1 [symbolic = %impl.elem0.loc24_34.2 (constants.%impl.elem0.d9c3a6.2)] {
// CHECK:STDOUT:       %T.ref.loc24_33: %J.type = name_ref T, %T.loc24_17.1 [symbolic = %T.loc24_17.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc24_34: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:       %T.as_type.loc24_34: type = facet_access_type %T.ref.loc24_33 [symbolic = %T.as_type.loc24_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc24_34.2: type = converted %T.ref.loc24_33, %T.as_type.loc24_34 [symbolic = %T.as_type.loc24_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %T.as_wit.iface0.loc24_34.1: <witness> = facet_access_witness %T.ref.loc24_33, element0 [symbolic = %T.as_wit.iface0.loc24_34.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc24_34.1: type = impl_witness_access %T.as_wit.iface0.loc24_34.1, element0 [symbolic = %impl.elem0.loc24_34.2 (constants.%impl.elem0.d9c3a6.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %e.patt: %E = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: %E = value_param_pattern %e.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:     %u.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c3a6.1) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c3a6.1) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c3a6.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c3a6.1) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_17: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access %Self.as_wit.iface0.loc5_17, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c3a6.1) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c3a6.1)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_11.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access %Self.as_wit.iface0.loc5_11.2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c3a6.1) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c3a6.1) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c3a6.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:   %G.decl: %G.type.285 = fn_decl @G.1 [concrete = constants.%G.e4b] {
// CHECK:STDOUT:     %self.patt: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %v.patt: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.d9c3a6.1) = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.d9c3a6.1) = value_param_pattern %v.patt, call_param1
// CHECK:STDOUT:     %return.patt: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.d9c3a6.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.d9c3a6.1) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc6_29: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc6_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc6_29: type = impl_witness_access %Self.as_wit.iface0.loc6_29, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:     %U.ref.loc6_29: type = name_ref U, %impl.elem0.loc6_29 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:     %self.param: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc6_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.d9c3a6.1) = value_param call_param1
// CHECK:STDOUT:     %.loc6_23: type = splice_block %U.ref.loc6_23 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.d9c3a6.1)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc6_23.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc6_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc6_23.2: type = impl_witness_access %Self.as_wit.iface0.loc6_23.2, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:       %U.ref.loc6_23: type = name_ref U, %impl.elem0.loc6_23.2 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.d9c3a6.1) = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.d9c3a6.1) = out_param call_param2
// CHECK:STDOUT:     %return: ref @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.d9c3a6.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc2: %J.assoc_type = assoc_entity element2, %G.decl [concrete = constants.%assoc2]
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
// CHECK:STDOUT: impl @impl.2dd: %Self.ref as %.loc10_20 {
// CHECK:STDOUT:   %F.decl: %F.type.e5e = fn_decl @F.2 [concrete = constants.%F.c81] {
// CHECK:STDOUT:     %u.patt: %i32 = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %i32 = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:   %G.decl: %G.type.89e = fn_decl @G.2 [concrete = constants.%G.105] {
// CHECK:STDOUT:     %self.patt: %E = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %E = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %v.patt: %i32 = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: %i32 = value_param_pattern %v.patt, call_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param2
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
// CHECK:STDOUT:   witness = @E.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @impl.2dd [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_26: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.53a]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_20: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%i32, @impl.2dd.%F.decl, @impl.2dd.%G.decl) [concrete = constants.%impl_witness.71e]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   extend @impl.2dd.%.loc10_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %Self.as_wit.iface0.loc5_11.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c3a6.1)) -> @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c3a6.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc6_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc6_23.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc6_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc6_23.1: type = impl_witness_access %Self.as_wit.iface0.loc6_23.1, element0 [symbolic = %impl.elem0.loc6_23.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.1.%Self.as_type.loc6_14.1 (%Self.as_type.3df)](%v.param_patt: @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.d9c3a6.1)) -> @G.1.%impl.elem0.loc6_23.1 (%impl.elem0.d9c3a6.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%u.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %i32 = name_ref u, %u
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc12_16: %.f1a = impl_witness_access constants.%impl_witness.01d, element0 [concrete = constants.%Op.c82]
// CHECK:STDOUT:   %bound_method.loc12_16.1: <bound method> = bound_method %u.ref, %impl.elem0.loc12_16
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %specific_fn.loc12_16: <specific function> = specific_function %impl.elem0.loc12_16, @Op.2(constants.%int_32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_16.2: <bound method> = bound_method %u.ref, %specific_fn.loc12_16
// CHECK:STDOUT:   %impl.elem0.loc12_18: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_18.1: <bound method> = bound_method %int_1, %impl.elem0.loc12_18 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc12_18: <specific function> = specific_function %impl.elem0.loc12_18, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_18.2: <bound method> = bound_method %int_1, %specific_fn.loc12_18 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc12_18.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_18.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_18.2: %i32 = converted %int_1, %.loc12_18.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int.sadd: init %i32 = call %bound_method.loc12_16.2(%u.ref, %.loc12_18.2)
// CHECK:STDOUT:   %.loc12_19.1: %i32 = value_of_initializer %int.sadd
// CHECK:STDOUT:   %.loc12_19.2: %i32 = converted %int.sadd, %.loc12_19.1
// CHECK:STDOUT:   return %.loc12_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[%self.param_patt: %E](%v.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: %i32 = name_ref v, %v
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc15_16: %.f1a = impl_witness_access constants.%impl_witness.01d, element0 [concrete = constants.%Op.c82]
// CHECK:STDOUT:   %bound_method.loc15_16.1: <bound method> = bound_method %v.ref, %impl.elem0.loc15_16
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %specific_fn.loc15_16: <specific function> = specific_function %impl.elem0.loc15_16, @Op.2(constants.%int_32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_16.2: <bound method> = bound_method %v.ref, %specific_fn.loc15_16
// CHECK:STDOUT:   %impl.elem0.loc15_18: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_18.1: <bound method> = bound_method %int_2, %impl.elem0.loc15_18 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc15_18: <specific function> = specific_function %impl.elem0.loc15_18, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_18.2: <bound method> = bound_method %int_2, %specific_fn.loc15_18 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc15_18.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc15_18.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc15_18.2: %i32 = converted %int_2, %.loc15_18.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int.sadd: init %i32 = call %bound_method.loc15_16.2(%v.ref, %.loc15_18.2)
// CHECK:STDOUT:   %.loc15_19.1: %i32 = value_of_initializer %int.sadd
// CHECK:STDOUT:   %.loc15_19.2: %i32 = converted %int.sadd, %.loc15_19.1
// CHECK:STDOUT:   return %.loc15_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallBoth(%e.param_patt: %E) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %e.ref.loc21_10: %E = name_ref e, %e
// CHECK:STDOUT:   %F.ref.loc21_11: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl.elem1.loc21_11: %.dea = impl_witness_access constants.%impl_witness.71e, element1 [concrete = constants.%F.c81]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc21_14: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_14.1: <bound method> = bound_method %int_2, %impl.elem0.loc21_14 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc21_14: <specific function> = specific_function %impl.elem0.loc21_14, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_14.2: <bound method> = bound_method %int_2, %specific_fn.loc21_14 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc21_14: init %i32 = call %bound_method.loc21_14.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc21_14.1: %i32 = value_of_initializer %int.convert_checked.loc21_14 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc21_14.2: %i32 = converted %int_2, %.loc21_14.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %F.call.loc21_15: init %i32 = call %impl.elem1.loc21_11(%.loc21_14.2)
// CHECK:STDOUT:   %e.ref.loc21_19: %E = name_ref e, %e
// CHECK:STDOUT:   %G.ref.loc21_20: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc21_20: %.4bc = impl_witness_access constants.%impl_witness.71e, element2 [concrete = constants.%G.105]
// CHECK:STDOUT:   %bound_method.loc21_20: <bound method> = bound_method %e.ref.loc21_19, %impl.elem2.loc21_20
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc21_23: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_23.1: <bound method> = bound_method %int_3, %impl.elem0.loc21_23 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc21_23: <specific function> = specific_function %impl.elem0.loc21_23, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_23.2: <bound method> = bound_method %int_3, %specific_fn.loc21_23 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc21_23: init %i32 = call %bound_method.loc21_23.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc21_23.1: %i32 = value_of_initializer %int.convert_checked.loc21_23 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc21_23.2: %i32 = converted %int_3, %.loc21_23.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %G.call.loc21_24: init %i32 = call %bound_method.loc21_20(%e.ref.loc21_19, %.loc21_23.2)
// CHECK:STDOUT:   %impl.elem0.loc21_17: %.f1a = impl_witness_access constants.%impl_witness.01d, element0 [concrete = constants.%Op.c82]
// CHECK:STDOUT:   %bound_method.loc21_17.1: <bound method> = bound_method %F.call.loc21_15, %impl.elem0.loc21_17
// CHECK:STDOUT:   %specific_fn.loc21_17: <specific function> = specific_function %impl.elem0.loc21_17, @Op.2(constants.%int_32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_17.2: <bound method> = bound_method %F.call.loc21_15, %specific_fn.loc21_17
// CHECK:STDOUT:   %.loc21_15.1: %i32 = value_of_initializer %F.call.loc21_15
// CHECK:STDOUT:   %.loc21_15.2: %i32 = converted %F.call.loc21_15, %.loc21_15.1
// CHECK:STDOUT:   %.loc21_24.1: %i32 = value_of_initializer %G.call.loc21_24
// CHECK:STDOUT:   %.loc21_24.2: %i32 = converted %G.call.loc21_24, %.loc21_24.1
// CHECK:STDOUT:   %int.sadd.loc21_17: init %i32 = call %bound_method.loc21_17.2(%.loc21_15.2, %.loc21_24.2)
// CHECK:STDOUT:   %E.ref.loc21_28: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %F.ref.loc21_29: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl.elem1.loc21_29: %.dea = impl_witness_access constants.%impl_witness.71e, element1 [concrete = constants.%F.c81]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc21_32: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_32.1: <bound method> = bound_method %int_4, %impl.elem0.loc21_32 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc21_32: <specific function> = specific_function %impl.elem0.loc21_32, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_32.2: <bound method> = bound_method %int_4, %specific_fn.loc21_32 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc21_32: init %i32 = call %bound_method.loc21_32.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc21_32.1: %i32 = value_of_initializer %int.convert_checked.loc21_32 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc21_32.2: %i32 = converted %int_4, %.loc21_32.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %F.call.loc21_33: init %i32 = call %impl.elem1.loc21_29(%.loc21_32.2)
// CHECK:STDOUT:   %impl.elem0.loc21_26: %.f1a = impl_witness_access constants.%impl_witness.01d, element0 [concrete = constants.%Op.c82]
// CHECK:STDOUT:   %bound_method.loc21_26.1: <bound method> = bound_method %int.sadd.loc21_17, %impl.elem0.loc21_26
// CHECK:STDOUT:   %specific_fn.loc21_26: <specific function> = specific_function %impl.elem0.loc21_26, @Op.2(constants.%int_32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_26.2: <bound method> = bound_method %int.sadd.loc21_17, %specific_fn.loc21_26
// CHECK:STDOUT:   %.loc21_17.1: %i32 = value_of_initializer %int.sadd.loc21_17
// CHECK:STDOUT:   %.loc21_17.2: %i32 = converted %int.sadd.loc21_17, %.loc21_17.1
// CHECK:STDOUT:   %.loc21_33.1: %i32 = value_of_initializer %F.call.loc21_33
// CHECK:STDOUT:   %.loc21_33.2: %i32 = converted %F.call.loc21_33, %.loc21_33.1
// CHECK:STDOUT:   %int.sadd.loc21_26: init %i32 = call %bound_method.loc21_26.2(%.loc21_17.2, %.loc21_33.2)
// CHECK:STDOUT:   %e.ref.loc21_37: %E = name_ref e, %e
// CHECK:STDOUT:   %E.ref.loc21_40: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %G.ref.loc21_41: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc21_41: %.4bc = impl_witness_access constants.%impl_witness.71e, element2 [concrete = constants.%G.105]
// CHECK:STDOUT:   %bound_method.loc21_38: <bound method> = bound_method %e.ref.loc21_37, %impl.elem2.loc21_41
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc21_45: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_45.1: <bound method> = bound_method %int_5, %impl.elem0.loc21_45 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc21_45: <specific function> = specific_function %impl.elem0.loc21_45, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_45.2: <bound method> = bound_method %int_5, %specific_fn.loc21_45 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %int.convert_checked.loc21_45: init %i32 = call %bound_method.loc21_45.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc21_45.1: %i32 = value_of_initializer %int.convert_checked.loc21_45 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc21_45.2: %i32 = converted %int_5, %.loc21_45.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %G.call.loc21_46: init %i32 = call %bound_method.loc21_38(%e.ref.loc21_37, %.loc21_45.2)
// CHECK:STDOUT:   %impl.elem0.loc21_35: %.f1a = impl_witness_access constants.%impl_witness.01d, element0 [concrete = constants.%Op.c82]
// CHECK:STDOUT:   %bound_method.loc21_35.1: <bound method> = bound_method %int.sadd.loc21_26, %impl.elem0.loc21_35
// CHECK:STDOUT:   %specific_fn.loc21_35: <specific function> = specific_function %impl.elem0.loc21_35, @Op.2(constants.%int_32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_35.2: <bound method> = bound_method %int.sadd.loc21_26, %specific_fn.loc21_35
// CHECK:STDOUT:   %.loc21_26.1: %i32 = value_of_initializer %int.sadd.loc21_26
// CHECK:STDOUT:   %.loc21_26.2: %i32 = converted %int.sadd.loc21_26, %.loc21_26.1
// CHECK:STDOUT:   %.loc21_46.1: %i32 = value_of_initializer %G.call.loc21_46
// CHECK:STDOUT:   %.loc21_46.2: %i32 = converted %G.call.loc21_46, %.loc21_46.1
// CHECK:STDOUT:   %int.sadd.loc21_35: init %i32 = call %bound_method.loc21_35.2(%.loc21_26.2, %.loc21_46.2)
// CHECK:STDOUT:   %e.ref.loc21_50: %E = name_ref e, %e
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %G.ref.loc21_54: %J.assoc_type = name_ref G, @J.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2.loc21_51: %.4bc = impl_witness_access constants.%impl_witness.71e, element2 [concrete = constants.%G.105]
// CHECK:STDOUT:   %bound_method.loc21_51: <bound method> = bound_method %e.ref.loc21_50, %impl.elem2.loc21_51
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %impl.elem0.loc21_58: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_58.1: <bound method> = bound_method %int_6, %impl.elem0.loc21_58 [concrete = constants.%Convert.bound.ce9]
// CHECK:STDOUT:   %specific_fn.loc21_58: <specific function> = specific_function %impl.elem0.loc21_58, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_58.2: <bound method> = bound_method %int_6, %specific_fn.loc21_58 [concrete = constants.%bound_method.efa]
// CHECK:STDOUT:   %int.convert_checked.loc21_58: init %i32 = call %bound_method.loc21_58.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc21_58.1: %i32 = value_of_initializer %int.convert_checked.loc21_58 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc21_58.2: %i32 = converted %int_6, %.loc21_58.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %G.call.loc21_59: init %i32 = call %bound_method.loc21_51(%e.ref.loc21_50, %.loc21_58.2)
// CHECK:STDOUT:   %impl.elem0.loc21_48: %.f1a = impl_witness_access constants.%impl_witness.01d, element0 [concrete = constants.%Op.c82]
// CHECK:STDOUT:   %bound_method.loc21_48.1: <bound method> = bound_method %int.sadd.loc21_35, %impl.elem0.loc21_48
// CHECK:STDOUT:   %specific_fn.loc21_48: <specific function> = specific_function %impl.elem0.loc21_48, @Op.2(constants.%int_32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_48.2: <bound method> = bound_method %int.sadd.loc21_35, %specific_fn.loc21_48
// CHECK:STDOUT:   %.loc21_35.1: %i32 = value_of_initializer %int.sadd.loc21_35
// CHECK:STDOUT:   %.loc21_35.2: %i32 = converted %int.sadd.loc21_35, %.loc21_35.1
// CHECK:STDOUT:   %.loc21_59.1: %i32 = value_of_initializer %G.call.loc21_59
// CHECK:STDOUT:   %.loc21_59.2: %i32 = converted %G.call.loc21_59, %.loc21_59.1
// CHECK:STDOUT:   %int.sadd.loc21_48: init %i32 = call %bound_method.loc21_48.2(%.loc21_35.2, %.loc21_59.2)
// CHECK:STDOUT:   %.loc21_60.1: %i32 = value_of_initializer %int.sadd.loc21_48
// CHECK:STDOUT:   %.loc21_60.2: %i32 = converted %int.sadd.loc21_48, %.loc21_60.1
// CHECK:STDOUT:   return %.loc21_60.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallF(%T.loc24_17.1: %J.type) {
// CHECK:STDOUT:   %T.loc24_17.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc24_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc24_17.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc24_27.2: type = facet_access_type %T.loc24_17.2 [symbolic = %T.as_type.loc24_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc24_34.2: <witness> = facet_access_witness %T.loc24_17.2, element0 [symbolic = %T.as_wit.iface0.loc24_34.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc24_34.2: type = impl_witness_access %T.as_wit.iface0.loc24_34.2, element0 [symbolic = %impl.elem0.loc24_34.2 (constants.%impl.elem0.d9c3a6.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc24_38: <witness> = require_complete_type @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) [symbolic = %require_complete.loc24_38 (constants.%require_complete.d9a)]
// CHECK:STDOUT:   %require_complete.loc24_25: <witness> = require_complete_type @GenericCallF.%T.as_type.loc24_27.2 (%T.as_type) [symbolic = %require_complete.loc24_25 (constants.%require_complete.2a5)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type.loc24_27.2, (%T.as_wit.iface0.loc24_34.2) [symbolic = %J.facet (constants.%J.facet.085e97.2)]
// CHECK:STDOUT:   %.loc25_11.2: type = fn_type_with_self_type constants.%F.type.c14, %J.facet [symbolic = %.loc25_11.2 (constants.%.4d2)]
// CHECK:STDOUT:   %impl.elem1.loc25_11.2: @GenericCallF.%.loc25_11.2 (%.4d2) = impl_witness_access %T.as_wit.iface0.loc24_34.2, element1 [symbolic = %impl.elem1.loc25_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_11.2: <specific function> = specific_impl_function %impl.elem1.loc25_11.2, @F.1(%J.facet) [symbolic = %specific_impl_fn.loc25_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc24_17.1: %J.type](%t.param_patt: @GenericCallF.%T.as_type.loc24_27.2 (%T.as_type), %u.param_patt: @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2)) -> @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallF.%T.as_type.loc24_27.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc25: type = facet_access_type constants.%T [symbolic = %T.as_type.loc24_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc25_11.1: type = converted constants.%T, %T.as_type.loc25 [symbolic = %T.as_type.loc24_27.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc25: <witness> = facet_access_witness constants.%T, element0 [symbolic = %T.as_wit.iface0.loc24_34.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem1.loc25_11.1: @GenericCallF.%.loc25_11.2 (%.4d2) = impl_witness_access %T.as_wit.iface0.loc25, element1 [symbolic = %impl.elem1.loc25_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref: @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = name_ref u, %u
// CHECK:STDOUT:     %as_wit.iface0.1: <witness> = facet_access_witness constants.%J.facet.085e97.2, element0 [symbolic = %T.as_wit.iface0.loc24_34.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.1: type = impl_witness_access constants.%T.as_wit.iface0, element0 [symbolic = %impl.elem0.loc24_34.2 (constants.%impl.elem0.d9c3a6.2)]
// CHECK:STDOUT:     %as_wit.iface0.2: <witness> = facet_access_witness constants.%J.facet.085e97.2, element0 [symbolic = %T.as_wit.iface0.loc24_34.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.2: type = impl_witness_access constants.%T.as_wit.iface0, element0 [symbolic = %impl.elem0.loc24_34.2 (constants.%impl.elem0.d9c3a6.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc25_11.1: <specific function> = specific_impl_function %impl.elem1.loc25_11.1, @F.1(constants.%J.facet.085e97.2) [symbolic = %specific_impl_fn.loc25_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc25_15: init @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = call %specific_impl_fn.loc25_11.1(%u.ref)
// CHECK:STDOUT:     %.loc25_16.1: @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = value_of_initializer %.loc25_15
// CHECK:STDOUT:     %.loc25_16.2: @GenericCallF.%impl.elem0.loc24_34.2 (%impl.elem0.d9c3a6.2) = converted %.loc25_15, %.loc25_16.1
// CHECK:STDOUT:     return %.loc25_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGeneric(%e.param_patt: %E) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %GenericCallF.ref: %GenericCallF.type = name_ref GenericCallF, file.%GenericCallF.decl [concrete = constants.%GenericCallF]
// CHECK:STDOUT:   %e.ref: %E = name_ref e, %e
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %J.facet.loc29_27.1: %J.type = facet_value constants.%E, (constants.%impl_witness.71e) [concrete = constants.%J.facet.ccb]
// CHECK:STDOUT:   %.loc29_27.1: %J.type = converted constants.%E, %J.facet.loc29_27.1 [concrete = constants.%J.facet.ccb]
// CHECK:STDOUT:   %J.facet.loc29_27.2: %J.type = facet_value constants.%E, (constants.%impl_witness.71e) [concrete = constants.%J.facet.ccb]
// CHECK:STDOUT:   %.loc29_27.2: %J.type = converted constants.%E, %J.facet.loc29_27.2 [concrete = constants.%J.facet.ccb]
// CHECK:STDOUT:   %GenericCallF.specific_fn: <specific function> = specific_function %GenericCallF.ref, @GenericCallF(constants.%J.facet.ccb) [concrete = constants.%GenericCallF.specific_fn]
// CHECK:STDOUT:   %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc29_26.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29_26.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc29_26.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc29_26.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc29_26.2: %i32 = converted %int_2, %.loc29_26.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %GenericCallF.call: init %i32 = call %GenericCallF.specific_fn(%e.ref, %.loc29_26.2)
// CHECK:STDOUT:   %.loc29_28.1: %i32 = value_of_initializer %GenericCallF.call
// CHECK:STDOUT:   %.loc29_28.2: %i32 = converted %GenericCallF.call, %.loc29_28.1
// CHECK:STDOUT:   return %.loc29_28.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.085e97.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.d9c3a6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc6_14.1 => constants.%Self.as_type.3df
// CHECK:STDOUT:   %Self.as_wit.iface0.loc6_23.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%impl.elem0.d9c3a6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.28c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.ccb) {
// CHECK:STDOUT:   %Self => constants.%J.facet.ccb
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%impl_witness.71e
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%J.facet.ccb) {
// CHECK:STDOUT:   %Self => constants.%J.facet.ccb
// CHECK:STDOUT:   %Self.as_type.loc6_14.1 => constants.%E
// CHECK:STDOUT:   %Self.as_wit.iface0.loc6_23.1 => constants.%impl_witness.71e
// CHECK:STDOUT:   %impl.elem0.loc6_23.1 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.085e97.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallF(constants.%T) {
// CHECK:STDOUT:   %T.loc24_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc24_17.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc24_27.2 => constants.%T.as_type
// CHECK:STDOUT:   %T.as_wit.iface0.loc24_34.2 => constants.%T.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc24_34.2 => constants.%impl.elem0.d9c3a6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.085e97.2) {
// CHECK:STDOUT:   %Self => constants.%J.facet.085e97.2
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%T.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.d9c3a6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(@GenericCallF.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallF(constants.%J.facet.ccb) {
// CHECK:STDOUT:   %T.loc24_17.2 => constants.%J.facet.ccb
// CHECK:STDOUT:   %T.patt.loc24_17.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc24_27.2 => constants.%E
// CHECK:STDOUT:   %T.as_wit.iface0.loc24_34.2 => constants.%impl_witness.71e
// CHECK:STDOUT:   %impl.elem0.loc24_34.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc24_38 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc24_25 => constants.%complete_type.357
// CHECK:STDOUT:   %J.facet => constants.%J.facet.ccb
// CHECK:STDOUT:   %.loc25_11.2 => constants.%.dea
// CHECK:STDOUT:   %impl.elem1.loc25_11.2 => constants.%F.c81
// CHECK:STDOUT:   %specific_impl_fn.loc25_11.2 => constants.%F.c81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_constraint_on_associated_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.304: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.085e97.1: %J.type = facet_value %Self.as_type.3df, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.c57618.1: %Add.type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %as_type.6b96b1.1: type = facet_access_type %impl.elem0.c57618.1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %T: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0: <witness> = facet_access_witness %T, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.085e97.2: %J.type = facet_value %T.as_type, (%T.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.c57618.2: %Add.type = impl_witness_access %T.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %as_type.6b96b1.2: type = facet_access_type %impl.elem0.c57618.2 [symbolic]
// CHECK:STDOUT:   %GenericAddResult.type: type = fn_type @GenericAddResult [concrete]
// CHECK:STDOUT:   %GenericAddResult: %GenericAddResult.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ec7: <witness> = require_complete_type %as_type.6b96b1.2 [symbolic]
// CHECK:STDOUT:   %require_complete.2a5: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.4d2: type = fn_type_with_self_type %F.type, %J.facet.085e97.2 [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.4d2 = impl_witness_access %T.as_wit.iface0, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5d8: <specific function> = specific_impl_function %impl.elem1, @F(%J.facet.085e97.2) [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op [concrete]
// CHECK:STDOUT:   %as_wit.iface0: <witness> = facet_access_witness %impl.elem0.c57618.2, element0 [symbolic]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %as_type.6b96b1.2, (%as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %.5a3: type = fn_type_with_self_type %Op.type, %Add.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.ab3: %.5a3 = impl_witness_access %as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e8e: <specific function> = specific_impl_function %impl.elem0.ab3, @Op(%Add.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .GenericAddResult = %GenericAddResult.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %GenericAddResult.decl: %GenericAddResult.type = fn_decl @GenericAddResult [concrete = constants.%GenericAddResult] {
// CHECK:STDOUT:     %T.patt.loc8_21.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_21.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: @GenericAddResult.%T.as_type.loc8_31.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @GenericAddResult.%T.as_type.loc8_31.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %u.patt: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_45: %J.type = name_ref T, %T.loc8_21.1 [symbolic = %T.loc8_21.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc8_46: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.304]
// CHECK:STDOUT:     %T.as_type.loc8_46: type = facet_access_type %T.ref.loc8_45 [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8_46.1: type = converted %T.ref.loc8_45, %T.as_type.loc8_46 [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc8_46: <witness> = facet_access_witness %T.ref.loc8_45, element0 [symbolic = %T.as_wit.iface0.loc8_38.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc8_46: %Add.type = impl_witness_access %T.as_wit.iface0.loc8_46, element0 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:     %as_type.loc8_46: type = facet_access_type %impl.elem0.loc8_46 [symbolic = %as_type.loc8_38.2 (constants.%as_type.6b96b1.2)]
// CHECK:STDOUT:     %.loc8_46.2: type = converted %impl.elem0.loc8_46, %as_type.loc8_46 [symbolic = %as_type.loc8_38.2 (constants.%as_type.6b96b1.2)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc8_21.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_21.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericAddResult.%T.as_type.loc8_31.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_31.1: type = splice_block %.loc8_31.2 [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc8_31: %J.type = name_ref T, %T.loc8_21.1 [symbolic = %T.loc8_21.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8_31.1: type = facet_access_type %T.ref.loc8_31 [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_31.2: type = converted %T.ref.loc8_31, %T.as_type.loc8_31.1 [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericAddResult.%T.as_type.loc8_31.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = value_param call_param1
// CHECK:STDOUT:     %.loc8_38.1: type = splice_block %.loc8_38.3 [symbolic = %as_type.loc8_38.2 (constants.%as_type.6b96b1.2)] {
// CHECK:STDOUT:       %T.ref.loc8_37: %J.type = name_ref T, %T.loc8_21.1 [symbolic = %T.loc8_21.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc8_38: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.304]
// CHECK:STDOUT:       %T.as_type.loc8_38: type = facet_access_type %T.ref.loc8_37 [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_38.2: type = converted %T.ref.loc8_37, %T.as_type.loc8_38 [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %T.as_wit.iface0.loc8_38.1: <witness> = facet_access_witness %T.ref.loc8_37, element0 [symbolic = %T.as_wit.iface0.loc8_38.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc8_38.1: %Add.type = impl_witness_access %T.as_wit.iface0.loc8_38.1, element0 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:       %as_type.loc8_38.1: type = facet_access_type %impl.elem0.loc8_38.1 [symbolic = %as_type.loc8_38.2 (constants.%as_type.6b96b1.2)]
// CHECK:STDOUT:       %.loc8_38.3: type = converted %impl.elem0.loc8_38.1, %as_type.loc8_38.1 [symbolic = %as_type.loc8_38.2 (constants.%as_type.6b96b1.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: %Add.type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.304]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %u.patt: @F.%as_type (%as_type.6b96b1.1) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.%as_type (%as_type.6b96b1.1) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%as_type (%as_type.6b96b1.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%as_type (%as_type.6b96b1.1) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_17: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_17: %Add.type = impl_witness_access %Self.as_wit.iface0.loc5_17, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.c57618.1)]
// CHECK:STDOUT:     %U.ref.loc5_17: %Add.type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.c57618.1)]
// CHECK:STDOUT:     %U.as_type.loc5_17: type = facet_access_type %U.ref.loc5_17 [symbolic = %as_type (constants.%as_type.6b96b1.1)]
// CHECK:STDOUT:     %.loc5_17: type = converted %U.ref.loc5_17, %U.as_type.loc5_17 [symbolic = %as_type (constants.%as_type.6b96b1.1)]
// CHECK:STDOUT:     %u.param: @F.%as_type (%as_type.6b96b1.1) = value_param call_param0
// CHECK:STDOUT:     %.loc5_11.1: type = splice_block %.loc5_11.2 [symbolic = %as_type (constants.%as_type.6b96b1.1)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_11.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: %Add.type = impl_witness_access %Self.as_wit.iface0.loc5_11.2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.c57618.1)]
// CHECK:STDOUT:       %U.ref.loc5_11: %Add.type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.c57618.1)]
// CHECK:STDOUT:       %U.as_type.loc5_11: type = facet_access_type %U.ref.loc5_11 [symbolic = %as_type (constants.%as_type.6b96b1.1)]
// CHECK:STDOUT:       %.loc5_11.2: type = converted %U.ref.loc5_11, %U.as_type.loc5_11 [symbolic = %as_type (constants.%as_type.6b96b1.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.%as_type (%as_type.6b96b1.1) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.%as_type (%as_type.6b96b1.1) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%as_type (%as_type.6b96b1.1) = return_slot %return.param
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
// CHECK:STDOUT:   assoc_const U:! %Add.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: %Add.type = impl_witness_access %Self.as_wit.iface0.loc5_11.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.c57618.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc5_11.1 [symbolic = %as_type (constants.%as_type.6b96b1.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param_patt: @F.%as_type (%as_type.6b96b1.1)) -> @F.%as_type (%as_type.6b96b1.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericAddResult(%T.loc8_21.1: %J.type) {
// CHECK:STDOUT:   %T.loc8_21.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_21.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_21.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc8_31.2: type = facet_access_type %T.loc8_21.2 [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc8_38.2: <witness> = facet_access_witness %T.loc8_21.2, element0 [symbolic = %T.as_wit.iface0.loc8_38.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc8_38.2: %Add.type = impl_witness_access %T.as_wit.iface0.loc8_38.2, element0 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:   %as_type.loc8_38.2: type = facet_access_type %impl.elem0.loc8_38.2 [symbolic = %as_type.loc8_38.2 (constants.%as_type.6b96b1.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_42: <witness> = require_complete_type @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) [symbolic = %require_complete.loc8_42 (constants.%require_complete.ec7)]
// CHECK:STDOUT:   %require_complete.loc8_29: <witness> = require_complete_type @GenericAddResult.%T.as_type.loc8_31.2 (%T.as_type) [symbolic = %require_complete.loc8_29 (constants.%require_complete.2a5)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type.loc8_31.2, (%T.as_wit.iface0.loc8_38.2) [symbolic = %J.facet (constants.%J.facet.085e97.2)]
// CHECK:STDOUT:   %.loc9_11.2: type = fn_type_with_self_type constants.%F.type, %J.facet [symbolic = %.loc9_11.2 (constants.%.4d2)]
// CHECK:STDOUT:   %impl.elem1.loc9_11.2: @GenericAddResult.%.loc9_11.2 (%.4d2) = impl_witness_access %T.as_wit.iface0.loc8_38.2, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem1.loc9_11.2, @F(%J.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.5d8)]
// CHECK:STDOUT:   %as_wit.iface0.loc9_17.2: <witness> = facet_access_witness %impl.elem0.loc8_38.2, element0 [symbolic = %as_wit.iface0.loc9_17.2 (constants.%as_wit.iface0)]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %as_type.loc8_38.2, (%as_wit.iface0.loc9_17.2) [symbolic = %Add.facet (constants.%Add.facet)]
// CHECK:STDOUT:   %.loc9_17.4: type = fn_type_with_self_type constants.%Op.type, %Add.facet [symbolic = %.loc9_17.4 (constants.%.5a3)]
// CHECK:STDOUT:   %impl.elem0.loc9_17.2: @GenericAddResult.%.loc9_17.4 (%.5a3) = impl_witness_access %as_wit.iface0.loc9_17.2, element0 [symbolic = %impl.elem0.loc9_17.2 (constants.%impl.elem0.ab3)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_17.2: <specific function> = specific_impl_function %impl.elem0.loc9_17.2, @Op(%Add.facet) [symbolic = %specific_impl_fn.loc9_17.2 (constants.%specific_impl_fn.e8e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_21.1: %J.type](%t.param_patt: @GenericAddResult.%T.as_type.loc8_31.2 (%T.as_type), %u.param_patt: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2)) -> @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericAddResult.%T.as_type.loc8_31.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref.loc9_11: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc9_11: type = facet_access_type constants.%T [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_11.1: type = converted constants.%T, %T.as_type.loc9_11 [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc9_11: <witness> = facet_access_witness constants.%T, element0 [symbolic = %T.as_wit.iface0.loc8_38.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem1.loc9_11.1: @GenericAddResult.%.loc9_11.2 (%.4d2) = impl_witness_access %T.as_wit.iface0.loc9_11, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref.loc9_14: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = name_ref u, %u
// CHECK:STDOUT:     %.loc9_15.1: %Add.type = converted constants.%as_type.6b96b1.2, constants.%impl.elem0.c57618.2 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:     %as_wit.iface0.1: <witness> = facet_access_witness constants.%J.facet.085e97.2, element0 [symbolic = %T.as_wit.iface0.loc8_38.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.1: %Add.type = impl_witness_access constants.%T.as_wit.iface0, element0 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:     %.loc9_15.2: %Add.type = converted constants.%as_type.6b96b1.2, constants.%impl.elem0.c57618.2 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:     %as_wit.iface0.2: <witness> = facet_access_witness constants.%J.facet.085e97.2, element0 [symbolic = %T.as_wit.iface0.loc8_38.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.2: %Add.type = impl_witness_access constants.%T.as_wit.iface0, element0 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem1.loc9_11.1, @F(constants.%J.facet.085e97.2) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.5d8)]
// CHECK:STDOUT:     %.loc9_15.3: init @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = call %specific_impl_fn.loc9_11.1(%u.ref.loc9_14)
// CHECK:STDOUT:     %T.ref.loc9: %J.type = name_ref T, %T.loc8_21.1 [symbolic = %T.loc8_21.2 (constants.%T)]
// CHECK:STDOUT:     %F.ref.loc9_20: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc9_20: type = facet_access_type %T.ref.loc9 [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_20: type = converted %T.ref.loc9, %T.as_type.loc9_20 [symbolic = %T.as_type.loc8_31.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc9_20: <witness> = facet_access_witness %T.ref.loc9, element0 [symbolic = %T.as_wit.iface0.loc8_38.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem1.loc9_20: @GenericAddResult.%.loc9_11.2 (%.4d2) = impl_witness_access %T.as_wit.iface0.loc9_20, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %u.ref.loc9_23: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = name_ref u, %u
// CHECK:STDOUT:     %.loc9_24.1: %Add.type = converted constants.%as_type.6b96b1.2, constants.%impl.elem0.c57618.2 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:     %as_wit.iface0.3: <witness> = facet_access_witness constants.%J.facet.085e97.2, element0 [symbolic = %T.as_wit.iface0.loc8_38.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.3: %Add.type = impl_witness_access constants.%T.as_wit.iface0, element0 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:     %.loc9_24.2: %Add.type = converted constants.%as_type.6b96b1.2, constants.%impl.elem0.c57618.2 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:     %as_wit.iface0.4: <witness> = facet_access_witness constants.%J.facet.085e97.2, element0 [symbolic = %T.as_wit.iface0.loc8_38.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.4: %Add.type = impl_witness_access constants.%T.as_wit.iface0, element0 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_20: <specific function> = specific_impl_function %impl.elem1.loc9_20, @F(constants.%J.facet.085e97.2) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn.5d8)]
// CHECK:STDOUT:     %.loc9_24.3: init @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = call %specific_impl_fn.loc9_20(%u.ref.loc9_23)
// CHECK:STDOUT:     %as_wit.iface0.loc9_17.1: <witness> = facet_access_witness constants.%impl.elem0.c57618.2, element0 [symbolic = %as_wit.iface0.loc9_17.2 (constants.%as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc9_17.1: @GenericAddResult.%.loc9_17.4 (%.5a3) = impl_witness_access %as_wit.iface0.loc9_17.1, element0 [symbolic = %impl.elem0.loc9_17.2 (constants.%impl.elem0.ab3)]
// CHECK:STDOUT:     %bound_method.loc9_17.1: <bound method> = bound_method %.loc9_15.3, %impl.elem0.loc9_17.1
// CHECK:STDOUT:     %.loc9_17.1: %Add.type = converted constants.%as_type.6b96b1.2, constants.%impl.elem0.c57618.2 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:     %.loc9_17.2: %Add.type = converted constants.%as_type.6b96b1.2, constants.%impl.elem0.c57618.2 [symbolic = %impl.elem0.loc8_38.2 (constants.%impl.elem0.c57618.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_17.1: <specific function> = specific_impl_function %impl.elem0.loc9_17.1, @Op(constants.%Add.facet) [symbolic = %specific_impl_fn.loc9_17.2 (constants.%specific_impl_fn.e8e)]
// CHECK:STDOUT:     %bound_method.loc9_17.2: <bound method> = bound_method %.loc9_15.3, %specific_impl_fn.loc9_17.1
// CHECK:STDOUT:     %.loc9_15.4: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = value_of_initializer %.loc9_15.3
// CHECK:STDOUT:     %.loc9_15.5: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = converted %.loc9_15.3, %.loc9_15.4
// CHECK:STDOUT:     %.loc9_24.4: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = value_of_initializer %.loc9_24.3
// CHECK:STDOUT:     %.loc9_24.5: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = converted %.loc9_24.3, %.loc9_24.4
// CHECK:STDOUT:     %.loc9_17.3: init @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = call %bound_method.loc9_17.2(%.loc9_15.5, %.loc9_24.5)
// CHECK:STDOUT:     %.loc9_25.1: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = value_of_initializer %.loc9_17.3
// CHECK:STDOUT:     %.loc9_25.2: @GenericAddResult.%as_type.loc8_38.2 (%as_type.6b96b1.2) = converted %.loc9_17.3, %.loc9_25.1
// CHECK:STDOUT:     return %.loc9_25.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.085e97.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.c57618.1
// CHECK:STDOUT:   %as_type => constants.%as_type.6b96b1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.085e97.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAddResult(constants.%T) {
// CHECK:STDOUT:   %T.loc8_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_21.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc8_31.2 => constants.%T.as_type
// CHECK:STDOUT:   %T.as_wit.iface0.loc8_38.2 => constants.%T.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc8_38.2 => constants.%impl.elem0.c57618.2
// CHECK:STDOUT:   %as_type.loc8_38.2 => constants.%as_type.6b96b1.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%J.facet.085e97.2) {
// CHECK:STDOUT:   %Self => constants.%J.facet.085e97.2
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%T.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.c57618.2
// CHECK:STDOUT:   %as_type => constants.%as_type.6b96b1.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(@GenericAddResult.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface_qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.085e97.1: %J.type = facet_value %Self.as_type, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d9c3a6.1: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%G.decl [concrete]
// CHECK:STDOUT:   %T: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0: <witness> = facet_access_witness %T, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.085e97.2: %J.type = facet_value %T.as_type, (%T.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d9c3a6.2: type = impl_witness_access %T.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %GenericCallInterfaceQualified.type: type = fn_type @GenericCallInterfaceQualified [concrete]
// CHECK:STDOUT:   %GenericCallInterfaceQualified: %GenericCallInterfaceQualified.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d9a: <witness> = require_complete_type %impl.elem0.d9c3a6.2 [symbolic]
// CHECK:STDOUT:   %require_complete.2a5: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.9ac: type = fn_type_with_self_type %G.type, %J.facet.085e97.2 [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.9ac = impl_witness_access %T.as_wit.iface0, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem1, @G(%J.facet.085e97.2) [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc8_34.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_34.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: @GenericCallInterfaceQualified.%T.as_type.loc8_44.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @GenericCallInterfaceQualified.%T.as_type.loc8_44.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %u.patt: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_58: %J.type = name_ref T, %T.loc8_34.1 [symbolic = %T.loc8_34.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc8_59: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc8_59: type = facet_access_type %T.ref.loc8_58 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8_59: type = converted %T.ref.loc8_58, %T.as_type.loc8_59 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc8_59: <witness> = facet_access_witness %T.ref.loc8_58, element0 [symbolic = %T.as_wit.iface0.loc8_51.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc8_59: type = impl_witness_access %T.as_wit.iface0.loc8_59, element0 [symbolic = %impl.elem0.loc8_51.2 (constants.%impl.elem0.d9c3a6.2)]
// CHECK:STDOUT:     %J.ref.loc8: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc8_34.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_34.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericCallInterfaceQualified.%T.as_type.loc8_44.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_44.1: type = splice_block %.loc8_44.2 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc8_44: %J.type = name_ref T, %T.loc8_34.1 [symbolic = %T.loc8_34.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8_44.1: type = facet_access_type %T.ref.loc8_44 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_44.2: type = converted %T.ref.loc8_44, %T.as_type.loc8_44.1 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericCallInterfaceQualified.%T.as_type.loc8_44.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = value_param call_param1
// CHECK:STDOUT:     %.loc8_51.1: type = splice_block %impl.elem0.loc8_51.1 [symbolic = %impl.elem0.loc8_51.2 (constants.%impl.elem0.d9c3a6.2)] {
// CHECK:STDOUT:       %T.ref.loc8_50: %J.type = name_ref T, %T.loc8_34.1 [symbolic = %T.loc8_34.2 (constants.%T)]
// CHECK:STDOUT:       %U.ref.loc8_51: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %T.as_type.loc8_51: type = facet_access_type %T.ref.loc8_50 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_51.2: type = converted %T.ref.loc8_50, %T.as_type.loc8_51 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %T.as_wit.iface0.loc8_51.1: <witness> = facet_access_witness %T.ref.loc8_50, element0 [symbolic = %T.as_wit.iface0.loc8_51.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc8_51.1: type = impl_witness_access %T.as_wit.iface0.loc8_51.1, element0 [symbolic = %impl.elem0.loc8_51.2 (constants.%impl.elem0.d9c3a6.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = out_param call_param2
// CHECK:STDOUT:     %return: ref @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %self.patt: @G.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %v.patt: @G.%impl.elem0.loc5_23.1 (%impl.elem0.d9c3a6.1) = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: @G.%impl.elem0.loc5_23.1 (%impl.elem0.d9c3a6.1) = value_param_pattern %v.patt, call_param1
// CHECK:STDOUT:     %return.patt: @G.%impl.elem0.loc5_23.1 (%impl.elem0.d9c3a6.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @G.%impl.elem0.loc5_23.1 (%impl.elem0.d9c3a6.1) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_29: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access %Self.as_wit.iface0.loc5_29, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:     %U.ref.loc5_29: type = name_ref U, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:     %self.param: @G.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: @G.%impl.elem0.loc5_23.1 (%impl.elem0.d9c3a6.1) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %U.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c3a6.1)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_23.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access %Self.as_wit.iface0.loc5_23.2, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:       %U.ref.loc5_23: type = name_ref U, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @G.%impl.elem0.loc5_23.1 (%impl.elem0.d9c3a6.1) = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref @G.%impl.elem0.loc5_23.1 (%impl.elem0.d9c3a6.1) = out_param call_param2
// CHECK:STDOUT:     %return: ref @G.%impl.elem0.loc5_23.1 (%impl.elem0.d9c3a6.1) = return_slot %return.param
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
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %Self.as_wit.iface0.loc5_23.1, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.d9c3a6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.%Self.as_type.loc5_14.1 (%Self.as_type)](%v.param_patt: @G.%impl.elem0.loc5_23.1 (%impl.elem0.d9c3a6.1)) -> @G.%impl.elem0.loc5_23.1 (%impl.elem0.d9c3a6.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallInterfaceQualified(%T.loc8_34.1: %J.type) {
// CHECK:STDOUT:   %T.loc8_34.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_34.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_34.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_34.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc8_44.2: type = facet_access_type %T.loc8_34.2 [symbolic = %T.as_type.loc8_44.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc8_51.2: <witness> = facet_access_witness %T.loc8_34.2, element0 [symbolic = %T.as_wit.iface0.loc8_51.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc8_51.2: type = impl_witness_access %T.as_wit.iface0.loc8_51.2, element0 [symbolic = %impl.elem0.loc8_51.2 (constants.%impl.elem0.d9c3a6.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_55: <witness> = require_complete_type @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) [symbolic = %require_complete.loc8_55 (constants.%require_complete.d9a)]
// CHECK:STDOUT:   %require_complete.loc8_42: <witness> = require_complete_type @GenericCallInterfaceQualified.%T.as_type.loc8_44.2 (%T.as_type) [symbolic = %require_complete.loc8_42 (constants.%require_complete.2a5)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type.loc8_44.2, (%T.as_wit.iface0.loc8_51.2) [symbolic = %J.facet (constants.%J.facet.085e97.2)]
// CHECK:STDOUT:   %.loc9_11: type = fn_type_with_self_type constants.%G.type, %J.facet [symbolic = %.loc9_11 (constants.%.9ac)]
// CHECK:STDOUT:   %impl.elem1.loc9_11.2: @GenericCallInterfaceQualified.%.loc9_11 (%.9ac) = impl_witness_access %T.as_wit.iface0.loc8_51.2, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem1.loc9_11.2, @G(%J.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_34.1: %J.type](%t.param_patt: @GenericCallInterfaceQualified.%T.as_type.loc8_44.2 (%T.as_type), %u.param_patt: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2)) -> @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallInterfaceQualified.%T.as_type.loc8_44.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %J.ref.loc9: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %G.ref: %J.assoc_type = name_ref G, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_wit.iface0.loc9: <witness> = facet_access_witness constants.%T, element0 [symbolic = %T.as_wit.iface0.loc8_51.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem1.loc9_11.1: @GenericCallInterfaceQualified.%.loc9_11 (%.9ac) = impl_witness_access %T.as_wit.iface0.loc9, element1 [symbolic = %impl.elem1.loc9_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc9_11: <bound method> = bound_method %t.ref, %impl.elem1.loc9_11.1
// CHECK:STDOUT:     %u.ref: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = name_ref u, %u
// CHECK:STDOUT:     %as_wit.iface0.1: <witness> = facet_access_witness constants.%J.facet.085e97.2, element0 [symbolic = %T.as_wit.iface0.loc8_51.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.1: type = impl_witness_access constants.%T.as_wit.iface0, element0 [symbolic = %impl.elem0.loc8_51.2 (constants.%impl.elem0.d9c3a6.2)]
// CHECK:STDOUT:     %as_wit.iface0.2: <witness> = facet_access_witness constants.%J.facet.085e97.2, element0 [symbolic = %T.as_wit.iface0.loc8_51.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.2: type = impl_witness_access constants.%T.as_wit.iface0, element0 [symbolic = %impl.elem0.loc8_51.2 (constants.%impl.elem0.d9c3a6.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem1.loc9_11.1, @G(constants.%J.facet.085e97.2) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc9_19: <bound method> = bound_method %t.ref, %specific_impl_fn.loc9_11.1
// CHECK:STDOUT:     %.loc9_19: init @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = call %bound_method.loc9_19(%t.ref, %u.ref)
// CHECK:STDOUT:     %.loc9_20.1: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = value_of_initializer %.loc9_19
// CHECK:STDOUT:     %.loc9_20.2: @GenericCallInterfaceQualified.%impl.elem0.loc8_51.2 (%impl.elem0.d9c3a6.2) = converted %.loc9_19, %.loc9_20.1
// CHECK:STDOUT:     return %.loc9_20.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.085e97.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.d9c3a6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.085e97.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallInterfaceQualified(constants.%T) {
// CHECK:STDOUT:   %T.loc8_34.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_34.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc8_44.2 => constants.%T.as_type
// CHECK:STDOUT:   %T.as_wit.iface0.loc8_51.2 => constants.%T.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc8_51.2 => constants.%impl.elem0.d9c3a6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%J.facet.085e97.2) {
// CHECK:STDOUT:   %Self => constants.%J.facet.085e97.2
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%T.as_type
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1 => constants.%T.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.d9c3a6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(@GenericCallInterfaceQualified.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.085: %J.type = facet_value %Self.as_type.3df, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d9c: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %J.facet.28c: %J.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.53a: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.53a = %i32> [concrete]
// CHECK:STDOUT:   %T: %J_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %J_where.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %GenericCallFI32.type: type = fn_type @GenericCallFI32 [concrete]
// CHECK:STDOUT:   %GenericCallFI32: %GenericCallFI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.3e7: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0: <witness> = facet_access_witness %T, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.869: %J.type = facet_value %T.as_type, (%T.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %.861: type = fn_type_with_self_type %F.type, %J.facet.869 [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.861 = impl_witness_access %T.as_wit.iface0, element1 [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %impl.elem0.37e: type = impl_witness_access %T.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem1, @F(%J.facet.869) [symbolic]
// CHECK:STDOUT:   %require_complete.8bd: <witness> = require_complete_type %impl.elem0.37e [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.820: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.37e)> [symbolic]
// CHECK:STDOUT:   %require_complete.1ce: <witness> = require_complete_type %ImplicitAs.type.820 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc8_20.1: %J_where.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_20.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: @GenericCallFI32.%T.as_type.loc8_45.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @GenericCallFI32.%T.as_type.loc8_45.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc8_51: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_51: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_26.1: type = splice_block %.loc8_26.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc8_32: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:       %impl.elem0.loc8: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.53a]
// CHECK:STDOUT:       %int_32.loc8_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc8_26.2: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc8, %i32.loc8_37
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_20.1: %J_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_20.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @GenericCallFI32.%T.as_type.loc8_45.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_45.1: type = splice_block %.loc8_45.2 [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %J_where.type = name_ref T, %T.loc8_20.1 [symbolic = %T.loc8_20.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8_45.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_45.2: type = converted %T.ref, %T.as_type.loc8_45.1 [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @GenericCallFI32.%T.as_type.loc8_45.2 (%T.as_type) = bind_name t, %t.param
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
// CHECK:STDOUT:     %u.patt: @F.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_17: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access %Self.as_wit.iface0.loc5_17, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     %u.param: @F.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_11.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access %Self.as_wit.iface0.loc5_11.2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = return_slot %return.param
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
// CHECK:STDOUT: generic fn @F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %Self.as_wit.iface0.loc5_11.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param_patt: @F.%impl.elem0.loc5_11.1 (%impl.elem0.d9c)) -> @F.%impl.elem0.loc5_11.1 (%impl.elem0.d9c);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericCallFI32(%T.loc8_20.1: %J_where.type) {
// CHECK:STDOUT:   %T.loc8_20.2: %J_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_20.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_20.2: %J_where.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_20.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc8_45.2: type = facet_access_type %T.loc8_20.2 [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type @GenericCallFI32.%T.as_type.loc8_45.2 (%T.as_type) [symbolic = %require_complete.loc8 (constants.%require_complete.3e7)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc26_11.2: <witness> = facet_access_witness %T.loc8_20.2, element0 [symbolic = %T.as_wit.iface0.loc26_11.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type.loc8_45.2, (%T.as_wit.iface0.loc26_11.2) [symbolic = %J.facet (constants.%J.facet.869)]
// CHECK:STDOUT:   %.loc26_11.2: type = fn_type_with_self_type constants.%F.type, %J.facet [symbolic = %.loc26_11.2 (constants.%.861)]
// CHECK:STDOUT:   %impl.elem1.loc26_11.2: @GenericCallFI32.%.loc26_11.2 (%.861) = impl_witness_access %T.as_wit.iface0.loc26_11.2, element1 [symbolic = %impl.elem1.loc26_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %impl.elem0.3: type = impl_witness_access %T.as_wit.iface0.loc26_11.2, element0 [symbolic = %impl.elem0.3 (constants.%impl.elem0.37e)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_11.2: <specific function> = specific_impl_function %impl.elem1.loc26_11.2, @F(%J.facet) [symbolic = %specific_impl_fn.loc26_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %require_complete.loc26_15: <witness> = require_complete_type @GenericCallFI32.%impl.elem0.3 (%impl.elem0.37e) [symbolic = %require_complete.loc26_15 (constants.%require_complete.8bd)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.3)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.820)]
// CHECK:STDOUT:   %require_complete.loc26_14: <witness> = require_complete_type @GenericCallFI32.%ImplicitAs.type (%ImplicitAs.type.820) [symbolic = %require_complete.loc26_14 (constants.%require_complete.1ce)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_20.1: %J_where.type](%t.param_patt: @GenericCallFI32.%T.as_type.loc8_45.2 (%T.as_type)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @GenericCallFI32.%T.as_type.loc8_45.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %F.ref: %J.assoc_type = name_ref F, @J.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc26: type = facet_access_type constants.%T [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_11.1: type = converted constants.%T, %T.as_type.loc26 [symbolic = %T.as_type.loc8_45.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc26_11.1: <witness> = facet_access_witness constants.%T, element0 [symbolic = %T.as_wit.iface0.loc26_11.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem1.loc26_11.1: @GenericCallFI32.%.loc26_11.2 (%.861) = impl_witness_access %T.as_wit.iface0.loc26_11.1, element1 [symbolic = %impl.elem1.loc26_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %as_wit.iface0.1: <witness> = facet_access_witness constants.%J.facet.869, element0 [symbolic = %T.as_wit.iface0.loc26_11.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.1: type = impl_witness_access constants.%T.as_wit.iface0, element0 [symbolic = %impl.elem0.3 (constants.%impl.elem0.37e)]
// CHECK:STDOUT:     %as_wit.iface0.2: <witness> = facet_access_witness constants.%J.facet.869, element0 [symbolic = %T.as_wit.iface0.loc26_11.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.2: type = impl_witness_access constants.%T.as_wit.iface0, element0 [symbolic = %impl.elem0.3 (constants.%impl.elem0.37e)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_11.1: <specific function> = specific_impl_function %impl.elem1.loc26_11.1, @F(constants.%J.facet.869) [symbolic = %specific_impl_fn.loc26_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc26_14: @GenericCallFI32.%impl.elem0.3 (%impl.elem0.37e) = converted %int_2, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc26_15: init @GenericCallFI32.%impl.elem0.3 (%impl.elem0.37e) = call %specific_impl_fn.loc26_11.1(<error>) [concrete = <error>]
// CHECK:STDOUT:     %.loc26_16: %i32 = converted %.loc26_15, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.085) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.d9c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.28c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericCallFI32(constants.%T) {
// CHECK:STDOUT:   %T.loc8_20.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc8_45.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%J.facet.869) {
// CHECK:STDOUT:   %Self => constants.%J.facet.869
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%T.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.37e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(@GenericCallFI32.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_associated_constant_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.411: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.085: %J.type = facet_value %Self.as_type.3df, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d9c: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %J.facet.28c: %J.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.53a: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.53a = %i32> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%i32, <error>) [concrete]
// CHECK:STDOUT:   %F.type.e5e: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c81: %F.type.e5e = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.441: %J.type = facet_value %E, (%impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Add = %Core.Add
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
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.411]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %u.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = value_param_pattern %u.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_17: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_17: type = impl_witness_access %Self.as_wit.iface0.loc5_17, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     %U.ref.loc5_17: type = name_ref U, %impl.elem0.loc5_17 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %U.ref.loc5_11 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_11.2: <witness> = facet_access_witness @J.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_11.2: type = impl_witness_access %Self.as_wit.iface0.loc5_11.2, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:       %U.ref.loc5_11: type = name_ref U, %impl.elem0.loc5_11.2 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c) = return_slot %return.param
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
// CHECK:STDOUT: impl @impl: %Self.ref as %.loc9_20 {
// CHECK:STDOUT:   %F.decl: %F.type.e5e = fn_decl @F.2 [concrete = constants.%F.c81] {
// CHECK:STDOUT:     %u.patt: <error> = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: <error> = value_param_pattern %u.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc24_19: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.loc24_19: type = converted %U.ref.loc24_19, <error> [concrete = <error>]
// CHECK:STDOUT:     %u.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %U.ref.loc24_13: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:       %.loc24_13: type = converted %U.ref.loc24_13, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: <error> = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param1
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .U = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @E.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.411]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc9_26: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.53a]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc9_20: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%i32, <error>) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .U = <poisoned>
// CHECK:STDOUT:   extend @impl.%.loc9_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_11.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_11.1: type = impl_witness_access %Self.as_wit.iface0.loc5_11.1, element0 [symbolic = %impl.elem0.loc5_11.1 (constants.%impl.elem0.d9c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param_patt: @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c)) -> @F.1.%impl.elem0.loc5_11.1 (%impl.elem0.d9c);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%u.param_patt: <error>) -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: <error> = name_ref u, %u [concrete = <error>]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0: <error> = impl_witness_access <error>, element0 [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.085) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%impl.elem0.d9c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.28c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.441) {
// CHECK:STDOUT:   %Self => constants.%J.facet.441
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_11.1 => constants.%impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_11.1 => constants.%i32
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
// CHECK:STDOUT:   %F.type.c41: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.932: %F.type.c41 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J2.assoc_type = assoc_entity element1, @J2.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %J2.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %J2.facet.afd: %J2.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %J2_where.type.a84: type = facet_type <@J2 where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %impl_witness.f71: <witness> = impl_witness (%empty_struct_type, <error>) [concrete]
// CHECK:STDOUT:   %F.type.aee: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.243: %F.type.aee = struct_value () [concrete]
// CHECK:STDOUT:   %J2.facet.fdd: %J2.type = facet_value %empty_tuple.type, (%impl_witness.f71) [concrete]
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J2_where.type.4f1: type = facet_type <@J2 where %impl.elem0 = %C2> [concrete]
// CHECK:STDOUT:   %impl_witness.d84: <witness> = impl_witness (%C2, <error>) [concrete]
// CHECK:STDOUT:   %F.type.ea3: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.81f: %F.type.ea3 = struct_value () [concrete]
// CHECK:STDOUT:   %J2.facet.f36: %J2.type = facet_value %C2, (%impl_witness.d84) [concrete]
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
// CHECK:STDOUT:   impl_decl @impl.8b6 [concrete] {} {
// CHECK:STDOUT:     %.loc22_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_7.2: type = converted %.loc22_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J2.ref: type = name_ref J2, file.%J2.decl [concrete = constants.%J2.type]
// CHECK:STDOUT:     %.Self: %J2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U2.ref: %J2.assoc_type = name_ref U2, @U2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc22_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc22_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc22_28.2: type = converted %.loc22_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc22_15: type = where_expr %.Self [concrete = constants.%J2_where.type.a84] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc22_28.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc22: <witness> = impl_witness (constants.%empty_struct_type, <error>) [concrete = constants.%impl_witness.f71]
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [concrete = constants.%C2] {} {}
// CHECK:STDOUT:   impl_decl @impl.3df [concrete] {} {
// CHECK:STDOUT:     %C2.ref.loc31_6: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %J2.ref: type = name_ref J2, file.%J2.decl [concrete = constants.%J2.type]
// CHECK:STDOUT:     %.Self: %J2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U2.ref: %J2.assoc_type = name_ref U2, @U2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc31_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C2.ref.loc31_27: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %.loc31_15: type = where_expr %.Self [concrete = constants.%J2_where.type.4f1] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C2.ref.loc31_27
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc31: <witness> = impl_witness (constants.%C2, <error>) [concrete = constants.%impl_witness.d84]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J2 {
// CHECK:STDOUT:   %Self: %J2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U2: type = assoc_const_decl @U2 [concrete] {
// CHECK:STDOUT:     %assoc0: %J2.assoc_type = assoc_entity element0, @J2.%U2 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c41 = fn_decl @F.1 [concrete = constants.%F.932] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %z.patt: <error> = binding_pattern z
// CHECK:STDOUT:     %z.param_patt: <error> = value_param_pattern %z.patt, call_param1 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param2
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
// CHECK:STDOUT:     %z: <error> = bind_name z, %z.param
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
// CHECK:STDOUT: impl @impl.8b6: %.loc22_7.2 as %.loc22_15 {
// CHECK:STDOUT:   %F.decl: %F.type.aee = fn_decl @F.2 [concrete = constants.%F.243] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %z.patt: %empty_struct_type = binding_pattern z
// CHECK:STDOUT:     %z.param_patt: %empty_struct_type = value_param_pattern %z.patt, call_param1
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc23_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc23_31.2: type = converted %.loc23_31.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.8b6.%.loc22_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %z.param: %empty_struct_type = value_param call_param1
// CHECK:STDOUT:     %.loc23_24.1: type = splice_block %.loc23_24.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc23_24.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc23_24.3: type = converted %.loc23_24.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %z: %empty_struct_type = bind_name z, %z.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param2
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3df: %C2.ref.loc31_6 as %.loc31_15 {
// CHECK:STDOUT:   %F.decl: %F.type.ea3 = fn_decl @F.3 [concrete = constants.%F.81f] {
// CHECK:STDOUT:     %self.patt: %C2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C2 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %z.patt: %C2 = binding_pattern z
// CHECK:STDOUT:     %z.param_patt: %C2 = value_param_pattern %z.patt, call_param1
// CHECK:STDOUT:     %return.patt: %C2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C2 = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C2.ref.loc32_30: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %self.param: %C2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.3df.%C2.ref.loc31_6 [concrete = constants.%C2]
// CHECK:STDOUT:     %self: %C2 = bind_name self, %self.param
// CHECK:STDOUT:     %z.param: %C2 = value_param call_param1
// CHECK:STDOUT:     %C2.ref.loc32_23: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %z: %C2 = bind_name z, %z.param
// CHECK:STDOUT:     %return.param: ref %C2 = out_param call_param2
// CHECK:STDOUT:     %return: ref %C2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C2 = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %.loc28_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc28_11: type = converted %.loc28_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc28_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J2.%Self: %J2.type) {
// CHECK:STDOUT:   %Self: %J2.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc19_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type)](%z.param_patt: <error>) -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type](%z.param_patt: %empty_struct_type) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %z.ref: %empty_struct_type = name_ref z, %z
// CHECK:STDOUT:   return %z.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.param_patt: %C2](%z.param_patt: %C2) -> %C2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C2 = name_ref self, %self
// CHECK:STDOUT:   return %self.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U2(constants.%J2.facet.afd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J2.facet.fdd) {
// CHECK:STDOUT:   %Self => constants.%J2.facet.fdd
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J2.facet.f36) {
// CHECK:STDOUT:   %Self => constants.%J2.facet.f36
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%C2
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
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self, element0 [symbolic]
// CHECK:STDOUT:   %K.facet.3e1: %K.type = facet_value %Self.as_type, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.c2f: type = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type.cf5: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.06b: %F.type.cf5 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %K.assoc_type = assoc_entity element1, @K.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %K.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %K.facet.c18: %K.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e09: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %K_where.type: type = facet_type <@K where %impl.elem0.e09 = %struct_type.a> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%struct_type.a, <error>) [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %F.type.894: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.62c: %F.type.894 = struct_value () [concrete]
// CHECK:STDOUT:   %K.facet.28f: %K.type = facet_value %empty_tuple.type, (%impl_witness) [concrete]
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
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.e09]
// CHECK:STDOUT:     %.loc8_31.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_31.2: type = converted %.loc8_31.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%K_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %struct_type.a
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%struct_type.a, <error>) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %V: type = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %K.assoc_type = assoc_entity element0, @K.%V [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.cf5 = fn_decl @F.1 [concrete = constants.%F.06b] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %v.patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.c2f) = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.c2f) = value_param_pattern %v.patt, call_param1
// CHECK:STDOUT:     %return.patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.c2f) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.c2f) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_29: <witness> = facet_access_witness @K.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access %Self.as_wit.iface0.loc5_29, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.c2f)]
// CHECK:STDOUT:     %V.ref.loc5_29: type = name_ref V, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.c2f)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %K.type = name_ref Self, @K.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.c2f) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %V.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.c2f)] {
// CHECK:STDOUT:       %Self.as_wit.iface0.loc5_23.2: <witness> = facet_access_witness @K.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access %Self.as_wit.iface0.loc5_23.2, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.c2f)]
// CHECK:STDOUT:       %V.ref.loc5_23: type = name_ref V, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.c2f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.c2f) = bind_name v, %v.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.c2f) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.c2f) = return_slot %return.param
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
// CHECK:STDOUT:   %F.decl: %F.type.894 = fn_decl @F.2 [concrete = constants.%F.62c] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %v.patt: %struct_type.x = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: %struct_type.x = value_param_pattern %v.patt, call_param1
// CHECK:STDOUT:     %return.patt: %struct_type.x = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %struct_type.x = out_param_pattern %return.patt, call_param2
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@K.%Self: %K.type) {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_23.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %Self.as_wit.iface0.loc5_23.1, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.c2f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type)](%v.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.c2f)) -> @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.c2f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type](%v.param_patt: %struct_type.x) -> %struct_type.x {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: %struct_type.x = name_ref v, %v
// CHECK:STDOUT:   return %v.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%K.facet.3e1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.c2f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%K.facet.c18) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%K.facet.28f) {
// CHECK:STDOUT:   %Self => constants.%K.facet.28f
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_23.1 => constants.%impl_witness
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%struct_type.a
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
// CHECK:STDOUT:   %G.type.020: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.91c: %G.type.020 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%G.decl [concrete]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %M.facet.407: %M.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.b.347 = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %M_where.type: type = facet_type <@M where %impl.elem0 = %struct> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%struct, @impl.%G.decl) [concrete]
// CHECK:STDOUT:   %G.type.bf5: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.44e: %G.type.bf5 = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.993: %M.type = facet_value %empty_tuple.type, (%impl_witness) [concrete]
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
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.347 = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%struct, @impl.%G.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Z: %struct_type.b.347 = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.020 = fn_decl @G.1 [concrete = constants.%G.91c] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param0
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
// CHECK:STDOUT:   %G.decl: %G.type.bf5 = fn_decl @G.2 [concrete = constants.%G.44e] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param0
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
// CHECK:STDOUT:   witness = file.%impl_witness
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
// CHECK:STDOUT:   %M.facet: %M.type = facet_value constants.%empty_tuple.type, (constants.%impl_witness) [concrete = constants.%M.facet.993]
// CHECK:STDOUT:   %.loc10_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.993]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_18 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc10_23: type = converted %.loc10_18, %as_type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %as_wit.iface0: <witness> = facet_access_witness %.loc10_18, element0 [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access %as_wit.iface0, element0 [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_25: %empty_struct_type = struct_access %impl.elem0, element0 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc10_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.407) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%M.facet.993) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.993) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_as_uses_correct_rewrite_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
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
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %M.facet.407: %M.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.86f = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.0f3: %struct_type.b.86f = struct_value (%empty_struct_type) [concrete]
// CHECK:STDOUT:   %M_where.type.132: type = facet_type <@M where %impl.elem0 = %struct.0f3> [concrete]
// CHECK:STDOUT:   %impl_witness.ae0: <witness> = impl_witness (%struct.0f3, @impl.9c5.%G.decl) [concrete]
// CHECK:STDOUT:   %G.type.160: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.fc4: %G.type.160 = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.508: %M.type = facet_value %C.7a7, (%impl_witness.ae0) [concrete]
// CHECK:STDOUT:   %C.3a0: type = class_type @C, @C(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %struct_type.b.161: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct.c94: %struct_type.b.86f = struct_value (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %M_where.type.363: type = facet_type <@M where %impl.elem0 = %struct.c94> [concrete]
// CHECK:STDOUT:   %impl_witness.6b9: <witness> = impl_witness (%struct.c94, @impl.f46.%G.decl) [concrete]
// CHECK:STDOUT:   %G.type.18e: type = fn_type @G.3 [concrete]
// CHECK:STDOUT:   %G.066: %G.type.18e = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.a97: %M.type = facet_value %C.3a0, (%impl_witness.6b9) [concrete]
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
// CHECK:STDOUT:     %T.patt.loc3_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.9c5 [concrete] {} {
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
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.86f = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc10_35: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_36.1: %struct_type.b.347 = struct_literal (%.loc10_35)
// CHECK:STDOUT:     %.loc10_36.2: type = converted %.loc10_35, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct: %struct_type.b.86f = struct_value (%.loc10_36.2) [concrete = constants.%struct.0f3]
// CHECK:STDOUT:     %.loc10_36.3: %struct_type.b.86f = converted %.loc10_36.1, %struct [concrete = constants.%struct.0f3]
// CHECK:STDOUT:     %.loc10_17: type = where_expr %.Self [concrete = constants.%M_where.type.132] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc10_36.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc10: <witness> = impl_witness (constants.%struct.0f3, @impl.9c5.%G.decl) [concrete = constants.%impl_witness.ae0]
// CHECK:STDOUT:   impl_decl @impl.f46 [concrete] {} {
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
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.86f = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc16_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_36.1: %struct_type.b.161 = struct_literal (%.loc16_35)
// CHECK:STDOUT:     %.loc16_36.2: type = converted %.loc16_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct: %struct_type.b.86f = struct_value (%.loc16_36.2) [concrete = constants.%struct.c94]
// CHECK:STDOUT:     %.loc16_36.3: %struct_type.b.86f = converted %.loc16_36.1, %struct [concrete = constants.%struct.c94]
// CHECK:STDOUT:     %.loc16_17: type = where_expr %.Self [concrete = constants.%M_where.type.363] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc16_36.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness (constants.%struct.c94, @impl.f46.%G.decl) [concrete = constants.%impl_witness.6b9]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Z: %struct_type.b.86f = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.020 = fn_decl @G.1 [concrete = constants.%G.91c] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
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
// CHECK:STDOUT: impl @impl.9c5: %C as %.loc10_17 {
// CHECK:STDOUT:   %G.decl: %G.type.160 = fn_decl @G.2 [concrete = constants.%G.fc4] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = file.%impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.f46: %C as %.loc16_17 {
// CHECK:STDOUT:   %G.decl: %G.type.18e = fn_decl @G.3 [concrete = constants.%G.066] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = file.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc3_9.1: type) {
// CHECK:STDOUT:   %T.loc3_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc3_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.9c5.%C [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value constants.%C.7a7, (constants.%impl_witness.ae0) [concrete = constants.%M.facet.508]
// CHECK:STDOUT:   %.loc12_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.508]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc12_18 [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %.loc12_23: type = converted %.loc12_18, %as_type [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %as_wit.iface0: <witness> = facet_access_witness %.loc12_18, element0 [concrete = constants.%impl_witness.ae0]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.86f = impl_witness_access %as_wit.iface0, element0 [concrete = constants.%struct.0f3]
// CHECK:STDOUT:   %.loc12_25: type = struct_access %impl.elem0, element0 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   return %.loc12_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.3() -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.f46.%C [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value constants.%C.3a0, (constants.%impl_witness.6b9) [concrete = constants.%M.facet.a97]
// CHECK:STDOUT:   %.loc18_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.a97]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc18_18 [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %.loc18_23: type = converted %.loc18_18, %as_type [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %as_wit.iface0: <witness> = facet_access_witness %.loc18_18, element0 [concrete = constants.%impl_witness.6b9]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.86f = impl_witness_access %as_wit.iface0, element0 [concrete = constants.%struct.c94]
// CHECK:STDOUT:   %.loc18_25: type = struct_access %impl.elem0, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   return %.loc18_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc3_9.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc3_9.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.407) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%M.facet.508) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.508) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %T.patt.loc3_9.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%M.facet.a97) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.a97) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_int_in_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.995: %I.assoc_type = assoc_entity element0, @I.%N [concrete]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self.826, element0 [symbolic]
// CHECK:STDOUT:   %I.facet.335: %I.type = facet_value %Self.as_type.b70, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.ace: %i32 = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %impl_witness.023: <witness> = impl_witness (imports.%Core.import_ref.85c), @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ffe: %ImplicitAs.type.2fd = facet_value %i32, (%impl_witness.023) [concrete]
// CHECK:STDOUT:   %.4d8: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.ffe [concrete]
// CHECK:STDOUT:   %Convert.bound.69c: <bound method> = bound_method %impl.elem0.ace, %Convert.960 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.8a8: <specific function> = specific_function %Convert.960, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.37e: <bound method> = bound_method %impl.elem0.ace, %Convert.specific_fn.8a8 [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %bound_method.37e(%impl.elem0.ace) [symbolic]
// CHECK:STDOUT:   %array_type.a05: type = array_type %int.convert_checked, bool [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%F.decl [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet.d87: %I.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.0ad: %i32 = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2b8: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.2b8 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b6f: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn.b6f [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.0ad = %int_2.ef8> [concrete]
// CHECK:STDOUT:   %impl_witness.d8a: <witness> = impl_witness (%int_2.ef8, @impl.dd2.%F.decl) [concrete]
// CHECK:STDOUT:   %array_type.c9b: type = array_type %int_2.ecc, bool [concrete]
// CHECK:STDOUT:   %F.type.925: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.2a6: %F.type.925 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.ed1: %I.type = facet_value %empty_tuple.type, (%impl_witness.d8a) [concrete]
// CHECK:STDOUT:   %Convert.bound.fbf: <bound method> = bound_method %int_2.ef8, %Convert.960 [concrete]
// CHECK:STDOUT:   %bound_method.992: <bound method> = bound_method %int_2.ef8, %Convert.specific_fn.8a8 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (bool, bool) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.dd2 [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %N.ref: %I.assoc_type = name_ref N, @N.%assoc0 [concrete = constants.%assoc0.995]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc8_20: %i32 = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.0ad]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc8_25: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc8_25.1: <bound method> = bound_method %int_2, %impl.elem0.loc8_25 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc8_25, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:     %bound_method.loc8_25.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc8_25.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_25.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_25.2: %i32 = converted %int_2, %.loc8_25.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8_20, %.loc8_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%int_2.ef8, @impl.dd2.%F.decl) [concrete = constants.%impl_witness.d8a]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl @N [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%N [concrete = constants.%assoc0.995]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.1.%array_type.loc5_38.1 (%array_type.a05) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%array_type.loc5_38.1 (%array_type.a05) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %Self.as_wit.iface0.loc5_37.2: <witness> = facet_access_witness @I.%Self, element0 [symbolic = %Self.as_wit.iface0.loc5_37.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc5_37.2: %i32 = impl_witness_access %Self.as_wit.iface0.loc5_37.2, element0 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.ace)]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %impl.elem0.loc5_37.2 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.ace)]
// CHECK:STDOUT:     %.loc5_31.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc5_31.2: type = converted %bool.make_type, %.loc5_31.1 [concrete = bool]
// CHECK:STDOUT:     %impl.elem0.loc5_37.3: %.4d8 = impl_witness_access constants.%impl_witness.023, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:     %bound_method.loc5_37.2: <bound method> = bound_method %N.ref, %impl.elem0.loc5_37.3 [symbolic = %Convert.bound (constants.%Convert.bound.69c)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc5_37.3, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn.8a8]
// CHECK:STDOUT:     %bound_method.loc5_37.3: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc5_37.1 (constants.%bound_method.37e)]
// CHECK:STDOUT:     %int.convert_checked.loc5_37.2: init Core.IntLiteral = call %bound_method.loc5_37.3(%N.ref) [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc5_37.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc5_37.2 [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc5_37.2: Core.IntLiteral = converted %N.ref, %.loc5_37.1 [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %array_type.loc5_38.2: type = array_type %.loc5_37.2, %.loc5_31.2 [symbolic = %array_type.loc5_38.1 (constants.%array_type.a05)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref @F.1.%array_type.loc5_38.1 (%array_type.a05) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%array_type.loc5_38.1 (%array_type.a05) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
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
// CHECK:STDOUT: impl @impl.dd2: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %F.decl: %F.type.925 = fn_decl @F.2 [concrete = constants.%F.2a6] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %array_type.c9b = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type.c9b = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc9_31.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc9_31.2: type = converted %bool.make_type, %.loc9_31.1 [concrete = bool]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, %.loc9_31.2 [concrete = constants.%array_type.c9b]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.dd2.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %array_type.c9b = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.c9b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_37.1: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc5_37.1 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc5_37.1: %i32 = impl_witness_access %Self.as_wit.iface0.loc5_37.1, element0 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.ace)]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %impl.elem0.loc5_37.1, constants.%Convert.960 [symbolic = %Convert.bound (constants.%Convert.bound.69c)]
// CHECK:STDOUT:   %bound_method.loc5_37.1: <bound method> = bound_method %impl.elem0.loc5_37.1, constants.%Convert.specific_fn.8a8 [symbolic = %bound_method.loc5_37.1 (constants.%bound_method.37e)]
// CHECK:STDOUT:   %int.convert_checked.loc5_37.1: init Core.IntLiteral = call %bound_method.loc5_37.1(%impl.elem0.loc5_37.1) [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc5_38.1: type = array_type %int.convert_checked.loc5_37.1, bool [symbolic = %array_type.loc5_38.1 (constants.%array_type.a05)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70)]() -> @F.1.%array_type.loc5_38.1 (%array_type.a05);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type]() -> %return.param_patt: %array_type.c9b {
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
// CHECK:STDOUT: specific @N(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%I.facet.335) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type.b70
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_37.1 => constants.%Self.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc5_37.1 => constants.%impl.elem0.ace
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound.69c
// CHECK:STDOUT:   %bound_method.loc5_37.1 => constants.%bound_method.37e
// CHECK:STDOUT:   %int.convert_checked.loc5_37.1 => constants.%int.convert_checked
// CHECK:STDOUT:   %array_type.loc5_38.1 => constants.%array_type.a05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%I.facet.d87) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.ed1) {
// CHECK:STDOUT:   %Self => constants.%I.facet.ed1
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %Self.as_wit.iface0.loc5_37.1 => constants.%impl_witness.d8a
// CHECK:STDOUT:   %impl.elem0.loc5_37.1 => constants.%int_2.ef8
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound.fbf
// CHECK:STDOUT:   %bound_method.loc5_37.1 => constants.%bound_method.992
// CHECK:STDOUT:   %int.convert_checked.loc5_37.1 => constants.%int_2.ecc
// CHECK:STDOUT:   %array_type.loc5_38.1 => constants.%array_type.c9b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_symbolic_associated_type_in_concrete_context.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.6e6: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.659: %Z.assoc_type = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.Self: %Z.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %Z.facet.bf2: %Z.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.b30: type = facet_type <@Z where %impl.elem0 = %C.f2e> [symbolic]
// CHECK:STDOUT:   %require_complete.f02: <witness> = require_complete_type %Z_where.type.b30 [symbolic]
// CHECK:STDOUT:   %impl_witness.00e: <witness> = impl_witness (%C.f2e), @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.131: type = class_type @C, @C(%D) [concrete]
// CHECK:STDOUT:   %Z_where.type.116: type = facet_type <@Z where %impl.elem0 = %C.131> [concrete]
// CHECK:STDOUT:   %complete_type.fc8: <witness> = complete_type_witness %Z_where.type.116 [concrete]
// CHECK:STDOUT:   %impl_witness.b6f: <witness> = impl_witness (%C.f2e), @impl(%D) [concrete]
// CHECK:STDOUT:   %Z.facet.94c: %Z.type = facet_value %D, (%impl_witness.b6f) [concrete]
// CHECK:STDOUT:   %C.val: %C.131 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
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
// CHECK:STDOUT:     %T.patt.loc6_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc9_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_24: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self: %Z.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.659]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc9_37: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref.loc9_44: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:     %C.loc9_45.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc9_45.2 (constants.%C.f2e)]
// CHECK:STDOUT:     %.loc9_31: type = where_expr %.Self [symbolic = %Z_where.type (constants.%Z_where.type.b30)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.loc9_45.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%C.f2e), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness.00e)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.6e6]
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
// CHECK:STDOUT: generic impl @impl(%T.loc9_14.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc9_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %C.loc9_45.2: type = class_type @C, @C(%T.loc9_14.2) [symbolic = %C.loc9_45.2 (constants.%C.f2e)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z where constants.%impl.elem0 = %C.loc9_45.2 (constants.%C.f2e)> [symbolic = %Z_where.type (constants.%Z_where.type.b30)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%Z_where.type (%Z_where.type.b30) [symbolic = %require_complete (constants.%require_complete.f02)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%C.f2e), @impl(%T.loc9_14.2) [symbolic = %impl_witness (constants.%impl_witness.00e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc9_24 as %.loc9_31 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc6_9.1: type) {
// CHECK:STDOUT:   %T.loc6_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
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
// CHECK:STDOUT:     %a.patt: %C.f2e = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %D.ref.loc22_28: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%D) [concrete = constants.%C.131]
// CHECK:STDOUT:   %.loc22_21.2: ref %C.131 = temporary_storage
// CHECK:STDOUT:   %.loc22_21.3: init %C.131 = class_init (), %.loc22_21.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc22_21.4: ref %C.131 = temporary %.loc22_21.2, %.loc22_21.3
// CHECK:STDOUT:   %.loc22_23.1: ref %C.131 = converted %.loc22_21.1, %.loc22_21.4
// CHECK:STDOUT:   %.loc22_11.1: type = splice_block %impl.elem0 [symbolic = constants.%C.f2e] {
// CHECK:STDOUT:     %D.ref.loc22_10: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.659]
// CHECK:STDOUT:     %Z.facet: %Z.type = facet_value constants.%D, (constants.%impl_witness.b6f) [concrete = constants.%Z.facet.94c]
// CHECK:STDOUT:     %.loc22_11.2: %Z.type = converted %D.ref.loc22_10, %Z.facet [concrete = constants.%Z.facet.94c]
// CHECK:STDOUT:     %as_wit.iface0: <witness> = facet_access_witness %.loc22_11.2, element0 [concrete = constants.%impl_witness.b6f]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %as_wit.iface0, element0 [symbolic = constants.%C.f2e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_23.2: %C.f2e = converted %.loc22_23.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %a: %C.f2e = bind_name a, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Self.6e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc6_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_9.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Z.facet.bf2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc9_14.2 => constants.%T.patt
// CHECK:STDOUT:   %C.loc9_45.2 => constants.%C.f2e
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.b30
// CHECK:STDOUT:   %require_complete => constants.%require_complete.f02
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.00e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@impl.%T.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%D) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc9_14.2 => constants.%T.patt
// CHECK:STDOUT:   %C.loc9_45.2 => constants.%C.131
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.116
// CHECK:STDOUT:   %require_complete => constants.%complete_type.fc8
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b6f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%D) {
// CHECK:STDOUT:   %T.loc6_9.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc6_9.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Z.facet.94c) {}
// CHECK:STDOUT:
