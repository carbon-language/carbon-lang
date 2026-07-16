// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/designator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/designator.carbon

// --- success.carbon

library "[[@TEST_NAME]]";

interface I {
  let Member: type;
}

//@dump-sem-ir-begin
fn PeriodSelf(generic T: I where .Self == ());

fn PeriodMember(generic U: I where .Member == ());

fn TypeSelfImpls(generic V: type where .Self impls I);
//@dump-sem-ir-end

// --- fail_wrong_member.carbon

library "[[@TEST_NAME]]";

interface J {
  let Member: type;
}

// CHECK:STDERR: fail_wrong_member.carbon:[[@LINE+4]]:38: error: member name `Mismatch` not found in `J` [MemberNameNotFoundInSpecificScope]
// CHECK:STDERR: fn PeriodMismatch(generic W: J where .Mismatch = {});
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR:
fn PeriodMismatch(generic W: J where .Mismatch = {});

// --- fail_designator_matches_var.carbon

library "[[@TEST_NAME]]";

fn Foo() -> () {
  var unused x: ();
  // CHECK:STDERR: fail_designator_matches_var.carbon:[[@LINE+5]]:10: error: name `.Self` not found [NameNotFound]
  // CHECK:STDERR:   return .x;
  // CHECK:STDERR:          ^~
  // CHECK:STDERR: fail_designator_matches_var.carbon: note: designator may only be used when `.Self` is in scope [NoPeriodSelfForDesignator]
  // CHECK:STDERR:
  return .x;
}

// --- fail_unknown_designator.carbon

library "[[@TEST_NAME]]";

fn Bar() -> () {
  // CHECK:STDERR: fail_unknown_designator.carbon:[[@LINE+5]]:10: error: name `.Self` not found [NameNotFound]
  // CHECK:STDERR:   return .undef;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_unknown_designator.carbon: note: designator may only be used when `.Self` is in scope [NoPeriodSelfForDesignator]
  // CHECK:STDERR:
  return .undef;
}

// --- fail_dot_self_method_return_value.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_dot_self_method_return_value.carbon:[[@LINE+4]]:27: error: name `.Self` not found [NameNotFound]
  // CHECK:STDERR:   fn F() -> Self { return .Self; }
  // CHECK:STDERR:                           ^~~~~
  // CHECK:STDERR:
  fn F() -> Self { return .Self; }
}

// --- fail_dot_self_method_return_type.carbon

library "[[@TEST_NAME]]";

class D {
  // CHECK:STDERR: fail_dot_self_method_return_type.carbon:[[@LINE+4]]:13: error: name `.Self` not found [NameNotFound]
  // CHECK:STDERR:   fn G() -> .Self { return Self; }
  // CHECK:STDERR:             ^~~~~
  // CHECK:STDERR:
  fn G() -> .Self { return Self; }
}

// --- fail_impls_constraint_does_not_constrain_self.carbon
library "[[@TEST_NAME]]";

interface I {}

// CHECK:STDERR: fail_impls_constraint_does_not_constrain_self.carbon:[[@LINE+4]]:52: error: constraint in `where` clause without a designator; expected `.Self` or a member access like `.M` [WhereWithoutDesignator]
// CHECK:STDERR: fn F(generic U: type, unused generic T: type where U impls I) {}
// CHECK:STDERR:                                                    ^~~~~~~~~
// CHECK:STDERR:
fn F(generic U: type, unused generic T: type where U impls I) {}

// --- fail_equality_constraint_does_not_constrain_self.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
}

// CHECK:STDERR: fail_equality_constraint_does_not_constrain_self.carbon:[[@LINE+4]]:49: error: constraint in `where` clause without a designator; expected `.Self` or a member access like `.M` [WhereWithoutDesignator]
// CHECK:STDERR: fn F(generic U: I, unused generic T: type where U.X == ()) {}
// CHECK:STDERR:                                                 ^~~~~~~~~
// CHECK:STDERR:
fn F(generic U: I, unused generic T: type where U.X == ()) {}

// --- fail_combined_constraint_does_not_constrain_self.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
}

// CHECK:STDERR: fail_combined_constraint_does_not_constrain_self.carbon:[[@LINE+8]]:49: error: constraint in `where` clause without a designator; expected `.Self` or a member access like `.M` [WhereWithoutDesignator]
// CHECK:STDERR: fn F(generic U: I, unused generic T: type where U impls I and U.X == ()) {}
// CHECK:STDERR:                                                 ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_combined_constraint_does_not_constrain_self.carbon:[[@LINE+4]]:63: error: constraint in `where` clause without a designator; expected `.Self` or a member access like `.M` [WhereWithoutDesignator]
// CHECK:STDERR: fn F(generic U: I, unused generic T: type where U impls I and U.X == ()) {}
// CHECK:STDERR:                                                               ^~~~~~~~~
// CHECK:STDERR:
fn F(generic U: I, unused generic T: type where U impls I and U.X == ()) {}

// --- impls_constraint_does_constrain_self.carbon
library "[[@TEST_NAME]]";

interface I(T: type) {}

class C(T: type);

fn F(unused generic T: type where C(.Self) impls I(())) {}

fn G(unused generic T: type where C(()) impls I(.Self)) {}

// --- fail_where_without_designator_in_one_equiv_constraint.carbon
library "[[@TEST_NAME]]";

interface I {}
interface J(T: type) {}
class A(T: type);
class B;
class C;

// CHECK:STDERR: fail_where_without_designator_in_one_equiv_constraint.carbon:[[@LINE+4]]:53: error: constraint in `where` clause without a designator; expected `.Self` or a member access like `.M` [WhereWithoutDesignator]
// CHECK:STDERR: fn F(unused generic T: type where A(.Self) == B and A(B) == B) {}
// CHECK:STDERR:                                                     ^~~~~~~~~
// CHECK:STDERR:
fn F(unused generic T: type where A(.Self) == B and A(B) == B) {}

interface Z {
  let Z0: type;
}

// CHECK:STDERR: fail_where_without_designator_in_one_equiv_constraint.carbon:[[@LINE+4]]:48: error: constraint in `where` clause without a designator; expected `.Self` or a member access like `.M` [WhereWithoutDesignator]
// CHECK:STDERR: fn G(unused generic T: Z where A(.Z0) == B and A(B) == B) {}
// CHECK:STDERR:                                                ^~~~~~~~~
// CHECK:STDERR:
fn G(unused generic T: Z where A(.Z0) == B and A(B) == B) {}

// --- fail_where_without_self_designator_in_one_impls_interface.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
interface I {}
interface J(T: type) {}
class C;

// CHECK:STDERR: fail_where_without_self_designator_in_one_impls_interface.carbon:[[@LINE+4]]:35: error: constraint in `where` clause without a designator; expected `.Self` or a member access like `.M` [WhereWithoutDesignator]
// CHECK:STDERR: fn F(unused generic T: type where C impls (I & J(.Self))) {}
// CHECK:STDERR:                                   ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(unused generic T: type where C impls (I & J(.Self))) {}

// --- fail_where_without_member_designator_in_one_impls_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z0: type;
}
interface I {}
interface J(T: type) {}
class C;

// CHECK:STDERR: fail_where_without_member_designator_in_one_impls_interface.carbon:[[@LINE+4]]:32: error: constraint in `where` clause without a designator; expected `.Self` or a member access like `.M` [WhereWithoutDesignator]
// CHECK:STDERR: fn F(unused generic T: Z where C impls (I & J(.Z0))) {}
// CHECK:STDERR:                                ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(unused generic T: Z where C impls (I & J(.Z0))) {}

// --- fail_where_without_self_designator_in_one_impls_named_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {}
constraint I {}
interface J(T: type) {}
constraint K(T: type) {
  require T impls J(Self);
}
class C;

// CHECK:STDERR: fail_where_without_self_designator_in_one_impls_named_constraint.carbon:[[@LINE+4]]:35: error: constraint in `where` clause without a designator; expected `.Self` or a member access like `.M` [WhereWithoutDesignator]
// CHECK:STDERR: fn F(unused generic T: type where C impls (I & K(.Self))) {}
// CHECK:STDERR:                                   ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(unused generic T: type where C impls (I & K(.Self))) {}

// --- fail_where_without_member_designator_in_one_impls_named_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z0: type;
}
constraint I {}
interface J(T: type) {}
constraint K(T: type) {
  require T impls J(Self);
}
class C;


// CHECK:STDERR: fail_where_without_member_designator_in_one_impls_named_constraint.carbon:[[@LINE+4]]:32: error: constraint in `where` clause without a designator; expected `.Self` or a member access like `.M` [WhereWithoutDesignator]
// CHECK:STDERR: fn F(unused generic T: Z where C impls (I & K(.Z0))) {}
// CHECK:STDERR:                                ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(unused generic T: Z where C impls (I & K(.Z0))) {}

// --- constraint_does_constrain_designator_as_self.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
constraint N(T: type) {
  require T impls Z(Self);
}
class C;

fn F(unused generic T: type where C impls N(.Self)) {}

// --- constraint_does_constrain_designator_as_specific.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
constraint N(T: type) {
  require impls Z(T);
}
class C;

fn F(unused generic T: type where C impls N(.Self)) {}

// --- todo_fail_constraint_does_not_constrain_designator.carbon
library "[[@TEST_NAME]]";

interface Z {}
constraint N(T: type) {
  require impls Z;
}
class C;

// TODO: This `.Self` is not actually constrained by `C impls N(.Self)`, so it
// should fail.
fn F(unused generic T: type where C impls N(.Self)) {}

// CHECK:STDOUT: --- success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%Member [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.3a0: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.f73: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.603: type = pattern_type %I_where.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.603 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I_where.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %PeriodSelf.type: type = fn_type @PeriodSelf [concrete]
// CHECK:STDOUT:   %PeriodSelf: %PeriodSelf.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.e90: type = facet_access_type %.Self.frozen.3a0 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.ced: <witness> = lookup_impl_witness %.Self.frozen.3a0, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.125: type = impl_witness_access %I.lookup_impl_witness.ced, element0 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.78b: <witness> = lookup_impl_witness %.Self.f73, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e55: type = impl_witness_access %I.lookup_impl_witness.78b, element0 [symbolic_self]
// CHECK:STDOUT:   %U.patt: %pattern_type.603 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: %I_where.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %PeriodMember.type: type = fn_type @PeriodMember [concrete]
// CHECK:STDOUT:   %PeriodMember: %PeriodMember.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.bce: type = facet_access_type %.Self.frozen.197 [symbolic_self]
// CHECK:STDOUT:   %.Self.e0b: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.e0b [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls @I> [concrete]
// CHECK:STDOUT:   %pattern_type.951: type = pattern_type %type_where [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.951 = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V: %type_where = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %TypeSelfImpls.type: type = fn_type @TypeSelfImpls [concrete]
// CHECK:STDOUT:   %TypeSelfImpls: %TypeSelfImpls.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %PeriodSelf.decl: %PeriodSelf.type = fn_decl @PeriodSelf [concrete = constants.%PeriodSelf] {
// CHECK:STDOUT:     %T.patt.loc9_24.1: %pattern_type.603 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_24.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_28.1: type = splice_block %.loc9_28.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc9_24: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.frozen.loc9_28: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc9_34.1: %I.type = name_ref .Self, %.Self.frozen.loc9_28 [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %.loc9_44: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %equiv.loc9_40.1 = requirement_equivalent %.Self.ref.loc9_34.1, %.loc9_44 [concrete]
// CHECK:STDOUT:       %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.f73]
// CHECK:STDOUT:       %.Self.ref.loc9_34.2: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.f73]
// CHECK:STDOUT:       %equiv.loc9_40.2 = requirement_equivalent %.Self.ref.loc9_34.2, %.loc9_44 [concrete]
// CHECK:STDOUT:       %.loc9_28.2: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:         %equiv.loc9_40.2 = requirement_equivalent %.Self.ref.loc9_34.2, %.loc9_44 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_24.2: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc9_24.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PeriodMember.decl: %PeriodMember.type = fn_decl @PeriodMember [concrete = constants.%PeriodMember] {
// CHECK:STDOUT:     %U.patt.loc11_26.1: %pattern_type.603 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_26.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_30.1: type = splice_block %.loc11_30.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc11_26: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.frozen.loc11_30: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self.frozen.loc11_30 [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type.e90]
// CHECK:STDOUT:       %.loc11_36: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type.e90]
// CHECK:STDOUT:       %Member.ref: %I.assoc_type = name_ref Member, @Member.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc11_36.1: type = impl_witness_access constants.%I.lookup_impl_witness.ced, element0 [symbolic_self = constants.%impl.elem0.125]
// CHECK:STDOUT:       %.loc11_48: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %equiv.loc11_44.1 = requirement_equivalent %impl.elem0.loc11_36.1, %.loc11_48 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc11_36.2: type = impl_witness_access constants.%I.lookup_impl_witness.78b, element0 [symbolic_self = constants.%impl.elem0.e55]
// CHECK:STDOUT:       %equiv.loc11_44.2 = requirement_equivalent %impl.elem0.loc11_36.2, %.loc11_48 [concrete]
// CHECK:STDOUT:       %.loc11_30.2: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:         %equiv.loc11_44.2 = requirement_equivalent %impl.elem0.loc11_36.2, %.loc11_48 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc11_26.2: %I_where.type = symbolic_binding U, 0 [symbolic = %U.loc11_26.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TypeSelfImpls.decl: %TypeSelfImpls.type = fn_decl @TypeSelfImpls [concrete = constants.%TypeSelfImpls] {
// CHECK:STDOUT:     %V.patt.loc13_27.1: %pattern_type.951 = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_27.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_34.1: type = splice_block %.loc13_34.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       %.Self.frozen.loc13_27: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc13_29: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.Self.frozen.loc13_34: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %.loc13_29 [concrete]
// CHECK:STDOUT:       %.Self.ref.loc13_40.1: %type = name_ref .Self, %.Self.frozen.loc13_34 [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.as_type.loc13_40.1: type = facet_access_type %.Self.ref.loc13_40.1 [symbolic_self = constants.%.Self.frozen.as_type.bce]
// CHECK:STDOUT:       %.loc13_40.1: type = converted %.Self.ref.loc13_40.1, %.Self.as_type.loc13_40.1 [symbolic_self = constants.%.Self.frozen.as_type.bce]
// CHECK:STDOUT:       %impls.loc13_46.1 = requirement_impls %.loc13_40.1, %I.ref [concrete]
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.e0b]
// CHECK:STDOUT:       %.Self.ref.loc13_40.2: %type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.e0b]
// CHECK:STDOUT:       %.Self.as_type.loc13_40.2: type = facet_access_type %.Self.ref.loc13_40.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_40.2: type = converted %.Self.ref.loc13_40.1, %.Self.as_type.loc13_40.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impls.loc13_46.2 = requirement_impls %.loc13_40.2, %I.ref [concrete]
// CHECK:STDOUT:       %.loc13_34.2: type = where_expr [concrete = constants.%type_where] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %.loc13_29 [concrete]
// CHECK:STDOUT:         %impls.loc13_46.2 = requirement_impls %.loc13_40.2, %I.ref [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc13_27.2: %type_where = symbolic_binding V, 0 [symbolic = %V.loc13_27.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PeriodSelf(%T.loc9_24.2: %I_where.type) {
// CHECK:STDOUT:   %T.patt.loc9_24.2: %pattern_type.603 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_24.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc9_24.1: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc9_24.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PeriodMember(%U.loc11_26.2: %I_where.type) {
// CHECK:STDOUT:   %U.patt.loc11_26.2: %pattern_type.603 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_26.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc11_26.1: %I_where.type = symbolic_binding U, 0 [symbolic = %U.loc11_26.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TypeSelfImpls(%V.loc13_27.2: %type_where) {
// CHECK:STDOUT:   %V.patt.loc13_27.2: %pattern_type.951 = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_27.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc13_27.1: %type_where = symbolic_binding V, 0 [symbolic = %V.loc13_27.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PeriodSelf(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc9_24.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc9_24.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PeriodMember(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc11_26.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc11_26.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TypeSelfImpls(constants.%V) {
// CHECK:STDOUT:   %V.patt.loc13_27.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc13_27.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
