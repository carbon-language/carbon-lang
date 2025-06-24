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
  let Member:! type;
}

//@dump-sem-ir-begin
fn PeriodSelf(T:! I where .Self == ());

fn PeriodMember(U:! I where .Member == ());

fn TypeSelfImpls(V:! type where .Self impls I);
//@dump-sem-ir-end

// --- fail_wrong_member.carbon

library "[[@TEST_NAME]]";

interface J {
  let Member:! type;
}

// CHECK:STDERR: fail_wrong_member.carbon:[[@LINE+4]]:31: error: member name `Mismatch` not found in `J` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn PeriodMismatch(W:! J where .Mismatch = {});
// CHECK:STDERR:                               ^~~~~~~~~
// CHECK:STDERR:
fn PeriodMismatch(W:! J where .Mismatch = {});

// --- fail_designator_matches_var.carbon

library "[[@TEST_NAME]]";

fn Foo() -> () {
  var x: ();
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

// CHECK:STDOUT: --- success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Member [concrete]
// CHECK:STDOUT:   %.Self.258: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where TODO> [concrete]
// CHECK:STDOUT:   %T: %I_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f03: type = pattern_type %I_where.type [concrete]
// CHECK:STDOUT:   %PeriodSelf.type: type = fn_type @PeriodSelf [concrete]
// CHECK:STDOUT:   %PeriodSelf: %PeriodSelf.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.258 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.258, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %U: %I_where.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %PeriodMember.type: type = fn_type @PeriodMember [concrete]
// CHECK:STDOUT:   %PeriodMember: %PeriodMember.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.644: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls @I> [concrete]
// CHECK:STDOUT:   %V: %type_where = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b5: type = pattern_type %type_where [concrete]
// CHECK:STDOUT:   %TypeSelfImpls.type: type = fn_type @TypeSelfImpls [concrete]
// CHECK:STDOUT:   %TypeSelfImpls: %TypeSelfImpls.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %PeriodSelf.decl: %PeriodSelf.type = fn_decl @PeriodSelf [concrete = constants.%PeriodSelf] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f03 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_21.1: type = splice_block %.loc9_21.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %.loc9_37: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc9_21.2: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %.loc9_37
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_15.1: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PeriodMember.decl: %PeriodMember.type = fn_decl @PeriodMember [concrete = constants.%PeriodMember] {
// CHECK:STDOUT:     %U.patt: %pattern_type.f03 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_23.1: type = splice_block %.loc11_23.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %Member.ref: %I.assoc_type = name_ref Member, @Member.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_29: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc11_41: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc11_23.2: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_equivalent %impl.elem0, %.loc11_41
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc11_17.1: %I_where.type = bind_symbolic_name U, 0 [symbolic = %U.loc11_17.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TypeSelfImpls.decl: %TypeSelfImpls.type = fn_decl @TypeSelfImpls [concrete = constants.%TypeSelfImpls] {
// CHECK:STDOUT:     %V.patt: %pattern_type.2b5 = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_27.1: type = splice_block %.loc13_27.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.644]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.loc13_27.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, %I.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc13_18.1: %type_where = bind_symbolic_name V, 0 [symbolic = %V.loc13_18.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PeriodSelf(%T.loc9_15.1: %I_where.type) {
// CHECK:STDOUT:   %T.loc9_15.2: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_15.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PeriodMember(%U.loc11_17.1: %I_where.type) {
// CHECK:STDOUT:   %U.loc11_17.2: %I_where.type = bind_symbolic_name U, 0 [symbolic = %U.loc11_17.2 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TypeSelfImpls(%V.loc13_18.1: %type_where) {
// CHECK:STDOUT:   %V.loc13_18.2: %type_where = bind_symbolic_name V, 0 [symbolic = %V.loc13_18.2 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PeriodSelf(constants.%T) {
// CHECK:STDOUT:   %T.loc9_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PeriodMember(constants.%U) {
// CHECK:STDOUT:   %U.loc11_17.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TypeSelfImpls(constants.%V) {
// CHECK:STDOUT:   %V.loc13_18.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
