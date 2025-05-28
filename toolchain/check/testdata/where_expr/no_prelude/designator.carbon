// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/no_prelude/designator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/no_prelude/designator.carbon

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

// --- dot_self_index.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

interface Empty(W:! type) {
  let A:! type;
}

// T has index 0
// .Self has index invalid, not 1
// V has index 1, does not match .Self
//@dump-sem-ir-begin
fn H(T:! type, U: Empty(T) where .A = T*, V:! type) {}

fn G(U: Empty(C) where .A = C*) {
  H(C, U, D);
}
//@dump-sem-ir-end

// --- non_generic_arg.carbon

library "[[@TEST_NAME]]";

interface I { let T:! type; }

class C {}

// Ensure that we don't crash when checking this `where` in a non-generic context.
//@dump-sem-ir-begin
fn NotGenericF(U: I where .T == C) {}
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
// CHECK:STDOUT: --- dot_self_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Empty.type.d5a: type = generic_interface_type @Empty [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Empty.generic: %Empty.type.d5a = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Empty.type.3e5fde.2: type = facet_type <@Empty, @Empty(%T)> [symbolic]
// CHECK:STDOUT:   %.Self.c95: %Empty.type.3e5fde.2 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %Empty.assoc_type.3cf698.2: type = assoc_entity_type @Empty, @Empty(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.3715ce.2: %Empty.assoc_type.3cf698.2 = assoc_entity element0, @Empty.%A [symbolic]
// CHECK:STDOUT:   %require_complete.22f: <witness> = require_complete_type %Empty.type.3e5fde.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.a75: type = facet_access_type %.Self.c95 [symbolic]
// CHECK:STDOUT:   %Empty.lookup_impl_witness.b1d: <witness> = lookup_impl_witness %.Self.c95, @Empty, @Empty(%T) [symbolic]
// CHECK:STDOUT:   %impl.elem0.c6e: type = impl_witness_access %Empty.lookup_impl_witness.b1d, element0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Empty_where.type.b73: type = facet_type <@Empty, @Empty(%T) where %impl.elem0.c6e = %ptr.79f> [symbolic]
// CHECK:STDOUT:   %pattern_type.668: type = pattern_type %Empty_where.type.b73 [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 1 [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.af8: <witness> = require_complete_type %Empty_where.type.b73 [symbolic]
// CHECK:STDOUT:   %Empty.type.14d: type = facet_type <@Empty, @Empty(%C)> [concrete]
// CHECK:STDOUT:   %.Self.1b2: %Empty.type.14d = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Empty.assoc_type.b9c: type = assoc_entity_type @Empty, @Empty(%C) [concrete]
// CHECK:STDOUT:   %assoc0.984: %Empty.assoc_type.b9c = assoc_entity element0, @Empty.%A [concrete]
// CHECK:STDOUT:   %.Self.as_type.623: type = facet_access_type %.Self.1b2 [symbolic_self]
// CHECK:STDOUT:   %Empty.lookup_impl_witness.8df: <witness> = lookup_impl_witness %.Self.1b2, @Empty, @Empty(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f7a: type = impl_witness_access %Empty.lookup_impl_witness.8df, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Empty_where.type.c85: type = facet_type <@Empty, @Empty(%C) where %impl.elem0.f7a = %ptr.019> [concrete]
// CHECK:STDOUT:   %pattern_type.a97: type = pattern_type %Empty_where.type.c85 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.1ae: <witness> = complete_type_witness %Empty.type.14d [concrete]
// CHECK:STDOUT:   %H.specific_fn: <specific function> = specific_function %H, @H(%C, %D) [concrete]
// CHECK:STDOUT:   %complete_type.72d: <witness> = complete_type_witness %Empty_where.type.c85 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @H.%pattern_type (%pattern_type.668) = binding_pattern U [concrete]
// CHECK:STDOUT:     %U.param_patt: @H.%pattern_type (%pattern_type.668) = value_param_pattern %U.patt, call_param0 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc15_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: @H.%Empty_where.type (%Empty_where.type.b73) = value_param call_param0
// CHECK:STDOUT:     %.loc15_28.1: type = splice_block %.loc15_28.2 [symbolic = %Empty_where.type (constants.%Empty_where.type.b73)] {
// CHECK:STDOUT:       %Empty.ref: %Empty.type.d5a = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.generic]
// CHECK:STDOUT:       %T.ref.loc15_25: type = name_ref T, %T.loc15_6.1 [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:       %Empty.type.loc15_26.1: type = facet_type <@Empty, @Empty(constants.%T)> [symbolic = %Empty.type.loc15_26.2 (constants.%Empty.type.3e5fde.2)]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: @H.%Empty.type.loc15_26.2 (%Empty.type.3e5fde.2) = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.c95)]
// CHECK:STDOUT:       %.loc15_34.1: @H.%Empty.assoc_type (%Empty.assoc_type.3cf698.2) = specific_constant @A.%assoc0, @Empty(constants.%T) [symbolic = %assoc0 (constants.%assoc0.3715ce.2)]
// CHECK:STDOUT:       %A.ref: @H.%Empty.assoc_type (%Empty.assoc_type.3cf698.2) = name_ref A, %.loc15_34.1 [symbolic = %assoc0 (constants.%assoc0.3715ce.2)]
// CHECK:STDOUT:       %.Self.as_type.loc15_34.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc15_34.2 (constants.%.Self.as_type.a75)]
// CHECK:STDOUT:       %.loc15_34.2: type = converted %.Self.ref, %.Self.as_type.loc15_34.1 [symbolic = %.Self.as_type.loc15_34.2 (constants.%.Self.as_type.a75)]
// CHECK:STDOUT:       %impl.elem0.loc15_34.1: type = impl_witness_access constants.%Empty.lookup_impl_witness.b1d, element0 [symbolic = %impl.elem0.loc15_34.2 (constants.%impl.elem0.c6e)]
// CHECK:STDOUT:       %T.ref.loc15_39: type = name_ref T, %T.loc15_6.1 [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc15_40.1: type = ptr_type %T.ref.loc15_39 [symbolic = %ptr.loc15_40.2 (constants.%ptr.79f)]
// CHECK:STDOUT:       %.loc15_28.2: type = where_expr %.Self.1 [symbolic = %Empty_where.type (constants.%Empty_where.type.b73)] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc15_34.1, %ptr.loc15_40.1
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: @H.%Empty_where.type (%Empty_where.type.b73) = bind_name U, %U.param
// CHECK:STDOUT:     %V.loc15_43.1: type = bind_symbolic_name V, 1 [symbolic = %V.loc15_43.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %U.patt: %pattern_type.a97 = binding_pattern U [concrete]
// CHECK:STDOUT:     %U.param_patt: %pattern_type.a97 = value_param_pattern %U.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: %Empty_where.type.c85 = value_param call_param0
// CHECK:STDOUT:     %.loc17_18.1: type = splice_block %.loc17_18.2 [concrete = constants.%Empty_where.type.c85] {
// CHECK:STDOUT:       %Empty.ref: %Empty.type.d5a = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.generic]
// CHECK:STDOUT:       %C.ref.loc17_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Empty.type: type = facet_type <@Empty, @Empty(constants.%C)> [concrete = constants.%Empty.type.14d]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %Empty.type.14d = name_ref .Self, %.Self [symbolic_self = constants.%.Self.1b2]
// CHECK:STDOUT:       %.loc17_24.1: %Empty.assoc_type.b9c = specific_constant @A.%assoc0, @Empty(constants.%C) [concrete = constants.%assoc0.984]
// CHECK:STDOUT:       %A.ref: %Empty.assoc_type.b9c = name_ref A, %.loc17_24.1 [concrete = constants.%assoc0.984]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.623]
// CHECK:STDOUT:       %.loc17_24.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.623]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%Empty.lookup_impl_witness.8df, element0 [symbolic_self = constants.%impl.elem0.f7a]
// CHECK:STDOUT:       %C.ref.loc17_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref.loc17_29 [concrete = constants.%ptr.019]
// CHECK:STDOUT:       %.loc17_18.2: type = where_expr %.Self [concrete = constants.%Empty_where.type.c85] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %ptr
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: %Empty_where.type.c85 = bind_name U, %U.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%T.loc15_6.1: type, %V.loc15_43.1: type) {
// CHECK:STDOUT:   %T.loc15_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:   %Empty.type.loc15_26.2: type = facet_type <@Empty, @Empty(%T.loc15_6.2)> [symbolic = %Empty.type.loc15_26.2 (constants.%Empty.type.3e5fde.2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc15_34: <witness> = require_complete_type %Empty.type.loc15_26.2 [symbolic = %require_complete.loc15_34 (constants.%require_complete.22f)]
// CHECK:STDOUT:   %Empty.assoc_type: type = assoc_entity_type @Empty, @Empty(%T.loc15_6.2) [symbolic = %Empty.assoc_type (constants.%Empty.assoc_type.3cf698.2)]
// CHECK:STDOUT:   %assoc0: @H.%Empty.assoc_type (%Empty.assoc_type.3cf698.2) = assoc_entity element0, @Empty.%A [symbolic = %assoc0 (constants.%assoc0.3715ce.2)]
// CHECK:STDOUT:   %.Self.as_type.loc15_34.2: type = facet_access_type %.Self.2 [symbolic = %.Self.as_type.loc15_34.2 (constants.%.Self.as_type.a75)]
// CHECK:STDOUT:   %Empty.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.2, @Empty, @Empty(%T.loc15_6.2) [symbolic = %Empty.lookup_impl_witness (constants.%Empty.lookup_impl_witness.b1d)]
// CHECK:STDOUT:   %impl.elem0.loc15_34.2: type = impl_witness_access %Empty.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_34.2 (constants.%impl.elem0.c6e)]
// CHECK:STDOUT:   %ptr.loc15_40.2: type = ptr_type %T.loc15_6.2 [symbolic = %ptr.loc15_40.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %Empty_where.type: type = facet_type <@Empty, @Empty(%T.loc15_6.2) where %impl.elem0.loc15_34.2 = %ptr.loc15_40.2> [symbolic = %Empty_where.type (constants.%Empty_where.type.b73)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Empty_where.type [symbolic = %pattern_type (constants.%pattern_type.668)]
// CHECK:STDOUT:   %V.loc15_43.2: type = bind_symbolic_name V, 1 [symbolic = %V.loc15_43.2 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_17: <witness> = require_complete_type %Empty_where.type [symbolic = %require_complete.loc15_17 (constants.%require_complete.af8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param: @H.%Empty_where.type (%Empty_where.type.b73)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%U.param: %Empty_where.type.c85) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, file.%H.decl [concrete = constants.%H]
// CHECK:STDOUT:   %C.ref.loc18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %U.ref: %Empty_where.type.c85 = name_ref U, %U
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %H.specific_fn: <specific function> = specific_function %H.ref, @H(constants.%C, constants.%D) [concrete = constants.%H.specific_fn]
// CHECK:STDOUT:   %H.call: init %empty_tuple.type = call %H.specific_fn(%U.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%T, constants.%V) {
// CHECK:STDOUT:   %T.loc15_6.2 => constants.%T
// CHECK:STDOUT:   %Empty.type.loc15_26.2 => constants.%Empty.type.3e5fde.2
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.c95
// CHECK:STDOUT:   %require_complete.loc15_34 => constants.%require_complete.22f
// CHECK:STDOUT:   %Empty.assoc_type => constants.%Empty.assoc_type.3cf698.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.3715ce.2
// CHECK:STDOUT:   %.Self.as_type.loc15_34.2 => constants.%.Self.as_type.a75
// CHECK:STDOUT:   %Empty.lookup_impl_witness => constants.%Empty.lookup_impl_witness.b1d
// CHECK:STDOUT:   %impl.elem0.loc15_34.2 => constants.%impl.elem0.c6e
// CHECK:STDOUT:   %ptr.loc15_40.2 => constants.%ptr.79f
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.b73
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.668
// CHECK:STDOUT:   %V.loc15_43.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.loc15_6.2 => constants.%C
// CHECK:STDOUT:   %Empty.type.loc15_26.2 => constants.%Empty.type.14d
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.1b2
// CHECK:STDOUT:   %require_complete.loc15_34 => constants.%complete_type.1ae
// CHECK:STDOUT:   %Empty.assoc_type => constants.%Empty.assoc_type.b9c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.984
// CHECK:STDOUT:   %.Self.as_type.loc15_34.2 => constants.%.Self.as_type.623
// CHECK:STDOUT:   %Empty.lookup_impl_witness => constants.%Empty.lookup_impl_witness.8df
// CHECK:STDOUT:   %impl.elem0.loc15_34.2 => constants.%impl.elem0.f7a
// CHECK:STDOUT:   %ptr.loc15_40.2 => constants.%ptr.019
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.c85
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a97
// CHECK:STDOUT:   %V.loc15_43.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_17 => constants.%complete_type.72d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_generic_arg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I_where.type [concrete]
// CHECK:STDOUT:   %NotGenericF.type: type = fn_type @NotGenericF [concrete]
// CHECK:STDOUT:   %NotGenericF: %NotGenericF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %NotGenericF.decl: %NotGenericF.type = fn_decl @NotGenericF [concrete = constants.%NotGenericF] {
// CHECK:STDOUT:     %U.patt: %pattern_type = binding_pattern U [concrete]
// CHECK:STDOUT:     %U.param_patt: %pattern_type = value_param_pattern %U.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: %I_where.type = value_param call_param0
// CHECK:STDOUT:     %.loc10_21.1: type = splice_block %.loc10_21.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc10_27: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc10_21.2: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_equivalent %impl.elem0, %C.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: %I_where.type = bind_name U, %U.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotGenericF(%U.param: %I_where.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
