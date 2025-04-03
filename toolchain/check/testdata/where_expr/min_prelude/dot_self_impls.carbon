// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/min_prelude/dot_self_impls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/min_prelude/dot_self_impls.carbon

// --- name_lookup_success.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F();
}

interface J {
  fn G();
}

fn INotJ(T:! I where .Self impls J) {
  T.F();
  T.(I.F)();

  T.(J.G)();
}

fn TypeNotJ(T:! type where .Self impls J) {
  T.(J.G)();
}

// --- fail_name_lookup_with_i.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F();
}

interface J {
  fn G();
}

fn INotJ(T:! I where .Self impls J) {
  // CHECK:STDERR: fail_name_lookup_with_i.carbon:[[@LINE+4]]:3: error: member name `G` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   T.G();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  T.G();
}

// --- fail_name_lookup_with_type.carbon
library "[[@TEST_NAME]]";

interface J {
  fn G();
}

fn TypeNotJ(T:! type where .Self impls J) {
  // CHECK:STDERR: fail_name_lookup_with_type.carbon:[[@LINE+4]]:3: error: member name `G` not found [MemberNameNotFound]
  // CHECK:STDERR:   T.G();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  T.G();
}

// --- name_lookup_instance_success.carbon
library "[[@TEST_NAME]]";

interface I {
  fn FNonInstance();
  fn FSelf[self: Self]();
}

interface J {
  fn GNonInstance();
  fn GSelf[self: Self]();
}

fn INotJ[T:! I where .Self impls J](x: T) {
  x.FNonInstance();
  x.FSelf();
  x.(I.FSelf)();

  x.(J.GSelf)();
}

fn TypeNotJ[T:! type where .Self impls J](x: T) {
  x.(J.GSelf)();
}

// --- fail_name_lookup_instance.carbon
library "[[@TEST_NAME]]";

interface I {
  fn FNonInstance();
  fn FSelf[self: Self]();
}

interface J {
  fn GNonInstance();
  fn GSelf[self: Self]();
}

fn INotJ[T:! I where .Self impls J](x: T) {
  // CHECK:STDERR: fail_name_lookup_instance.carbon:[[@LINE+4]]:3: error: member name `GNonInstance` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   x.GNonInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  x.GNonInstance();

  // CHECK:STDERR: fail_name_lookup_instance.carbon:[[@LINE+4]]:3: error: member name `GSelf` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   x.GSelf();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  x.GSelf();
}

// --- compare_equal.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn AssertSame[T:! type](a: WrapType(T), b: WrapType(T)) {}
fn Type(T:! type) -> WrapType(T) { return {}; }

interface I;
interface J;
interface K;

fn Test() {
  AssertSame(Type(I), Type(I where .Self impls I));
  AssertSame(Type(I), Type(I where .Self impls I & I));
  AssertSame(Type(I & J), Type(J & I where .Self impls I));
  AssertSame(Type(I & J), Type(J & I where .Self impls J));
  AssertSame(Type(I & J), Type(J & I where .Self impls I & J));
  AssertSame(Type(I & J where .Self impls K), Type(J & I where .Self impls I & K));
  AssertSame(Type(I & J where .Self impls K), Type(J & I where .Self impls J & K));
  AssertSame(Type(I & J where .Self impls K), Type(J & I where .Self impls K & I & J));
  AssertSame(Type(I where .Self impls J & K), Type(I where .Self impls K & I & J));
  AssertSame(Type(I where .Self impls J & K),
             Type(I where .Self impls (J where .Self impls K)));
  AssertSame(Type(I where .Self impls J & K),
             Type(I where .Self impls (K where .Self impls J)));
  AssertSame(Type(I where .Self impls J & K),
             Type(I where .Self impls (type where .Self impls (J where .Self impls K))));
}

interface I {}
interface J {}
interface K {}

// --- fail_compare_not_equal.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn Same[T:! type](a: WrapType(T), b: WrapType(T)) {}
fn Type(T:! type) -> WrapType(T) { return {}; }

interface I {}
interface J {}

fn Test() {
  // CHECK:STDERR: fail_compare_not_equal.carbon:[[@LINE+7]]:3: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   Same(Type(I where .Self impls J), Type(J where .Self impls I));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_compare_not_equal.carbon:[[@LINE-10]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Same[T:! type](a: WrapType(T), b: WrapType(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Same(Type(I where .Self impls J), Type(J where .Self impls I));
}

// CHECK:STDOUT: --- name_lookup_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type.733 [concrete]
// CHECK:STDOUT:   %assoc0.a5e: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0.4b0: %J.assoc_type = assoc_entity element0, @J.%G.decl [concrete]
// CHECK:STDOUT:   %.Self.258: %I.type.733 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.258 [symbolic_self]
// CHECK:STDOUT:   %I.type.100: type = facet_type <@I where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T.527: %I.type.100 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.6c2: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %INotJ.type: type = fn_type @INotJ [concrete]
// CHECK:STDOUT:   %INotJ: %INotJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.0d1: type = facet_access_type %T.527 [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0.fd0: <witness> = facet_access_witness %T.527, element0 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.733 = facet_value %T.as_type.0d1, (%T.as_wit.iface0.fd0) [symbolic]
// CHECK:STDOUT:   %.5a0: type = fn_type_with_self_type %F.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.594: %.5a0 = impl_witness_access %T.as_wit.iface0.fd0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.577: <specific function> = specific_impl_function %impl.elem0.594, @F(%I.facet) [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface1: <witness> = facet_access_witness %T.527, element1 [symbolic]
// CHECK:STDOUT:   %J.facet.09e: %J.type = facet_value %T.as_type.0d1, (%T.as_wit.iface1) [symbolic]
// CHECK:STDOUT:   %.201: type = fn_type_with_self_type %G.type, %J.facet.09e [symbolic]
// CHECK:STDOUT:   %impl.elem0.ff4: %.201 = impl_witness_access %T.as_wit.iface1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.907: <specific function> = specific_impl_function %impl.elem0.ff4, @G(%J.facet.09e) [symbolic]
// CHECK:STDOUT:   %.Self.644: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %facet_type: type = facet_type <type where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T.ccd: %facet_type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.371: %facet_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %TypeNotJ.type: type = fn_type @TypeNotJ [concrete]
// CHECK:STDOUT:   %TypeNotJ: %TypeNotJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_wit.iface0.bba: <witness> = facet_access_witness %T.ccd, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.3df: type = facet_access_type %T.ccd [symbolic]
// CHECK:STDOUT:   %J.facet.085: %J.type = facet_value %T.as_type.3df, (%T.as_wit.iface0.bba) [symbolic]
// CHECK:STDOUT:   %.9ac: type = fn_type_with_self_type %G.type, %J.facet.085 [symbolic]
// CHECK:STDOUT:   %impl.elem0.7ca: %.9ac = impl_witness_access %T.as_wit.iface0.bba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a7f: <specific function> = specific_impl_function %impl.elem0.7ca, @G(%J.facet.085) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .INotJ = %INotJ.decl
// CHECK:STDOUT:     .TypeNotJ = %TypeNotJ.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %INotJ.decl: %INotJ.type = fn_decl @INotJ [concrete = constants.%INotJ] {
// CHECK:STDOUT:     %T.patt.loc11_10.1: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt.6c2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_16.1: type = splice_block %.loc11_16.2 [concrete = constants.%I.type.100] {
// CHECK:STDOUT:       %I.ref.loc11: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:       %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %.Self.ref: %I.type.733 = name_ref .Self, %.Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %J.ref.loc11: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_16.2: type = where_expr %.Self [concrete = constants.%I.type.100] {
// CHECK:STDOUT:         requirement_impls %.loc11_22, %J.ref.loc11
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_10.1: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc11_10.2 (constants.%T.527)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TypeNotJ.decl: %TypeNotJ.type = fn_decl @TypeNotJ [concrete = constants.%TypeNotJ] {
// CHECK:STDOUT:     %T.patt.loc18_13.1: %facet_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_13.2 (constants.%T.patt.371)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_22.1: type = splice_block %.loc18_22.2 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.644]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.644]
// CHECK:STDOUT:       %J.ref.loc18: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.loc18_22.2: type = where_expr %.Self [concrete = constants.%facet_type] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, %J.ref.loc18
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_13.1: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc18_13.2 (constants.%T.ccd)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.a5e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0.4b0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @INotJ(%T.loc11_10.1: %I.type.100) {
// CHECK:STDOUT:   %T.loc11_10.2: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc11_10.2 (constants.%T.527)]
// CHECK:STDOUT:   %T.patt.loc11_10.2: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt.6c2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc12_4.2: type = facet_access_type %T.loc11_10.2 [symbolic = %T.as_type.loc12_4.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc12_4.2: <witness> = facet_access_witness %T.loc11_10.2, element0 [symbolic = %T.as_wit.iface0.loc12_4.2 (constants.%T.as_wit.iface0.fd0)]
// CHECK:STDOUT:   %I.facet.loc12: %I.type.733 = facet_value %T.as_type.loc12_4.2, (%T.as_wit.iface0.loc12_4.2) [symbolic = %I.facet.loc12 (constants.%I.facet)]
// CHECK:STDOUT:   %.loc12_4.2: type = fn_type_with_self_type constants.%F.type, %I.facet.loc12 [symbolic = %.loc12_4.2 (constants.%.5a0)]
// CHECK:STDOUT:   %impl.elem0.loc12_4.2: @INotJ.%.loc12_4.2 (%.5a0) = impl_witness_access %T.as_wit.iface0.loc12_4.2, element0 [symbolic = %impl.elem0.loc12_4.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_4.2: <specific function> = specific_impl_function %impl.elem0.loc12_4.2, @F(%I.facet.loc12) [symbolic = %specific_impl_fn.loc12_4.2 (constants.%specific_impl_fn.577)]
// CHECK:STDOUT:   %T.as_wit.iface1.loc15_4.2: <witness> = facet_access_witness %T.loc11_10.2, element1 [symbolic = %T.as_wit.iface1.loc15_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:   %J.facet.loc15_4.2: %J.type = facet_value %T.as_type.loc12_4.2, (%T.as_wit.iface1.loc15_4.2) [symbolic = %J.facet.loc15_4.2 (constants.%J.facet.09e)]
// CHECK:STDOUT:   %.loc15_4.2: type = fn_type_with_self_type constants.%G.type, %J.facet.loc15_4.2 [symbolic = %.loc15_4.2 (constants.%.201)]
// CHECK:STDOUT:   %impl.elem0.loc15_4.2: @INotJ.%.loc15_4.2 (%.201) = impl_witness_access %T.as_wit.iface1.loc15_4.2, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0.ff4)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_4.2: <specific function> = specific_impl_function %impl.elem0.loc15_4.2, @G(%J.facet.loc15_4.2) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn.907)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc11_10.1: %I.type.100) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc12: %I.type.100 = name_ref T, %T.loc11_10.1 [symbolic = %T.loc11_10.2 (constants.%T.527)]
// CHECK:STDOUT:     %F.ref.loc12: %I.assoc_type = name_ref F, @I.%assoc0 [concrete = constants.%assoc0.a5e]
// CHECK:STDOUT:     %T.as_type.loc12_4.1: type = facet_access_type %T.ref.loc12 [symbolic = %T.as_type.loc12_4.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:     %.loc12_4.1: type = converted %T.ref.loc12, %T.as_type.loc12_4.1 [symbolic = %T.as_type.loc12_4.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc12_4.1: <witness> = facet_access_witness %T.ref.loc12, element0 [symbolic = %T.as_wit.iface0.loc12_4.2 (constants.%T.as_wit.iface0.fd0)]
// CHECK:STDOUT:     %impl.elem0.loc12_4.1: @INotJ.%.loc12_4.2 (%.5a0) = impl_witness_access %T.as_wit.iface0.loc12_4.1, element0 [symbolic = %impl.elem0.loc12_4.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %specific_impl_fn.loc12_4.1: <specific function> = specific_impl_function %impl.elem0.loc12_4.1, @F(constants.%I.facet) [symbolic = %specific_impl_fn.loc12_4.2 (constants.%specific_impl_fn.577)]
// CHECK:STDOUT:     %.loc12_7: init %empty_tuple.type = call %specific_impl_fn.loc12_4.1()
// CHECK:STDOUT:     %T.ref.loc13: %I.type.100 = name_ref T, %T.loc11_10.1 [symbolic = %T.loc11_10.2 (constants.%T.527)]
// CHECK:STDOUT:     %I.ref.loc13: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:     %F.ref.loc13: %I.assoc_type = name_ref F, @I.%assoc0 [concrete = constants.%assoc0.a5e]
// CHECK:STDOUT:     %T.as_wit.iface0.loc13: <witness> = facet_access_witness constants.%T.527, element0 [symbolic = %T.as_wit.iface0.loc12_4.2 (constants.%T.as_wit.iface0.fd0)]
// CHECK:STDOUT:     %T.as_type.loc13: type = facet_access_type constants.%T.527 [symbolic = %T.as_type.loc12_4.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:     %I.facet.loc13: %I.type.733 = facet_value %T.as_type.loc13, (%T.as_wit.iface0.loc13) [symbolic = %I.facet.loc12 (constants.%I.facet)]
// CHECK:STDOUT:     %.loc13_4: %I.type.733 = converted %T.ref.loc13, %I.facet.loc13 [symbolic = %I.facet.loc12 (constants.%I.facet)]
// CHECK:STDOUT:     %as_wit.iface0.loc13: <witness> = facet_access_witness %.loc13_4, element0 [symbolic = %T.as_wit.iface0.loc12_4.2 (constants.%T.as_wit.iface0.fd0)]
// CHECK:STDOUT:     %impl.elem0.loc13: @INotJ.%.loc12_4.2 (%.5a0) = impl_witness_access %as_wit.iface0.loc13, element0 [symbolic = %impl.elem0.loc12_4.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %specific_impl_fn.loc13: <specific function> = specific_impl_function %impl.elem0.loc13, @F(constants.%I.facet) [symbolic = %specific_impl_fn.loc12_4.2 (constants.%specific_impl_fn.577)]
// CHECK:STDOUT:     %.loc13_11: init %empty_tuple.type = call %specific_impl_fn.loc13()
// CHECK:STDOUT:     %T.ref.loc15: %I.type.100 = name_ref T, %T.loc11_10.1 [symbolic = %T.loc11_10.2 (constants.%T.527)]
// CHECK:STDOUT:     %J.ref.loc15: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %G.ref: %J.assoc_type = name_ref G, @J.%assoc0 [concrete = constants.%assoc0.4b0]
// CHECK:STDOUT:     %T.as_wit.iface1.loc15_4.1: <witness> = facet_access_witness constants.%T.527, element1 [symbolic = %T.as_wit.iface1.loc15_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:     %T.as_type.loc15: type = facet_access_type constants.%T.527 [symbolic = %T.as_type.loc12_4.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:     %J.facet.loc15_4.1: %J.type = facet_value %T.as_type.loc15, (%T.as_wit.iface1.loc15_4.1) [symbolic = %J.facet.loc15_4.2 (constants.%J.facet.09e)]
// CHECK:STDOUT:     %.loc15_4.1: %J.type = converted %T.ref.loc15, %J.facet.loc15_4.1 [symbolic = %J.facet.loc15_4.2 (constants.%J.facet.09e)]
// CHECK:STDOUT:     %as_wit.iface0.loc15: <witness> = facet_access_witness %.loc15_4.1, element0 [symbolic = %T.as_wit.iface1.loc15_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:     %impl.elem0.loc15_4.1: @INotJ.%.loc15_4.2 (%.201) = impl_witness_access %as_wit.iface0.loc15, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0.ff4)]
// CHECK:STDOUT:     %specific_impl_fn.loc15_4.1: <specific function> = specific_impl_function %impl.elem0.loc15_4.1, @G(constants.%J.facet.09e) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn.907)]
// CHECK:STDOUT:     %.loc15_11: init %empty_tuple.type = call %specific_impl_fn.loc15_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TypeNotJ(%T.loc18_13.1: %facet_type) {
// CHECK:STDOUT:   %T.loc18_13.2: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc18_13.2 (constants.%T.ccd)]
// CHECK:STDOUT:   %T.patt.loc18_13.2: %facet_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_13.2 (constants.%T.patt.371)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_wit.iface0.loc19_4.2: <witness> = facet_access_witness %T.loc18_13.2, element0 [symbolic = %T.as_wit.iface0.loc19_4.2 (constants.%T.as_wit.iface0.bba)]
// CHECK:STDOUT:   %T.as_type.loc19_4.2: type = facet_access_type %T.loc18_13.2 [symbolic = %T.as_type.loc19_4.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:   %J.facet.loc19_4.2: %J.type = facet_value %T.as_type.loc19_4.2, (%T.as_wit.iface0.loc19_4.2) [symbolic = %J.facet.loc19_4.2 (constants.%J.facet.085)]
// CHECK:STDOUT:   %.loc19_4.2: type = fn_type_with_self_type constants.%G.type, %J.facet.loc19_4.2 [symbolic = %.loc19_4.2 (constants.%.9ac)]
// CHECK:STDOUT:   %impl.elem0.loc19_4.2: @TypeNotJ.%.loc19_4.2 (%.9ac) = impl_witness_access %T.as_wit.iface0.loc19_4.2, element0 [symbolic = %impl.elem0.loc19_4.2 (constants.%impl.elem0.7ca)]
// CHECK:STDOUT:   %specific_impl_fn.loc19_4.2: <specific function> = specific_impl_function %impl.elem0.loc19_4.2, @G(%J.facet.loc19_4.2) [symbolic = %specific_impl_fn.loc19_4.2 (constants.%specific_impl_fn.a7f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc18_13.1: %facet_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %facet_type = name_ref T, %T.loc18_13.1 [symbolic = %T.loc18_13.2 (constants.%T.ccd)]
// CHECK:STDOUT:     %J.ref.loc19: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %G.ref: %J.assoc_type = name_ref G, @J.%assoc0 [concrete = constants.%assoc0.4b0]
// CHECK:STDOUT:     %T.as_wit.iface0.loc19_4.1: <witness> = facet_access_witness constants.%T.ccd, element0 [symbolic = %T.as_wit.iface0.loc19_4.2 (constants.%T.as_wit.iface0.bba)]
// CHECK:STDOUT:     %T.as_type.loc19_4.1: type = facet_access_type constants.%T.ccd [symbolic = %T.as_type.loc19_4.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     %J.facet.loc19_4.1: %J.type = facet_value %T.as_type.loc19_4.1, (%T.as_wit.iface0.loc19_4.1) [symbolic = %J.facet.loc19_4.2 (constants.%J.facet.085)]
// CHECK:STDOUT:     %.loc19_4.1: %J.type = converted %T.ref, %J.facet.loc19_4.1 [symbolic = %J.facet.loc19_4.2 (constants.%J.facet.085)]
// CHECK:STDOUT:     %as_wit.iface0: <witness> = facet_access_witness %.loc19_4.1, element0 [symbolic = %T.as_wit.iface0.loc19_4.2 (constants.%T.as_wit.iface0.bba)]
// CHECK:STDOUT:     %impl.elem0.loc19_4.1: @TypeNotJ.%.loc19_4.2 (%.9ac) = impl_witness_access %as_wit.iface0, element0 [symbolic = %impl.elem0.loc19_4.2 (constants.%impl.elem0.7ca)]
// CHECK:STDOUT:     %specific_impl_fn.loc19_4.1: <specific function> = specific_impl_function %impl.elem0.loc19_4.1, @G(constants.%J.facet.085) [symbolic = %specific_impl_fn.loc19_4.2 (constants.%specific_impl_fn.a7f)]
// CHECK:STDOUT:     %.loc19_11: init %empty_tuple.type = call %specific_impl_fn.loc19_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @INotJ(constants.%T.527) {
// CHECK:STDOUT:   %T.loc11_10.2 => constants.%T.527
// CHECK:STDOUT:   %T.patt.loc11_10.2 => constants.%T.patt.6c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%J.facet.09e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(@INotJ.%I.facet.loc12) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(@INotJ.%J.facet.loc15_4.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TypeNotJ(constants.%T.ccd) {
// CHECK:STDOUT:   %T.loc18_13.2 => constants.%T.ccd
// CHECK:STDOUT:   %T.patt.loc18_13.2 => constants.%T.patt.371
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%J.facet.085) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(@TypeNotJ.%J.facet.loc19_4.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_lookup_with_i.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type.733 [concrete]
// CHECK:STDOUT:   %assoc0.a5e: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0.4b0: %J.assoc_type = assoc_entity element0, @J.%G.decl [concrete]
// CHECK:STDOUT:   %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.type.100: type = facet_type <@I where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T: %I.type.100 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %INotJ.type: type = fn_type @INotJ [concrete]
// CHECK:STDOUT:   %INotJ: %INotJ.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .INotJ = %INotJ.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %INotJ.decl: %INotJ.type = fn_decl @INotJ [concrete = constants.%INotJ] {
// CHECK:STDOUT:     %T.patt.loc11_10.1: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_16.1: type = splice_block %.loc11_16.2 [concrete = constants.%I.type.100] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:       %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %I.type.733 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_16.2: type = where_expr %.Self [concrete = constants.%I.type.100] {
// CHECK:STDOUT:         requirement_impls %.loc11_22, %J.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_10.1: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc11_10.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.a5e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0.4b0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @INotJ(%T.loc11_10.1: %I.type.100) {
// CHECK:STDOUT:   %T.loc11_10.2: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc11_10.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_10.2: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc11_10.1: %I.type.100) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type.100 = name_ref T, %T.loc11_10.1 [symbolic = %T.loc11_10.2 (constants.%T)]
// CHECK:STDOUT:     %G.ref: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @INotJ(constants.%T) {
// CHECK:STDOUT:   %T.loc11_10.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_10.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_lookup_with_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%G.decl [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %facet_type: type = facet_type <type where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T: %facet_type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %facet_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %TypeNotJ.type: type = fn_type @TypeNotJ [concrete]
// CHECK:STDOUT:   %TypeNotJ: %TypeNotJ.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .TypeNotJ = %TypeNotJ.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %TypeNotJ.decl: %TypeNotJ.type = fn_decl @TypeNotJ [concrete = constants.%TypeNotJ] {
// CHECK:STDOUT:     %T.patt.loc7_13.1: %facet_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_22.1: type = splice_block %.loc7_22.2 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.loc7_22.2: type = where_expr %.Self [concrete = constants.%facet_type] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, %J.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_13.1: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc7_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TypeNotJ(%T.loc7_13.1: %facet_type) {
// CHECK:STDOUT:   %T.loc7_13.2: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc7_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_13.2: %facet_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc7_13.1: %facet_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %facet_type = name_ref T, %T.loc7_13.1 [symbolic = %T.loc7_13.2 (constants.%T)]
// CHECK:STDOUT:     %G.ref: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TypeNotJ(constants.%T) {
// CHECK:STDOUT:   %T.loc7_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_13.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- name_lookup_instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %FNonInstance.type: type = fn_type @FNonInstance [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %FNonInstance: %FNonInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type.733 [concrete]
// CHECK:STDOUT:   %assoc0.8b6: %I.assoc_type = assoc_entity element0, @I.%FNonInstance.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %FSelf.type: type = fn_type @FSelf [concrete]
// CHECK:STDOUT:   %FSelf: %FSelf.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.685: %I.assoc_type = assoc_entity element1, @I.%FSelf.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %GNonInstance.type: type = fn_type @GNonInstance [concrete]
// CHECK:STDOUT:   %GNonInstance: %GNonInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0.a91: %J.assoc_type = assoc_entity element0, @J.%GNonInstance.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %GSelf.type: type = fn_type @GSelf [concrete]
// CHECK:STDOUT:   %GSelf: %GSelf.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.26a: %J.assoc_type = assoc_entity element1, @J.%GSelf.decl [concrete]
// CHECK:STDOUT:   %.Self.258: %I.type.733 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.258 [symbolic_self]
// CHECK:STDOUT:   %I.type.100: type = facet_type <@I where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T.527: %I.type.100 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.6c2: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.0d1: type = facet_access_type %T.527 [symbolic]
// CHECK:STDOUT:   %INotJ.type: type = fn_type @INotJ [concrete]
// CHECK:STDOUT:   %INotJ: %INotJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.df2: <witness> = require_complete_type %T.as_type.0d1 [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0.fd0: <witness> = facet_access_witness %T.527, element0 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.733 = facet_value %T.as_type.0d1, (%T.as_wit.iface0.fd0) [symbolic]
// CHECK:STDOUT:   %.b38: type = fn_type_with_self_type %FNonInstance.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.b38 = impl_witness_access %T.as_wit.iface0.fd0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e7d: <specific function> = specific_impl_function %impl.elem0, @FNonInstance(%I.facet) [symbolic]
// CHECK:STDOUT:   %.e5a: type = fn_type_with_self_type %FSelf.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1.d3e: %.e5a = impl_witness_access %T.as_wit.iface0.fd0, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3d3: <specific function> = specific_impl_function %impl.elem1.d3e, @FSelf(%I.facet) [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface1: <witness> = facet_access_witness %T.527, element1 [symbolic]
// CHECK:STDOUT:   %J.facet.09e: %J.type = facet_value %T.as_type.0d1, (%T.as_wit.iface1) [symbolic]
// CHECK:STDOUT:   %.5b4: type = fn_type_with_self_type %GSelf.type, %J.facet.09e [symbolic]
// CHECK:STDOUT:   %impl.elem1.d26: %.5b4 = impl_witness_access %T.as_wit.iface1, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.83c: <specific function> = specific_impl_function %impl.elem1.d26, @GSelf(%J.facet.09e) [symbolic]
// CHECK:STDOUT:   %.Self.644: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %facet_type: type = facet_type <type where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T.ccd: %facet_type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.371: %facet_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.3df: type = facet_access_type %T.ccd [symbolic]
// CHECK:STDOUT:   %TypeNotJ.type: type = fn_type @TypeNotJ [concrete]
// CHECK:STDOUT:   %TypeNotJ: %TypeNotJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.2a5: <witness> = require_complete_type %T.as_type.3df [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0.bba: <witness> = facet_access_witness %T.ccd, element0 [symbolic]
// CHECK:STDOUT:   %J.facet.085: %J.type = facet_value %T.as_type.3df, (%T.as_wit.iface0.bba) [symbolic]
// CHECK:STDOUT:   %.f3d: type = fn_type_with_self_type %GSelf.type, %J.facet.085 [symbolic]
// CHECK:STDOUT:   %impl.elem1.ed8: %.f3d = impl_witness_access %T.as_wit.iface0.bba, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.8fa: <specific function> = specific_impl_function %impl.elem1.ed8, @GSelf(%J.facet.085) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .INotJ = %INotJ.decl
// CHECK:STDOUT:     .TypeNotJ = %TypeNotJ.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %INotJ.decl: %INotJ.type = fn_decl @INotJ [concrete = constants.%INotJ] {
// CHECK:STDOUT:     %T.patt.loc13_10.1: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_10.2 (constants.%T.patt.6c2)]
// CHECK:STDOUT:     %x.patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type.0d1) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type.0d1) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_16.1: type = splice_block %.loc13_16.2 [concrete = constants.%I.type.100] {
// CHECK:STDOUT:       %I.ref.loc13: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:       %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %.Self.ref: %I.type.733 = name_ref .Self, %.Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %J.ref.loc13: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_16.2: type = where_expr %.Self [concrete = constants.%I.type.100] {
// CHECK:STDOUT:         requirement_impls %.loc13_22, %J.ref.loc13
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_10.1: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc13_10.2 (constants.%T.527)]
// CHECK:STDOUT:     %x.param: @INotJ.%T.as_type.loc13_40.2 (%T.as_type.0d1) = value_param call_param0
// CHECK:STDOUT:     %.loc13_40.1: type = splice_block %.loc13_40.2 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type.0d1)] {
// CHECK:STDOUT:       %T.ref: %I.type.100 = name_ref T, %T.loc13_10.1 [symbolic = %T.loc13_10.2 (constants.%T.527)]
// CHECK:STDOUT:       %T.as_type.loc13_40.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:       %.loc13_40.2: type = converted %T.ref, %T.as_type.loc13_40.1 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @INotJ.%T.as_type.loc13_40.2 (%T.as_type.0d1) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TypeNotJ.decl: %TypeNotJ.type = fn_decl @TypeNotJ [concrete = constants.%TypeNotJ] {
// CHECK:STDOUT:     %T.patt.loc21_13.1: %facet_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_13.2 (constants.%T.patt.371)]
// CHECK:STDOUT:     %x.patt: @TypeNotJ.%T.as_type.loc21_46.2 (%T.as_type.3df) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @TypeNotJ.%T.as_type.loc21_46.2 (%T.as_type.3df) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_22.1: type = splice_block %.loc21_22.2 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.644]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.644]
// CHECK:STDOUT:       %J.ref.loc21: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.loc21_22.2: type = where_expr %.Self [concrete = constants.%facet_type] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, %J.ref.loc21
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_13.1: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc21_13.2 (constants.%T.ccd)]
// CHECK:STDOUT:     %x.param: @TypeNotJ.%T.as_type.loc21_46.2 (%T.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc21_46.1: type = splice_block %.loc21_46.2 [symbolic = %T.as_type.loc21_46.2 (constants.%T.as_type.3df)] {
// CHECK:STDOUT:       %T.ref: %facet_type = name_ref T, %T.loc21_13.1 [symbolic = %T.loc21_13.2 (constants.%T.ccd)]
// CHECK:STDOUT:       %T.as_type.loc21_46.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc21_46.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:       %.loc21_46.2: type = converted %T.ref, %T.as_type.loc21_46.1 [symbolic = %T.as_type.loc21_46.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TypeNotJ.%T.as_type.loc21_46.2 (%T.as_type.3df) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %FNonInstance.decl: %FNonInstance.type = fn_decl @FNonInstance [concrete = constants.%FNonInstance] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %FNonInstance.decl [concrete = constants.%assoc0.8b6]
// CHECK:STDOUT:   %FSelf.decl: %FSelf.type = fn_decl @FSelf [concrete = constants.%FSelf] {
// CHECK:STDOUT:     %self.patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc5_18.1: type = splice_block %.loc5_18.2 [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type.733 = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc5_18.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc5_18.2: type = converted %Self.ref, %Self.as_type.loc5_18.2 [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %FSelf.decl [concrete = constants.%assoc1.685]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .FNonInstance = %assoc0
// CHECK:STDOUT:   .FSelf = %assoc1
// CHECK:STDOUT:   witness = (%FNonInstance.decl, %FSelf.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %GNonInstance.decl: %GNonInstance.type = fn_decl @GNonInstance [concrete = constants.%GNonInstance] {} {}
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %GNonInstance.decl [concrete = constants.%assoc0.a91]
// CHECK:STDOUT:   %GSelf.decl: %GSelf.type = fn_decl @GSelf [concrete = constants.%GSelf] {
// CHECK:STDOUT:     %self.patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc10_18.1: type = splice_block %.loc10_18.2 [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc10_18.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc10_18.2: type = converted %Self.ref, %Self.as_type.loc10_18.2 [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %GSelf.decl [concrete = constants.%assoc1.26a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .GNonInstance = %assoc0
// CHECK:STDOUT:   .GSelf = %assoc1
// CHECK:STDOUT:   witness = (%GNonInstance.decl, %GSelf.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FNonInstance(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FSelf(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc5_18.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GNonInstance(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GSelf(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc10_18.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @INotJ(%T.loc13_10.1: %I.type.100) {
// CHECK:STDOUT:   %T.loc13_10.2: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc13_10.2 (constants.%T.527)]
// CHECK:STDOUT:   %T.patt.loc13_10.2: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_10.2 (constants.%T.patt.6c2)]
// CHECK:STDOUT:   %T.as_type.loc13_40.2: type = facet_access_type %T.loc13_10.2 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @INotJ.%T.as_type.loc13_40.2 (%T.as_type.0d1) [symbolic = %require_complete (constants.%require_complete.df2)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc14_4.2: <witness> = facet_access_witness %T.loc13_10.2, element0 [symbolic = %T.as_wit.iface0.loc14_4.2 (constants.%T.as_wit.iface0.fd0)]
// CHECK:STDOUT:   %I.facet: %I.type.733 = facet_value %T.as_type.loc13_40.2, (%T.as_wit.iface0.loc14_4.2) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc14_4.2: type = fn_type_with_self_type constants.%FNonInstance.type, %I.facet [symbolic = %.loc14_4.2 (constants.%.b38)]
// CHECK:STDOUT:   %impl.elem0.loc14_4.2: @INotJ.%.loc14_4.2 (%.b38) = impl_witness_access %T.as_wit.iface0.loc14_4.2, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2: <specific function> = specific_impl_function %impl.elem0.loc14_4.2, @FNonInstance(%I.facet) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.e7d)]
// CHECK:STDOUT:   %.loc15_4.2: type = fn_type_with_self_type constants.%FSelf.type, %I.facet [symbolic = %.loc15_4.2 (constants.%.e5a)]
// CHECK:STDOUT:   %impl.elem1.loc15_4.2: @INotJ.%.loc15_4.2 (%.e5a) = impl_witness_access %T.as_wit.iface0.loc14_4.2, element1 [symbolic = %impl.elem1.loc15_4.2 (constants.%impl.elem1.d3e)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_4.2: <specific function> = specific_impl_function %impl.elem1.loc15_4.2, @FSelf(%I.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn.3d3)]
// CHECK:STDOUT:   %T.as_wit.iface1.loc18_4.2: <witness> = facet_access_witness %T.loc13_10.2, element1 [symbolic = %T.as_wit.iface1.loc18_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type.loc13_40.2, (%T.as_wit.iface1.loc18_4.2) [symbolic = %J.facet (constants.%J.facet.09e)]
// CHECK:STDOUT:   %.loc18_4: type = fn_type_with_self_type constants.%GSelf.type, %J.facet [symbolic = %.loc18_4 (constants.%.5b4)]
// CHECK:STDOUT:   %impl.elem1.loc18_4.2: @INotJ.%.loc18_4 (%.5b4) = impl_witness_access %T.as_wit.iface1.loc18_4.2, element1 [symbolic = %impl.elem1.loc18_4.2 (constants.%impl.elem1.d26)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_4.2: <specific function> = specific_impl_function %impl.elem1.loc18_4.2, @GSelf(%J.facet) [symbolic = %specific_impl_fn.loc18_4.2 (constants.%specific_impl_fn.83c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc13_10.1: %I.type.100](%x.param_patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type.0d1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc14: @INotJ.%T.as_type.loc13_40.2 (%T.as_type.0d1) = name_ref x, %x
// CHECK:STDOUT:     %FNonInstance.ref: %I.assoc_type = name_ref FNonInstance, @I.%assoc0 [concrete = constants.%assoc0.8b6]
// CHECK:STDOUT:     %T.as_type.loc14: type = facet_access_type constants.%T.527 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:     %.loc14_4.1: type = converted constants.%T.527, %T.as_type.loc14 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc14_4.1: <witness> = facet_access_witness constants.%T.527, element0 [symbolic = %T.as_wit.iface0.loc14_4.2 (constants.%T.as_wit.iface0.fd0)]
// CHECK:STDOUT:     %impl.elem0.loc14_4.1: @INotJ.%.loc14_4.2 (%.b38) = impl_witness_access %T.as_wit.iface0.loc14_4.1, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_4.1: <specific function> = specific_impl_function %impl.elem0.loc14_4.1, @FNonInstance(constants.%I.facet) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.e7d)]
// CHECK:STDOUT:     %.loc14_18: init %empty_tuple.type = call %specific_impl_fn.loc14_4.1()
// CHECK:STDOUT:     %x.ref.loc15: @INotJ.%T.as_type.loc13_40.2 (%T.as_type.0d1) = name_ref x, %x
// CHECK:STDOUT:     %FSelf.ref.loc15: %I.assoc_type = name_ref FSelf, @I.%assoc1 [concrete = constants.%assoc1.685]
// CHECK:STDOUT:     %T.as_type.loc15: type = facet_access_type constants.%T.527 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:     %.loc15_4.1: type = converted constants.%T.527, %T.as_type.loc15 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc15: <witness> = facet_access_witness constants.%T.527, element0 [symbolic = %T.as_wit.iface0.loc14_4.2 (constants.%T.as_wit.iface0.fd0)]
// CHECK:STDOUT:     %impl.elem1.loc15_4.1: @INotJ.%.loc15_4.2 (%.e5a) = impl_witness_access %T.as_wit.iface0.loc15, element1 [symbolic = %impl.elem1.loc15_4.2 (constants.%impl.elem1.d3e)]
// CHECK:STDOUT:     %bound_method.loc15_4: <bound method> = bound_method %x.ref.loc15, %impl.elem1.loc15_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_4.1: <specific function> = specific_impl_function %impl.elem1.loc15_4.1, @FSelf(constants.%I.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn.3d3)]
// CHECK:STDOUT:     %bound_method.loc15_11: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_4.1
// CHECK:STDOUT:     %.loc15_11: init %empty_tuple.type = call %bound_method.loc15_11(%x.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @INotJ.%T.as_type.loc13_40.2 (%T.as_type.0d1) = name_ref x, %x
// CHECK:STDOUT:     %I.ref.loc16: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:     %FSelf.ref.loc16: %I.assoc_type = name_ref FSelf, @I.%assoc1 [concrete = constants.%assoc1.685]
// CHECK:STDOUT:     %T.as_wit.iface0.loc16: <witness> = facet_access_witness constants.%T.527, element0 [symbolic = %T.as_wit.iface0.loc14_4.2 (constants.%T.as_wit.iface0.fd0)]
// CHECK:STDOUT:     %impl.elem1.loc16: @INotJ.%.loc15_4.2 (%.e5a) = impl_witness_access %T.as_wit.iface0.loc16, element1 [symbolic = %impl.elem1.loc15_4.2 (constants.%impl.elem1.d3e)]
// CHECK:STDOUT:     %bound_method.loc16_4: <bound method> = bound_method %x.ref.loc16, %impl.elem1.loc16
// CHECK:STDOUT:     %specific_impl_fn.loc16: <specific function> = specific_impl_function %impl.elem1.loc16, @FSelf(constants.%I.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn.3d3)]
// CHECK:STDOUT:     %bound_method.loc16_15: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16
// CHECK:STDOUT:     %.loc16: init %empty_tuple.type = call %bound_method.loc16_15(%x.ref.loc16)
// CHECK:STDOUT:     %x.ref.loc18: @INotJ.%T.as_type.loc13_40.2 (%T.as_type.0d1) = name_ref x, %x
// CHECK:STDOUT:     %J.ref.loc18: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %GSelf.ref: %J.assoc_type = name_ref GSelf, @J.%assoc1 [concrete = constants.%assoc1.26a]
// CHECK:STDOUT:     %T.as_wit.iface1.loc18_4.1: <witness> = facet_access_witness constants.%T.527, element1 [symbolic = %T.as_wit.iface1.loc18_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:     %impl.elem1.loc18_4.1: @INotJ.%.loc18_4 (%.5b4) = impl_witness_access %T.as_wit.iface1.loc18_4.1, element1 [symbolic = %impl.elem1.loc18_4.2 (constants.%impl.elem1.d26)]
// CHECK:STDOUT:     %bound_method.loc18_4: <bound method> = bound_method %x.ref.loc18, %impl.elem1.loc18_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_4.1: <specific function> = specific_impl_function %impl.elem1.loc18_4.1, @GSelf(constants.%J.facet.09e) [symbolic = %specific_impl_fn.loc18_4.2 (constants.%specific_impl_fn.83c)]
// CHECK:STDOUT:     %bound_method.loc18_15: <bound method> = bound_method %x.ref.loc18, %specific_impl_fn.loc18_4.1
// CHECK:STDOUT:     %.loc18_15: init %empty_tuple.type = call %bound_method.loc18_15(%x.ref.loc18)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TypeNotJ(%T.loc21_13.1: %facet_type) {
// CHECK:STDOUT:   %T.loc21_13.2: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc21_13.2 (constants.%T.ccd)]
// CHECK:STDOUT:   %T.patt.loc21_13.2: %facet_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_13.2 (constants.%T.patt.371)]
// CHECK:STDOUT:   %T.as_type.loc21_46.2: type = facet_access_type %T.loc21_13.2 [symbolic = %T.as_type.loc21_46.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @TypeNotJ.%T.as_type.loc21_46.2 (%T.as_type.3df) [symbolic = %require_complete (constants.%require_complete.2a5)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc22_4.2: <witness> = facet_access_witness %T.loc21_13.2, element0 [symbolic = %T.as_wit.iface0.loc22_4.2 (constants.%T.as_wit.iface0.bba)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type.loc21_46.2, (%T.as_wit.iface0.loc22_4.2) [symbolic = %J.facet (constants.%J.facet.085)]
// CHECK:STDOUT:   %.loc22_4: type = fn_type_with_self_type constants.%GSelf.type, %J.facet [symbolic = %.loc22_4 (constants.%.f3d)]
// CHECK:STDOUT:   %impl.elem1.loc22_4.2: @TypeNotJ.%.loc22_4 (%.f3d) = impl_witness_access %T.as_wit.iface0.loc22_4.2, element1 [symbolic = %impl.elem1.loc22_4.2 (constants.%impl.elem1.ed8)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_4.2: <specific function> = specific_impl_function %impl.elem1.loc22_4.2, @GSelf(%J.facet) [symbolic = %specific_impl_fn.loc22_4.2 (constants.%specific_impl_fn.8fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc21_13.1: %facet_type](%x.param_patt: @TypeNotJ.%T.as_type.loc21_46.2 (%T.as_type.3df)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TypeNotJ.%T.as_type.loc21_46.2 (%T.as_type.3df) = name_ref x, %x
// CHECK:STDOUT:     %J.ref.loc22: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %GSelf.ref: %J.assoc_type = name_ref GSelf, @J.%assoc1 [concrete = constants.%assoc1.26a]
// CHECK:STDOUT:     %T.as_wit.iface0.loc22_4.1: <witness> = facet_access_witness constants.%T.ccd, element0 [symbolic = %T.as_wit.iface0.loc22_4.2 (constants.%T.as_wit.iface0.bba)]
// CHECK:STDOUT:     %impl.elem1.loc22_4.1: @TypeNotJ.%.loc22_4 (%.f3d) = impl_witness_access %T.as_wit.iface0.loc22_4.1, element1 [symbolic = %impl.elem1.loc22_4.2 (constants.%impl.elem1.ed8)]
// CHECK:STDOUT:     %bound_method.loc22_4: <bound method> = bound_method %x.ref, %impl.elem1.loc22_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc22_4.1: <specific function> = specific_impl_function %impl.elem1.loc22_4.1, @GSelf(constants.%J.facet.085) [symbolic = %specific_impl_fn.loc22_4.2 (constants.%specific_impl_fn.8fa)]
// CHECK:STDOUT:     %bound_method.loc22_15: <bound method> = bound_method %x.ref, %specific_impl_fn.loc22_4.1
// CHECK:STDOUT:     %.loc22_15: init %empty_tuple.type = call %bound_method.loc22_15(%x.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FNonInstance(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FSelf(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc5_18.1 => constants.%Self.as_type.b70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GNonInstance(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GSelf(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc10_18.1 => constants.%Self.as_type.3df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @INotJ(constants.%T.527) {
// CHECK:STDOUT:   %T.loc13_10.2 => constants.%T.527
// CHECK:STDOUT:   %T.patt.loc13_10.2 => constants.%T.patt.6c2
// CHECK:STDOUT:   %T.as_type.loc13_40.2 => constants.%T.as_type.0d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FNonInstance(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FSelf(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_18.1 => constants.%T.as_type.0d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GSelf(constants.%J.facet.09e) {
// CHECK:STDOUT:   %Self => constants.%J.facet.09e
// CHECK:STDOUT:   %Self.as_type.loc10_18.1 => constants.%T.as_type.0d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FNonInstance(@INotJ.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FSelf(@INotJ.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GSelf(@INotJ.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TypeNotJ(constants.%T.ccd) {
// CHECK:STDOUT:   %T.loc21_13.2 => constants.%T.ccd
// CHECK:STDOUT:   %T.patt.loc21_13.2 => constants.%T.patt.371
// CHECK:STDOUT:   %T.as_type.loc21_46.2 => constants.%T.as_type.3df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GSelf(constants.%J.facet.085) {
// CHECK:STDOUT:   %Self => constants.%J.facet.085
// CHECK:STDOUT:   %Self.as_type.loc10_18.1 => constants.%T.as_type.3df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GSelf(@TypeNotJ.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_lookup_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %FNonInstance.type: type = fn_type @FNonInstance [concrete]
// CHECK:STDOUT:   %FNonInstance: %FNonInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type.733 [concrete]
// CHECK:STDOUT:   %assoc0.8b6: %I.assoc_type = assoc_entity element0, @I.%FNonInstance.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %FSelf.type: type = fn_type @FSelf [concrete]
// CHECK:STDOUT:   %FSelf: %FSelf.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.685: %I.assoc_type = assoc_entity element1, @I.%FSelf.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %GNonInstance.type: type = fn_type @GNonInstance [concrete]
// CHECK:STDOUT:   %GNonInstance: %GNonInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0.a91: %J.assoc_type = assoc_entity element0, @J.%GNonInstance.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %GSelf.type: type = fn_type @GSelf [concrete]
// CHECK:STDOUT:   %GSelf: %GSelf.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.26a: %J.assoc_type = assoc_entity element1, @J.%GSelf.decl [concrete]
// CHECK:STDOUT:   %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.type.100: type = facet_type <@I where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T: %I.type.100 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %INotJ.type: type = fn_type @INotJ [concrete]
// CHECK:STDOUT:   %INotJ: %INotJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .INotJ = %INotJ.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %INotJ.decl: %INotJ.type = fn_decl @INotJ [concrete = constants.%INotJ] {
// CHECK:STDOUT:     %T.patt.loc13_10.1: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_16.1: type = splice_block %.loc13_16.2 [concrete = constants.%I.type.100] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:       %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %I.type.733 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_16.2: type = where_expr %.Self [concrete = constants.%I.type.100] {
// CHECK:STDOUT:         requirement_impls %.loc13_22, %J.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_10.1: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc13_10.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc13_40.1: type = splice_block %.loc13_40.2 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.100 = name_ref T, %T.loc13_10.1 [symbolic = %T.loc13_10.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc13_40.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc13_40.2: type = converted %T.ref, %T.as_type.loc13_40.1 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %FNonInstance.decl: %FNonInstance.type = fn_decl @FNonInstance [concrete = constants.%FNonInstance] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %FNonInstance.decl [concrete = constants.%assoc0.8b6]
// CHECK:STDOUT:   %FSelf.decl: %FSelf.type = fn_decl @FSelf [concrete = constants.%FSelf] {
// CHECK:STDOUT:     %self.patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc5_18.1: type = splice_block %.loc5_18.2 [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type.733 = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc5_18.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc5_18.2: type = converted %Self.ref, %Self.as_type.loc5_18.2 [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %FSelf.decl [concrete = constants.%assoc1.685]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .FNonInstance = %assoc0
// CHECK:STDOUT:   .FSelf = %assoc1
// CHECK:STDOUT:   .GNonInstance = <poisoned>
// CHECK:STDOUT:   .GSelf = <poisoned>
// CHECK:STDOUT:   witness = (%FNonInstance.decl, %FSelf.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %GNonInstance.decl: %GNonInstance.type = fn_decl @GNonInstance [concrete = constants.%GNonInstance] {} {}
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %GNonInstance.decl [concrete = constants.%assoc0.a91]
// CHECK:STDOUT:   %GSelf.decl: %GSelf.type = fn_decl @GSelf [concrete = constants.%GSelf] {
// CHECK:STDOUT:     %self.patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc10_18.1: type = splice_block %.loc10_18.2 [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc10_18.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc10_18.2: type = converted %Self.ref, %Self.as_type.loc10_18.2 [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %GSelf.decl [concrete = constants.%assoc1.26a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .GNonInstance = %assoc0
// CHECK:STDOUT:   .GSelf = %assoc1
// CHECK:STDOUT:   witness = (%GNonInstance.decl, %GSelf.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FNonInstance(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FSelf(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc5_18.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GNonInstance(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GSelf(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc10_18.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @INotJ(%T.loc13_10.1: %I.type.100) {
// CHECK:STDOUT:   %T.loc13_10.2: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc13_10.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_10.2: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc13_40.2: type = facet_access_type %T.loc13_10.2 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @INotJ.%T.as_type.loc13_40.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc13_10.1: %I.type.100](%x.param_patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc18: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %GNonInstance.ref: <error> = name_ref GNonInstance, <error> [concrete = <error>]
// CHECK:STDOUT:     %x.ref.loc24: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %GSelf.ref: <error> = name_ref GSelf, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FNonInstance(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FSelf(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc5_18.1 => constants.%Self.as_type.b70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GNonInstance(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GSelf(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc10_18.1 => constants.%Self.as_type.3df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @INotJ(constants.%T) {
// CHECK:STDOUT:   %T.loc13_10.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc13_40.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compare_equal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %WrapType.type: type = generic_class_type @WrapType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %WrapType.generic: %WrapType.type = struct_value () [concrete]
// CHECK:STDOUT:   %WrapType.aa1: type = class_type @WrapType, @WrapType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AssertSame.type: type = fn_type @AssertSame [concrete]
// CHECK:STDOUT:   %AssertSame: %AssertSame.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.5a3: <witness> = require_complete_type %WrapType.aa1 [symbolic]
// CHECK:STDOUT:   %Type.type: type = fn_type @Type [concrete]
// CHECK:STDOUT:   %Type: %Type.type = struct_value () [concrete]
// CHECK:STDOUT:   %WrapType.val.892: %WrapType.aa1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %J.type.eee: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %K.type.693: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %WrapType.fca: type = class_type @WrapType, @WrapType(%I.type.733) [concrete]
// CHECK:STDOUT:   %Type.specific_fn.605: <specific function> = specific_function %Type, @Type(%I.type.733) [concrete]
// CHECK:STDOUT:   %.Self.258: %I.type.733 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.541: type = facet_access_type %.Self.258 [symbolic_self]
// CHECK:STDOUT:   %AssertSame.specific_fn.0ab: <specific function> = specific_function %AssertSame, @AssertSame(%I.type.733) [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.25f: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, imports.%Core.import_ref.a93 [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.25f [symbolic]
// CHECK:STDOUT:   %impl_witness.db8: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %impl_witness.3ea: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl(type) [concrete]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.2, @impl(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.473: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%impl_witness.3ea) [concrete]
// CHECK:STDOUT:   %.2ac: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound.86c: <bound method> = bound_method %I.type.733, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.444, @Op.2(type) [concrete]
// CHECK:STDOUT:   %bound_method.f92: <bound method> = bound_method %I.type.733, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type.100: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT:   %WrapType.4f4: type = class_type @WrapType, @WrapType(%facet_type.100) [concrete]
// CHECK:STDOUT:   %Type.specific_fn.3cd: <specific function> = specific_function %Type, @Type(%facet_type.100) [concrete]
// CHECK:STDOUT:   %Op.bound.9e8: <bound method> = bound_method %J.type.eee, %Op.444 [concrete]
// CHECK:STDOUT:   %bound_method.739: <bound method> = bound_method %J.type.eee, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %.Self.d59: %facet_type.100 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.330: type = facet_access_type %.Self.d59 [symbolic_self]
// CHECK:STDOUT:   %AssertSame.specific_fn.bb9: <specific function> = specific_function %AssertSame, @AssertSame(%facet_type.100) [concrete]
// CHECK:STDOUT:   %facet_type.6ed91a.1: type = facet_type <@I & @J where .Self impls @K> [concrete]
// CHECK:STDOUT:   %WrapType.46a0db.1: type = class_type @WrapType, @WrapType(%facet_type.6ed91a.1) [concrete]
// CHECK:STDOUT:   %Type.specific_fn.bf4e16.1: <specific function> = specific_function %Type, @Type(%facet_type.6ed91a.1) [concrete]
// CHECK:STDOUT:   %facet_type.a90: type = facet_type <@I & @K> [concrete]
// CHECK:STDOUT:   %AssertSame.specific_fn.b64196.1: <specific function> = specific_function %AssertSame, @AssertSame(%facet_type.6ed91a.1) [concrete]
// CHECK:STDOUT:   %facet_type.2cb99f.1: type = facet_type <@J & @K> [concrete]
// CHECK:STDOUT:   %Op.bound.6d4: <bound method> = bound_method %K.type.693, %Op.444 [concrete]
// CHECK:STDOUT:   %bound_method.2ce: <bound method> = bound_method %K.type.693, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %Op.bound.fe2: <bound method> = bound_method %facet_type.a90, %Op.444 [concrete]
// CHECK:STDOUT:   %bound_method.0d5: <bound method> = bound_method %facet_type.a90, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type.6ed91a.2: type = facet_type <@I & @J & @K> [concrete]
// CHECK:STDOUT:   %I.type.6ed: type = facet_type <@I where .Self impls @J & @K> [concrete]
// CHECK:STDOUT:   %WrapType.46a0db.2: type = class_type @WrapType, @WrapType(%I.type.6ed) [concrete]
// CHECK:STDOUT:   %Type.specific_fn.bf4e16.2: <specific function> = specific_function %Type, @Type(%I.type.6ed) [concrete]
// CHECK:STDOUT:   %AssertSame.specific_fn.b64196.2: <specific function> = specific_function %AssertSame, @AssertSame(%I.type.6ed) [concrete]
// CHECK:STDOUT:   %.Self.968: %J.type.eee = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.78d: type = facet_access_type %.Self.968 [symbolic_self]
// CHECK:STDOUT:   %J.type.2cb: type = facet_type <@J where .Self impls @K> [concrete]
// CHECK:STDOUT:   %.Self.02e: %K.type.693 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.b21: type = facet_access_type %.Self.02e [symbolic_self]
// CHECK:STDOUT:   %K.type.c26: type = facet_type <@K where .Self impls @J> [concrete]
// CHECK:STDOUT:   %.Self.644: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %facet_type.2cb99f.2: type = facet_type <type where .Self impls @J & @K> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.ccd: %J.type.eee = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.09f: %K.type.693 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %WrapType.val.26a: %WrapType.fca = struct_value () [concrete]
// CHECK:STDOUT:   %WrapType.val.77e: %WrapType.4f4 = struct_value () [concrete]
// CHECK:STDOUT:   %WrapType.val.751449.1: %WrapType.46a0db.1 = struct_value () [concrete]
// CHECK:STDOUT:   %WrapType.val.751449.2: %WrapType.46a0db.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst101 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.08d: %BitAnd.assoc_type = import_ref Core//prelude, loc18_41, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.040: %BitAnd.type = import_ref Core//prelude, inst101 [no loc], loaded [symbolic = constants.%Self.25f]
// CHECK:STDOUT:   %Core.import_ref.51c: <witness> = import_ref Core//prelude, loc21_36, loaded [symbolic = @impl.%impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.%Op.type (%Op.type.f99) = import_ref Core//prelude, loc22_42, loaded [symbolic = @impl.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .WrapType = %WrapType.decl
// CHECK:STDOUT:     .AssertSame = %AssertSame.decl
// CHECK:STDOUT:     .Type = %Type.decl
// CHECK:STDOUT:     .I = %I.decl.loc7
// CHECK:STDOUT:     .J = %J.decl.loc8
// CHECK:STDOUT:     .K = %K.decl.loc9
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %WrapType.decl: %WrapType.type = class_decl @WrapType [concrete = constants.%WrapType.generic] {
// CHECK:STDOUT:     %T.patt.loc3_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_16.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AssertSame.decl: %AssertSame.type = fn_decl @AssertSame [concrete = constants.%AssertSame] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.patt: @AssertSame.%WrapType.loc4_38.2 (%WrapType.aa1) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @AssertSame.%WrapType.loc4_38.2 (%WrapType.aa1) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: @AssertSame.%WrapType.loc4_38.2 (%WrapType.aa1) = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: @AssertSame.%WrapType.loc4_38.2 (%WrapType.aa1) = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @AssertSame.%WrapType.loc4_38.2 (%WrapType.aa1) = value_param call_param0
// CHECK:STDOUT:     %.loc4_38: type = splice_block %WrapType.loc4_38.1 [symbolic = %WrapType.loc4_38.2 (constants.%WrapType.aa1)] {
// CHECK:STDOUT:       %WrapType.ref.loc4_28: %WrapType.type = name_ref WrapType, file.%WrapType.decl [concrete = constants.%WrapType.generic]
// CHECK:STDOUT:       %T.ref.loc4_37: type = name_ref T, %T.loc4_15.1 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:       %WrapType.loc4_38.1: type = class_type @WrapType, @WrapType(constants.%T) [symbolic = %WrapType.loc4_38.2 (constants.%WrapType.aa1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @AssertSame.%WrapType.loc4_38.2 (%WrapType.aa1) = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: @AssertSame.%WrapType.loc4_38.2 (%WrapType.aa1) = value_param call_param1
// CHECK:STDOUT:     %.loc4_54: type = splice_block %WrapType.loc4_54 [symbolic = %WrapType.loc4_38.2 (constants.%WrapType.aa1)] {
// CHECK:STDOUT:       %WrapType.ref.loc4_44: %WrapType.type = name_ref WrapType, file.%WrapType.decl [concrete = constants.%WrapType.generic]
// CHECK:STDOUT:       %T.ref.loc4_53: type = name_ref T, %T.loc4_15.1 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:       %WrapType.loc4_54: type = class_type @WrapType, @WrapType(constants.%T) [symbolic = %WrapType.loc4_38.2 (constants.%WrapType.aa1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: @AssertSame.%WrapType.loc4_38.2 (%WrapType.aa1) = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.decl: %Type.type = fn_decl @Type [concrete = constants.%Type] {
// CHECK:STDOUT:     %T.patt.loc5_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %WrapType.ref: %WrapType.type = name_ref WrapType, file.%WrapType.decl [concrete = constants.%WrapType.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_9.1 [symbolic = %T.loc5_9.2 (constants.%T)]
// CHECK:STDOUT:     %WrapType.loc5_32.1: type = class_type @WrapType, @WrapType(constants.%T) [symbolic = %WrapType.loc5_32.2 (constants.%WrapType.aa1)]
// CHECK:STDOUT:     %T.loc5_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = out_param call_param0
// CHECK:STDOUT:     %return: ref @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc7: type = interface_decl @I [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %J.decl.loc8: type = interface_decl @J [concrete = constants.%J.type.eee] {} {}
// CHECK:STDOUT:   %K.decl.loc9: type = interface_decl @K [concrete = constants.%K.type.693] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {} {}
// CHECK:STDOUT:   %I.decl.loc29: type = interface_decl @I [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %J.decl.loc30: type = interface_decl @J [concrete = constants.%J.type.eee] {} {}
// CHECK:STDOUT:   %K.decl.loc31: type = interface_decl @K [concrete = constants.%K.type.693] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type.eee = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type.693 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.09f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.08d
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl(%T) [symbolic = %impl_witness (constants.%impl_witness.db8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T) [symbolic = %Op.type (constants.%Op.type.f99)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.f99) = struct_value () [symbolic = %Op (constants.%Op.05a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%T (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.51c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @WrapType(%T.loc3_16.1: type) {
// CHECK:STDOUT:   %T.loc3_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc3_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%WrapType.aa1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AssertSame(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %WrapType.loc4_38.2: type = class_type @WrapType, @WrapType(%T.loc4_15.2) [symbolic = %WrapType.loc4_38.2 (constants.%WrapType.aa1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @AssertSame.%WrapType.loc4_38.2 (%WrapType.aa1) [symbolic = %require_complete (constants.%require_complete.5a3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_15.1: type](%a.param_patt: @AssertSame.%WrapType.loc4_38.2 (%WrapType.aa1), %b.param_patt: @AssertSame.%WrapType.loc4_38.2 (%WrapType.aa1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Type(%T.loc5_9.1: type) {
// CHECK:STDOUT:   %T.loc5_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   %WrapType.loc5_32.2: type = class_type @WrapType, @WrapType(%T.loc5_9.2) [symbolic = %WrapType.loc5_32.2 (constants.%WrapType.aa1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Type.%WrapType.loc5_32.2 (%WrapType.aa1) [symbolic = %require_complete (constants.%require_complete.5a3)]
// CHECK:STDOUT:   %WrapType.val: @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = struct_value () [symbolic = %WrapType.val (constants.%WrapType.val.892)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc5_9.1: type) -> %return.param_patt: @Type.%WrapType.loc5_32.2 (%WrapType.aa1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_44.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_44.2: init @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = class_init (), %return [symbolic = %WrapType.val (constants.%WrapType.val.892)]
// CHECK:STDOUT:     %.loc5_45: init @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = converted %.loc5_44.1, %.loc5_44.2 [symbolic = %WrapType.val (constants.%WrapType.val.892)]
// CHECK:STDOUT:     return %.loc5_45 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AssertSame.ref.loc12: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc12_14: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc12_19: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %Type.specific_fn.loc12_14: <specific function> = specific_function %Type.ref.loc12_14, @Type(constants.%I.type.733) [concrete = constants.%Type.specific_fn.605]
// CHECK:STDOUT:   %.loc12_20.1: ref %WrapType.fca = temporary_storage
// CHECK:STDOUT:   %Type.call.loc12_20: init %WrapType.fca = call %Type.specific_fn.loc12_14() to %.loc12_20.1
// CHECK:STDOUT:   %Type.ref.loc12_23: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc12_28: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.1: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.ref.loc12: %I.type.733 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %I.ref.loc12_48: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.as_type.loc12: type = facet_access_type %.Self.ref.loc12 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc12_36: type = converted %.Self.ref.loc12, %.Self.as_type.loc12 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc12_30: type = where_expr %.Self.1 [concrete = constants.%I.type.733] {
// CHECK:STDOUT:     requirement_impls %.loc12_36, %I.ref.loc12_48
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc12_23: <specific function> = specific_function %Type.ref.loc12_23, @Type(constants.%I.type.733) [concrete = constants.%Type.specific_fn.605]
// CHECK:STDOUT:   %.loc12_49.1: ref %WrapType.fca = temporary_storage
// CHECK:STDOUT:   %Type.call.loc12_49: init %WrapType.fca = call %Type.specific_fn.loc12_23() to %.loc12_49.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc12: <specific function> = specific_function %AssertSame.ref.loc12, @AssertSame(constants.%I.type.733) [concrete = constants.%AssertSame.specific_fn.0ab]
// CHECK:STDOUT:   %.loc12_20.2: ref %WrapType.fca = temporary %.loc12_20.1, %Type.call.loc12_20
// CHECK:STDOUT:   %.loc12_20.3: %WrapType.fca = bind_value %.loc12_20.2
// CHECK:STDOUT:   %.loc12_49.2: ref %WrapType.fca = temporary %.loc12_49.1, %Type.call.loc12_49
// CHECK:STDOUT:   %.loc12_49.3: %WrapType.fca = bind_value %.loc12_49.2
// CHECK:STDOUT:   %AssertSame.call.loc12: init %empty_tuple.type = call %AssertSame.specific_fn.loc12(%.loc12_20.3, %.loc12_49.3)
// CHECK:STDOUT:   %AssertSame.ref.loc13: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc13_14: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc13_19: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %Type.specific_fn.loc13_14: <specific function> = specific_function %Type.ref.loc13_14, @Type(constants.%I.type.733) [concrete = constants.%Type.specific_fn.605]
// CHECK:STDOUT:   %.loc13_20.1: ref %WrapType.fca = temporary_storage
// CHECK:STDOUT:   %Type.call.loc13_20: init %WrapType.fca = call %Type.specific_fn.loc13_14() to %.loc13_20.1
// CHECK:STDOUT:   %Type.ref.loc13_23: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc13_28: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.2: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.ref.loc13: %I.type.733 = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %I.ref.loc13_48: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %I.ref.loc13_52: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %impl.elem0.loc13: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc13_50.1: <bound method> = bound_method %I.ref.loc13_48, %impl.elem0.loc13 [concrete = constants.%Op.bound.86c]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_50.2: <bound method> = bound_method %I.ref.loc13_48, %specific_fn.loc13 [concrete = constants.%bound_method.f92]
// CHECK:STDOUT:   %type.and.loc13: init type = call %bound_method.loc13_50.2(%I.ref.loc13_48, %I.ref.loc13_52) [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.as_type.loc13: type = facet_access_type %.Self.ref.loc13 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc13_36: type = converted %.Self.ref.loc13, %.Self.as_type.loc13 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc13_50.1: type = value_of_initializer %type.and.loc13 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.loc13_50.2: type = converted %type.and.loc13, %.loc13_50.1 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.loc13_30: type = where_expr %.Self.2 [concrete = constants.%I.type.733] {
// CHECK:STDOUT:     requirement_impls %.loc13_36, %.loc13_50.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc13_23: <specific function> = specific_function %Type.ref.loc13_23, @Type(constants.%I.type.733) [concrete = constants.%Type.specific_fn.605]
// CHECK:STDOUT:   %.loc13_53.1: ref %WrapType.fca = temporary_storage
// CHECK:STDOUT:   %Type.call.loc13_53: init %WrapType.fca = call %Type.specific_fn.loc13_23() to %.loc13_53.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc13: <specific function> = specific_function %AssertSame.ref.loc13, @AssertSame(constants.%I.type.733) [concrete = constants.%AssertSame.specific_fn.0ab]
// CHECK:STDOUT:   %.loc13_20.2: ref %WrapType.fca = temporary %.loc13_20.1, %Type.call.loc13_20
// CHECK:STDOUT:   %.loc13_20.3: %WrapType.fca = bind_value %.loc13_20.2
// CHECK:STDOUT:   %.loc13_53.2: ref %WrapType.fca = temporary %.loc13_53.1, %Type.call.loc13_53
// CHECK:STDOUT:   %.loc13_53.3: %WrapType.fca = bind_value %.loc13_53.2
// CHECK:STDOUT:   %AssertSame.call.loc13: init %empty_tuple.type = call %AssertSame.specific_fn.loc13(%.loc13_20.3, %.loc13_53.3)
// CHECK:STDOUT:   %AssertSame.ref.loc14: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc14_14: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc14_19: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %J.ref.loc14_23: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %impl.elem0.loc14_21: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc14_21.1: <bound method> = bound_method %I.ref.loc14_19, %impl.elem0.loc14_21 [concrete = constants.%Op.bound.86c]
// CHECK:STDOUT:   %specific_fn.loc14_21: <specific function> = specific_function %impl.elem0.loc14_21, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_21.2: <bound method> = bound_method %I.ref.loc14_19, %specific_fn.loc14_21 [concrete = constants.%bound_method.f92]
// CHECK:STDOUT:   %type.and.loc14_21: init type = call %bound_method.loc14_21.2(%I.ref.loc14_19, %J.ref.loc14_23) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc14_24.1: type = value_of_initializer %type.and.loc14_21 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc14_24.2: type = converted %type.and.loc14_21, %.loc14_24.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %Type.specific_fn.loc14_14: <specific function> = specific_function %Type.ref.loc14_14, @Type(constants.%facet_type.100) [concrete = constants.%Type.specific_fn.3cd]
// CHECK:STDOUT:   %.loc14_24.3: ref %WrapType.4f4 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc14_24: init %WrapType.4f4 = call %Type.specific_fn.loc14_14() to %.loc14_24.3
// CHECK:STDOUT:   %Type.ref.loc14_27: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %J.ref.loc14_32: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %I.ref.loc14_36: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %J.ref.loc14_32, %impl.elem0.loc14_34 [concrete = constants.%Op.bound.9e8]
// CHECK:STDOUT:   %specific_fn.loc14_34: <specific function> = specific_function %impl.elem0.loc14_34, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %J.ref.loc14_32, %specific_fn.loc14_34 [concrete = constants.%bound_method.739]
// CHECK:STDOUT:   %type.and.loc14_34: init type = call %bound_method.loc14_34.2(%J.ref.loc14_32, %I.ref.loc14_36) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc14_34.1: type = value_of_initializer %type.and.loc14_34 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc14_34.2: type = converted %type.and.loc14_34, %.loc14_34.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.Self.3: %facet_type.100 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %.Self.ref.loc14: %facet_type.100 = name_ref .Self, %.Self.3 [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %I.ref.loc14_56: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.as_type.loc14: type = facet_access_type %.Self.ref.loc14 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc14_44: type = converted %.Self.ref.loc14, %.Self.as_type.loc14 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc14_38: type = where_expr %.Self.3 [concrete = constants.%facet_type.100] {
// CHECK:STDOUT:     requirement_impls %.loc14_44, %I.ref.loc14_56
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc14_27: <specific function> = specific_function %Type.ref.loc14_27, @Type(constants.%facet_type.100) [concrete = constants.%Type.specific_fn.3cd]
// CHECK:STDOUT:   %.loc14_57.1: ref %WrapType.4f4 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc14_57: init %WrapType.4f4 = call %Type.specific_fn.loc14_27() to %.loc14_57.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc14: <specific function> = specific_function %AssertSame.ref.loc14, @AssertSame(constants.%facet_type.100) [concrete = constants.%AssertSame.specific_fn.bb9]
// CHECK:STDOUT:   %.loc14_24.4: ref %WrapType.4f4 = temporary %.loc14_24.3, %Type.call.loc14_24
// CHECK:STDOUT:   %.loc14_24.5: %WrapType.4f4 = bind_value %.loc14_24.4
// CHECK:STDOUT:   %.loc14_57.2: ref %WrapType.4f4 = temporary %.loc14_57.1, %Type.call.loc14_57
// CHECK:STDOUT:   %.loc14_57.3: %WrapType.4f4 = bind_value %.loc14_57.2
// CHECK:STDOUT:   %AssertSame.call.loc14: init %empty_tuple.type = call %AssertSame.specific_fn.loc14(%.loc14_24.5, %.loc14_57.3)
// CHECK:STDOUT:   %AssertSame.ref.loc15: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc15_14: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc15_19: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %J.ref.loc15_23: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %impl.elem0.loc15_21: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc15_21.1: <bound method> = bound_method %I.ref.loc15_19, %impl.elem0.loc15_21 [concrete = constants.%Op.bound.86c]
// CHECK:STDOUT:   %specific_fn.loc15_21: <specific function> = specific_function %impl.elem0.loc15_21, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_21.2: <bound method> = bound_method %I.ref.loc15_19, %specific_fn.loc15_21 [concrete = constants.%bound_method.f92]
// CHECK:STDOUT:   %type.and.loc15_21: init type = call %bound_method.loc15_21.2(%I.ref.loc15_19, %J.ref.loc15_23) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc15_24.1: type = value_of_initializer %type.and.loc15_21 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc15_24.2: type = converted %type.and.loc15_21, %.loc15_24.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %Type.specific_fn.loc15_14: <specific function> = specific_function %Type.ref.loc15_14, @Type(constants.%facet_type.100) [concrete = constants.%Type.specific_fn.3cd]
// CHECK:STDOUT:   %.loc15_24.3: ref %WrapType.4f4 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc15_24: init %WrapType.4f4 = call %Type.specific_fn.loc15_14() to %.loc15_24.3
// CHECK:STDOUT:   %Type.ref.loc15_27: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %J.ref.loc15_32: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %I.ref.loc15_36: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %impl.elem0.loc15_34: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc15_34.1: <bound method> = bound_method %J.ref.loc15_32, %impl.elem0.loc15_34 [concrete = constants.%Op.bound.9e8]
// CHECK:STDOUT:   %specific_fn.loc15_34: <specific function> = specific_function %impl.elem0.loc15_34, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_34.2: <bound method> = bound_method %J.ref.loc15_32, %specific_fn.loc15_34 [concrete = constants.%bound_method.739]
// CHECK:STDOUT:   %type.and.loc15_34: init type = call %bound_method.loc15_34.2(%J.ref.loc15_32, %I.ref.loc15_36) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc15_34.1: type = value_of_initializer %type.and.loc15_34 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc15_34.2: type = converted %type.and.loc15_34, %.loc15_34.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.Self.4: %facet_type.100 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %.Self.ref.loc15: %facet_type.100 = name_ref .Self, %.Self.4 [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %J.ref.loc15_56: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %.Self.as_type.loc15: type = facet_access_type %.Self.ref.loc15 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc15_44: type = converted %.Self.ref.loc15, %.Self.as_type.loc15 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc15_38: type = where_expr %.Self.4 [concrete = constants.%facet_type.100] {
// CHECK:STDOUT:     requirement_impls %.loc15_44, %J.ref.loc15_56
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc15_27: <specific function> = specific_function %Type.ref.loc15_27, @Type(constants.%facet_type.100) [concrete = constants.%Type.specific_fn.3cd]
// CHECK:STDOUT:   %.loc15_57.1: ref %WrapType.4f4 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc15_57: init %WrapType.4f4 = call %Type.specific_fn.loc15_27() to %.loc15_57.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc15: <specific function> = specific_function %AssertSame.ref.loc15, @AssertSame(constants.%facet_type.100) [concrete = constants.%AssertSame.specific_fn.bb9]
// CHECK:STDOUT:   %.loc15_24.4: ref %WrapType.4f4 = temporary %.loc15_24.3, %Type.call.loc15_24
// CHECK:STDOUT:   %.loc15_24.5: %WrapType.4f4 = bind_value %.loc15_24.4
// CHECK:STDOUT:   %.loc15_57.2: ref %WrapType.4f4 = temporary %.loc15_57.1, %Type.call.loc15_57
// CHECK:STDOUT:   %.loc15_57.3: %WrapType.4f4 = bind_value %.loc15_57.2
// CHECK:STDOUT:   %AssertSame.call.loc15: init %empty_tuple.type = call %AssertSame.specific_fn.loc15(%.loc15_24.5, %.loc15_57.3)
// CHECK:STDOUT:   %AssertSame.ref.loc16: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc16_14: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc16_19: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %J.ref.loc16_23: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %impl.elem0.loc16_21: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc16_21.1: <bound method> = bound_method %I.ref.loc16_19, %impl.elem0.loc16_21 [concrete = constants.%Op.bound.86c]
// CHECK:STDOUT:   %specific_fn.loc16_21: <specific function> = specific_function %impl.elem0.loc16_21, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_21.2: <bound method> = bound_method %I.ref.loc16_19, %specific_fn.loc16_21 [concrete = constants.%bound_method.f92]
// CHECK:STDOUT:   %type.and.loc16_21: init type = call %bound_method.loc16_21.2(%I.ref.loc16_19, %J.ref.loc16_23) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc16_24.1: type = value_of_initializer %type.and.loc16_21 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc16_24.2: type = converted %type.and.loc16_21, %.loc16_24.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %Type.specific_fn.loc16_14: <specific function> = specific_function %Type.ref.loc16_14, @Type(constants.%facet_type.100) [concrete = constants.%Type.specific_fn.3cd]
// CHECK:STDOUT:   %.loc16_24.3: ref %WrapType.4f4 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc16_24: init %WrapType.4f4 = call %Type.specific_fn.loc16_14() to %.loc16_24.3
// CHECK:STDOUT:   %Type.ref.loc16_27: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %J.ref.loc16_32: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %I.ref.loc16_36: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %impl.elem0.loc16_34: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc16_34.1: <bound method> = bound_method %J.ref.loc16_32, %impl.elem0.loc16_34 [concrete = constants.%Op.bound.9e8]
// CHECK:STDOUT:   %specific_fn.loc16_34: <specific function> = specific_function %impl.elem0.loc16_34, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_34.2: <bound method> = bound_method %J.ref.loc16_32, %specific_fn.loc16_34 [concrete = constants.%bound_method.739]
// CHECK:STDOUT:   %type.and.loc16_34: init type = call %bound_method.loc16_34.2(%J.ref.loc16_32, %I.ref.loc16_36) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc16_34.1: type = value_of_initializer %type.and.loc16_34 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc16_34.2: type = converted %type.and.loc16_34, %.loc16_34.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.Self.5: %facet_type.100 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %.Self.ref.loc16: %facet_type.100 = name_ref .Self, %.Self.5 [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %I.ref.loc16_56: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %J.ref.loc16_60: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %impl.elem0.loc16_58: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc16_58.1: <bound method> = bound_method %I.ref.loc16_56, %impl.elem0.loc16_58 [concrete = constants.%Op.bound.86c]
// CHECK:STDOUT:   %specific_fn.loc16_58: <specific function> = specific_function %impl.elem0.loc16_58, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_58.2: <bound method> = bound_method %I.ref.loc16_56, %specific_fn.loc16_58 [concrete = constants.%bound_method.f92]
// CHECK:STDOUT:   %type.and.loc16_58: init type = call %bound_method.loc16_58.2(%I.ref.loc16_56, %J.ref.loc16_60) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.Self.as_type.loc16: type = facet_access_type %.Self.ref.loc16 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc16_44: type = converted %.Self.ref.loc16, %.Self.as_type.loc16 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc16_58.1: type = value_of_initializer %type.and.loc16_58 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc16_58.2: type = converted %type.and.loc16_58, %.loc16_58.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc16_38: type = where_expr %.Self.5 [concrete = constants.%facet_type.100] {
// CHECK:STDOUT:     requirement_impls %.loc16_44, %.loc16_58.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc16_27: <specific function> = specific_function %Type.ref.loc16_27, @Type(constants.%facet_type.100) [concrete = constants.%Type.specific_fn.3cd]
// CHECK:STDOUT:   %.loc16_61.1: ref %WrapType.4f4 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc16_61: init %WrapType.4f4 = call %Type.specific_fn.loc16_27() to %.loc16_61.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc16: <specific function> = specific_function %AssertSame.ref.loc16, @AssertSame(constants.%facet_type.100) [concrete = constants.%AssertSame.specific_fn.bb9]
// CHECK:STDOUT:   %.loc16_24.4: ref %WrapType.4f4 = temporary %.loc16_24.3, %Type.call.loc16_24
// CHECK:STDOUT:   %.loc16_24.5: %WrapType.4f4 = bind_value %.loc16_24.4
// CHECK:STDOUT:   %.loc16_61.2: ref %WrapType.4f4 = temporary %.loc16_61.1, %Type.call.loc16_61
// CHECK:STDOUT:   %.loc16_61.3: %WrapType.4f4 = bind_value %.loc16_61.2
// CHECK:STDOUT:   %AssertSame.call.loc16: init %empty_tuple.type = call %AssertSame.specific_fn.loc16(%.loc16_24.5, %.loc16_61.3)
// CHECK:STDOUT:   %AssertSame.ref.loc17: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc17_14: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc17_19: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %J.ref.loc17_23: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %impl.elem0.loc17_21: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc17_21.1: <bound method> = bound_method %I.ref.loc17_19, %impl.elem0.loc17_21 [concrete = constants.%Op.bound.86c]
// CHECK:STDOUT:   %specific_fn.loc17_21: <specific function> = specific_function %impl.elem0.loc17_21, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_21.2: <bound method> = bound_method %I.ref.loc17_19, %specific_fn.loc17_21 [concrete = constants.%bound_method.f92]
// CHECK:STDOUT:   %type.and.loc17_21: init type = call %bound_method.loc17_21.2(%I.ref.loc17_19, %J.ref.loc17_23) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc17_21.1: type = value_of_initializer %type.and.loc17_21 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc17_21.2: type = converted %type.and.loc17_21, %.loc17_21.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.Self.6: %facet_type.100 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %.Self.ref.loc17_31: %facet_type.100 = name_ref .Self, %.Self.6 [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %K.ref.loc17_43: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %.Self.as_type.loc17_31: type = facet_access_type %.Self.ref.loc17_31 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc17_31: type = converted %.Self.ref.loc17_31, %.Self.as_type.loc17_31 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc17_25: type = where_expr %.Self.6 [concrete = constants.%facet_type.6ed91a.1] {
// CHECK:STDOUT:     requirement_impls %.loc17_31, %K.ref.loc17_43
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc17_14: <specific function> = specific_function %Type.ref.loc17_14, @Type(constants.%facet_type.6ed91a.1) [concrete = constants.%Type.specific_fn.bf4e16.1]
// CHECK:STDOUT:   %.loc17_44.1: ref %WrapType.46a0db.1 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc17_44: init %WrapType.46a0db.1 = call %Type.specific_fn.loc17_14() to %.loc17_44.1
// CHECK:STDOUT:   %Type.ref.loc17_47: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %J.ref.loc17_52: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %I.ref.loc17_56: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %impl.elem0.loc17_54: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc17_54.1: <bound method> = bound_method %J.ref.loc17_52, %impl.elem0.loc17_54 [concrete = constants.%Op.bound.9e8]
// CHECK:STDOUT:   %specific_fn.loc17_54: <specific function> = specific_function %impl.elem0.loc17_54, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_54.2: <bound method> = bound_method %J.ref.loc17_52, %specific_fn.loc17_54 [concrete = constants.%bound_method.739]
// CHECK:STDOUT:   %type.and.loc17_54: init type = call %bound_method.loc17_54.2(%J.ref.loc17_52, %I.ref.loc17_56) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc17_54.1: type = value_of_initializer %type.and.loc17_54 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc17_54.2: type = converted %type.and.loc17_54, %.loc17_54.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.Self.7: %facet_type.100 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %.Self.ref.loc17_64: %facet_type.100 = name_ref .Self, %.Self.7 [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %I.ref.loc17_76: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %K.ref.loc17_80: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %impl.elem0.loc17_78: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc17_78.1: <bound method> = bound_method %I.ref.loc17_76, %impl.elem0.loc17_78 [concrete = constants.%Op.bound.86c]
// CHECK:STDOUT:   %specific_fn.loc17_78: <specific function> = specific_function %impl.elem0.loc17_78, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_78.2: <bound method> = bound_method %I.ref.loc17_76, %specific_fn.loc17_78 [concrete = constants.%bound_method.f92]
// CHECK:STDOUT:   %type.and.loc17_78: init type = call %bound_method.loc17_78.2(%I.ref.loc17_76, %K.ref.loc17_80) [concrete = constants.%facet_type.a90]
// CHECK:STDOUT:   %.Self.as_type.loc17_64: type = facet_access_type %.Self.ref.loc17_64 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc17_64: type = converted %.Self.ref.loc17_64, %.Self.as_type.loc17_64 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc17_78.1: type = value_of_initializer %type.and.loc17_78 [concrete = constants.%facet_type.a90]
// CHECK:STDOUT:   %.loc17_78.2: type = converted %type.and.loc17_78, %.loc17_78.1 [concrete = constants.%facet_type.a90]
// CHECK:STDOUT:   %.loc17_58: type = where_expr %.Self.7 [concrete = constants.%facet_type.6ed91a.1] {
// CHECK:STDOUT:     requirement_impls %.loc17_64, %.loc17_78.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc17_47: <specific function> = specific_function %Type.ref.loc17_47, @Type(constants.%facet_type.6ed91a.1) [concrete = constants.%Type.specific_fn.bf4e16.1]
// CHECK:STDOUT:   %.loc17_81.1: ref %WrapType.46a0db.1 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc17_81: init %WrapType.46a0db.1 = call %Type.specific_fn.loc17_47() to %.loc17_81.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc17: <specific function> = specific_function %AssertSame.ref.loc17, @AssertSame(constants.%facet_type.6ed91a.1) [concrete = constants.%AssertSame.specific_fn.b64196.1]
// CHECK:STDOUT:   %.loc17_44.2: ref %WrapType.46a0db.1 = temporary %.loc17_44.1, %Type.call.loc17_44
// CHECK:STDOUT:   %.loc17_44.3: %WrapType.46a0db.1 = bind_value %.loc17_44.2
// CHECK:STDOUT:   %.loc17_81.2: ref %WrapType.46a0db.1 = temporary %.loc17_81.1, %Type.call.loc17_81
// CHECK:STDOUT:   %.loc17_81.3: %WrapType.46a0db.1 = bind_value %.loc17_81.2
// CHECK:STDOUT:   %AssertSame.call.loc17: init %empty_tuple.type = call %AssertSame.specific_fn.loc17(%.loc17_44.3, %.loc17_81.3)
// CHECK:STDOUT:   %AssertSame.ref.loc18: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc18_14: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc18_19: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %J.ref.loc18_23: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %impl.elem0.loc18_21: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc18_21.1: <bound method> = bound_method %I.ref.loc18_19, %impl.elem0.loc18_21 [concrete = constants.%Op.bound.86c]
// CHECK:STDOUT:   %specific_fn.loc18_21: <specific function> = specific_function %impl.elem0.loc18_21, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_21.2: <bound method> = bound_method %I.ref.loc18_19, %specific_fn.loc18_21 [concrete = constants.%bound_method.f92]
// CHECK:STDOUT:   %type.and.loc18_21: init type = call %bound_method.loc18_21.2(%I.ref.loc18_19, %J.ref.loc18_23) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc18_21.1: type = value_of_initializer %type.and.loc18_21 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc18_21.2: type = converted %type.and.loc18_21, %.loc18_21.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.Self.8: %facet_type.100 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %.Self.ref.loc18_31: %facet_type.100 = name_ref .Self, %.Self.8 [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %K.ref.loc18_43: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %.Self.as_type.loc18_31: type = facet_access_type %.Self.ref.loc18_31 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc18_31: type = converted %.Self.ref.loc18_31, %.Self.as_type.loc18_31 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc18_25: type = where_expr %.Self.8 [concrete = constants.%facet_type.6ed91a.1] {
// CHECK:STDOUT:     requirement_impls %.loc18_31, %K.ref.loc18_43
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc18_14: <specific function> = specific_function %Type.ref.loc18_14, @Type(constants.%facet_type.6ed91a.1) [concrete = constants.%Type.specific_fn.bf4e16.1]
// CHECK:STDOUT:   %.loc18_44.1: ref %WrapType.46a0db.1 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc18_44: init %WrapType.46a0db.1 = call %Type.specific_fn.loc18_14() to %.loc18_44.1
// CHECK:STDOUT:   %Type.ref.loc18_47: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %J.ref.loc18_52: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %I.ref.loc18_56: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %impl.elem0.loc18_54: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc18_54.1: <bound method> = bound_method %J.ref.loc18_52, %impl.elem0.loc18_54 [concrete = constants.%Op.bound.9e8]
// CHECK:STDOUT:   %specific_fn.loc18_54: <specific function> = specific_function %impl.elem0.loc18_54, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_54.2: <bound method> = bound_method %J.ref.loc18_52, %specific_fn.loc18_54 [concrete = constants.%bound_method.739]
// CHECK:STDOUT:   %type.and.loc18_54: init type = call %bound_method.loc18_54.2(%J.ref.loc18_52, %I.ref.loc18_56) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc18_54.1: type = value_of_initializer %type.and.loc18_54 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc18_54.2: type = converted %type.and.loc18_54, %.loc18_54.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.Self.9: %facet_type.100 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %.Self.ref.loc18_64: %facet_type.100 = name_ref .Self, %.Self.9 [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %J.ref.loc18_76: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %K.ref.loc18_80: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %impl.elem0.loc18_78: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc18_78.1: <bound method> = bound_method %J.ref.loc18_76, %impl.elem0.loc18_78 [concrete = constants.%Op.bound.9e8]
// CHECK:STDOUT:   %specific_fn.loc18_78: <specific function> = specific_function %impl.elem0.loc18_78, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_78.2: <bound method> = bound_method %J.ref.loc18_76, %specific_fn.loc18_78 [concrete = constants.%bound_method.739]
// CHECK:STDOUT:   %type.and.loc18_78: init type = call %bound_method.loc18_78.2(%J.ref.loc18_76, %K.ref.loc18_80) [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.Self.as_type.loc18_64: type = facet_access_type %.Self.ref.loc18_64 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc18_64: type = converted %.Self.ref.loc18_64, %.Self.as_type.loc18_64 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc18_78.1: type = value_of_initializer %type.and.loc18_78 [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.loc18_78.2: type = converted %type.and.loc18_78, %.loc18_78.1 [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.loc18_58: type = where_expr %.Self.9 [concrete = constants.%facet_type.6ed91a.1] {
// CHECK:STDOUT:     requirement_impls %.loc18_64, %.loc18_78.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc18_47: <specific function> = specific_function %Type.ref.loc18_47, @Type(constants.%facet_type.6ed91a.1) [concrete = constants.%Type.specific_fn.bf4e16.1]
// CHECK:STDOUT:   %.loc18_81.1: ref %WrapType.46a0db.1 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc18_81: init %WrapType.46a0db.1 = call %Type.specific_fn.loc18_47() to %.loc18_81.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc18: <specific function> = specific_function %AssertSame.ref.loc18, @AssertSame(constants.%facet_type.6ed91a.1) [concrete = constants.%AssertSame.specific_fn.b64196.1]
// CHECK:STDOUT:   %.loc18_44.2: ref %WrapType.46a0db.1 = temporary %.loc18_44.1, %Type.call.loc18_44
// CHECK:STDOUT:   %.loc18_44.3: %WrapType.46a0db.1 = bind_value %.loc18_44.2
// CHECK:STDOUT:   %.loc18_81.2: ref %WrapType.46a0db.1 = temporary %.loc18_81.1, %Type.call.loc18_81
// CHECK:STDOUT:   %.loc18_81.3: %WrapType.46a0db.1 = bind_value %.loc18_81.2
// CHECK:STDOUT:   %AssertSame.call.loc18: init %empty_tuple.type = call %AssertSame.specific_fn.loc18(%.loc18_44.3, %.loc18_81.3)
// CHECK:STDOUT:   %AssertSame.ref.loc19: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc19_14: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc19_19: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %J.ref.loc19_23: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %impl.elem0.loc19_21: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc19_21.1: <bound method> = bound_method %I.ref.loc19_19, %impl.elem0.loc19_21 [concrete = constants.%Op.bound.86c]
// CHECK:STDOUT:   %specific_fn.loc19_21: <specific function> = specific_function %impl.elem0.loc19_21, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_21.2: <bound method> = bound_method %I.ref.loc19_19, %specific_fn.loc19_21 [concrete = constants.%bound_method.f92]
// CHECK:STDOUT:   %type.and.loc19_21: init type = call %bound_method.loc19_21.2(%I.ref.loc19_19, %J.ref.loc19_23) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc19_21.1: type = value_of_initializer %type.and.loc19_21 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc19_21.2: type = converted %type.and.loc19_21, %.loc19_21.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.Self.10: %facet_type.100 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %.Self.ref.loc19_31: %facet_type.100 = name_ref .Self, %.Self.10 [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %K.ref.loc19_43: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %.Self.as_type.loc19_31: type = facet_access_type %.Self.ref.loc19_31 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc19_31: type = converted %.Self.ref.loc19_31, %.Self.as_type.loc19_31 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc19_25: type = where_expr %.Self.10 [concrete = constants.%facet_type.6ed91a.1] {
// CHECK:STDOUT:     requirement_impls %.loc19_31, %K.ref.loc19_43
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc19_14: <specific function> = specific_function %Type.ref.loc19_14, @Type(constants.%facet_type.6ed91a.1) [concrete = constants.%Type.specific_fn.bf4e16.1]
// CHECK:STDOUT:   %.loc19_44.1: ref %WrapType.46a0db.1 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc19_44: init %WrapType.46a0db.1 = call %Type.specific_fn.loc19_14() to %.loc19_44.1
// CHECK:STDOUT:   %Type.ref.loc19_47: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %J.ref.loc19_52: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %I.ref.loc19_56: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %impl.elem0.loc19_54: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc19_54.1: <bound method> = bound_method %J.ref.loc19_52, %impl.elem0.loc19_54 [concrete = constants.%Op.bound.9e8]
// CHECK:STDOUT:   %specific_fn.loc19_54: <specific function> = specific_function %impl.elem0.loc19_54, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_54.2: <bound method> = bound_method %J.ref.loc19_52, %specific_fn.loc19_54 [concrete = constants.%bound_method.739]
// CHECK:STDOUT:   %type.and.loc19_54: init type = call %bound_method.loc19_54.2(%J.ref.loc19_52, %I.ref.loc19_56) [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc19_54.1: type = value_of_initializer %type.and.loc19_54 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.loc19_54.2: type = converted %type.and.loc19_54, %.loc19_54.1 [concrete = constants.%facet_type.100]
// CHECK:STDOUT:   %.Self.11: %facet_type.100 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %.Self.ref.loc19_64: %facet_type.100 = name_ref .Self, %.Self.11 [symbolic_self = constants.%.Self.d59]
// CHECK:STDOUT:   %K.ref.loc19_76: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %I.ref.loc19_80: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %impl.elem0.loc19_78: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc19_78.1: <bound method> = bound_method %K.ref.loc19_76, %impl.elem0.loc19_78 [concrete = constants.%Op.bound.6d4]
// CHECK:STDOUT:   %specific_fn.loc19_78: <specific function> = specific_function %impl.elem0.loc19_78, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_78.2: <bound method> = bound_method %K.ref.loc19_76, %specific_fn.loc19_78 [concrete = constants.%bound_method.2ce]
// CHECK:STDOUT:   %type.and.loc19_78: init type = call %bound_method.loc19_78.2(%K.ref.loc19_76, %I.ref.loc19_80) [concrete = constants.%facet_type.a90]
// CHECK:STDOUT:   %J.ref.loc19_84: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %impl.elem0.loc19_82: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc19_82.1: <bound method> = bound_method %type.and.loc19_78, %impl.elem0.loc19_82 [concrete = constants.%Op.bound.fe2]
// CHECK:STDOUT:   %specific_fn.loc19_82: <specific function> = specific_function %impl.elem0.loc19_82, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_82.2: <bound method> = bound_method %type.and.loc19_78, %specific_fn.loc19_82 [concrete = constants.%bound_method.0d5]
// CHECK:STDOUT:   %.loc19_78.1: type = value_of_initializer %type.and.loc19_78 [concrete = constants.%facet_type.a90]
// CHECK:STDOUT:   %.loc19_78.2: type = converted %type.and.loc19_78, %.loc19_78.1 [concrete = constants.%facet_type.a90]
// CHECK:STDOUT:   %type.and.loc19_82: init type = call %bound_method.loc19_82.2(%.loc19_78.2, %J.ref.loc19_84) [concrete = constants.%facet_type.6ed91a.2]
// CHECK:STDOUT:   %.Self.as_type.loc19_64: type = facet_access_type %.Self.ref.loc19_64 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc19_64: type = converted %.Self.ref.loc19_64, %.Self.as_type.loc19_64 [symbolic_self = constants.%.Self.as_type.330]
// CHECK:STDOUT:   %.loc19_82.1: type = value_of_initializer %type.and.loc19_82 [concrete = constants.%facet_type.6ed91a.2]
// CHECK:STDOUT:   %.loc19_82.2: type = converted %type.and.loc19_82, %.loc19_82.1 [concrete = constants.%facet_type.6ed91a.2]
// CHECK:STDOUT:   %.loc19_58: type = where_expr %.Self.11 [concrete = constants.%facet_type.6ed91a.1] {
// CHECK:STDOUT:     requirement_impls %.loc19_64, %.loc19_82.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc19_47: <specific function> = specific_function %Type.ref.loc19_47, @Type(constants.%facet_type.6ed91a.1) [concrete = constants.%Type.specific_fn.bf4e16.1]
// CHECK:STDOUT:   %.loc19_85.1: ref %WrapType.46a0db.1 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc19_85: init %WrapType.46a0db.1 = call %Type.specific_fn.loc19_47() to %.loc19_85.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc19: <specific function> = specific_function %AssertSame.ref.loc19, @AssertSame(constants.%facet_type.6ed91a.1) [concrete = constants.%AssertSame.specific_fn.b64196.1]
// CHECK:STDOUT:   %.loc19_44.2: ref %WrapType.46a0db.1 = temporary %.loc19_44.1, %Type.call.loc19_44
// CHECK:STDOUT:   %.loc19_44.3: %WrapType.46a0db.1 = bind_value %.loc19_44.2
// CHECK:STDOUT:   %.loc19_85.2: ref %WrapType.46a0db.1 = temporary %.loc19_85.1, %Type.call.loc19_85
// CHECK:STDOUT:   %.loc19_85.3: %WrapType.46a0db.1 = bind_value %.loc19_85.2
// CHECK:STDOUT:   %AssertSame.call.loc19: init %empty_tuple.type = call %AssertSame.specific_fn.loc19(%.loc19_44.3, %.loc19_85.3)
// CHECK:STDOUT:   %AssertSame.ref.loc20: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc20_14: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc20_19: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.12: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.ref.loc20_27: %I.type.733 = name_ref .Self, %.Self.12 [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %J.ref.loc20_39: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %K.ref.loc20_43: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %impl.elem0.loc20_41: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc20_41.1: <bound method> = bound_method %J.ref.loc20_39, %impl.elem0.loc20_41 [concrete = constants.%Op.bound.9e8]
// CHECK:STDOUT:   %specific_fn.loc20_41: <specific function> = specific_function %impl.elem0.loc20_41, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_41.2: <bound method> = bound_method %J.ref.loc20_39, %specific_fn.loc20_41 [concrete = constants.%bound_method.739]
// CHECK:STDOUT:   %type.and.loc20_41: init type = call %bound_method.loc20_41.2(%J.ref.loc20_39, %K.ref.loc20_43) [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.Self.as_type.loc20_27: type = facet_access_type %.Self.ref.loc20_27 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc20_27: type = converted %.Self.ref.loc20_27, %.Self.as_type.loc20_27 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc20_41.1: type = value_of_initializer %type.and.loc20_41 [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.loc20_41.2: type = converted %type.and.loc20_41, %.loc20_41.1 [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.loc20_21: type = where_expr %.Self.12 [concrete = constants.%I.type.6ed] {
// CHECK:STDOUT:     requirement_impls %.loc20_27, %.loc20_41.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc20_14: <specific function> = specific_function %Type.ref.loc20_14, @Type(constants.%I.type.6ed) [concrete = constants.%Type.specific_fn.bf4e16.2]
// CHECK:STDOUT:   %.loc20_44.1: ref %WrapType.46a0db.2 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc20_44: init %WrapType.46a0db.2 = call %Type.specific_fn.loc20_14() to %.loc20_44.1
// CHECK:STDOUT:   %Type.ref.loc20_47: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc20_52: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.13: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.ref.loc20_60: %I.type.733 = name_ref .Self, %.Self.13 [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %K.ref.loc20_72: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %I.ref.loc20_76: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %impl.elem0.loc20_74: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc20_74.1: <bound method> = bound_method %K.ref.loc20_72, %impl.elem0.loc20_74 [concrete = constants.%Op.bound.6d4]
// CHECK:STDOUT:   %specific_fn.loc20_74: <specific function> = specific_function %impl.elem0.loc20_74, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_74.2: <bound method> = bound_method %K.ref.loc20_72, %specific_fn.loc20_74 [concrete = constants.%bound_method.2ce]
// CHECK:STDOUT:   %type.and.loc20_74: init type = call %bound_method.loc20_74.2(%K.ref.loc20_72, %I.ref.loc20_76) [concrete = constants.%facet_type.a90]
// CHECK:STDOUT:   %J.ref.loc20_80: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %impl.elem0.loc20_78: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc20_78.1: <bound method> = bound_method %type.and.loc20_74, %impl.elem0.loc20_78 [concrete = constants.%Op.bound.fe2]
// CHECK:STDOUT:   %specific_fn.loc20_78: <specific function> = specific_function %impl.elem0.loc20_78, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_78.2: <bound method> = bound_method %type.and.loc20_74, %specific_fn.loc20_78 [concrete = constants.%bound_method.0d5]
// CHECK:STDOUT:   %.loc20_74.1: type = value_of_initializer %type.and.loc20_74 [concrete = constants.%facet_type.a90]
// CHECK:STDOUT:   %.loc20_74.2: type = converted %type.and.loc20_74, %.loc20_74.1 [concrete = constants.%facet_type.a90]
// CHECK:STDOUT:   %type.and.loc20_78: init type = call %bound_method.loc20_78.2(%.loc20_74.2, %J.ref.loc20_80) [concrete = constants.%facet_type.6ed91a.2]
// CHECK:STDOUT:   %.Self.as_type.loc20_60: type = facet_access_type %.Self.ref.loc20_60 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc20_60: type = converted %.Self.ref.loc20_60, %.Self.as_type.loc20_60 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc20_78.1: type = value_of_initializer %type.and.loc20_78 [concrete = constants.%facet_type.6ed91a.2]
// CHECK:STDOUT:   %.loc20_78.2: type = converted %type.and.loc20_78, %.loc20_78.1 [concrete = constants.%facet_type.6ed91a.2]
// CHECK:STDOUT:   %.loc20_54: type = where_expr %.Self.13 [concrete = constants.%I.type.6ed] {
// CHECK:STDOUT:     requirement_impls %.loc20_60, %.loc20_78.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc20_47: <specific function> = specific_function %Type.ref.loc20_47, @Type(constants.%I.type.6ed) [concrete = constants.%Type.specific_fn.bf4e16.2]
// CHECK:STDOUT:   %.loc20_81.1: ref %WrapType.46a0db.2 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc20_81: init %WrapType.46a0db.2 = call %Type.specific_fn.loc20_47() to %.loc20_81.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc20: <specific function> = specific_function %AssertSame.ref.loc20, @AssertSame(constants.%I.type.6ed) [concrete = constants.%AssertSame.specific_fn.b64196.2]
// CHECK:STDOUT:   %.loc20_44.2: ref %WrapType.46a0db.2 = temporary %.loc20_44.1, %Type.call.loc20_44
// CHECK:STDOUT:   %.loc20_44.3: %WrapType.46a0db.2 = bind_value %.loc20_44.2
// CHECK:STDOUT:   %.loc20_81.2: ref %WrapType.46a0db.2 = temporary %.loc20_81.1, %Type.call.loc20_81
// CHECK:STDOUT:   %.loc20_81.3: %WrapType.46a0db.2 = bind_value %.loc20_81.2
// CHECK:STDOUT:   %AssertSame.call.loc20: init %empty_tuple.type = call %AssertSame.specific_fn.loc20(%.loc20_44.3, %.loc20_81.3)
// CHECK:STDOUT:   %AssertSame.ref.loc21: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc21: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc21: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.14: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.ref.loc21: %I.type.733 = name_ref .Self, %.Self.14 [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %J.ref.loc21: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %K.ref.loc21: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %impl.elem0.loc21: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc21_41.1: <bound method> = bound_method %J.ref.loc21, %impl.elem0.loc21 [concrete = constants.%Op.bound.9e8]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_41.2: <bound method> = bound_method %J.ref.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.739]
// CHECK:STDOUT:   %type.and.loc21: init type = call %bound_method.loc21_41.2(%J.ref.loc21, %K.ref.loc21) [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.Self.as_type.loc21: type = facet_access_type %.Self.ref.loc21 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc21_27: type = converted %.Self.ref.loc21, %.Self.as_type.loc21 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc21_41.1: type = value_of_initializer %type.and.loc21 [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.loc21_41.2: type = converted %type.and.loc21, %.loc21_41.1 [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.loc21_21: type = where_expr %.Self.14 [concrete = constants.%I.type.6ed] {
// CHECK:STDOUT:     requirement_impls %.loc21_27, %.loc21_41.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc21: <specific function> = specific_function %Type.ref.loc21, @Type(constants.%I.type.6ed) [concrete = constants.%Type.specific_fn.bf4e16.2]
// CHECK:STDOUT:   %.loc21_44.1: ref %WrapType.46a0db.2 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc21: init %WrapType.46a0db.2 = call %Type.specific_fn.loc21() to %.loc21_44.1
// CHECK:STDOUT:   %Type.ref.loc22: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc22: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.15: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.ref.loc22_27: %I.type.733 = name_ref .Self, %.Self.15 [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %J.ref.loc22: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %.Self.16: %J.type.eee = bind_symbolic_name .Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:   %.Self.ref.loc22_48: %J.type.eee = name_ref .Self, %.Self.16 [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:   %K.ref.loc22: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %.Self.as_type.loc22_48: type = facet_access_type %.Self.ref.loc22_48 [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:   %.loc22_48: type = converted %.Self.ref.loc22_48, %.Self.as_type.loc22_48 [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:   %.loc22_42: type = where_expr %.Self.16 [concrete = constants.%J.type.2cb] {
// CHECK:STDOUT:     requirement_impls %.loc22_48, %K.ref.loc22
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.Self.as_type.loc22_27: type = facet_access_type %.Self.ref.loc22_27 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc22_27: type = converted %.Self.ref.loc22_27, %.Self.as_type.loc22_27 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc22_21: type = where_expr %.Self.15 [concrete = constants.%I.type.6ed] {
// CHECK:STDOUT:     requirement_impls %.loc22_27, %.loc22_42
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc22: <specific function> = specific_function %Type.ref.loc22, @Type(constants.%I.type.6ed) [concrete = constants.%Type.specific_fn.bf4e16.2]
// CHECK:STDOUT:   %.loc22_62.1: ref %WrapType.46a0db.2 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc22: init %WrapType.46a0db.2 = call %Type.specific_fn.loc22() to %.loc22_62.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc21: <specific function> = specific_function %AssertSame.ref.loc21, @AssertSame(constants.%I.type.6ed) [concrete = constants.%AssertSame.specific_fn.b64196.2]
// CHECK:STDOUT:   %.loc21_44.2: ref %WrapType.46a0db.2 = temporary %.loc21_44.1, %Type.call.loc21
// CHECK:STDOUT:   %.loc21_44.3: %WrapType.46a0db.2 = bind_value %.loc21_44.2
// CHECK:STDOUT:   %.loc22_62.2: ref %WrapType.46a0db.2 = temporary %.loc22_62.1, %Type.call.loc22
// CHECK:STDOUT:   %.loc22_62.3: %WrapType.46a0db.2 = bind_value %.loc22_62.2
// CHECK:STDOUT:   %AssertSame.call.loc22: init %empty_tuple.type = call %AssertSame.specific_fn.loc21(%.loc21_44.3, %.loc22_62.3)
// CHECK:STDOUT:   %AssertSame.ref.loc23: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc23: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc23: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.17: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.ref.loc23: %I.type.733 = name_ref .Self, %.Self.17 [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %J.ref.loc23: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %K.ref.loc23: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %impl.elem0.loc23: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc23_41.1: <bound method> = bound_method %J.ref.loc23, %impl.elem0.loc23 [concrete = constants.%Op.bound.9e8]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_41.2: <bound method> = bound_method %J.ref.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.739]
// CHECK:STDOUT:   %type.and.loc23: init type = call %bound_method.loc23_41.2(%J.ref.loc23, %K.ref.loc23) [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.Self.as_type.loc23: type = facet_access_type %.Self.ref.loc23 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc23_27: type = converted %.Self.ref.loc23, %.Self.as_type.loc23 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc23_41.1: type = value_of_initializer %type.and.loc23 [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.loc23_41.2: type = converted %type.and.loc23, %.loc23_41.1 [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.loc23_21: type = where_expr %.Self.17 [concrete = constants.%I.type.6ed] {
// CHECK:STDOUT:     requirement_impls %.loc23_27, %.loc23_41.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc23: <specific function> = specific_function %Type.ref.loc23, @Type(constants.%I.type.6ed) [concrete = constants.%Type.specific_fn.bf4e16.2]
// CHECK:STDOUT:   %.loc23_44.1: ref %WrapType.46a0db.2 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc23: init %WrapType.46a0db.2 = call %Type.specific_fn.loc23() to %.loc23_44.1
// CHECK:STDOUT:   %Type.ref.loc24: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc24: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.18: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.ref.loc24_27: %I.type.733 = name_ref .Self, %.Self.18 [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %K.ref.loc24: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %.Self.19: %K.type.693 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.02e]
// CHECK:STDOUT:   %.Self.ref.loc24_48: %K.type.693 = name_ref .Self, %.Self.19 [symbolic_self = constants.%.Self.02e]
// CHECK:STDOUT:   %J.ref.loc24: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %.Self.as_type.loc24_48: type = facet_access_type %.Self.ref.loc24_48 [symbolic_self = constants.%.Self.as_type.b21]
// CHECK:STDOUT:   %.loc24_48: type = converted %.Self.ref.loc24_48, %.Self.as_type.loc24_48 [symbolic_self = constants.%.Self.as_type.b21]
// CHECK:STDOUT:   %.loc24_42: type = where_expr %.Self.19 [concrete = constants.%K.type.c26] {
// CHECK:STDOUT:     requirement_impls %.loc24_48, %J.ref.loc24
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.Self.as_type.loc24_27: type = facet_access_type %.Self.ref.loc24_27 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc24_27: type = converted %.Self.ref.loc24_27, %.Self.as_type.loc24_27 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc24_21: type = where_expr %.Self.18 [concrete = constants.%I.type.6ed] {
// CHECK:STDOUT:     requirement_impls %.loc24_27, %.loc24_42
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc24: <specific function> = specific_function %Type.ref.loc24, @Type(constants.%I.type.6ed) [concrete = constants.%Type.specific_fn.bf4e16.2]
// CHECK:STDOUT:   %.loc24_62.1: ref %WrapType.46a0db.2 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc24: init %WrapType.46a0db.2 = call %Type.specific_fn.loc24() to %.loc24_62.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc23: <specific function> = specific_function %AssertSame.ref.loc23, @AssertSame(constants.%I.type.6ed) [concrete = constants.%AssertSame.specific_fn.b64196.2]
// CHECK:STDOUT:   %.loc23_44.2: ref %WrapType.46a0db.2 = temporary %.loc23_44.1, %Type.call.loc23
// CHECK:STDOUT:   %.loc23_44.3: %WrapType.46a0db.2 = bind_value %.loc23_44.2
// CHECK:STDOUT:   %.loc24_62.2: ref %WrapType.46a0db.2 = temporary %.loc24_62.1, %Type.call.loc24
// CHECK:STDOUT:   %.loc24_62.3: %WrapType.46a0db.2 = bind_value %.loc24_62.2
// CHECK:STDOUT:   %AssertSame.call.loc24: init %empty_tuple.type = call %AssertSame.specific_fn.loc23(%.loc23_44.3, %.loc24_62.3)
// CHECK:STDOUT:   %AssertSame.ref.loc25: %AssertSame.type = name_ref AssertSame, file.%AssertSame.decl [concrete = constants.%AssertSame]
// CHECK:STDOUT:   %Type.ref.loc25: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc25: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.20: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.ref.loc25: %I.type.733 = name_ref .Self, %.Self.20 [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %J.ref.loc25: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %K.ref.loc25: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %impl.elem0.loc25: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc25_41.1: <bound method> = bound_method %J.ref.loc25, %impl.elem0.loc25 [concrete = constants.%Op.bound.9e8]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_41.2: <bound method> = bound_method %J.ref.loc25, %specific_fn.loc25 [concrete = constants.%bound_method.739]
// CHECK:STDOUT:   %type.and.loc25: init type = call %bound_method.loc25_41.2(%J.ref.loc25, %K.ref.loc25) [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.Self.as_type.loc25: type = facet_access_type %.Self.ref.loc25 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc25_27: type = converted %.Self.ref.loc25, %.Self.as_type.loc25 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc25_41.1: type = value_of_initializer %type.and.loc25 [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.loc25_41.2: type = converted %type.and.loc25, %.loc25_41.1 [concrete = constants.%facet_type.2cb99f.1]
// CHECK:STDOUT:   %.loc25_21: type = where_expr %.Self.20 [concrete = constants.%I.type.6ed] {
// CHECK:STDOUT:     requirement_impls %.loc25_27, %.loc25_41.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc25: <specific function> = specific_function %Type.ref.loc25, @Type(constants.%I.type.6ed) [concrete = constants.%Type.specific_fn.bf4e16.2]
// CHECK:STDOUT:   %.loc25_44.1: ref %WrapType.46a0db.2 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc25: init %WrapType.46a0db.2 = call %Type.specific_fn.loc25() to %.loc25_44.1
// CHECK:STDOUT:   %Type.ref.loc26: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc26: type = name_ref I, file.%I.decl.loc7 [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.21: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.ref.loc26_27: %I.type.733 = name_ref .Self, %.Self.21 [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.22: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.644]
// CHECK:STDOUT:   %.Self.ref.loc26_51: type = name_ref .Self, %.Self.22 [symbolic_self = constants.%.Self.644]
// CHECK:STDOUT:   %J.ref.loc26: type = name_ref J, file.%J.decl.loc8 [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %.Self.23: %J.type.eee = bind_symbolic_name .Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:   %.Self.ref.loc26_72: %J.type.eee = name_ref .Self, %.Self.23 [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:   %K.ref.loc26: type = name_ref K, file.%K.decl.loc9 [concrete = constants.%K.type.693]
// CHECK:STDOUT:   %.Self.as_type.loc26_72: type = facet_access_type %.Self.ref.loc26_72 [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:   %.loc26_72: type = converted %.Self.ref.loc26_72, %.Self.as_type.loc26_72 [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:   %.loc26_66: type = where_expr %.Self.23 [concrete = constants.%J.type.2cb] {
// CHECK:STDOUT:     requirement_impls %.loc26_72, %K.ref.loc26
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26_45: type = where_expr %.Self.22 [concrete = constants.%facet_type.2cb99f.2] {
// CHECK:STDOUT:     requirement_impls %.Self.ref.loc26_51, %.loc26_66
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.Self.as_type.loc26_27: type = facet_access_type %.Self.ref.loc26_27 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc26_27: type = converted %.Self.ref.loc26_27, %.Self.as_type.loc26_27 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc26_21: type = where_expr %.Self.21 [concrete = constants.%I.type.6ed] {
// CHECK:STDOUT:     requirement_impls %.loc26_27, %.loc26_45
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc26: <specific function> = specific_function %Type.ref.loc26, @Type(constants.%I.type.6ed) [concrete = constants.%Type.specific_fn.bf4e16.2]
// CHECK:STDOUT:   %.loc26_87.1: ref %WrapType.46a0db.2 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc26: init %WrapType.46a0db.2 = call %Type.specific_fn.loc26() to %.loc26_87.1
// CHECK:STDOUT:   %AssertSame.specific_fn.loc25: <specific function> = specific_function %AssertSame.ref.loc25, @AssertSame(constants.%I.type.6ed) [concrete = constants.%AssertSame.specific_fn.b64196.2]
// CHECK:STDOUT:   %.loc25_44.2: ref %WrapType.46a0db.2 = temporary %.loc25_44.1, %Type.call.loc25
// CHECK:STDOUT:   %.loc25_44.3: %WrapType.46a0db.2 = bind_value %.loc25_44.2
// CHECK:STDOUT:   %.loc26_87.2: ref %WrapType.46a0db.2 = temporary %.loc26_87.1, %Type.call.loc26
// CHECK:STDOUT:   %.loc26_87.3: %WrapType.46a0db.2 = bind_value %.loc26_87.2
// CHECK:STDOUT:   %AssertSame.call.loc26: init %empty_tuple.type = call %AssertSame.specific_fn.loc25(%.loc25_44.3, %.loc26_87.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.040: %BitAnd.type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.25f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type (%Self.as_type)](%other.param_patt: @Op.1.%Self.as_type (%Self.as_type)) -> @Op.1.%Self.as_type (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T)](%other.param_patt: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(constants.%T) {
// CHECK:STDOUT:   %T.loc3_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc3_16.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AssertSame(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc4_38.2 => constants.%WrapType.aa1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(@AssertSame.%T.loc4_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%T) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc5_9.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc5_32.2 => constants.%WrapType.aa1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(@Type.%T.loc5_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%I.type.733) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%I.type.733
// CHECK:STDOUT:   %T.patt.loc5_9.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc5_32.2 => constants.%WrapType.fca
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %WrapType.val => constants.%WrapType.val.26a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(constants.%I.type.733) {
// CHECK:STDOUT:   %T.loc3_16.2 => constants.%I.type.733
// CHECK:STDOUT:   %T.patt.loc3_16.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AssertSame(constants.%I.type.733) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%I.type.733
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc4_38.2 => constants.%WrapType.fca
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.25f) {
// CHECK:STDOUT:   %Self => constants.%Self.25f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.db8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3ea
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.eb8
// CHECK:STDOUT:   %Op => constants.%Op.444
// CHECK:STDOUT:   %require_complete => constants.%complete_type.473
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%facet_type.100) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%facet_type.100
// CHECK:STDOUT:   %T.patt.loc5_9.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc5_32.2 => constants.%WrapType.4f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %WrapType.val => constants.%WrapType.val.77e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(constants.%facet_type.100) {
// CHECK:STDOUT:   %T.loc3_16.2 => constants.%facet_type.100
// CHECK:STDOUT:   %T.patt.loc3_16.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AssertSame(constants.%facet_type.100) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%facet_type.100
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc4_38.2 => constants.%WrapType.4f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%facet_type.6ed91a.1) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%facet_type.6ed91a.1
// CHECK:STDOUT:   %T.patt.loc5_9.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc5_32.2 => constants.%WrapType.46a0db.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %WrapType.val => constants.%WrapType.val.751449.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(constants.%facet_type.6ed91a.1) {
// CHECK:STDOUT:   %T.loc3_16.2 => constants.%facet_type.6ed91a.1
// CHECK:STDOUT:   %T.patt.loc3_16.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AssertSame(constants.%facet_type.6ed91a.1) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%facet_type.6ed91a.1
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc4_38.2 => constants.%WrapType.46a0db.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%I.type.6ed) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%I.type.6ed
// CHECK:STDOUT:   %T.patt.loc5_9.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc5_32.2 => constants.%WrapType.46a0db.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %WrapType.val => constants.%WrapType.val.751449.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(constants.%I.type.6ed) {
// CHECK:STDOUT:   %T.loc3_16.2 => constants.%I.type.6ed
// CHECK:STDOUT:   %T.patt.loc3_16.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AssertSame(constants.%I.type.6ed) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%I.type.6ed
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc4_38.2 => constants.%WrapType.46a0db.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_compare_not_equal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %WrapType.type: type = generic_class_type @WrapType [concrete]
// CHECK:STDOUT:   %WrapType.generic: %WrapType.type = struct_value () [concrete]
// CHECK:STDOUT:   %WrapType.aa1: type = class_type @WrapType, @WrapType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Same.type: type = fn_type @Same [concrete]
// CHECK:STDOUT:   %Same: %Same.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %WrapType.aa1 [symbolic]
// CHECK:STDOUT:   %Type.type: type = fn_type @Type [concrete]
// CHECK:STDOUT:   %Type: %Type.type = struct_value () [concrete]
// CHECK:STDOUT:   %WrapType.val.892: %WrapType.aa1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type.eee: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type.eee = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.258: %I.type.733 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.541: type = facet_access_type %.Self.258 [symbolic_self]
// CHECK:STDOUT:   %I.type.100: type = facet_type <@I where .Self impls @J> [concrete]
// CHECK:STDOUT:   %WrapType.4f4: type = class_type @WrapType, @WrapType(%I.type.100) [concrete]
// CHECK:STDOUT:   %Type.specific_fn.3cd: <specific function> = specific_function %Type, @Type(%I.type.100) [concrete]
// CHECK:STDOUT:   %.Self.968: %J.type.eee = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.78d: type = facet_access_type %.Self.968 [symbolic_self]
// CHECK:STDOUT:   %J.type.c41: type = facet_type <@J where .Self impls @I> [concrete]
// CHECK:STDOUT:   %WrapType.802: type = class_type @WrapType, @WrapType(%J.type.c41) [concrete]
// CHECK:STDOUT:   %Type.specific_fn.6fb: <specific function> = specific_function %Type, @Type(%J.type.c41) [concrete]
// CHECK:STDOUT:   %WrapType.val.77e: %WrapType.4f4 = struct_value () [concrete]
// CHECK:STDOUT:   %WrapType.val.0de: %WrapType.802 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .WrapType = %WrapType.decl
// CHECK:STDOUT:     .Same = %Same.decl
// CHECK:STDOUT:     .Type = %Type.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %WrapType.decl: %WrapType.type = class_decl @WrapType [concrete = constants.%WrapType.generic] {
// CHECK:STDOUT:     %T.patt.loc3_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_16.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Same.decl: %Same.type = fn_decl @Same [concrete = constants.%Same] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.patt: @Same.%WrapType.loc4_32.2 (%WrapType.aa1) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @Same.%WrapType.loc4_32.2 (%WrapType.aa1) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: @Same.%WrapType.loc4_32.2 (%WrapType.aa1) = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: @Same.%WrapType.loc4_32.2 (%WrapType.aa1) = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @Same.%WrapType.loc4_32.2 (%WrapType.aa1) = value_param call_param0
// CHECK:STDOUT:     %.loc4_32: type = splice_block %WrapType.loc4_32.1 [symbolic = %WrapType.loc4_32.2 (constants.%WrapType.aa1)] {
// CHECK:STDOUT:       %WrapType.ref.loc4_22: %WrapType.type = name_ref WrapType, file.%WrapType.decl [concrete = constants.%WrapType.generic]
// CHECK:STDOUT:       %T.ref.loc4_31: type = name_ref T, %T.loc4_9.1 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:       %WrapType.loc4_32.1: type = class_type @WrapType, @WrapType(constants.%T) [symbolic = %WrapType.loc4_32.2 (constants.%WrapType.aa1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @Same.%WrapType.loc4_32.2 (%WrapType.aa1) = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: @Same.%WrapType.loc4_32.2 (%WrapType.aa1) = value_param call_param1
// CHECK:STDOUT:     %.loc4_48: type = splice_block %WrapType.loc4_48 [symbolic = %WrapType.loc4_32.2 (constants.%WrapType.aa1)] {
// CHECK:STDOUT:       %WrapType.ref.loc4_38: %WrapType.type = name_ref WrapType, file.%WrapType.decl [concrete = constants.%WrapType.generic]
// CHECK:STDOUT:       %T.ref.loc4_47: type = name_ref T, %T.loc4_9.1 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:       %WrapType.loc4_48: type = class_type @WrapType, @WrapType(constants.%T) [symbolic = %WrapType.loc4_32.2 (constants.%WrapType.aa1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: @Same.%WrapType.loc4_32.2 (%WrapType.aa1) = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.decl: %Type.type = fn_decl @Type [concrete = constants.%Type] {
// CHECK:STDOUT:     %T.patt.loc5_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %WrapType.ref: %WrapType.type = name_ref WrapType, file.%WrapType.decl [concrete = constants.%WrapType.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_9.1 [symbolic = %T.loc5_9.2 (constants.%T)]
// CHECK:STDOUT:     %WrapType.loc5_32.1: type = class_type @WrapType, @WrapType(constants.%T) [symbolic = %WrapType.loc5_32.2 (constants.%WrapType.aa1)]
// CHECK:STDOUT:     %T.loc5_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = out_param call_param0
// CHECK:STDOUT:     %return: ref @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type.eee] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type.eee = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @WrapType(%T.loc3_16.1: type) {
// CHECK:STDOUT:   %T.loc3_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc3_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%WrapType.aa1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Same(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   %WrapType.loc4_32.2: type = class_type @WrapType, @WrapType(%T.loc4_9.2) [symbolic = %WrapType.loc4_32.2 (constants.%WrapType.aa1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Same.%WrapType.loc4_32.2 (%WrapType.aa1) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_9.1: type](%a.param_patt: @Same.%WrapType.loc4_32.2 (%WrapType.aa1), %b.param_patt: @Same.%WrapType.loc4_32.2 (%WrapType.aa1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Type(%T.loc5_9.1: type) {
// CHECK:STDOUT:   %T.loc5_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   %WrapType.loc5_32.2: type = class_type @WrapType, @WrapType(%T.loc5_9.2) [symbolic = %WrapType.loc5_32.2 (constants.%WrapType.aa1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Type.%WrapType.loc5_32.2 (%WrapType.aa1) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %WrapType.val: @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = struct_value () [symbolic = %WrapType.val (constants.%WrapType.val.892)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc5_9.1: type) -> %return.param_patt: @Type.%WrapType.loc5_32.2 (%WrapType.aa1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_44.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_44.2: init @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = class_init (), %return [symbolic = %WrapType.val (constants.%WrapType.val.892)]
// CHECK:STDOUT:     %.loc5_45: init @Type.%WrapType.loc5_32.2 (%WrapType.aa1) = converted %.loc5_44.1, %.loc5_44.2 [symbolic = %WrapType.val (constants.%WrapType.val.892)]
// CHECK:STDOUT:     return %.loc5_45 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Same.ref: %Same.type = name_ref Same, file.%Same.decl [concrete = constants.%Same]
// CHECK:STDOUT:   %Type.ref.loc18_8: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %I.ref.loc18_13: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.1: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %.Self.ref.loc18_21: %I.type.733 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:   %J.ref.loc18_33: type = name_ref J, file.%J.decl [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %.Self.as_type.loc18_21: type = facet_access_type %.Self.ref.loc18_21 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc18_21: type = converted %.Self.ref.loc18_21, %.Self.as_type.loc18_21 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:   %.loc18_15: type = where_expr %.Self.1 [concrete = constants.%I.type.100] {
// CHECK:STDOUT:     requirement_impls %.loc18_21, %J.ref.loc18_33
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc18_8: <specific function> = specific_function %Type.ref.loc18_8, @Type(constants.%I.type.100) [concrete = constants.%Type.specific_fn.3cd]
// CHECK:STDOUT:   %.loc18_34: ref %WrapType.4f4 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc18_34: init %WrapType.4f4 = call %Type.specific_fn.loc18_8() to %.loc18_34
// CHECK:STDOUT:   %Type.ref.loc18_37: %Type.type = name_ref Type, file.%Type.decl [concrete = constants.%Type]
// CHECK:STDOUT:   %J.ref.loc18_42: type = name_ref J, file.%J.decl [concrete = constants.%J.type.eee]
// CHECK:STDOUT:   %.Self.2: %J.type.eee = bind_symbolic_name .Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:   %.Self.ref.loc18_50: %J.type.eee = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:   %I.ref.loc18_62: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:   %.Self.as_type.loc18_50: type = facet_access_type %.Self.ref.loc18_50 [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:   %.loc18_50: type = converted %.Self.ref.loc18_50, %.Self.as_type.loc18_50 [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:   %.loc18_44: type = where_expr %.Self.2 [concrete = constants.%J.type.c41] {
// CHECK:STDOUT:     requirement_impls %.loc18_50, %I.ref.loc18_62
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Type.specific_fn.loc18_37: <specific function> = specific_function %Type.ref.loc18_37, @Type(constants.%J.type.c41) [concrete = constants.%Type.specific_fn.6fb]
// CHECK:STDOUT:   %.loc18_63: ref %WrapType.802 = temporary_storage
// CHECK:STDOUT:   %Type.call.loc18_63: init %WrapType.802 = call %Type.specific_fn.loc18_37() to %.loc18_63
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(constants.%T) {
// CHECK:STDOUT:   %T.loc3_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc3_16.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Same(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc4_32.2 => constants.%WrapType.aa1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(@Same.%T.loc4_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%T) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc5_9.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc5_32.2 => constants.%WrapType.aa1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(@Type.%T.loc5_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%I.type.100) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%I.type.100
// CHECK:STDOUT:   %T.patt.loc5_9.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc5_32.2 => constants.%WrapType.4f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %WrapType.val => constants.%WrapType.val.77e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(constants.%I.type.100) {
// CHECK:STDOUT:   %T.loc3_16.2 => constants.%I.type.100
// CHECK:STDOUT:   %T.patt.loc3_16.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%J.type.c41) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%J.type.c41
// CHECK:STDOUT:   %T.patt.loc5_9.2 => constants.%T.patt
// CHECK:STDOUT:   %WrapType.loc5_32.2 => constants.%WrapType.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %WrapType.val => constants.%WrapType.val.0de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrapType(constants.%J.type.c41) {
// CHECK:STDOUT:   %T.loc3_16.2 => constants.%J.type.c41
// CHECK:STDOUT:   %T.patt.loc3_16.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/facet_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type %As.type.8ba [symbolic]
// CHECK:STDOUT:   %assoc0.ac5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0.a50: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.e44: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.560: type = facet_access_type %Self.e44 [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0.121: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, (%impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt.loc9_14.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc9_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc9_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc13_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc13_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc13_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc21_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc21_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc9_14.1: type) {
// CHECK:STDOUT:   %Dest.loc9_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc9_14.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc9_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc9_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc9_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.8ba) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc10_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc9_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc10_20.1: type = splice_block %.loc10_20.3 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc10_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc10_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc10_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc10_20.3: type = converted %Self.ref, %Self.as_type.loc10_20.2 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc10_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc13_22.1: type) {
// CHECK:STDOUT:   %Dest.loc13_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc13_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc13_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc13_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc14_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc13_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.3 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc14_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc14_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc14_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc14_20.3: type = converted %Self.ref, %Self.as_type.loc14_20.2 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc14_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e44]
// CHECK:STDOUT:   %Op.decl: %Op.type.613 = fn_decl @Op.1 [concrete = constants.%Op.d98] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18_37: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:     %Self.as_type.loc18_37: type = facet_access_type %Self.ref.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %.loc18_37: type = converted %Self.ref.loc18_37, %Self.as_type.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param0
// CHECK:STDOUT:     %.loc18_15.1: type = splice_block %.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_15: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_15.2: type = facet_access_type %Self.ref.loc18_15 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_15.2: type = converted %Self.ref.loc18_15, %Self.as_type.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param1
// CHECK:STDOUT:     %.loc18_28.1: type = splice_block %.loc18_28.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_28: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_28: type = facet_access_type %Self.ref.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_28.2: type = converted %Self.ref.loc18_28, %Self.as_type.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.121]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T.loc21_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc21_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Op.decl), @impl(%T.loc21_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc21_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%T.loc21_14.2 (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%T (%T) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %other.patt: @Op.2.%T (%T) = binding_pattern other
// CHECK:STDOUT:       %other.param_patt: @Op.2.%T (%T) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:       %return.patt: @Op.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Op.2.%T (%T) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc22_37: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc22_15: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc22_28: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %other: @Op.2.%T (%T) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%T (%T) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc9_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: @Convert.1.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc10_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc13_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419)]() -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:   %Self.as_type.loc18_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)](%other.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)) -> @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T)](%other.param_patt: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc9_14.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc9_14.2 => constants.%Dest.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc10_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc13_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc13_22.2 => constants.%Dest.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc13_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.e44) {
// CHECK:STDOUT:   %Self => constants.%Self.e44
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%Self.as_type.560
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc21_14.2 => constants.%T.patt
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc21_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
