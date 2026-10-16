// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/nested_facet_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/nested_facet_types.carbon

// --- nested_facet_types.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
  let U: type;
}

fn F(generic FF: (Z where .T = ()) where .U = .T) -> FF.U {
  return ();
}

// --- nested_facet_types_same.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
}

fn F(generic FF: ((Z where .T = ()) where .T = ()) where .T = ()) -> FF.T {
  return ();
}

// --- nested_facet_types_same_associated.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
  let U: type;
}

fn F(unused generic FF: ((Z where .T = .U) where .T = .U) where .T = .U) {}

// --- nested_facet_types_same_two_associated.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
  let U: type;
  let V: type;
}

fn F(unused generic FF: ((Z where .T = .U and .U = .V) where .T = .U and .U = .V) where .T = .U and .U = .V) {}

// --- nested_facet_types_same_associated_in_generic_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
  let U: type;
}
class C(T: type) {}

fn F(unused generic FF: ((Z where .T = C(.U)) where .T = C(.U)) where .T = C(.U)) {}

// --- fail_nested_facet_types_different.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
}

// CHECK:STDERR: fail_nested_facet_types_different.carbon:[[@LINE+2]]:25-55: error: associated constant `.(Z.T)` given two different values `()` and `{}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:
fn F(unused generic FF: (Z where .T = ()) where .T = {}) {}

// --- fail_nested_facet_types_different_with_associated.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
  let U: type;
}

// CHECK:STDERR: fail_nested_facet_types_different_with_associated.carbon:[[@LINE+2]]:25-55: error: associated constant `.(Z.T)` given two different values `.(Z.U)` and `{}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:
fn F(unused generic FF: (Z where .T = .U) where .T = {}) {}

// --- fail_nested_facet_types_different_with_impls_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
  let U: type;
}
interface Y {}

// CHECK:STDERR: fail_nested_facet_types_different_with_impls_interface.carbon:[[@LINE+2]]:25-73: error: associated constant `.(Z.T)` given two different values `()` and `{}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:
fn F(unused generic FF: (Z where .Self impls Y and .T = ()) where .T = {}) {}

// --- fail_nested_facet_types_different_with_impls_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
  let U: type;
}
constraint Y {}

// CHECK:STDERR: fail_nested_facet_types_different_with_impls_constraint.carbon:[[@LINE+2]]:25-73: error: associated constant `.(Z.T)` given two different values `()` and `{}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:
fn F(unused generic FF: (Z where .Self impls Y and .T = ()) where .T = {}) {}

// --- fail_nested_facet_types_different_with_associated_in_generic_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
  let U: type;
}
class C(T: type) {}

// CHECK:STDERR: fail_nested_facet_types_different_with_associated_in_generic_parameter.carbon:[[@LINE+2]]:25-58: error: associated constant `.(Z.T)` given two different values `C(.(Z.U))` and `{}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:
fn F(unused generic FF: (Z where .T = C(.U)) where .T = {}) {}

// --- nested_facet_types_same_with_bitand.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
  let U: type;
}

fn F(unused generic FF: ((Z where .T = .U) where .T = .U) & (Z where .T = .U)) {}

// --- fail_nested_facet_types_different_with_bitand.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
  let U: type;
}

// CHECK:STDERR: fail_nested_facet_types_different_with_bitand.carbon:[[@LINE+2]]:25-77: error: associated constant `.(Z.T)` given two different values `.(Z.U)` and `{}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:
fn F(unused generic FF: ((Z where .T = .U) where .T = .U) & (Z where .T = {})) {}

// --- fail_nested_facet_type_cycle.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T: type;
  let U: type;
}

// CHECK:STDERR: fail_nested_facet_type_cycle.carbon:[[@LINE+2]]:25-55: error: found cycle in facet type constraint for `.(Z.U)` [FacetTypeConstraintCycle]
// CHECK:STDERR:
fn F(unused generic FF: (Z where .T = .U) where .U = .T) {}

// --- fail_nested_facet_types_different_separated_order_one.carbon
library "[[@TEST_NAME]]";

interface Z {
  let S: type;
  let T: type;
  let U: type;
  let V: type;
}

// The ordering here is meant to find a way to insert a LHS between the two `.S`
// ids, by producing different orderings to get an id between them.

// CHECK:STDERR: fail_nested_facet_types_different_separated_order_one.carbon:[[@LINE+2]]:25-93: error: associated constant `.(Z.S)` given two different values `.(Z.U)` and `{}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:
fn F(unused generic FF: (((Z where .S = .U) where .V = {}) where .T = .U) & (Z where .S = {})) {}

// --- fail_nested_facet_types_different_separated_order_two.carbon
library "[[@TEST_NAME]]";

interface Z {
  let S: type;
  let T: type;
  let U: type;
  let V: type;
}

// The ordering here is meant to find a way to insert a LHS between the two `.S`
// ids, by producing different orderings to get an id between them. As of today,
// this is the case where `.T` is sorted between the two `.S` rewrite rules.

// CHECK:STDERR: fail_nested_facet_types_different_separated_order_two.carbon:[[@LINE+2]]:25-93: error: associated constant `.(Z.S)` given two different values `{}` and `.(Z.U)` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:
fn F(unused generic FF: (((Z where .V = {}) where .S = .U) where .T = .U) & (Z where .S = {})) {}

// --- fail_nested_facet_types_different_separated_order_three.carbon
library "[[@TEST_NAME]]";

interface Z {
  let S: type;
  let T: type;
  let U: type;
  let V: type;
}

// The ordering here is meant to find a way to insert a LHS between the two `.S`
// ids, by producing different orderings to get an id between them.

// CHECK:STDERR: fail_nested_facet_types_different_separated_order_three.carbon:[[@LINE+2]]:25-93: error: associated constant `.(Z.S)` given two different values `{}` and `.(Z.U)` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:
fn F(unused generic FF: (((Z where .V = {}) where .T = .U) where .S = .U) & (Z where .S = {})) {}

// --- repeated_with_value_available.carbon
library "[[@TEST_NAME]]";

interface Z { let X: type; let Y: type; }

fn F(generic T: ((Z where .Y = ()) where .X = .Y) where .X = .Y) -> T.X {
  return ();
}

class C(T: type) { adapt (); }

fn F1(generic T: (Z where .Y = C(()) and .X = C(.Y)) where .X = C(C(()))) -> T.X {
  return () as C(C(()));
}

fn F2(generic T: ((Z where .Y = C(())) where .X = C(.Y)) where .X = C(C(()))) -> T.X {
  return () as C(C(()));
}

fn F3(generic T: Z where .Y = C(()) and .X = C(.Y) and .X = C(C(()))) -> T.X {
  return () as C(C(()));
}

// --- conflicting_syntax_same_value.carbon
library "[[@TEST_NAME]]";

interface Z { let X: type; let Y: type; }

fn F1(generic T: ((Z where .Y = ()) where .X = .Y) where .X = ()) -> T.X {
  return ();
}

fn F2(generic T: ((Z where .Y = ()) where .X = ()) where .X = .Y) -> T.X {
  return ();
}

// --- nested_impls_repeated.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let Z1: type;
  let Z2: type;
}
class C(T: type) {}

constraint NZ {
  extend require impls Z where .Z1 = C(.Z2);
}

fn F(unused generic FF: ((Y where .Self impls NZ)
                  where .Self impls NZ)
                 where .Self impls NZ) {}

// --- fail_todo_nested_impl_combine_rewrites_from_named_constraints.carbon
library "[[@TEST_NAME]]";

interface Y {
  let Y1: type;
}
interface Z {
  let Z1: type;
  let Z2: type;
}
class C(T: type) { adapt (); }

constraint NZ1 {
  // This uses the rewrite from NZ2.
  extend require impls Z where .Z1 = C(.Z2);
}
constraint NZ2 {
  extend require impls Z where .Z2 = ();
}

fn F(generic FF: ((Y where .Y1 = ()) where .Self impls NZ1) where .Self impls NZ2) -> FF.(Z.Z1) {
  // CHECK:STDERR: fail_todo_nested_impl_combine_rewrites_from_named_constraints.carbon:[[@LINE+4]]:10-20: error: cannot implicitly convert expression of type `C(())` to `FF.(Z.Z1)` [ConversionFailure]
  // CHECK:STDERR: fail_todo_nested_impl_combine_rewrites_from_named_constraints.carbon:[[@LINE+3]]:3-21: note: type `C(())` does not implement interface `Core.ImplicitAs(FF.(Z.Z1))` [MissingImplForType]
  // CHECK:STDERR: fail_todo_nested_impl_combine_rewrites_from_named_constraints.carbon:[[@LINE-3]]:87-95: note: return type of function is `FF.(Z.Z1)` [ReturnTypeHereNote]
  // CHECK:STDERR:
  return () as C(());
}

// --- fail_where_nested_inside_where_impls.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_where_nested_inside_where_impls.carbon:[[@LINE+3]]:41-67: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fail_where_nested_inside_where_impls.carbon:[[@LINE+2]]:17-26: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR:
fn F(generic _: type where .Self impls (type where .Self impls type)) {}

// --- fail_where_nested_inside_where_rewrite.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

// CHECK:STDERR: fail_where_nested_inside_where_rewrite.carbon:[[@LINE+3]]:32-58: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fail_where_nested_inside_where_rewrite.carbon:[[@LINE+2]]:17-23: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR:
fn F(generic _: Z where .Z1 = (type where .Self impls type)) {}

// --- fail_where_nested_inside_where_same_type.carbon
library "[[@TEST_NAME]]";

interface Z {}
class C;

// CHECK:STDERR: fail_where_nested_inside_where_same_type.carbon:[[@LINE+3]]:31-57: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fail_where_nested_inside_where_same_type.carbon:[[@LINE+2]]:17-23: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR:
fn F(generic _: Z where C == (type where .Self impls type)) {}

// --- fail_where_nested_inside_binding.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
interface Y {}

// CHECK:STDERR: fail_where_nested_inside_binding.carbon:[[@LINE+2]]:17-43: error: found `where` expression in binding's type that does not constrain the binding [InvalidWhereInsideBinding]
// CHECK:STDERR:
fn F(generic _: Z(type where .Self impls Y)) {}

// --- fail_where_double_nested_inside_binding.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
interface Y {}

// CHECK:STDERR: fail_where_double_nested_inside_binding.carbon:[[@LINE+2]]:17-43: error: found `where` expression in binding's type that does not constrain the binding [InvalidWhereInsideBinding]
// CHECK:STDERR:
fn F(generic _: Z(type where .Self impls Y) where .Self impls Y) {}

// --- fail_where_triple_nested_inside_binding.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
interface Y {}

// CHECK:STDERR: fail_where_triple_nested_inside_binding.carbon:[[@LINE+2]]:18-44: error: found `where` expression in binding's type that does not constrain the binding [InvalidWhereInsideBinding]
// CHECK:STDERR:
fn F(generic _: (Z(type where .Self impls Y) where .Self impls Y) where .Self impls Y) {}

// --- fail_where_nested_inside_binding_combined.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
interface Y {}

// CHECK:STDERR: fail_where_nested_inside_binding_combined.carbon:[[@LINE+2]]:17-47: error: found `where` expression in binding's type that does not constrain the binding [InvalidWhereInsideBinding]
// CHECK:STDERR:
fn F(generic _: Z(type where .Self impls Y) & Y) {}

// --- fail_where_double_nested_inside_binding_combined.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
interface Y {}

// CHECK:STDERR: fail_where_double_nested_inside_binding_combined.carbon:[[@LINE+2]]:17-69: error: found `where` expression in binding's type that does not constrain the binding [InvalidWhereInsideBinding]
// CHECK:STDERR:
fn F(generic _: (Z(type where .Self impls Y) where .Self impls Y) & Y) {}

// --- fail_where_triple_nested_inside_binding_combined.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
interface Y {}

// CHECK:STDERR: fail_where_triple_nested_inside_binding_combined.carbon:[[@LINE+2]]:17-91: error: found `where` expression in binding's type that does not constrain the binding [InvalidWhereInsideBinding]
// CHECK:STDERR:
fn F(generic _: ((Z(type where .Self impls Y) where .Self impls Y) where .Self impls Y) & Y) {}

// --- fail_where_nested_inside_binding_of_non_facet_type.carbon
library "[[@TEST_NAME]]";

class C(T: type);
interface Y {}

// CHECK:STDERR: fail_where_nested_inside_binding_of_non_facet_type.carbon:[[@LINE+2]]:17-43: error: found `where` expression in binding's type that does not constrain the binding [InvalidWhereInsideBinding]
// CHECK:STDERR:
fn F(generic _: C(type where .Self impls Y)) {}

// --- fail_where_nested_inside_runtime_let_binding.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
interface Y {}

// CHECK:STDERR: fail_where_nested_inside_runtime_let_binding.carbon:[[@LINE+2]]:9-35: error: found `where` expression in binding's type that does not constrain the binding [InvalidWhereInsideBinding]
// CHECK:STDERR:
fn F(_: Z(type where .Self impls Y)) {}

// --- fail_where_nested_inside_form_binding.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
interface Y {}

// CHECK:STDERR: fail_where_nested_inside_form_binding.carbon:[[@LINE+2]]:10-36: error: found `where` expression in binding's type that does not constrain the binding [InvalidWhereInsideBinding]
// CHECK:STDERR:
fn F(_:? Z(type where .Self impls Y)) {}

// --- fail_where_nested_inside_var_binding.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
interface Y {}

// CHECK:STDERR: fail_where_nested_inside_var_binding.carbon:[[@LINE+2]]:13-39: error: found `where` expression in binding's type that does not constrain the binding [InvalidWhereInsideBinding]
// CHECK:STDERR:
fn F(var _: Z(type where .Self impls Y)) {}
