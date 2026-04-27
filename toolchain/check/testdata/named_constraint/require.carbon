// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/require.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/require.carbon

// --- extend.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

//@dump-sem-ir-begin
constraint Z {
  extend require impls Y;
}
//@dump-sem-ir-end

fn F(T:! Z, t: T) {
  //@dump-sem-ir-begin
  T.YY();
  T.(Y.YY)();

  t.YY();
  //@dump-sem-ir-end
}

// --- extend_with_self.carbon
library "[[@TEST_NAME]]";

interface Y(T:! type) {
  fn YY();
}

//@dump-sem-ir-begin
constraint Z {
  extend require impls Y(Self);
}
//@dump-sem-ir-end

fn F(T:! Z, t: T) {
  //@dump-sem-ir-begin
  T.YY();
  T.(Y(T).YY)();

  t.YY();
  //@dump-sem-ir-end
}

// --- implicit_self_impls.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

//@dump-sem-ir-begin
constraint Z {
  require impls Y;
}
//@dump-sem-ir-end

fn F(T:! Z) {
  T.(Y.YY)();
}

// --- explicit_self_impls.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

//@dump-sem-ir-begin
constraint Z {
  require Self impls Y;
}
//@dump-sem-ir-end

fn F(T:! Z) {
  T.(Y.YY)();
}

// --- fail_implicit_self_no_extend_name_lookup_fails.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

constraint Z {
  require impls Y;
}

fn F(T:! Z) {
  // This should fail name lookup since Z does not extend Y.
  //
  // CHECK:STDERR: fail_implicit_self_no_extend_name_lookup_fails.carbon:[[@LINE+4]]:3: error: member name `YY` not found in `Z` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   T.YY();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.YY();
}

// --- fail_explicit_self_no_extend_name_lookup_fails.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

constraint Z {
  require Self impls Y;
}

fn F(T:! Z) {
  // This should fail name lookup since Z does not extend Y.
  //
  // CHECK:STDERR: fail_explicit_self_no_extend_name_lookup_fails.carbon:[[@LINE+4]]:3: error: member name `YY` not found in `Z` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   T.YY();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.YY();
}

// --- explicit_self_specific_impls.carbon
library "[[@TEST_NAME]]";

interface Y {}

class C(T:! type);

//@dump-sem-ir-begin
constraint Z {
  require C(Self) impls Y;
}
//@dump-sem-ir-end

// --- require_impls_where.carbon
library "[[@TEST_NAME]]";

interface Y { let Y1:! type; }

//@dump-sem-ir-begin
constraint Z {
  require impls Y where .Y1 = ();
}
//@dump-sem-ir-end

// --- fail_require_impls_incomplete_constraint.carbon
library "[[@TEST_NAME]]";

constraint Y;

//@dump-sem-ir-begin
constraint Z {
  // CHECK:STDERR: fail_require_impls_incomplete_constraint.carbon:[[@LINE+7]]:17: error: facet type `Y` cannot be identified in `require` declaration [RequireImplsUnidentifiedFacetType]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR: fail_require_impls_incomplete_constraint.carbon:[[@LINE-7]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint Y;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  require impls Y;
}
//@dump-sem-ir-end

// --- fail_require_impls_incomplete_self.carbon
library "[[@TEST_NAME]]";

constraint Z {
  // CHECK:STDERR: fail_require_impls_incomplete_self.carbon:[[@LINE+4]]:17: error: facet type in `require` declaration refers to the named constraint `Z` from within its definition [RequireImplsReferenceCycle]
  // CHECK:STDERR:   require impls Z;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  require impls Z;
}

// --- fail_require_impls_incomplete_self_through_alias.carbon
library "[[@TEST_NAME]]";

constraint Z;

alias Y = Z;
constraint Z {
  // CHECK:STDERR: fail_require_impls_incomplete_self_through_alias.carbon:[[@LINE+4]]:17: error: facet type in `require` declaration refers to the named constraint `Z` from within its definition [RequireImplsReferenceCycle]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  require impls Y;
}

// --- fail_require_impls_incomplete_self_forward_declared.carbon
library "[[@TEST_NAME]]";

constraint Z;

constraint Z {
  // CHECK:STDERR: fail_require_impls_incomplete_self_forward_declared.carbon:[[@LINE+4]]:17: error: facet type in `require` declaration refers to the named constraint `Z` from within its definition [RequireImplsReferenceCycle]
  // CHECK:STDERR:   require impls Z;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  require impls Z;
}

// --- fail_require_impls_incomplete_self_in_period_self_impls.carbon
library "[[@TEST_NAME]]";

constraint Z {
  // CHECK:STDERR: fail_require_impls_incomplete_self_in_period_self_impls.carbon:[[@LINE+4]]:17: error: facet type in `require` declaration refers to the named constraint `Z` from within its definition [RequireImplsReferenceCycle]
  // CHECK:STDERR:   require impls type where .Self impls Z;
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require impls type where .Self impls Z;
}

// --- fail_require_impls_incomplete_self_in_class_impls.carbon
library "[[@TEST_NAME]]";

class C(T:! type);
constraint Z {
  // CHECK:STDERR: fail_require_impls_incomplete_self_in_class_impls.carbon:[[@LINE+4]]:17: error: facet type in `require` declaration refers to the named constraint `Z` from within its definition [RequireImplsReferenceCycle]
  // CHECK:STDERR:   require impls type where C(.Self) impls Z;
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require impls type where C(.Self) impls Z;
}

// --- fail_require_impls_incomplete_indirect.carbon
library "[[@TEST_NAME]]";

constraint Z;

constraint Y {
  // CHECK:STDERR: fail_require_impls_incomplete_indirect.carbon:[[@LINE+7]]:17: error: facet type `Z` cannot be identified in `require` declaration [RequireImplsUnidentifiedFacetType]
  // CHECK:STDERR:   require impls Z;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR: fail_require_impls_incomplete_indirect.carbon:[[@LINE-6]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint Z;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  require impls Z;
}

constraint Z {
  require impls Y;
}

// --- fail_require_impls_incomplete_self_specific.carbon
library "[[@TEST_NAME]]";

constraint Z(T:! type) {
  // CHECK:STDERR: fail_require_impls_incomplete_self_specific.carbon:[[@LINE+4]]:19: error: facet type in `require` declaration refers to the named constraint `Z` from within its definition [RequireImplsReferenceCycle]
  // CHECK:STDERR:   require T impls Z(Self);
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR:
  require T impls Z(Self);
}

// --- fail_require_impls_without_self.carbon
library "[[@TEST_NAME]]";

interface Y {
  let Y1:! type;
}

//@dump-sem-ir-begin
constraint Z(T:! type) {
  // Either the type `T` or the facet type `Y` must mention `Self` in a way that
  // it would appear in the type structure used for impl lookup (so inside a
  // `where` does not count). But they don't.
  //
  // CHECK:STDERR: fail_require_impls_without_self.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic argument for each required interface, but found interface `Y` without a `Self` argument [RequireImplsMissingSelf]
  // CHECK:STDERR:   require T impls Y where .Y1 = Self;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require T impls Y where .Y1 = Self;
}
//@dump-sem-ir-end

// --- fail_require_impls_without_self_in_one_interface.carbon
library "[[@TEST_NAME]]";

interface Y(T:! type) {}

constraint Z(T:! type) {
  // Self is in one interface but not the other.
  //
  // CHECK:STDERR: fail_require_impls_without_self_in_one_interface.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic argument for each required interface, but found interface `Y({})` without a `Self` argument [RequireImplsMissingSelf]
  // CHECK:STDERR:   require T impls Y(Self) & Y({});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require T impls Y(Self) & Y({});
}

// --- fail_require_impls_without_self_in_multiple_interfaces.carbon
library "[[@TEST_NAME]]";

interface Y(T:! type) {}

constraint Z(T:! type) {
  // Self does not appear in either interface, we should get an error for each.
  //
  // CHECK:STDERR: fail_require_impls_without_self_in_multiple_interfaces.carbon:[[@LINE+8]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic argument for each required interface, but found interface `Y(())` without a `Self` argument [RequireImplsMissingSelf]
  // CHECK:STDERR:   require T impls Y(()) & Y({});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_require_impls_without_self_in_multiple_interfaces.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic argument for each required interface, but found interface `Y({})` without a `Self` argument [RequireImplsMissingSelf]
  // CHECK:STDERR:   require T impls Y(()) & Y({});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require T impls Y(()) & Y({});
}

// --- fail_self_impls_self.carbon
library "[[@TEST_NAME]]";

constraint Z(T:! type) {
  // CHECK:STDERR: fail_self_impls_self.carbon:[[@LINE+4]]:17: error: `require` declaration constrained by a non-facet type; expected an `interface` or `constraint` name after `impls` [RequireImplsMissingFacetType]
  // CHECK:STDERR:   require impls Self;
  // CHECK:STDERR:                 ^~~~
  // CHECK:STDERR:
  require impls Self;
}

// --- fail_impls_type.carbon
library "[[@TEST_NAME]]";

class C(T:! type) {}

constraint Z(T:! type) {
  // CHECK:STDERR: fail_impls_type.carbon:[[@LINE+4]]:19: error: `require` declaration constrained by a non-facet type; expected an `interface` or `constraint` name after `impls` [RequireImplsMissingFacetType]
  // CHECK:STDERR:   require T impls C(Self);
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR:
  require T impls C(Self);
}

// --- fail_non_type_impls.carbon
library "[[@TEST_NAME]]";

interface Y {}

constraint Z(T:! type) {
  // CHECK:STDERR: fail_non_type_impls.carbon:[[@LINE+7]]:11: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   require 1 impls Y;
  // CHECK:STDERR:           ^
  // CHECK:STDERR: fail_non_type_impls.carbon:[[@LINE+4]]:11: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   require 1 impls Y;
  // CHECK:STDERR:           ^
  // CHECK:STDERR:
  require 1 impls Y;
}

// --- fail_impls_non_type.carbon
library "[[@TEST_NAME]]";

interface Y {}

constraint Z(T:! type) {
  // CHECK:STDERR: fail_impls_non_type.carbon:[[@LINE+4]]:17: error: `require` declaration constrained by a non-facet type; expected an `interface` or `constraint` name after `impls` [RequireImplsMissingFacetType]
  // CHECK:STDERR:   require impls 1;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  require impls 1;
}

// --- require_self_in_requirement.carbon
library "[[@TEST_NAME]]";

interface Y {
  let Y1:! type;
}

//@dump-sem-ir-begin
constraint Z {
  // Self can appear in a requirement.
  require impls Y where .Y1 = Self;
}
//@dump-sem-ir-end

// --- require_self.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

constraint N {
  require impls Z(Self);
}

impl () as Z(()) {}

fn F() {
  () as N;
}

// --- fail_require_different_self.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

constraint N {
  require impls Z(Self);
}

impl () as Z({}) {}

fn F() {
  // N requires () impls Z(Self) which is Z(()) for this query, but it impls
  // Z({}).
  //
  // CHECK:STDERR: fail_require_different_self.carbon:[[@LINE+4]]:3: error: cannot convert type `()` into type implementing `N` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   () as N;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  () as N;
}

// --- fail_require_different_parameter.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

constraint N {
  require impls Z({});
}

impl () as Z(()) {}

fn F() {
  // N requires () impls Z({}) but it impls Z(()).
  //
  // CHECK:STDERR: fail_require_different_parameter.carbon:[[@LINE+4]]:3: error: cannot convert type `()` into type implementing `N` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   () as N;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  () as N;
}

// --- require_different_type_impls.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

constraint N {
  require {} impls Z(Self);
}

impl {} as Z(()) {}

fn F() {
  () as N;
}

// --- require_different_type_impls_nested_constraint.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

constraint M(T:! type) {
  extend require impls Z(T);
}

constraint N {
  require {} impls M(Self);
}

impl {} as Z(()) {}

fn F() {
  () as N;
}

// --- fail_require_different_type_impls_nested_constraint_without_reference_to_outer_self.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

constraint M(T:! type) {
  extend require impls Z(Self);
}

constraint N {
  // `M(T)` does not make use of the `T`, which is assigned `Self`, so we end up
  // without any interface specific referencing the `Self` of `N`.
  //
  // CHECK:STDERR: fail_require_different_type_impls_nested_constraint_without_reference_to_outer_self.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic argument for each required interface, but found interface `Z({})` without a `Self` argument [RequireImplsMissingSelf]
  // CHECK:STDERR:   require {} impls M(Self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require {} impls M(Self);
}

// --- fail_require_different_type_impls_nested_constraint_empty.carbon
library "[[@TEST_NAME]]";

constraint M(T:! type) {}

constraint N {
  // `M(T)` does not make use of the `T`, which is assigned `Self`, so we end up
  // without any interface specific referencing the `Self` of `N`.
  //
  // CHECK:STDERR: fail_require_different_type_impls_nested_constraint_empty.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic argument for each required interface, but no interfaces were found [RequireImplsMissingSelfEmptyFacetType]
  // CHECK:STDERR:   require {} impls M(Self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require {} impls M(Self);
}

// --- fail_require_different_type_impls_different_parameter.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

constraint N {
  require {} impls Z(Self);
}

impl {} as Z({}) {}

fn F() {
  // N requires {} impls Z(Self) which is Z(()) for this query, but it impls
  // Z({}).
  //
  // CHECK:STDERR: fail_require_different_type_impls_different_parameter.carbon:[[@LINE+4]]:3: error: cannot convert type `()` into type implementing `N` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   () as N;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  () as N;
}

// --- require_with_rewrite_constraint.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

constraint N {
  require {} impls Z(Self) where .X = Self;
}

// Will satisfy `() as N`.
impl {} as Z(()) where .X = () {}

fn F() {
  () as N;
}

// --- require_with_rewrite_constraint_period_self.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

constraint N(T:! type) {
  require T impls Z(Self) where .X = .Self;
}

// Will satisfy `() as N({})`.
impl {} as Z(()) where .X = {} {}

fn F() {
  () as N({});
}

// --- todo_fail_require_with_mismatching_rewrite_constraint.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

constraint N {
  require {} impls Z(Self) where .X = Self;
}

// Will not satisfy `() as N` as the rewrite does not match.
impl {} as Z(()) where .X = {} {}

fn F() {
  // TODO: Should fail, but we're currently checking the requirements of `N`
  // rather than of the identified facet type of `N` (which would include the
  // require decl's requirements).
  () as N;
}

// --- todo_fail_require_with_mismatching_rewrite_constraint_period_self.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

constraint N(T:! type) {
  require T impls Z(Self) where .X = .Self;
}

// Will not satisfy `() as N({})` as the rewrite does not match.
impl {} as Z(()) where .X = () {}

fn F() {
  // TODO: Should fail, but we're currently checking the requirements of `N`
  // rather than of the identified facet type of `N` (which would include the
  // require decl's requirements).
  () as N({});
}

// --- fail_require_with_period_self_impls_not_satisfied.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface Y(T:! type) {}

class C(T:! type);

constraint N {
  require impls Z where .Self impls Y(.Self);
}

fn F(_:! N) {}

fn G(T:! Z) {
  // CHECK:STDERR: fail_require_with_period_self_impls_not_satisfied.carbon:[[@LINE+4]]:3: error: cannot convert type `T` that implements `Z` into type implementing `N` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   T as N;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  T as N;
}

// --- require_with_period_self_impls.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface Y(T:! type) {}

class C(T:! type);

//@dump-sem-ir-begin
constraint N {
  require impls Z where .Self impls Y(.Self);
}
//@dump-sem-ir-end

fn F(_:! N) {}

fn G(T:! Z & Y(.Self)) {
  T as N;
}

// --- fail_require_with_specific_period_self_not_satisfied.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface Y(T:! type) {}

class C(T:! type);

constraint N {
  require impls Z where C(.Self) impls Y(.Self);
}

fn F(_:! N) {}

fn G(T:! Z) {
  // CHECK:STDERR: fail_require_with_specific_period_self_not_satisfied.carbon:[[@LINE+4]]:3: error: cannot convert type `T` that implements `Z` into type implementing `N` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   T as N;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  T as N;
}

// --- require_with_specific_period_self.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface Y(T:! type) {}

class C(T:! type);

//@dump-sem-ir-begin
constraint N {
  require impls Z where C(.Self) impls Y(.Self);
}
//@dump-sem-ir-end

fn F(_:! N) {}

fn G(T:! Z) {
  impl C(T) as Y(T) {}
  T as N;
}

// CHECK:STDOUT: --- extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, @Y.WithSelf.%Y.WithSelf.YY.decl [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.550: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.550 [symbolic]
// CHECK:STDOUT:   %T: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.349: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Y [symbolic]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %T.binding.as_type, (%Y.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Y.WithSelf.YY.type.7ea: type = fn_type @Y.WithSelf.YY, @Y.WithSelf(%Y.facet) [symbolic]
// CHECK:STDOUT:   %.1b4: type = fn_type_with_self_type %Y.WithSelf.YY.type.7ea, %Y.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.1b4 = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Y.WithSelf.YY(%Y.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.WithSelf.decl = constraint_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @Z.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: type = specific_constant @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.ref, @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.550) [concrete = constants.%Y.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .YY = <poisoned>
// CHECK:STDOUT:   extend @Z.WithSelf.%.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Self.as_type impls @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_7.2: %Z.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_7.1, @Y [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness)]
// CHECK:STDOUT:   %Y.facet.loc15: %Y.type = facet_value %T.binding.as_type, (%Y.lookup_impl_witness) [symbolic = %Y.facet.loc15 (constants.%Y.facet)]
// CHECK:STDOUT:   %Y.WithSelf.YY.type: type = fn_type @Y.WithSelf.YY, @Y.WithSelf(%Y.facet.loc15) [symbolic = %Y.WithSelf.YY.type (constants.%Y.WithSelf.YY.type.7ea)]
// CHECK:STDOUT:   %.loc15_4.2: type = fn_type_with_self_type %Y.WithSelf.YY.type, %Y.facet.loc15 [symbolic = %.loc15_4.2 (constants.%.1b4)]
// CHECK:STDOUT:   %impl.elem0.loc15_4.2: @F.%.loc15_4.2 (%.1b4) = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_4.2: <specific function> = specific_impl_function %impl.elem0.loc15_4.2, @Y.WithSelf.YY(%Y.facet.loc15) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc15: %Z.type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc15_4.1: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %YY.ref.loc15: %Y.assoc_type = name_ref YY, @Y.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc15_4.1: @F.%.loc15_4.2 (%.1b4) = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc15_4.1: <specific function> = specific_impl_function %impl.elem0.loc15_4.1, @Y.WithSelf.YY(constants.%Y.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Y.WithSelf.YY.call.loc15: init %empty_tuple.type = call %specific_impl_fn.loc15_4.1()
// CHECK:STDOUT:     %T.ref.loc16: %Z.type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     %YY.ref.loc16: %Y.assoc_type = name_ref YY, @Y.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc16: type = facet_access_type %T.ref.loc16 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Y.facet.loc16: %Y.type = facet_value %T.as_type.loc16, (constants.%Y.lookup_impl_witness) [symbolic = %Y.facet.loc15 (constants.%Y.facet)]
// CHECK:STDOUT:     %.loc16: %Y.type = converted %T.ref.loc16, %Y.facet.loc16 [symbolic = %Y.facet.loc15 (constants.%Y.facet)]
// CHECK:STDOUT:     %impl.elem0.loc16: @F.%.loc15_4.2 (%.1b4) = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc16: <specific function> = specific_impl_function %impl.elem0.loc16, @Y.WithSelf.YY(constants.%Y.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Y.WithSelf.YY.call.loc16: init %empty_tuple.type = call %specific_impl_fn.loc16()
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %YY.ref.loc18: %Y.assoc_type = name_ref YY, @Y.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc18: @F.%.loc15_4.2 (%.1b4) = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc18: <specific function> = specific_impl_function %impl.elem0.loc18, @Y.WithSelf.YY(constants.%Y.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Y.WithSelf.YY.call.loc18: init %empty_tuple.type = call %specific_impl_fn.loc18()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.550) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc13_7.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.349
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_with_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Y.type.82b: type = generic_interface_type @Y [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Y.generic: %Y.type.82b = struct_value () [concrete]
// CHECK:STDOUT:   %Y.type.a16: type = facet_type <@Y, @Y(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.6f4: %Y.type.a16 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.550: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.550 [symbolic]
// CHECK:STDOUT:   %Y.type.83d: type = facet_type <@Y, @Y(%Self.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %require_complete.057: <witness> = require_complete_type %Y.type.83d [symbolic]
// CHECK:STDOUT:   %T.f45: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.f45 [symbolic]
// CHECK:STDOUT:   %pattern_type.349: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Y.type.55c: type = facet_type <@Y, @Y(%T.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %require_complete.ec2: <witness> = require_complete_type %Y.type.55c [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.eee: type = assoc_entity_type @Y, @Y(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %assoc0.e42: %Y.assoc_type.eee = assoc_entity element0, @Y.WithSelf.%Y.WithSelf.YY.decl [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %T.f45, @Y, @Y(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %Y.facet: %Y.type.55c = facet_value %T.binding.as_type, (%Y.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Y.WithSelf.YY.type.aeb: type = fn_type @Y.WithSelf.YY, @Y.WithSelf(%T.binding.as_type, %Y.facet) [symbolic]
// CHECK:STDOUT:   %.b61: type = fn_type_with_self_type %Y.WithSelf.YY.type.aeb, %Y.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.b61 = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Y.WithSelf.YY(%T.binding.as_type, %Y.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.WithSelf.decl = constraint_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type.loc9_18 impls %Y.type.loc9_30.1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc9_18: type = facet_access_type @Z.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: %Y.type.82b = name_ref Y, file.%Y.decl [concrete = constants.%Y.generic]
// CHECK:STDOUT:     %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:     %Self.as_type.loc9_30: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc9: type = converted %Self.ref, %Self.as_type.loc9_30 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.type.loc9_30.1: type = facet_type <@Y, @Y(constants.%Self.binding.as_type)> [symbolic = %Y.type.loc9_30.2 (constants.%Y.type.83d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: type = specific_constant @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.type.loc9_30.1, @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.550) [symbolic = %Y.type (constants.%Y.type.83d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .YY = <poisoned>
// CHECK:STDOUT:   extend @Z.WithSelf.%.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Self.as_type.loc9_18 impls @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.type.loc9_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Y.type.loc9_30.2: type = facet_type <@Y, @Y(%Self.binding.as_type)> [symbolic = %Y.type.loc9_30.2 (constants.%Y.type.83d)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_7.2: %Z.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Y.type.loc15: type = facet_type <@Y, @Y(%T.binding.as_type)> [symbolic = %Y.type.loc15 (constants.%Y.type.55c)]
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %Y.type.loc15 [symbolic = %require_complete.loc15 (constants.%require_complete.ec2)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%T.binding.as_type) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.eee)]
// CHECK:STDOUT:   %assoc0: @F.%Y.assoc_type (%Y.assoc_type.eee) = assoc_entity element0, @Y.WithSelf.%Y.WithSelf.YY.decl [symbolic = %assoc0 (constants.%assoc0.e42)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_7.1, @Y, @Y(%T.binding.as_type) [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness)]
// CHECK:STDOUT:   %Y.facet.loc15: @F.%Y.type.loc15 (%Y.type.55c) = facet_value %T.binding.as_type, (%Y.lookup_impl_witness) [symbolic = %Y.facet.loc15 (constants.%Y.facet)]
// CHECK:STDOUT:   %Y.WithSelf.YY.type: type = fn_type @Y.WithSelf.YY, @Y.WithSelf(%T.binding.as_type, %Y.facet.loc15) [symbolic = %Y.WithSelf.YY.type (constants.%Y.WithSelf.YY.type.aeb)]
// CHECK:STDOUT:   %.loc15_4.3: type = fn_type_with_self_type %Y.WithSelf.YY.type, %Y.facet.loc15 [symbolic = %.loc15_4.3 (constants.%.b61)]
// CHECK:STDOUT:   %impl.elem0.loc15_4.2: @F.%.loc15_4.3 (%.b61) = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_4.2: <specific function> = specific_impl_function %impl.elem0.loc15_4.2, @Y.WithSelf.YY(%T.binding.as_type, %Y.facet.loc15) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc15: %Z.type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T.f45)]
// CHECK:STDOUT:     %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc15_4.1: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc15_4.2: @F.%Y.assoc_type (%Y.assoc_type.eee) = specific_constant @Y.WithSelf.%assoc0.loc4_10.1, @Y.WithSelf(constants.%T.binding.as_type, constants.%T.f45) [symbolic = %assoc0 (constants.%assoc0.e42)]
// CHECK:STDOUT:     %YY.ref.loc15: @F.%Y.assoc_type (%Y.assoc_type.eee) = name_ref YY, %.loc15_4.2 [symbolic = %assoc0 (constants.%assoc0.e42)]
// CHECK:STDOUT:     %impl.elem0.loc15_4.1: @F.%.loc15_4.3 (%.b61) = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc15_4.1: <specific function> = specific_impl_function %impl.elem0.loc15_4.1, @Y.WithSelf.YY(constants.%T.binding.as_type, constants.%Y.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Y.WithSelf.YY.call.loc15: init %empty_tuple.type = call %specific_impl_fn.loc15_4.1()
// CHECK:STDOUT:     %T.ref.loc16_3: %Z.type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T.f45)]
// CHECK:STDOUT:     %Y.ref: %Y.type.82b = name_ref Y, file.%Y.decl [concrete = constants.%Y.generic]
// CHECK:STDOUT:     %T.ref.loc16_8: %Z.type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T.f45)]
// CHECK:STDOUT:     %T.as_type.loc16_9: type = facet_access_type %T.ref.loc16_8 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc16_9: type = converted %T.ref.loc16_8, %T.as_type.loc16_9 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Y.type.loc16: type = facet_type <@Y, @Y(constants.%T.binding.as_type)> [symbolic = %Y.type.loc15 (constants.%Y.type.55c)]
// CHECK:STDOUT:     %.loc16_10: @F.%Y.assoc_type (%Y.assoc_type.eee) = specific_constant @Y.WithSelf.%assoc0.loc4_10.1, @Y.WithSelf(constants.%T.binding.as_type, constants.%Self.6f4) [symbolic = %assoc0 (constants.%assoc0.e42)]
// CHECK:STDOUT:     %YY.ref.loc16: @F.%Y.assoc_type (%Y.assoc_type.eee) = name_ref YY, %.loc16_10 [symbolic = %assoc0 (constants.%assoc0.e42)]
// CHECK:STDOUT:     %T.as_type.loc16_4: type = facet_access_type %T.ref.loc16_3 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Y.facet.loc16: @F.%Y.type.loc15 (%Y.type.55c) = facet_value %T.as_type.loc16_4, (constants.%Y.lookup_impl_witness) [symbolic = %Y.facet.loc15 (constants.%Y.facet)]
// CHECK:STDOUT:     %.loc16_4: @F.%Y.type.loc15 (%Y.type.55c) = converted %T.ref.loc16_3, %Y.facet.loc16 [symbolic = %Y.facet.loc15 (constants.%Y.facet)]
// CHECK:STDOUT:     %impl.elem0.loc16: @F.%.loc15_4.3 (%.b61) = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc16: <specific function> = specific_impl_function %impl.elem0.loc16, @Y.WithSelf.YY(constants.%T.binding.as_type, constants.%Y.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Y.WithSelf.YY.call.loc16: init %empty_tuple.type = call %specific_impl_fn.loc16()
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc18: @F.%Y.assoc_type (%Y.assoc_type.eee) = specific_constant @Y.WithSelf.%assoc0.loc4_10.1, @Y.WithSelf(constants.%T.binding.as_type, constants.%T.f45) [symbolic = %assoc0 (constants.%assoc0.e42)]
// CHECK:STDOUT:     %YY.ref.loc18: @F.%Y.assoc_type (%Y.assoc_type.eee) = name_ref YY, %.loc18 [symbolic = %assoc0 (constants.%assoc0.e42)]
// CHECK:STDOUT:     %impl.elem0.loc18: @F.%.loc15_4.3 (%.b61) = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc18: <specific function> = specific_impl_function %impl.elem0.loc18, @Y.WithSelf.YY(constants.%T.binding.as_type, constants.%Y.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Y.WithSelf.YY.call.loc18: init %empty_tuple.type = call %specific_impl_fn.loc18()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.550) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.83d
// CHECK:STDOUT:   %require_complete => constants.%require_complete.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %Y.type.loc9_30.2 => constants.%Y.type.83d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.f45) {
// CHECK:STDOUT:   %T.loc13_7.1 => constants.%T.f45
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.349
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%T.f45) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T.f45
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.55c
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ec2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%T.f45) {
// CHECK:STDOUT:   %Self => constants.%T.f45
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %Y.type.loc9_30.2 => constants.%Y.type.55c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_self_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.550: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.550 [symbolic]
// CHECK:STDOUT:   %T: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.WithSelf.decl = constraint_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @Z.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Self.as_type impls @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.550) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_self_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.550: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.550 [symbolic]
// CHECK:STDOUT:   %T: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.WithSelf.decl = constraint_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %.loc9 impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc9: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%.loc9 impls @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.550) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_self_specific_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.550: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.550 [symbolic]
// CHECK:STDOUT:   %C.237: type = class_type @C, @C(%Self.binding.as_type) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.WithSelf.decl = constraint_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.C.impls.Y.type.require.decl = require_decl @Z.WithSelf.C.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %C.loc9_17.1 impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc9: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %C.loc9_17.1: type = class_type @C, @C(constants.%Self.binding.as_type) [symbolic = %C.loc9_17.2 (constants.%C.237)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.C.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.C.impls.Y.type.require.%C.loc9_17.1 impls @Z.WithSelf.C.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.C.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %C.loc9_17.2: type = class_type @C, @C(%Self.binding.as_type) [symbolic = %C.loc9_17.2 (constants.%C.237)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.550) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.C.impls.Y.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %C.loc9_17.2 => constants.%C.237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_impls_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, @Y.WithSelf.%Y1 [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.550: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.550 [symbolic]
// CHECK:STDOUT:   %.Self: %Y.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @Y [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.WithSelf.decl = constraint_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %.loc7_19
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @Z.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.Self.ref: %Y.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_25: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %Y1.ref: %Y.assoc_type = name_ref Y1, @Y1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc7_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_32.2: type = converted %.loc7_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc7_19: type = where_expr %.Self [concrete = constants.%Y_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Y.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc7_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.%Self.as_type impls @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.%.loc7_19
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.550) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_require_impls_incomplete_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Z.WithSelf.decl = constraint_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_require_impls_without_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Z.type.7a8: type = generic_named_constaint_type @Z [concrete]
// CHECK:STDOUT:   %empty_struct: %Z.type.7a8 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.d68: type = facet_type <@Z, @Z(%T)> [symbolic]
// CHECK:STDOUT:   %Self.31c: %Z.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: %Z.type.7a8 = constraint_decl @Z [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @Z(%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc8_15.1)> [symbolic = %Z.type (constants.%Z.type.d68)]
// CHECK:STDOUT:   %Self.loc8_24.2: @Z.%Z.type (%Z.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc8_24.2 (constants.%Self.31c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc8_24.1: @Z.%Z.type (%Z.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc8_24.2 (constants.%Self.31c)]
// CHECK:STDOUT:     %Z.WithSelf.decl = constraint_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc8_24.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Y = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%T, constants.%Self.31c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_self_in_requirement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, @Y.WithSelf.%Y1 [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.550: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.550 [symbolic]
// CHECK:STDOUT:   %.Self: %Y.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @Y [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y where %impl.elem0 = %Self.binding.as_type> [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.WithSelf.decl = constraint_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type.loc10_11 impls %.loc10_19
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc10_11: type = facet_access_type @Z.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.Self.ref: %Y.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_25: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %Y1.ref: %Y.assoc_type = name_ref Y1, @Y1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:     %Self.as_type.loc10_31: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc10_31: type = converted %Self.ref, %Self.as_type.loc10_31 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc10_19: type = where_expr %.Self [symbolic = %Y_where.type (constants.%Y_where.type)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Y.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc10_31
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.%Self.as_type.loc10_11 impls @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.%.loc10_19
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y where constants.%impl.elem0 = %Self.binding.as_type> [symbolic = %Y_where.type (constants.%Y_where.type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.550) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_with_period_self_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Y.type.82b: type = generic_interface_type @Y [concrete]
// CHECK:STDOUT:   %Y.generic: %Y.type.82b = struct_value () [concrete]
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [concrete]
// CHECK:STDOUT:   %Self.550: %N.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.5cb: type = symbolic_binding_type Self, 0, %Self.550 [symbolic]
// CHECK:STDOUT:   %.Self.aac: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type.5a0: type = symbolic_binding_type .Self, %.Self.aac [symbolic_self]
// CHECK:STDOUT:   %Y.type.dfc: type = facet_type <@Y, @Y(%.Self.binding.as_type.5a0)> [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.455: type = facet_type <@Z where .Self impls @Y, @Y(%.Self.binding.as_type.5a0)> [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.820: type = facet_type <@Z where .Self impls @Y, @Y(%Self.binding.as_type.5cb)> [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type.8db: type = symbolic_binding_type .Self, %.Self.c39 [symbolic_self]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Z & @Y, @Y(%.Self.binding.as_type.8db)> [symbolic_self]
// CHECK:STDOUT:   %T.e40: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.e40 [symbolic]
// CHECK:STDOUT:   %Z_where.type.c36: type = facet_type <@Z where .Self impls @Y, @Y(%T.binding.as_type)> [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %N.decl: type = constraint_decl @N [concrete = constants.%N.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @N {
// CHECK:STDOUT:   %Self: %N.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %N.WithSelf.Self.binding.as_type.impls.Z_where.type.require.decl = require_decl @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %Z_where.type.loc10_19.1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @N.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.5cb)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.Self.ref.loc10_25: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.aac]
// CHECK:STDOUT:     %Y.ref: %Y.type.82b = name_ref Y, file.%Y.decl [concrete = constants.%Y.generic]
// CHECK:STDOUT:     %.Self.ref.loc10_39: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.aac]
// CHECK:STDOUT:     %.Self.as_type.loc10_44: type = facet_access_type %.Self.ref.loc10_39 [symbolic_self = constants.%.Self.binding.as_type.5a0]
// CHECK:STDOUT:     %.loc10_44: type = converted %.Self.ref.loc10_39, %.Self.as_type.loc10_44 [symbolic_self = constants.%.Self.binding.as_type.5a0]
// CHECK:STDOUT:     %Y.type: type = facet_type <@Y, @Y(constants.%.Self.binding.as_type.5a0)> [symbolic_self = constants.%Y.type.dfc]
// CHECK:STDOUT:     %.Self.as_type.loc10_25: type = facet_access_type %.Self.ref.loc10_25 [symbolic_self = constants.%.Self.binding.as_type.5a0]
// CHECK:STDOUT:     %.loc10_25: type = converted %.Self.ref.loc10_25, %.Self.as_type.loc10_25 [symbolic_self = constants.%.Self.binding.as_type.5a0]
// CHECK:STDOUT:     %.loc10_19: type = where_expr %.Self [symbolic_self = constants.%Z_where.type.455] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type
// CHECK:STDOUT:       requirement_impls %.loc10_25, %Y.type
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Z_where.type.loc10_19.1: type = facet_type <@Z where .Self impls @Y, @Y(constants.%Self.binding.as_type.5cb)> [symbolic = %Z_where.type.loc10_19.2 (constants.%Z_where.type.820)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require {
// CHECK:STDOUT:     require @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require.%Self.as_type impls @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require.%Z_where.type.loc10_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require(@N.%Self: %N.type) {
// CHECK:STDOUT:   %Self: %N.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.5cb)]
// CHECK:STDOUT:   %Z_where.type.loc10_19.2: type = facet_type <@Z where .Self impls @Y, @Y(%Self.binding.as_type)> [symbolic = %Z_where.type.loc10_19.2 (constants.%Z_where.type.820)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%Self.550) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.5cb
// CHECK:STDOUT:   %Z_where.type.loc10_19.2 => constants.%Z_where.type.820
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T.e40) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require(constants.%T.e40) {
// CHECK:STDOUT:   %Self => constants.%T.e40
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %Z_where.type.loc10_19.2 => constants.%Z_where.type.c36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_with_specific_period_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Y.type.82b: type = generic_interface_type @Y [concrete]
// CHECK:STDOUT:   %Y.generic: %Y.type.82b = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [concrete]
// CHECK:STDOUT:   %Self.550: %N.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.550 [symbolic]
// CHECK:STDOUT:   %.Self.aac: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.aac [symbolic_self]
// CHECK:STDOUT:   %C.646: type = class_type @C, @C(%.Self.binding.as_type) [symbolic_self]
// CHECK:STDOUT:   %Y.type.dfc: type = facet_type <@Y, @Y(%.Self.binding.as_type)> [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.a6d533.1: type = facet_type <@Z where %C.646 impls @Y, @Y(%.Self.binding.as_type)> [symbolic_self]
// CHECK:STDOUT:   %C.237: type = class_type @C, @C(%Self.binding.as_type) [symbolic]
// CHECK:STDOUT:   %Z_where.type.a6d533.2: type = facet_type <@Z where %C.237 impls @Y, @Y(%Self.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %T.ea3: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.ea3 [symbolic]
// CHECK:STDOUT:   %C.3e1: type = class_type @C, @C(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %Z_where.type.a6d533.3: type = facet_type <@Z where %C.3e1 impls @Y, @Y(%T.binding.as_type)> [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %N.decl: type = constraint_decl @N [concrete = constants.%N.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @N {
// CHECK:STDOUT:   %Self: %N.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %N.WithSelf.Self.binding.as_type.impls.Z_where.type.require.decl = require_decl @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %Z_where.type.loc10_19.1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @N.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.Self.ref.loc10_27: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.aac]
// CHECK:STDOUT:     %.Self.as_type.loc10_32: type = facet_access_type %.Self.ref.loc10_27 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_32: type = converted %.Self.ref.loc10_27, %.Self.as_type.loc10_32 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %C.loc10_32: type = class_type @C, @C(constants.%.Self.binding.as_type) [symbolic_self = constants.%C.646]
// CHECK:STDOUT:     %Y.ref: %Y.type.82b = name_ref Y, file.%Y.decl [concrete = constants.%Y.generic]
// CHECK:STDOUT:     %.Self.ref.loc10_42: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.aac]
// CHECK:STDOUT:     %.Self.as_type.loc10_47: type = facet_access_type %.Self.ref.loc10_42 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_47: type = converted %.Self.ref.loc10_42, %.Self.as_type.loc10_47 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %Y.type: type = facet_type <@Y, @Y(constants.%.Self.binding.as_type)> [symbolic_self = constants.%Y.type.dfc]
// CHECK:STDOUT:     %.loc10_19: type = where_expr %.Self [symbolic_self = constants.%Z_where.type.a6d533.1] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type
// CHECK:STDOUT:       requirement_impls %C.loc10_32, %Y.type
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Z_where.type.loc10_19.1: type = facet_type <@Z where constants.%C.237 impls @Y, @Y(constants.%Self.binding.as_type)> [symbolic = %Z_where.type.loc10_19.2 (constants.%Z_where.type.a6d533.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require {
// CHECK:STDOUT:     require @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require.%Self.as_type impls @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require.%Z_where.type.loc10_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require(@N.%Self: %N.type) {
// CHECK:STDOUT:   %Self: %N.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %C.loc10_19: type = class_type @C, @C(%Self.binding.as_type) [symbolic = %C.loc10_19 (constants.%C.237)]
// CHECK:STDOUT:   %Z_where.type.loc10_19.2: type = facet_type <@Z where %C.loc10_19 impls @Y, @Y(%Self.binding.as_type)> [symbolic = %Z_where.type.loc10_19.2 (constants.%Z_where.type.a6d533.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%Self.550) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %C.loc10_19 => constants.%C.237
// CHECK:STDOUT:   %Z_where.type.loc10_19.2 => constants.%Z_where.type.a6d533.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T.ea3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.Z_where.type.require(constants.%T.ea3) {
// CHECK:STDOUT:   %Self => constants.%T.ea3
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %C.loc10_19 => constants.%C.3e1
// CHECK:STDOUT:   %Z_where.type.loc10_19.2 => constants.%Z_where.type.a6d533.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
