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
  // CHECK:STDERR: fail_implicit_self_no_extend_name_lookup_fails.carbon:[[@LINE+4]]:3: error: member name `YY` not found in `Z` [MemberNameNotFoundInInstScope]
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
  // CHECK:STDERR: fail_explicit_self_no_extend_name_lookup_fails.carbon:[[@LINE+4]]:3: error: member name `YY` not found in `Z` [MemberNameNotFoundInInstScope]
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

//@dump-sem-ir-begin
constraint Z {
  // CHECK:STDERR: fail_require_impls_incomplete_self.carbon:[[@LINE+7]]:17: error: facet type `Z` cannot be identified in `require` declaration [RequireImplsUnidentifiedFacetType]
  // CHECK:STDERR:   require impls Z;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR: fail_require_impls_incomplete_self.carbon:[[@LINE-4]]:1: note: constraint is currently being defined [NamedConstraintIncompleteWithinDefinition]
  // CHECK:STDERR: constraint Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  require impls Z;
}
//@dump-sem-ir-end

// --- fail_require_impls_incomplete_self_specific.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
constraint Z(T:! type) {
  // CHECK:STDERR: fail_require_impls_incomplete_self_specific.carbon:[[@LINE+7]]:19: error: facet type `Z(Self)` cannot be identified in `require` declaration [RequireImplsUnidentifiedFacetType]
  // CHECK:STDERR:   require T impls Z(Self);
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR: fail_require_impls_incomplete_self_specific.carbon:[[@LINE-4]]:1: note: constraint is currently being defined [NamedConstraintIncompleteWithinDefinition]
  // CHECK:STDERR: constraint Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require T impls Z(Self);
}
//@dump-sem-ir-end

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
  // CHECK:STDERR: fail_require_impls_without_self.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic parameter for each `interface` or `constraint` [RequireImplsMissingSelf]
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
  // CHECK:STDERR: fail_require_impls_without_self_in_one_interface.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic parameter for each `interface` or `constraint` [RequireImplsMissingSelf]
  // CHECK:STDERR:   require T impls Y(Self) & Y({});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require T impls Y(Self) & Y({});
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
  // CHECK:STDERR: fail_non_type_impls.carbon:[[@LINE+4]]:11: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
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

// CHECK:STDOUT: --- extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Y.YY.type: type = fn_type @Y.YY [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, @Y.%Y.YY.decl [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.550: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.550 [symbolic]
// CHECK:STDOUT:   %T: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.349: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Y [symbolic]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %T.binding.as_type, (%Y.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.bb4: type = fn_type_with_self_type %Y.YY.type, %Y.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.bb4 = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Y.YY(%Y.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.Self.binding.as_type.impls.Y.type.require.decl = require_decl @Z.Self.binding.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @Z.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: type = specific_constant @Z.Self.binding.as_type.impls.Y.type.require.%Y.ref, @Z.Self.binding.as_type.impls.Y.type.require(constants.%Self.550) [concrete = constants.%Y.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .YY = <poisoned>
// CHECK:STDOUT:   extend %.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.Self.binding.as_type.impls.Y.type.require.%Self.as_type impls @Z.Self.binding.as_type.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.Self.binding.as_type.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_6.2: %Z.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_6.1, @Y [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness)]
// CHECK:STDOUT:   %Y.facet.loc15: %Y.type = facet_value %T.binding.as_type, (%Y.lookup_impl_witness) [symbolic = %Y.facet.loc15 (constants.%Y.facet)]
// CHECK:STDOUT:   %.loc15_4.2: type = fn_type_with_self_type constants.%Y.YY.type, %Y.facet.loc15 [symbolic = %.loc15_4.2 (constants.%.bb4)]
// CHECK:STDOUT:   %impl.elem0.loc15_4.2: @F.%.loc15_4.2 (%.bb4) = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_4.2: <specific function> = specific_impl_function %impl.elem0.loc15_4.2, @Y.YY(%Y.facet.loc15) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc15: %Z.type = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc15_4.1: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %YY.ref.loc15: %Y.assoc_type = name_ref YY, @Y.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc15_4.1: @F.%.loc15_4.2 (%.bb4) = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc15_4.1: <specific function> = specific_impl_function %impl.elem0.loc15_4.1, @Y.YY(constants.%Y.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Y.YY.call.loc15: init %empty_tuple.type = call %specific_impl_fn.loc15_4.1()
// CHECK:STDOUT:     %T.ref.loc16: %Z.type = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     %YY.ref.loc16: %Y.assoc_type = name_ref YY, @Y.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc16: type = facet_access_type %T.ref.loc16 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Y.facet.loc16: %Y.type = facet_value %T.as_type.loc16, (constants.%Y.lookup_impl_witness) [symbolic = %Y.facet.loc15 (constants.%Y.facet)]
// CHECK:STDOUT:     %.loc16: %Y.type = converted %T.ref.loc16, %Y.facet.loc16 [symbolic = %Y.facet.loc15 (constants.%Y.facet)]
// CHECK:STDOUT:     %impl.elem0.loc16: @F.%.loc15_4.2 (%.bb4) = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc16: <specific function> = specific_impl_function %impl.elem0.loc16, @Y.YY(constants.%Y.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Y.YY.call.loc16: init %empty_tuple.type = call %specific_impl_fn.loc16()
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %YY.ref.loc18: %Y.assoc_type = name_ref YY, @Y.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc18: @F.%.loc15_4.2 (%.bb4) = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc18: <specific function> = specific_impl_function %impl.elem0.loc18, @Y.YY(constants.%Y.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Y.YY.call.loc18: init %empty_tuple.type = call %specific_impl_fn.loc18()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Self.binding.as_type.impls.Y.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.349
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Self.binding.as_type.impls.Y.type.require(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.Self.binding.as_type.impls.Y.type.require.decl = require_decl @Z.Self.binding.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @Z.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.Self.binding.as_type.impls.Y.type.require.%Self.as_type impls @Z.Self.binding.as_type.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.Self.binding.as_type.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Self.binding.as_type.impls.Y.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Self.binding.as_type.impls.Y.type.require(constants.%T) {
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.Self.binding.as_type.impls.Y.type.require.decl = require_decl @Z.Self.binding.as_type.impls.Y.type.require [concrete] {
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
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.Self.binding.as_type.impls.Y.type.require.%.loc9 impls @Z.Self.binding.as_type.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.Self.binding.as_type.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Self.binding.as_type.impls.Y.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Self.binding.as_type.impls.Y.type.require(constants.%T) {
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.C.impls.Y.type.require.decl = require_decl @Z.C.impls.Y.type.require [concrete] {
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
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.C.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.C.impls.Y.type.require.%C.loc9_17.1 impls @Z.C.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.C.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %C.loc9_17.2: type = class_type @C, @C(%Self.binding.as_type) [symbolic = %C.loc9_17.2 (constants.%C.237)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.C.impls.Y.type.require(constants.%Self.550) {
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
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, @Y.%Y1 [concrete]
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.Self.binding.as_type.impls.Y_where.type.require.decl = require_decl @Z.Self.binding.as_type.impls.Y_where.type.require [concrete] {
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
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.Self.binding.as_type.impls.Y_where.type.require {
// CHECK:STDOUT:     require @Z.Self.binding.as_type.impls.Y_where.type.require.%Self.as_type impls @Z.Self.binding.as_type.impls.Y_where.type.require.%.loc7_19
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.Self.binding.as_type.impls.Y_where.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Self.binding.as_type.impls.Y_where.type.require(constants.%Self.550) {
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_require_impls_incomplete_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_require_impls_incomplete_self_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.7a8: type = generic_named_constaint_type @Z [concrete]
// CHECK:STDOUT:   %empty_struct: %Z.type.7a8 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.d682d6.1: type = facet_type <@Z, @Z(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Z.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: %Z.type.7a8 = constraint_decl @Z [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @Z(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc4_14.1)> [symbolic = %Z.type (constants.%Z.type.d682d6.1)]
// CHECK:STDOUT:   %Self.loc4_24.2: @Z.%Z.type (%Z.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_24.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc4_24.1: @Z.%Z.type (%Z.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_24.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_24.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Z = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.binding.as_type) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_require_impls_without_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.7a8: type = generic_named_constaint_type @Z [concrete]
// CHECK:STDOUT:   %empty_struct: %Z.type.7a8 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.d68: type = facet_type <@Z, @Z(%T)> [symbolic]
// CHECK:STDOUT:   %Self.31c: %Z.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: %Z.type.7a8 = constraint_decl @Z [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc8_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @Z(%T.loc8_14.2: type) {
// CHECK:STDOUT:   %T.loc8_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc8_14.1)> [symbolic = %Z.type (constants.%Z.type.d68)]
// CHECK:STDOUT:   %Self.loc8_24.2: @Z.%Z.type (%Z.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc8_24.2 (constants.%Self.31c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc8_24.1: @Z.%Z.type (%Z.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc8_24.2 (constants.%Self.31c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc8_24.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Y = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_self_in_requirement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, @Y.%Y1 [concrete]
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = constraint_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   %Z.Self.binding.as_type.impls.Y_where.type.require.decl = require_decl @Z.Self.binding.as_type.impls.Y_where.type.require [concrete] {
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
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.Self.binding.as_type.impls.Y_where.type.require {
// CHECK:STDOUT:     require @Z.Self.binding.as_type.impls.Y_where.type.require.%Self.as_type.loc10_11 impls @Z.Self.binding.as_type.impls.Y_where.type.require.%.loc10_19
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.Self.binding.as_type.impls.Y_where.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y where constants.%impl.elem0 = %Self.binding.as_type> [symbolic = %Y_where.type (constants.%Y_where.type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Self.binding.as_type.impls.Y_where.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
