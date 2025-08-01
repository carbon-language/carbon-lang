// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/early_rewrites.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/early_rewrites.carbon

// Tests where `ImplWitnessAccess` on the RHS of a rewrite constraint is used in
// other constraints within the same facet type in ways that require the RHS to
// be evaluated to a concrete value before the facet type is resolved (such as
// by passing it as a parameter to a generic class).

// --- nested_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

interface J {
  let J1:! (Z where .Z1 = ()) where .Z2 = C(.Self);
}

// --- fail_todo_earlier_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

// TODO: Should this pass? Or do we need a `where` between .Z1 and .Z2 to make
// .Z1's value visible to .Z2?
// See https://github.com/carbon-language/carbon-lang/issues/5884.
interface J {
  // CHECK:STDERR: fail_todo_earlier_constraint_visible_in_type_parameter.carbon:[[@LINE+7]]:39: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   let J1:! Z where .Z1 = () and .Z2 = C(.Self);
  // CHECK:STDERR:                                       ^~~~~~~~
  // CHECK:STDERR: fail_todo_earlier_constraint_visible_in_type_parameter.carbon:[[@LINE-9]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  let J1:! Z where .Z1 = () and .Z2 = C(.Self);
}

// --- fail_todo_later_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}

class C(T:! Z where .Z1 = ()) {}

// TODO: Should this pass? Or do we need a `where` between .Z1 and .Z2 to make
// .Z1's value visible to .Z2?
// See https://github.com/carbon-language/carbon-lang/issues/5884.
interface J {
  // CHECK:STDERR: fail_todo_later_constraint_visible_in_type_parameter.carbon:[[@LINE+7]]:26: error: cannot convert type `.Self` that implements `Z` into type implementing `Z where .(Z.Z1) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   let J1:! Z where .Z2 = C(.Self) and .Z1 = ();
  // CHECK:STDERR:                          ^~~~~~~~
  // CHECK:STDERR: fail_todo_later_constraint_visible_in_type_parameter.carbon:[[@LINE-9]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Z where .Z1 = ()) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  let J1:! Z where .Z2 = C(.Self) and .Z1 = ();
}

// --- fail_todo_resolved_constraint_visible_in_type_parameter.carbon
library "[[@TEST_NAME]]";

interface Y {
  let Y1:! type;
}

interface Z {
  let Z1:! Y;
  let Z2:! type;
  let Z3:! type;
}

interface Tuple {}
impl () as Tuple {}
class C(T:! Tuple) {}

// TODO: Should this pass? Or do we need a `where` between .Z2 and .Z3 to make
// .Z2's value visible to .Z3?
// See https://github.com/carbon-language/carbon-lang/issues/5884.
interface J {
  // CHECK:STDERR: fail_todo_resolved_constraint_visible_in_type_parameter.carbon:[[@LINE+7]]:80: error: cannot convert type `.(Z.Z2)` into type implementing `Tuple` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   let J1:! (Z & Y where .Y1 = ()) where .Z1 = .Self and .Z2 = .Z1.Y1 and .Z3 = C(.Z2);
  // CHECK:STDERR:                                                                                ^~~~~~
  // CHECK:STDERR: fail_todo_resolved_constraint_visible_in_type_parameter.carbon:[[@LINE-9]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Tuple) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  let J1:! (Z & Y where .Y1 = ()) where .Z1 = .Self and .Z2 = .Z1.Y1 and .Z3 = C(.Z2);
}
