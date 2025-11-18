// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/fail_incomplete.carbon

// --- fail_incomplete_interface.carbon
library "[[@TEST_NAME]]";

interface X;
class C {}

// Requires X identified.
impl C as X;

// Requires X complete.
// CHECK:STDERR: fail_incomplete_interface.carbon:[[@LINE+7]]:1: error: definition of impl as incomplete facet type `X` [ImplAsIncompleteFacetTypeDefinition]
// CHECK:STDERR: impl C as X {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_incomplete_interface.carbon:[[@LINE-10]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface X;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
impl C as X {}

// --- fail_incomplete_interface_without_forward_decl.carbon
library "[[@TEST_NAME]]";

interface X;
class C {}

// Requires X complete.
// CHECK:STDERR: fail_incomplete_interface_without_forward_decl.carbon:[[@LINE+7]]:1: error: definition of impl as incomplete facet type `X` [ImplAsIncompleteFacetTypeDefinition]
// CHECK:STDERR: impl C as X {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_incomplete_interface_without_forward_decl.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface X;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
impl C as X {}

// --- fail_unidentified_constraint.carbon
library "[[@TEST_NAME]]";

constraint X;
class C {}

// Requires X identified.
// CHECK:STDERR: fail_unidentified_constraint.carbon:[[@LINE+7]]:1: error: facet type `X` cannot be identified in `impl as` [ImplOfUnidentifiedFacetType]
// CHECK:STDERR: impl C as X;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_unidentified_constraint.carbon:[[@LINE-7]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
// CHECK:STDERR: constraint X;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C as X;

// --- nested_require_incomplete_interface.carbon
library "[[@TEST_NAME]]";

interface Z;
constraint Y {
  require impls Z;
}
interface X {
  require impls Y;
}

class C {}

// Requires X identified.
impl C as X;

// Requires X complete.
impl C as X {}

// --- fail_incomplete_through_constraint.carbon
library "[[@TEST_NAME]]";

interface Z;
constraint Y {
  extend require impls Z;
}

class C {}

// Requires Y identified.
impl C as Y;

// Requires Y complete.
// CHECK:STDERR: fail_incomplete_through_constraint.carbon:[[@LINE+7]]:1: error: definition of impl as incomplete facet type `Y` [ImplAsIncompleteFacetTypeDefinition]
// CHECK:STDERR: impl C as Y {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_incomplete_through_constraint.carbon:[[@LINE-14]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
impl C as Y {}

// --- fail_impl_lookup_incomplete.carbon
library "[[@TEST_NAME]]";

constraint Z;

fn AsZ(T:! Z) {}

fn F() {
  // Requires Z identified.
  // CHECK:STDERR: fail_impl_lookup_incomplete.carbon:[[@LINE+10]]:3: error: facet type `Z` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   AsZ(());
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_impl_lookup_incomplete.carbon:[[@LINE-9]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint Z;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_lookup_incomplete.carbon:[[@LINE-10]]:8: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn AsZ(T:! Z) {}
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  AsZ(());
}
