// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/require_satisified.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/require_satisified.carbon

// --- fail_missing_require_for_concrete_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// () does not need to impl Z yet.
impl () as Y;

// () needs to impl Z at the start of the definition.
// CHECK:STDERR: fail_missing_require_for_concrete_type.carbon:[[@LINE+4]]:1: error: interface `Y` being implemented requires that `()` implements `Z` [RequireImplsNotImplemented]
// CHECK:STDERR: impl () as Y {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as Y {}

// --- fail_missing_require_for_symbolic_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// T does not need to impl Z yet.
impl forall [T:! type] T as Y;

// T needs to impl Z at the start of the definition.
// CHECK:STDERR: fail_missing_require_for_symbolic_type.carbon:[[@LINE+4]]:1: error: interface `Y` being implemented requires that `T` implements `Z` [RequireImplsNotImplemented]
// CHECK:STDERR: impl forall [T:! type] T as Y {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Y {}

// --- fail_missing_require_double.carbon
library "[[@TEST_NAME]]";

interface Z1(T:! type) {}
interface Z2(T:! type) {}

class A;
class B;

interface Y(U:! type) {
  extend require impls Z1(U) & Z2(B);
}

// CHECK:STDERR: fail_missing_require_double.carbon:[[@LINE+4]]:1: error: interface `Y(A)` being implemented requires that `()` implements `Z1(A) & Z2(B)` [RequireImplsNotImplemented]
// CHECK:STDERR: impl () as Y(A) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as Y(A) {}

// --- require_for_concrete_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// () does not need to impl Z yet.
impl () as Y;

// Now we know () impls Z.
impl () as Z;

// So no error here.
impl () as Y {}

impl () as Z {}

// --- require_for_symbolic_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// T does not need to impl Z yet.
impl forall [T:! type] T as Y;

// Now we know T impls Z.
impl forall [T:! type] T as Z;

// So no error here.
impl forall [T:! type] T as Y {}

impl forall [T:! type] T as Z {}

// --- require_for_symbolic_type_impl_matches_same_interface.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// This only matches T that impl Z so no error here.
impl forall [T:! Z] T as Y {}

// --- require_for_generic_interfaces.carbon
library "[[@TEST_NAME]]";

interface Z1(T:! type) {}
interface Z2(T:! type) {}

class A;
class B;

interface Y(U:! type) {
  extend require impls Z1(U) & Z2(B);
}

// () does not need to impl Z1(A) and Z2(B) yet.
impl () as Y(A);

// Now we know () impls Z1(A) and Z2(B).
impl () as Z1(A);
impl () as Z2(B);

// So no error here.
impl () as Y(A) {}

impl () as Z1(A) {}
impl () as Z2(B) {}

// --- fail_require_non_self_missing.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

class C;

interface Y(U:! type) {
  // `Self` must appear somewhere in the require decl, so it's in the interface
  // specific because we want to test a different self-type.
  require C impls Z(Self);
}

// Y requires that C impls Z(Self), but it does not do so.
// CHECK:STDERR: fail_require_non_self_missing.carbon:[[@LINE+4]]:1: error: interface `Y(())` being implemented requires that `C` implements `Z(())` [RequireImplsNotImplemented]
// CHECK:STDERR: impl () as Y(()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as Y(()) {}

// --- require_non_self.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {}

class C;

interface Y(U:! type) {
  // `Self` must appear somewhere in the require decl, so it's in the interface
  // specific because we want to test a different self-type.
  require C impls Z(Self);
}

impl C as Z(());

// Y requires that C impls Z(Self), which is done above.
impl () as Y(Self) {}

impl C as Z(()) {}
