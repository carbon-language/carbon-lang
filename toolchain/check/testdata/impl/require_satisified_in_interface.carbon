// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/require_satisified_in_interface.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/require_satisified_in_interface.carbon

// --- fail_missing_extend_require_for_concrete_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// () does not need to impl Z yet.
impl () as Y;

// () needs to impl Z at the start of the definition.
// CHECK:STDERR: fail_missing_extend_require_for_concrete_type.carbon:[[@LINE+4]]:1: error: interface `Y` being implemented requires that `()` implements `Z` [RequireImplsNotImplemented]
// CHECK:STDERR: impl () as Y {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as Y {}

// --- fail_missing_require_for_concrete_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  require impls Z;
}

// () does not need to impl Z yet.
impl () as Y;

// () needs to impl Z at the start of the definition.
// CHECK:STDERR: fail_missing_require_for_concrete_type.carbon:[[@LINE+4]]:1: error: interface `Y` being implemented requires that `()` implements `Z` [RequireImplsNotImplemented]
// CHECK:STDERR: impl () as Y {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as Y {}

// --- fail_missing_extend_require_for_symbolic_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  extend require impls Z;
}

// T does not need to impl Z yet.
impl forall [T:! type] T as Y;

// T needs to impl Z at the start of the definition.
// CHECK:STDERR: fail_missing_extend_require_for_symbolic_type.carbon:[[@LINE+4]]:1: error: interface `Y` being implemented requires that `T` implements `Z` [RequireImplsNotImplemented]
// CHECK:STDERR: impl forall [T:! type] T as Y {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Y {}

// --- fail_missing_require_for_symbolic_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  require impls Z;
}

// T does not need to impl Z yet.
impl forall [T:! type] T as Y;

// T needs to impl Z at the start of the definition.
// CHECK:STDERR: fail_missing_require_for_symbolic_type.carbon:[[@LINE+4]]:1: error: interface `Y` being implemented requires that `T` implements `Z` [RequireImplsNotImplemented]
// CHECK:STDERR: impl forall [T:! type] T as Y {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Y {}

// --- fail_missing_extend_require_double.carbon
library "[[@TEST_NAME]]";

interface Z1(T:! type) {}
interface Z2(T:! type) {}

class A;
class B;

interface Y(U:! type) {
  extend require impls Z1(U) & Z2(B);
}

// CHECK:STDERR: fail_missing_extend_require_double.carbon:[[@LINE+4]]:1: error: interface `Y(A)` being implemented requires that `()` implements `Z1(A) & Z2(B)` [RequireImplsNotImplemented]
// CHECK:STDERR: impl () as Y(A) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as Y(A) {}

// --- fail_missing_require_double.carbon
library "[[@TEST_NAME]]";

interface Z1(T:! type) {}
interface Z2(T:! type) {}

class A;
class B;

interface Y(U:! type) {
  require impls Z1(U) & Z2(B);
}

// CHECK:STDERR: fail_missing_require_double.carbon:[[@LINE+4]]:1: error: interface `Y(A)` being implemented requires that `()` implements `Z1(A) & Z2(B)` [RequireImplsNotImplemented]
// CHECK:STDERR: impl () as Y(A) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as Y(A) {}

// --- require_for_concrete_type.carbon
library "[[@TEST_NAME]]";

interface Z1 {}
interface Z2 {}

interface Y {
  extend require impls Z1;
  require impls Z2;
}

// () does not need to impl Z1/Z2 yet.
impl () as Y;

// Now we know () impls Z1 and Z2.
impl () as Z1;
impl () as Z2;

// So no error here.
impl () as Y {}

impl () as Z1 {}
impl () as Z2 {}

// --- require_for_symbolic_type.carbon
library "[[@TEST_NAME]]";

interface Z1 {}
interface Z2 {}

interface Y {
  extend require impls Z1;
  require impls Z2;
}

// T does not need to impl Z1/Z2 yet.
impl forall [T:! type] T as Y;

// Now we know T impls Z1 and Z2.
impl forall [T:! type] T as Z1;
impl forall [T:! type] T as Z2;

// So no error here.
impl forall [T:! type] T as Y {}

impl forall [T:! type] T as Z1 {}
impl forall [T:! type] T as Z2 {}

// --- require_for_symbolic_type_impl_matches_same_interface.carbon
library "[[@TEST_NAME]]";

interface Z1 {}
interface Z2 {}

interface Y {
  extend require impls Z1;
  require impls Z2;
}

// This only matches T that impl Z1 and Z2 so no error here.
impl forall [T:! Z1 & Z2] T as Y {}

// --- require_for_generic_interfaces.carbon
library "[[@TEST_NAME]]";

interface Z1(T:! type) {}
interface Z2(T:! type) {}

class A;
class B;

interface Y(U:! type) {
  extend require impls Z1(U);
  require impls Z2(B);
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

// --- impl_requires_itself_cycle.carbon
library "[[@TEST_NAME]]";

interface A { fn AA(); }
interface B {
  require impls A;
}
interface C {
  require impls B;
}

impl forall [T:! B] T as A { fn AA() {} }

// When we get to the definition we check that anything B requires is satisfied.
// - The interface B requires A, so we must check T impls A.
// - The impl for A requires T impls B.
// - This impl is what provides T impls B.
impl forall [T:! C] T as B {}

fn F(T:! C) {
  // If things go wrong, we find the impl `T as B`, but inside its definition is
  // a lookup for `T as A`, which (through deducing `T`) includes a lookup for
  // `T as B`. This creates a cycle of evaluating `T as B` recursively.
  //
  // This was solved by doing the check for requirements of `B` outside the
  // definition of `T as B`.
  // https://discord.com/channels/655572317891461132/941071822756143115/1463189087598022861
  T.(A.AA)();
}

// --- fail_impl_requires_itself_cycle_with_monomorphization_error.carbon
library "[[@TEST_NAME]]";

class W(T:! type) {
  adapt {};
}

interface A(N:! Core.IntLiteral) {
  fn AA() -> W(array((), N));
}
interface B(N:! Core.IntLiteral) {
  require impls A(N);
}
interface C(N:! Core.IntLiteral) {
  require impls B(N);
}

impl forall [N:! Core.IntLiteral, T:! B(N)] T as A(N) {
  fn AA() -> W(array((), N)) { return {} as W(array((), N)); }
}

impl forall [N:! Core.IntLiteral, T:! C(N)] T as B(N) {
  // The definition here does not contain the lookups verifying that C(N) impls
  // `A(N)`, so they do not get re-evaluated for a specific `N`. That doesn't
  // prevent us from producing a reasonable diagnostic when that `N` causes an
  // error in the specific use of this impl, which we use to get from `C(N)` to
  // `A(N)` (in the deduction of `T` in the impl as `A(N)`). The error just
  // happens where we use `N` in `A(N)` instead of inside the verification that
  // `T as B(N)` implies `T as A(N)`.
}

fn F(T:! C(-1)) {
  // CHECK:STDERR: fail_impl_requires_itself_cycle_with_monomorphization_error.carbon:[[@LINE+7]]:3: error: unable to monomorphize specific `AA()` [ResolvingSpecificHere]
  // CHECK:STDERR:   T.(A(-1).AA)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_requires_itself_cycle_with_monomorphization_error.carbon:[[@LINE-27]]:26: note: array bound of -1 is negative [ArrayBoundNegative]
  // CHECK:STDERR:   fn AA() -> W(array((), N));
  // CHECK:STDERR:                          ^
  // CHECK:STDERR:
  T.(A(-1).AA)();
}
