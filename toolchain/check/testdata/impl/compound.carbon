// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/compound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/compound.carbon

// --- non-instance_success.carbon
library "[[@TEST_NAME]]";

interface NonInstance1 {
  fn F1();
}

impl {.a: ()} as NonInstance1 {
  fn F1() {}
}

fn NonInstanceCall1() {
  {.a: ()}.(NonInstance1.F1)();
}

// --- fail_non-instance.carbon
library "[[@TEST_NAME]]";

interface NonInstance2 {
  fn F2();
}

impl {.b: ()} as NonInstance2 {
  fn F2() {}
}

fn NonInstanceCall2(n: {.b: ()}) {
  // CHECK:STDERR:        ╭─┤ fail_non-instance.carbon:[[@LINE+11]]:3
  // CHECK:STDERR: ->     │   n.(NonInstance2.F2)();
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert non-type value of type `{.b: ()}` into
  // CHECK:STDERR:        ·   │  ┃        type implementing `NonInstance2`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `{.b: ()}` does not implement interface
  // CHECK:STDERR:        ·        │ `Core.ImplicitAs(NonInstance2)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  n.(NonInstance2.F2)();
}

// --- fail_non-instance_indirect.carbon
library "[[@TEST_NAME]]";

interface NonInstance3 {
  fn F3();
}

impl {.c: ()} as NonInstance3 {
  fn F3() {}
}

fn NonInstanceCallIndirect(p: {.c: ()}*) {
  // CHECK:STDERR:        ╭─┤ fail_non-instance_indirect.carbon:[[@LINE+11]]:3
  // CHECK:STDERR: ->     │   p->(NonInstance3.F3)();
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert non-type value of type `{.c: ()}` into
  // CHECK:STDERR:        ·   │  ┃        type implementing `NonInstance3`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `{.c: ()}` does not implement interface
  // CHECK:STDERR:        ·        │ `Core.ImplicitAs(NonInstance3)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  p->(NonInstance3.F3)();
}

// --- instance_success.carbon
library "[[@TEST_NAME]]";

interface Instance1 {
  fn G1(self);
}

impl {.d: ()} as Instance1 {
  fn G1(unused self) {}
}

fn InstanceCall(n: {.d: ()}) {
  n.(Instance1.G1)();
}

fn InstanceCallIndirect(p: {.d: ()}*) {
  p->(Instance1.G1)();
}

// --- fail_instance.carbon
library "[[@TEST_NAME]]";

interface Instance2 {
  fn G2(self);
}

impl {.e: ()} as Instance2 {
  fn G2(unused self) {}
}

fn InstanceCallFail() {
  // CHECK:STDERR:        ╭─┤ fail_instance.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   {.e: ()}.(Instance2.G2)();
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot access member of interface `Instance2` in type `type` that
  // CHECK:STDERR:        ·      ┃        does not implement that interface
  // CHECK:STDERR:        ·      ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  {.e: ()}.(Instance2.G2)();
}

// --- compound_lookup_uses_self_specific.carbon
library "[[@TEST_NAME]]";

interface X {
  let T: type;
}

// A generic impl for `U impls X`. Which is used inside Z.
impl forall [U: type] U as X where .T = () {}

interface Z {
  // Uses `U impls X` above, with a symbolic Self. It should not resolve to the
  // concrete value of `.T`.
  let V: Self.(X.T);
}

fn F(generic T: Z) {
  // Does name lookup `Z.V` which forms a specific of `Z`. It should use the
  // self specific, not a specific where the facet type `Z` is the Self of the
  // interface `Z`. If that mistake is made, the specific now has a concrete
  // Self, which resolves the `.Self.(X.T)` to a concrete value, which poisons
  // the next impl below.
  T.(Z.V);
}

// Set a concrete value for `Z.(X.T)`, which is not something we should have
// tried to access yet, unless the name lookup in F went wrong.
impl Z as X where .T = {} {}
