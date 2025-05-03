// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --no-dump-sem-ir --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/unused_generic_binding.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/unused_generic_binding.carbon

// --- fail_no_binding_used.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}

impl forall [U:! type] U as I where .T = () {}

class C {}

// This impl can never deduce its implicit generic parameter `V`, which
// should be diagnosed.
//
// CHECK:STDERR: fail_no_binding_used.carbon:[[@LINE+4]]:13: error: `impl` with unused generic binding [ImplUnusedBinding]
// CHECK:STDERR: impl forall [V:! type] C as I where .T = {.unmatched: ()} {}
// CHECK:STDERR:             ^~~~~~~~~~
// CHECK:STDERR:
impl forall [V:! type] C as I where .T = {.unmatched: ()} {}

fn F() {
  // This shows that the first impl declaration was matched even though the
  // second is more specific. But since it has an unused generic parameter, it
  // will never match. If the second was matched, this line would not type
  // check.
  let x: C.(I.T) = ();
}

// --- fail_one_binding_unused.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}

impl forall [U:! type] U as I where .T = () {}

class C {}

// This impl can never deduce its implicit generic parameter `W`, which
// should be diagnosed.
// https://discord.com/channels/655572317891461132/941071822756143115/1354563497731686550
//
// CHECK:STDERR: fail_one_binding_unused.carbon:[[@LINE+4]]:13: error: `impl` with unused generic binding [ImplUnusedBinding]
// CHECK:STDERR: impl forall [V:! type, W:! type] V as I where .T = {.unmatched: ()} {}
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [V:! type, W:! type] V as I where .T = {.unmatched: ()} {}

fn F() {
  // This shows that the first impl declaration was matched even though the
  // second is more specific. If the second was matched, it would not type
  // check.
  let x: C.(I.T) = ();
}

// --- fail_inherited_binding_unused.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}

impl forall [U:! type] U as I where .T = () {}

class C(U:! type) {
  // This impl can never deduce its inherited generic binding `U`, which
  // should be diagnosed.
  //
  // TODO: We should point the diagnostic at the binding `U` in C.
  //
  // CHECK:STDERR: fail_inherited_binding_unused.carbon:[[@LINE+4]]:3: error: `impl` with unused generic binding [ImplUnusedBinding]
  // CHECK:STDERR:   impl C(()) as I where .T = {.unmatched: ()} {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl C(()) as I where .T = {.unmatched: ()} {}
}

fn F() {
  let x: C(()).(I.T) = ();
}

// --- fail_inherited_binding_in_forall_unused.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}

impl forall [U:! type] U as I where .T = () {}

class C(U:! type) {
  // This impl can never deduce its inherited generic binding `U`, which
  // should be diagnosed.
  //
  // TODO: We should point the diagnostic at the binding `U` in C.
  //
  // CHECK:STDERR: fail_inherited_binding_in_forall_unused.carbon:[[@LINE+4]]:15: error: `impl` with unused generic binding [ImplUnusedBinding]
  // CHECK:STDERR:   impl forall [V:! type] C(V) as I where .T = {.unmatched: ()} {}
  // CHECK:STDERR:               ^~~~~~~~~~
  // CHECK:STDERR:
  impl forall [V:! type] C(V) as I where .T = {.unmatched: ()} {}
}

fn F() {
  let x: C(()).(I.T) = ();
}

// --- inherited_binding_used.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}

impl forall [U:! type] U as I where .T = {.unmatched: ()} {}

class C(U:! type) {
  // `U` can be deduced here, so no diagnostic issued.
  impl C(U) as I where .T = () {}
}

fn F() {
  // The `impl` inside C is matched here.
  let x: C(()).(I.T) = ();
}

// --- inherited_binding_in_forall_used.carbon
library "[[@TEST_NAME]]";

interface I(W:! type) {
  let T:! type;
}

impl forall [U:! type] U as I(U) where .T = {.unmatched: ()} {}

class C(U:! type) {
  // `U` can be deduced here, so no diagnostic issued.
  impl forall [V:! type] C(U) as I(V) where .T = () {}
}

fn F() {
  // The `impl` inside C is matched here.
  let x: C(()).(I({}).T) = ();
}
