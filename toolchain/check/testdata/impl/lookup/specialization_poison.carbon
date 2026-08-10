// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/specialization_poison.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/specialization_poison.carbon

// --- fail_final_poisoned_concrete_query_in_specific.carbon
library "[[@TEST_NAME]]";

interface I {
  let T: type;
  fn F(self) -> T;
}

impl forall [U: type] U as I where .T = () {
  fn F(unused self) -> () { return (); }
}

// CHECK:STDERR: fail_final_poisoned_concrete_query_in_specific.carbon:[[@LINE+1]]:24-30: error: found `impl` that would change the result of an earlier use of `C* as I` [PoisonedImplLookupConcreteResult]
fn H[W: type](v: W) -> W.(I.T) {
  return v.(I.F)();
}

class C { adapt (); }

fn G(p: C*) -> () {
  // This concrete impl lookup query poisons any further specializations.
  return H(p);
}

// CHECK:STDERR: fail_final_poisoned_concrete_query_in_specific.carbon:[[@LINE+3]]:1-27: note: the use would select the `impl` here but it was not found yet [PoisonedImplLookupConcreteResultNoteBadImpl]
// CHECK:STDERR: fail_final_poisoned_concrete_query_in_specific.carbon:[[@LINE-17]]:1-44: note: the use had selected the `impl` here [PoisonedImplLookupConcreteResultNotePreviousImpl]
// CHECK:STDERR:
impl C* as I where .T = C {
  fn F(self) -> C { return *self; }
}

// --- fail_final_poisoned_concrete_query_in_generic.carbon
library "[[@TEST_NAME]]";

interface I {
  let T: type;
}

impl forall [U: type] U as I where .T = () {}

class C {}

fn H[W: type](unused v: W) {
  // This concrete impl lookup query poisons any further specializations.
  // CHECK:STDERR: fail_final_poisoned_concrete_query_in_generic.carbon:[[@LINE+1]]:17-23: error: found `impl` that would change the result of an earlier use of `C as I` [PoisonedImplLookupConcreteResult]
  let unused a: C.(I.T) = ();
}

// CHECK:STDERR: fail_final_poisoned_concrete_query_in_generic.carbon:[[@LINE+3]]:1-26: note: the use would select the `impl` here but it was not found yet [PoisonedImplLookupConcreteResultNoteBadImpl]
// CHECK:STDERR: fail_final_poisoned_concrete_query_in_generic.carbon:[[@LINE-11]]:1-44: note: the use had selected the `impl` here [PoisonedImplLookupConcreteResultNotePreviousImpl]
// CHECK:STDERR:
impl C as I where .T = C {}

// --- fail_final_poisoned_concrete_query_nested_type_in_self.carbon
library "[[@TEST_NAME]]";

interface I {
  let T: type;
  fn F(self) -> T;
}

class C(U: type) {}

impl forall [U: type] C(U) as I where .T = () {
  fn F(unused self) -> () { return (); }
}

fn G(c: C(())) -> () {
  // This concrete impl lookup query poisons any further specializations.
  // CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_self.carbon:[[@LINE+1]]:10-16: error: found `impl` that would change the result of an earlier use of `C(()) as I` [PoisonedImplLookupConcreteResult]
  return c.(I.F)();
}

// CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_self.carbon:[[@LINE+3]]:1-31: note: the use would select the `impl` here but it was not found yet [PoisonedImplLookupConcreteResultNoteBadImpl]
// CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_self.carbon:[[@LINE-11]]:1-47: note: the use had selected the `impl` here [PoisonedImplLookupConcreteResultNotePreviousImpl]
// CHECK:STDERR:
impl C(()) as I where .T = {} {
  fn F(unused self) -> {} { return {}; }
}

// --- fail_final_poisoned_concrete_query_nested_type_in_interface.carbon
library "[[@TEST_NAME]]";

interface I(U: type) {
  let T: type;
  fn F(self) -> T;
}

impl forall [U: type] U as I(U) where .T = () {
  fn F(unused self) -> () { return (); }
}

class C {}

fn G(c: C) -> () {
  // This concrete impl lookup query poisons any further specializations.
  // CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_interface.carbon:[[@LINE+1]]:10-19: error: found `impl` that would change the result of an earlier use of `C as I(C)` [PoisonedImplLookupConcreteResult]
  return c.(I(C).F)();
}

// CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_interface.carbon:[[@LINE+3]]:1-30: note: the use would select the `impl` here but it was not found yet [PoisonedImplLookupConcreteResultNoteBadImpl]
// CHECK:STDERR: fail_final_poisoned_concrete_query_nested_type_in_interface.carbon:[[@LINE-13]]:1-47: note: the use had selected the `impl` here [PoisonedImplLookupConcreteResultNotePreviousImpl]
// CHECK:STDERR:
impl C as I(C) where .T = {} {
  fn F(unused self) -> {} { return {}; }
}

// --- todo_fail_final_poisoned_by_generic_query.carbon
library "[[@TEST_NAME]]";

interface I {
  let T: type;
  fn F(self) -> T;
}

impl forall [U: type] U as I where .T = () {
  fn F(unused self) -> () { return (); }
}

fn H[W: type](v: W) -> W.(I.T) {
  return v.(I.F)();
}

// This function could return a concrete `X` if it saw the `final` impl below.
fn G[X: type](p: X*) -> (X*).(I.T) {
  return H(p);
}

// TODO: This should be diagnosed as poisoned, as H() uses an associated
// constant from the `impl` which was treated as a symbolic, but this would
// change it to be a concrete type.
final impl forall [V: Core.Copy] V* as I where .T = V {
  fn F(self) -> V { return *self; }
}

// --- fail_todo_poisoned_concrete_lookup_from_eval_fn.carbon
library "[[@TEST_NAME]]";

fn F() {
  class D {}

  // CHECK:STDERR: fail_todo_poisoned_concrete_lookup_from_eval_fn.carbon:[[@LINE+2]]:22-25: error: semantics TODO: `HandleAutoTypeLiteral` [SemanticsTodo]
  // CHECK:STDERR:
  musteval fn G() -> auto {
    class C(T: type) {}
    impl forall [T: type] C(T) as Z {}
    C(D) as Z;
    return C(D);
  }

  // TODO: error: found `impl` that would change the result of an earlier use of `C(D) as Z`.
  impl G() as Z {}
}
