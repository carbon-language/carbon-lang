// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/types.carbon

// --- todo_trivial_associated_entity.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F();
}

fn Test() {
  // TODO: fix crash
  // typeof(I.F) as Core.SubobjectDestroy;
  // typeof(I.F) as Core.Destroy;
  // typeof(I.F) as Core.TrivialDestroy;
}

// --- todo_trivial_generic_interface.carbon
library "[[@TEST_NAME]]";

interface GenericInterface(T: type) {}

fn Test() {
  // TODO: fix crash
  // typeof(GenericInterface) as Core.SubobjectDestroy;
  // typeof(GenericInterface) as Core.Destroy;
  // typeof(GenericInterface) as Core.TrivialDestroy;
}

// --- todo_trivial_generic_named_constraint.carbon
library "[[@TEST_NAME]]";

constraint GenericConstraint(T: type) {}

fn Test() {
  // TODO: fix crash
  // typeof(GenericConstraint) as Core.SubobjectDestroy;
  // typeof(GenericConstraint) as Core.Destroy;
  // typeof(GenericConstraint) as Core.TrivialDestroy;
}

// --- todo_trivial_namespace.carbon
library "[[@TEST_NAME]]";

namespace N;

fn Test() {
  // TODO: fix crash
  // typeof(N) as Core.SubobjectDestroy;
  // typeof(N) as Core.Destroy;
  // typeof(N) as Core.TrivialDestroy;
}

// --- todo_trivial_parameterised_class.carbon
library "[[@TEST_NAME]]";

class GenericClass(T: type) {}

fn Test() {
  // TODO: fix crash
  // typeof(GenericClass) as Core.SubobjectDestroy;
  // typeof(GenericClass) as Core.Destroy;
  // typeof(GenericClass) as Core.TrivialDestroy;
}

// --- todo_trivial_unbound_element.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class C {
  var x: HasTrivialDestruction;
}

fn Test() {
  // TODO: fix crash
  // typeof(C.x) as Core.SubobjectDestroy;
  // typeof(C.x) as Core.Destroy;
  // typeof(C.x) as Core.TrivialDestroy;
}

// --- todo_trivial_call.carbon
library "[[@TEST_NAME]]";

// A function that returns a type. A call to it with a symbolic argument
// can't be resolved until the argument is known.
eval fn Identity(generic T: type) -> type { return T; }

// A generic class with a member whose type is a call to `Identity`.
class GenericClass(T: type) {
  var t: Identity(T);
}

// `GenericClass(T)` is used instead of `Identity(T)`, because a check that
// uses `Identity(T)` directly waits until `T` is known.
fn Test(generic T: type) {
  typeof(GenericClass(T)) as Core.SubobjectDestroy;
  typeof(GenericClass(T)) as Core.Destroy;
  typeof(GenericClass(T)) as Core.TrivialDestroy;
}

// --- todo_trivial_impl_witness_access.carbon
library "[[@TEST_NAME]]";

// An interface with an associated type that has no constraints. For a symbolic
// `T`, `T.U` is a type that isn't known to be destroyable.
interface HasUnconstrainedAssociatedType {
  let U: type;
}

fn Test(generic T: HasUnconstrainedAssociatedType) {
  typeof(T.U) as Core.SubobjectDestroy;
  typeof(T.U) as Core.Destroy;
  typeof(T.U) as Core.TrivialDestroy;
}

// --- todo_trivial_symbolic_binding.carbon
library "[[@TEST_NAME]]";

fn Test(generic T: type) {
  typeof(T) as Core.SubobjectDestroy;
  typeof(T) as Core.Destroy;
  typeof(T) as Core.TrivialDestroy;
}

// --- todo_trivial_trivial_tuple_access.carbon
library "[[@TEST_NAME]]";

fn Test(unused generic T: (type, type)) {
  // TODO: fix crash
  // typeof(T.0) as Core.SubobjectDestroy;
  // typeof(T.0) as Core.Destroy;
  // typeof(T.0) as Core.TrivialDestroy;
}

// --- todo_trivial_incomplete_type.carbon
library "[[@TEST_NAME]]";

class Incomplete;

fn Test() {
  typeof(Incomplete) as Core.SubobjectDestroy;
  typeof(Incomplete) as Core.Destroy;
  typeof(Incomplete) as Core.TrivialDestroy;
}

// --- todo_trivial_abstract_class.carbon
library "[[@TEST_NAME]]";

abstract class Abstract {}

fn Test() {
  typeof(Abstract) as Core.Destroy;
  typeof(Abstract) as Core.TrivialDestroy;
}
