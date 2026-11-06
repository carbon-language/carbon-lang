// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/types.carbon

// --- todo_trivial_associated_entity.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F();
}

fn Test() {
  // TODO: fix crash
  // typeof(I.F) as Core.Destroy;
}

// --- todo_trivial_generic_interface.carbon

library "[[@TEST_NAME]]";

interface GenericInterface(T: type) {}

fn Test() {
  // TODO: fix crash
  // typeof(GenericInterface) as Core.Destroy;
}

// --- todo_trivial_generic_named_constraint.carbon

library "[[@TEST_NAME]]";

constraint GenericConstraint(T: type) {}

fn Test() {
  // TODO: fix crash
  // typeof(GenericConstraint) as Core.Destroy;
}

// --- todo_trivial_namespace.carbon

library "[[@TEST_NAME]]";

namespace N;

fn Test() {
  // TODO: fix crash
  // typeof(N) as Core.Destroy;
}

// --- todo_trivial_parameterised_class.carbon

library "[[@TEST_NAME]]";

class GenericClass(T: type) {}

fn Test() {
  // TODO: fix crash
  // typeof(GenericClass) as Core.Destroy;
}

// --- todo_trivial_unbound_element.carbon

library "[[@TEST_NAME]]";

class C {
  var x: i32;
}

fn Test() {
  // TODO: fix crash
  // typeof(C.x) as Core.Destroy;
}

// --- trivial_call.carbon

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
fn Test(generic T: Core.Destroy) {
  typeof(GenericClass(T)) as Core.Destroy;
}

// --- trivial_impl_witness_access.carbon

library "[[@TEST_NAME]]";

// An interface with an associated type that has no constraints. For a symbolic
// `T`, `T.U` is a type that isn't known to be destroyable.
interface HasUnconstrainedAssociatedType {
  let U: type;
}

fn Test(generic T: HasUnconstrainedAssociatedType) {
  typeof(T.U) as Core.Destroy;
}

// --- trivial_symbolic_binding.carbon

library "[[@TEST_NAME]]";

fn Test(generic T: type) {
  typeof(T) as Core.Destroy;
}

// --- todo_trivial_trivial_tuple_access.carbon

library "[[@TEST_NAME]]";

fn Test(unused generic T: (type, type)) {
  // TODO: fix crash
  // typeof(T.0) as Core.Destroy;
}

// --- todo_trivial_incomplete_type.carbon

library "[[@TEST_NAME]]";

class Incomplete;

fn Test() {
  typeof(Incomplete) as Core.Destroy;
}

// --- todo_trivial_abstract_class.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {}

fn Test() {
  typeof(Abstract) as Core.Destroy;
}
