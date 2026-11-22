// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/types_without_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/types_without_values.carbon

// --- fail_nodestroy_incomplete_class.carbon
library "[[@TEST_NAME]]";

class Incomplete;

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nodestroy_incomplete_class.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Incomplete as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Incomplete` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Destroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Incomplete as Core.TrivialDestroy;
}

// --- todo_fail_nodestroy_associated_entity.carbon
library "[[@TEST_NAME]]";

// TODO: add `AssociatedEntityType` to `IsBuiltinWithTrivialDestruction`.

interface I {
  fn F();
}

alias AssociatedEntityType = typeof(I.F);

fn Test() {
  // AssociatedEntityType as Core.TrivialDestroy;
}

// --- todo_fail_nodestroy_generic_interface.carbon
library "[[@TEST_NAME]]";

// TODO: add `GenericInterfaceType` to `IsBuiltinWithTrivialDestruction`.

interface GenericInterface(T: type) {}
alias GenericInterfaceType = typeof(GenericInterface);

fn Test() {
  // GenericInterfaceType as Core.TrivialDestroy;
}

// --- todo_fail_nodestroy_generic_named_constraint.carbon
library "[[@TEST_NAME]]";

// TODO: add `GenericNamedConstraintType` to `IsBuiltinWithTrivialDestruction`.

constraint GenericClass(T: type) {}
alias GenericNamedConstraintType = typeof(GenericClass);

fn Test() {
  // GenericNamedConstraintType as Core.TrivialDestroy;
}

// --- todo_fail_nodestroy_namespace.carbon
library "[[@TEST_NAME]]";

// TODO: add `NamespaceType` to `IsBuiltinWithTrivialDestruction`.

namespace N;
alias NamespaceType = typeof(N);

fn Test() {
  // NamespaceType as Core.TrivialDestroy;
}

// --- todo_fail_nodestroy_parameterised_class.carbon
library "[[@TEST_NAME]]";

// TODO: add `ParameterizedClassType` to `IsBuiltinWithTrivialDestruction`.

class ParameterizedClassType(T: type) {}

fn Test() {
  // ParameterizedClassType as Core.TrivialDestroy;
}

// --- todo_fail_nodestroy_unbound_element.carbon
library "[[@TEST_NAME]]";

// TODO: add `UnboundElementType` to `IsBuiltinWithTrivialDestruction`.

class HasTrivialDestruction {}

class C {
  var x: HasTrivialDestruction;
}

alias UnboundElementType = typeof(C.x);

fn Test() {
  // UnboundElementType as Core.TrivialDestroy;
}

// --- todo_fail_nodestroy_function_with_self_type.carbon
library "[[@TEST_NAME]]";

// TODO: add `FunctionTypeWithSelfType` to `IsBuiltinWithTrivialDestruction`.

interface I { fn F(self); }
impl () as I { fn F(unused self) {} }
alias FunctionTypeWithSelfType = typeof((() as I).F);

fn Test() {
  // FunctionTypeWithSelfType as Core.TrivialDestroy;
}

// --- fail_nodestroy_tuple_access_with_facet_type_first.carbon
library "[[@TEST_NAME]]";

// TODO: add `TupleAccess` to `IsBuiltinWithTrivialDestruction`.

class HasTrivialDestruction {}

fn Test(generic T: (type, HasTrivialDestruction)) {
  // alias TupleAccessFirst = T.0;
  // TupleAccessFirst as Core.TrivialDestroy;

  alias TupleAccessSecond = T.1;
  // CHECK:STDERR:        ╭─┤ fail_nodestroy_tuple_access_with_facet_type_first.carbon:[[@LINE+11]]:3
  // CHECK:STDERR: ->     │   TupleAccessSecond as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert non-type value of type `HasTrivialDestruction` into
  // CHECK:STDERR:        ·   │  ┃        type implementing `Core.Destroy` with `as`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `HasTrivialDestruction` does not implement interface
  // CHECK:STDERR:        ·        │ `Core.As(Core.Destroy)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  TupleAccessSecond as Core.TrivialDestroy;
}

// --- fail_nodestroy_tuple_access_with_facet_type_second.carbon
library "[[@TEST_NAME]]";

// TODO: add `TupleAccess` to `IsBuiltinWithTrivialDestruction`.

class HasTrivialDestruction {}

fn Test(generic T: (HasTrivialDestruction, type)) {
  alias TupleAccessFirst = T.0;
  // CHECK:STDERR:        ╭─┤ fail_nodestroy_tuple_access_with_facet_type_second.carbon:[[@LINE+11]]:3
  // CHECK:STDERR: ->     │   TupleAccessFirst as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert non-type value of type `HasTrivialDestruction` into
  // CHECK:STDERR:        ·   │  ┃        type implementing `Core.Destroy` with `as`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `HasTrivialDestruction` does not implement interface
  // CHECK:STDERR:        ·        │ `Core.As(Core.Destroy)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  TupleAccessFirst as Core.TrivialDestroy;

  // alias TupleAccessSecond = T.1;
  // TupleAccessSecond as Core.TrivialDestroy;
}

// --- fail_nodestroy_symbolic_type.carbon
library "[[@TEST_NAME]]";

fn Test(generic T: type) {
  // CHECK:STDERR:        ╭─┤ fail_nodestroy_symbolic_type.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   T as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ━━━━━┳━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·        ┗━━┫ error: cannot convert type `T` into type implementing `Core.Destroy`
  // CHECK:STDERR:        ·           ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  T as Core.TrivialDestroy;
}
