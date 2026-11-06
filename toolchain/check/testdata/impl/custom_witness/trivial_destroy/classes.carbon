// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/classes.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/classes.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

// Use when mixing with something with non-trivial destruction.
class HasTrivialDestruction {}

class TrivialFields {
  var a: A;
  var b: B;
}

class NestedFields {
  var x: TrivialFields;
}

base class TrivialBase {}

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

base class NonTrivialBase {
  var x: ImplsDestroy;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

base class DerivedFromNonTrivialBase {
  extend base: NonTrivialBase;
  var y: ImplsDestroy;
}

class GenericClass(T: Core.Destroy) {
  var x: T;
}

base class TrivialDynamicClass {
  virtual fn F(self);
}

base class NonTrivialDynamicClass {
  virtual fn F(self);

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_trivial_empty_class.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Make() -> HasTrivialDestruction;

fn Test() {
  HasTrivialDestruction as Core.Destroy;
  HasTrivialDestruction as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: HasTrivialDestruction = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_empty_class_impls_destroy.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  ImplsDestroy as Core.TrivialDestroy;
}

// --- todo_trivial_all_trivial_fields.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Make() -> TrivialFields;

fn Test() {
  TrivialFields as Core.Destroy;
  TrivialFields as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: TrivialFields = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_field_first.carbon

library "[[@TEST_NAME]]";

import library "types";

class NonTrivialFieldFirst {
  var x: ImplsDestroy;
  var y: HasTrivialDestruction;
}

fn Test() {
  NonTrivialFieldFirst as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_field_second.carbon

library "[[@TEST_NAME]]";

import library "types";

class NonTrivialFieldSecond {
  var x: HasTrivialDestruction;
  var y: ImplsDestroy;
}

fn Test() {
  NonTrivialFieldSecond as Core.TrivialDestroy;
}

// --- todo_trivial_nested_trivial_fields.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Make() -> NestedFields;

fn Test() {
  NestedFields as Core.Destroy;
  NestedFields as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: NestedFields = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_nested_nontrivial_member.carbon

library "[[@TEST_NAME]]";

import library "types";

class HasNonTrivialField {
  var x: ImplsDestroy;
}

class NestedNonTrivialMember {
  var x: HasNonTrivialField;
}

fn Test() {
  NestedNonTrivialMember as Core.TrivialDestroy;
}

// --- todo_trivial_derived_from_trivial_base.carbon

library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: TrivialBase;
  var a: HasTrivialDestruction;
}

fn Make() -> Derived;

fn Test() {
  Derived as Core.Destroy;
  Derived as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Derived = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_derived_from_nontrivial_base.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  DerivedFromNonTrivialBase as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_grandparent.carbon

library "[[@TEST_NAME]]";

import library "types";

class NonTrivialGrandparent {
  extend base: DerivedFromNonTrivialBase;
}

fn Test() {
  NonTrivialGrandparent as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_derived_has_trivial_base_and_impls_destroy.carbon

library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: TrivialBase;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test() {
  Derived as Core.TrivialDestroy;
}

// --- todo_trivial_generic_class.carbon

library "[[@TEST_NAME]]";

import library "types";

alias TrivialGenericClass = GenericClass(HasTrivialDestruction);
fn Make() -> TrivialGenericClass;

fn Test() {
  TrivialGenericClass as Core.Destroy;
  TrivialGenericClass as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: TrivialGenericClass = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_generic_class.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  GenericClass(ImplsDestroy) as Core.Destroy;
}

// --- todo_fail_nontrivial_base_and_trivial_field.carbon

library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: NonTrivialBase;
  var a: HasTrivialDestruction;
}

fn Test() {
  Derived as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_field_and_trivial_base.carbon

library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: TrivialBase;
  var n: ImplsDestroy;
}

fn Test() {
  Derived as Core.TrivialDestroy;
}

// --- todo_trivial_dynamic_class.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Make() -> TrivialDynamicClass;

fn Test() {
  TrivialDynamicClass as Core.Destroy;
  TrivialDynamicClass as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: TrivialDynamicClass = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_dynamic_base.carbon

library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: TrivialDynamicClass;
}

fn Make() -> Derived;

fn Test() {
  Derived as Core.Destroy;
  Derived as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Derived = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_dynamic_class.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  NonTrivialDynamicClass as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_dynamic_base.carbon

library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: NonTrivialDynamicClass;
}

fn Test() {
  Derived as Core.TrivialDestroy;
}

// --- fail_todo_trivial_class_derived_from_abstract.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {}

class Derived {
  extend base: Abstract;
}

fn Make() -> Derived;

// TODO: Support destroying classes that derive from abstract types.
fn Test() {
  Derived as Core.Destroy;
  Derived as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Derived = Make();
  //@dump-sem-ir-end
}

// --- fail_nontrivial_class_derived_from_nontrivial_abstract.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class Derived {
  extend base: Abstract;
}

fn Make() -> Derived;

fn Test() {
  Derived as Core.TrivialDestroy;
}
