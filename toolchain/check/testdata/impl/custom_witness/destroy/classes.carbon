// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
// 
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/classes.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/classes.carbon

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

// --- trivial_empty_class.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: HasTrivialDestruction;
}

// --- todo_nontrivial_empty_class_impls_destroy.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: ImplsDestroy;
  //@dump-sem-ir-end
}


// --- trivial_all_trivial_fields.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: TrivialFields;
}

// --- todo_nontrivial_all_nontrivial_fields.carbon

library "[[@TEST_NAME]]";

import library "types";

class AlsoImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

class AllNonTrivialFields {
  var a: ImplsDestroy;
  var b: AlsoImplsDestroy;
  var c: AlsoImplsDestroy;
  var d: ImplsDestroy;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: AllNonTrivialFields;
  //@dump-sem-ir-end
}

// --- todo_nontrivial_field_first.carbon

library "[[@TEST_NAME]]";

import library "types";

class NonTrivialFieldFirst {
  var x: ImplsDestroy;
  var y: HasTrivialDestruction;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: NonTrivialFieldFirst;
  //@dump-sem-ir-end
}

// --- todo_nontrivial_field_second.carbon

library "[[@TEST_NAME]]";

import library "types";

class NonTrivialFieldSecond {
  var x: HasTrivialDestruction;
  var y: ImplsDestroy;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: NonTrivialFieldSecond;
  //@dump-sem-ir-end
}

// --- trivial_nested_trivial_fields.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: NestedFields;
}

// --- todo_nontrivial_nested_nontrivial_member.carbon

library "[[@TEST_NAME]]";

import library "types";

class HasNonTrivialField {
  var x: ImplsDestroy;
}

class NestedNonTrivialMember {
  var x: HasNonTrivialField;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: NestedNonTrivialMember;
  //@dump-sem-ir-end
}

// --- trivial_derived_from_trivial_base.carbon

library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: TrivialBase;
  var a: HasTrivialDestruction;
}

fn Test() {
  var unused x: Derived;
}

// --- todo_nontrivial_derived_from_nontrivial_base.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: DerivedFromNonTrivialBase;
  //@dump-sem-ir-end
}

// --- todo_nontrivial_grandparent.carbon

library "[[@TEST_NAME]]";

import library "types";

class NonTrivialGrandparent {
  extend base: DerivedFromNonTrivialBase;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: NonTrivialGrandparent;
  //@dump-sem-ir-end
}

// --- todo_nontrivial_derived_has_trivial_base_and_impls_destroy.carbon

library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: TrivialBase;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Derived;
  //@dump-sem-ir-end
}

// --- trivial_generic_class.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: GenericClass(HasTrivialDestruction);
}

// --- todo_nontrivial_generic_class.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: GenericClass(ImplsDestroy);
  //@dump-sem-ir-end
}

// --- todo_nontrivial_base_and_trivial_field.carbon

library "[[@TEST_NAME]]";

import library "types";

class NonTrivialBaseAndTrivialMember {
  extend base: NonTrivialBase;
  var y: HasTrivialDestruction;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: NonTrivialBaseAndTrivialMember;
  //@dump-sem-ir-end
}

// --- todo_nontrivial_trivial_base_and_nontrivial_field.carbon

library "[[@TEST_NAME]]";

import library "types";

class TrivialBaseAndNonTrivialMember {
  extend base: TrivialBase;
  var x: ImplsDestroy;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: TrivialBaseAndNonTrivialMember;
  //@dump-sem-ir-end
}

// --- trivial_dynamic_class.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: TrivialDynamicClass = {};
}

// --- trivial_dynamic_base.carbon

library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: TrivialDynamicClass;
}

fn Test() {
  var unused x: Derived;
}

// --- todo_nontrivial_class_dynamic.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: NonTrivialDynamicClass;
  //@dump-sem-ir-end
}

// --- todo_nontrivial_dynamic_base.carbon

library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: NonTrivialDynamicClass;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Derived;
  //@dump-sem-ir-end
}

// --- fail_todo_trivial_class_derived_from_abstract.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {}

class Derived {
  extend base: Abstract;
}

// TODO: Support destroying classes that derive from abstract types.
fn Test() {
  var unused x: Derived;
}

// --- fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon

library "[[@TEST_NAME]]";


abstract class Abstract {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class Derived {
  extend base: Abstract;
}

// TODO: Support destroying classes that derive from abstract types.
fn Test() {
  //@dump-sem-ir-begin
  var unused x: Derived;
  //@dump-sem-ir-end
}
