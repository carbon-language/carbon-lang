// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/classes.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/classes.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

base class TrivialBase {}

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
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

fn MakeHasTrivialDestruction() -> HasTrivialDestruction;

fn Test() {
  //@dump-sem-ir-begin
  HasTrivialDestruction as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  HasTrivialDestruction as Core.Destroy;
  HasTrivialDestruction as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeHasTrivialDestruction();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_empty_class_impls_destroy.carbon
library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_empty_class_impls_destroy.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   HasNonTrivialDestruction as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `HasNonTrivialDestruction` into type
  // CHECK:STDERR:        ·      ┃        implementing `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  HasNonTrivialDestruction as Core.TrivialDestroy;
}

// --- trivial_all_trivial_fields.carbon
library "[[@TEST_NAME]]";

class TrivialFields {
  class HasTrivialDestruction {}
  class AlsoHasTrivialDestruction {}

  var a: HasTrivialDestruction;
  var b: AlsoHasTrivialDestruction;
}

fn MakeTrivialFields() -> TrivialFields;

fn Test() {
  //@dump-sem-ir-begin
  TrivialFields as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  TrivialFields as Core.Destroy;
  TrivialFields as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeTrivialFields();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_field_first.carbon
library "[[@TEST_NAME]]";

import library "types";

class NonTrivialFieldFirst {
  var x: HasNonTrivialDestruction;
  var y: HasTrivialDestruction;
}

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_field_first.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   NonTrivialFieldFirst as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `NonTrivialFieldFirst` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  NonTrivialFieldFirst as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_field_second.carbon
library "[[@TEST_NAME]]";

import library "types";

class NonTrivialFieldSecond {
  var x: HasTrivialDestruction;
  var y: HasNonTrivialDestruction;
}

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_field_second.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   NonTrivialFieldSecond as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `NonTrivialFieldSecond` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  NonTrivialFieldSecond as Core.TrivialDestroy;
}

// --- trivial_nested_trivial_fields.carbon
library "[[@TEST_NAME]]";

class TrivialFields {
  class HasTrivialDestruction {}
  class AlsoHasTrivialDestruction {}

  var a: HasTrivialDestruction;
  var b: AlsoHasTrivialDestruction;
}

class NestedTrivialFields {
  var x: TrivialFields;
}

fn MakeNestedTrivialFields() -> NestedTrivialFields;

fn Test() {
  //@dump-sem-ir-begin
  NestedTrivialFields as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  NestedTrivialFields as Core.Destroy;
  NestedTrivialFields as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeNestedTrivialFields();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_nested_nontrivial_member.carbon
library "[[@TEST_NAME]]";

import library "types";

class HasNonTrivialField {
  var x: HasNonTrivialDestruction;
}

class NestedNonTrivialMember {
  var x: HasNonTrivialField;
}

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_nested_nontrivial_member.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   NestedNonTrivialMember as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `NestedNonTrivialMember` into type
  // CHECK:STDERR:        ·      ┃        implementing `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  NestedNonTrivialMember as Core.TrivialDestroy;
}

// --- trivial_derived_from_trivial_base.carbon
library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: TrivialBase;
  var a: HasTrivialDestruction;
}

fn MakeDerived() -> Derived;

fn Test() {
  //@dump-sem-ir-begin
  Derived as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Derived as Core.Destroy;
  Derived as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeDerived();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_derived_from_nontrivial_base.carbon
library "[[@TEST_NAME]]";

import library "types";

base class NonTrivialBase {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

base class DerivedFromNonTrivialBase {
  extend base: NonTrivialBase;
  var y: HasNonTrivialDestruction;
}

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_derived_from_nontrivial_base.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   DerivedFromNonTrivialBase as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `DerivedFromNonTrivialBase` into type
  // CHECK:STDERR:        ·      ┃        implementing `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  DerivedFromNonTrivialBase as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_grandparent.carbon
library "[[@TEST_NAME]]";

import library "types";

base class NonTrivialBase {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

base class DerivedFromNonTrivialBase {
  extend base: NonTrivialBase;
  var y: HasNonTrivialDestruction;
}

class NonTrivialGrandparent {
  extend base: DerivedFromNonTrivialBase;
}

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_grandparent.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   NonTrivialGrandparent as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `NonTrivialGrandparent` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
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
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_derived_has_trivial_base_and_impls_destroy.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Derived as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.TrivialDestroy;
}

// --- trivial_generic_class.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}
class GenericClass(T: Core.Destroy) {
  var x: T;
}

alias TrivialGenericClass = GenericClass(HasTrivialDestruction);
fn MakeTrivialGenericClass() -> TrivialGenericClass;

fn Test() {
  //@dump-sem-ir-begin
  TrivialGenericClass as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  TrivialGenericClass as Core.Destroy;
  TrivialGenericClass as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeTrivialGenericClass();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_generic_class.carbon
library "[[@TEST_NAME]]";

import library "types";

class GenericClass(T: Core.Destroy) {
  var x: T;
}

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_generic_class.carbon:[[@LINE+9]]:3
  // CHECK:STDERR: ->     │   GenericClass(HasNonTrivialDestruction) as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type
  // CHECK:STDERR:        ·      ┃        `GenericClass(HasNonTrivialDestruction as Core.Destroy)` into type
  // CHECK:STDERR:        ·      ┃        implementing `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  GenericClass(HasNonTrivialDestruction) as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_base_and_trivial_field.carbon
library "[[@TEST_NAME]]";

import library "types";

base class NonTrivialBase {
  var x: HasNonTrivialDestruction;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class Derived {
  extend base: NonTrivialBase;
  class HasTrivialDestruction {}

  var a: HasTrivialDestruction;
}

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_base_and_trivial_field.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Derived as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_field_and_trivial_base.carbon
library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: TrivialBase;
  var n: HasNonTrivialDestruction;
}

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_field_and_trivial_base.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Derived as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.TrivialDestroy;
}

// --- trivial_dynamic_class.carbon
library "[[@TEST_NAME]]";

import library "types";

fn MakeTrivialDynamicClass() -> TrivialDynamicClass;

fn Test() {
  //@dump-sem-ir-begin
  TrivialDynamicClass as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  TrivialDynamicClass as Core.Destroy;
  TrivialDynamicClass as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeTrivialDynamicClass();
  //@dump-sem-ir-end
}

// --- trivial_dynamic_base.carbon
library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: TrivialDynamicClass;
}

fn MakeDerived() -> Derived;

fn Test() {
  //@dump-sem-ir-begin
  Derived as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Derived as Core.Destroy;
  Derived as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeDerived();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_dynamic_class.carbon
library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_dynamic_class.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   NonTrivialDynamicClass as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `NonTrivialDynamicClass` into type
  // CHECK:STDERR:        ·      ┃        implementing `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  NonTrivialDynamicClass as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_dynamic_base.carbon
library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: NonTrivialDynamicClass;
}

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_dynamic_base.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Derived as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.TrivialDestroy;
}

// --- fail_todo_trivial_class_derived_from_abstract.carbon
library "[[@TEST_NAME]]";

// TODO: Support destroying classes that derive from abstract types.

abstract class TrivialAbstract {}

class Derived {
  extend base: TrivialAbstract;
}

fn MakeDerived() -> Derived;

fn Test() {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_todo_trivial_class_derived_from_abstract.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Derived as Core.SubobjectDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.SubobjectDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  // CHECK:STDERR:        ╭─┤ fail_todo_trivial_class_derived_from_abstract.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Derived as Core.Destroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Destroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.Destroy;
  // CHECK:STDERR:        ╭─┤ fail_todo_trivial_class_derived_from_abstract.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Derived as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_todo_trivial_class_derived_from_abstract.carbon:[[@LINE+14]]:3
  // CHECK:STDERR: ->     │   MakeDerived();
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Destroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ min_prelude/parts/destroy.carbon
  // CHECK:STDERR:        │ fn SelfDestruct[T: Destroy](ref t: T) {
  // CHECK:STDERR:        ·    ──────┬─────
  // CHECK:STDERR:        ·          ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·             │ [DeductionGenericHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  MakeDerived();
  //@dump-sem-ir-end
}

// --- fail_nontrivial_class_derived_from_nontrivial_abstract.carbon
library "[[@TEST_NAME]]";

abstract class TrivialAbstract {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class Derived {
  extend base: TrivialAbstract;
}

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nontrivial_class_derived_from_nontrivial_abstract.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Derived as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- trivial_empty_class.carbon = $trivial_empty_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %MakeHasTrivialDestruction.type: type = fn_type @MakeHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeHasTrivialDestruction: %MakeHasTrivialDestruction.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_25.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.HasTrivialDestruction: type = import_ref Main//types, HasTrivialDestruction, loaded [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %HasTrivialDestruction.ref.loc9: type = name_ref HasTrivialDestruction, imports.%Main.HasTrivialDestruction [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %HasTrivialDestruction.ref.loc9, (constants.%custom_witness.b80ef6.2) [concrete = constants.%SubobjectDestroy.facet.5bc]
// CHECK:STDOUT:   %.loc9: %SubobjectDestroy.type = converted %HasTrivialDestruction.ref.loc9, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.5bc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeHasTrivialDestruction.ref: %MakeHasTrivialDestruction.type = name_ref MakeHasTrivialDestruction, file.%MakeHasTrivialDestruction.decl [concrete = constants.%MakeHasTrivialDestruction]
// CHECK:STDOUT:   %.loc16_29.1: ref %HasTrivialDestruction = temporary_storage
// CHECK:STDOUT:   %MakeHasTrivialDestruction.call: init %HasTrivialDestruction to %.loc16_29.1 = call %MakeHasTrivialDestruction.ref()
// CHECK:STDOUT:   %.loc16_29.2: ref %HasTrivialDestruction = temporary %.loc16_29.1, %MakeHasTrivialDestruction.call
// CHECK:STDOUT:   %Destroy.facet.loc16_29.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_29.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc16_29.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc16_29.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_29.4: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc16_29.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc16_29.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_25.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_25.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_all_trivial_fields.carbon = $trivial_all_trivial_fields
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialFields: type = class_type @TrivialFields [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %AlsoHasTrivialDestruction: type = class_type @AlsoHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.841: type = struct_type {.a: %HasTrivialDestruction, .b: %AlsoHasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %MakeTrivialFields.type: type = fn_type @MakeTrivialFields [concrete]
// CHECK:STDOUT:   %MakeTrivialFields: %MakeTrivialFields.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_17.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.16d: %SubobjectDestroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c88: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.16d) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.c88) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TrivialFields.ref.loc15: type = name_ref TrivialFields, file.%TrivialFields.decl [concrete = constants.%TrivialFields]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %TrivialFields.ref.loc15, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.16d]
// CHECK:STDOUT:   %.loc15: %SubobjectDestroy.type = converted %TrivialFields.ref.loc15, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.16d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeTrivialFields.ref: %MakeTrivialFields.type = name_ref MakeTrivialFields, file.%MakeTrivialFields.decl [concrete = constants.%MakeTrivialFields]
// CHECK:STDOUT:   %.loc22_21.1: ref %TrivialFields = temporary_storage
// CHECK:STDOUT:   %MakeTrivialFields.call: init %TrivialFields to %.loc22_21.1 = call %MakeTrivialFields.ref()
// CHECK:STDOUT:   %.loc22_21.2: ref %TrivialFields = temporary %.loc22_21.1, %MakeTrivialFields.call
// CHECK:STDOUT:   %Destroy.facet.loc22_21.1: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc22_21.3: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc22_21.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc22_21.2: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc22_21.4: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc22_21.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc22_21.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_17.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_17.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_17.3(%self.param: ref %AlsoHasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_17.4(%self.param: ref %struct_type.a.b.841) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_17.5(%self.param: ref %TrivialFields) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_nested_trivial_fields.carbon = $trivial_nested_trivial_fields
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialFields: type = class_type @TrivialFields [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %AlsoHasTrivialDestruction: type = class_type @AlsoHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.841: type = struct_type {.a: %HasTrivialDestruction, .b: %AlsoHasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %NestedTrivialFields: type = class_type @NestedTrivialFields [concrete]
// CHECK:STDOUT:   %struct_type.x.156: type = struct_type {.x: %TrivialFields} [concrete]
// CHECK:STDOUT:   %MakeNestedTrivialFields.type: type = fn_type @MakeNestedTrivialFields [concrete]
// CHECK:STDOUT:   %MakeNestedTrivialFields: %MakeNestedTrivialFields.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_23.7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.23d: %SubobjectDestroy.type = facet_value %NestedTrivialFields, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.810: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.23d) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %NestedTrivialFields, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.810) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NestedTrivialFields.ref.loc19: type = name_ref NestedTrivialFields, file.%NestedTrivialFields.decl [concrete = constants.%NestedTrivialFields]
// CHECK:STDOUT:   %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %NestedTrivialFields.ref.loc19, (constants.%custom_witness.b80ef6.7) [concrete = constants.%SubobjectDestroy.facet.23d]
// CHECK:STDOUT:   %.loc19: %SubobjectDestroy.type = converted %NestedTrivialFields.ref.loc19, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.23d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeNestedTrivialFields.ref: %MakeNestedTrivialFields.type = name_ref MakeNestedTrivialFields, file.%MakeNestedTrivialFields.decl [concrete = constants.%MakeNestedTrivialFields]
// CHECK:STDOUT:   %.loc26_27.1: ref %NestedTrivialFields = temporary_storage
// CHECK:STDOUT:   %MakeNestedTrivialFields.call: init %NestedTrivialFields to %.loc26_27.1 = call %MakeNestedTrivialFields.ref()
// CHECK:STDOUT:   %.loc26_27.2: ref %NestedTrivialFields = temporary %.loc26_27.1, %MakeNestedTrivialFields.call
// CHECK:STDOUT:   %Destroy.facet.loc26_27.1: %Destroy.type = facet_value constants.%NestedTrivialFields, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.810) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc26_27.3: %Destroy.type = converted constants.%NestedTrivialFields, %Destroy.facet.loc26_27.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc26_27.2: %Destroy.type = facet_value constants.%NestedTrivialFields, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.810) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc26_27.4: %Destroy.type = converted constants.%NestedTrivialFields, %Destroy.facet.loc26_27.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc26_27.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.3(%self.param: ref %AlsoHasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.4(%self.param: ref %struct_type.a.b.841) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.5(%self.param: ref %TrivialFields) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.6(%self.param: ref %struct_type.x.156) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.7(%self.param: ref %NestedTrivialFields) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_derived_from_trivial_base.carbon = $trivial_derived_from_trivial_base
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %TrivialBase: type = class_type @TrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.base.a.e10: type = struct_type {.base: %TrivialBase, .a: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %MakeDerived.type: type = fn_type @MakeDerived [concrete]
// CHECK:STDOUT:   %MakeDerived: %MakeDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_11.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %Derived, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.660: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Derived, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.660) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Derived.ref.loc14: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc14: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Derived.ref.loc14, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   %.loc14: %SubobjectDestroy.type = converted %Derived.ref.loc14, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeDerived.ref: %MakeDerived.type = name_ref MakeDerived, file.%MakeDerived.decl [concrete = constants.%MakeDerived]
// CHECK:STDOUT:   %.loc21_15.1: ref %Derived = temporary_storage
// CHECK:STDOUT:   %MakeDerived.call: init %Derived to %.loc21_15.1 = call %MakeDerived.ref()
// CHECK:STDOUT:   %.loc21_15.2: ref %Derived = temporary %.loc21_15.1, %MakeDerived.call
// CHECK:STDOUT:   %Destroy.facet.loc21_15.1: %Destroy.type = facet_value constants.%Derived, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc21_15.3: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc21_15.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc21_15.2: %Destroy.type = facet_value constants.%Derived, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc21_15.4: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc21_15.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc21_15.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.2(%self.param: ref %TrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.4(%self.param: ref %struct_type.base.a.e10) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.5(%self.param: ref %Derived) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_generic_class.carbon = $trivial_generic_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_63.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dba: %Destroy.type = facet_value %HasTrivialDestruction, (%DestroyImpl.impl_witness.4ca, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %GenericClass.422: type = class_type @GenericClass, @GenericClass(%Destroy.facet.dba) [concrete]
// CHECK:STDOUT:   %MakeTrivialGenericClass.type: type = fn_type @MakeTrivialGenericClass [concrete]
// CHECK:STDOUT:   %MakeTrivialGenericClass: %MakeTrivialGenericClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.f48: type = struct_type {.x: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_23.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.542: %SubobjectDestroy.type = facet_value %GenericClass.422, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f8b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.542) [concrete]
// CHECK:STDOUT:   %Destroy.facet.43b: %Destroy.type = facet_value %GenericClass.422, (%DestroyImpl.impl_witness.f8b, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TrivialGenericClass.ref.loc13: type = name_ref TrivialGenericClass, file.%TrivialGenericClass [concrete = constants.%GenericClass.422]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %TrivialGenericClass.ref.loc13, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.542]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %TrivialGenericClass.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.542]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeTrivialGenericClass.ref: %MakeTrivialGenericClass.type = name_ref MakeTrivialGenericClass, file.%MakeTrivialGenericClass.decl [concrete = constants.%MakeTrivialGenericClass]
// CHECK:STDOUT:   %.loc20_27.1: ref %GenericClass.422 = temporary_storage
// CHECK:STDOUT:   %MakeTrivialGenericClass.call: init %GenericClass.422 to %.loc20_27.1 = call %MakeTrivialGenericClass.ref()
// CHECK:STDOUT:   %.loc20_27.2: ref %GenericClass.422 = temporary %.loc20_27.1, %MakeTrivialGenericClass.call
// CHECK:STDOUT:   %Destroy.facet.loc20_27.1: %Destroy.type = facet_value constants.%GenericClass.422, (constants.%DestroyImpl.impl_witness.f8b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   %.loc20_27.3: %Destroy.type = converted constants.%GenericClass.422, %Destroy.facet.loc20_27.1 [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   %Destroy.facet.loc20_27.2: %Destroy.type = facet_value constants.%GenericClass.422, (constants.%DestroyImpl.impl_witness.f8b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   %.loc20_27.4: %Destroy.type = converted constants.%GenericClass.422, %Destroy.facet.loc20_27.2 [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_27.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_23.1(%self.param: ref %struct_type.x.f48) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_23.2(%self.param: ref %GenericClass.422) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_dynamic_class.carbon = $trivial_dynamic_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialDynamicClass: type = class_type @TrivialDynamicClass [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %MakeTrivialDynamicClass.type: type = fn_type @MakeTrivialDynamicClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeTrivialDynamicClass: %MakeTrivialDynamicClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_23.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.52e: %SubobjectDestroy.type = facet_value %TrivialDynamicClass, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1a4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.52e) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %TrivialDynamicClass, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.TrivialDynamicClass: type = import_ref Main//types, TrivialDynamicClass, loaded [concrete = constants.%TrivialDynamicClass]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TrivialDynamicClass.ref.loc9: type = name_ref TrivialDynamicClass, imports.%Main.TrivialDynamicClass [concrete = constants.%TrivialDynamicClass]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %TrivialDynamicClass.ref.loc9, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.52e]
// CHECK:STDOUT:   %.loc9: %SubobjectDestroy.type = converted %TrivialDynamicClass.ref.loc9, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.52e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeTrivialDynamicClass.ref: %MakeTrivialDynamicClass.type = name_ref MakeTrivialDynamicClass, file.%MakeTrivialDynamicClass.decl [concrete = constants.%MakeTrivialDynamicClass]
// CHECK:STDOUT:   %.loc16_27.1: ref %TrivialDynamicClass = temporary_storage
// CHECK:STDOUT:   %MakeTrivialDynamicClass.call: init %TrivialDynamicClass to %.loc16_27.1 = call %MakeTrivialDynamicClass.ref()
// CHECK:STDOUT:   %.loc16_27.2: ref %TrivialDynamicClass = temporary %.loc16_27.1, %MakeTrivialDynamicClass.call
// CHECK:STDOUT:   %Destroy.facet.loc16_27.1: %Destroy.type = facet_value constants.%TrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.1a4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_27.3: %Destroy.type = converted constants.%TrivialDynamicClass, %Destroy.facet.loc16_27.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc16_27.2: %Destroy.type = facet_value constants.%TrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.1a4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_27.4: %Destroy.type = converted constants.%TrivialDynamicClass, %Destroy.facet.loc16_27.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc16_27.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_23.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_23.2(%self.param: ref %struct_type.vptr) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_23.3(%self.param: ref %TrivialDynamicClass) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_dynamic_base.carbon = $trivial_dynamic_base
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %TrivialDynamicClass: type = class_type @TrivialDynamicClass [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %TrivialDynamicClass.F.type: type = fn_type @TrivialDynamicClass.F [concrete]
// CHECK:STDOUT:   %TrivialDynamicClass.F: %TrivialDynamicClass.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.6c4: type = struct_type {.base: %TrivialDynamicClass} [concrete]
// CHECK:STDOUT:   %MakeDerived.type: type = fn_type @MakeDerived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeDerived: %MakeDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %Derived, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.660: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Derived, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.660) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @TrivialDynamicClass.vtable {
// CHECK:STDOUT:   constants.%TrivialDynamicClass.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   constants.%TrivialDynamicClass.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Derived.ref.loc13: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Derived.ref.loc13, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %Derived.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeDerived.ref: %MakeDerived.type = name_ref MakeDerived, file.%MakeDerived.decl [concrete = constants.%MakeDerived]
// CHECK:STDOUT:   %.loc20_15.1: ref %Derived = temporary_storage
// CHECK:STDOUT:   %MakeDerived.call: init %Derived to %.loc20_15.1 = call %MakeDerived.ref()
// CHECK:STDOUT:   %.loc20_15.2: ref %Derived = temporary %.loc20_15.1, %MakeDerived.call
// CHECK:STDOUT:   %Destroy.facet.loc20_15.1: %Destroy.type = facet_value constants.%Derived, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_15.3: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc20_15.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc20_15.2: %Destroy.type = facet_value constants.%Derived, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_15.4: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc20_15.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_15.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.2(%self.param: ref %struct_type.vptr) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.3(%self.param: ref %TrivialDynamicClass) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.4(%self.param: ref %struct_type.base.6c4) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.5(%self.param: ref %Derived) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_trivial_class_derived_from_abstract.carbon = $fail_todo_trivial_class_derived_from_abstract
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %MakeDerived.type: type = fn_type @MakeDerived [concrete]
// CHECK:STDOUT:   %MakeDerived: %MakeDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Derived.ref.loc23: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc23: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeDerived.ref: %MakeDerived.type = name_ref MakeDerived, file.%MakeDerived.decl [concrete = constants.%MakeDerived]
// CHECK:STDOUT:   %.loc60_15.1: ref %Derived = temporary_storage
// CHECK:STDOUT:   %MakeDerived.call: init %Derived to %.loc60_15.1 = call %MakeDerived.ref()
// CHECK:STDOUT:   %.loc60_15.2: ref %Derived = temporary %.loc60_15.1, %MakeDerived.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
