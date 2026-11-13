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
  //@dump-sem-ir-begin
  HasTrivialDestruction as Core.SubobjectDestroy;
  HasTrivialDestruction as Core.Destroy;
  HasTrivialDestruction as Core.TrivialDestroy;

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
  //@dump-sem-ir-begin
  TrivialFields as Core.SubobjectDestroy;
  TrivialFields as Core.Destroy;
  TrivialFields as Core.TrivialDestroy;

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
  //@dump-sem-ir-begin
  NestedFields as Core.SubobjectDestroy;
  NestedFields as Core.Destroy;
  NestedFields as Core.TrivialDestroy;

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
  //@dump-sem-ir-begin
  Derived as Core.SubobjectDestroy;
  Derived as Core.Destroy;
  Derived as Core.TrivialDestroy;

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
  //@dump-sem-ir-begin
  TrivialGenericClass as Core.SubobjectDestroy;
  TrivialGenericClass as Core.Destroy;
  TrivialGenericClass as Core.TrivialDestroy;

  var unused x: TrivialGenericClass = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_generic_class.carbon
library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  GenericClass(ImplsDestroy) as Core.TrivialDestroy;
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
  //@dump-sem-ir-begin
  TrivialDynamicClass as Core.SubobjectDestroy;
  TrivialDynamicClass as Core.Destroy;
  TrivialDynamicClass as Core.TrivialDestroy;

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
  //@dump-sem-ir-begin
  Derived as Core.SubobjectDestroy;
  Derived as Core.Destroy;
  Derived as Core.TrivialDestroy;

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
  // CHECK:STDERR:        ·      ┃        `Core.Destroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.TrivialDestroy;

  // CHECK:STDERR:        ╭─┤ fail_todo_trivial_class_derived_from_abstract.carbon:[[@LINE+14]]:3
  // CHECK:STDERR: ->     │   var unused x: Derived = Make();
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Destroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ min_prelude/parts/destroy.carbon
  // CHECK:STDERR:        │ fn SelfDestruct[T: Destroy](ref t: T) {
  // CHECK:STDERR:        · ───────────────────┬───────────────────
  // CHECK:STDERR:        ·                    ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·                       │ [DeductionGenericHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
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

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nontrivial_class_derived_from_nontrivial_abstract.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Derived as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Destroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- todo_trivial_empty_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_25.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.HasTrivialDestruction: type = import_ref Main//types, HasTrivialDestruction, loaded [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:   %HasTrivialDestruction.ref.loc10: type = name_ref HasTrivialDestruction, imports.%Main.HasTrivialDestruction [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc10: %Destroy.type = facet_value %HasTrivialDestruction.ref.loc10, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10: %Destroy.type = converted %HasTrivialDestruction.ref.loc10, %Destroy.facet.loc10 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %HasTrivialDestruction.ref.loc11: type = name_ref HasTrivialDestruction, imports.%Main.HasTrivialDestruction [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %HasTrivialDestruction.ref.loc11, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %HasTrivialDestruction.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %HasTrivialDestruction = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %HasTrivialDestruction = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %HasTrivialDestruction to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %HasTrivialDestruction.ref.loc13: type = name_ref HasTrivialDestruction, imports.%Main.HasTrivialDestruction [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:   %x: ref %HasTrivialDestruction = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_25.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_25.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_all_trivial_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialFields: type = class_type @TrivialFields [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %struct_type.a.b.06d: type = struct_type {.a: %A, .b: %B} [concrete]
// CHECK:STDOUT:   %pattern_type.063: type = pattern_type %TrivialFields [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_17.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_17.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.92e: %SubobjectDestroy.type = facet_value %B, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.131: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.92e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dc1: %Destroy.type = facet_value %B, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.131) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9bf: %SubobjectDestroy.type = facet_value %A, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9bf) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3d0: %Destroy.type = facet_value %A, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.6d1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_17.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialFields.elem.3ba: type = unbound_element_type %TrivialFields, %B [concrete]
// CHECK:STDOUT:   %TrivialFields.elem.8f7: type = unbound_element_type %TrivialFields, %A [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.16d: %SubobjectDestroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c88: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.16d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.bc5: %Destroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.c88) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.063 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.063 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.TrivialFields: type = import_ref Main//types, TrivialFields, loaded [concrete = constants.%TrivialFields]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.f3e: %TrivialFields.elem.8f7 = import_ref Main//types, loc10_8, loaded [concrete = %field_decl.1fe]
// CHECK:STDOUT:   %Main.import_ref.51c: %TrivialFields.elem.3ba = import_ref Main//types, loc11_8, loaded [concrete = %field_decl.556]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.c8c = import_ref Main//types, loc11_10, unloaded
// CHECK:STDOUT:   %field_decl.556: %TrivialFields.elem.3ba = field_decl b, element1, %Main.import_ref.c8c in [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.35d = import_ref Main//types, loc10_10, unloaded
// CHECK:STDOUT:   %field_decl.1fe: %TrivialFields.elem.8f7 = field_decl a, element0, %Main.import_ref.35d in [concrete] {}
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TrivialFields.ref.loc9: type = name_ref TrivialFields, imports.%Main.TrivialFields [concrete = constants.%TrivialFields]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %TrivialFields.ref.loc9, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.16d]
// CHECK:STDOUT:   %.loc9: %SubobjectDestroy.type = converted %TrivialFields.ref.loc9, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.16d]
// CHECK:STDOUT:   %TrivialFields.ref.loc10: type = name_ref TrivialFields, imports.%Main.TrivialFields [concrete = constants.%TrivialFields]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc10: %Destroy.type = facet_value %TrivialFields.ref.loc10, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc10: %Destroy.type = converted %TrivialFields.ref.loc10, %Destroy.facet.loc10 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %TrivialFields.ref.loc11: type = name_ref TrivialFields, imports.%Main.TrivialFields [concrete = constants.%TrivialFields]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %TrivialFields.ref.loc11, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %TrivialFields.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %x.var: ref %TrivialFields = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %TrivialFields = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %TrivialFields to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %TrivialFields.ref.loc13: type = name_ref TrivialFields, imports.%Main.TrivialFields [concrete = constants.%TrivialFields]
// CHECK:STDOUT:   %x: ref %TrivialFields = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.063 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.063 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_17.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_17.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_17.3(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_17.4(%self.param: ref %struct_type.a.b.06d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_17.1: ref %B = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc9_17.1: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc9_17.2: %Destroy.type = converted constants.%B, %Destroy.facet.loc9_17.1 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %Destroy.facet.loc9_17.2: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc9_17.3: %Destroy.type = converted constants.%B, %Destroy.facet.loc9_17.2 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_17.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc9_17.1)
// CHECK:STDOUT:   %.loc9_17.4: ref %A = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_17.3: %Destroy.type = facet_value constants.%A, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d1) [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %.loc9_17.5: %Destroy.type = converted constants.%A, %Destroy.facet.loc9_17.3 [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %Destroy.facet.loc9_17.4: %Destroy.type = facet_value constants.%A, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d1) [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %.loc9_17.6: %Destroy.type = converted constants.%A, %Destroy.facet.loc9_17.4 [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_17.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc9_17.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_17.5(%self.param: ref %TrivialFields) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: %TrivialFields.elem.3ba = name_ref b, imports.%Main.import_ref.51c [concrete = imports.%field_decl.556]
// CHECK:STDOUT:   %.loc9_17.1: ref %B = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc9_17.1: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc9_17.2: %Destroy.type = converted constants.%B, %Destroy.facet.loc9_17.1 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %Destroy.facet.loc9_17.2: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc9_17.3: %Destroy.type = converted constants.%B, %Destroy.facet.loc9_17.2 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_17.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc9_17.1)
// CHECK:STDOUT:   %a.ref: %TrivialFields.elem.8f7 = name_ref a, imports.%Main.import_ref.f3e [concrete = imports.%field_decl.1fe]
// CHECK:STDOUT:   %.loc9_17.4: ref %A = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_17.3: %Destroy.type = facet_value constants.%A, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d1) [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %.loc9_17.5: %Destroy.type = converted constants.%A, %Destroy.facet.loc9_17.3 [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %Destroy.facet.loc9_17.4: %Destroy.type = facet_value constants.%A, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d1) [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %.loc9_17.6: %Destroy.type = converted constants.%A, %Destroy.facet.loc9_17.4 [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_17.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc9_17.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_nested_trivial_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NestedFields: type = class_type @NestedFields [concrete]
// CHECK:STDOUT:   %TrivialFields: type = class_type @TrivialFields [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %struct_type.a.b.06d: type = struct_type {.a: %A, .b: %B} [concrete]
// CHECK:STDOUT:   %struct_type.x.156: type = struct_type {.x: %TrivialFields} [concrete]
// CHECK:STDOUT:   %pattern_type.ab4: type = pattern_type %NestedFields [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_16.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_16.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.92e: %SubobjectDestroy.type = facet_value %B, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.131: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.92e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dc1: %Destroy.type = facet_value %B, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.131) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9bf: %SubobjectDestroy.type = facet_value %A, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9bf) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3d0: %Destroy.type = facet_value %A, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.6d1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_16.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialFields.elem.3ba: type = unbound_element_type %TrivialFields, %B [concrete]
// CHECK:STDOUT:   %TrivialFields.elem.8f7: type = unbound_element_type %TrivialFields, %A [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.16d: %SubobjectDestroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c88: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.16d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.bc5: %Destroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.c88) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_16.7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %NestedFields.elem: type = unbound_element_type %NestedFields, %TrivialFields [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.697: %SubobjectDestroy.type = facet_value %NestedFields, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.512: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.697) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9aa: %Destroy.type = facet_value %NestedFields, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.512) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.ab4 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.ab4 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NestedFields: type = import_ref Main//types, NestedFields, loaded [concrete = constants.%NestedFields]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.f3e: %TrivialFields.elem.8f7 = import_ref Main//types, loc10_8, loaded [concrete = %field_decl.1fe]
// CHECK:STDOUT:   %Main.import_ref.51c: %TrivialFields.elem.3ba = import_ref Main//types, loc11_8, loaded [concrete = %field_decl.556]
// CHECK:STDOUT:   %Main.import_ref.e6d: %NestedFields.elem = import_ref Main//types, loc15_8, loaded [concrete = %field_decl.e44]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.c8c = import_ref Main//types, loc11_10, unloaded
// CHECK:STDOUT:   %field_decl.556: %TrivialFields.elem.3ba = field_decl b, element1, %Main.import_ref.c8c in [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.35d = import_ref Main//types, loc10_10, unloaded
// CHECK:STDOUT:   %field_decl.1fe: %TrivialFields.elem.8f7 = field_decl a, element0, %Main.import_ref.35d in [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.46d9 = import_ref Main//types, loc15_10, unloaded
// CHECK:STDOUT:   %field_decl.e44: %NestedFields.elem = field_decl x, element0, %Main.import_ref.46d9 in [concrete] {}
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NestedFields.ref.loc9: type = name_ref NestedFields, imports.%Main.NestedFields [concrete = constants.%NestedFields]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %NestedFields.ref.loc9, (constants.%custom_witness.b80ef6.7) [concrete = constants.%SubobjectDestroy.facet.697]
// CHECK:STDOUT:   %.loc9: %SubobjectDestroy.type = converted %NestedFields.ref.loc9, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.697]
// CHECK:STDOUT:   %NestedFields.ref.loc10: type = name_ref NestedFields, imports.%Main.NestedFields [concrete = constants.%NestedFields]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc10: %Destroy.type = facet_value %NestedFields.ref.loc10, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.512) [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   %.loc10: %Destroy.type = converted %NestedFields.ref.loc10, %Destroy.facet.loc10 [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   %NestedFields.ref.loc11: type = name_ref NestedFields, imports.%Main.NestedFields [concrete = constants.%NestedFields]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %NestedFields.ref.loc11, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.512) [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %NestedFields.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   %x.var: ref %NestedFields = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %NestedFields = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NestedFields to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %NestedFields.ref.loc13: type = name_ref NestedFields, imports.%Main.NestedFields [concrete = constants.%NestedFields]
// CHECK:STDOUT:   %x: ref %NestedFields = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.ab4 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.ab4 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%NestedFields, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.512) [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%NestedFields, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%NestedFields, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.512) [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%NestedFields, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_16.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_16.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_16.3(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_16.4(%self.param: ref %struct_type.a.b.06d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_16.1: ref %B = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc9_16.1: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc9_16.2: %Destroy.type = converted constants.%B, %Destroy.facet.loc9_16.1 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %Destroy.facet.loc9_16.2: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc9_16.3: %Destroy.type = converted constants.%B, %Destroy.facet.loc9_16.2 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_16.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc9_16.1)
// CHECK:STDOUT:   %.loc9_16.4: ref %A = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_16.3: %Destroy.type = facet_value constants.%A, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d1) [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %.loc9_16.5: %Destroy.type = converted constants.%A, %Destroy.facet.loc9_16.3 [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %Destroy.facet.loc9_16.4: %Destroy.type = facet_value constants.%A, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d1) [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %.loc9_16.6: %Destroy.type = converted constants.%A, %Destroy.facet.loc9_16.4 [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_16.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc9_16.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_16.5(%self.param: ref %TrivialFields) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: %TrivialFields.elem.3ba = name_ref b, imports.%Main.import_ref.51c [concrete = imports.%field_decl.556]
// CHECK:STDOUT:   %.loc9_16.1: ref %B = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc9_16.1: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc9_16.2: %Destroy.type = converted constants.%B, %Destroy.facet.loc9_16.1 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %Destroy.facet.loc9_16.2: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc9_16.3: %Destroy.type = converted constants.%B, %Destroy.facet.loc9_16.2 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_16.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc9_16.1)
// CHECK:STDOUT:   %a.ref: %TrivialFields.elem.8f7 = name_ref a, imports.%Main.import_ref.f3e [concrete = imports.%field_decl.1fe]
// CHECK:STDOUT:   %.loc9_16.4: ref %A = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_16.3: %Destroy.type = facet_value constants.%A, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d1) [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %.loc9_16.5: %Destroy.type = converted constants.%A, %Destroy.facet.loc9_16.3 [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %Destroy.facet.loc9_16.4: %Destroy.type = facet_value constants.%A, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d1) [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %.loc9_16.6: %Destroy.type = converted constants.%A, %Destroy.facet.loc9_16.4 [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_16.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc9_16.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_16.6(%self.param: ref %struct_type.x.156) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_16.1: ref %TrivialFields = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_16.1: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc9_16.2: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc9_16.1 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %Destroy.facet.loc9_16.2: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc9_16.3: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc9_16.2 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc9_16.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_16.7(%self.param: ref %NestedFields) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NestedFields.elem = name_ref x, imports.%Main.import_ref.e6d [concrete = imports.%field_decl.e44]
// CHECK:STDOUT:   %.loc9_16.1: ref %TrivialFields = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_16.1: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc9_16.2: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc9_16.1 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %Destroy.facet.loc9_16.2: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc9_16.3: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc9_16.2 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc9_16.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_derived_from_trivial_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %TrivialBase: type = class_type @TrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Derived.elem.dbc: type = unbound_element_type %Derived, %TrivialBase [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %Derived.elem.179: type = unbound_element_type %Derived, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.base.a.e10: type = struct_type {.base: %TrivialBase, .a: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %pattern_type.746d: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_11.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_11.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.23e: %SubobjectDestroy.type = facet_value %TrivialBase, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7c0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.23e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.033: %Destroy.type = facet_value %TrivialBase, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.7c0) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_11.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %Derived, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.660: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f2a: %Destroy.type = facet_value %Derived, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.660) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.746d = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.746d = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Derived.ref.loc14: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc14: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Derived.ref.loc14, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   %.loc14: %SubobjectDestroy.type = converted %Derived.ref.loc14, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   %Derived.ref.loc15: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc15: %Destroy.type = facet_value %Derived.ref.loc15, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %.loc15: %Destroy.type = converted %Derived.ref.loc15, %Destroy.facet.loc15 [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %Derived.ref.loc16: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc16: %Destroy.type = facet_value %Derived.ref.loc16, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %.loc16: %Destroy.type = converted %Derived.ref.loc16, %Destroy.facet.loc16 [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %x.var: ref %Derived = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc18_3.1: ref %Derived = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc18_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Derived.ref.loc18: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %x: ref %Derived = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.746d = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.746d = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%Derived, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%Derived, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.2(%self.param: ref %TrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.4(%self.param: ref %struct_type.base.a.e10) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_11.1: ref %HasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_11.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_11.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_11.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc14_11.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_11.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_11.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_11.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_11.1)
// CHECK:STDOUT:   %.loc14_11.4: ref %TrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_11.3: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc14_11.5: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc14_11.3 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %Destroy.facet.loc14_11.4: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc14_11.6: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc14_11.4 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_11.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_11.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.5(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Derived.elem.179 = name_ref a, @Derived.%field_decl [concrete = @Derived.%field_decl]
// CHECK:STDOUT:   %.loc14_11.1: ref %HasTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_11.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_11.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_11.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc14_11.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_11.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_11.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_11.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_11.1)
// CHECK:STDOUT:   %base.ref: %Derived.elem.dbc = name_ref base, @Derived.%.loc6 [concrete = @Derived.%.loc6]
// CHECK:STDOUT:   %.loc14_11.4: ref %TrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_11.3: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc14_11.5: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc14_11.3 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %Destroy.facet.loc14_11.4: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc14_11.6: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc14_11.4 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_11.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_11.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.f45: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190: type = facet_access_type %T.f45 [symbolic]
// CHECK:STDOUT:   %GenericClass.9e4: type = class_type @GenericClass, @GenericClass(%T.f45) [symbolic]
// CHECK:STDOUT:   %GenericClass.elem.70a: type = unbound_element_type %GenericClass.9e4, %T.as_type.190 [symbolic]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc5_63.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %GenericClass.cd3: type = class_type @GenericClass, @GenericClass(%Destroy.facet.b89) [concrete]
// CHECK:STDOUT:   %pattern_type.112: type = pattern_type %GenericClass.cd3 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericClass.elem.826: type = unbound_element_type %GenericClass.cd3, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.f48: type = struct_type {.x: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_23.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.28c: %SubobjectDestroy.type = facet_value %GenericClass.cd3, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e38: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.28c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.5c4: %Destroy.type = facet_value %GenericClass.cd3, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.e38) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.112 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.112 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.f7a: @GenericClass.%GenericClass.elem (%GenericClass.elem.70a) = import_ref Main//types, loc40_8, loaded [concrete = %field_decl]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Main.import_ref.eb2 = import_ref Main//types, loc40_10, unloaded
// CHECK:STDOUT:   %field_decl: @GenericClass.%GenericClass.elem (%GenericClass.elem.70a) = field_decl x, element0, %Main.import_ref.eb2 in [concrete] {}
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TrivialGenericClass.ref.loc10: type = name_ref TrivialGenericClass, file.%TrivialGenericClass [concrete = constants.%GenericClass.cd3]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %TrivialGenericClass.ref.loc10, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.28c]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %TrivialGenericClass.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.28c]
// CHECK:STDOUT:   %TrivialGenericClass.ref.loc11: type = name_ref TrivialGenericClass, file.%TrivialGenericClass [concrete = constants.%GenericClass.cd3]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %TrivialGenericClass.ref.loc11, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e38) [concrete = constants.%Destroy.facet.5c4]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %TrivialGenericClass.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.5c4]
// CHECK:STDOUT:   %TrivialGenericClass.ref.loc12: type = name_ref TrivialGenericClass, file.%TrivialGenericClass [concrete = constants.%GenericClass.cd3]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %TrivialGenericClass.ref.loc12, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e38) [concrete = constants.%Destroy.facet.5c4]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %TrivialGenericClass.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet.5c4]
// CHECK:STDOUT:   %x.var: ref %GenericClass.cd3 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %GenericClass.cd3 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %GenericClass.cd3 to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %TrivialGenericClass.ref.loc14: type = name_ref TrivialGenericClass, file.%TrivialGenericClass [concrete = constants.%GenericClass.cd3]
// CHECK:STDOUT:   %x: ref %GenericClass.cd3 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.112 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.112 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%GenericClass.cd3, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e38) [concrete = constants.%Destroy.facet.5c4]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%GenericClass.cd3, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.5c4]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%GenericClass.cd3, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e38) [concrete = constants.%Destroy.facet.5c4]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%GenericClass.cd3, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.5c4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_23.1(%self.param: ref %struct_type.x.f48) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_23.1: ref %HasTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_23.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_23.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_23.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc10_23.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_23.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_23.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_23.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_23.2(%self.param: ref %GenericClass.cd3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %GenericClass.elem.826 = name_ref x, imports.%Main.import_ref.f7a [concrete = imports.%field_decl]
// CHECK:STDOUT:   %.loc10_23.1: ref %HasTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_23.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_23.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_23.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc10_23.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_23.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_23.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_23.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_dynamic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialDynamicClass: type = class_type @TrivialDynamicClass [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %pattern_type.077: type = pattern_type %TrivialDynamicClass [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_23.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c1a: %SubobjectDestroy.type = facet_value %ptr.454, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cad: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c1a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.50f: %Destroy.type = facet_value %ptr.454, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.cad) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_23.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.52e: %SubobjectDestroy.type = facet_value %TrivialDynamicClass, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1a4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.52e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.1f0: %Destroy.type = facet_value %TrivialDynamicClass, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.077 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.077 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.TrivialDynamicClass: type = import_ref Main//types, TrivialDynamicClass, loaded [concrete = constants.%TrivialDynamicClass]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TrivialDynamicClass.ref.loc9: type = name_ref TrivialDynamicClass, imports.%Main.TrivialDynamicClass [concrete = constants.%TrivialDynamicClass]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %TrivialDynamicClass.ref.loc9, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.52e]
// CHECK:STDOUT:   %.loc9: %SubobjectDestroy.type = converted %TrivialDynamicClass.ref.loc9, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.52e]
// CHECK:STDOUT:   %TrivialDynamicClass.ref.loc10: type = name_ref TrivialDynamicClass, imports.%Main.TrivialDynamicClass [concrete = constants.%TrivialDynamicClass]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc10: %Destroy.type = facet_value %TrivialDynamicClass.ref.loc10, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.1a4) [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %.loc10: %Destroy.type = converted %TrivialDynamicClass.ref.loc10, %Destroy.facet.loc10 [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %TrivialDynamicClass.ref.loc11: type = name_ref TrivialDynamicClass, imports.%Main.TrivialDynamicClass [concrete = constants.%TrivialDynamicClass]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %TrivialDynamicClass.ref.loc11, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.1a4) [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %TrivialDynamicClass.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %x.var: ref %TrivialDynamicClass = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %TrivialDynamicClass = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %TrivialDynamicClass to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %TrivialDynamicClass.ref.loc13: type = name_ref TrivialDynamicClass, imports.%Main.TrivialDynamicClass [concrete = constants.%TrivialDynamicClass]
// CHECK:STDOUT:   %x: ref %TrivialDynamicClass = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.077 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.077 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%TrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.1a4) [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%TrivialDynamicClass, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%TrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.1a4) [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%TrivialDynamicClass, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_23.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_23.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_23.1: ref %ptr.454 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_23.1: %Destroy.type = facet_value constants.%ptr.454, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.cad) [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %.loc9_23.2: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc9_23.1 [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %Destroy.facet.loc9_23.2: %Destroy.type = facet_value constants.%ptr.454, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.cad) [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %.loc9_23.3: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc9_23.2 [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc9_23.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_23.3(%self.param: ref %TrivialDynamicClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_dynamic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %TrivialDynamicClass: type = class_type @TrivialDynamicClass [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %TrivialDynamicClass [concrete]
// CHECK:STDOUT:   %TrivialDynamicClass.F.type: type = fn_type @TrivialDynamicClass.F [concrete]
// CHECK:STDOUT:   %TrivialDynamicClass.F: %TrivialDynamicClass.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.6c4: type = struct_type {.base: %TrivialDynamicClass} [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c1a: %SubobjectDestroy.type = facet_value %ptr.454, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cad: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c1a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.50f: %Destroy.type = facet_value %ptr.454, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.cad) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.52e: %SubobjectDestroy.type = facet_value %TrivialDynamicClass, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1a4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.52e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.1f0: %Destroy.type = facet_value %TrivialDynamicClass, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %Derived, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.660: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f2a: %Destroy.type = facet_value %Derived, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.660) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:   %Derived.ref.loc14: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc14: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc14: %Destroy.type = facet_value %Derived.ref.loc14, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %.loc14: %Destroy.type = converted %Derived.ref.loc14, %Destroy.facet.loc14 [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %Derived.ref.loc15: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc15: %Destroy.type = facet_value %Derived.ref.loc15, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %.loc15: %Destroy.type = converted %Derived.ref.loc15, %Destroy.facet.loc15 [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %x.var: ref %Derived = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc17_3.1: ref %Derived = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc17_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Derived.ref.loc17: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %x: ref %Derived = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%Derived, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%Derived, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %.loc17_3.3: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_11.1: ref %ptr.454 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_11.1: %Destroy.type = facet_value constants.%ptr.454, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.cad) [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %.loc13_11.2: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc13_11.1 [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %Destroy.facet.loc13_11.2: %Destroy.type = facet_value constants.%ptr.454, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.cad) [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %.loc13_11.3: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc13_11.2 [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_11.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.3(%self.param: ref %TrivialDynamicClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.4(%self.param: ref %struct_type.base.6c4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_11.1: ref %TrivialDynamicClass = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_11.1: %Destroy.type = facet_value constants.%TrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.1a4) [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %.loc13_11.2: %Destroy.type = converted constants.%TrivialDynamicClass, %Destroy.facet.loc13_11.1 [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %Destroy.facet.loc13_11.2: %Destroy.type = facet_value constants.%TrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.1a4) [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %.loc13_11.3: %Destroy.type = converted constants.%TrivialDynamicClass, %Destroy.facet.loc13_11.2 [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_11.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.5(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %Derived.elem = name_ref base, @Derived.%.loc6 [concrete = @Derived.%.loc6]
// CHECK:STDOUT:   %.loc13_11.1: ref %TrivialDynamicClass = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_11.1: %Destroy.type = facet_value constants.%TrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.1a4) [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %.loc13_11.2: %Destroy.type = converted constants.%TrivialDynamicClass, %Destroy.facet.loc13_11.1 [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %Destroy.facet.loc13_11.2: %Destroy.type = facet_value constants.%TrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.1a4) [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   %.loc13_11.3: %Destroy.type = converted constants.%TrivialDynamicClass, %Destroy.facet.loc13_11.2 [concrete = constants.%Destroy.facet.1f0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_11.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_trivial_class_derived_from_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Derived.ref.loc22: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc22: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Derived.ref.loc31: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc31: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Derived.ref.loc40: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc40: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %x.var: ref %Derived = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc56: ref %Derived = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc56 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Derived.ref.loc56: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %x: ref %Derived = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
