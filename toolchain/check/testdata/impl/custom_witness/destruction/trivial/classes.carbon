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

class ImplsDestroy {
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

class TrivialFields {
  class HasTrivialDestruction {}
  class AlsoHasTrivialDestruction {}

  var a: HasTrivialDestruction;
  var b: AlsoHasTrivialDestruction;
}

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

class TrivialFields {
  class HasTrivialDestruction {}
  class AlsoHasTrivialDestruction {}

  var a: HasTrivialDestruction;
  var b: AlsoHasTrivialDestruction;
}

class NestedTrivialFields {
  var x: TrivialFields;
}

fn Make() -> NestedTrivialFields;

fn Test() {
  //@dump-sem-ir-begin
  NestedTrivialFields as Core.SubobjectDestroy;
  NestedTrivialFields as Core.Destroy;
  NestedTrivialFields as Core.TrivialDestroy;

  var unused x: NestedTrivialFields = Make();
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

base class NonTrivialBase {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

base class DerivedFromNonTrivialBase {
  extend base: NonTrivialBase;
  var y: ImplsDestroy;
}

fn Test() {
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
  var y: ImplsDestroy;
}

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

class HasTrivialDestruction {}
class GenericClass(T: Core.Destroy) {
  var x: T;
}

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

class GenericClass(T: Core.Destroy) {
  var x: T;
}

fn Test() {
  GenericClass(ImplsDestroy) as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_base_and_trivial_field.carbon
library "[[@TEST_NAME]]";

import library "types";

base class NonTrivialBase {
  var x: ImplsDestroy;

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

abstract class TrivialAbstract {}

class Derived {
  extend base: TrivialAbstract;
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
  // CHECK:STDERR:        ·      ┃        `Core.Destroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- todo_trivial_empty_class.carbon = $todo_trivial_empty_class
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
// CHECK:STDOUT: --- todo_trivial_all_trivial_fields.carbon = $todo_trivial_all_trivial_fields
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialFields: type = class_type @TrivialFields [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %AlsoHasTrivialDestruction: type = class_type @AlsoHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TrivialFields.elem.64f: type = unbound_element_type %TrivialFields, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %TrivialFields.elem.fcd: type = unbound_element_type %TrivialFields, %AlsoHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.a.b.841: type = struct_type {.a: %HasTrivialDestruction, .b: %AlsoHasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %pattern_type.063: type = pattern_type %TrivialFields [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_17.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_17.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9b3: %SubobjectDestroy.type = facet_value %AlsoHasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.092: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9b3) [concrete]
// CHECK:STDOUT:   %Destroy.facet.79d: %Destroy.type = facet_value %AlsoHasTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.092) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.344: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.03b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.344) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c0d: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.03b) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_17.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.16d: %SubobjectDestroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c88: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.16d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.bc5: %Destroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.c88) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.063 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.063 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %TrivialFields.ref.loc15: type = name_ref TrivialFields, file.%TrivialFields.decl [concrete = constants.%TrivialFields]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %TrivialFields.ref.loc15, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.16d]
// CHECK:STDOUT:   %.loc15: %SubobjectDestroy.type = converted %TrivialFields.ref.loc15, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.16d]
// CHECK:STDOUT:   %TrivialFields.ref.loc16: type = name_ref TrivialFields, file.%TrivialFields.decl [concrete = constants.%TrivialFields]
// CHECK:STDOUT:   %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc16: %Destroy.type = facet_value %TrivialFields.ref.loc16, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc16: %Destroy.type = converted %TrivialFields.ref.loc16, %Destroy.facet.loc16 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %TrivialFields.ref.loc17: type = name_ref TrivialFields, file.%TrivialFields.decl [concrete = constants.%TrivialFields]
// CHECK:STDOUT:   %Core.ref.loc17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc17: %Destroy.type = facet_value %TrivialFields.ref.loc17, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc17: %Destroy.type = converted %TrivialFields.ref.loc17, %Destroy.facet.loc17 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %x.var: ref %TrivialFields = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc19_3.1: ref %TrivialFields = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %TrivialFields to %.loc19_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %TrivialFields.ref.loc19: type = name_ref TrivialFields, file.%TrivialFields.decl [concrete = constants.%TrivialFields]
// CHECK:STDOUT:   %x: ref %TrivialFields = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.063 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.063 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc19_3.1: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc19_3.2: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc19_3.1 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %Destroy.facet.loc19_3.2: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc19_3.3: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc19_3.2 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_17.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_17.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_17.3(%self.param: ref %AlsoHasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_17.4(%self.param: ref %struct_type.a.b.841) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_17.1: ref %AlsoHasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc15_17.1: %Destroy.type = facet_value constants.%AlsoHasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.092) [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %.loc15_17.2: %Destroy.type = converted constants.%AlsoHasTrivialDestruction, %Destroy.facet.loc15_17.1 [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %Destroy.facet.loc15_17.2: %Destroy.type = facet_value constants.%AlsoHasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.092) [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %.loc15_17.3: %Destroy.type = converted constants.%AlsoHasTrivialDestruction, %Destroy.facet.loc15_17.2 [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_17.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc15_17.1)
// CHECK:STDOUT:   %.loc15_17.4: ref %HasTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_17.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.03b) [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %.loc15_17.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_17.3 [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %Destroy.facet.loc15_17.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.03b) [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %.loc15_17.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_17.4 [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_17.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc15_17.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_17.5(%self.param: ref %TrivialFields) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: %TrivialFields.elem.fcd = name_ref b, @TrivialFields.%field_decl.loc8 [concrete = @TrivialFields.%field_decl.loc8]
// CHECK:STDOUT:   %.loc15_17.1: ref %AlsoHasTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc15_17.1: %Destroy.type = facet_value constants.%AlsoHasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.092) [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %.loc15_17.2: %Destroy.type = converted constants.%AlsoHasTrivialDestruction, %Destroy.facet.loc15_17.1 [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %Destroy.facet.loc15_17.2: %Destroy.type = facet_value constants.%AlsoHasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.092) [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %.loc15_17.3: %Destroy.type = converted constants.%AlsoHasTrivialDestruction, %Destroy.facet.loc15_17.2 [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_17.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc15_17.1)
// CHECK:STDOUT:   %a.ref: %TrivialFields.elem.64f = name_ref a, @TrivialFields.%field_decl.loc7 [concrete = @TrivialFields.%field_decl.loc7]
// CHECK:STDOUT:   %.loc15_17.4: ref %HasTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_17.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.03b) [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %.loc15_17.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_17.3 [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %Destroy.facet.loc15_17.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.03b) [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %.loc15_17.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_17.4 [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_17.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc15_17.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_nested_trivial_fields.carbon = $todo_trivial_nested_trivial_fields
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialFields: type = class_type @TrivialFields [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %AlsoHasTrivialDestruction: type = class_type @AlsoHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TrivialFields.elem.64f: type = unbound_element_type %TrivialFields, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %TrivialFields.elem.fcd: type = unbound_element_type %TrivialFields, %AlsoHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.a.b.841: type = struct_type {.a: %HasTrivialDestruction, .b: %AlsoHasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %NestedTrivialFields: type = class_type @NestedTrivialFields [concrete]
// CHECK:STDOUT:   %NestedTrivialFields.elem: type = unbound_element_type %NestedTrivialFields, %TrivialFields [concrete]
// CHECK:STDOUT:   %struct_type.x.156: type = struct_type {.x: %TrivialFields} [concrete]
// CHECK:STDOUT:   %pattern_type.d18: type = pattern_type %NestedTrivialFields [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_23.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_23.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9b3: %SubobjectDestroy.type = facet_value %AlsoHasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.092: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9b3) [concrete]
// CHECK:STDOUT:   %Destroy.facet.79d: %Destroy.type = facet_value %AlsoHasTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.092) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.344: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.03b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.344) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c0d: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.03b) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_23.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.16d: %SubobjectDestroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c88: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.16d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.bc5: %Destroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.c88) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_23.7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.23d: %SubobjectDestroy.type = facet_value %NestedTrivialFields, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.810: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.23d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.1c1: %Destroy.type = facet_value %NestedTrivialFields, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.810) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d18 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.d18 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %NestedTrivialFields.ref.loc19: type = name_ref NestedTrivialFields, file.%NestedTrivialFields.decl [concrete = constants.%NestedTrivialFields]
// CHECK:STDOUT:   %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %NestedTrivialFields.ref.loc19, (constants.%custom_witness.b80ef6.7) [concrete = constants.%SubobjectDestroy.facet.23d]
// CHECK:STDOUT:   %.loc19: %SubobjectDestroy.type = converted %NestedTrivialFields.ref.loc19, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.23d]
// CHECK:STDOUT:   %NestedTrivialFields.ref.loc20: type = name_ref NestedTrivialFields, file.%NestedTrivialFields.decl [concrete = constants.%NestedTrivialFields]
// CHECK:STDOUT:   %Core.ref.loc20: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc20: %Destroy.type = facet_value %NestedTrivialFields.ref.loc20, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.810) [concrete = constants.%Destroy.facet.1c1]
// CHECK:STDOUT:   %.loc20: %Destroy.type = converted %NestedTrivialFields.ref.loc20, %Destroy.facet.loc20 [concrete = constants.%Destroy.facet.1c1]
// CHECK:STDOUT:   %NestedTrivialFields.ref.loc21: type = name_ref NestedTrivialFields, file.%NestedTrivialFields.decl [concrete = constants.%NestedTrivialFields]
// CHECK:STDOUT:   %Core.ref.loc21: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc21: %Destroy.type = facet_value %NestedTrivialFields.ref.loc21, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.810) [concrete = constants.%Destroy.facet.1c1]
// CHECK:STDOUT:   %.loc21: %Destroy.type = converted %NestedTrivialFields.ref.loc21, %Destroy.facet.loc21 [concrete = constants.%Destroy.facet.1c1]
// CHECK:STDOUT:   %x.var: ref %NestedTrivialFields = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc23_3.1: ref %NestedTrivialFields = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NestedTrivialFields to %.loc23_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %NestedTrivialFields.ref.loc23: type = name_ref NestedTrivialFields, file.%NestedTrivialFields.decl [concrete = constants.%NestedTrivialFields]
// CHECK:STDOUT:   %x: ref %NestedTrivialFields = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d18 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d18 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc23_3.1: %Destroy.type = facet_value constants.%NestedTrivialFields, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.810) [concrete = constants.%Destroy.facet.1c1]
// CHECK:STDOUT:   %.loc23_3.2: %Destroy.type = converted constants.%NestedTrivialFields, %Destroy.facet.loc23_3.1 [concrete = constants.%Destroy.facet.1c1]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.2: %Destroy.type = facet_value constants.%NestedTrivialFields, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.810) [concrete = constants.%Destroy.facet.1c1]
// CHECK:STDOUT:   %.loc23_3.3: %Destroy.type = converted constants.%NestedTrivialFields, %Destroy.facet.loc23_3.2 [concrete = constants.%Destroy.facet.1c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.3(%self.param: ref %AlsoHasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.4(%self.param: ref %struct_type.a.b.841) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_23.1: ref %AlsoHasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc19_23.1: %Destroy.type = facet_value constants.%AlsoHasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.092) [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %.loc19_23.2: %Destroy.type = converted constants.%AlsoHasTrivialDestruction, %Destroy.facet.loc19_23.1 [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %Destroy.facet.loc19_23.2: %Destroy.type = facet_value constants.%AlsoHasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.092) [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %.loc19_23.3: %Destroy.type = converted constants.%AlsoHasTrivialDestruction, %Destroy.facet.loc19_23.2 [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_23.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc19_23.1)
// CHECK:STDOUT:   %.loc19_23.4: ref %HasTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_23.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.03b) [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %.loc19_23.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc19_23.3 [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %Destroy.facet.loc19_23.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.03b) [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %.loc19_23.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc19_23.4 [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_23.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc19_23.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.5(%self.param: ref %TrivialFields) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: %TrivialFields.elem.fcd = name_ref b, @TrivialFields.%field_decl.loc8 [concrete = @TrivialFields.%field_decl.loc8]
// CHECK:STDOUT:   %.loc19_23.1: ref %AlsoHasTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc19_23.1: %Destroy.type = facet_value constants.%AlsoHasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.092) [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %.loc19_23.2: %Destroy.type = converted constants.%AlsoHasTrivialDestruction, %Destroy.facet.loc19_23.1 [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %Destroy.facet.loc19_23.2: %Destroy.type = facet_value constants.%AlsoHasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.092) [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   %.loc19_23.3: %Destroy.type = converted constants.%AlsoHasTrivialDestruction, %Destroy.facet.loc19_23.2 [concrete = constants.%Destroy.facet.79d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_23.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc19_23.1)
// CHECK:STDOUT:   %a.ref: %TrivialFields.elem.64f = name_ref a, @TrivialFields.%field_decl.loc7 [concrete = @TrivialFields.%field_decl.loc7]
// CHECK:STDOUT:   %.loc19_23.4: ref %HasTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_23.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.03b) [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %.loc19_23.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc19_23.3 [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %Destroy.facet.loc19_23.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.03b) [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   %.loc19_23.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc19_23.4 [concrete = constants.%Destroy.facet.c0d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_23.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc19_23.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.6(%self.param: ref %struct_type.x.156) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_23.1: ref %TrivialFields = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_23.1: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc19_23.2: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc19_23.1 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %Destroy.facet.loc19_23.2: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc19_23.3: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc19_23.2 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc19_23.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_23.7(%self.param: ref %NestedTrivialFields) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NestedTrivialFields.elem = name_ref x, @NestedTrivialFields.%field_decl [concrete = @NestedTrivialFields.%field_decl]
// CHECK:STDOUT:   %.loc19_23.1: ref %TrivialFields = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_23.1: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc19_23.2: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc19_23.1 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %Destroy.facet.loc19_23.2: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc19_23.3: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc19_23.2 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc19_23.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_derived_from_trivial_base.carbon = $todo_trivial_derived_from_trivial_base
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
// CHECK:STDOUT: --- todo_trivial_generic_class.carbon = $todo_trivial_generic_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
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
// CHECK:STDOUT:   %pattern_type.ff3: type = pattern_type %GenericClass.422 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericClass.elem.570: type = unbound_element_type %GenericClass.422, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.f48: type = struct_type {.x: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_23.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.542: %SubobjectDestroy.type = facet_value %GenericClass.422, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f8b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.542) [concrete]
// CHECK:STDOUT:   %Destroy.facet.43b: %Destroy.type = facet_value %GenericClass.422, (%DestroyImpl.impl_witness.f8b, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.ff3 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.ff3 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TrivialGenericClass.ref.loc13: type = name_ref TrivialGenericClass, file.%TrivialGenericClass [concrete = constants.%GenericClass.422]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %TrivialGenericClass.ref.loc13, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.542]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %TrivialGenericClass.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.542]
// CHECK:STDOUT:   %TrivialGenericClass.ref.loc14: type = name_ref TrivialGenericClass, file.%TrivialGenericClass [concrete = constants.%GenericClass.422]
// CHECK:STDOUT:   %Core.ref.loc14: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc14: %Destroy.type = facet_value %TrivialGenericClass.ref.loc14, (constants.%DestroyImpl.impl_witness.f8b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   %.loc14: %Destroy.type = converted %TrivialGenericClass.ref.loc14, %Destroy.facet.loc14 [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   %TrivialGenericClass.ref.loc15: type = name_ref TrivialGenericClass, file.%TrivialGenericClass [concrete = constants.%GenericClass.422]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc15: %Destroy.type = facet_value %TrivialGenericClass.ref.loc15, (constants.%DestroyImpl.impl_witness.f8b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   %.loc15: %Destroy.type = converted %TrivialGenericClass.ref.loc15, %Destroy.facet.loc15 [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   %x.var: ref %GenericClass.422 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc17_3.1: ref %GenericClass.422 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %GenericClass.422 to %.loc17_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %TrivialGenericClass.ref.loc17: type = name_ref TrivialGenericClass, file.%TrivialGenericClass [concrete = constants.%GenericClass.422]
// CHECK:STDOUT:   %x: ref %GenericClass.422 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.ff3 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.ff3 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%GenericClass.422, (constants.%DestroyImpl.impl_witness.f8b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%GenericClass.422, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%GenericClass.422, (constants.%DestroyImpl.impl_witness.f8b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   %.loc17_3.3: %Destroy.type = converted constants.%GenericClass.422, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet.43b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_23.1(%self.param: ref %struct_type.x.f48) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_23.1: ref %HasTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_23.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc13_23.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_23.1 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc13_23.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc13_23.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_23.2 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_23.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_23.2(%self.param: ref %GenericClass.422) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %GenericClass.elem.570 = name_ref x, @GenericClass.%field_decl [concrete = @GenericClass.%field_decl]
// CHECK:STDOUT:   %.loc13_23.1: ref %HasTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_23.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc13_23.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_23.1 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc13_23.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc13_23.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_23.2 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_23.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_dynamic_class.carbon = $todo_trivial_dynamic_class
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
// CHECK:STDOUT: --- todo_trivial_dynamic_base.carbon = $todo_trivial_dynamic_base
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
// CHECK:STDOUT: --- fail_todo_trivial_class_derived_from_abstract.carbon = $fail_todo_trivial_class_derived_from_abstract
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
