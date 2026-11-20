// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/nontrivial/classes.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/nontrivial/classes.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

// `HasNonTrivialDestruction.(SubobjectDestroy.Op)` is a no-op because it
// doesn't have any fields that require it to be non-trivial.
// Similarly for other types like this.
class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- nontrivial_empty_class_impls_destroy.carbon
library "[[@TEST_NAME]]";

import library "types";

fn Make() -> HasNonTrivialDestruction;

fn Test() {
  //@dump-sem-ir-begin
  HasNonTrivialDestruction as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  HasNonTrivialDestruction as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: HasNonTrivialDestruction = Make();
  //@dump-sem-ir-end

}

// --- nontrivial_all_nontrivial_fields.carbon
library "[[@TEST_NAME]]";

import library "types";

class AlsoHasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

class AllNonTrivialFields {
  var a: HasNonTrivialDestruction;
  var b: AlsoHasNonTrivialDestruction;
  var c: AlsoHasNonTrivialDestruction;
  var d: HasNonTrivialDestruction;
}

fn Make() -> AllNonTrivialFields;

fn Test() {
  //@dump-sem-ir-begin
  AllNonTrivialFields as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  AllNonTrivialFields as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: AllNonTrivialFields = Make();
  //@dump-sem-ir-end
}

// --- nontrivial_one_trivial_field.carbon
library "[[@TEST_NAME]]";

import library "types";

class AlsoHasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

class HasTrivialDestruction {}

class OneTrivialField {
  var a: HasNonTrivialDestruction;
  var b: AlsoHasNonTrivialDestruction;
  var c: HasTrivialDestruction;
  var d: HasNonTrivialDestruction;
}

fn Make() -> OneTrivialField;

fn Test() {
  //@dump-sem-ir-begin
  OneTrivialField as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  OneTrivialField as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: OneTrivialField = Make();
  //@dump-sem-ir-end
}

// --- nontrivial_nested_nontrivial_member.carbon
library "[[@TEST_NAME]]";

import library "types";

class HasNonTrivialField {
  var x: HasNonTrivialDestruction;
}

class NestedNonTrivialMember {
  var x: HasNonTrivialField;
}

fn Make() -> NestedNonTrivialMember;

fn Test() {
  //@dump-sem-ir-begin
  NestedNonTrivialMember as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  NestedNonTrivialMember as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: NestedNonTrivialMember = Make();
  //@dump-sem-ir-end
}

// --- nontrivial_derived_from_nontrivial_base.carbon
library "[[@TEST_NAME]]";

import library "types";

base class NonTrivialBase {
  var x: HasNonTrivialDestruction;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

base class DerivedFromNonTrivialBase {
  extend base: NonTrivialBase;
  var y: HasNonTrivialDestruction;
}

fn Make() -> DerivedFromNonTrivialBase;

fn Test() {
  //@dump-sem-ir-begin
  DerivedFromNonTrivialBase as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  DerivedFromNonTrivialBase as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: DerivedFromNonTrivialBase = Make();
  //@dump-sem-ir-end
}

// --- nontrivial_grandparent.carbon
library "[[@TEST_NAME]]";

import library "types";

base class NonTrivialBase {
  var x: HasNonTrivialDestruction;

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

fn Make() -> NonTrivialGrandparent;

fn Test() {
  //@dump-sem-ir-begin
  NonTrivialGrandparent as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  NonTrivialGrandparent as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: NonTrivialGrandparent = Make();
  //@dump-sem-ir-end
}

// --- nontrivial_derived_has_trivial_base_and_impls_destroy.carbon
library "[[@TEST_NAME]]";

base class TrivialBase {}

class Derived {
  extend base: TrivialBase;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Make() -> Derived;

fn Test() {
  //@dump-sem-ir-begin
  Derived as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  Derived as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: Derived = Make();
  //@dump-sem-ir-end
}

// --- nontrivial_generic_class.carbon
library "[[@TEST_NAME]]";

import library "types";

class GenericClass(T: Core.Destroy) {
  var x: T;
}

fn Make() -> GenericClass(HasNonTrivialDestruction);

fn Test() {
  //@dump-sem-ir-begin
  GenericClass(HasNonTrivialDestruction) as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  GenericClass(HasNonTrivialDestruction) as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: GenericClass(HasNonTrivialDestruction) = Make();
  //@dump-sem-ir-end
}

// --- nontrivial_base_and_trivial_field.carbon
library "[[@TEST_NAME]]";

import library "types";

base class NonTrivialBase {
  var x: HasNonTrivialDestruction;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class HasTrivialDestruction {}

class NonTrivialBaseAndTrivialMember {
  extend base: NonTrivialBase;
  var y: HasTrivialDestruction;
}

fn Make() -> NonTrivialBaseAndTrivialMember;

fn Test() {
  //@dump-sem-ir-begin
  NonTrivialBaseAndTrivialMember as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  NonTrivialBaseAndTrivialMember as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: NonTrivialBaseAndTrivialMember = Make();
  //@dump-sem-ir-end
}

// --- nontrivial_trivial_base_and_nontrivial_field.carbon
library "[[@TEST_NAME]]";

import library "types";

base class TrivialBase {}

class TrivialBaseAndNonTrivialMember {
  extend base: TrivialBase;
  var x: HasNonTrivialDestruction;
}

fn Make() -> TrivialBaseAndNonTrivialMember;

fn Test() {
  //@dump-sem-ir-begin
  TrivialBaseAndNonTrivialMember as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  TrivialBaseAndNonTrivialMember as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: TrivialBaseAndNonTrivialMember = Make();
  //@dump-sem-ir-end
}

// --- nontrivial_class_dynamic.carbon
library "[[@TEST_NAME]]";

base class NonTrivialDynamicClass {
  virtual fn F(self);

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Make() -> NonTrivialDynamicClass;

fn Test() {
  //@dump-sem-ir-begin
  NonTrivialDynamicClass as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  NonTrivialDynamicClass as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: NonTrivialDynamicClass = Make();
  //@dump-sem-ir-end
}

// --- nontrivial_dynamic_base.carbon
library "[[@TEST_NAME]]";

base class NonTrivialDynamicClass {
  virtual fn F(self);

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class Derived {
  extend base: NonTrivialDynamicClass;
}

fn Make() -> Derived;

fn Test() {
  //@dump-sem-ir-begin
  Derived as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  Derived as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: Derived = Make();
  //@dump-sem-ir-end
}

// --- fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon
library "[[@TEST_NAME]]";

// TODO: Support destroying classes that derive from abstract types.

abstract class NonTrivialAbstract {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class Derived {
  extend base: NonTrivialAbstract;
}

fn Make() -> Derived;

fn Test() {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Derived as Core.SubobjectDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.SubobjectDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.SubobjectDestroy;
  //@dump-sem-ir-end
  // CHECK:STDERR:        ╭─┤ fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Derived as Core.Destroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Derived` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Destroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Derived as Core.Destroy;

  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon:[[@LINE+6]]:17
  // CHECK:STDERR: ->     │   var unused x: Class = Make();
  // CHECK:STDERR:        ·                 ━━┳━━
  // CHECK:STDERR:        ·                   ┗━━┫ error: name `Class` not found [NameNotFound]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused x: Class = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- nontrivial_empty_class_impls_destroy.carbon = $nontrivial_empty_class_impls_destroy
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.186: type = pattern_type %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_28.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.69c: %SubobjectDestroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.186 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.186 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.HasNonTrivialDestruction: type = import_ref Main//types, HasNonTrivialDestruction, loaded [concrete = constants.%HasNonTrivialDestruction]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %HasNonTrivialDestruction.ref.loc9: type = name_ref HasNonTrivialDestruction, imports.%Main.HasNonTrivialDestruction [concrete = constants.%HasNonTrivialDestruction]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %HasNonTrivialDestruction.ref.loc9, (constants.%custom_witness.b80ef6.2) [concrete = constants.%SubobjectDestroy.facet.69c]
// CHECK:STDOUT:   %.loc9: %SubobjectDestroy.type = converted %HasNonTrivialDestruction.ref.loc9, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.69c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %HasNonTrivialDestruction = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %HasNonTrivialDestruction = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %HasNonTrivialDestruction to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %HasNonTrivialDestruction.ref.loc14: type = name_ref HasNonTrivialDestruction, imports.%Main.HasNonTrivialDestruction [concrete = constants.%HasNonTrivialDestruction]
// CHECK:STDOUT:   %x: ref %HasNonTrivialDestruction = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.186 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.186 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_28.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_28.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_all_nontrivial_fields.carbon = $nontrivial_all_nontrivial_fields
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction: type = class_type @AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b58: <witness> = impl_witness @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %AllNonTrivialFields: type = class_type @AllNonTrivialFields [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %AllNonTrivialFields.elem.605: type = unbound_element_type %AllNonTrivialFields, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %AllNonTrivialFields.elem.256: type = unbound_element_type %AllNonTrivialFields, %AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.7ad: type = struct_type {.a: %HasNonTrivialDestruction, .b: %AlsoHasNonTrivialDestruction, .c: %AlsoHasNonTrivialDestruction, .d: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %pattern_type.c66: type = pattern_type %AllNonTrivialFields [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22_23.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22_23.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a2f: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.490, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.facet.88e: %Destroy.type = facet_value %AlsoHasNonTrivialDestruction, (%DestroyImpl.impl_witness.b58, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22_23.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.075: %SubobjectDestroy.type = facet_value %AllNonTrivialFields, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.124: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.075) [concrete]
// CHECK:STDOUT:   %Destroy.facet.468: %Destroy.type = facet_value %AllNonTrivialFields, (%DestroyImpl.impl_witness.124, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.c66 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.c66 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AllNonTrivialFields.ref.loc22: type = name_ref AllNonTrivialFields, file.%AllNonTrivialFields.decl [concrete = constants.%AllNonTrivialFields]
// CHECK:STDOUT:   %Core.ref.loc22: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %AllNonTrivialFields.ref.loc22, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.075]
// CHECK:STDOUT:   %.loc22: %SubobjectDestroy.type = converted %AllNonTrivialFields.ref.loc22, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.075]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %AllNonTrivialFields = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc27_3.1: ref %AllNonTrivialFields = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %AllNonTrivialFields to %.loc27_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %AllNonTrivialFields.ref.loc27: type = name_ref AllNonTrivialFields, file.%AllNonTrivialFields.decl [concrete = constants.%AllNonTrivialFields]
// CHECK:STDOUT:   %x: ref %AllNonTrivialFields = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.c66 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c66 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc27_3.1: %Destroy.type = facet_value constants.%AllNonTrivialFields, (constants.%DestroyImpl.impl_witness.124, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.468]
// CHECK:STDOUT:   %.loc27_3.2: %Destroy.type = converted constants.%AllNonTrivialFields, %Destroy.facet.loc27_3.1 [concrete = constants.%Destroy.facet.468]
// CHECK:STDOUT:   %Destroy.facet.loc27_3.2: %Destroy.type = facet_value constants.%AllNonTrivialFields, (constants.%DestroyImpl.impl_witness.124, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.468]
// CHECK:STDOUT:   %.loc27_3.3: %Destroy.type = converted constants.%AllNonTrivialFields, %Destroy.facet.loc27_3.2 [concrete = constants.%Destroy.facet.468]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_23.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_23.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_23.3(%self.param: ref %AlsoHasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_23.4(%self.param: ref %struct_type.a.b.c.d.7ad) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_23.1: ref %HasNonTrivialDestruction = struct_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc22_23.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_23.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_23.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc22_23.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_23.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_23.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_23.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc22_23.1)
// CHECK:STDOUT:   %.loc22_23.4: ref %AlsoHasNonTrivialDestruction = struct_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc22_23.3: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc22_23.5: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc22_23.3 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %Destroy.facet.loc22_23.4: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc22_23.6: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc22_23.4 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_23.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc22_23.4)
// CHECK:STDOUT:   %.loc22_23.7: ref %AlsoHasNonTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc22_23.5: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc22_23.8: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc22_23.5 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %Destroy.facet.loc22_23.6: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc22_23.9: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc22_23.6 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_23.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc22_23.7)
// CHECK:STDOUT:   %.loc22_23.10: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc22_23.7: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_23.11: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_23.7 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc22_23.8: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_23.12: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_23.8 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_23.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc22_23.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_23.5(%self.param: ref %AllNonTrivialFields) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %AllNonTrivialFields.elem.605 = name_ref d, @AllNonTrivialFields.%field_decl.loc15 [concrete = @AllNonTrivialFields.%field_decl.loc15]
// CHECK:STDOUT:   %.loc22_23.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc22_23.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_23.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_23.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc22_23.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_23.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_23.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_23.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc22_23.1)
// CHECK:STDOUT:   %c.ref: %AllNonTrivialFields.elem.256 = name_ref c, @AllNonTrivialFields.%field_decl.loc14 [concrete = @AllNonTrivialFields.%field_decl.loc14]
// CHECK:STDOUT:   %.loc22_23.4: ref %AlsoHasNonTrivialDestruction = class_element_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc22_23.3: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc22_23.5: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc22_23.3 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %Destroy.facet.loc22_23.4: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc22_23.6: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc22_23.4 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_23.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc22_23.4)
// CHECK:STDOUT:   %b.ref: %AllNonTrivialFields.elem.256 = name_ref b, @AllNonTrivialFields.%field_decl.loc13 [concrete = @AllNonTrivialFields.%field_decl.loc13]
// CHECK:STDOUT:   %.loc22_23.7: ref %AlsoHasNonTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc22_23.5: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc22_23.8: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc22_23.5 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %Destroy.facet.loc22_23.6: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc22_23.9: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc22_23.6 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_23.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc22_23.7)
// CHECK:STDOUT:   %a.ref: %AllNonTrivialFields.elem.605 = name_ref a, @AllNonTrivialFields.%field_decl.loc12 [concrete = @AllNonTrivialFields.%field_decl.loc12]
// CHECK:STDOUT:   %.loc22_23.10: ref %HasNonTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc22_23.7: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_23.11: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_23.7 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc22_23.8: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_23.12: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_23.8 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_23.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc22_23.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_one_trivial_field.carbon = $nontrivial_one_trivial_field
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction: type = class_type @AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b58: <witness> = impl_witness @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %OneTrivialField: type = class_type @OneTrivialField [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %OneTrivialField.elem.197: type = unbound_element_type %OneTrivialField, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %OneTrivialField.elem.88e: type = unbound_element_type %OneTrivialField, %AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %OneTrivialField.elem.310: type = unbound_element_type %OneTrivialField, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.23f: type = struct_type {.a: %HasNonTrivialDestruction, .b: %AlsoHasNonTrivialDestruction, .c: %HasTrivialDestruction, .d: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %pattern_type.494: type = pattern_type %OneTrivialField [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_19.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_19.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_19.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a2f: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.490, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dba: %Destroy.type = facet_value %HasTrivialDestruction, (%DestroyImpl.impl_witness.4ca, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.facet.88e: %Destroy.type = facet_value %AlsoHasNonTrivialDestruction, (%DestroyImpl.impl_witness.b58, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_19.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c4c: %SubobjectDestroy.type = facet_value %OneTrivialField, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6ed: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c4c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0bd: %Destroy.type = facet_value %OneTrivialField, (%DestroyImpl.impl_witness.6ed, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.494 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.494 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %OneTrivialField.ref.loc24: type = name_ref OneTrivialField, file.%OneTrivialField.decl [concrete = constants.%OneTrivialField]
// CHECK:STDOUT:   %Core.ref.loc24: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %OneTrivialField.ref.loc24, (constants.%custom_witness.b80ef6.6) [concrete = constants.%SubobjectDestroy.facet.c4c]
// CHECK:STDOUT:   %.loc24: %SubobjectDestroy.type = converted %OneTrivialField.ref.loc24, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.c4c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %OneTrivialField = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc29_3.1: ref %OneTrivialField = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %OneTrivialField to %.loc29_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %OneTrivialField.ref.loc29: type = name_ref OneTrivialField, file.%OneTrivialField.decl [concrete = constants.%OneTrivialField]
// CHECK:STDOUT:   %x: ref %OneTrivialField = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.494 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.494 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc29_3.1: %Destroy.type = facet_value constants.%OneTrivialField, (constants.%DestroyImpl.impl_witness.6ed, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:   %.loc29_3.2: %Destroy.type = converted constants.%OneTrivialField, %Destroy.facet.loc29_3.1 [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:   %Destroy.facet.loc29_3.2: %Destroy.type = facet_value constants.%OneTrivialField, (constants.%DestroyImpl.impl_witness.6ed, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:   %.loc29_3.3: %Destroy.type = converted constants.%OneTrivialField, %Destroy.facet.loc29_3.2 [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.3(%self.param: ref %AlsoHasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.4(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.5(%self.param: ref %struct_type.a.b.c.d.23f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_19.1: ref %HasNonTrivialDestruction = struct_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc24_19.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_19.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_19.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc24_19.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_19.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_19.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_19.1)
// CHECK:STDOUT:   %.loc24_19.4: ref %HasTrivialDestruction = struct_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc24_19.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc24_19.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc24_19.3 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc24_19.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc24_19.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc24_19.4 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_19.4)
// CHECK:STDOUT:   %.loc24_19.7: ref %AlsoHasNonTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc24_19.5: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc24_19.8: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc24_19.5 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %Destroy.facet.loc24_19.6: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc24_19.9: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc24_19.6 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc24_19.7)
// CHECK:STDOUT:   %.loc24_19.10: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc24_19.7: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_19.11: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_19.7 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc24_19.8: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_19.12: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_19.8 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc24_19.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.6(%self.param: ref %OneTrivialField) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %OneTrivialField.elem.197 = name_ref d, @OneTrivialField.%field_decl.loc17 [concrete = @OneTrivialField.%field_decl.loc17]
// CHECK:STDOUT:   %.loc24_19.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc24_19.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_19.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_19.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc24_19.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_19.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_19.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_19.1)
// CHECK:STDOUT:   %c.ref: %OneTrivialField.elem.310 = name_ref c, @OneTrivialField.%field_decl.loc16 [concrete = @OneTrivialField.%field_decl.loc16]
// CHECK:STDOUT:   %.loc24_19.4: ref %HasTrivialDestruction = class_element_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc24_19.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc24_19.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc24_19.3 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc24_19.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc24_19.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc24_19.4 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_19.4)
// CHECK:STDOUT:   %b.ref: %OneTrivialField.elem.88e = name_ref b, @OneTrivialField.%field_decl.loc15 [concrete = @OneTrivialField.%field_decl.loc15]
// CHECK:STDOUT:   %.loc24_19.7: ref %AlsoHasNonTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc24_19.5: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc24_19.8: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc24_19.5 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %Destroy.facet.loc24_19.6: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.b58, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   %.loc24_19.9: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc24_19.6 [concrete = constants.%Destroy.facet.88e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc24_19.7)
// CHECK:STDOUT:   %a.ref: %OneTrivialField.elem.197 = name_ref a, @OneTrivialField.%field_decl.loc14 [concrete = @OneTrivialField.%field_decl.loc14]
// CHECK:STDOUT:   %.loc24_19.10: ref %HasNonTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc24_19.7: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_19.11: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_19.7 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc24_19.8: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_19.12: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_19.8 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc24_19.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_nested_nontrivial_member.carbon = $nontrivial_nested_nontrivial_member
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialField: type = class_type @HasNonTrivialField [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasNonTrivialField.elem: type = unbound_element_type %HasNonTrivialField, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.11f: type = struct_type {.x: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %NestedNonTrivialMember: type = class_type @NestedNonTrivialMember [concrete]
// CHECK:STDOUT:   %NestedNonTrivialMember.elem: type = unbound_element_type %NestedNonTrivialMember, %HasNonTrivialField [concrete]
// CHECK:STDOUT:   %struct_type.x.3f9: type = struct_type {.x: %HasNonTrivialField} [concrete]
// CHECK:STDOUT:   %pattern_type.03f: type = pattern_type %NestedNonTrivialMember [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_26.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_26.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.959: %SubobjectDestroy.type = facet_value %HasNonTrivialField, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d61: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.959) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ab1: %Destroy.type = facet_value %HasNonTrivialField, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.d61) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_26.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.446: %SubobjectDestroy.type = facet_value %NestedNonTrivialMember, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c6f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.446) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7fa: %Destroy.type = facet_value %NestedNonTrivialMember, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.c6f) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.03f = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.03f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NestedNonTrivialMember.ref.loc17: type = name_ref NestedNonTrivialMember, file.%NestedNonTrivialMember.decl [concrete = constants.%NestedNonTrivialMember]
// CHECK:STDOUT:   %Core.ref.loc17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %NestedNonTrivialMember.ref.loc17, (constants.%custom_witness.b80ef6.6) [concrete = constants.%SubobjectDestroy.facet.446]
// CHECK:STDOUT:   %.loc17: %SubobjectDestroy.type = converted %NestedNonTrivialMember.ref.loc17, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.446]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %NestedNonTrivialMember = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc22_3.1: ref %NestedNonTrivialMember = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NestedNonTrivialMember to %.loc22_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %NestedNonTrivialMember.ref.loc22: type = name_ref NestedNonTrivialMember, file.%NestedNonTrivialMember.decl [concrete = constants.%NestedNonTrivialMember]
// CHECK:STDOUT:   %x: ref %NestedNonTrivialMember = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.03f = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.03f = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc22_3.1: %Destroy.type = facet_value constants.%NestedNonTrivialMember, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.c6f) [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %.loc22_3.2: %Destroy.type = converted constants.%NestedNonTrivialMember, %Destroy.facet.loc22_3.1 [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %Destroy.facet.loc22_3.2: %Destroy.type = facet_value constants.%NestedNonTrivialMember, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.c6f) [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %.loc22_3.3: %Destroy.type = converted constants.%NestedNonTrivialMember, %Destroy.facet.loc22_3.2 [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_26.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_26.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_26.3(%self.param: ref %struct_type.x.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_26.1: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_26.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc17_26.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc17_26.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc17_26.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc17_26.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc17_26.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_26.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_26.4(%self.param: ref %HasNonTrivialField) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %HasNonTrivialField.elem = name_ref x, @HasNonTrivialField.%field_decl [concrete = @HasNonTrivialField.%field_decl]
// CHECK:STDOUT:   %.loc17_26.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_26.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc17_26.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc17_26.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc17_26.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc17_26.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc17_26.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_26.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_26.5(%self.param: ref %struct_type.x.3f9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_26.1: ref %HasNonTrivialField = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_26.1: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d61) [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %.loc17_26.2: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc17_26.1 [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %Destroy.facet.loc17_26.2: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d61) [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %.loc17_26.3: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc17_26.2 [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_26.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_26.6(%self.param: ref %NestedNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NestedNonTrivialMember.elem = name_ref x, @NestedNonTrivialMember.%field_decl [concrete = @NestedNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc17_26.1: ref %HasNonTrivialField = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_26.1: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d61) [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %.loc17_26.2: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc17_26.1 [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %Destroy.facet.loc17_26.2: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d61) [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %.loc17_26.3: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc17_26.2 [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_26.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_derived_from_nontrivial_base.carbon = $nontrivial_derived_from_nontrivial_base
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialBase: type = class_type @NonTrivialBase [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.elem: type = unbound_element_type %NonTrivialBase, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.25b: <witness> = impl_witness @NonTrivialBase.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.x.11f: type = struct_type {.x: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase: type = class_type @DerivedFromNonTrivialBase [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.elem.932: type = unbound_element_type %DerivedFromNonTrivialBase, %NonTrivialBase [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.elem.954: type = unbound_element_type %DerivedFromNonTrivialBase, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.base.y.329: type = struct_type {.base: %NonTrivialBase, .y: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %pattern_type.dbe: type = pattern_type %DerivedFromNonTrivialBase [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22_29.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a2f: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.490, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22_29.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.da1: %Destroy.type = facet_value %NonTrivialBase, (%DestroyImpl.impl_witness.25b, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22_29.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c4c: %SubobjectDestroy.type = facet_value %DerivedFromNonTrivialBase, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7a2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c4c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.14e: %Destroy.type = facet_value %DerivedFromNonTrivialBase, (%DestroyImpl.impl_witness.7a2, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.dbe = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.dbe = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.ref.loc22: type = name_ref DerivedFromNonTrivialBase, file.%DerivedFromNonTrivialBase.decl [concrete = constants.%DerivedFromNonTrivialBase]
// CHECK:STDOUT:   %Core.ref.loc22: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %DerivedFromNonTrivialBase.ref.loc22, (constants.%custom_witness.b80ef6.6) [concrete = constants.%SubobjectDestroy.facet.c4c]
// CHECK:STDOUT:   %.loc22: %SubobjectDestroy.type = converted %DerivedFromNonTrivialBase.ref.loc22, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.c4c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %DerivedFromNonTrivialBase = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc27_3.1: ref %DerivedFromNonTrivialBase = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %DerivedFromNonTrivialBase to %.loc27_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.ref.loc27: type = name_ref DerivedFromNonTrivialBase, file.%DerivedFromNonTrivialBase.decl [concrete = constants.%DerivedFromNonTrivialBase]
// CHECK:STDOUT:   %x: ref %DerivedFromNonTrivialBase = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.dbe = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.dbe = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc27_3.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc27_3.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc27_3.1 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %Destroy.facet.loc27_3.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc27_3.3: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc27_3.2 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_29.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_29.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_29.3(%self.param: ref %struct_type.x.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_29.1: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc22_29.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_29.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_29.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc22_29.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_29.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_29.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc22_29.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_29.4(%self.param: ref %NonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NonTrivialBase.elem = name_ref x, @NonTrivialBase.%field_decl [concrete = @NonTrivialBase.%field_decl]
// CHECK:STDOUT:   %.loc22_29.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc22_29.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_29.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_29.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc22_29.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_29.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_29.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc22_29.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_29.5(%self.param: ref %struct_type.base.y.329) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_29.1: ref %HasNonTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc22_29.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_29.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_29.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc22_29.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_29.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_29.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_29.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc22_29.1)
// CHECK:STDOUT:   %.loc22_29.4: ref %NonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc22_29.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc22_29.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc22_29.3 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %Destroy.facet.loc22_29.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc22_29.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc22_29.4 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_29.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc22_29.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_29.6(%self.param: ref %DerivedFromNonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.ref: %DerivedFromNonTrivialBase.elem.954 = name_ref y, @DerivedFromNonTrivialBase.%field_decl [concrete = @DerivedFromNonTrivialBase.%field_decl]
// CHECK:STDOUT:   %.loc22_29.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc22_29.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_29.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_29.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc22_29.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc22_29.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc22_29.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_29.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc22_29.1)
// CHECK:STDOUT:   %base.ref: %DerivedFromNonTrivialBase.elem.932 = name_ref base, @DerivedFromNonTrivialBase.%.loc14 [concrete = @DerivedFromNonTrivialBase.%.loc14]
// CHECK:STDOUT:   %.loc22_29.4: ref %NonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc22_29.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc22_29.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc22_29.3 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %Destroy.facet.loc22_29.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc22_29.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc22_29.4 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc22_29.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc22_29.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_grandparent.carbon = $nontrivial_grandparent
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialBase: type = class_type @NonTrivialBase [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.elem: type = unbound_element_type %NonTrivialBase, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.25b: <witness> = impl_witness @NonTrivialBase.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.x.11f: type = struct_type {.x: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase: type = class_type @DerivedFromNonTrivialBase [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.elem.932: type = unbound_element_type %DerivedFromNonTrivialBase, %NonTrivialBase [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.elem.954: type = unbound_element_type %DerivedFromNonTrivialBase, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.base.y.329: type = struct_type {.base: %NonTrivialBase, .y: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %NonTrivialGrandparent: type = class_type @NonTrivialGrandparent [concrete]
// CHECK:STDOUT:   %NonTrivialGrandparent.elem: type = unbound_element_type %NonTrivialGrandparent, %DerivedFromNonTrivialBase [concrete]
// CHECK:STDOUT:   %struct_type.base.3eb: type = struct_type {.base: %DerivedFromNonTrivialBase} [concrete]
// CHECK:STDOUT:   %pattern_type.42e: type = pattern_type %NonTrivialGrandparent [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_25.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a2f: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.490, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_25.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.da1: %Destroy.type = facet_value %NonTrivialBase, (%DestroyImpl.impl_witness.25b, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_25.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c4c: %SubobjectDestroy.type = facet_value %DerivedFromNonTrivialBase, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7a2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c4c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.14e: %Destroy.type = facet_value %DerivedFromNonTrivialBase, (%DestroyImpl.impl_witness.7a2, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_25.8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f13: %SubobjectDestroy.type = facet_value %NonTrivialGrandparent, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0d6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f13) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f1d: %Destroy.type = facet_value %NonTrivialGrandparent, (%DestroyImpl.impl_witness.0d6, %custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.42e = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.42e = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NonTrivialGrandparent.ref.loc26: type = name_ref NonTrivialGrandparent, file.%NonTrivialGrandparent.decl [concrete = constants.%NonTrivialGrandparent]
// CHECK:STDOUT:   %Core.ref.loc26: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %NonTrivialGrandparent.ref.loc26, (constants.%custom_witness.b80ef6.8) [concrete = constants.%SubobjectDestroy.facet.f13]
// CHECK:STDOUT:   %.loc26: %SubobjectDestroy.type = converted %NonTrivialGrandparent.ref.loc26, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.f13]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %NonTrivialGrandparent = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc31_3.1: ref %NonTrivialGrandparent = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NonTrivialGrandparent to %.loc31_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %NonTrivialGrandparent.ref.loc31: type = name_ref NonTrivialGrandparent, file.%NonTrivialGrandparent.decl [concrete = constants.%NonTrivialGrandparent]
// CHECK:STDOUT:   %x: ref %NonTrivialGrandparent = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.42e = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.42e = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc31_3.1: %Destroy.type = facet_value constants.%NonTrivialGrandparent, (constants.%DestroyImpl.impl_witness.0d6, constants.%custom_witness.b80ef6.8) [concrete = constants.%Destroy.facet.f1d]
// CHECK:STDOUT:   %.loc31_3.2: %Destroy.type = converted constants.%NonTrivialGrandparent, %Destroy.facet.loc31_3.1 [concrete = constants.%Destroy.facet.f1d]
// CHECK:STDOUT:   %Destroy.facet.loc31_3.2: %Destroy.type = facet_value constants.%NonTrivialGrandparent, (constants.%DestroyImpl.impl_witness.0d6, constants.%custom_witness.b80ef6.8) [concrete = constants.%Destroy.facet.f1d]
// CHECK:STDOUT:   %.loc31_3.3: %Destroy.type = converted constants.%NonTrivialGrandparent, %Destroy.facet.loc31_3.2 [concrete = constants.%Destroy.facet.f1d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_25.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_25.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_25.3(%self.param: ref %struct_type.x.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc26_25.1: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc26_25.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc26_25.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc26_25.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc26_25.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc26_25.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc26_25.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc26_25.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_25.4(%self.param: ref %NonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NonTrivialBase.elem = name_ref x, @NonTrivialBase.%field_decl [concrete = @NonTrivialBase.%field_decl]
// CHECK:STDOUT:   %.loc26_25.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc26_25.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc26_25.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc26_25.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc26_25.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc26_25.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc26_25.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc26_25.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_25.5(%self.param: ref %struct_type.base.y.329) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc26_25.1: ref %HasNonTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc26_25.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc26_25.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc26_25.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc26_25.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc26_25.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc26_25.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_25.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc26_25.1)
// CHECK:STDOUT:   %.loc26_25.4: ref %NonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc26_25.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc26_25.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc26_25.3 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %Destroy.facet.loc26_25.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc26_25.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc26_25.4 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_25.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc26_25.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_25.6(%self.param: ref %DerivedFromNonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.ref: %DerivedFromNonTrivialBase.elem.954 = name_ref y, @DerivedFromNonTrivialBase.%field_decl [concrete = @DerivedFromNonTrivialBase.%field_decl]
// CHECK:STDOUT:   %.loc26_25.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc26_25.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc26_25.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc26_25.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc26_25.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc26_25.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc26_25.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_25.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc26_25.1)
// CHECK:STDOUT:   %base.ref: %DerivedFromNonTrivialBase.elem.932 = name_ref base, @DerivedFromNonTrivialBase.%.loc14 [concrete = @DerivedFromNonTrivialBase.%.loc14]
// CHECK:STDOUT:   %.loc26_25.4: ref %NonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc26_25.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc26_25.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc26_25.3 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %Destroy.facet.loc26_25.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc26_25.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc26_25.4 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_25.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc26_25.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_25.7(%self.param: ref %struct_type.base.3eb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc26_25.1: ref %DerivedFromNonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc26_25.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc26_25.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc26_25.1 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %Destroy.facet.loc26_25.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc26_25.3: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc26_25.2 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc26_25.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_25.8(%self.param: ref %NonTrivialGrandparent) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %NonTrivialGrandparent.elem = name_ref base, @NonTrivialGrandparent.%.loc19 [concrete = @NonTrivialGrandparent.%.loc19]
// CHECK:STDOUT:   %.loc26_25.1: ref %DerivedFromNonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc26_25.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc26_25.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc26_25.1 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %Destroy.facet.loc26_25.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc26_25.3: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc26_25.2 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc26_25.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_derived_has_trivial_base_and_impls_destroy.carbon = $nontrivial_derived_has_trivial_base_and_impls_destroy
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialBase: type = class_type @TrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %TrivialBase [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.17e: <witness> = impl_witness @Derived.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %struct_type.base.223: type = struct_type {.base: %TrivialBase} [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_11.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.23e: %SubobjectDestroy.type = facet_value %TrivialBase, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7c0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.23e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0fd: %Destroy.type = facet_value %TrivialBase, (%DestroyImpl.impl_witness.7c0, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_11.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %Derived, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b2e: %Destroy.type = facet_value %Derived, (%DestroyImpl.impl_witness.17e, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Derived.ref.loc17: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Derived.ref.loc17, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   %.loc17: %SubobjectDestroy.type = converted %Derived.ref.loc17, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %Derived = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc22_3.1: ref %Derived = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc22_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Derived.ref.loc22: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %x: ref %Derived = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc22_3.1: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.17e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   %.loc22_3.2: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc22_3.1 [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   %Destroy.facet.loc22_3.2: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.17e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   %.loc22_3.3: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc22_3.2 [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_11.2(%self.param: ref %TrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_11.3(%self.param: ref %struct_type.base.223) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_11.1: ref %TrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_11.1: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc17_11.2: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc17_11.1 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %Destroy.facet.loc17_11.2: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc17_11.3: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc17_11.2 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_11.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_11.4(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %Derived.elem = name_ref base, @Derived.%.loc6 [concrete = @Derived.%.loc6]
// CHECK:STDOUT:   %.loc17_11.1: ref %TrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_11.1: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc17_11.2: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc17_11.1 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %Destroy.facet.loc17_11.2: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc17_11.3: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc17_11.2 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_11.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_generic_class.carbon = $nontrivial_generic_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %GenericClass.type: type = parameterized_class_name_type @GenericClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %GenericClass.generic: %GenericClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_51.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.a2f: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.490, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %GenericClass.38b: type = class_type @GenericClass, @GenericClass(%Destroy.facet.a2f) [concrete]
// CHECK:STDOUT:   %pattern_type.e4f: type = pattern_type %GenericClass.38b [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericClass.elem.1c8: type = unbound_element_type %GenericClass.38b, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.11f: type = struct_type {.x: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_42.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d8b: %SubobjectDestroy.type = facet_value %GenericClass.38b, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c8a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d8b) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6a8: %Destroy.type = facet_value %GenericClass.38b, (%DestroyImpl.impl_witness.c8a, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.e4f = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.e4f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.HasNonTrivialDestruction: type = import_ref Main//types, HasNonTrivialDestruction, loaded [concrete = constants.%HasNonTrivialDestruction]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %GenericClass.ref.loc13: %GenericClass.type = name_ref GenericClass, file.%GenericClass.decl [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:   %HasNonTrivialDestruction.ref.loc13: type = name_ref HasNonTrivialDestruction, imports.%Main.HasNonTrivialDestruction [concrete = constants.%HasNonTrivialDestruction]
// CHECK:STDOUT:   %Destroy.facet.loc13: %Destroy.type = facet_value %HasNonTrivialDestruction.ref.loc13, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc13_40: %Destroy.type = converted %HasNonTrivialDestruction.ref.loc13, %Destroy.facet.loc13 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %GenericClass.loc13: type = class_type @GenericClass, @GenericClass(constants.%Destroy.facet.a2f) [concrete = constants.%GenericClass.38b]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %GenericClass.loc13, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.d8b]
// CHECK:STDOUT:   %.loc13_42: %SubobjectDestroy.type = converted %GenericClass.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.d8b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %GenericClass.38b = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc18_3.1: ref %GenericClass.38b = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %GenericClass.38b to %.loc18_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc18_54.1: type = splice_block %GenericClass.loc18 [concrete = constants.%GenericClass.38b] {
// CHECK:STDOUT:     %GenericClass.ref.loc18: %GenericClass.type = name_ref GenericClass, file.%GenericClass.decl [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:     %HasNonTrivialDestruction.ref.loc18: type = name_ref HasNonTrivialDestruction, imports.%Main.HasNonTrivialDestruction [concrete = constants.%HasNonTrivialDestruction]
// CHECK:STDOUT:     %Destroy.facet.loc18_54: %Destroy.type = facet_value %HasNonTrivialDestruction.ref.loc18, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:     %.loc18_54.2: %Destroy.type = converted %HasNonTrivialDestruction.ref.loc18, %Destroy.facet.loc18_54 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:     %GenericClass.loc18: type = class_type @GenericClass, @GenericClass(constants.%Destroy.facet.a2f) [concrete = constants.%GenericClass.38b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %GenericClass.38b = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.e4f = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.e4f = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%GenericClass.38b, (constants.%DestroyImpl.impl_witness.c8a, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.6a8]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%GenericClass.38b, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.6a8]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%GenericClass.38b, (constants.%DestroyImpl.impl_witness.c8a, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.6a8]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%GenericClass.38b, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.6a8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_42.1(%self.param: ref %struct_type.x.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_42.1: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_42.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc13_42.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc13_42.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc13_42.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc13_42.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc13_42.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_42.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_42.2(%self.param: ref %GenericClass.38b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %GenericClass.elem.1c8 = name_ref x, @GenericClass.%field_decl [concrete = @GenericClass.%field_decl]
// CHECK:STDOUT:   %.loc13_42.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_42.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc13_42.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc13_42.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc13_42.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc13_42.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc13_42.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_42.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_base_and_trivial_field.carbon = $nontrivial_base_and_trivial_field
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialBase: type = class_type @NonTrivialBase [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.elem: type = unbound_element_type %NonTrivialBase, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.25b: <witness> = impl_witness @NonTrivialBase.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.x.11f: type = struct_type {.x: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember: type = class_type @NonTrivialBaseAndTrivialMember [concrete]
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember.elem.e27: type = unbound_element_type %NonTrivialBaseAndTrivialMember, %NonTrivialBase [concrete]
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember.elem.354: type = unbound_element_type %NonTrivialBaseAndTrivialMember, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.base.y.6e0: type = struct_type {.base: %NonTrivialBase, .y: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %pattern_type.c4f: type = pattern_type %NonTrivialBaseAndTrivialMember [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_34.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a2f: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.490, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_34.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_34.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dba: %Destroy.type = facet_value %HasTrivialDestruction, (%DestroyImpl.impl_witness.4ca, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.da1: %Destroy.type = facet_value %NonTrivialBase, (%DestroyImpl.impl_witness.25b, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_34.7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.78d: %SubobjectDestroy.type = facet_value %NonTrivialBaseAndTrivialMember, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.75e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.78d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7eb: %Destroy.type = facet_value %NonTrivialBaseAndTrivialMember, (%DestroyImpl.impl_witness.75e, %custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.c4f = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.c4f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember.ref.loc24: type = name_ref NonTrivialBaseAndTrivialMember, file.%NonTrivialBaseAndTrivialMember.decl [concrete = constants.%NonTrivialBaseAndTrivialMember]
// CHECK:STDOUT:   %Core.ref.loc24: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %NonTrivialBaseAndTrivialMember.ref.loc24, (constants.%custom_witness.b80ef6.7) [concrete = constants.%SubobjectDestroy.facet.78d]
// CHECK:STDOUT:   %.loc24: %SubobjectDestroy.type = converted %NonTrivialBaseAndTrivialMember.ref.loc24, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.78d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %NonTrivialBaseAndTrivialMember = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc29_3.1: ref %NonTrivialBaseAndTrivialMember = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NonTrivialBaseAndTrivialMember to %.loc29_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember.ref.loc29: type = name_ref NonTrivialBaseAndTrivialMember, file.%NonTrivialBaseAndTrivialMember.decl [concrete = constants.%NonTrivialBaseAndTrivialMember]
// CHECK:STDOUT:   %x: ref %NonTrivialBaseAndTrivialMember = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.c4f = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c4f = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc29_3.1: %Destroy.type = facet_value constants.%NonTrivialBaseAndTrivialMember, (constants.%DestroyImpl.impl_witness.75e, constants.%custom_witness.b80ef6.7) [concrete = constants.%Destroy.facet.7eb]
// CHECK:STDOUT:   %.loc29_3.2: %Destroy.type = converted constants.%NonTrivialBaseAndTrivialMember, %Destroy.facet.loc29_3.1 [concrete = constants.%Destroy.facet.7eb]
// CHECK:STDOUT:   %Destroy.facet.loc29_3.2: %Destroy.type = facet_value constants.%NonTrivialBaseAndTrivialMember, (constants.%DestroyImpl.impl_witness.75e, constants.%custom_witness.b80ef6.7) [concrete = constants.%Destroy.facet.7eb]
// CHECK:STDOUT:   %.loc29_3.3: %Destroy.type = converted constants.%NonTrivialBaseAndTrivialMember, %Destroy.facet.loc29_3.2 [concrete = constants.%Destroy.facet.7eb]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_34.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_34.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_34.3(%self.param: ref %struct_type.x.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_34.1: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc24_34.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_34.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_34.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc24_34.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_34.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_34.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc24_34.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_34.4(%self.param: ref %NonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NonTrivialBase.elem = name_ref x, @NonTrivialBase.%field_decl [concrete = @NonTrivialBase.%field_decl]
// CHECK:STDOUT:   %.loc24_34.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc24_34.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_34.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_34.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc24_34.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc24_34.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc24_34.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc24_34.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_34.5(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_34.6(%self.param: ref %struct_type.base.y.6e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_34.1: ref %HasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc24_34.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc24_34.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc24_34.1 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc24_34.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc24_34.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc24_34.2 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_34.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_34.1)
// CHECK:STDOUT:   %.loc24_34.4: ref %NonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc24_34.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc24_34.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc24_34.3 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %Destroy.facet.loc24_34.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc24_34.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc24_34.4 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_34.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_34.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_34.7(%self.param: ref %NonTrivialBaseAndTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.ref: %NonTrivialBaseAndTrivialMember.elem.354 = name_ref y, @NonTrivialBaseAndTrivialMember.%field_decl [concrete = @NonTrivialBaseAndTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc24_34.1: ref %HasTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc24_34.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc24_34.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc24_34.1 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc24_34.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc24_34.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc24_34.2 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_34.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_34.1)
// CHECK:STDOUT:   %base.ref: %NonTrivialBaseAndTrivialMember.elem.e27 = name_ref base, @NonTrivialBaseAndTrivialMember.%.loc16 [concrete = @NonTrivialBaseAndTrivialMember.%.loc16]
// CHECK:STDOUT:   %.loc24_34.4: ref %NonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc24_34.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc24_34.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc24_34.3 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %Destroy.facet.loc24_34.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.25b, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   %.loc24_34.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc24_34.4 [concrete = constants.%Destroy.facet.da1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_34.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_34.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_trivial_base_and_nontrivial_field.carbon = $nontrivial_trivial_base_and_nontrivial_field
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialBase: type = class_type @TrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %TrivialBaseAndNonTrivialMember: type = class_type @TrivialBaseAndNonTrivialMember [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TrivialBaseAndNonTrivialMember.elem.8b0: type = unbound_element_type %TrivialBaseAndNonTrivialMember, %TrivialBase [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %TrivialBaseAndNonTrivialMember.elem.303: type = unbound_element_type %TrivialBaseAndNonTrivialMember, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.base.x.9c9: type = struct_type {.base: %TrivialBase, .x: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %pattern_type.575: type = pattern_type %TrivialBaseAndNonTrivialMember [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_34.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_34.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.23e: %SubobjectDestroy.type = facet_value %TrivialBase, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7c0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.23e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.033: %Destroy.type = facet_value %TrivialBase, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.7c0) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_34.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a42: %SubobjectDestroy.type = facet_value %TrivialBaseAndNonTrivialMember, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4c7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a42) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6ca: %Destroy.type = facet_value %TrivialBaseAndNonTrivialMember, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.4c7) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.575 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.575 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TrivialBaseAndNonTrivialMember.ref.loc16: type = name_ref TrivialBaseAndNonTrivialMember, file.%TrivialBaseAndNonTrivialMember.decl [concrete = constants.%TrivialBaseAndNonTrivialMember]
// CHECK:STDOUT:   %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %TrivialBaseAndNonTrivialMember.ref.loc16, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.a42]
// CHECK:STDOUT:   %.loc16: %SubobjectDestroy.type = converted %TrivialBaseAndNonTrivialMember.ref.loc16, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.a42]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %TrivialBaseAndNonTrivialMember = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc21_3.1: ref %TrivialBaseAndNonTrivialMember = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %TrivialBaseAndNonTrivialMember to %.loc21_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %TrivialBaseAndNonTrivialMember.ref.loc21: type = name_ref TrivialBaseAndNonTrivialMember, file.%TrivialBaseAndNonTrivialMember.decl [concrete = constants.%TrivialBaseAndNonTrivialMember]
// CHECK:STDOUT:   %x: ref %TrivialBaseAndNonTrivialMember = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.575 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.575 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc21_3.1: %Destroy.type = facet_value constants.%TrivialBaseAndNonTrivialMember, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.4c7) [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %.loc21_3.2: %Destroy.type = converted constants.%TrivialBaseAndNonTrivialMember, %Destroy.facet.loc21_3.1 [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %Destroy.facet.loc21_3.2: %Destroy.type = facet_value constants.%TrivialBaseAndNonTrivialMember, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.4c7) [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %.loc21_3.3: %Destroy.type = converted constants.%TrivialBaseAndNonTrivialMember, %Destroy.facet.loc21_3.2 [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_34.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_34.2(%self.param: ref %TrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_34.3(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_34.4(%self.param: ref %struct_type.base.x.9c9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_34.1: ref %HasNonTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc16_34.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc16_34.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc16_34.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc16_34.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc16_34.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc16_34.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_34.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc16_34.1)
// CHECK:STDOUT:   %.loc16_34.4: ref %TrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc16_34.3: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc16_34.5: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc16_34.3 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %Destroy.facet.loc16_34.4: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc16_34.6: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc16_34.4 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_34.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc16_34.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_34.5(%self.param: ref %TrivialBaseAndNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %TrivialBaseAndNonTrivialMember.elem.303 = name_ref x, @TrivialBaseAndNonTrivialMember.%field_decl [concrete = @TrivialBaseAndNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc16_34.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc16_34.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc16_34.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc16_34.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc16_34.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc16_34.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc16_34.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_34.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc16_34.1)
// CHECK:STDOUT:   %base.ref: %TrivialBaseAndNonTrivialMember.elem.8b0 = name_ref base, @TrivialBaseAndNonTrivialMember.%.loc8 [concrete = @TrivialBaseAndNonTrivialMember.%.loc8]
// CHECK:STDOUT:   %.loc16_34.4: ref %TrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc16_34.3: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc16_34.5: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc16_34.3 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %Destroy.facet.loc16_34.4: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc16_34.6: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc16_34.4 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_34.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc16_34.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_class_dynamic.carbon = $nontrivial_class_dynamic
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass: type = class_type @NonTrivialDynamicClass [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %NonTrivialDynamicClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a13: <witness> = impl_witness @NonTrivialDynamicClass.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_26.1 [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.194: %Destroy.type = facet_value %ptr.454, (%DestroyImpl.impl_witness.cad, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_26.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.27c: %SubobjectDestroy.type = facet_value %NonTrivialDynamicClass, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %Destroy.facet.153: %Destroy.type = facet_value %NonTrivialDynamicClass, (%DestroyImpl.impl_witness.a13, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.157 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.157 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @NonTrivialDynamicClass.vtable {
// CHECK:STDOUT:   @NonTrivialDynamicClass.%NonTrivialDynamicClass.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NonTrivialDynamicClass.ref.loc15: type = name_ref NonTrivialDynamicClass, file.%NonTrivialDynamicClass.decl [concrete = constants.%NonTrivialDynamicClass]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %NonTrivialDynamicClass.ref.loc15, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.27c]
// CHECK:STDOUT:   %.loc15: %SubobjectDestroy.type = converted %NonTrivialDynamicClass.ref.loc15, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.27c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %NonTrivialDynamicClass = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc20_3.1: ref %NonTrivialDynamicClass = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NonTrivialDynamicClass to %.loc20_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %NonTrivialDynamicClass.ref.loc20: type = name_ref NonTrivialDynamicClass, file.%NonTrivialDynamicClass.decl [concrete = constants.%NonTrivialDynamicClass]
// CHECK:STDOUT:   %x: ref %NonTrivialDynamicClass = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.157 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.157 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc20_3.1: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.a13, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   %.loc20_3.2: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc20_3.1 [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   %Destroy.facet.loc20_3.2: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.a13, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   %.loc20_3.3: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc20_3.2 [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_26.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_26.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_26.1: ref %ptr.454 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_26.1: %Destroy.type = facet_value constants.%ptr.454, (constants.%DestroyImpl.impl_witness.cad, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %.loc15_26.2: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc15_26.1 [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %Destroy.facet.loc15_26.2: %Destroy.type = facet_value constants.%ptr.454, (constants.%DestroyImpl.impl_witness.cad, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %.loc15_26.3: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc15_26.2 [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc15_26.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_26.3(%self.param: ref %NonTrivialDynamicClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_dynamic_base.carbon = $nontrivial_dynamic_base
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass: type = class_type @NonTrivialDynamicClass [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.F.type: type = fn_type @NonTrivialDynamicClass.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.F: %NonTrivialDynamicClass.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a13: <witness> = impl_witness @NonTrivialDynamicClass.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %NonTrivialDynamicClass [concrete]
// CHECK:STDOUT:   %struct_type.base.7c1: type = struct_type {.base: %NonTrivialDynamicClass} [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_11.1 [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.194: %Destroy.type = facet_value %ptr.454, (%DestroyImpl.impl_witness.cad, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_11.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.153: %Destroy.type = facet_value %NonTrivialDynamicClass, (%DestroyImpl.impl_witness.a13, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_11.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %Derived, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.660: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.35d: %Destroy.type = facet_value %Derived, (%DestroyImpl.impl_witness.660, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @NonTrivialDynamicClass.vtable {
// CHECK:STDOUT:   @NonTrivialDynamicClass.%NonTrivialDynamicClass.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   constants.%NonTrivialDynamicClass.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Derived.ref.loc19: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Derived.ref.loc19, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   %.loc19: %SubobjectDestroy.type = converted %Derived.ref.loc19, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %Derived = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc24_3.1: ref %Derived = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc24_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Derived.ref.loc24: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %x: ref %Derived = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc24_3.1: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.660, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %.loc24_3.2: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc24_3.1 [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %Destroy.facet.loc24_3.2: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.660, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %.loc24_3.3: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc24_3.2 [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_11.1: ref %ptr.454 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_11.1: %Destroy.type = facet_value constants.%ptr.454, (constants.%DestroyImpl.impl_witness.cad, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %.loc19_11.2: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc19_11.1 [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %Destroy.facet.loc19_11.2: %Destroy.type = facet_value constants.%ptr.454, (constants.%DestroyImpl.impl_witness.cad, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %.loc19_11.3: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc19_11.2 [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc19_11.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.3(%self.param: ref %NonTrivialDynamicClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.4(%self.param: ref %struct_type.base.7c1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_11.1: ref %NonTrivialDynamicClass = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_11.1: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.a13, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   %.loc19_11.2: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc19_11.1 [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   %Destroy.facet.loc19_11.2: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.a13, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   %.loc19_11.3: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc19_11.2 [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc19_11.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.5(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %Derived.elem = name_ref base, @Derived.%.loc12 [concrete = @Derived.%.loc12]
// CHECK:STDOUT:   %.loc19_11.1: ref %NonTrivialDynamicClass = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_11.1: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.a13, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   %.loc19_11.2: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc19_11.1 [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   %Destroy.facet.loc19_11.2: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.a13, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   %.loc19_11.3: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc19_11.2 [concrete = constants.%Destroy.facet.153]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc19_11.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon = $fail_todo_nontrivial_class_derived_from_nontrivial_abstract
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Derived.ref.loc27: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc27: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref <error> = var_storage %x.var_patt [concrete = <error>]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc46: ref %Derived = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc46 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   %Class.ref: <error> = name_ref Class, <error> [concrete = <error>]
// CHECK:STDOUT:   %x: <error> = wrapper_binding x, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: <error> = ref_binding_pattern x [concrete = <error>]
// CHECK:STDOUT:     %x.var_patt: <error> = var_pattern %x.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
