// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// 
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/classes.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/classes.carbon

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

// `ImplsDestroy.(SubobjectDestroy.Op)` is a no-op because it
// doesn't have any fields that require it to be non-trivial.
// Similarly for other types like this.
class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class AlsoImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
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

// --- todo_nontrivial_empty_class_impls_destroy.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Class = ImplsDestroy;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end

}

// --- todo_nontrivial_all_nontrivial_fields.carbon
library "[[@TEST_NAME]]";

import library "types";

class AllNonTrivialFields {
  var a: ImplsDestroy;
  var b: AlsoImplsDestroy;
  var c: AlsoImplsDestroy;
  var d: ImplsDestroy;
}

alias Class = AllNonTrivialFields;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_one_trivial_field.carbon
library "[[@TEST_NAME]]";

import library "types";

class OneTrivialField {
  var a: ImplsDestroy;
  var b: AlsoImplsDestroy;
  var c: HasTrivialDestruction;
  var d: ImplsDestroy;
}

alias Class = OneTrivialField;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
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

alias Class = NestedNonTrivialMember;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_derived_from_nontrivial_base.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Class = DerivedFromNonTrivialBase;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_grandparent.carbon
library "[[@TEST_NAME]]";

import library "types";

class NonTrivialGrandparent {
  extend base: DerivedFromNonTrivialBase;
}

alias Class = NonTrivialGrandparent;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
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

alias Class = Derived;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_generic_class.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Class = GenericClass(ImplsDestroy);
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_base_and_trivial_field.carbon
library "[[@TEST_NAME]]";

import library "types";

class NonTrivialBaseAndTrivialMember {
  extend base: NonTrivialBase;
  var y: HasTrivialDestruction;
}

alias Class = NonTrivialBaseAndTrivialMember;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_trivial_base_and_nontrivial_field.carbon
library "[[@TEST_NAME]]";

import library "types";

class TrivialBaseAndNonTrivialMember {
  extend base: TrivialBase;
  var x: ImplsDestroy;
}

alias Class = TrivialBaseAndNonTrivialMember;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_class_dynamic.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Class = NonTrivialDynamicClass;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_dynamic_base.carbon
library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: NonTrivialDynamicClass;
}

alias Class = Derived;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
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

alias Class = Derived;
fn Make() -> Class;

// TODO: Support destroying classes that derive from abstract types.
fn Test() {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Class as Core.SubobjectDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Class` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.SubobjectDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Class as Core.SubobjectDestroy;
  // CHECK:STDERR:        ╭─┤ fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   Class as Core.Destroy;
  // CHECK:STDERR:        ·   ━┳━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·    ┗━━┫ error: cannot convert type `Class` into type implementing `Core.Destroy`
  // CHECK:STDERR:        ·       ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Class as Core.Destroy;

  // CHECK:STDERR:        ╭─┤ fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon:[[@LINE+14]]:3
  // CHECK:STDERR: ->     │   var unused x: Class = Make();
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━
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
  var unused x: Class = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_empty_class_impls_destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.6b9: type = pattern_type %ImplsDestroy [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.df0: %SubobjectDestroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b9 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.6b9 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc25_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc10: type = name_ref Class, file.%Class [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc10, (constants.%custom_witness.b80ef6.2) [concrete = constants.%SubobjectDestroy.facet.df0]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Class.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.df0]
// CHECK:STDOUT:   %Class.ref.loc11: type = name_ref Class, file.%Class [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Class.ref.loc11, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Class.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %ImplsDestroy = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %ImplsDestroy = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %ImplsDestroy to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc13: type = name_ref Class, file.%Class [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:   %x: ref %ImplsDestroy = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6b9 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.6b9 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_all_nontrivial_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AllNonTrivialFields: type = class_type @AllNonTrivialFields [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AllNonTrivialFields.elem.b78: type = unbound_element_type %AllNonTrivialFields, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %AllNonTrivialFields.elem.8d2: type = unbound_element_type %AllNonTrivialFields, %AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.f50: type = struct_type {.a: %ImplsDestroy, .b: %AlsoImplsDestroy, .c: %AlsoImplsDestroy, .d: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %pattern_type.c66: type = pattern_type %AllNonTrivialFields [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c98: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.09d [concrete]
// CHECK:STDOUT:   %Destroy.facet.4a8: %Destroy.type = facet_value %AlsoImplsDestroy, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.c98) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_9.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.075: %SubobjectDestroy.type = facet_value %AllNonTrivialFields, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.124: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.075) [concrete]
// CHECK:STDOUT:   %Destroy.facet.933: %Destroy.type = facet_value %AllNonTrivialFields, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.124) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.c66 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.c66 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @AlsoImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.as.DestroyImpl.impl.Op: %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc25_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.0c0: %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc31_28, loaded [concrete = constants.%AlsoImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.09d = impl_witness_table (%Main.import_ref.0c0), @AlsoImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc17: type = name_ref Class, file.%Class [concrete = constants.%AllNonTrivialFields]
// CHECK:STDOUT:   %Core.ref.loc17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc17, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.075]
// CHECK:STDOUT:   %.loc17: %SubobjectDestroy.type = converted %Class.ref.loc17, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.075]
// CHECK:STDOUT:   %Class.ref.loc18: type = name_ref Class, file.%Class [concrete = constants.%AllNonTrivialFields]
// CHECK:STDOUT:   %Core.ref.loc18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc18: %Destroy.type = facet_value %Class.ref.loc18, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.124) [concrete = constants.%Destroy.facet.933]
// CHECK:STDOUT:   %.loc18: %Destroy.type = converted %Class.ref.loc18, %Destroy.facet.loc18 [concrete = constants.%Destroy.facet.933]
// CHECK:STDOUT:   %x.var: ref %AllNonTrivialFields = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc20_3.1: ref %AllNonTrivialFields = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %AllNonTrivialFields to %.loc20_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc20: type = name_ref Class, file.%Class [concrete = constants.%AllNonTrivialFields]
// CHECK:STDOUT:   %x: ref %AllNonTrivialFields = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.c66 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c66 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc20_3.1: %Destroy.type = facet_value constants.%AllNonTrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.124) [concrete = constants.%Destroy.facet.933]
// CHECK:STDOUT:   %.loc20_3.2: %Destroy.type = converted constants.%AllNonTrivialFields, %Destroy.facet.loc20_3.1 [concrete = constants.%Destroy.facet.933]
// CHECK:STDOUT:   %Destroy.facet.loc20_3.2: %Destroy.type = facet_value constants.%AllNonTrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.124) [concrete = constants.%Destroy.facet.933]
// CHECK:STDOUT:   %.loc20_3.3: %Destroy.type = converted constants.%AllNonTrivialFields, %Destroy.facet.loc20_3.2 [concrete = constants.%Destroy.facet.933]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_9.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_9.3(%self.param: ref %AlsoImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_9.4(%self.param: ref %struct_type.a.b.c.d.f50) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_9.1: ref %ImplsDestroy = struct_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc17_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc17_9.1)
// CHECK:STDOUT:   %.loc17_9.4: ref %AlsoImplsDestroy = struct_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc17_9.3: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.5: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.3 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.4: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.6: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.4 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc17_9.4)
// CHECK:STDOUT:   %.loc17_9.7: ref %AlsoImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc17_9.5: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.8: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.5 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.6: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.9: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.6 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc17_9.7)
// CHECK:STDOUT:   %.loc17_9.10: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_9.7: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.11: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.7 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.8: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.12: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.8 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc17_9.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_9.5(%self.param: ref %AllNonTrivialFields) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %AllNonTrivialFields.elem.b78 = name_ref d, @AllNonTrivialFields.%field_decl.loc9 [concrete = @AllNonTrivialFields.%field_decl.loc9]
// CHECK:STDOUT:   %.loc17_9.1: ref %ImplsDestroy = class_element_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc17_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc17_9.1)
// CHECK:STDOUT:   %c.ref: %AllNonTrivialFields.elem.8d2 = name_ref c, @AllNonTrivialFields.%field_decl.loc8 [concrete = @AllNonTrivialFields.%field_decl.loc8]
// CHECK:STDOUT:   %.loc17_9.4: ref %AlsoImplsDestroy = class_element_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc17_9.3: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.5: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.3 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.4: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.6: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.4 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc17_9.4)
// CHECK:STDOUT:   %b.ref: %AllNonTrivialFields.elem.8d2 = name_ref b, @AllNonTrivialFields.%field_decl.loc7 [concrete = @AllNonTrivialFields.%field_decl.loc7]
// CHECK:STDOUT:   %.loc17_9.7: ref %AlsoImplsDestroy = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc17_9.5: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.8: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.5 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.6: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.9: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.6 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc17_9.7)
// CHECK:STDOUT:   %a.ref: %AllNonTrivialFields.elem.b78 = name_ref a, @AllNonTrivialFields.%field_decl.loc6 [concrete = @AllNonTrivialFields.%field_decl.loc6]
// CHECK:STDOUT:   %.loc17_9.10: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_9.7: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.11: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.7 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.8: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.12: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.8 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc17_9.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_one_trivial_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %OneTrivialField: type = class_type @OneTrivialField [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %OneTrivialField.elem.d63: type = unbound_element_type %OneTrivialField, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %OneTrivialField.elem.5db: type = unbound_element_type %OneTrivialField, %AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %OneTrivialField.elem.310: type = unbound_element_type %OneTrivialField, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.c6a: type = struct_type {.a: %ImplsDestroy, .b: %AlsoImplsDestroy, .c: %HasTrivialDestruction, .d: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %pattern_type.494: type = pattern_type %OneTrivialField [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c98: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.09d [concrete]
// CHECK:STDOUT:   %Destroy.facet.4a8: %Destroy.type = facet_value %AlsoImplsDestroy, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.c98) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_9.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c4c: %SubobjectDestroy.type = facet_value %OneTrivialField, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6ed: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c4c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c8b: %Destroy.type = facet_value %OneTrivialField, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.6ed) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.494 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.494 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @AlsoImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.as.DestroyImpl.impl.Op: %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc25_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.0c0: %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc31_28, loaded [concrete = constants.%AlsoImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.09d = impl_witness_table (%Main.import_ref.0c0), @AlsoImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc17: type = name_ref Class, file.%Class [concrete = constants.%OneTrivialField]
// CHECK:STDOUT:   %Core.ref.loc17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc17, (constants.%custom_witness.b80ef6.6) [concrete = constants.%SubobjectDestroy.facet.c4c]
// CHECK:STDOUT:   %.loc17: %SubobjectDestroy.type = converted %Class.ref.loc17, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.c4c]
// CHECK:STDOUT:   %Class.ref.loc18: type = name_ref Class, file.%Class [concrete = constants.%OneTrivialField]
// CHECK:STDOUT:   %Core.ref.loc18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc18: %Destroy.type = facet_value %Class.ref.loc18, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.6ed) [concrete = constants.%Destroy.facet.c8b]
// CHECK:STDOUT:   %.loc18: %Destroy.type = converted %Class.ref.loc18, %Destroy.facet.loc18 [concrete = constants.%Destroy.facet.c8b]
// CHECK:STDOUT:   %x.var: ref %OneTrivialField = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc20_3.1: ref %OneTrivialField = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %OneTrivialField to %.loc20_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc20: type = name_ref Class, file.%Class [concrete = constants.%OneTrivialField]
// CHECK:STDOUT:   %x: ref %OneTrivialField = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.494 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.494 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc20_3.1: %Destroy.type = facet_value constants.%OneTrivialField, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.6ed) [concrete = constants.%Destroy.facet.c8b]
// CHECK:STDOUT:   %.loc20_3.2: %Destroy.type = converted constants.%OneTrivialField, %Destroy.facet.loc20_3.1 [concrete = constants.%Destroy.facet.c8b]
// CHECK:STDOUT:   %Destroy.facet.loc20_3.2: %Destroy.type = facet_value constants.%OneTrivialField, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.6ed) [concrete = constants.%Destroy.facet.c8b]
// CHECK:STDOUT:   %.loc20_3.3: %Destroy.type = converted constants.%OneTrivialField, %Destroy.facet.loc20_3.2 [concrete = constants.%Destroy.facet.c8b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_9.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_9.3(%self.param: ref %AlsoImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_9.4(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_9.5(%self.param: ref %struct_type.a.b.c.d.c6a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_9.1: ref %ImplsDestroy = struct_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc17_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc17_9.1)
// CHECK:STDOUT:   %.loc17_9.4: ref %HasTrivialDestruction = struct_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc17_9.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc17_9.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc17_9.3 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc17_9.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc17_9.4 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc17_9.4)
// CHECK:STDOUT:   %.loc17_9.7: ref %AlsoImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc17_9.5: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.8: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.5 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.6: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.9: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.6 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc17_9.7)
// CHECK:STDOUT:   %.loc17_9.10: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_9.7: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.11: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.7 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.8: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.12: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.8 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc17_9.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_9.6(%self.param: ref %OneTrivialField) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %OneTrivialField.elem.d63 = name_ref d, @OneTrivialField.%field_decl.loc9 [concrete = @OneTrivialField.%field_decl.loc9]
// CHECK:STDOUT:   %.loc17_9.1: ref %ImplsDestroy = class_element_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc17_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc17_9.1)
// CHECK:STDOUT:   %c.ref: %OneTrivialField.elem.310 = name_ref c, @OneTrivialField.%field_decl.loc8 [concrete = @OneTrivialField.%field_decl.loc8]
// CHECK:STDOUT:   %.loc17_9.4: ref %HasTrivialDestruction = class_element_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc17_9.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc17_9.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc17_9.3 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc17_9.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc17_9.4 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc17_9.4)
// CHECK:STDOUT:   %b.ref: %OneTrivialField.elem.5db = name_ref b, @OneTrivialField.%field_decl.loc7 [concrete = @OneTrivialField.%field_decl.loc7]
// CHECK:STDOUT:   %.loc17_9.7: ref %AlsoImplsDestroy = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc17_9.5: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.8: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.5 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.6: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c98) [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   %.loc17_9.9: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_9.6 [concrete = constants.%Destroy.facet.4a8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc17_9.7)
// CHECK:STDOUT:   %a.ref: %OneTrivialField.elem.d63 = name_ref a, @OneTrivialField.%field_decl.loc6 [concrete = @OneTrivialField.%field_decl.loc6]
// CHECK:STDOUT:   %.loc17_9.10: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_9.7: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.11: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.7 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc17_9.8: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc17_9.12: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_9.8 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_9.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc17_9.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_nested_nontrivial_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialField: type = class_type @HasNonTrivialField [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasNonTrivialField.elem: type = unbound_element_type %HasNonTrivialField, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.x.c57: type = struct_type {.x: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %NestedNonTrivialMember: type = class_type @NestedNonTrivialMember [concrete]
// CHECK:STDOUT:   %NestedNonTrivialMember.elem: type = unbound_element_type %NestedNonTrivialMember, %HasNonTrivialField [concrete]
// CHECK:STDOUT:   %struct_type.x.3f9: type = struct_type {.x: %HasNonTrivialField} [concrete]
// CHECK:STDOUT:   %pattern_type.03f: type = pattern_type %NestedNonTrivialMember [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.959: %SubobjectDestroy.type = facet_value %HasNonTrivialField, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d61: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.959) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ab1: %Destroy.type = facet_value %HasNonTrivialField, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.d61) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_9.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.446: %SubobjectDestroy.type = facet_value %NestedNonTrivialMember, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c6f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.446) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7fa: %Destroy.type = facet_value %NestedNonTrivialMember, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.c6f) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.03f = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.03f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc25_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc18: type = name_ref Class, file.%Class [concrete = constants.%NestedNonTrivialMember]
// CHECK:STDOUT:   %Core.ref.loc18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc18, (constants.%custom_witness.b80ef6.6) [concrete = constants.%SubobjectDestroy.facet.446]
// CHECK:STDOUT:   %.loc18: %SubobjectDestroy.type = converted %Class.ref.loc18, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.446]
// CHECK:STDOUT:   %Class.ref.loc19: type = name_ref Class, file.%Class [concrete = constants.%NestedNonTrivialMember]
// CHECK:STDOUT:   %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc19: %Destroy.type = facet_value %Class.ref.loc19, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.c6f) [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %.loc19: %Destroy.type = converted %Class.ref.loc19, %Destroy.facet.loc19 [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %x.var: ref %NestedNonTrivialMember = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc21_3.1: ref %NestedNonTrivialMember = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NestedNonTrivialMember to %.loc21_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc21: type = name_ref Class, file.%Class [concrete = constants.%NestedNonTrivialMember]
// CHECK:STDOUT:   %x: ref %NestedNonTrivialMember = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.03f = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.03f = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc21_3.1: %Destroy.type = facet_value constants.%NestedNonTrivialMember, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.c6f) [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %.loc21_3.2: %Destroy.type = converted constants.%NestedNonTrivialMember, %Destroy.facet.loc21_3.1 [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %Destroy.facet.loc21_3.2: %Destroy.type = facet_value constants.%NestedNonTrivialMember, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.c6f) [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %.loc21_3.3: %Destroy.type = converted constants.%NestedNonTrivialMember, %Destroy.facet.loc21_3.2 [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_9.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_9.3(%self.param: ref %struct_type.x.c57) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_9.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc18_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc18_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc18_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc18_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc18_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_9.4(%self.param: ref %HasNonTrivialField) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %HasNonTrivialField.elem = name_ref x, @HasNonTrivialField.%field_decl [concrete = @HasNonTrivialField.%field_decl]
// CHECK:STDOUT:   %.loc18_9.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc18_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc18_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc18_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc18_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc18_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_9.5(%self.param: ref %struct_type.x.3f9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_9.1: ref %HasNonTrivialField = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_9.1: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d61) [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %.loc18_9.2: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc18_9.1 [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %Destroy.facet.loc18_9.2: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d61) [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %.loc18_9.3: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc18_9.2 [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_9.6(%self.param: ref %NestedNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NestedNonTrivialMember.elem = name_ref x, @NestedNonTrivialMember.%field_decl [concrete = @NestedNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc18_9.1: ref %HasNonTrivialField = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_9.1: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d61) [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %.loc18_9.2: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc18_9.1 [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %Destroy.facet.loc18_9.2: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d61) [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   %.loc18_9.3: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc18_9.2 [concrete = constants.%Destroy.facet.ab1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_derived_from_nontrivial_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase: type = class_type @DerivedFromNonTrivialBase [concrete]
// CHECK:STDOUT:   %NonTrivialBase: type = class_type @NonTrivialBase [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.x.c57: type = struct_type {.x: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.elem.932: type = unbound_element_type %DerivedFromNonTrivialBase, %NonTrivialBase [concrete]
// CHECK:STDOUT:   %struct_type.base.y.2df: type = struct_type {.base: %NonTrivialBase, .y: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %pattern_type.dbe: type = pattern_type %DerivedFromNonTrivialBase [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.elem: type = unbound_element_type %NonTrivialBase, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6e0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.9e6: %Destroy.type = facet_value %NonTrivialBase, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.6d9) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.elem.60a: type = unbound_element_type %DerivedFromNonTrivialBase, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c4c: %SubobjectDestroy.type = facet_value %DerivedFromNonTrivialBase, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7a2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c4c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6f5: %Destroy.type = facet_value %DerivedFromNonTrivialBase, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.7a2) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.dbe = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.dbe = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialBase.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op: %NonTrivialBase.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.1aa: %NonTrivialBase.elem = import_ref Main//types, loc36_8, loaded [concrete = %field_decl.54f]
// CHECK:STDOUT:   %Main.import_ref.c61: type = import_ref Main//types, loc44_16, loaded [concrete = constants.%NonTrivialBase]
// CHECK:STDOUT:   %.c90: %DerivedFromNonTrivialBase.elem.932 = base_decl %Main.import_ref.c61, element0 [concrete]
// CHECK:STDOUT:   %Main.import_ref.0ac: %DerivedFromNonTrivialBase.elem.932 = import_ref Main//types, loc44_30, loaded [concrete = %.c90]
// CHECK:STDOUT:   %Main.import_ref.43f: %DerivedFromNonTrivialBase.elem.60a = import_ref Main//types, loc45_8, loaded [concrete = %field_decl.63f]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc25_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.fbd8bb.1 = import_ref Main//types, loc36_10, unloaded
// CHECK:STDOUT:   %field_decl.54f: %NonTrivialBase.elem = field_decl x, element0, %Main.import_ref.fbd8bb.1 in [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.9c2: %NonTrivialBase.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc39_24, loaded [concrete = constants.%NonTrivialBase.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6e0 = impl_witness_table (%Main.import_ref.9c2), @NonTrivialBase.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.fbd8bb.2 = import_ref Main//types, loc45_10, unloaded
// CHECK:STDOUT:   %field_decl.63f: %DerivedFromNonTrivialBase.elem.60a = field_decl y, element1, %Main.import_ref.fbd8bb.2 in [concrete] {}
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc10: type = name_ref Class, file.%Class [concrete = constants.%DerivedFromNonTrivialBase]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc10, (constants.%custom_witness.b80ef6.6) [concrete = constants.%SubobjectDestroy.facet.c4c]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Class.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.c4c]
// CHECK:STDOUT:   %Class.ref.loc11: type = name_ref Class, file.%Class [concrete = constants.%DerivedFromNonTrivialBase]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Class.ref.loc11, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Class.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %x.var: ref %DerivedFromNonTrivialBase = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %DerivedFromNonTrivialBase = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %DerivedFromNonTrivialBase to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc13: type = name_ref Class, file.%Class [concrete = constants.%DerivedFromNonTrivialBase]
// CHECK:STDOUT:   %x: ref %DerivedFromNonTrivialBase = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.dbe = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.dbe = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %struct_type.x.c57) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_9.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.4(%self.param: ref %NonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NonTrivialBase.elem = name_ref x, imports.%Main.import_ref.1aa [concrete = imports.%field_decl.54f]
// CHECK:STDOUT:   %.loc10_9.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.5(%self.param: ref %struct_type.base.y.2df) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_9.1: ref %ImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_9.1)
// CHECK:STDOUT:   %.loc10_9.4: ref %NonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc10_9.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc10_9.3 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc10_9.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc10_9.4 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.6(%self.param: ref %DerivedFromNonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.ref: %DerivedFromNonTrivialBase.elem.60a = name_ref y, imports.%Main.import_ref.43f [concrete = imports.%field_decl.63f]
// CHECK:STDOUT:   %.loc10_9.1: ref %ImplsDestroy = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_9.1)
// CHECK:STDOUT:   %base.ref: %DerivedFromNonTrivialBase.elem.932 = name_ref base, imports.%Main.import_ref.0ac [concrete = imports.%.c90]
// CHECK:STDOUT:   %.loc10_9.4: ref %NonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc10_9.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc10_9.3 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc10_9.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc10_9.4 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_grandparent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialGrandparent: type = class_type @NonTrivialGrandparent [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase: type = class_type @DerivedFromNonTrivialBase [concrete]
// CHECK:STDOUT:   %NonTrivialBase: type = class_type @NonTrivialBase [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.x.c57: type = struct_type {.x: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.elem.932: type = unbound_element_type %DerivedFromNonTrivialBase, %NonTrivialBase [concrete]
// CHECK:STDOUT:   %struct_type.base.y.2df: type = struct_type {.base: %NonTrivialBase, .y: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonTrivialGrandparent.elem: type = unbound_element_type %NonTrivialGrandparent, %DerivedFromNonTrivialBase [concrete]
// CHECK:STDOUT:   %struct_type.base.3eb: type = struct_type {.base: %DerivedFromNonTrivialBase} [concrete]
// CHECK:STDOUT:   %pattern_type.42e: type = pattern_type %NonTrivialGrandparent [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.elem: type = unbound_element_type %NonTrivialBase, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6e0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.9e6: %Destroy.type = facet_value %NonTrivialBase, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.6d9) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_9.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.elem.60a: type = unbound_element_type %DerivedFromNonTrivialBase, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c4c: %SubobjectDestroy.type = facet_value %DerivedFromNonTrivialBase, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7a2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c4c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6f5: %Destroy.type = facet_value %DerivedFromNonTrivialBase, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.7a2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_9.8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f13: %SubobjectDestroy.type = facet_value %NonTrivialGrandparent, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0d6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f13) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b1d: %Destroy.type = facet_value %NonTrivialGrandparent, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.0d6) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.42e = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.42e = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialBase.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op: %NonTrivialBase.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.1aa: %NonTrivialBase.elem = import_ref Main//types, loc36_8, loaded [concrete = %field_decl.54f]
// CHECK:STDOUT:   %Main.import_ref.c61: type = import_ref Main//types, loc44_16, loaded [concrete = constants.%NonTrivialBase]
// CHECK:STDOUT:   %.c90: %DerivedFromNonTrivialBase.elem.932 = base_decl %Main.import_ref.c61, element0 [concrete]
// CHECK:STDOUT:   %Main.import_ref.0ac: %DerivedFromNonTrivialBase.elem.932 = import_ref Main//types, loc44_30, loaded [concrete = %.c90]
// CHECK:STDOUT:   %Main.import_ref.43f: %DerivedFromNonTrivialBase.elem.60a = import_ref Main//types, loc45_8, loaded [concrete = %field_decl.63f]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc25_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.fbd8bb.1 = import_ref Main//types, loc36_10, unloaded
// CHECK:STDOUT:   %field_decl.54f: %NonTrivialBase.elem = field_decl x, element0, %Main.import_ref.fbd8bb.1 in [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.9c2: %NonTrivialBase.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc39_24, loaded [concrete = constants.%NonTrivialBase.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6e0 = impl_witness_table (%Main.import_ref.9c2), @NonTrivialBase.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.fbd8bb.2 = import_ref Main//types, loc45_10, unloaded
// CHECK:STDOUT:   %field_decl.63f: %DerivedFromNonTrivialBase.elem.60a = field_decl y, element1, %Main.import_ref.fbd8bb.2 in [concrete] {}
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc14: type = name_ref Class, file.%Class [concrete = constants.%NonTrivialGrandparent]
// CHECK:STDOUT:   %Core.ref.loc14: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc14, (constants.%custom_witness.b80ef6.8) [concrete = constants.%SubobjectDestroy.facet.f13]
// CHECK:STDOUT:   %.loc14: %SubobjectDestroy.type = converted %Class.ref.loc14, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.f13]
// CHECK:STDOUT:   %Class.ref.loc15: type = name_ref Class, file.%Class [concrete = constants.%NonTrivialGrandparent]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc15: %Destroy.type = facet_value %Class.ref.loc15, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.0d6) [concrete = constants.%Destroy.facet.b1d]
// CHECK:STDOUT:   %.loc15: %Destroy.type = converted %Class.ref.loc15, %Destroy.facet.loc15 [concrete = constants.%Destroy.facet.b1d]
// CHECK:STDOUT:   %x.var: ref %NonTrivialGrandparent = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc17_3.1: ref %NonTrivialGrandparent = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NonTrivialGrandparent to %.loc17_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc17: type = name_ref Class, file.%Class [concrete = constants.%NonTrivialGrandparent]
// CHECK:STDOUT:   %x: ref %NonTrivialGrandparent = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.42e = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.42e = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%NonTrivialGrandparent, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.0d6) [concrete = constants.%Destroy.facet.b1d]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%NonTrivialGrandparent, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet.b1d]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%NonTrivialGrandparent, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.0d6) [concrete = constants.%Destroy.facet.b1d]
// CHECK:STDOUT:   %.loc17_3.3: %Destroy.type = converted constants.%NonTrivialGrandparent, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet.b1d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.3(%self.param: ref %struct_type.x.c57) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_9.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc14_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.4(%self.param: ref %NonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NonTrivialBase.elem = name_ref x, imports.%Main.import_ref.1aa [concrete = imports.%field_decl.54f]
// CHECK:STDOUT:   %.loc14_9.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc14_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.5(%self.param: ref %struct_type.base.y.2df) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_9.1: ref %ImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc14_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_9.1)
// CHECK:STDOUT:   %.loc14_9.4: ref %NonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_9.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc14_9.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_9.3 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %Destroy.facet.loc14_9.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc14_9.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_9.4 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.6(%self.param: ref %DerivedFromNonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.ref: %DerivedFromNonTrivialBase.elem.60a = name_ref y, imports.%Main.import_ref.43f [concrete = imports.%field_decl.63f]
// CHECK:STDOUT:   %.loc14_9.1: ref %ImplsDestroy = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc14_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_9.1)
// CHECK:STDOUT:   %base.ref: %DerivedFromNonTrivialBase.elem.932 = name_ref base, imports.%Main.import_ref.0ac [concrete = imports.%.c90]
// CHECK:STDOUT:   %.loc14_9.4: ref %NonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_9.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc14_9.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_9.3 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %Destroy.facet.loc14_9.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc14_9.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_9.4 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.7(%self.param: ref %struct_type.base.3eb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_9.1: ref %DerivedFromNonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_9.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc14_9.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc14_9.1 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %Destroy.facet.loc14_9.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc14_9.3: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc14_9.2 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.8(%self.param: ref %NonTrivialGrandparent) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %NonTrivialGrandparent.elem = name_ref base, @NonTrivialGrandparent.%.loc6 [concrete = @NonTrivialGrandparent.%.loc6]
// CHECK:STDOUT:   %.loc14_9.1: ref %DerivedFromNonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_9.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc14_9.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc14_9.1 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %Destroy.facet.loc14_9.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc14_9.3: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc14_9.2 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_derived_has_trivial_base_and_impls_destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %TrivialBase: type = class_type @TrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %TrivialBase [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.17e: <witness> = impl_witness @Derived.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %struct_type.base.223: type = struct_type {.base: %TrivialBase} [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_9.2 [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_9.4 [concrete]
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
// CHECK:STDOUT:   %Class.ref.loc18: type = name_ref Class, file.%Class [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc18, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   %.loc18: %SubobjectDestroy.type = converted %Class.ref.loc18, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   %Class.ref.loc19: type = name_ref Class, file.%Class [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc19: %Destroy.type = facet_value %Class.ref.loc19, (constants.%DestroyImpl.impl_witness.17e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   %.loc19: %Destroy.type = converted %Class.ref.loc19, %Destroy.facet.loc19 [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   %x.var: ref %Derived = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc21_3.1: ref %Derived = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc21_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc21: type = name_ref Class, file.%Class [concrete = constants.%Derived]
// CHECK:STDOUT:   %x: ref %Derived = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc21_3.1: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.17e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   %.loc21_3.2: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc21_3.1 [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   %Destroy.facet.loc21_3.2: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.17e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   %.loc21_3.3: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc21_3.2 [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_9.2(%self.param: ref %TrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_9.3(%self.param: ref %struct_type.base.223) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_9.1: ref %TrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_9.1: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc18_9.2: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc18_9.1 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %Destroy.facet.loc18_9.2: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc18_9.3: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc18_9.2 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_9.4(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %Derived.elem = name_ref base, @Derived.%.loc6 [concrete = @Derived.%.loc6]
// CHECK:STDOUT:   %.loc18_9.1: ref %TrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_9.1: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc18_9.2: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc18_9.1 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %Destroy.facet.loc18_9.2: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc18_9.3: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc18_9.2 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_generic_class.carbon
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
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc5_40.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %struct_type.x.c57: type = struct_type {.x: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %GenericClass.62d: type = class_type @GenericClass, @GenericClass(%Destroy.facet.c9e) [concrete]
// CHECK:STDOUT:   %pattern_type.9f9: type = pattern_type %GenericClass.62d [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericClass.elem.e29: type = unbound_element_type %GenericClass.62d, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e88: %SubobjectDestroy.type = facet_value %GenericClass.62d, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.52a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e88) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7d0: %Destroy.type = facet_value %GenericClass.62d, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.52a) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.9f9 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.9f9 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.f7a: @GenericClass.%GenericClass.elem (%GenericClass.elem.70a) = import_ref Main//types, loc49_8, loaded [concrete = %field_decl]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc25_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Main.import_ref.eb2 = import_ref Main//types, loc49_10, unloaded
// CHECK:STDOUT:   %field_decl: @GenericClass.%GenericClass.elem (%GenericClass.elem.70a) = field_decl x, element0, %Main.import_ref.eb2 in [concrete] {}
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc10: type = name_ref Class, file.%Class [concrete = constants.%GenericClass.62d]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc10, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.e88]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Class.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.e88]
// CHECK:STDOUT:   %Class.ref.loc11: type = name_ref Class, file.%Class [concrete = constants.%GenericClass.62d]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Class.ref.loc11, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.52a) [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Class.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %x.var: ref %GenericClass.62d = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %GenericClass.62d = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %GenericClass.62d to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc13: type = name_ref Class, file.%Class [concrete = constants.%GenericClass.62d]
// CHECK:STDOUT:   %x: ref %GenericClass.62d = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.9f9 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.9f9 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%GenericClass.62d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.52a) [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%GenericClass.62d, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%GenericClass.62d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.52a) [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%GenericClass.62d, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %struct_type.x.c57) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_9.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %GenericClass.62d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %GenericClass.elem.e29 = name_ref x, imports.%Main.import_ref.f7a [concrete = imports.%field_decl]
// CHECK:STDOUT:   %.loc10_9.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_base_and_trivial_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember: type = class_type @NonTrivialBaseAndTrivialMember [concrete]
// CHECK:STDOUT:   %NonTrivialBase: type = class_type @NonTrivialBase [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.x.c57: type = struct_type {.x: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember.elem.e27: type = unbound_element_type %NonTrivialBaseAndTrivialMember, %NonTrivialBase [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember.elem.354: type = unbound_element_type %NonTrivialBaseAndTrivialMember, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.base.y.6e0: type = struct_type {.base: %NonTrivialBase, .y: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %pattern_type.c4f: type = pattern_type %NonTrivialBaseAndTrivialMember [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.elem: type = unbound_element_type %NonTrivialBase, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_9.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6e0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.9e6: %Destroy.type = facet_value %NonTrivialBase, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.6d9) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_9.7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.78d: %SubobjectDestroy.type = facet_value %NonTrivialBaseAndTrivialMember, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.75e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.78d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3cb: %Destroy.type = facet_value %NonTrivialBaseAndTrivialMember, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.75e) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.c4f = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.c4f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialBase.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op: %NonTrivialBase.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.1aa: %NonTrivialBase.elem = import_ref Main//types, loc36_8, loaded [concrete = %field_decl]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc25_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.fbd = import_ref Main//types, loc36_10, unloaded
// CHECK:STDOUT:   %field_decl: %NonTrivialBase.elem = field_decl x, element0, %Main.import_ref.fbd in [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.9c2: %NonTrivialBase.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc39_24, loaded [concrete = constants.%NonTrivialBase.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6e0 = impl_witness_table (%Main.import_ref.9c2), @NonTrivialBase.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc15: type = name_ref Class, file.%Class [concrete = constants.%NonTrivialBaseAndTrivialMember]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc15, (constants.%custom_witness.b80ef6.7) [concrete = constants.%SubobjectDestroy.facet.78d]
// CHECK:STDOUT:   %.loc15: %SubobjectDestroy.type = converted %Class.ref.loc15, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.78d]
// CHECK:STDOUT:   %Class.ref.loc16: type = name_ref Class, file.%Class [concrete = constants.%NonTrivialBaseAndTrivialMember]
// CHECK:STDOUT:   %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc16: %Destroy.type = facet_value %Class.ref.loc16, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.75e) [concrete = constants.%Destroy.facet.3cb]
// CHECK:STDOUT:   %.loc16: %Destroy.type = converted %Class.ref.loc16, %Destroy.facet.loc16 [concrete = constants.%Destroy.facet.3cb]
// CHECK:STDOUT:   %x.var: ref %NonTrivialBaseAndTrivialMember = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc18_3.1: ref %NonTrivialBaseAndTrivialMember = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NonTrivialBaseAndTrivialMember to %.loc18_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc18: type = name_ref Class, file.%Class [concrete = constants.%NonTrivialBaseAndTrivialMember]
// CHECK:STDOUT:   %x: ref %NonTrivialBaseAndTrivialMember = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.c4f = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c4f = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%NonTrivialBaseAndTrivialMember, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.75e) [concrete = constants.%Destroy.facet.3cb]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%NonTrivialBaseAndTrivialMember, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.3cb]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%NonTrivialBaseAndTrivialMember, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.75e) [concrete = constants.%Destroy.facet.3cb]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%NonTrivialBaseAndTrivialMember, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.3cb]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.3(%self.param: ref %struct_type.x.c57) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_9.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc15_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc15_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc15_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc15_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.4(%self.param: ref %NonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NonTrivialBase.elem = name_ref x, imports.%Main.import_ref.1aa [concrete = imports.%field_decl]
// CHECK:STDOUT:   %.loc15_9.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc15_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc15_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc15_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc15_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.5(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.6(%self.param: ref %struct_type.base.y.6e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_9.1: ref %HasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc15_9.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc15_9.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_9.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc15_9.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc15_9.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_9.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc15_9.1)
// CHECK:STDOUT:   %.loc15_9.4: ref %NonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_9.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc15_9.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc15_9.3 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %Destroy.facet.loc15_9.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc15_9.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc15_9.4 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc15_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.7(%self.param: ref %NonTrivialBaseAndTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.ref: %NonTrivialBaseAndTrivialMember.elem.354 = name_ref y, @NonTrivialBaseAndTrivialMember.%field_decl [concrete = @NonTrivialBaseAndTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc15_9.1: ref %HasTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc15_9.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc15_9.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_9.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc15_9.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc15_9.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_9.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc15_9.1)
// CHECK:STDOUT:   %base.ref: %NonTrivialBaseAndTrivialMember.elem.e27 = name_ref base, @NonTrivialBaseAndTrivialMember.%.loc6 [concrete = @NonTrivialBaseAndTrivialMember.%.loc6]
// CHECK:STDOUT:   %.loc15_9.4: ref %NonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_9.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc15_9.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc15_9.3 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %Destroy.facet.loc15_9.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc15_9.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc15_9.4 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc15_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_trivial_base_and_nontrivial_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialBaseAndNonTrivialMember: type = class_type @TrivialBaseAndNonTrivialMember [concrete]
// CHECK:STDOUT:   %TrivialBase: type = class_type @TrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TrivialBaseAndNonTrivialMember.elem.8b0: type = unbound_element_type %TrivialBaseAndNonTrivialMember, %TrivialBase [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %TrivialBaseAndNonTrivialMember.elem.9f5: type = unbound_element_type %TrivialBaseAndNonTrivialMember, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.base.x.bb9: type = struct_type {.base: %TrivialBase, .x: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %pattern_type.575: type = pattern_type %TrivialBaseAndNonTrivialMember [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.23e: %SubobjectDestroy.type = facet_value %TrivialBase, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7c0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.23e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.033: %Destroy.type = facet_value %TrivialBase, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.7c0) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_9.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a42: %SubobjectDestroy.type = facet_value %TrivialBaseAndNonTrivialMember, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4c7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a42) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6ca: %Destroy.type = facet_value %TrivialBaseAndNonTrivialMember, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.4c7) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.575 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.575 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc25_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc15: type = name_ref Class, file.%Class [concrete = constants.%TrivialBaseAndNonTrivialMember]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc15, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.a42]
// CHECK:STDOUT:   %.loc15: %SubobjectDestroy.type = converted %Class.ref.loc15, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.a42]
// CHECK:STDOUT:   %Class.ref.loc16: type = name_ref Class, file.%Class [concrete = constants.%TrivialBaseAndNonTrivialMember]
// CHECK:STDOUT:   %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc16: %Destroy.type = facet_value %Class.ref.loc16, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.4c7) [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %.loc16: %Destroy.type = converted %Class.ref.loc16, %Destroy.facet.loc16 [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %x.var: ref %TrivialBaseAndNonTrivialMember = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc18_3.1: ref %TrivialBaseAndNonTrivialMember = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %TrivialBaseAndNonTrivialMember to %.loc18_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc18: type = name_ref Class, file.%Class [concrete = constants.%TrivialBaseAndNonTrivialMember]
// CHECK:STDOUT:   %x: ref %TrivialBaseAndNonTrivialMember = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.575 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.575 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%TrivialBaseAndNonTrivialMember, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.4c7) [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%TrivialBaseAndNonTrivialMember, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%TrivialBaseAndNonTrivialMember, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.4c7) [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%TrivialBaseAndNonTrivialMember, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.2(%self.param: ref %TrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.3(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.4(%self.param: ref %struct_type.base.x.bb9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_9.1: ref %ImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc15_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc15_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc15_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc15_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc15_9.1)
// CHECK:STDOUT:   %.loc15_9.4: ref %TrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_9.3: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc15_9.5: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc15_9.3 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %Destroy.facet.loc15_9.4: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc15_9.6: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc15_9.4 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc15_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_9.5(%self.param: ref %TrivialBaseAndNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %TrivialBaseAndNonTrivialMember.elem.9f5 = name_ref x, @TrivialBaseAndNonTrivialMember.%field_decl [concrete = @TrivialBaseAndNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc15_9.1: ref %ImplsDestroy = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc15_9.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc15_9.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_9.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc15_9.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc15_9.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_9.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc15_9.1)
// CHECK:STDOUT:   %base.ref: %TrivialBaseAndNonTrivialMember.elem.8b0 = name_ref base, @TrivialBaseAndNonTrivialMember.%.loc6 [concrete = @TrivialBaseAndNonTrivialMember.%.loc6]
// CHECK:STDOUT:   %.loc15_9.4: ref %TrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_9.3: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc15_9.5: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc15_9.3 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %Destroy.facet.loc15_9.4: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc15_9.6: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc15_9.4 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc15_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_class_dynamic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass: type = class_type @NonTrivialDynamicClass [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %NonTrivialDynamicClass [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c1a: %SubobjectDestroy.type = facet_value %ptr.454, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cad: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c1a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.50f: %Destroy.type = facet_value %ptr.454, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.cad) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.27c: %SubobjectDestroy.type = facet_value %NonTrivialDynamicClass, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.501: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6ec [concrete]
// CHECK:STDOUT:   %Destroy.facet.22e: %Destroy.type = facet_value %NonTrivialDynamicClass, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.501) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.157 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.157 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialDynamicClass.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.as.DestroyImpl.impl.Op: %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.133: %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc60_24, loaded [concrete = constants.%NonTrivialDynamicClass.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6ec = impl_witness_table (%Main.import_ref.133), @NonTrivialDynamicClass.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc10: type = name_ref Class, file.%Class [concrete = constants.%NonTrivialDynamicClass]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc10, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.27c]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Class.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.27c]
// CHECK:STDOUT:   %Class.ref.loc11: type = name_ref Class, file.%Class [concrete = constants.%NonTrivialDynamicClass]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Class.ref.loc11, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.501) [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Class.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   %x.var: ref %NonTrivialDynamicClass = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %NonTrivialDynamicClass = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NonTrivialDynamicClass to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc13: type = name_ref Class, file.%Class [concrete = constants.%NonTrivialDynamicClass]
// CHECK:STDOUT:   %x: ref %NonTrivialDynamicClass = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.157 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.157 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.501) [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.501) [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_9.1: ref %ptr.454 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%ptr.454, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.cad) [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%ptr.454, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.cad) [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %NonTrivialDynamicClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_dynamic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass: type = class_type @NonTrivialDynamicClass [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %NonTrivialDynamicClass [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.F.type: type = fn_type @NonTrivialDynamicClass.F [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.F: %NonTrivialDynamicClass.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.7c1: type = struct_type {.base: %NonTrivialDynamicClass} [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_9.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c1a: %SubobjectDestroy.type = facet_value %ptr.454, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cad: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c1a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.50f: %Destroy.type = facet_value %ptr.454, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.cad) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.501: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6ec [concrete]
// CHECK:STDOUT:   %Destroy.facet.22e: %Destroy.type = facet_value %NonTrivialDynamicClass, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.501) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_9.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %Derived, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.660: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f2a: %Destroy.type = facet_value %Derived, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.660) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialDynamicClass.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.as.DestroyImpl.impl.Op: %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.133: %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc60_24, loaded [concrete = constants.%NonTrivialDynamicClass.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6ec = impl_witness_table (%Main.import_ref.133), @NonTrivialDynamicClass.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @NonTrivialDynamicClass.vtable {
// CHECK:STDOUT:   constants.%NonTrivialDynamicClass.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   constants.%NonTrivialDynamicClass.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc14: type = name_ref Class, file.%Class [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc14: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc14, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   %.loc14: %SubobjectDestroy.type = converted %Class.ref.loc14, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.1fc]
// CHECK:STDOUT:   %Class.ref.loc15: type = name_ref Class, file.%Class [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc15: %Destroy.type = facet_value %Class.ref.loc15, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %.loc15: %Destroy.type = converted %Class.ref.loc15, %Destroy.facet.loc15 [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %x.var: ref %Derived = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc17_3.1: ref %Derived = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc17_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc17: type = name_ref Class, file.%Class [concrete = constants.%Derived]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_9.1: ref %ptr.454 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_9.1: %Destroy.type = facet_value constants.%ptr.454, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.cad) [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %.loc14_9.2: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc14_9.1 [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %Destroy.facet.loc14_9.2: %Destroy.type = facet_value constants.%ptr.454, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.cad) [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   %.loc14_9.3: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc14_9.2 [concrete = constants.%Destroy.facet.50f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.3(%self.param: ref %NonTrivialDynamicClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.4(%self.param: ref %struct_type.base.7c1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_9.1: ref %NonTrivialDynamicClass = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_9.1: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.501) [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   %.loc14_9.2: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc14_9.1 [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   %Destroy.facet.loc14_9.2: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.501) [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   %.loc14_9.3: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc14_9.2 [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_9.5(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %Derived.elem = name_ref base, @Derived.%.loc6 [concrete = @Derived.%.loc6]
// CHECK:STDOUT:   %.loc14_9.1: ref %NonTrivialDynamicClass = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_9.1: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.501) [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   %.loc14_9.2: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc14_9.1 [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   %Destroy.facet.loc14_9.2: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.501) [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   %.loc14_9.3: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc14_9.2 [concrete = constants.%Destroy.facet.22e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
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
// CHECK:STDOUT:   %Class.ref.loc27: type = name_ref Class, file.%Class [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc27: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Class.ref.loc35: type = name_ref Class, file.%Class [concrete = constants.%Derived]
// CHECK:STDOUT:   %Core.ref.loc35: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %x.var: ref %Derived = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc51: ref %Derived = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc51 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc51: type = name_ref Class, file.%Class [concrete = constants.%Derived]
// CHECK:STDOUT:   %x: ref %Derived = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
