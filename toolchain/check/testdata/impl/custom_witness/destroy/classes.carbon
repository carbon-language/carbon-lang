// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
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

// --- todo_nontrivial_empty_class_impls_destroy.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Make() -> ImplsDestroy;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: ImplsDestroy = Make();
  //@dump-sem-ir-end
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

fn Make() -> AllNonTrivialFields;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: AllNonTrivialFields = Make();
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

fn Make() -> NestedNonTrivialMember;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: NestedNonTrivialMember = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_derived_from_nontrivial_base.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Make() -> DerivedFromNonTrivialBase;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: DerivedFromNonTrivialBase = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_grandparent.carbon

library "[[@TEST_NAME]]";

import library "types";

class NonTrivialGrandparent {
  extend base: DerivedFromNonTrivialBase;
}

fn Make() -> NonTrivialGrandparent;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: NonTrivialGrandparent = Make();
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

fn Make() -> Derived;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Derived = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_generic_class.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Make() -> GenericClass(ImplsDestroy);

fn Test() {
  //@dump-sem-ir-begin
  var unused x: GenericClass(ImplsDestroy) = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_base_and_trivial_field.carbon

library "[[@TEST_NAME]]";

import library "types";

class NonTrivialBaseAndTrivialMember {
  extend base: NonTrivialBase;
  var y: HasTrivialDestruction;
}

fn Make() -> NonTrivialBaseAndTrivialMember;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: NonTrivialBaseAndTrivialMember = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_trivial_base_and_nontrivial_field.carbon

library "[[@TEST_NAME]]";

import library "types";

class TrivialBaseAndNonTrivialMember {
  extend base: TrivialBase;
  var x: ImplsDestroy;
}

fn Make() -> TrivialBaseAndNonTrivialMember;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: TrivialBaseAndNonTrivialMember = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_class_dynamic.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Make() -> NonTrivialDynamicClass;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: NonTrivialDynamicClass = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_dynamic_base.carbon

library "[[@TEST_NAME]]";

import library "types";

class Derived {
  extend base: NonTrivialDynamicClass;
}

fn Make() -> Derived;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Derived = Make();
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

fn Make() -> Derived;

// TODO: Support destroying classes that derive from abstract types.
fn Test() {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon:[[@LINE+14]]:3
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
// CHECK:STDOUT:   %x.patt: %pattern_type.6b9 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.6b9 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.ImplsDestroy: type = import_ref Main//types, ImplsDestroy, loaded [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc23_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %ImplsDestroy = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_3.1: ref %ImplsDestroy = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %ImplsDestroy to %.loc10_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %ImplsDestroy.ref: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:   %x: ref %ImplsDestroy = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6b9 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.6b9 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_all_nontrivial_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e7f: <witness> = impl_witness @AlsoImplsDestroy.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %AllNonTrivialFields: type = class_type @AllNonTrivialFields [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %AllNonTrivialFields.elem.b78: type = unbound_element_type %AllNonTrivialFields, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %AllNonTrivialFields.elem.8d2: type = unbound_element_type %AllNonTrivialFields, %AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.f50: type = struct_type {.a: %ImplsDestroy, .b: %AlsoImplsDestroy, .c: %AlsoImplsDestroy, .d: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %pattern_type.c66: type = pattern_type %AllNonTrivialFields [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.c66 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.c66 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.facet.122: %Destroy.type = facet_value %AlsoImplsDestroy, (%DestroyImpl.impl_witness.e7f, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23_3.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.075: %SubobjectDestroy.type = facet_value %AllNonTrivialFields, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.124: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.075) [concrete]
// CHECK:STDOUT:   %Destroy.facet.468: %Destroy.type = facet_value %AllNonTrivialFields, (%DestroyImpl.impl_witness.124, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc23_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %AllNonTrivialFields = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc23_3.1: ref %AllNonTrivialFields = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %AllNonTrivialFields to %.loc23_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %AllNonTrivialFields.ref: type = name_ref AllNonTrivialFields, file.%AllNonTrivialFields.decl [concrete = constants.%AllNonTrivialFields]
// CHECK:STDOUT:   %x: ref %AllNonTrivialFields = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.c66 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c66 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc23_3.1: %Destroy.type = facet_value constants.%AllNonTrivialFields, (constants.%DestroyImpl.impl_witness.124, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.468]
// CHECK:STDOUT:   %.loc23_3.2: %Destroy.type = converted constants.%AllNonTrivialFields, %Destroy.facet.loc23_3.1 [concrete = constants.%Destroy.facet.468]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.2: %Destroy.type = facet_value constants.%AllNonTrivialFields, (constants.%DestroyImpl.impl_witness.124, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.468]
// CHECK:STDOUT:   %.loc23_3.3: %Destroy.type = converted constants.%AllNonTrivialFields, %Destroy.facet.loc23_3.2 [concrete = constants.%Destroy.facet.468]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_3.3(%self.param: ref %AlsoImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_3.4(%self.param: ref %struct_type.a.b.c.d.f50) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23_3.1: ref %ImplsDestroy = struct_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc23_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc23_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc23_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc23_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc23_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc23_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc23_3.1)
// CHECK:STDOUT:   %.loc23_3.4: ref %AlsoImplsDestroy = struct_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc23_3.3: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc23_3.5: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc23_3.3 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.4: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc23_3.6: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc23_3.4 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc23_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc23_3.4)
// CHECK:STDOUT:   %.loc23_3.7: ref %AlsoImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc23_3.5: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc23_3.8: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc23_3.5 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.6: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc23_3.9: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc23_3.6 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc23_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc23_3.7)
// CHECK:STDOUT:   %.loc23_3.10: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc23_3.7: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc23_3.11: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc23_3.7 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.8: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc23_3.12: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc23_3.8 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc23_3.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc23_3.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_3.5(%self.param: ref %AllNonTrivialFields) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %AllNonTrivialFields.elem.b78 = name_ref d, @AllNonTrivialFields.%field_decl.loc16 [concrete = @AllNonTrivialFields.%field_decl.loc16]
// CHECK:STDOUT:   %.loc23_3.1: ref %ImplsDestroy = class_element_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc23_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc23_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc23_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc23_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc23_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc23_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc23_3.1)
// CHECK:STDOUT:   %c.ref: %AllNonTrivialFields.elem.8d2 = name_ref c, @AllNonTrivialFields.%field_decl.loc15 [concrete = @AllNonTrivialFields.%field_decl.loc15]
// CHECK:STDOUT:   %.loc23_3.4: ref %AlsoImplsDestroy = class_element_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc23_3.3: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc23_3.5: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc23_3.3 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.4: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc23_3.6: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc23_3.4 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc23_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc23_3.4)
// CHECK:STDOUT:   %b.ref: %AllNonTrivialFields.elem.8d2 = name_ref b, @AllNonTrivialFields.%field_decl.loc14 [concrete = @AllNonTrivialFields.%field_decl.loc14]
// CHECK:STDOUT:   %.loc23_3.7: ref %AlsoImplsDestroy = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc23_3.5: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc23_3.8: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc23_3.5 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.6: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc23_3.9: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc23_3.6 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc23_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc23_3.7)
// CHECK:STDOUT:   %a.ref: %AllNonTrivialFields.elem.b78 = name_ref a, @AllNonTrivialFields.%field_decl.loc13 [concrete = @AllNonTrivialFields.%field_decl.loc13]
// CHECK:STDOUT:   %.loc23_3.10: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc23_3.7: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc23_3.11: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc23_3.7 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.8: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc23_3.12: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc23_3.8 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc23_3.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc23_3.10)
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
// CHECK:STDOUT:   %x.patt: %pattern_type.03f = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.03f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.959: %SubobjectDestroy.type = facet_value %HasNonTrivialField, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d61: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.959) [concrete]
// CHECK:STDOUT:   %Destroy.facet.54a: %Destroy.type = facet_value %HasNonTrivialField, (%DestroyImpl.impl_witness.d61, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_3.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.446: %SubobjectDestroy.type = facet_value %NestedNonTrivialMember, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c6f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.446) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f68: %Destroy.type = facet_value %NestedNonTrivialMember, (%DestroyImpl.impl_witness.c6f, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc23_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %NestedNonTrivialMember = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc18_3.1: ref %NestedNonTrivialMember = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NestedNonTrivialMember to %.loc18_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %NestedNonTrivialMember.ref: type = name_ref NestedNonTrivialMember, file.%NestedNonTrivialMember.decl [concrete = constants.%NestedNonTrivialMember]
// CHECK:STDOUT:   %x: ref %NestedNonTrivialMember = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.03f = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.03f = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%NestedNonTrivialMember, (constants.%DestroyImpl.impl_witness.c6f, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.f68]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%NestedNonTrivialMember, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.f68]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%NestedNonTrivialMember, (constants.%DestroyImpl.impl_witness.c6f, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.f68]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%NestedNonTrivialMember, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.f68]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.3(%self.param: ref %struct_type.x.c57) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_3.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.4(%self.param: ref %HasNonTrivialField) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %HasNonTrivialField.elem = name_ref x, @HasNonTrivialField.%field_decl [concrete = @HasNonTrivialField.%field_decl]
// CHECK:STDOUT:   %.loc18_3.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.5(%self.param: ref %struct_type.x.3f9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_3.1: ref %HasNonTrivialField = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%DestroyImpl.impl_witness.d61, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.54a]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.54a]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%DestroyImpl.impl_witness.d61, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.54a]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.54a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.6(%self.param: ref %NestedNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NestedNonTrivialMember.elem = name_ref x, @NestedNonTrivialMember.%field_decl [concrete = @NestedNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc18_3.1: ref %HasNonTrivialField = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%DestroyImpl.impl_witness.d61, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.54a]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.54a]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%HasNonTrivialField, (constants.%DestroyImpl.impl_witness.d61, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.54a]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%HasNonTrivialField, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.54a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_3.1)
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
// CHECK:STDOUT:   %x.patt: %pattern_type.dbe = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.dbe = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.elem: type = unbound_element_type %NonTrivialBase, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6e0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c8f: %Destroy.type = facet_value %NonTrivialBase, (%DestroyImpl.impl_witness.6d9, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.elem.60a: type = unbound_element_type %DerivedFromNonTrivialBase, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c4c: %SubobjectDestroy.type = facet_value %DerivedFromNonTrivialBase, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7a2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c4c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.14e: %Destroy.type = facet_value %DerivedFromNonTrivialBase, (%DestroyImpl.impl_witness.7a2, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialBase.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op: %NonTrivialBase.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.DerivedFromNonTrivialBase: type = import_ref Main//types, DerivedFromNonTrivialBase, loaded [concrete = constants.%DerivedFromNonTrivialBase]
// CHECK:STDOUT:   %Main.import_ref.1aa: %NonTrivialBase.elem = import_ref Main//types, loc28_8, loaded [concrete = %field_decl.54f]
// CHECK:STDOUT:   %Main.import_ref.c61: type = import_ref Main//types, loc36_16, loaded [concrete = constants.%NonTrivialBase]
// CHECK:STDOUT:   %.c90: %DerivedFromNonTrivialBase.elem.932 = base_decl %Main.import_ref.c61, element0 [concrete]
// CHECK:STDOUT:   %Main.import_ref.0ac: %DerivedFromNonTrivialBase.elem.932 = import_ref Main//types, loc36_30, loaded [concrete = %.c90]
// CHECK:STDOUT:   %Main.import_ref.43f: %DerivedFromNonTrivialBase.elem.60a = import_ref Main//types, loc37_8, loaded [concrete = %field_decl.63f]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc23_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.fbd8bb.1 = import_ref Main//types, loc28_10, unloaded
// CHECK:STDOUT:   %field_decl.54f: %NonTrivialBase.elem = field_decl x, element0, %Main.import_ref.fbd8bb.1 in [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.9c2: %NonTrivialBase.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc31_24, loaded [concrete = constants.%NonTrivialBase.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6e0 = impl_witness_table (%Main.import_ref.9c2), @NonTrivialBase.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.fbd8bb.2 = import_ref Main//types, loc37_10, unloaded
// CHECK:STDOUT:   %field_decl.63f: %DerivedFromNonTrivialBase.elem.60a = field_decl y, element1, %Main.import_ref.fbd8bb.2 in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %DerivedFromNonTrivialBase = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_3.1: ref %DerivedFromNonTrivialBase = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %DerivedFromNonTrivialBase to %.loc10_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.ref: type = name_ref DerivedFromNonTrivialBase, imports.%Main.DerivedFromNonTrivialBase [concrete = constants.%DerivedFromNonTrivialBase]
// CHECK:STDOUT:   %x: ref %DerivedFromNonTrivialBase = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.dbe = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.dbe = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.3(%self.param: ref %struct_type.x.c57) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_3.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.4(%self.param: ref %NonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NonTrivialBase.elem = name_ref x, imports.%Main.import_ref.1aa [concrete = imports.%field_decl.54f]
// CHECK:STDOUT:   %.loc10_3.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.5(%self.param: ref %struct_type.base.y.2df) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_3.1: ref %ImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_3.1)
// CHECK:STDOUT:   %.loc10_3.4: ref %NonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_3.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc10_3.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc10_3.3 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc10_3.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc10_3.4 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.6(%self.param: ref %DerivedFromNonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.ref: %DerivedFromNonTrivialBase.elem.60a = name_ref y, imports.%Main.import_ref.43f [concrete = imports.%field_decl.63f]
// CHECK:STDOUT:   %.loc10_3.1: ref %ImplsDestroy = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_3.1)
// CHECK:STDOUT:   %base.ref: %DerivedFromNonTrivialBase.elem.932 = name_ref base, imports.%Main.import_ref.0ac [concrete = imports.%.c90]
// CHECK:STDOUT:   %.loc10_3.4: ref %NonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_3.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc10_3.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc10_3.3 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc10_3.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc10_3.4 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_3.4)
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
// CHECK:STDOUT:   %x.patt: %pattern_type.42e = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.42e = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.elem: type = unbound_element_type %NonTrivialBase, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6e0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c8f: %Destroy.type = facet_value %NonTrivialBase, (%DestroyImpl.impl_witness.6d9, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_3.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.elem.60a: type = unbound_element_type %DerivedFromNonTrivialBase, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c4c: %SubobjectDestroy.type = facet_value %DerivedFromNonTrivialBase, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7a2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c4c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.14e: %Destroy.type = facet_value %DerivedFromNonTrivialBase, (%DestroyImpl.impl_witness.7a2, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_3.8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f13: %SubobjectDestroy.type = facet_value %NonTrivialGrandparent, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0d6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f13) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f1d: %Destroy.type = facet_value %NonTrivialGrandparent, (%DestroyImpl.impl_witness.0d6, %custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialBase.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op: %NonTrivialBase.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.import_ref.1aa: %NonTrivialBase.elem = import_ref Main//types, loc28_8, loaded [concrete = %field_decl.54f]
// CHECK:STDOUT:   %Main.import_ref.c61: type = import_ref Main//types, loc36_16, loaded [concrete = constants.%NonTrivialBase]
// CHECK:STDOUT:   %.c90: %DerivedFromNonTrivialBase.elem.932 = base_decl %Main.import_ref.c61, element0 [concrete]
// CHECK:STDOUT:   %Main.import_ref.0ac: %DerivedFromNonTrivialBase.elem.932 = import_ref Main//types, loc36_30, loaded [concrete = %.c90]
// CHECK:STDOUT:   %Main.import_ref.43f: %DerivedFromNonTrivialBase.elem.60a = import_ref Main//types, loc37_8, loaded [concrete = %field_decl.63f]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc23_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.fbd8bb.1 = import_ref Main//types, loc28_10, unloaded
// CHECK:STDOUT:   %field_decl.54f: %NonTrivialBase.elem = field_decl x, element0, %Main.import_ref.fbd8bb.1 in [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.9c2: %NonTrivialBase.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc31_24, loaded [concrete = constants.%NonTrivialBase.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6e0 = impl_witness_table (%Main.import_ref.9c2), @NonTrivialBase.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.fbd8bb.2 = import_ref Main//types, loc37_10, unloaded
// CHECK:STDOUT:   %field_decl.63f: %DerivedFromNonTrivialBase.elem.60a = field_decl y, element1, %Main.import_ref.fbd8bb.2 in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %NonTrivialGrandparent = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %NonTrivialGrandparent = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NonTrivialGrandparent to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %NonTrivialGrandparent.ref: type = name_ref NonTrivialGrandparent, file.%NonTrivialGrandparent.decl [concrete = constants.%NonTrivialGrandparent]
// CHECK:STDOUT:   %x: ref %NonTrivialGrandparent = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.42e = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.42e = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%NonTrivialGrandparent, (constants.%DestroyImpl.impl_witness.0d6, constants.%custom_witness.b80ef6.8) [concrete = constants.%Destroy.facet.f1d]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%NonTrivialGrandparent, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.f1d]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%NonTrivialGrandparent, (constants.%DestroyImpl.impl_witness.0d6, constants.%custom_witness.b80ef6.8) [concrete = constants.%Destroy.facet.f1d]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%NonTrivialGrandparent, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.f1d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.3(%self.param: ref %struct_type.x.c57) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_3.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.4(%self.param: ref %NonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NonTrivialBase.elem = name_ref x, imports.%Main.import_ref.1aa [concrete = imports.%field_decl.54f]
// CHECK:STDOUT:   %.loc14_3.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.5(%self.param: ref %struct_type.base.y.2df) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_3.1: ref %ImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_3.1)
// CHECK:STDOUT:   %.loc14_3.4: ref %NonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_3.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc14_3.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_3.3 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc14_3.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_3.4 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.6(%self.param: ref %DerivedFromNonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.ref: %DerivedFromNonTrivialBase.elem.60a = name_ref y, imports.%Main.import_ref.43f [concrete = imports.%field_decl.63f]
// CHECK:STDOUT:   %.loc14_3.1: ref %ImplsDestroy = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_3.1)
// CHECK:STDOUT:   %base.ref: %DerivedFromNonTrivialBase.elem.932 = name_ref base, imports.%Main.import_ref.0ac [concrete = imports.%.c90]
// CHECK:STDOUT:   %.loc14_3.4: ref %NonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_3.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc14_3.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_3.3 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc14_3.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_3.4 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.7(%self.param: ref %struct_type.base.3eb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_3.1: ref %DerivedFromNonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.8(%self.param: ref %NonTrivialGrandparent) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %NonTrivialGrandparent.elem = name_ref base, @NonTrivialGrandparent.%.loc7 [concrete = @NonTrivialGrandparent.%.loc7]
// CHECK:STDOUT:   %.loc14_3.1: ref %DerivedFromNonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%DestroyImpl.impl_witness.7a2, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.14e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_3.1)
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
// CHECK:STDOUT:   %DestroyImpl.impl_witness.17e: <witness> = impl_witness @Derived.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %struct_type.base.223: type = struct_type {.base: %TrivialBase} [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.23e: %SubobjectDestroy.type = facet_value %TrivialBase, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7c0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.23e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0fd: %Destroy.type = facet_value %TrivialBase, (%DestroyImpl.impl_witness.7c0, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.b2e: %Destroy.type = facet_value %Derived, (%DestroyImpl.impl_witness.17e, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %Derived = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc18_3.1: ref %Derived = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc18_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %x: ref %Derived = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.17e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.17e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.b2e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.2(%self.param: ref %TrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.3(%self.param: ref %struct_type.base.223) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_3.1: ref %TrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.4(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %Derived.elem = name_ref base, @Derived.%.loc7 [concrete = @Derived.%.loc7]
// CHECK:STDOUT:   %.loc18_3.1: ref %TrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %GenericClass.type: type = parameterized_class_name_type @GenericClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %GenericClass.generic: %GenericClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.f45: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190: type = facet_access_type %T.f45 [symbolic]
// CHECK:STDOUT:   %GenericClass.9e4: type = class_type @GenericClass, @GenericClass(%T.f45) [symbolic]
// CHECK:STDOUT:   %GenericClass.elem.70a: type = unbound_element_type %GenericClass.9e4, %T.as_type.190 [symbolic]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc6_39.2 [concrete]
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
// CHECK:STDOUT:   %x.patt: %pattern_type.9f9 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.9f9 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e88: %SubobjectDestroy.type = facet_value %GenericClass.62d, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.52a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e88) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7d0: %Destroy.type = facet_value %GenericClass.62d, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.52a) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.ImplsDestroy: type = import_ref Main//types, ImplsDestroy, loaded [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:   %Main.GenericClass: %GenericClass.type = import_ref Main//types, GenericClass, loaded [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:   %Main.import_ref.f7a: @GenericClass.%GenericClass.elem (%GenericClass.elem.70a) = import_ref Main//types, loc41_8, loaded [concrete = %field_decl]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc23_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.eb2 = import_ref Main//types, loc41_10, unloaded
// CHECK:STDOUT:   %field_decl: @GenericClass.%GenericClass.elem (%GenericClass.elem.70a) = field_decl x, element0, %Main.import_ref.eb2 in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %GenericClass.62d = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_3.1: ref %GenericClass.62d = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %GenericClass.62d to %.loc10_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc10_42.1: type = splice_block %GenericClass [concrete = constants.%GenericClass.62d] {
// CHECK:STDOUT:     %GenericClass.ref: %GenericClass.type = name_ref GenericClass, imports.%Main.GenericClass [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:     %ImplsDestroy.ref: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:     %Destroy.facet.loc10_42: %Destroy.type = facet_value %ImplsDestroy.ref, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:     %.loc10_42.2: %Destroy.type = converted %ImplsDestroy.ref, %Destroy.facet.loc10_42 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:     %GenericClass: type = class_type @GenericClass, @GenericClass(constants.%Destroy.facet.c9e) [concrete = constants.%GenericClass.62d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %GenericClass.62d = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.9f9 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.9f9 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%GenericClass.62d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.52a) [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%GenericClass.62d, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%GenericClass.62d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.52a) [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%GenericClass.62d, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.1(%self.param: ref %struct_type.x.c57) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_3.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.2(%self.param: ref %GenericClass.62d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %GenericClass.elem.e29 = name_ref x, imports.%Main.import_ref.f7a [concrete = imports.%field_decl]
// CHECK:STDOUT:   %.loc10_3.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_3.1)
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
// CHECK:STDOUT:   %x.patt: %pattern_type.c4f = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.c4f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.elem: type = unbound_element_type %NonTrivialBase, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_3.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dba: %Destroy.type = facet_value %HasTrivialDestruction, (%DestroyImpl.impl_witness.4ca, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6e0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c8f: %Destroy.type = facet_value %NonTrivialBase, (%DestroyImpl.impl_witness.6d9, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_3.7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.78d: %SubobjectDestroy.type = facet_value %NonTrivialBaseAndTrivialMember, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.75e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.78d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7eb: %Destroy.type = facet_value %NonTrivialBaseAndTrivialMember, (%DestroyImpl.impl_witness.75e, %custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialBase.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op: %NonTrivialBase.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.import_ref.1aa: %NonTrivialBase.elem = import_ref Main//types, loc28_8, loaded [concrete = %field_decl]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc23_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.fbd = import_ref Main//types, loc28_10, unloaded
// CHECK:STDOUT:   %field_decl: %NonTrivialBase.elem = field_decl x, element0, %Main.import_ref.fbd in [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.9c2: %NonTrivialBase.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc31_24, loaded [concrete = constants.%NonTrivialBase.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6e0 = impl_witness_table (%Main.import_ref.9c2), @NonTrivialBase.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %NonTrivialBaseAndTrivialMember = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15_3.1: ref %NonTrivialBaseAndTrivialMember = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NonTrivialBaseAndTrivialMember to %.loc15_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember.ref: type = name_ref NonTrivialBaseAndTrivialMember, file.%NonTrivialBaseAndTrivialMember.decl [concrete = constants.%NonTrivialBaseAndTrivialMember]
// CHECK:STDOUT:   %x: ref %NonTrivialBaseAndTrivialMember = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.c4f = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c4f = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%NonTrivialBaseAndTrivialMember, (constants.%DestroyImpl.impl_witness.75e, constants.%custom_witness.b80ef6.7) [concrete = constants.%Destroy.facet.7eb]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%NonTrivialBaseAndTrivialMember, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet.7eb]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%NonTrivialBaseAndTrivialMember, (constants.%DestroyImpl.impl_witness.75e, constants.%custom_witness.b80ef6.7) [concrete = constants.%Destroy.facet.7eb]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%NonTrivialBaseAndTrivialMember, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet.7eb]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.3(%self.param: ref %struct_type.x.c57) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_3.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc15_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.4(%self.param: ref %NonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %NonTrivialBase.elem = name_ref x, imports.%Main.import_ref.1aa [concrete = imports.%field_decl]
// CHECK:STDOUT:   %.loc15_3.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc15_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.5(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.6(%self.param: ref %struct_type.base.y.6e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_3.1: ref %HasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc15_3.1)
// CHECK:STDOUT:   %.loc15_3.4: ref %NonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_3.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc15_3.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc15_3.3 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc15_3.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc15_3.4 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc15_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.7(%self.param: ref %NonTrivialBaseAndTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.ref: %NonTrivialBaseAndTrivialMember.elem.354 = name_ref y, @NonTrivialBaseAndTrivialMember.%field_decl [concrete = @NonTrivialBaseAndTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc15_3.1: ref %HasTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc15_3.1)
// CHECK:STDOUT:   %base.ref: %NonTrivialBaseAndTrivialMember.elem.e27 = name_ref base, @NonTrivialBaseAndTrivialMember.%.loc7 [concrete = @NonTrivialBaseAndTrivialMember.%.loc7]
// CHECK:STDOUT:   %.loc15_3.4: ref %NonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_3.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc15_3.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc15_3.3 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%DestroyImpl.impl_witness.6d9, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   %.loc15_3.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc15_3.4 [concrete = constants.%Destroy.facet.c8f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc15_3.4)
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
// CHECK:STDOUT:   %x.patt: %pattern_type.575 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.575 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.23e: %SubobjectDestroy.type = facet_value %TrivialBase, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7c0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.23e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0fd: %Destroy.type = facet_value %TrivialBase, (%DestroyImpl.impl_witness.7c0, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_3.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a42: %SubobjectDestroy.type = facet_value %TrivialBaseAndNonTrivialMember, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4c7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a42) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3c5: %Destroy.type = facet_value %TrivialBaseAndNonTrivialMember, (%DestroyImpl.impl_witness.4c7, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc23_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %TrivialBaseAndNonTrivialMember = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15_3.1: ref %TrivialBaseAndNonTrivialMember = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %TrivialBaseAndNonTrivialMember to %.loc15_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %TrivialBaseAndNonTrivialMember.ref: type = name_ref TrivialBaseAndNonTrivialMember, file.%TrivialBaseAndNonTrivialMember.decl [concrete = constants.%TrivialBaseAndNonTrivialMember]
// CHECK:STDOUT:   %x: ref %TrivialBaseAndNonTrivialMember = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.575 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.575 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%TrivialBaseAndNonTrivialMember, (constants.%DestroyImpl.impl_witness.4c7, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.3c5]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%TrivialBaseAndNonTrivialMember, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet.3c5]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%TrivialBaseAndNonTrivialMember, (constants.%DestroyImpl.impl_witness.4c7, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.3c5]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%TrivialBaseAndNonTrivialMember, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet.3c5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.2(%self.param: ref %TrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.3(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.4(%self.param: ref %struct_type.base.x.bb9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_3.1: ref %ImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc15_3.1)
// CHECK:STDOUT:   %.loc15_3.4: ref %TrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_3.3: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc15_3.5: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc15_3.3 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.4: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc15_3.6: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc15_3.4 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc15_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_3.5(%self.param: ref %TrivialBaseAndNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %TrivialBaseAndNonTrivialMember.elem.9f5 = name_ref x, @TrivialBaseAndNonTrivialMember.%field_decl [concrete = @TrivialBaseAndNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc15_3.1: ref %ImplsDestroy = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc15_3.1)
// CHECK:STDOUT:   %base.ref: %TrivialBaseAndNonTrivialMember.elem.8b0 = name_ref base, @TrivialBaseAndNonTrivialMember.%.loc7 [concrete = @TrivialBaseAndNonTrivialMember.%.loc7]
// CHECK:STDOUT:   %.loc15_3.4: ref %TrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_3.3: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc15_3.5: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc15_3.3 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.4: %Destroy.type = facet_value constants.%TrivialBase, (constants.%DestroyImpl.impl_witness.7c0, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   %.loc15_3.6: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc15_3.4 [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc15_3.4)
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
// CHECK:STDOUT:   %x.patt: %pattern_type.157 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.157 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.501: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6ec [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c1a: %SubobjectDestroy.type = facet_value %ptr.454, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cad: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c1a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.194: %Destroy.type = facet_value %ptr.454, (%DestroyImpl.impl_witness.cad, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.af1: %Destroy.type = facet_value %NonTrivialDynamicClass, (%DestroyImpl.impl_witness.501, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialDynamicClass.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.as.DestroyImpl.impl.Op: %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NonTrivialDynamicClass: type = import_ref Main//types, NonTrivialDynamicClass, loaded [concrete = constants.%NonTrivialDynamicClass]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.133: %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc52_24, loaded [concrete = constants.%NonTrivialDynamicClass.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6ec = impl_witness_table (%Main.import_ref.133), @NonTrivialDynamicClass.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %NonTrivialDynamicClass = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_3.1: ref %NonTrivialDynamicClass = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %NonTrivialDynamicClass to %.loc10_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %NonTrivialDynamicClass.ref: type = name_ref NonTrivialDynamicClass, imports.%Main.NonTrivialDynamicClass [concrete = constants.%NonTrivialDynamicClass]
// CHECK:STDOUT:   %x: ref %NonTrivialDynamicClass = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.157 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.157 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.501, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.501, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_3.1: ref %ptr.454 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%ptr.454, (constants.%DestroyImpl.impl_witness.cad, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%ptr.454, (constants.%DestroyImpl.impl_witness.cad, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.3(%self.param: ref %NonTrivialDynamicClass) {
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
// CHECK:STDOUT:   %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_3.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c1a: %SubobjectDestroy.type = facet_value %ptr.454, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cad: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c1a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.194: %Destroy.type = facet_value %ptr.454, (%DestroyImpl.impl_witness.cad, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.501: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6ec [concrete]
// CHECK:STDOUT:   %Destroy.facet.af1: %Destroy.type = facet_value %NonTrivialDynamicClass, (%DestroyImpl.impl_witness.501, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_3.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %Derived, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.660: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.35d: %Destroy.type = facet_value %Derived, (%DestroyImpl.impl_witness.660, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialDynamicClass.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialDynamicClass.as.DestroyImpl.impl.Op: %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.133: %NonTrivialDynamicClass.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc52_24, loaded [concrete = constants.%NonTrivialDynamicClass.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6ec = impl_witness_table (%Main.import_ref.133), @NonTrivialDynamicClass.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:   %x.var: ref %Derived = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %Derived = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %x: ref %Derived = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.660, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.660, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_3.1: ref %ptr.454 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%ptr.454, (constants.%DestroyImpl.impl_witness.cad, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%ptr.454, (constants.%DestroyImpl.impl_witness.cad, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%ptr.454, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.194]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.3(%self.param: ref %NonTrivialDynamicClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.4(%self.param: ref %struct_type.base.7c1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_3.1: ref %NonTrivialDynamicClass = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.501, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.501, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.5(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %Derived.elem = name_ref base, @Derived.%.loc7 [concrete = @Derived.%.loc7]
// CHECK:STDOUT:   %.loc14_3.1: ref %NonTrivialDynamicClass = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.501, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%NonTrivialDynamicClass, (constants.%DestroyImpl.impl_witness.501, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%NonTrivialDynamicClass, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.af1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc14_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_nontrivial_class_derived_from_nontrivial_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %Derived = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc34: ref %Derived = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Derived to %.loc34 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %x: ref %Derived = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.746 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.746 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
