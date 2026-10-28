// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/array.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/non_trivial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/non_trivial.carbon

// --- core.carbon

package Core library "core";

export import library "prelude/parts/array";
export import library "prelude/parts/int";

// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a vaule's type can be destroyed.
fn CheckImplsDestroy[T: Core.Destroy](_: T) {}

// Identifies whether a type can be destroyed.
fn CheckTypeImplsDestroy(unused T: Core.Destroy) {}

// --- types.carbon

library "[[@TEST_NAME]]";

import Core library "core";

class HasTrivialDestruction {}

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

base class NonTrivialBase {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class GenericClass(T: Core.Destroy) {
  var t: T;
}

class TrivialAndNonTrivialMembers {
  var a: HasTrivialDestruction;
  var n: ImplsDestroy;
}

base class TrivialBase {}

// `array(T, N)` is the class `Core.Array(T, N)`, so this is an adapter of an
// adapter.
class AdaptNonTrivialArray {
  adapt array(ImplsDestroy, 2);
}

// --- todo_fail_class_impls_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestClassDestroyImpl(x: ImplsDestroy) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_class_has_non_trivial_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class HasNonTrivialMember {
  var n: ImplsDestroy;
}

fn TestContainsNonTrivialMember(x: HasNonTrivialMember) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

class NestedNonTrivialMember {
  var inner: HasNonTrivialMember;
}

fn TestNestedNonTrivialMember(x: NestedNonTrivialMember) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_class_derived_from_non_trivial_base.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

base class DerivedFromNonTrivialBase {
  extend base: NonTrivialBase;
}

fn TestDerivedFromNonTrivialBase(x: DerivedFromNonTrivialBase) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy

  // TODO: add CheckImplsDynamicDestroy
}

class NonTrivialGrandparent {
  extend base: DerivedFromNonTrivialBase;
}

fn TestNonTrivialGrandparent(x: NonTrivialGrandparent) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy

  // TODO: add CheckImplsDynamicDestroy
}

// --- todo_fail_class_derived_from_trivial_base_impls_destroy.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class DestroyImplAndTrivialBase {
  extend base: TrivialBase;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn TestDestroyImplAndTrivialBase(x: DestroyImplAndTrivialBase) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_generic_class_with_non_trivial_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestGenericClassWithNonTrivialMember(x: GenericClass(ImplsDestroy)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_non_trivial_and_trivial_members.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class NonTrivialMemberFirst {
  var x: ImplsDestroy;
  var y: HasTrivialDestruction;
}

fn TestNonTrivialAndTrivialMembers(x: NonTrivialMemberFirst) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

class NonTrivialMemberSecond {
  var x: HasTrivialDestruction;
  var y: ImplsDestroy;
}

fn TestTrivialAndNonTrivialMembers(x: NonTrivialMemberSecond) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_non_trivial_base_and_trivial_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class NonTrivialBaseAndTrivialMember {
  extend base: NonTrivialBase;
  var a: HasTrivialDestruction;
}

fn TestNonTrivialBaseAndTrivialMember(x: NonTrivialBaseAndTrivialMember) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_trivial_base_and_non_trivial_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class TrivialBaseAndNonTrivialMember {
  extend base: TrivialBase;
  var n: ImplsDestroy;
}

fn TestTrivialBaseAndNonTrivialMember(x: TrivialBaseAndNonTrivialMember) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_int_adapter_with_destroy_impl.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";
import library "types";

class IntAdapterWithDestroyImpl {
  adapt i32;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn TestIntAdapterWithDestroyImpl(x: IntAdapterWithDestroyImpl) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_array.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";
import library "types";

fn TestArray(x: array(ImplsDestroy, 3)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

fn TestArrayOfArray(x: array(array(ImplsDestroy, 2), 3)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_class_adapter.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class ClassAdapter {
  adapt ImplsDestroy;
}

fn TestClassAdapter(x: ClassAdapter) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_struct_adapter.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class StructAdapter {
  adapt {.n: ImplsDestroy};
}

fn TestNonTrivialStructAdapter(x: StructAdapter) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_const.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestConstNonTrivial(x: const ImplsDestroy) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_maybe_unformed.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestMaybeUnformedNonTrivial(x: Core.MaybeUnformed(ImplsDestroy)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_struct.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestStructWithNonTrivialFieldFirst(x: {.a: ImplsDestroy, .b: HasTrivialDestruction}) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy

}

fn TestStructWithNonTrivialFieldSecond(x: {.a: HasTrivialDestruction, .b: ImplsDestroy}) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

fn TestStructWithNestedNonTrivial(x: {.a: {.b: ImplsDestroy}}) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_tuple_with_non_trivial_element.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestTupleWithNonTrivialElementFirst(x: (ImplsDestroy, HasTrivialDestruction)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

fn TestTupleWithNonTrivialElementSecond(x: (HasTrivialDestruction, ImplsDestroy)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_tuple_with_nested_non_trivial_element.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestTupleWithNestedNonTrivial(x: ((ImplsDestroy,),)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy
}

// --- todo_fail_non_trivial_tuple_adapter.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class NonTrivialTupleAdapter {
  adapt (ImplsDestroy, HasTrivialDestruction);
}

fn TestNonTrivialTupleAdapter(x: NonTrivialTupleAdapter) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy

}

// --- constrained_facet.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";
import library "types";

fn TestDestroy(generic T: Core.Destroy) {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy(T);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy

  // TODO: add CheckImplsDynamicDestroy
}

fn TestStructOfDestroy(generic T: Core.Destroy) {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy({.a: T});
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy

  // TODO: add CheckImplsDynamicDestroy
}

fn TestTupleOfDestroy(generic T: Core.Destroy) {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy((T, HasTrivialDestruction));
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy

  // TODO: add CheckImplsDynamicDestroy
}

fn TestArrayOfDestroy(generic T: Core.Destroy) {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy(array(T, 2));
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy

  // TODO: add CheckImplsDynamicDestroy
}

fn TestGenericClassOfDestroy(generic T: Core.Destroy) {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy(GenericClass(T));
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy

  // TODO: add CheckImplsDynamicDestroy
}

// --- todo_fail_constrained_facet_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

// TODO: This should fail, because `T` might not have trivial destruction.
fn TestDestroy(generic T: Core.Destroy) {
  CheckImplsDestroy(T);

  // TODO: add CheckImplsTrivialDestroy

  // TODO: add CheckImplsDynamicDestroy
}

// --- todo_fail_partial_class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

base class BaseClassWithNonTrivialMember {
  var n: ImplsDestroy;
}

fn TestPartialNonTrivialMemberBase() {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy(partial BaseClassWithNonTrivialMember);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy

  // TODO: add CheckImplsDynamicDestroy
}

// --- todo_fail_partial_destroy_impl.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

// TODO: `CheckTypeImplsDestroy` should fail once `DestroyImpl.Op`
// takes `partial Self` instead of `Self`.
fn TestPartialNonTrivialBase() {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy(partial NonTrivialBase);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy

  // TODO: add CheckImplsDynamicDestroy
}

// CHECK:STDOUT: --- todo_fail_class_impls_destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Main.import_ref.d8f = import_ref Main//types, loc10_24, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.d8f), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestClassDestroyImpl(%x.param: %ImplsDestroy) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %ImplsDestroy = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_class_has_non_trivial_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialMember: type = class_type @HasNonTrivialMember [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasNonTrivialMember.elem: type = unbound_element_type %HasNonTrivialMember, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.n.81f: type = struct_type {.n: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8a0: %SubobjectDestroy.type = facet_value %HasNonTrivialMember, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.9da: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8a0) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e80: %Destroy.type = facet_value %HasNonTrivialMember, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.9da) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.e63: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.e80) [concrete]
// CHECK:STDOUT:   %NestedNonTrivialMember: type = class_type @NestedNonTrivialMember [concrete]
// CHECK:STDOUT:   %NestedNonTrivialMember.elem: type = unbound_element_type %NestedNonTrivialMember, %HasNonTrivialMember [concrete]
// CHECK:STDOUT:   %struct_type.inner.975: type = struct_type {.inner: %HasNonTrivialMember} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.446: %SubobjectDestroy.type = facet_value %NestedNonTrivialMember, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c6f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.446) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7fa: %Destroy.type = facet_value %NestedNonTrivialMember, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.c6f) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.2c9: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.7fa) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestContainsNonTrivialMember(%x.param: %HasNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %HasNonTrivialMember = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%HasNonTrivialMember, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.9da) [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%HasNonTrivialMember, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%HasNonTrivialMember, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.9da) [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%HasNonTrivialMember, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.e80) [concrete = constants.%CheckImplsDestroy.specific_fn.e63]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.3(%self.param: ref %struct_type.n.81f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_22.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.4(%self.param: ref %HasNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %HasNonTrivialMember.elem = name_ref n, @HasNonTrivialMember.%field_decl [concrete = @HasNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc13_22.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNestedNonTrivialMember(%x.param: %NestedNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %NestedNonTrivialMember = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc26_22.1: %Destroy.type = facet_value constants.%NestedNonTrivialMember, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.c6f) [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %.loc26_22.1: %Destroy.type = converted constants.%NestedNonTrivialMember, %Destroy.facet.loc26_22.1 [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %Destroy.facet.loc26_22.2: %Destroy.type = facet_value constants.%NestedNonTrivialMember, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.c6f) [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %.loc26_22.2: %Destroy.type = converted constants.%NestedNonTrivialMember, %Destroy.facet.loc26_22.2 [concrete = constants.%Destroy.facet.7fa]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.7fa) [concrete = constants.%CheckImplsDestroy.specific_fn.2c9]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_22.1(%self.param: ref %struct_type.inner.975) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc26_22.1: ref %HasNonTrivialMember = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc26_22.1: %Destroy.type = facet_value constants.%HasNonTrivialMember, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.9da) [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   %.loc26_22.2: %Destroy.type = converted constants.%HasNonTrivialMember, %Destroy.facet.loc26_22.1 [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   %Destroy.facet.loc26_22.2: %Destroy.type = facet_value constants.%HasNonTrivialMember, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.9da) [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   %.loc26_22.3: %Destroy.type = converted constants.%HasNonTrivialMember, %Destroy.facet.loc26_22.2 [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc26_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_22.2(%self.param: ref %NestedNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %inner.ref: %NestedNonTrivialMember.elem = name_ref inner, @NestedNonTrivialMember.%field_decl [concrete = @NestedNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc26_22.1: ref %HasNonTrivialMember = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc26_22.1: %Destroy.type = facet_value constants.%HasNonTrivialMember, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.9da) [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   %.loc26_22.2: %Destroy.type = converted constants.%HasNonTrivialMember, %Destroy.facet.loc26_22.1 [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   %Destroy.facet.loc26_22.2: %Destroy.type = facet_value constants.%HasNonTrivialMember, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.9da) [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   %.loc26_22.3: %Destroy.type = converted constants.%HasNonTrivialMember, %Destroy.facet.loc26_22.2 [concrete = constants.%Destroy.facet.e80]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc26_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_class_derived_from_non_trivial_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase: type = class_type @DerivedFromNonTrivialBase [concrete]
// CHECK:STDOUT:   %NonTrivialBase: type = class_type @NonTrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialBase.elem: type = unbound_element_type %DerivedFromNonTrivialBase, %NonTrivialBase [concrete]
// CHECK:STDOUT:   %struct_type.base.25c: type = struct_type {.base: %NonTrivialBase} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6e0 [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialBase.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op: %NonTrivialBase.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.9e6: %Destroy.type = facet_value %NonTrivialBase, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.6d9) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c4c: %SubobjectDestroy.type = facet_value %DerivedFromNonTrivialBase, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7a2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c4c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6f5: %Destroy.type = facet_value %DerivedFromNonTrivialBase, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.7a2) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.d20: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.6f5) [concrete]
// CHECK:STDOUT:   %NonTrivialGrandparent: type = class_type @NonTrivialGrandparent [concrete]
// CHECK:STDOUT:   %NonTrivialGrandparent.elem: type = unbound_element_type %NonTrivialGrandparent, %DerivedFromNonTrivialBase [concrete]
// CHECK:STDOUT:   %struct_type.base.3eb: type = struct_type {.base: %DerivedFromNonTrivialBase} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f13: %SubobjectDestroy.type = facet_value %NonTrivialGrandparent, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0d6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f13) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b1d: %Destroy.type = facet_value %NonTrivialGrandparent, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.0d6) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.fe3: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.b1d) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.9c2: %NonTrivialBase.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc16_24, loaded [concrete = constants.%NonTrivialBase.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6e0 = impl_witness_table (%Main.import_ref.9c2), @NonTrivialBase.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerivedFromNonTrivialBase(%x.param: %DerivedFromNonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %DerivedFromNonTrivialBase = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.6f5) [concrete = constants.%CheckImplsDestroy.specific_fn.d20]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %NonTrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.3(%self.param: ref %struct_type.base.25c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_22.1: ref %NonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.4(%self.param: ref %DerivedFromNonTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %DerivedFromNonTrivialBase.elem = name_ref base, @DerivedFromNonTrivialBase.%.loc8 [concrete = @DerivedFromNonTrivialBase.%.loc8]
// CHECK:STDOUT:   %.loc13_22.1: ref %NonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNonTrivialGrandparent(%x.param: %NonTrivialGrandparent) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %NonTrivialGrandparent = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc27_22.1: %Destroy.type = facet_value constants.%NonTrivialGrandparent, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.0d6) [concrete = constants.%Destroy.facet.b1d]
// CHECK:STDOUT:   %.loc27_22.1: %Destroy.type = converted constants.%NonTrivialGrandparent, %Destroy.facet.loc27_22.1 [concrete = constants.%Destroy.facet.b1d]
// CHECK:STDOUT:   %Destroy.facet.loc27_22.2: %Destroy.type = facet_value constants.%NonTrivialGrandparent, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.0d6) [concrete = constants.%Destroy.facet.b1d]
// CHECK:STDOUT:   %.loc27_22.2: %Destroy.type = converted constants.%NonTrivialGrandparent, %Destroy.facet.loc27_22.2 [concrete = constants.%Destroy.facet.b1d]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.b1d) [concrete = constants.%CheckImplsDestroy.specific_fn.fe3]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_22.1(%self.param: ref %struct_type.base.3eb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27_22.1: ref %DerivedFromNonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc27_22.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc27_22.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc27_22.1 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %Destroy.facet.loc27_22.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc27_22.3: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc27_22.2 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc27_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_22.2(%self.param: ref %NonTrivialGrandparent) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %NonTrivialGrandparent.elem = name_ref base, @NonTrivialGrandparent.%.loc22 [concrete = @NonTrivialGrandparent.%.loc22]
// CHECK:STDOUT:   %.loc27_22.1: ref %DerivedFromNonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc27_22.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc27_22.2: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc27_22.1 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %Destroy.facet.loc27_22.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialBase, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7a2) [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   %.loc27_22.3: %Destroy.type = converted constants.%DerivedFromNonTrivialBase, %Destroy.facet.loc27_22.2 [concrete = constants.%Destroy.facet.6f5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc27_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_class_derived_from_trivial_base_impls_destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %DestroyImplAndTrivialBase: type = class_type @DestroyImplAndTrivialBase [concrete]
// CHECK:STDOUT:   %TrivialBase: type = class_type @TrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.2a8: <witness> = impl_witness @DestroyImplAndTrivialBase.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.base.223: type = struct_type {.base: %TrivialBase} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_22.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %DestroyImplAndTrivialBase, (%DestroyImpl.impl_witness.2a8, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDestroyImplAndTrivialBase(%x.param: %DestroyImplAndTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %DestroyImplAndTrivialBase = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc17_22.1: %Destroy.type = facet_value constants.%DestroyImplAndTrivialBase, (constants.%DestroyImpl.impl_witness.2a8, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_22.1: %Destroy.type = converted constants.%DestroyImplAndTrivialBase, %Destroy.facet.loc17_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_22.2: %Destroy.type = facet_value constants.%DestroyImplAndTrivialBase, (constants.%DestroyImpl.impl_witness.2a8, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_22.2: %Destroy.type = converted constants.%DestroyImplAndTrivialBase, %Destroy.facet.loc17_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_22.2(%self.param: ref %TrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_22.3(%self.param: ref %struct_type.base.223) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_22.4(%self.param: ref %DestroyImplAndTrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_generic_class_with_non_trivial_member.carbon
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
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_69.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %GenericClass.62d: type = class_type @GenericClass, @GenericClass(%Destroy.facet.c9e) [concrete]
// CHECK:STDOUT:   %GenericClass.elem.e29: type = unbound_element_type %GenericClass.62d, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.t.1ce: type = struct_type {.t: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e88: %SubobjectDestroy.type = facet_value %GenericClass.62d, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.52a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e88) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7d0: %Destroy.type = facet_value %GenericClass.62d, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.52a) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.7d0) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Main.import_ref.f7a: @GenericClass.%GenericClass.elem (%GenericClass.elem.70a) = import_ref Main//types, loc21_8, loaded [concrete = %field_decl]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.eb2 = import_ref Main//types, loc21_10, unloaded
// CHECK:STDOUT:   %field_decl: @GenericClass.%GenericClass.elem (%GenericClass.elem.70a) = field_decl t, element0, %Main.import_ref.eb2 in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGenericClassWithNonTrivialMember(%x.param: %GenericClass.62d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %GenericClass.62d = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%GenericClass.62d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.52a) [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%GenericClass.62d, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%GenericClass.62d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.52a) [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%GenericClass.62d, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.7d0]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.7d0) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %struct_type.t.1ce) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_22.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc9_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc9_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %GenericClass.62d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %t.ref: %GenericClass.elem.e29 = name_ref t, imports.%Main.import_ref.f7a [concrete = imports.%field_decl]
// CHECK:STDOUT:   %.loc9_22.1: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc9_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc9_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_non_trivial_and_trivial_members.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialMemberFirst: type = class_type @NonTrivialMemberFirst [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonTrivialMemberFirst.elem.c66: type = unbound_element_type %NonTrivialMemberFirst, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %NonTrivialMemberFirst.elem.269: type = unbound_element_type %NonTrivialMemberFirst, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.y.8c7: type = struct_type {.x: %ImplsDestroy, .y: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_22.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a88: %SubobjectDestroy.type = facet_value %NonTrivialMemberFirst, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.332: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a88) [concrete]
// CHECK:STDOUT:   %Destroy.facet.296: %Destroy.type = facet_value %NonTrivialMemberFirst, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.332) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.9a6: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.296) [concrete]
// CHECK:STDOUT:   %NonTrivialMemberSecond: type = class_type @NonTrivialMemberSecond [concrete]
// CHECK:STDOUT:   %NonTrivialMemberSecond.elem.1a5: type = unbound_element_type %NonTrivialMemberSecond, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %NonTrivialMemberSecond.elem.e20: type = unbound_element_type %NonTrivialMemberSecond, %ImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.x.y.8df: type = struct_type {.x: %HasTrivialDestruction, .y: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc28_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1db: %SubobjectDestroy.type = facet_value %NonTrivialMemberSecond, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.745: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1db) [concrete]
// CHECK:STDOUT:   %Destroy.facet.997: %Destroy.type = facet_value %NonTrivialMemberSecond, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.745) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.63a: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.997) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNonTrivialAndTrivialMembers(%x.param: %NonTrivialMemberFirst) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %NonTrivialMemberFirst = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc14_22.1: %Destroy.type = facet_value constants.%NonTrivialMemberFirst, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.332) [concrete = constants.%Destroy.facet.296]
// CHECK:STDOUT:   %.loc14_22.1: %Destroy.type = converted constants.%NonTrivialMemberFirst, %Destroy.facet.loc14_22.1 [concrete = constants.%Destroy.facet.296]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.2: %Destroy.type = facet_value constants.%NonTrivialMemberFirst, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.332) [concrete = constants.%Destroy.facet.296]
// CHECK:STDOUT:   %.loc14_22.2: %Destroy.type = converted constants.%NonTrivialMemberFirst, %Destroy.facet.loc14_22.2 [concrete = constants.%Destroy.facet.296]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.296) [concrete = constants.%CheckImplsDestroy.specific_fn.9a6]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.4(%self.param: ref %struct_type.x.y.8c7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_22.1: ref %HasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_22.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_22.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_22.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_22.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_22.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_22.1)
// CHECK:STDOUT:   %.loc14_22.4: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_22.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_22.5: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_22.3 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_22.6: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_22.4 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.5(%self.param: ref %NonTrivialMemberFirst) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.ref: %NonTrivialMemberFirst.elem.269 = name_ref y, @NonTrivialMemberFirst.%field_decl.loc9 [concrete = @NonTrivialMemberFirst.%field_decl.loc9]
// CHECK:STDOUT:   %.loc14_22.1: ref %HasTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_22.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_22.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_22.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_22.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_22.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_22.1)
// CHECK:STDOUT:   %x.ref: %NonTrivialMemberFirst.elem.c66 = name_ref x, @NonTrivialMemberFirst.%field_decl.loc8 [concrete = @NonTrivialMemberFirst.%field_decl.loc8]
// CHECK:STDOUT:   %.loc14_22.4: ref %ImplsDestroy = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_22.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_22.5: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_22.3 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_22.6: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_22.4 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTrivialAndNonTrivialMembers(%x.param: %NonTrivialMemberSecond) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %NonTrivialMemberSecond = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc28_22.1: %Destroy.type = facet_value constants.%NonTrivialMemberSecond, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.745) [concrete = constants.%Destroy.facet.997]
// CHECK:STDOUT:   %.loc28_22.1: %Destroy.type = converted constants.%NonTrivialMemberSecond, %Destroy.facet.loc28_22.1 [concrete = constants.%Destroy.facet.997]
// CHECK:STDOUT:   %Destroy.facet.loc28_22.2: %Destroy.type = facet_value constants.%NonTrivialMemberSecond, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.745) [concrete = constants.%Destroy.facet.997]
// CHECK:STDOUT:   %.loc28_22.2: %Destroy.type = converted constants.%NonTrivialMemberSecond, %Destroy.facet.loc28_22.2 [concrete = constants.%Destroy.facet.997]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.997) [concrete = constants.%CheckImplsDestroy.specific_fn.63a]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc28_22.1(%self.param: ref %struct_type.x.y.8df) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc28_22.1: ref %ImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc28_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc28_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc28_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc28_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc28_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc28_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc28_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc28_22.1)
// CHECK:STDOUT:   %.loc28_22.4: ref %HasTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc28_22.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc28_22.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc28_22.3 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc28_22.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc28_22.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc28_22.4 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc28_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc28_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc28_22.2(%self.param: ref %NonTrivialMemberSecond) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.ref: %NonTrivialMemberSecond.elem.e20 = name_ref y, @NonTrivialMemberSecond.%field_decl.loc23 [concrete = @NonTrivialMemberSecond.%field_decl.loc23]
// CHECK:STDOUT:   %.loc28_22.1: ref %ImplsDestroy = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc28_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc28_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc28_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc28_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc28_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc28_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc28_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc28_22.1)
// CHECK:STDOUT:   %x.ref: %NonTrivialMemberSecond.elem.1a5 = name_ref x, @NonTrivialMemberSecond.%field_decl.loc22 [concrete = @NonTrivialMemberSecond.%field_decl.loc22]
// CHECK:STDOUT:   %.loc28_22.4: ref %HasTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc28_22.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc28_22.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc28_22.3 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc28_22.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc28_22.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc28_22.4 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc28_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc28_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_non_trivial_base_and_trivial_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember: type = class_type @NonTrivialBaseAndTrivialMember [concrete]
// CHECK:STDOUT:   %NonTrivialBase: type = class_type @NonTrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember.elem.e27: type = unbound_element_type %NonTrivialBaseAndTrivialMember, %NonTrivialBase [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %NonTrivialBaseAndTrivialMember.elem.354: type = unbound_element_type %NonTrivialBaseAndTrivialMember, %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.base.a.0e0: type = struct_type {.base: %NonTrivialBase, .a: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.6e0 [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op.type: type = fn_type @NonTrivialBase.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %NonTrivialBase.as.DestroyImpl.impl.Op: %NonTrivialBase.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9e6: %Destroy.type = facet_value %NonTrivialBase, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.6d9) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_22.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.78d: %SubobjectDestroy.type = facet_value %NonTrivialBaseAndTrivialMember, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.75e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.78d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3cb: %Destroy.type = facet_value %NonTrivialBaseAndTrivialMember, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.75e) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.3cb) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.9c2: %NonTrivialBase.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc16_24, loaded [concrete = constants.%NonTrivialBase.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.6e0 = impl_witness_table (%Main.import_ref.9c2), @NonTrivialBase.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNonTrivialBaseAndTrivialMember(%x.param: %NonTrivialBaseAndTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %NonTrivialBaseAndTrivialMember = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc14_22.1: %Destroy.type = facet_value constants.%NonTrivialBaseAndTrivialMember, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.75e) [concrete = constants.%Destroy.facet.3cb]
// CHECK:STDOUT:   %.loc14_22.1: %Destroy.type = converted constants.%NonTrivialBaseAndTrivialMember, %Destroy.facet.loc14_22.1 [concrete = constants.%Destroy.facet.3cb]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.2: %Destroy.type = facet_value constants.%NonTrivialBaseAndTrivialMember, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.75e) [concrete = constants.%Destroy.facet.3cb]
// CHECK:STDOUT:   %.loc14_22.2: %Destroy.type = converted constants.%NonTrivialBaseAndTrivialMember, %Destroy.facet.loc14_22.2 [concrete = constants.%Destroy.facet.3cb]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.3cb) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.2(%self.param: ref %NonTrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.4(%self.param: ref %struct_type.base.a.0e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_22.1: ref %HasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_22.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_22.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_22.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_22.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_22.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_22.1)
// CHECK:STDOUT:   %.loc14_22.4: ref %NonTrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_22.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc14_22.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_22.3 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc14_22.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_22.4 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.5(%self.param: ref %NonTrivialBaseAndTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %NonTrivialBaseAndTrivialMember.elem.354 = name_ref a, @NonTrivialBaseAndTrivialMember.%field_decl [concrete = @NonTrivialBaseAndTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc14_22.1: ref %HasTrivialDestruction = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_22.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_22.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_22.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc14_22.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc14_22.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_22.1)
// CHECK:STDOUT:   %base.ref: %NonTrivialBaseAndTrivialMember.elem.e27 = name_ref base, @NonTrivialBaseAndTrivialMember.%.loc8 [concrete = @NonTrivialBaseAndTrivialMember.%.loc8]
// CHECK:STDOUT:   %.loc14_22.4: ref %NonTrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_22.3: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc14_22.5: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_22.3 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.4: %Destroy.type = facet_value constants.%NonTrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d9) [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   %.loc14_22.6: %Destroy.type = converted constants.%NonTrivialBase, %Destroy.facet.loc14_22.4 [concrete = constants.%Destroy.facet.9e6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_trivial_base_and_non_trivial_member.carbon
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
// CHECK:STDOUT:   %struct_type.base.n.b10: type = struct_type {.base: %TrivialBase, .n: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.23e: %SubobjectDestroy.type = facet_value %TrivialBase, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7c0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.23e) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %Destroy.facet.033: %Destroy.type = facet_value %TrivialBase, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.7c0) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_22.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a42: %SubobjectDestroy.type = facet_value %TrivialBaseAndNonTrivialMember, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4c7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a42) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6ca: %Destroy.type = facet_value %TrivialBaseAndNonTrivialMember, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.4c7) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.6ca) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTrivialBaseAndNonTrivialMember(%x.param: %TrivialBaseAndNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %TrivialBaseAndNonTrivialMember = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc14_22.1: %Destroy.type = facet_value constants.%TrivialBaseAndNonTrivialMember, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.4c7) [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %.loc14_22.1: %Destroy.type = converted constants.%TrivialBaseAndNonTrivialMember, %Destroy.facet.loc14_22.1 [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.2: %Destroy.type = facet_value constants.%TrivialBaseAndNonTrivialMember, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.4c7) [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %.loc14_22.2: %Destroy.type = converted constants.%TrivialBaseAndNonTrivialMember, %Destroy.facet.loc14_22.2 [concrete = constants.%Destroy.facet.6ca]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.6ca) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.2(%self.param: ref %TrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.3(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.4(%self.param: ref %struct_type.base.n.b10) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_22.1: ref %ImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_22.1)
// CHECK:STDOUT:   %.loc14_22.4: ref %TrivialBase = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_22.3: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc14_22.5: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc14_22.3 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.4: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc14_22.6: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc14_22.4 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.5(%self.param: ref %TrivialBaseAndNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %TrivialBaseAndNonTrivialMember.elem.9f5 = name_ref n, @TrivialBaseAndNonTrivialMember.%field_decl [concrete = @TrivialBaseAndNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc14_22.1: ref %ImplsDestroy = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc14_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_22.1)
// CHECK:STDOUT:   %base.ref: %TrivialBaseAndNonTrivialMember.elem.8b0 = name_ref base, @TrivialBaseAndNonTrivialMember.%.loc8 [concrete = @TrivialBaseAndNonTrivialMember.%.loc8]
// CHECK:STDOUT:   %.loc14_22.4: ref %TrivialBase = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_22.3: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc14_22.5: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc14_22.3 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.4: %Destroy.type = facet_value constants.%TrivialBase, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7c0) [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   %.loc14_22.6: %Destroy.type = converted constants.%TrivialBase, %Destroy.facet.loc14_22.4 [concrete = constants.%Destroy.facet.033]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_int_adapter_with_destroy_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %IntAdapterWithDestroyImpl: type = class_type @IntAdapterWithDestroyImpl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.03b: <witness> = impl_witness @IntAdapterWithDestroyImpl.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %IntAdapterWithDestroyImpl, (%DestroyImpl.impl_witness.03b, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIntAdapterWithDestroyImpl(%x.param: %IntAdapterWithDestroyImpl) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %IntAdapterWithDestroyImpl = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc18_22.1: %Destroy.type = facet_value constants.%IntAdapterWithDestroyImpl, (constants.%DestroyImpl.impl_witness.03b, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc18_22.1: %Destroy.type = converted constants.%IntAdapterWithDestroyImpl, %Destroy.facet.loc18_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc18_22.2: %Destroy.type = facet_value constants.%IntAdapterWithDestroyImpl, (constants.%DestroyImpl.impl_witness.03b, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc18_22.2: %Destroy.type = converted constants.%IntAdapterWithDestroyImpl, %Destroy.facet.loc18_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_22.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_22.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_22.3(%self.param: ref %IntAdapterWithDestroyImpl) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.875: type = class_type @Array, @Array(%ImplsDestroy, %int_3) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.134: %SubobjectDestroy.type = facet_value %Array.875, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.70b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.134) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ab9: %Destroy.type = facet_value %Array.875, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.70b) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.a7c: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.ab9) [concrete]
// CHECK:STDOUT:   %Array.c9e: type = class_type @Array, @Array(%ImplsDestroy, %int_2) [concrete]
// CHECK:STDOUT:   %Array.419: type = class_type @Array, @Array(%Array.c9e, %int_3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_22.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7bc: %SubobjectDestroy.type = facet_value %Array.c9e, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.907: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f6b: %Destroy.type = facet_value %Array.c9e, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.907) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.46f: %SubobjectDestroy.type = facet_value %Array.419, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.8b0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.46f) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9ff: %Destroy.type = facet_value %Array.419, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.8b0) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.401: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.9ff) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestArray(%x.param: %Array.875) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Array.875 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc10_22.1: %Destroy.type = facet_value constants.%Array.875, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.70b) [concrete = constants.%Destroy.facet.ab9]
// CHECK:STDOUT:   %.loc10_22.1: %Destroy.type = converted constants.%Array.875, %Destroy.facet.loc10_22.1 [concrete = constants.%Destroy.facet.ab9]
// CHECK:STDOUT:   %Destroy.facet.loc10_22.2: %Destroy.type = facet_value constants.%Array.875, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.70b) [concrete = constants.%Destroy.facet.ab9]
// CHECK:STDOUT:   %.loc10_22.2: %Destroy.type = converted constants.%Array.875, %Destroy.facet.loc10_22.2 [concrete = constants.%Destroy.facet.ab9]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.ab9) [concrete = constants.%CheckImplsDestroy.specific_fn.a7c]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_22.3(%self.param: ref %Array.875) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc10_22.1: ref %ImplsDestroy = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.facet.loc10_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_22.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc10_22.4: ref %ImplsDestroy = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc10_22.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_22.5: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_22.3 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_22.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_22.6: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_22.4 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_22.4)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_22.7: ref %ImplsDestroy = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc10_22.5: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_22.8: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_22.5 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_22.6: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc10_22.9: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc10_22.6 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_22.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_22.7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestArrayOfArray(%x.param: %Array.419) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Array.419 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc19_22.1: %Destroy.type = facet_value constants.%Array.419, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.8b0) [concrete = constants.%Destroy.facet.9ff]
// CHECK:STDOUT:   %.loc19_22.1: %Destroy.type = converted constants.%Array.419, %Destroy.facet.loc19_22.1 [concrete = constants.%Destroy.facet.9ff]
// CHECK:STDOUT:   %Destroy.facet.loc19_22.2: %Destroy.type = facet_value constants.%Array.419, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.8b0) [concrete = constants.%Destroy.facet.9ff]
// CHECK:STDOUT:   %.loc19_22.2: %Destroy.type = converted constants.%Array.419, %Destroy.facet.loc19_22.2 [concrete = constants.%Destroy.facet.9ff]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.9ff) [concrete = constants.%CheckImplsDestroy.specific_fn.401]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_22.1(%self.param: ref %Array.c9e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc19_22.1: ref %ImplsDestroy = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc19_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc19_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc19_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc19_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc19_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc19_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc19_22.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc19_22.4: ref %ImplsDestroy = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc19_22.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc19_22.5: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc19_22.3 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc19_22.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc19_22.6: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc19_22.4 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc19_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_22.2(%self.param: ref %Array.419) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc19_22.1: ref %Array.c9e = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.facet.loc19_22.1: %Destroy.type = facet_value constants.%Array.c9e, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.907) [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   %.loc19_22.2: %Destroy.type = converted constants.%Array.c9e, %Destroy.facet.loc19_22.1 [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   %Destroy.facet.loc19_22.2: %Destroy.type = facet_value constants.%Array.c9e, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.907) [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   %.loc19_22.3: %Destroy.type = converted constants.%Array.c9e, %Destroy.facet.loc19_22.2 [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc19_22.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc19_22.4: ref %Array.c9e = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc19_22.3: %Destroy.type = facet_value constants.%Array.c9e, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.907) [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   %.loc19_22.5: %Destroy.type = converted constants.%Array.c9e, %Destroy.facet.loc19_22.3 [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   %Destroy.facet.loc19_22.4: %Destroy.type = facet_value constants.%Array.c9e, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.907) [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   %.loc19_22.6: %Destroy.type = converted constants.%Array.c9e, %Destroy.facet.loc19_22.4 [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc19_22.4)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc19_22.7: ref %Array.c9e = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc19_22.5: %Destroy.type = facet_value constants.%Array.c9e, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.907) [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   %.loc19_22.8: %Destroy.type = converted constants.%Array.c9e, %Destroy.facet.loc19_22.5 [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   %Destroy.facet.loc19_22.6: %Destroy.type = facet_value constants.%Array.c9e, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.907) [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   %.loc19_22.9: %Destroy.type = converted constants.%Array.c9e, %Destroy.facet.loc19_22.6 [concrete = constants.%Destroy.facet.f6b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_22.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc19_22.7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_class_adapter.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ClassAdapter: type = class_type @ClassAdapter [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.88c: %SubobjectDestroy.type = facet_value %ClassAdapter, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.423: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.88c) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ClassAdapter, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.423) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestClassAdapter(%x.param: %ClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %ClassAdapter = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%ClassAdapter, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.423) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%ClassAdapter, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%ClassAdapter, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.423) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%ClassAdapter, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.3(%self.param: ref %ClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_struct_adapter.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %StructAdapter: type = class_type @StructAdapter [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.n.81f: type = struct_type {.n: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d2b: %SubobjectDestroy.type = facet_value %StructAdapter, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.91a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d2b) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ca9: %Destroy.type = facet_value %StructAdapter, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.91a) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.ca9) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNonTrivialStructAdapter(%x.param: %StructAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %StructAdapter = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%StructAdapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.91a) [concrete = constants.%Destroy.facet.ca9]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%StructAdapter, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.ca9]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%StructAdapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.91a) [concrete = constants.%Destroy.facet.ca9]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%StructAdapter, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.ca9]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.ca9) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.3(%self.param: ref %struct_type.n.81f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_22.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.4(%self.param: ref %StructAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %const: type = const_type %ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c28: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.fa4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c28) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %const, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.fa4) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestConstNonTrivial(%x.param: %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %const = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.fa4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%const, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.fa4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.3(%self.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_maybe_unformed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %MaybeUnformed.439: type = class_type @MaybeUnformed, @MaybeUnformed(%ImplsDestroy) [concrete]
// CHECK:STDOUT:   %.98a: type = maybe_unformed_type %ImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.fa1: %SubobjectDestroy.type = facet_value %MaybeUnformed.439, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b7e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.fa1) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %MaybeUnformed.439, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.b7e) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestMaybeUnformedNonTrivial(%x.param: %MaybeUnformed.439) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %MaybeUnformed.439 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%MaybeUnformed.439, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.b7e) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%MaybeUnformed.439, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%MaybeUnformed.439, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.b7e) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%MaybeUnformed.439, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.3(%self.param: ref %.98a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.4(%self.param: ref %MaybeUnformed.439) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.a.b.02c: type = struct_type {.a: %ImplsDestroy, .b: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d56: %SubobjectDestroy.type = facet_value %struct_type.a.b.02c, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.65e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d56) [concrete]
// CHECK:STDOUT:   %Destroy.facet.38c: %Destroy.type = facet_value %struct_type.a.b.02c, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.65e) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.bb5: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.38c) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.bb2: type = struct_type {.a: %HasTrivialDestruction, .b: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.bae: %SubobjectDestroy.type = facet_value %struct_type.a.b.bb2, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.de5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.bae) [concrete]
// CHECK:STDOUT:   %Destroy.facet.afa: %Destroy.type = facet_value %struct_type.a.b.bb2, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.de5) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.3d0: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.afa) [concrete]
// CHECK:STDOUT:   %struct_type.b.0cd: type = struct_type {.b: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %struct_type.a.13f: type = struct_type {.a: %struct_type.b.0cd} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc28_22.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc28_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.135: %SubobjectDestroy.type = facet_value %struct_type.b.0cd, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.436: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.135) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ba8: %Destroy.type = facet_value %struct_type.b.0cd, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.436) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.985: %SubobjectDestroy.type = facet_value %struct_type.a.13f, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.973: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.985) [concrete]
// CHECK:STDOUT:   %Destroy.facet.49c: %Destroy.type = facet_value %struct_type.a.13f, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.973) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.394: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.49c) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestStructWithNonTrivialFieldFirst(%x.param: %struct_type.a.b.02c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %struct_type.a.b.02c = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%struct_type.a.b.02c, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.65e) [concrete = constants.%Destroy.facet.38c]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%struct_type.a.b.02c, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.38c]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%struct_type.a.b.02c, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.65e) [concrete = constants.%Destroy.facet.38c]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%struct_type.a.b.02c, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.38c]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.38c) [concrete = constants.%CheckImplsDestroy.specific_fn.bb5]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.4(%self.param: ref %struct_type.a.b.02c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_22.1: ref %HasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc9_22.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc9_22.1)
// CHECK:STDOUT:   %.loc9_22.4: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_22.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc9_22.5: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.3 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc9_22.6: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.4 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc9_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestStructWithNonTrivialFieldSecond(%x.param: %struct_type.a.b.bb2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %struct_type.a.b.bb2 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc19_22.1: %Destroy.type = facet_value constants.%struct_type.a.b.bb2, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.de5) [concrete = constants.%Destroy.facet.afa]
// CHECK:STDOUT:   %.loc19_22.1: %Destroy.type = converted constants.%struct_type.a.b.bb2, %Destroy.facet.loc19_22.1 [concrete = constants.%Destroy.facet.afa]
// CHECK:STDOUT:   %Destroy.facet.loc19_22.2: %Destroy.type = facet_value constants.%struct_type.a.b.bb2, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.de5) [concrete = constants.%Destroy.facet.afa]
// CHECK:STDOUT:   %.loc19_22.2: %Destroy.type = converted constants.%struct_type.a.b.bb2, %Destroy.facet.loc19_22.2 [concrete = constants.%Destroy.facet.afa]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.afa) [concrete = constants.%CheckImplsDestroy.specific_fn.3d0]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19(%self.param: ref %struct_type.a.b.bb2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_22.1: ref %ImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc19_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc19_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc19_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc19_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc19_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc19_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc19_22.1)
// CHECK:STDOUT:   %.loc19_22.4: ref %HasTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_22.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc19_22.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc19_22.3 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc19_22.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc19_22.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc19_22.4 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc19_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestStructWithNestedNonTrivial(%x.param: %struct_type.a.13f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %struct_type.a.13f = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc28_22.1: %Destroy.type = facet_value constants.%struct_type.a.13f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.973) [concrete = constants.%Destroy.facet.49c]
// CHECK:STDOUT:   %.loc28_22.1: %Destroy.type = converted constants.%struct_type.a.13f, %Destroy.facet.loc28_22.1 [concrete = constants.%Destroy.facet.49c]
// CHECK:STDOUT:   %Destroy.facet.loc28_22.2: %Destroy.type = facet_value constants.%struct_type.a.13f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.973) [concrete = constants.%Destroy.facet.49c]
// CHECK:STDOUT:   %.loc28_22.2: %Destroy.type = converted constants.%struct_type.a.13f, %Destroy.facet.loc28_22.2 [concrete = constants.%Destroy.facet.49c]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.49c) [concrete = constants.%CheckImplsDestroy.specific_fn.394]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc28_22.1(%self.param: ref %struct_type.b.0cd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc28_22.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc28_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc28_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc28_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc28_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc28_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc28_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc28_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc28_22.2(%self.param: ref %struct_type.a.13f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc28_22.1: ref %struct_type.b.0cd = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc28_22.1: %Destroy.type = facet_value constants.%struct_type.b.0cd, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.436) [concrete = constants.%Destroy.facet.ba8]
// CHECK:STDOUT:   %.loc28_22.2: %Destroy.type = converted constants.%struct_type.b.0cd, %Destroy.facet.loc28_22.1 [concrete = constants.%Destroy.facet.ba8]
// CHECK:STDOUT:   %Destroy.facet.loc28_22.2: %Destroy.type = facet_value constants.%struct_type.b.0cd, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.436) [concrete = constants.%Destroy.facet.ba8]
// CHECK:STDOUT:   %.loc28_22.3: %Destroy.type = converted constants.%struct_type.b.0cd, %Destroy.facet.loc28_22.2 [concrete = constants.%Destroy.facet.ba8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc28_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_tuple_with_non_trivial_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %tuple.type.cac: type = tuple_type (%ImplsDestroy, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d74: %SubobjectDestroy.type = facet_value %tuple.type.cac, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ab0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d74) [concrete]
// CHECK:STDOUT:   %Destroy.facet.fdf: %Destroy.type = facet_value %tuple.type.cac, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.ab0) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.d83: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.fdf) [concrete]
// CHECK:STDOUT:   %tuple.type.871: type = tuple_type (%HasTrivialDestruction, %ImplsDestroy) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b98: %SubobjectDestroy.type = facet_value %tuple.type.871, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.de7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b98) [concrete]
// CHECK:STDOUT:   %Destroy.facet.aa2: %Destroy.type = facet_value %tuple.type.871, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.de7) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.87a: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.aa2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTupleWithNonTrivialElementFirst(%x.param: %tuple.type.cac) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %tuple.type.cac = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%tuple.type.cac, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ab0) [concrete = constants.%Destroy.facet.fdf]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%tuple.type.cac, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.fdf]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%tuple.type.cac, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ab0) [concrete = constants.%Destroy.facet.fdf]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%tuple.type.cac, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.fdf]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.fdf) [concrete = constants.%CheckImplsDestroy.specific_fn.d83]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.4(%self.param: ref %tuple.type.cac) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %HasTrivialDestruction = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %ImplsDestroy = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_22.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc9_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.3 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc9_22.4: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.4 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTupleWithNonTrivialElementSecond(%x.param: %tuple.type.871) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %tuple.type.871 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc18_22.1: %Destroy.type = facet_value constants.%tuple.type.871, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.de7) [concrete = constants.%Destroy.facet.aa2]
// CHECK:STDOUT:   %.loc18_22.1: %Destroy.type = converted constants.%tuple.type.871, %Destroy.facet.loc18_22.1 [concrete = constants.%Destroy.facet.aa2]
// CHECK:STDOUT:   %Destroy.facet.loc18_22.2: %Destroy.type = facet_value constants.%tuple.type.871, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.de7) [concrete = constants.%Destroy.facet.aa2]
// CHECK:STDOUT:   %.loc18_22.2: %Destroy.type = converted constants.%tuple.type.871, %Destroy.facet.loc18_22.2 [concrete = constants.%Destroy.facet.aa2]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.aa2) [concrete = constants.%CheckImplsDestroy.specific_fn.87a]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18(%self.param: ref %tuple.type.871) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %ImplsDestroy = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc18_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc18_22.1: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc18_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc18_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc18_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc18_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc18_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %HasTrivialDestruction = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_22.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc18_22.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc18_22.3 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc18_22.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc18_22.4: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc18_22.4 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc18_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_tuple_with_nested_non_trivial_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.159: type = tuple_type (%ImplsDestroy) [concrete]
// CHECK:STDOUT:   %tuple.type.efa: type = tuple_type (%tuple.type.159) [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5c3: %SubobjectDestroy.type = facet_value %tuple.type.159, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.aa8: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5c3) [concrete]
// CHECK:STDOUT:   %Destroy.facet.93a: %Destroy.type = facet_value %tuple.type.159, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.aa8) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.778: %SubobjectDestroy.type = facet_value %tuple.type.efa, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a9f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.778) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c23: %Destroy.type = facet_value %tuple.type.efa, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.a9f) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.c23) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTupleWithNestedNonTrivial(%x.param: %tuple.type.efa) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %tuple.type.efa = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%tuple.type.efa, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.a9f) [concrete = constants.%Destroy.facet.c23]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%tuple.type.efa, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.c23]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%tuple.type.efa, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.a9f) [concrete = constants.%Destroy.facet.c23]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%tuple.type.efa, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.c23]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.c23) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.3(%self.param: ref %tuple.type.159) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %ImplsDestroy = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.4(%self.param: ref %tuple.type.efa) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.159 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%tuple.type.159, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.aa8) [concrete = constants.%Destroy.facet.93a]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%tuple.type.159, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.93a]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%tuple.type.159, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.aa8) [concrete = constants.%Destroy.facet.93a]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%tuple.type.159, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.93a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_non_trivial_tuple_adapter.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialTupleAdapter: type = class_type @NonTrivialTupleAdapter [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %tuple.type.cac: type = tuple_type (%ImplsDestroy, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.4c3: %SubobjectDestroy.type = facet_value %NonTrivialTupleAdapter, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f89: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.4c3) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ee2: %Destroy.type = facet_value %NonTrivialTupleAdapter, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.f89) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.ee2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNonTrivialTupleAdapter(%x.param: %NonTrivialTupleAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %NonTrivialTupleAdapter = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%NonTrivialTupleAdapter, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.f89) [concrete = constants.%Destroy.facet.ee2]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%NonTrivialTupleAdapter, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.ee2]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%NonTrivialTupleAdapter, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.f89) [concrete = constants.%Destroy.facet.ee2]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%NonTrivialTupleAdapter, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.ee2]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.ee2) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.4(%self.param: ref %tuple.type.cac) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %HasTrivialDestruction = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %ImplsDestroy = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_22.3 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc13_22.4: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_22.4 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.5(%self.param: ref %NonTrivialTupleAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- constrained_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.type: type = fn_type @CheckTypeImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.190: type = facet_access_type %T.f45 [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T.as_type.190} [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.070: <witness> = lookup_impl_witness %struct_type.a, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.049: <witness> = lookup_impl_witness %struct_type.a, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.4f8: %SubobjectDestroy.type = facet_value %struct_type.a, (%SubobjectDestroy.lookup_impl_witness.049) [symbolic]
// CHECK:STDOUT:   %.1cf: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.4f8) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.632: %Destroy.type = facet_value %struct_type.a, (%DestroyImpl.lookup_impl_witness.070, %SubobjectDestroy.lookup_impl_witness.049) [symbolic]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (%Destroy.type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.693 = tuple_value (%T.f45, %HasTrivialDestruction) [symbolic]
// CHECK:STDOUT:   %tuple.type.f53: type = tuple_type (%T.as_type.190, %HasTrivialDestruction) [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.832: <witness> = lookup_impl_witness %tuple.type.f53, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.787: <witness> = lookup_impl_witness %tuple.type.f53, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.2f4: %SubobjectDestroy.type = facet_value %tuple.type.f53, (%SubobjectDestroy.lookup_impl_witness.787) [symbolic]
// CHECK:STDOUT:   %.338: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.2f4) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e3d: %Destroy.type = facet_value %tuple.type.f53, (%DestroyImpl.lookup_impl_witness.832, %SubobjectDestroy.lookup_impl_witness.787) [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Array.549: type = class_type @Array, @Array(%T.as_type.190, %int_2) [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.c96: <witness> = lookup_impl_witness %Array.549, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.e49: <witness> = lookup_impl_witness %Array.549, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.2ae: %SubobjectDestroy.type = facet_value %Array.549, (%SubobjectDestroy.lookup_impl_witness.e49) [symbolic]
// CHECK:STDOUT:   %.b46: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.2ae) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.71f: %Destroy.type = facet_value %Array.549, (%DestroyImpl.lookup_impl_witness.c96, %SubobjectDestroy.lookup_impl_witness.e49) [symbolic]
// CHECK:STDOUT:   %GenericClass.type: type = parameterized_class_name_type @GenericClass [concrete]
// CHECK:STDOUT:   %GenericClass.generic: %GenericClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericClass: type = class_type @GenericClass, @GenericClass(%T.f45) [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.85c: <witness> = lookup_impl_witness %GenericClass, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.8a5: <witness> = lookup_impl_witness %GenericClass, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.dd6: %SubobjectDestroy.type = facet_value %GenericClass, (%SubobjectDestroy.lookup_impl_witness.8a5) [symbolic]
// CHECK:STDOUT:   %.8d9: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.dd6) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.15c: %Destroy.type = facet_value %GenericClass, (%DestroyImpl.lookup_impl_witness.85c, %SubobjectDestroy.lookup_impl_witness.8a5) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = import_ref Main//check_destroy_format, CheckTypeImplsDestroy, loaded [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %Main.HasTrivialDestruction: type = import_ref Main//types, HasTrivialDestruction, loaded [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:   %Main.GenericClass: %GenericClass.type = import_ref Main//types, GenericClass, loaded [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestDestroy(%T.loc8_25.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:     %T.ref: %Destroy.type = name_ref T, %T.loc8_25.2 [symbolic = %T.loc8_25.1 (constants.%T.f45)]
// CHECK:STDOUT:     %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%T.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestStructOfDestroy(%T.loc18_33.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc20_30.2: type = facet_access_type %T.loc18_33.1 [symbolic = %T.as_type.loc20_30.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:   %struct_type.a.loc20_31.2: type = struct_type {.a: @TestStructOfDestroy.%T.as_type.loc20_30.2 (%T.as_type.190)} [symbolic = %struct_type.a.loc20_31.2 (constants.%struct_type.a)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %struct_type.a.loc20_31.2, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.049)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %struct_type.a.loc20_31.2, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.4f8)]
// CHECK:STDOUT:   %.loc20_31.2: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc20_31.2 (constants.%.1cf)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %struct_type.a.loc20_31.2, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.070)]
// CHECK:STDOUT:   %Destroy.facet.loc20_31.2: %Destroy.type = facet_value %struct_type.a.loc20_31.2, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc20_31.2 (constants.%Destroy.facet.632)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:     %T.ref: %Destroy.type = name_ref T, %T.loc18_33.2 [symbolic = %T.loc18_33.1 (constants.%T.f45)]
// CHECK:STDOUT:     %T.as_type.loc20_30.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc20_30.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %.loc20_30: type = converted %T.ref, %T.as_type.loc20_30.1 [symbolic = %T.as_type.loc20_30.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %struct_type.a.loc20_31.1: type = struct_type {.a: @TestStructOfDestroy.%T.as_type.loc20_30.2 (%T.as_type.190)} [symbolic = %struct_type.a.loc20_31.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     %Destroy.facet.loc20_31.1: %Destroy.type = facet_value %struct_type.a.loc20_31.1, (constants.%DestroyImpl.lookup_impl_witness.070, constants.%SubobjectDestroy.lookup_impl_witness.049) [symbolic = %Destroy.facet.loc20_31.2 (constants.%Destroy.facet.632)]
// CHECK:STDOUT:     %.loc20_31.1: %Destroy.type = converted %struct_type.a.loc20_31.1, %Destroy.facet.loc20_31.1 [symbolic = %Destroy.facet.loc20_31.2 (constants.%Destroy.facet.632)]
// CHECK:STDOUT:     %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%.loc20_31.1)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestTupleOfDestroy(%T.loc28_32.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple: %tuple.type.693 = tuple_value (%T.loc28_32.1, constants.%HasTrivialDestruction) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:   %T.as_type.loc30_50.2: type = facet_access_type %T.loc28_32.1 [symbolic = %T.as_type.loc30_50.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.as_type.loc30_50.2, constants.%HasTrivialDestruction) [symbolic = %tuple.type (constants.%tuple.type.f53)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.787)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %tuple.type, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.2f4)]
// CHECK:STDOUT:   %.loc30_50.4: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc30_50.4 (constants.%.338)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.832)]
// CHECK:STDOUT:   %Destroy.facet.loc30_50.2: %Destroy.type = facet_value %tuple.type, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc30_50.2 (constants.%Destroy.facet.e3d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:     %T.ref: %Destroy.type = name_ref T, %T.loc28_32.2 [symbolic = %T.loc28_32.1 (constants.%T.f45)]
// CHECK:STDOUT:     %HasTrivialDestruction.ref: type = name_ref HasTrivialDestruction, imports.%Main.HasTrivialDestruction [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:     %.loc30_50.1: %tuple.type.693 = tuple_literal (%T.ref, %HasTrivialDestruction.ref) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:     %T.as_type.loc30_50.1: type = facet_access_type constants.%T.f45 [symbolic = %T.as_type.loc30_50.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %.loc30_50.2: type = converted constants.%T.f45, %T.as_type.loc30_50.1 [symbolic = %T.as_type.loc30_50.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %Destroy.facet.loc30_50.1: %Destroy.type = facet_value constants.%tuple.type.f53, (constants.%DestroyImpl.lookup_impl_witness.832, constants.%SubobjectDestroy.lookup_impl_witness.787) [symbolic = %Destroy.facet.loc30_50.2 (constants.%Destroy.facet.e3d)]
// CHECK:STDOUT:     %.loc30_50.3: %Destroy.type = converted %.loc30_50.1, %Destroy.facet.loc30_50.1 [symbolic = %Destroy.facet.loc30_50.2 (constants.%Destroy.facet.e3d)]
// CHECK:STDOUT:     %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%.loc30_50.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc30_50.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc30_50.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestArrayOfDestroy(%T.loc38_32.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc40_31.2: type = facet_access_type %T.loc38_32.1 [symbolic = %T.as_type.loc40_31.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:   %Array.loc40_35.2: type = class_type @Array, @Array(%T.as_type.loc40_31.2, constants.%int_2) [symbolic = %Array.loc40_35.2 (constants.%Array.549)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %Array.loc40_35.2, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.e49)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.loc40_35.2, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.2ae)]
// CHECK:STDOUT:   %.loc40_35.2: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc40_35.2 (constants.%.b46)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %Array.loc40_35.2, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.c96)]
// CHECK:STDOUT:   %Destroy.facet.loc40_35.2: %Destroy.type = facet_value %Array.loc40_35.2, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc40_35.2 (constants.%Destroy.facet.71f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:     %T.ref: %Destroy.type = name_ref T, %T.loc38_32.2 [symbolic = %T.loc38_32.1 (constants.%T.f45)]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %T.as_type.loc40_31.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc40_31.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %.loc40_31: type = converted %T.ref, %T.as_type.loc40_31.1 [symbolic = %T.as_type.loc40_31.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %Array.loc40_35.1: type = class_type @Array, @Array(constants.%T.as_type.190, constants.%int_2) [symbolic = %Array.loc40_35.2 (constants.%Array.549)]
// CHECK:STDOUT:     %Destroy.facet.loc40_35.1: %Destroy.type = facet_value %Array.loc40_35.1, (constants.%DestroyImpl.lookup_impl_witness.c96, constants.%SubobjectDestroy.lookup_impl_witness.e49) [symbolic = %Destroy.facet.loc40_35.2 (constants.%Destroy.facet.71f)]
// CHECK:STDOUT:     %.loc40_35.1: %Destroy.type = converted %Array.loc40_35.1, %Destroy.facet.loc40_35.1 [symbolic = %Destroy.facet.loc40_35.2 (constants.%Destroy.facet.71f)]
// CHECK:STDOUT:     %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%.loc40_35.1)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericClassOfDestroy(%T.loc48_39.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericClass.loc50_39.2: type = class_type @GenericClass, @GenericClass(%T.loc48_39.1) [symbolic = %GenericClass.loc50_39.2 (constants.%GenericClass)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %GenericClass.loc50_39.2, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.8a5)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %GenericClass.loc50_39.2, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.dd6)]
// CHECK:STDOUT:   %.loc50_39.2: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc50_39.2 (constants.%.8d9)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %GenericClass.loc50_39.2, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.85c)]
// CHECK:STDOUT:   %Destroy.facet.loc50_39.2: %Destroy.type = facet_value %GenericClass.loc50_39.2, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc50_39.2 (constants.%Destroy.facet.15c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:     %GenericClass.ref: %GenericClass.type = name_ref GenericClass, imports.%Main.GenericClass [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:     %T.ref: %Destroy.type = name_ref T, %T.loc48_39.2 [symbolic = %T.loc48_39.1 (constants.%T.f45)]
// CHECK:STDOUT:     %GenericClass.loc50_39.1: type = class_type @GenericClass, @GenericClass(constants.%T.f45) [symbolic = %GenericClass.loc50_39.2 (constants.%GenericClass)]
// CHECK:STDOUT:     %Destroy.facet.loc50_39.1: %Destroy.type = facet_value %GenericClass.loc50_39.1, (constants.%DestroyImpl.lookup_impl_witness.85c, constants.%SubobjectDestroy.lookup_impl_witness.8a5) [symbolic = %Destroy.facet.loc50_39.2 (constants.%Destroy.facet.15c)]
// CHECK:STDOUT:     %.loc50_39.1: %Destroy.type = converted %GenericClass.loc50_39.1, %Destroy.facet.loc50_39.1 [symbolic = %Destroy.facet.loc50_39.2 (constants.%Destroy.facet.15c)]
// CHECK:STDOUT:     %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%.loc50_39.1)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDestroy(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc8_25.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc8_25.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestStructOfDestroy(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc18_33.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc18_33.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestTupleOfDestroy(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc28_32.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc28_32.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestArrayOfDestroy(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc38_32.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc38_32.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericClassOfDestroy(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc48_39.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc48_39.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_partial_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %BaseClassWithNonTrivialMember: type = class_type @BaseClassWithNonTrivialMember [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.n.81f: type = struct_type {.n: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.type: type = fn_type @CheckTypeImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %.68c: type = partial_type %BaseClassWithNonTrivialMember [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_25.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9e: %Destroy.type = facet_value %ImplsDestroy, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.906) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_25.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.823: %SubobjectDestroy.type = facet_value %.68c, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.dc5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.823) [concrete]
// CHECK:STDOUT:   %Destroy.facet.15d: %Destroy.type = facet_value %.68c, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.dc5) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = import_ref Main//check_destroy_format, CheckTypeImplsDestroy, loaded [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestPartialNonTrivialMemberBase() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %BaseClassWithNonTrivialMember.ref: type = name_ref BaseClassWithNonTrivialMember, file.%BaseClassWithNonTrivialMember.decl [concrete = constants.%BaseClassWithNonTrivialMember]
// CHECK:STDOUT:   %.loc13_25.1: type = partial_type %BaseClassWithNonTrivialMember.ref [concrete = constants.%.68c]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.loc13_25.1, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.dc5) [concrete = constants.%Destroy.facet.15d]
// CHECK:STDOUT:   %.loc13_25.2: %Destroy.type = converted %.loc13_25.1, %Destroy.facet [concrete = constants.%Destroy.facet.15d]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%.loc13_25.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_25.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_25.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_25.3(%self.param: ref %struct_type.n.81f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_25.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_25.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc13_25.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_25.1 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %Destroy.facet.loc13_25.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.906) [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   %.loc13_25.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_25.2 [concrete = constants.%Destroy.facet.c9e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_25.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_25.4(%self.param: ref %.68c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_partial_destroy_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.type: type = fn_type @CheckTypeImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %NonTrivialBase: type = class_type @NonTrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.657: type = partial_type %NonTrivialBase [concrete]
// CHECK:STDOUT:   %T: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_25.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.cb2: %SubobjectDestroy.type = facet_value %.657, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.605: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.cb2) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.657, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.605) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = import_ref Main//check_destroy_format, CheckTypeImplsDestroy, loaded [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %Main.NonTrivialBase: type = import_ref Main//types, NonTrivialBase, loaded [concrete = constants.%NonTrivialBase]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestPartialNonTrivialBase() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %NonTrivialBase.ref: type = name_ref NonTrivialBase, imports.%Main.NonTrivialBase [concrete = constants.%NonTrivialBase]
// CHECK:STDOUT:   %.loc11_25.1: type = partial_type %NonTrivialBase.ref [concrete = constants.%.657]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.loc11_25.1, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.605) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11_25.2: %Destroy.type = converted %.loc11_25.1, %Destroy.facet [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%.loc11_25.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_25.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_25.2(%self.param: ref %.657) = "no_op";
// CHECK:STDOUT:
