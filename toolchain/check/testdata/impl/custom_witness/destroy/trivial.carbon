// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/array.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/bool.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/float.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/trivial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/trivial.carbon

// --- core.carbon

package Core library "core";

export import library "prelude/parts/array";
export import library "prelude/parts/bool";
export import library "prelude/parts/float";
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

class EmptyClass {}
base class TrivialBase {}
abstract class AbstractEmptyBase {}

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- bool.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestBool(x: bool) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- bound_method.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";
import library "types";

class C {
  fn Get(self) -> i32;
}

fn TestBoundMethod(c: C) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(c.Get);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- char_literal.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestCharLiteral() {
  //@dump-sem-ir-begin
  CheckImplsDestroy('a');
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- facet_type.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestFacetType() {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy(Core.Destroy);
  // TODO: add CheckTypeImplsTrivialDestroy
  // TODO: add CheckTypeImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- float.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestFloat(x: f64) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- float_literal.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestFloatLiteral() {
  //@dump-sem-ir-begin
  CheckImplsDestroy(1.0);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- form.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestForm() {
  //@dump-sem-ir-begin
  CheckImplsDestroy(form(var i32));
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- function.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn G() {}

fn TestFunction() {
  //@dump-sem-ir-begin
  CheckImplsDestroy(G);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- function_with_self.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestFunctionWithSelf() {
  //@dump-sem-ir-begin
  CheckImplsDestroy((i32 as Core.Copy).Op);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- int_literal.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestIntLiteral() {
  //@dump-sem-ir-begin
  CheckImplsDestroy(1);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- int.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestInt(x: i32) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- pointer.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestPointer(x: i32*) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- empty_array.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";
import library "types";

fn TestEmptyArray(x: array(i32, 0)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

fn TestEmptyArrayOfNonTrivial(x: array(HasNonTrivialDestruction, 0)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- populated_array.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";
import library "types";

fn TestArray(x: array(EmptyClass, 3)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

fn TestArrayOfArray(x: array(array(EmptyClass, 2), 3)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

fn TestArrayOfEmptyArrayOfNonTrivial(x: array(array(HasNonTrivialDestruction, 0), 3)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- class_empty.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestEmptyClass(x: EmptyClass) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- class_with_trivial_fields.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";
import library "types";

class TrivialFields {
  var a: i32;
  var b: bool;
}

fn TestTrivialFields(x: TrivialFields) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

class NestedFields {
  var member: TrivialFields;
}

fn TestContainsTrivialMember(x: NestedFields) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- class_derived_from_trivial_base.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class DerivedFromTrivialBase {
  extend base: TrivialBase;
  var a: EmptyClass;
}

fn TestDerivedFromTrivialBase(x: DerivedFromTrivialBase) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- class_generic_is_trivial_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class HasGenericMember(T: Core.Destroy) {
  var t: T;
}

fn TestGenericClassOfTrivial(x: HasGenericMember(EmptyClass)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- class_adapter.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class ClassAdapter {
  adapt EmptyClass;
}

fn TestClassAdapter(x: ClassAdapter) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- array_adapter.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";
import library "types";

class ArrayAdapter {
  adapt array(EmptyClass, 2);
}

fn TestTrivialArrayAdapter(x: ArrayAdapter) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- class_dynamic.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

base class VirtualFunction {
  virtual fn F(self);
}

fn TestVirtualFunction(x: VirtualFunction) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

class DerivedFromVirtualFunction {
  extend base: VirtualFunction;
}

fn TestDerivedFromVirtualFunction(x: DerivedFromVirtualFunction) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- fail_todo_class_derived_from_abstract.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

class DerivedFromAbstract {
  extend base: AbstractEmptyBase;
}

// TODO: Support destroying classes that derive from abstract types.
fn TestDerivedFromAbstract(x: DerivedFromAbstract) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_class_derived_from_abstract.carbon:[[@LINE+4]]:3-22: error: cannot convert type `DerivedFromAbstract` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_todo_class_derived_from_abstract.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- const.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestConst(x: const i32) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- maybe_unformed.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestMaybeUnformed(x: Core.MaybeUnformed(i32)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- struct_empty.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestEmptyStruct() {
  //@dump-sem-ir-begin
  CheckImplsDestroy({});
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- struct_with_trivial_members.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestStructWithTrivialMembers(x: {.a: i32, .b: bool}) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

fn TestStructWithNestedTrivialStruct(x: {.a: {.b: i32}}) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- tuple_empty.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestEmptyTuple(x: ()) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- tuple_with_trivial_elements.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestTupleWithTrivialElements(x: (i32, bool)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

fn TestTupleWithNestedTrivialTuple(x: (i32, (bool, i32))) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- constrained_facet.carbon

library "[[@TEST_NAME]]";

import Core library "core";
import library "check_destroy_format";

fn TestDestroyAndTrivialDestroy(generic T: Core.Destroy & Core.Copy) {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy(T);
  // TODO: add CheckTypeImplsTrivialDestroy
  // TODO: add CheckTypeImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- type.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestType() {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy(type);
  // TODO: add CheckTypeImplsTrivialDestroy
  // TODO: add CheckImplsDynamicDestroy
  //@dump-sem-ir-end
}

// --- partial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestPartialTrivialBase() {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy(partial TrivialBase);
  // TODO: add CheckTypeImplsTrivialDestroy
  // TODO: add CheckTypeImplsDynamicDestroy
  //@dump-sem-ir-end
}

fn TestPartialAbstract() {
  //@dump-sem-ir-begin
  CheckTypeImplsDestroy(partial AbstractEmptyBase);
  // TODO: add CheckTypeImplsTrivialDestroy
  // TODO: add CheckTypeImplsDynamicDestroy
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- bool.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value bool, (%custom_witness.b80, %DestroyImpl.impl_witness.778) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBool(%x.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: bool = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted bool, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted bool, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- bound_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.Get.type: type = fn_type @C.Get [concrete]
// CHECK:STDOUT:   %C.Get: %C.Get.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8ad: %SubobjectDestroy.type = facet_value <bound method>, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.691: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8ad) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value <bound method>, (%custom_witness.b80, %DestroyImpl.impl_witness.691) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBoundMethod(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Get.ref: %C.Get.type = name_ref Get, @C.%C.Get.decl [concrete = constants.%C.Get]
// CHECK:STDOUT:   %C.Get.bound: <bound method> = bound_method %c.ref, %Get.ref
// CHECK:STDOUT:   %Destroy.facet.loc14_26.1: %Destroy.type = facet_value <bound method>, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.691) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_26.1: %Destroy.type = converted <bound method>, %Destroy.facet.loc14_26.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_26.2: %Destroy.type = facet_value <bound method>, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.691) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_26.2: %Destroy.type = converted <bound method>, %Destroy.facet.loc14_26.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%C.Get.bound)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14(%self.param: ref <bound method>) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- char_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.576: %SubobjectDestroy.type = facet_value Core.CharLiteral, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.96a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.576) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value Core.CharLiteral, (%custom_witness.b80, %DestroyImpl.impl_witness.96a) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCharLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %.loc8_21: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %Destroy.facet.loc8_24.1: %Destroy.type = facet_value Core.CharLiteral, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.96a) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_24.1: %Destroy.type = converted Core.CharLiteral, %Destroy.facet.loc8_24.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_24.2: %Destroy.type = facet_value Core.CharLiteral, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.96a) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_24.2: %Destroy.type = converted Core.CharLiteral, %Destroy.facet.loc8_24.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%.loc8_21)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref Core.CharLiteral) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- facet_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.type: type = fn_type @CheckTypeImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f5: %SubobjectDestroy.type = facet_value %Destroy.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a54: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f5) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Destroy.type, (%custom_witness.b80, %DestroyImpl.impl_witness.a54) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = import_ref Main//check_destroy_format, CheckTypeImplsDestroy, loaded [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestFacetType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Destroy.ref, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8: %Destroy.type = converted %Destroy.ref, %Destroy.facet [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %Destroy.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- float.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %f64.761: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %f64.794: type = float_type %int_64, f64 [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.47f: %SubobjectDestroy.type = facet_value %f64.761, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a23: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.47f) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %f64.761, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestFloat(%x.param: %f64.761) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %f64.761 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%f64.761, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.a23) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%f64.761, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%f64.761, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.a23) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%f64.761, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %f64.794) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %f64.761) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- float_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d80: %SubobjectDestroy.type = facet_value Core.FloatLiteral, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.62f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d80) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value Core.FloatLiteral, (%custom_witness.b80, %DestroyImpl.impl_witness.62f) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestFloatLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float]
// CHECK:STDOUT:   %Destroy.facet.loc8_24.1: %Destroy.type = facet_value Core.FloatLiteral, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.62f) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_24.1: %Destroy.type = converted Core.FloatLiteral, %Destroy.facet.loc8_24.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_24.2: %Destroy.type = facet_value Core.FloatLiteral, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.62f) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_24.2: %Destroy.type = converted Core.FloatLiteral, %Destroy.facet.loc8_24.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%float)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref Core.FloatLiteral) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- form.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.4de: %SubobjectDestroy.type = facet_value Core.Form, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.fb5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.4de) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value Core.Form, (%custom_witness.b80, %DestroyImpl.impl_witness.fb5) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestForm() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9_26: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:   %Destroy.facet.loc9_34.1: %Destroy.type = facet_value Core.Form, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.fb5) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_34.1: %Destroy.type = converted Core.Form, %Destroy.facet.loc9_34.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_34.2: %Destroy.type = facet_value Core.Form, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.fb5) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_34.2: %Destroy.type = converted Core.Form, %Destroy.facet.loc9_34.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%.loc9_26)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref Core.Form) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d43: %SubobjectDestroy.type = facet_value %G.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.846: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d43) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %G.type, (%custom_witness.b80, %DestroyImpl.impl_witness.846) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestFunction() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %Destroy.facet.loc10_22.1: %Destroy.type = facet_value constants.%G.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.846) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_22.1: %Destroy.type = converted constants.%G.type, %Destroy.facet.loc10_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc10_22.2: %Destroy.type = facet_value constants.%G.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.846) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_22.2: %Destroy.type = converted constants.%G.type, %Destroy.facet.loc10_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%G.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %G.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- function_with_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Self.f84: %Copy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2f9c3.1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Self.f84) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.3f89b0.1: %Copy.WithSelf.Op.type.d2f9c3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Copy.assoc_type: type = assoc_entity_type @Copy [concrete]
// CHECK:STDOUT:   %assoc0: %Copy.assoc_type = assoc_entity element0, imports.%Core.import_ref.ea51 [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ab7: %SubobjectDestroy.type = facet_value %.2b9, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e48: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ab7) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.2b9, (%custom_witness.b80, %DestroyImpl.impl_witness.e48) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//core
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.258: %Copy.assoc_type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ea51: @Copy.WithSelf.%Copy.WithSelf.Op.type (%Copy.WithSelf.Op.type.d2f9c3.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @Copy.WithSelf.%Copy.WithSelf.Op (constants.%Copy.WithSelf.Op.3f89b0.1)]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestFunctionWithSelf() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:   %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc9_26: %Copy.type = converted %i32, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc9_26 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9_39: type = converted %.loc9_26, %as_type [concrete = constants.%i32]
// CHECK:STDOUT:   %Op.ref: %Copy.assoc_type = name_ref Op, imports.%Core.import_ref.258 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %Destroy.facet.loc9_42.1: %Destroy.type = facet_value constants.%.2b9, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.e48) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_42.1: %Destroy.type = converted constants.%.2b9, %Destroy.facet.loc9_42.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_42.2: %Destroy.type = facet_value constants.%.2b9, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.e48) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_42.2: %Destroy.type = converted constants.%.2b9, %Destroy.facet.loc9_42.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%impl.elem0)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %.2b9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.659: %SubobjectDestroy.type = facet_value Core.IntLiteral, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.9b4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.659) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value Core.IntLiteral, (%custom_witness.b80, %DestroyImpl.impl_witness.9b4) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value Core.IntLiteral, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.9b4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value Core.IntLiteral, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.9b4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%int_1)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref Core.IntLiteral) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %i32, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.077) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestInt(%x.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%i32, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%i32, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%i32, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d4d: %SubobjectDestroy.type = facet_value %ptr, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b8f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d4d) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr, (%custom_witness.b80, %DestroyImpl.impl_witness.b8f) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestPointer(%x.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %ptr = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%ptr, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.b8f) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%ptr, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%ptr, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.b8f) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%ptr, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %ptr) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- empty_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Array.68f: type = class_type @Array, @Array(%i32, %int_0) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8b7: %SubobjectDestroy.type = facet_value %Array.68f, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8b7) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c0c: %Destroy.type = facet_value %Array.68f, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.cca) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.9e1: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.c0c) [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %Array.fbd: type = class_type @Array, @Array(%HasNonTrivialDestruction, %int_0) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6c7: %SubobjectDestroy.type = facet_value %Array.fbd, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cf7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6c7) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e2e: %Destroy.type = facet_value %Array.fbd, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.cf7) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.887: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.e2e) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyArray(%x.param: %Array.68f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Array.68f = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc10_22.1: %Destroy.type = facet_value constants.%Array.68f, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.cca) [concrete = constants.%Destroy.facet.c0c]
// CHECK:STDOUT:   %.loc10_22.1: %Destroy.type = converted constants.%Array.68f, %Destroy.facet.loc10_22.1 [concrete = constants.%Destroy.facet.c0c]
// CHECK:STDOUT:   %Destroy.facet.loc10_22.2: %Destroy.type = facet_value constants.%Array.68f, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.cca) [concrete = constants.%Destroy.facet.c0c]
// CHECK:STDOUT:   %.loc10_22.2: %Destroy.type = converted constants.%Array.68f, %Destroy.facet.loc10_22.2 [concrete = constants.%Destroy.facet.c0c]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.c0c) [concrete = constants.%CheckImplsDestroy.specific_fn.9e1]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %Array.68f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyArrayOfNonTrivial(%x.param: %Array.fbd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Array.fbd = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc18_22.1: %Destroy.type = facet_value constants.%Array.fbd, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.cf7) [concrete = constants.%Destroy.facet.e2e]
// CHECK:STDOUT:   %.loc18_22.1: %Destroy.type = converted constants.%Array.fbd, %Destroy.facet.loc18_22.1 [concrete = constants.%Destroy.facet.e2e]
// CHECK:STDOUT:   %Destroy.facet.loc18_22.2: %Destroy.type = facet_value constants.%Array.fbd, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.cf7) [concrete = constants.%Destroy.facet.e2e]
// CHECK:STDOUT:   %.loc18_22.2: %Destroy.type = converted constants.%Array.fbd, %Destroy.facet.loc18_22.2 [concrete = constants.%Destroy.facet.e2e]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.e2e) [concrete = constants.%CheckImplsDestroy.specific_fn.887]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18(%self.param: ref %Array.fbd) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- populated_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %EmptyClass: type = class_type @EmptyClass [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.db5: type = class_type @Array, @Array(%EmptyClass, %int_3) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.905: %SubobjectDestroy.type = facet_value %Array.db5, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e19: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.905) [concrete]
// CHECK:STDOUT:   %Destroy.facet.22a: %Destroy.type = facet_value %Array.db5, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.e19) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.609: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.22a) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Array.dc0: type = class_type @Array, @Array(%EmptyClass, %int_2) [concrete]
// CHECK:STDOUT:   %Array.322: type = class_type @Array, @Array(%Array.dc0, %int_3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.422: %SubobjectDestroy.type = facet_value %Array.322, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cdb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.422) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c52: %Destroy.type = facet_value %Array.322, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.cdb) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.454: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.c52) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Array.fbd: type = class_type @Array, @Array(%HasNonTrivialDestruction, %int_0) [concrete]
// CHECK:STDOUT:   %Array.ef0: type = class_type @Array, @Array(%Array.fbd, %int_3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.fdd: %SubobjectDestroy.type = facet_value %Array.ef0, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.040: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.fdd) [concrete]
// CHECK:STDOUT:   %Destroy.facet.17d: %Destroy.type = facet_value %Array.ef0, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.040) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.3c4: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.17d) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestArray(%x.param: %Array.db5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Array.db5 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc10_22.1: %Destroy.type = facet_value constants.%Array.db5, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.e19) [concrete = constants.%Destroy.facet.22a]
// CHECK:STDOUT:   %.loc10_22.1: %Destroy.type = converted constants.%Array.db5, %Destroy.facet.loc10_22.1 [concrete = constants.%Destroy.facet.22a]
// CHECK:STDOUT:   %Destroy.facet.loc10_22.2: %Destroy.type = facet_value constants.%Array.db5, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.e19) [concrete = constants.%Destroy.facet.22a]
// CHECK:STDOUT:   %.loc10_22.2: %Destroy.type = converted constants.%Array.db5, %Destroy.facet.loc10_22.2 [concrete = constants.%Destroy.facet.22a]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.22a) [concrete = constants.%CheckImplsDestroy.specific_fn.609]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_22.2(%self.param: ref %EmptyClass) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_22.3(%self.param: ref %Array.db5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestArrayOfArray(%x.param: %Array.322) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Array.322 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc18_22.1: %Destroy.type = facet_value constants.%Array.322, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.cdb) [concrete = constants.%Destroy.facet.c52]
// CHECK:STDOUT:   %.loc18_22.1: %Destroy.type = converted constants.%Array.322, %Destroy.facet.loc18_22.1 [concrete = constants.%Destroy.facet.c52]
// CHECK:STDOUT:   %Destroy.facet.loc18_22.2: %Destroy.type = facet_value constants.%Array.322, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.cdb) [concrete = constants.%Destroy.facet.c52]
// CHECK:STDOUT:   %.loc18_22.2: %Destroy.type = converted constants.%Array.322, %Destroy.facet.loc18_22.2 [concrete = constants.%Destroy.facet.c52]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.c52) [concrete = constants.%CheckImplsDestroy.specific_fn.454]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_22.1(%self.param: ref %Array.dc0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_22.2(%self.param: ref %Array.322) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestArrayOfEmptyArrayOfNonTrivial(%x.param: %Array.ef0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Array.ef0 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc26_22.1: %Destroy.type = facet_value constants.%Array.ef0, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.040) [concrete = constants.%Destroy.facet.17d]
// CHECK:STDOUT:   %.loc26_22.1: %Destroy.type = converted constants.%Array.ef0, %Destroy.facet.loc26_22.1 [concrete = constants.%Destroy.facet.17d]
// CHECK:STDOUT:   %Destroy.facet.loc26_22.2: %Destroy.type = facet_value constants.%Array.ef0, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.040) [concrete = constants.%Destroy.facet.17d]
// CHECK:STDOUT:   %.loc26_22.2: %Destroy.type = converted constants.%Array.ef0, %Destroy.facet.loc26_22.2 [concrete = constants.%Destroy.facet.17d]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.17d) [concrete = constants.%CheckImplsDestroy.specific_fn.3c4]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_22.1(%self.param: ref %Array.fbd) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_22.2(%self.param: ref %Array.ef0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- class_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %EmptyClass: type = class_type @EmptyClass [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.5c8: %SubobjectDestroy.type = facet_value %EmptyClass, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ec0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5c8) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %EmptyClass, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.ec0) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyClass(%x.param: %EmptyClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %EmptyClass = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%EmptyClass, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.ec0) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%EmptyClass, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%EmptyClass, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.ec0) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%EmptyClass, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %EmptyClass) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- class_with_trivial_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialFields: type = class_type @TrivialFields [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: bool} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_22.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.16d: %SubobjectDestroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c88: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.16d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.bc5: %Destroy.type = facet_value %TrivialFields, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.c88) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.bdc: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.bc5) [concrete]
// CHECK:STDOUT:   %NestedFields: type = class_type @NestedFields [concrete]
// CHECK:STDOUT:   %struct_type.member.b2a: type = struct_type {.member: %TrivialFields} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.697: %SubobjectDestroy.type = facet_value %NestedFields, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.512: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.697) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9aa: %Destroy.type = facet_value %NestedFields, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.512) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.fa6: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.9aa) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTrivialFields(%x.param: %TrivialFields) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %TrivialFields = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc15_22.1: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc15_22.1: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc15_22.1 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %Destroy.facet.loc15_22.2: %Destroy.type = facet_value constants.%TrivialFields, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c88) [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %.loc15_22.2: %Destroy.type = converted constants.%TrivialFields, %Destroy.facet.loc15_22.2 [concrete = constants.%Destroy.facet.bc5]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.bc5) [concrete = constants.%CheckImplsDestroy.specific_fn.bdc]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_22.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_22.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_22.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_22.4(%self.param: ref %struct_type.a.b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_22.5(%self.param: ref %TrivialFields) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestContainsTrivialMember(%x.param: %NestedFields) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %NestedFields = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc27_22.1: %Destroy.type = facet_value constants.%NestedFields, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.512) [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   %.loc27_22.1: %Destroy.type = converted constants.%NestedFields, %Destroy.facet.loc27_22.1 [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   %Destroy.facet.loc27_22.2: %Destroy.type = facet_value constants.%NestedFields, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.512) [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   %.loc27_22.2: %Destroy.type = converted constants.%NestedFields, %Destroy.facet.loc27_22.2 [concrete = constants.%Destroy.facet.9aa]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.9aa) [concrete = constants.%CheckImplsDestroy.specific_fn.fa6]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_22.1(%self.param: ref %struct_type.member.b2a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_22.2(%self.param: ref %NestedFields) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- class_derived_from_trivial_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %DerivedFromTrivialBase: type = class_type @DerivedFromTrivialBase [concrete]
// CHECK:STDOUT:   %TrivialBase: type = class_type @TrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %EmptyClass: type = class_type @EmptyClass [concrete]
// CHECK:STDOUT:   %struct_type.base.a.5c0: type = struct_type {.base: %TrivialBase, .a: %EmptyClass} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_22.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.721: %SubobjectDestroy.type = facet_value %DerivedFromTrivialBase, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f18: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.721) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %DerivedFromTrivialBase, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.f18) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerivedFromTrivialBase(%x.param: %DerivedFromTrivialBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %DerivedFromTrivialBase = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc14_22.1: %Destroy.type = facet_value constants.%DerivedFromTrivialBase, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.f18) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_22.1: %Destroy.type = converted constants.%DerivedFromTrivialBase, %Destroy.facet.loc14_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.2: %Destroy.type = facet_value constants.%DerivedFromTrivialBase, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.f18) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_22.2: %Destroy.type = converted constants.%DerivedFromTrivialBase, %Destroy.facet.loc14_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.2(%self.param: ref %TrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.3(%self.param: ref %EmptyClass) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.4(%self.param: ref %struct_type.base.a.5c0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.5(%self.param: ref %DerivedFromTrivialBase) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- class_generic_is_trivial_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %EmptyClass: type = class_type @EmptyClass [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_60.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5c8: %SubobjectDestroy.type = facet_value %EmptyClass, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ec0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5c8) [concrete]
// CHECK:STDOUT:   %Destroy.facet.643: %Destroy.type = facet_value %EmptyClass, (%DestroyImpl.impl_witness.ec0, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %HasGenericMember.66d: type = class_type @HasGenericMember, @HasGenericMember(%Destroy.facet.643) [concrete]
// CHECK:STDOUT:   %struct_type.t.252: type = struct_type {.t: %EmptyClass} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.29f: %SubobjectDestroy.type = facet_value %HasGenericMember.66d, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0c3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.29f) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0c7: %Destroy.type = facet_value %HasGenericMember.66d, (%DestroyImpl.impl_witness.0c3, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.0c7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGenericClassOfTrivial(%x.param: %HasGenericMember.66d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %HasGenericMember.66d = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%HasGenericMember.66d, (constants.%DestroyImpl.impl_witness.0c3, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.0c7]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%HasGenericMember.66d, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.0c7]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%HasGenericMember.66d, (constants.%DestroyImpl.impl_witness.0c3, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.0c7]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%HasGenericMember.66d, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.0c7]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.0c7) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %struct_type.t.252) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %HasGenericMember.66d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- class_adapter.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ClassAdapter: type = class_type @ClassAdapter [concrete]
// CHECK:STDOUT:   %EmptyClass: type = class_type @EmptyClass [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.impl_witness.423: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.88c) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ClassAdapter, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.423) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %EmptyClass) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.3(%self.param: ref %ClassAdapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- array_adapter.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ArrayAdapter: type = class_type @ArrayAdapter [concrete]
// CHECK:STDOUT:   %EmptyClass: type = class_type @EmptyClass [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.dc0: type = class_type @Array, @Array(%EmptyClass, %int_2) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_22.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.2b2: %SubobjectDestroy.type = facet_value %ArrayAdapter, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e75: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.2b2) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ArrayAdapter, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.e75) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTrivialArrayAdapter(%x.param: %ArrayAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %ArrayAdapter = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc14_22.1: %Destroy.type = facet_value constants.%ArrayAdapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e75) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_22.1: %Destroy.type = converted constants.%ArrayAdapter, %Destroy.facet.loc14_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_22.2: %Destroy.type = facet_value constants.%ArrayAdapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e75) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_22.2: %Destroy.type = converted constants.%ArrayAdapter, %Destroy.facet.loc14_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.2(%self.param: ref %EmptyClass) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.3(%self.param: ref %Array.dc0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_22.4(%self.param: ref %ArrayAdapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- class_dynamic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %VirtualFunction: type = class_type @VirtualFunction [concrete]
// CHECK:STDOUT:   %VirtualFunction.F.type: type = fn_type @VirtualFunction.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %VirtualFunction.F: %VirtualFunction.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.0f2: %SubobjectDestroy.type = facet_value %VirtualFunction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a93: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0f2) [concrete]
// CHECK:STDOUT:   %Destroy.facet.d61: %Destroy.type = facet_value %VirtualFunction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a93) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.3d3: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.d61) [concrete]
// CHECK:STDOUT:   %DerivedFromVirtualFunction: type = class_type @DerivedFromVirtualFunction [concrete]
// CHECK:STDOUT:   %struct_type.base.95a: type = struct_type {.base: %VirtualFunction} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc25_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.440: %SubobjectDestroy.type = facet_value %DerivedFromVirtualFunction, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0ba: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.440) [concrete]
// CHECK:STDOUT:   %Destroy.facet.de0: %Destroy.type = facet_value %DerivedFromVirtualFunction, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.0ba) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.ef3: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.de0) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @VirtualFunction.vtable {
// CHECK:STDOUT:   @VirtualFunction.%VirtualFunction.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @DerivedFromVirtualFunction.vtable {
// CHECK:STDOUT:   constants.%VirtualFunction.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestVirtualFunction(%x.param: %VirtualFunction) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %VirtualFunction = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%VirtualFunction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a93) [concrete = constants.%Destroy.facet.d61]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%VirtualFunction, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.d61]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%VirtualFunction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a93) [concrete = constants.%Destroy.facet.d61]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%VirtualFunction, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.d61]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.d61) [concrete = constants.%CheckImplsDestroy.specific_fn.3d3]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %struct_type.vptr) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.3(%self.param: ref %VirtualFunction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerivedFromVirtualFunction(%x.param: %DerivedFromVirtualFunction) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %DerivedFromVirtualFunction = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc25_22.1: %Destroy.type = facet_value constants.%DerivedFromVirtualFunction, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.0ba) [concrete = constants.%Destroy.facet.de0]
// CHECK:STDOUT:   %.loc25_22.1: %Destroy.type = converted constants.%DerivedFromVirtualFunction, %Destroy.facet.loc25_22.1 [concrete = constants.%Destroy.facet.de0]
// CHECK:STDOUT:   %Destroy.facet.loc25_22.2: %Destroy.type = facet_value constants.%DerivedFromVirtualFunction, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.0ba) [concrete = constants.%Destroy.facet.de0]
// CHECK:STDOUT:   %.loc25_22.2: %Destroy.type = converted constants.%DerivedFromVirtualFunction, %Destroy.facet.loc25_22.2 [concrete = constants.%Destroy.facet.de0]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.de0) [concrete = constants.%CheckImplsDestroy.specific_fn.ef3]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc25_22.1(%self.param: ref %struct_type.base.95a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc25_22.2(%self.param: ref %DerivedFromVirtualFunction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_class_derived_from_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %DerivedFromAbstract: type = class_type @DerivedFromAbstract [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerivedFromAbstract(%x.param: %DerivedFromAbstract) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %DerivedFromAbstract = name_ref x, %x
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %const: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.7d5: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e17: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7d5) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %const, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.e17) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestConst(%x.param: %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %const = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.e17) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%const, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.e17) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.3(%self.param: ref %const) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- maybe_unformed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.752: type = class_type @MaybeUnformed, @MaybeUnformed(%i32) [concrete]
// CHECK:STDOUT:   %.b31: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.c42: %SubobjectDestroy.type = facet_value %MaybeUnformed.752, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.433: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c42) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %MaybeUnformed.752, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.433) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestMaybeUnformed(%x.param: %MaybeUnformed.752) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %MaybeUnformed.752 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%MaybeUnformed.752, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.433) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%MaybeUnformed.752, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%MaybeUnformed.752, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.433) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%MaybeUnformed.752, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.3(%self.param: ref %.b31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.4(%self.param: ref %MaybeUnformed.752) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- struct_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %empty_struct_type, (%custom_witness.b80, %DestroyImpl.impl_witness.011) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %.loc8_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Destroy.facet.loc8_23.1: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_23.1: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc8_23.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_23.2: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_23.2: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc8_23.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_22.2: %empty_struct_type = converted %.loc8_22.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%.loc8_22.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- struct_with_trivial_members.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: bool} [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.f95: %SubobjectDestroy.type = facet_value %struct_type.a.b, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cfb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f95) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e79: %Destroy.type = facet_value %struct_type.a.b, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.cfb) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.445: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.e79) [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %i32} [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %struct_type.b} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.91e: %SubobjectDestroy.type = facet_value %struct_type.a, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.19a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.91e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e68: %Destroy.type = facet_value %struct_type.a, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.19a) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.e27: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.e68) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestStructWithTrivialMembers(%x.param: %struct_type.a.b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %struct_type.a.b = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%struct_type.a.b, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.cfb) [concrete = constants.%Destroy.facet.e79]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%struct_type.a.b, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.e79]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%struct_type.a.b, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.cfb) [concrete = constants.%Destroy.facet.e79]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%struct_type.a.b, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.e79]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.e79) [concrete = constants.%CheckImplsDestroy.specific_fn.445]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.4(%self.param: ref %struct_type.a.b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestStructWithNestedTrivialStruct(%x.param: %struct_type.a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %struct_type.a = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc17_22.1: %Destroy.type = facet_value constants.%struct_type.a, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.19a) [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %.loc17_22.1: %Destroy.type = converted constants.%struct_type.a, %Destroy.facet.loc17_22.1 [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %Destroy.facet.loc17_22.2: %Destroy.type = facet_value constants.%struct_type.a, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.19a) [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %.loc17_22.2: %Destroy.type = converted constants.%struct_type.a, %Destroy.facet.loc17_22.2 [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.e68) [concrete = constants.%CheckImplsDestroy.specific_fn.e27]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_22.1(%self.param: ref %struct_type.b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_22.2(%self.param: ref %struct_type.a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %empty_tuple.type, (%custom_witness.b80, %DestroyImpl.impl_witness.674) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyTuple(%x.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.674) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.674) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_with_trivial_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.2e0: type = tuple_type (%i32, bool) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.554: %SubobjectDestroy.type = facet_value %tuple.type.2e0, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.750: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.554) [concrete]
// CHECK:STDOUT:   %Destroy.facet.94d: %Destroy.type = facet_value %tuple.type.2e0, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.750) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.25a: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.94d) [concrete]
// CHECK:STDOUT:   %tuple.type.02f: type = tuple_type (bool, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.79a: type = tuple_type (%i32, %tuple.type.02f) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.010: %SubobjectDestroy.type = facet_value %tuple.type.79a, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.abf: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.010) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f03: %Destroy.type = facet_value %tuple.type.79a, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.abf) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.59d: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.f03) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTupleWithTrivialElements(%x.param: %tuple.type.2e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %tuple.type.2e0 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%tuple.type.2e0, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.750) [concrete = constants.%Destroy.facet.94d]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%tuple.type.2e0, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.94d]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%tuple.type.2e0, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.750) [concrete = constants.%Destroy.facet.94d]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%tuple.type.2e0, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.94d]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.94d) [concrete = constants.%CheckImplsDestroy.specific_fn.25a]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.4(%self.param: ref %tuple.type.2e0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTupleWithNestedTrivialTuple(%x.param: %tuple.type.79a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %tuple.type.79a = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc17_22.1: %Destroy.type = facet_value constants.%tuple.type.79a, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.abf) [concrete = constants.%Destroy.facet.f03]
// CHECK:STDOUT:   %.loc17_22.1: %Destroy.type = converted constants.%tuple.type.79a, %Destroy.facet.loc17_22.1 [concrete = constants.%Destroy.facet.f03]
// CHECK:STDOUT:   %Destroy.facet.loc17_22.2: %Destroy.type = facet_value constants.%tuple.type.79a, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.abf) [concrete = constants.%Destroy.facet.f03]
// CHECK:STDOUT:   %.loc17_22.2: %Destroy.type = converted constants.%tuple.type.79a, %Destroy.facet.loc17_22.2 [concrete = constants.%Destroy.facet.f03]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.f03) [concrete = constants.%CheckImplsDestroy.specific_fn.59d]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_22.1(%self.param: ref %tuple.type.02f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_22.2(%self.param: ref %tuple.type.79a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- constrained_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %facet_type.bd0: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %facet_type.bd0 [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %facet_type.bd0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.type: type = fn_type @CheckTypeImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %T.f84, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.f84, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %T.as_type.c1c, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = import_ref Main//check_destroy_format, CheckTypeImplsDestroy, loaded [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestDestroyAndTrivialDestroy(%T.loc7_42.2: %facet_type.bd0) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc9_25.2: type = facet_access_type %T.loc7_42.1 [symbolic = %T.as_type.loc9_25.2 (constants.%T.as_type.c1c)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc7_42.1, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc7_42.1, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet.loc9_25.2: %Destroy.type = facet_value %T.as_type.loc9_25.2, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc9_25.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:     %T.ref: %facet_type.bd0 = name_ref T, %T.loc7_42.2 [symbolic = %T.loc7_42.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc9_25.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_25.2 (constants.%T.as_type.c1c)]
// CHECK:STDOUT:     %Destroy.facet.loc9_25.1: %Destroy.type = facet_value %T.as_type.loc9_25.1, (constants.%DestroyImpl.lookup_impl_witness, constants.%SubobjectDestroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc9_25.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc9: %Destroy.type = converted %T.ref, %Destroy.facet.loc9_25.1 [symbolic = %Destroy.facet.loc9_25.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%.loc9)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDestroyAndTrivialDestroy(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc7_42.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc7_42.1 => constants.%T.f84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.type: type = fn_type @CheckTypeImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f5: %SubobjectDestroy.type = facet_value type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a54: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f5) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value type, (%custom_witness.b80, %DestroyImpl.impl_witness.a54) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = import_ref Main//check_destroy_format, CheckTypeImplsDestroy, loaded [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %.loc8_25.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.loc8_25.1, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_25.2: %Destroy.type = converted %.loc8_25.1, %Destroy.facet [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%.loc8_25.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- partial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.type: type = fn_type @CheckTypeImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %TrivialBase: type = class_type @TrivialBase [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.f81: type = partial_type %TrivialBase [concrete]
// CHECK:STDOUT:   %T: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_25.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.833: %SubobjectDestroy.type = facet_value %.f81, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d81: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.833) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c96: %Destroy.type = facet_value %.f81, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.d81) [concrete]
// CHECK:STDOUT:   %AbstractEmptyBase: type = class_type @AbstractEmptyBase [concrete]
// CHECK:STDOUT:   %.3bd: type = partial_type %AbstractEmptyBase [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.32c: %SubobjectDestroy.type = facet_value %.3bd, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0f7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.32c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.2ad: %Destroy.type = facet_value %.3bd, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.0f7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckTypeImplsDestroy: %CheckTypeImplsDestroy.type = import_ref Main//check_destroy_format, CheckTypeImplsDestroy, loaded [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %Main.TrivialBase: type = import_ref Main//types, TrivialBase, loaded [concrete = constants.%TrivialBase]
// CHECK:STDOUT:   %Main.AbstractEmptyBase: type = import_ref Main//types, AbstractEmptyBase, loaded [concrete = constants.%AbstractEmptyBase]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestPartialTrivialBase() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %TrivialBase.ref: type = name_ref TrivialBase, imports.%Main.TrivialBase [concrete = constants.%TrivialBase]
// CHECK:STDOUT:   %.loc9_25.1: type = partial_type %TrivialBase.ref [concrete = constants.%.f81]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.loc9_25.1, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.d81) [concrete = constants.%Destroy.facet.c96]
// CHECK:STDOUT:   %.loc9_25.2: %Destroy.type = converted %.loc9_25.1, %Destroy.facet [concrete = constants.%Destroy.facet.c96]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%.loc9_25.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_25.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_25.2(%self.param: ref %.f81) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestPartialAbstract() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckTypeImplsDestroy.ref: %CheckTypeImplsDestroy.type = name_ref CheckTypeImplsDestroy, imports.%Main.CheckTypeImplsDestroy [concrete = constants.%CheckTypeImplsDestroy]
// CHECK:STDOUT:   %AbstractEmptyBase.ref: type = name_ref AbstractEmptyBase, imports.%Main.AbstractEmptyBase [concrete = constants.%AbstractEmptyBase]
// CHECK:STDOUT:   %.loc17_25.1: type = partial_type %AbstractEmptyBase.ref [concrete = constants.%.3bd]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.loc17_25.1, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.0f7) [concrete = constants.%Destroy.facet.2ad]
// CHECK:STDOUT:   %.loc17_25.2: %Destroy.type = converted %.loc17_25.1, %Destroy.facet [concrete = constants.%Destroy.facet.2ad]
// CHECK:STDOUT:   %CheckTypeImplsDestroy.call: init %empty_tuple.type = call %CheckTypeImplsDestroy.ref(%.loc17_25.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17(%self.param: ref %.3bd) = "no_op";
// CHECK:STDOUT:
