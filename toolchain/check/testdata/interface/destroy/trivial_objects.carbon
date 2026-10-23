// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy/trivial_objects.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy/trivial_objects.carbon

// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a type can be destroyed.
//
// Use this function to confirm non-trivial destruction (pass) and types that
// can't be destroyed (fail).
fn CheckHasDestruction[T: Core.Destroy](_: T) {}

// Identifies whether a type has trivial destruction.
//
// Use this function to confirm trivial destruction (pass), non-trivial
// destruction (fail), and types that can't be destroyed (fail).
//
// `CheckHasTrivialDestruction` should only be combined with
// `CheckHasDestruction` for types that are expected to not have trivial
// destruction. Using both on a trivially destroyable type (e.g. `i32`) is
// redundant.
fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}

// --- types.carbon

library "[[@TEST_NAME]]";

// Types used by the splits below.

// Has non-trivial destruction because of its own `DestroyImpl.Op`.
class NonTrivial {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

base class TrivialBase {}

abstract class Abstract {}

// A generic class with a member of type `T`. Tests that the destruction of a
// generic class comes from its type argument: `DestroyableBox(i32)` has trivial
// destruction, and `DestroyableBox(NonTrivial)` has non-trivial destruction.
// The generics tests also use it with a symbolic argument.
class DestroyableBox(T: Core.Destroy) {
  var t: T;
}

class C {
  fn Get(self) -> i32;
}

class EmptyClass {}

class TrivialFields {
  var a: i32;
  var b: bool;
}

class DerivedFromTrivialBase {
  extend base: TrivialBase;
  var a: i32;
}

class ContainsTrivialMember {
  var member: TrivialFields;
}

class AdaptInt {
  adapt i32;
}

class AdaptTrivialArray {
  adapt array(i32, 2);
}

base class VirtualFunction {
  virtual fn F(self);
}

class DerivedFromVirtualFunction {
  extend base: VirtualFunction;
}

class DerivedFromAbstract {
  extend base: Abstract;
}

// --- bool.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestBool(x: bool) {
  CheckHasTrivialDestruction(x);
}

// --- bound_method.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestBoundMethod(c: C) {
  CheckHasTrivialDestruction(c.Get);
}

// --- char_literal.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestCharLiteral() {
  CheckHasTrivialDestruction('a');
}

// --- float.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestF64(x: f64) {
  CheckHasTrivialDestruction(x);
}

// --- float_literal.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestFloatLiteral() {
  CheckHasTrivialDestruction(1.0);
}

// --- form.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestForm() {
  CheckHasTrivialDestruction(form(var i32));
}

// --- function.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn G() {}

fn TestFunction() {
  CheckHasTrivialDestruction(G);
}

// --- function_with_self.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestFunctionWithSelf() {
  CheckHasTrivialDestruction((i32 as Core.Copy).Op);
}

// --- int_literal.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestIntLiteral() {
  CheckHasTrivialDestruction(1);
}

// --- int.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestI32(x: i32) {
  CheckHasTrivialDestruction(x);
}

fn TestU8(x: u8) {
  CheckHasTrivialDestruction(x);
}

// --- pointer.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestPointer(x: i32*) {
  CheckHasTrivialDestruction(x);
}

// --- array.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestEmptyArray(x: array(i32, 0)) {
  CheckHasTrivialDestruction(x);
}

fn TestEmptyArrayOfNonTrivial(x: array(NonTrivial, 0)) {
  CheckHasTrivialDestruction(x);
}

fn TestArray(x: array(i32, 3)) {
  CheckHasTrivialDestruction(x);
}

fn TestArrayOfArray(x: array(array(i32, 2), 3)) {
  CheckHasTrivialDestruction(x);
}

fn TestArrayOfEmptyArrayOfNonTrivial(x: array(array(NonTrivial, 0), 3)) {
  CheckHasTrivialDestruction(x);
}

// --- class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

//@dump-sem-ir-begin
fn TestEmptyClass(x: EmptyClass) {
  CheckHasTrivialDestruction(x);
}
//@dump-sem-ir-end

fn TestTrivialFields(x: TrivialFields) {
  CheckHasTrivialDestruction(x);
}

fn TestDerivedFromTrivialBase(x: DerivedFromTrivialBase) {
  CheckHasTrivialDestruction(x);
}

fn TestContainsTrivialMember(x: ContainsTrivialMember) {
  CheckHasTrivialDestruction(x);
}

fn TestGenericClassOfTrivial(x: DestroyableBox(i32)) {
  CheckHasTrivialDestruction(x);
}

fn TestAdaptInt(x: AdaptInt) {
  CheckHasTrivialDestruction(x);
}

fn TestAdaptTrivialArray(x: AdaptTrivialArray) {
  CheckHasTrivialDestruction(x);
}

// --- todo_class_dynamic.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

// TODO: Decide what kind of destruction a dynamic class has. A dynamic class
// has a virtual function, either directly or through a base class. Today these
// have trivial destruction.

fn TestVirtualFunction(x: VirtualFunction) {
  CheckHasTrivialDestruction(x);
}

fn TestDerivedFromVirtualFunction(x: DerivedFromVirtualFunction) {
  CheckHasTrivialDestruction(x);
}

// --- todo_class_optional.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

// TODO: `Core.Optional` depends on `Core.MaybeUnformed` being correct, so this
// stays a TODO until `Core.MaybeUnformed` is complete.
fn TestOptional(x: Core.Optional(i32)) {
  CheckHasTrivialDestruction(x);
}

// --- fail_todo_class_derived_from_abstract.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

// TODO: Support destroying classes that derive from abstract types.
fn TestDerivedFromAbstract(x: DerivedFromAbstract) {
  // CHECK:STDERR: fail_todo_class_derived_from_abstract.carbon:[[@LINE+4]]:3-31: error: cannot convert type `DerivedFromAbstract` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_todo_class_derived_from_abstract.carbon:[[@LINE-6]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- const.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestConst(x: const i32) {
  CheckHasTrivialDestruction(x);
}

// --- todo_maybe_unformed.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

// TODO: Implement destruction for `Core.MaybeUnformed`.
fn TestMaybeUnformed(x: Core.MaybeUnformed(i32)) {
  CheckHasTrivialDestruction(x);
}

// --- struct.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

//@dump-sem-ir-begin
fn TestEmptyStruct() {
  CheckHasTrivialDestruction({});
}
//@dump-sem-ir-end

fn TestStruct(x: {.a: i32, .b: bool}) {
  CheckHasTrivialDestruction(x);
}

fn TestStructWithNestedTrivialStruct(x: {.a: {.b: i32}}) {
  CheckHasTrivialDestruction(x);
}

// --- tuple.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

//@dump-sem-ir-begin
fn TestEmptyTuple(x: ()) {
  CheckHasTrivialDestruction(x);
}
//@dump-sem-ir-end

fn TestTuple(x: (i32, bool)) {
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithNestedTrivialTuple(x: (i32, (bool, i32))) {
  CheckHasTrivialDestruction(x);
}

// CHECK:STDOUT: --- class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %EmptyClass: type = class_type @EmptyClass [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.dc9: type = pattern_type %EmptyClass [concrete]
// CHECK:STDOUT:   %x.param_patt.622: %pattern_type.dc9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.a86: %pattern_type.dc9 = wrapper_binding_pattern x, %x.param_patt.622 [concrete]
// CHECK:STDOUT:   %TestEmptyClass.type: type = fn_type @TestEmptyClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestEmptyClass: %TestEmptyClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.type: type = fn_type @CheckHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_31.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5c8: %SubobjectDestroy.type = facet_value %EmptyClass, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ec0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5c8) [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet.d78: %TrivialDestroy.type = facet_value %EmptyClass, (%custom_witness.447, %custom_witness.b80ef6.2, %DestroyImpl.impl_witness.ec0) [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn.6bc: <specific function> = specific_function %CheckHasTrivialDestruction, @CheckHasTrivialDestruction(%TrivialDestroy.facet.d78) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = import_ref Main//check_destroy_format, CheckHasTrivialDestruction, loaded [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %Main.EmptyClass: type = import_ref Main//types, EmptyClass, loaded [concrete = constants.%EmptyClass]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TestEmptyClass.decl: %TestEmptyClass.type = fn_decl @TestEmptyClass [concrete = constants.%TestEmptyClass] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.dc9 = value_param_pattern [concrete = constants.%x.param_patt.622]
// CHECK:STDOUT:     %x.patt: %pattern_type.dc9 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.a86]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %EmptyClass = value_param call_param0
// CHECK:STDOUT:     %EmptyClass.ref: type = name_ref EmptyClass, imports.%Main.EmptyClass [concrete = constants.%EmptyClass]
// CHECK:STDOUT:     %x: %EmptyClass = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyClass(%x.param: %EmptyClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckHasTrivialDestruction.ref: %CheckHasTrivialDestruction.type = name_ref CheckHasTrivialDestruction, imports.%Main.CheckHasTrivialDestruction [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %x.ref: %EmptyClass = name_ref x, %x
// CHECK:STDOUT:   %TrivialDestroy.facet.loc9_31.1: %TrivialDestroy.type = facet_value constants.%EmptyClass, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.ec0) [concrete = constants.%TrivialDestroy.facet.d78]
// CHECK:STDOUT:   %.loc9_31.1: %TrivialDestroy.type = converted constants.%EmptyClass, %TrivialDestroy.facet.loc9_31.1 [concrete = constants.%TrivialDestroy.facet.d78]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc9_31.2: %TrivialDestroy.type = facet_value constants.%EmptyClass, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.ec0) [concrete = constants.%TrivialDestroy.facet.d78]
// CHECK:STDOUT:   %.loc9_31.2: %TrivialDestroy.type = converted constants.%EmptyClass, %TrivialDestroy.facet.loc9_31.2 [concrete = constants.%TrivialDestroy.facet.d78]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction.ref, @CheckHasTrivialDestruction(constants.%TrivialDestroy.facet.d78) [concrete = constants.%CheckHasTrivialDestruction.specific_fn.6bc]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.call: init %empty_tuple.type = call %CheckHasTrivialDestruction.specific_fn(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_31.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_31.2(%self.param: ref %EmptyClass) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TestEmptyStruct.type: type = fn_type @TestEmptyStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestEmptyStruct: %TestEmptyStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.type: type = fn_type @CheckHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet.609: %TrivialDestroy.type = facet_value %empty_struct_type, (%custom_witness.447, %custom_witness.b80ef6.1, %DestroyImpl.impl_witness.011) [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn.30e: <specific function> = specific_function %CheckHasTrivialDestruction, @CheckHasTrivialDestruction(%TrivialDestroy.facet.609) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = import_ref Main//check_destroy_format, CheckHasTrivialDestruction, loaded [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TestEmptyStruct.decl: %TestEmptyStruct.type = fn_decl @TestEmptyStruct [concrete = constants.%TestEmptyStruct] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckHasTrivialDestruction.ref: %CheckHasTrivialDestruction.type = name_ref CheckHasTrivialDestruction, imports.%Main.CheckHasTrivialDestruction [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %.loc8_31.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_32.1: %TrivialDestroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%TrivialDestroy.facet.609]
// CHECK:STDOUT:   %.loc8_32.1: %TrivialDestroy.type = converted constants.%empty_struct_type, %TrivialDestroy.facet.loc8_32.1 [concrete = constants.%TrivialDestroy.facet.609]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_32.2: %TrivialDestroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%TrivialDestroy.facet.609]
// CHECK:STDOUT:   %.loc8_32.2: %TrivialDestroy.type = converted constants.%empty_struct_type, %TrivialDestroy.facet.loc8_32.2 [concrete = constants.%TrivialDestroy.facet.609]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction.ref, @CheckHasTrivialDestruction(constants.%TrivialDestroy.facet.609) [concrete = constants.%CheckHasTrivialDestruction.specific_fn.30e]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_31.2: %empty_struct_type = converted %.loc8_31.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.call: init %empty_tuple.type = call %CheckHasTrivialDestruction.specific_fn(%.loc8_31.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt.334: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.218: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt.334 [concrete]
// CHECK:STDOUT:   %TestEmptyTuple.type: type = fn_type @TestEmptyTuple [concrete]
// CHECK:STDOUT:   %TestEmptyTuple: %TestEmptyTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.type: type = fn_type @CheckHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet.3bb: %TrivialDestroy.type = facet_value %empty_tuple.type, (%custom_witness.447, %custom_witness.b80ef6.1, %DestroyImpl.impl_witness.674) [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn.6e1: <specific function> = specific_function %CheckHasTrivialDestruction, @CheckHasTrivialDestruction(%TrivialDestroy.facet.3bb) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = import_ref Main//check_destroy_format, CheckHasTrivialDestruction, loaded [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TestEmptyTuple.decl: %TestEmptyTuple.type = fn_decl @TestEmptyTuple [concrete = constants.%TestEmptyTuple] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%x.param_patt.334]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.218]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc7_23.1: type = splice_block %.loc7_23.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc7_23.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc7_23.3: type = converted %.loc7_23.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyTuple(%x.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckHasTrivialDestruction.ref: %CheckHasTrivialDestruction.type = name_ref CheckHasTrivialDestruction, imports.%Main.CheckHasTrivialDestruction [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_31.1: %TrivialDestroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.674) [concrete = constants.%TrivialDestroy.facet.3bb]
// CHECK:STDOUT:   %.loc8_31.1: %TrivialDestroy.type = converted constants.%empty_tuple.type, %TrivialDestroy.facet.loc8_31.1 [concrete = constants.%TrivialDestroy.facet.3bb]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_31.2: %TrivialDestroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.674) [concrete = constants.%TrivialDestroy.facet.3bb]
// CHECK:STDOUT:   %.loc8_31.2: %TrivialDestroy.type = converted constants.%empty_tuple.type, %TrivialDestroy.facet.loc8_31.2 [concrete = constants.%TrivialDestroy.facet.3bb]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction.ref, @CheckHasTrivialDestruction(constants.%TrivialDestroy.facet.3bb) [concrete = constants.%CheckHasTrivialDestruction.specific_fn.6e1]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.call: init %empty_tuple.type = call %CheckHasTrivialDestruction.specific_fn(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
