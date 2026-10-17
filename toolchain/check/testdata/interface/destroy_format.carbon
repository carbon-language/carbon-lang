// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy_format.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy_format.carbon

// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a type can be destroyed.
//
// Use this function to confirm `DestroyFormat::NonTrivial` (pass) and
// `DestroyFormat::NoDestroy` (fail).
fn CheckHasDestruction[T: Core.Destroy](_: T) {}

// Identifies whether a type has trivial destruction.
//
// Use this function to confirm `DestroyFormat::Trivial` (pass),
// `DestroyFormat::NonTrivial` (fail), and `DestroyFormat::NoDestroy`
// (fail).
//
// `CheckHasTrivialDestruction` should only be combined with
// `CheckHasDestruction` for types that are expected to not have trivial
// destruction. Using both on a trivially destroyable type (e.g. `i32`) is
// redundant.
fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}

fn CheckTypeHasDestruction(unused T: Core.Destroy) {}
fn CheckTypeHasTrivialDestruction(unused T: Core.TrivialDestroy) {}

// --- empty_struct.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

//@dump-sem-ir-begin
fn Test() {
  CheckHasTrivialDestruction({});
}
//@dump-sem-ir-end

// --- empty_tuple.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

//@dump-sem-ir-begin
fn Test(x: ()) {
  CheckHasTrivialDestruction(x);
}
//@dump-sem-ir-end

// --- empty_class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

//@dump-sem-ir-begin
class C {}

fn Test(x: C) {
  CheckHasTrivialDestruction(x);
}
//@dump-sem-ir-end

// --- fail_destroy_custom_impl.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class C {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test(x: C) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_destroy_custom_impl.carbon:[[@LINE+8]]:3: error: cannot convert type `C` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_destroy_custom_impl.carbon:[[@LINE-14]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_non_trivial_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class Member {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class C {
  var member: Member;
}

fn Test(x: C) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_non_trivial_member.carbon:[[@LINE+8]]:3: error: cannot convert type `C` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_non_trivial_member.carbon:[[@LINE-18]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_non_trivial_base.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

base class Base {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class C {
  extend base: Base;
}

fn Test(x: C) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_non_trivial_base.carbon:[[@LINE+8]]:3: error: cannot convert type `C` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_non_trivial_base.carbon:[[@LINE-18]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- no_destroy.h

struct DeletedDestructor {
  ~DeletedDestructor() = delete;
};

struct ProtectedDestructor {
protected:
  ~ProtectedDestructor() = default;
};

struct PrivateDestructor {
private:
  ~PrivateDestructor() = default;
};

// --- fail_deleted_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

fn Test(x: Cpp.DeletedDestructor) {
  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE+20]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    28 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./no_destroy.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE-29]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_protected_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

fn Test(x: Cpp.ProtectedDestructor) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- fail_private_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

fn Test(x: Cpp.PrivateDestructor) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- function_value.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn G() {}

fn Test() {
  CheckHasTrivialDestruction(G);
}

// --- function_with_self_value.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn Test() {
  CheckHasTrivialDestruction((i32 as Core.Copy).Op);
}

// --- fail_unbound_element_value.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class C {
  var x: i32;
}

fn Test() {
  // CHECK:STDERR: fail_unbound_element_value.carbon:[[@LINE+8]]:3: error: cannot convert type `<unbound element of class C>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(C.x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_unbound_element_value.carbon:[[@LINE-10]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(C.x);
}

// --- fail_todo_associated_entity_value.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

interface I {
  fn F();
}

fn Test() {
  // CHECK:STDERR: fail_todo_associated_entity_value.carbon:[[@LINE+16]]:3: error: semantics TODO: `unhandled type AssociatedEntityType` [SemanticsTodo]
  // CHECK:STDERR:   CheckHasTrivialDestruction(I.F);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_associated_entity_value.carbon:[[@LINE-10]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_associated_entity_value.carbon:[[@LINE+8]]:3: error: cannot convert type `<associated entity in I>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(I.F);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_associated_entity_value.carbon:[[@LINE-18]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(I.F);
}

// --- fail_todo_generic_class_value.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class Generic(T: type) {}

fn Test() {
  // CHECK:STDERR: fail_todo_generic_class_value.carbon:[[@LINE+16]]:3: error: semantics TODO: `unhandled type ParameterizedClassType` [SemanticsTodo]
  // CHECK:STDERR:   CheckHasTrivialDestruction(Generic);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_class_value.carbon:[[@LINE-8]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_generic_class_value.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Generic>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(Generic);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_class_value.carbon:[[@LINE-16]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(Generic);
}

// --- fail_todo_generic_interface_value.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

interface GenericInterface(T: type) {}

fn Test() {
  // CHECK:STDERR: fail_todo_generic_interface_value.carbon:[[@LINE+16]]:3: error: semantics TODO: `unhandled type GenericInterfaceType` [SemanticsTodo]
  // CHECK:STDERR:   CheckHasTrivialDestruction(GenericInterface);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_interface_value.carbon:[[@LINE-8]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_generic_interface_value.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of GenericInterface>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(GenericInterface);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_interface_value.carbon:[[@LINE-16]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(GenericInterface);
}

// --- fail_todo_generic_named_constraint_value.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

constraint GenericConstraint(T: type) {}

fn Test() {
  // CHECK:STDERR: fail_todo_generic_named_constraint_value.carbon:[[@LINE+16]]:3: error: semantics TODO: `unhandled type GenericNamedConstraintType` [SemanticsTodo]
  // CHECK:STDERR:   CheckHasTrivialDestruction(GenericConstraint);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_named_constraint_value.carbon:[[@LINE-8]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_generic_named_constraint_value.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of GenericConstraint>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(GenericConstraint);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_named_constraint_value.carbon:[[@LINE-16]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(GenericConstraint);
}

// --- fail_todo_namespace_value.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

namespace N;

fn Test() {
  // CHECK:STDERR: fail_todo_namespace_value.carbon:[[@LINE+16]]:3: error: semantics TODO: `unhandled type NamespaceType` [SemanticsTodo]
  // CHECK:STDERR:   CheckHasTrivialDestruction(N);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_namespace_value.carbon:[[@LINE-8]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_namespace_value.carbon:[[@LINE+8]]:3: error: cannot convert type `<namespace>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(N);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_namespace_value.carbon:[[@LINE-16]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(N);
}

// --- float_literal.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn Test() {
  CheckTypeHasTrivialDestruction(Core.FloatLiteral);
}

// --- char_literal.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn Test() {
  CheckTypeHasTrivialDestruction(Core.CharLiteral);
}

// --- unhandled.h

void Function();

template <typename T>
struct Template {};

// --- fail_todo_cpp_overload_set_value.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "unhandled.h";

fn Test() {
  // CHECK:STDERR: fail_todo_cpp_overload_set_value.carbon:[[@LINE+16]]:3: error: semantics TODO: `unhandled type CppOverloadSetType` [SemanticsTodo]
  // CHECK:STDERR:   CheckHasTrivialDestruction(Cpp.Function);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_cpp_overload_set_value.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_cpp_overload_set_value.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Cpp.Function>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(Cpp.Function);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_cpp_overload_set_value.carbon:[[@LINE-15]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(Cpp.Function);
}

// --- fail_cpp_template_name_value.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "unhandled.h";

fn Test() {
  // CHECK:STDERR: fail_cpp_template_name_value.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Cpp.Template>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(Cpp.Template);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_template_name_value.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:20:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(Cpp.Template);
}

// CHECK:STDOUT: --- empty_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.type: type = fn_type @CheckHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet: %TrivialDestroy.type = facet_value %empty_struct_type, (%custom_witness.447, %custom_witness.b80, %DestroyImpl.impl_witness.011) [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction, @CheckHasTrivialDestruction(%TrivialDestroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = import_ref Main//check_destroy_format, CheckHasTrivialDestruction, loaded [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckHasTrivialDestruction.ref: %CheckHasTrivialDestruction.type = name_ref CheckHasTrivialDestruction, imports.%Main.CheckHasTrivialDestruction [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %.loc8_31.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_32.1: %TrivialDestroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.447, constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %.loc8_32.1: %TrivialDestroy.type = converted constants.%empty_struct_type, %TrivialDestroy.facet.loc8_32.1 [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_32.2: %TrivialDestroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.447, constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %.loc8_32.2: %TrivialDestroy.type = converted constants.%empty_struct_type, %TrivialDestroy.facet.loc8_32.2 [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction.ref, @CheckHasTrivialDestruction(constants.%TrivialDestroy.facet) [concrete = constants.%CheckHasTrivialDestruction.specific_fn]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_31.2: %empty_struct_type = converted %.loc8_31.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.call: init %empty_tuple.type = call %CheckHasTrivialDestruction.specific_fn(%.loc8_31.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- empty_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.type: type = fn_type @CheckHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet: %TrivialDestroy.type = facet_value %empty_tuple.type, (%custom_witness.447, %custom_witness.b80, %DestroyImpl.impl_witness.674) [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction, @CheckHasTrivialDestruction(%TrivialDestroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = import_ref Main//check_destroy_format, CheckHasTrivialDestruction, loaded [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc7_13.1: type = splice_block %.loc7_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc7_13.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc7_13.3: type = converted %.loc7_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%x.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckHasTrivialDestruction.ref: %CheckHasTrivialDestruction.type = name_ref CheckHasTrivialDestruction, imports.%Main.CheckHasTrivialDestruction [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_31.1: %TrivialDestroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.447, constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.674) [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %.loc8_31.1: %TrivialDestroy.type = converted constants.%empty_tuple.type, %TrivialDestroy.facet.loc8_31.1 [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc8_31.2: %TrivialDestroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.447, constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.674) [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %.loc8_31.2: %TrivialDestroy.type = converted constants.%empty_tuple.type, %TrivialDestroy.facet.loc8_31.2 [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction.ref, @CheckHasTrivialDestruction(constants.%TrivialDestroy.facet) [concrete = constants.%CheckHasTrivialDestruction.specific_fn]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.call: init %empty_tuple.type = call %CheckHasTrivialDestruction.specific_fn(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- empty_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.type: type = fn_type @CheckHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_31.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.978: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.978) [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet: %TrivialDestroy.type = facet_value %C, (%custom_witness.447, %custom_witness.b80ef6.2, %DestroyImpl.impl_witness.1e5) [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction, @CheckHasTrivialDestruction(%TrivialDestroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = import_ref Main//check_destroy_format, CheckHasTrivialDestruction, loaded [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%x.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckHasTrivialDestruction.ref: %CheckHasTrivialDestruction.type = name_ref CheckHasTrivialDestruction, imports.%Main.CheckHasTrivialDestruction [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %x.ref: %C = name_ref x, %x
// CHECK:STDOUT:   %TrivialDestroy.facet.loc10_31.1: %TrivialDestroy.type = facet_value constants.%C, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.1e5) [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %.loc10_31.1: %TrivialDestroy.type = converted constants.%C, %TrivialDestroy.facet.loc10_31.1 [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc10_31.2: %TrivialDestroy.type = facet_value constants.%C, (constants.%custom_witness.447, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.1e5) [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %.loc10_31.2: %TrivialDestroy.type = converted constants.%C, %TrivialDestroy.facet.loc10_31.2 [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction.ref, @CheckHasTrivialDestruction(constants.%TrivialDestroy.facet) [concrete = constants.%CheckHasTrivialDestruction.specific_fn]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.call: init %empty_tuple.type = call %CheckHasTrivialDestruction.specific_fn(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_31.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_31.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
