// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy/cpp_no_destroy_objects.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy/cpp_no_destroy_objects.carbon

// --- types.h

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

// The destructor is implicitly deleted, because the member's is deleted.
struct HasDeletedMember {
  DeletedDestructor d;
};

void Function();

template <typename T>
struct Template {};

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

import Cpp library "types.h";

// Types used by the splits below.

base class TrivialBase {}

class DeletedDestructorField {
  var d: Cpp.DeletedDestructor;
}

class NestedDeletedDestructorMember {
  var inner: DeletedDestructorField;
}

class DeletedDestructorAndTrivialMembers {
  var d: Cpp.DeletedDestructor;
  var a: i32;
}

class TrivialAndDeletedDestructorMembers {
  var a: i32;
  var d: Cpp.DeletedDestructor;
}

base class DeletedDestructorFieldBase {
  var d: Cpp.DeletedDestructor;
}

class DeletedDestructorBaseAndTrivialMember {
  extend base: DeletedDestructorFieldBase;
  var a: i32;
}

class TrivialBaseAndDeletedDestructorMember {
  extend base: TrivialBase;
  var d: Cpp.DeletedDestructor;
}

class AdaptDeletedDestructor {
  adapt Cpp.DeletedDestructor;
}

class CarbonHoldsDeletedDestructor {
  var d: Cpp.DeletedDestructor;
}

// --- fail_cpp_overload_set.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestCppOverloadSet() {
  // CHECK:STDERR: fail_cpp_overload_set.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Cpp.Function>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(Cpp.Function);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_overload_set.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(Cpp.Function);

  // CHECK:STDERR: fail_cpp_overload_set.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Cpp.Function>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(Cpp.Function);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_overload_set.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(Cpp.Function);
}

// --- fail_cpp_template_name.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestCppTemplateName() {
  // CHECK:STDERR: fail_cpp_template_name.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Cpp.Template>` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(Cpp.Template);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_template_name.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(Cpp.Template);

  // CHECK:STDERR: fail_cpp_template_name.carbon:[[@LINE+8]]:3: error: cannot convert type `<type of Cpp.Template>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(Cpp.Template);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cpp_template_name.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(Cpp.Template);
}

// --- fail_array.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestArrayOfDeletedDestructor(x: array(Cpp.DeletedDestructor, 2)) {
  // CHECK:STDERR: fail_array.carbon:[[@LINE+20]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    28 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_array.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_array.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_array.carbon:[[@LINE+8]]:3: error: cannot convert type `array(Cpp.DeletedDestructor, 2)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_array.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_array.carbon:[[@LINE+8]]:3: error: cannot convert type `array(Cpp.DeletedDestructor, 2)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_array.carbon:[[@LINE-29]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestArrayOfArrayOfDeletedDestructor(
    x: array(array(Cpp.DeletedDestructor, 2), 3)) {
  // CHECK:STDERR: fail_array.carbon:[[@LINE+8]]:3: error: cannot convert type `array(array(Cpp.DeletedDestructor, 2), 3)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_array.carbon:[[@LINE-42]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_array.carbon:[[@LINE+8]]:3: error: cannot convert type `array(array(Cpp.DeletedDestructor, 2), 3)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_array.carbon:[[@LINE-52]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "types.h";

fn TestDeletedDestructor(x: Cpp.DeletedDestructor) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+20]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    29 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_class.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_class.carbon:[[@LINE-12]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-20]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-30]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestHasDeletedMember(x: Cpp.HasDeletedMember) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+24]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    67 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_class.carbon:[[@LINE-40]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:18:21: note: destructor of 'HasDeletedMember' is implicitly deleted because field 'd' has a deleted destructor [CppInteropParseNote]
  // CHECK:STDERR:    18 |   DeletedDestructor d;
  // CHECK:STDERR:       |                     ^
  // CHECK:STDERR: fail_class.carbon:[[@LINE-44]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_class.carbon:[[@LINE-50]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.HasDeletedMember` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-58]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.HasDeletedMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-68]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDeletedDestructorField(x: DeletedDestructorField) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorField` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-80]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorField` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-90]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestNestedDeletedDestructorMember(x: NestedDeletedDestructorMember) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `NestedDeletedDestructorMember` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-102]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `NestedDeletedDestructorMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-112]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDeletedDestructorAndTrivialMembers(
    x: DeletedDestructorAndTrivialMembers) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorAndTrivialMembers` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-125]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorAndTrivialMembers` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-135]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTrivialAndDeletedDestructorMembers(
    x: TrivialAndDeletedDestructorMembers) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `TrivialAndDeletedDestructorMembers` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-148]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `TrivialAndDeletedDestructorMembers` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-158]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDeletedDestructorBaseAndTrivialMember(
    x: DeletedDestructorBaseAndTrivialMember) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorBaseAndTrivialMember` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-171]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `DeletedDestructorBaseAndTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-181]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTrivialBaseAndDeletedDestructorMember(
    x: TrivialBaseAndDeletedDestructorMember) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `TrivialBaseAndDeletedDestructorMember` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-194]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:3: error: cannot convert type `TrivialBaseAndDeletedDestructorMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_class.carbon:[[@LINE-204]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- todo_fail_class_inaccessible.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

// TODO: `Core.Destroy` should fail, because the destructor isn't public.
fn TestProtectedDestructor(x: Cpp.ProtectedDestructor) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// TODO: `Core.Destroy` should fail, because the destructor isn't public.
fn TestPrivateDestructor(x: Cpp.PrivateDestructor) {
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- fail_todo_class_adapter.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "types.h";

// TODO: Both checks should fail. Today only the C++ deleted-destructor error is
// reported.
fn TestAdaptDeletedDestructor(x: AdaptDeletedDestructor) {
  // CHECK:STDERR: fail_todo_class_adapter.carbon:[[@LINE+12]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    23 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_todo_class_adapter.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_todo_class_adapter.carbon:[[@LINE-14]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  CheckHasTrivialDestruction(x);
}

// --- fail_todo_class_cpp_carbon_cpp.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+11]]:1: in import [InImport]
// CHECK:STDERR: types.carbon:46:1: error: cannot convert type `CarbonHoldsDeletedDestructor` into type implementing `Core.DestroyImpl & Core.SubobjectDestroy` [ConversionFailureTypeToFacet]
// CHECK:STDERR: class CarbonHoldsDeletedDestructor {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/destroy.carbon:33:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn SelfDestruct[T: Destroy](ref t: T) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: types.carbon:46:36: error: attempt to use a deleted function [CppInteropParseError]
// CHECK:STDERR:    46 | class CarbonHoldsDeletedDestructor {
// CHECK:STDERR:       |                                    ^
import library "check_destroy_format";
import library "types";
// CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
// CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
// CHECK:STDERR:       |   ^
// CHECK:STDERR: {{.*}}/prelude/destroy.carbon:33:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn SelfDestruct[T: Destroy](ref t: T) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "types.h";

// A C++ type that holds a Carbon type that holds a C++ type.
inline Cpp '''
struct CppHoldsCarbonHoldsDeletedDestructor {
  Carbon::CarbonHoldsDeletedDestructor c;
};
''';

// TODO: `CheckHasDestruction` should fail too, because
// `CarbonHoldsDeletedDestructor` can't be destroyed. Today it passes; the only
// errors are at the definition of `CarbonHoldsDeletedDestructor` and from
// `CheckHasTrivialDestruction`.
fn TestCppHoldsCarbonHoldsDeletedDestructor(
    x: Cpp.CppHoldsCarbonHoldsDeletedDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.CppHoldsCarbonHoldsDeletedDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE-30]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_struct.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestStructWithDeletedDestructor(x: {.d: Cpp.DeletedDestructor}) {
  // CHECK:STDERR: fail_struct.carbon:[[@LINE+20]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    28 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_struct.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_struct.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_struct.carbon:[[@LINE+8]]:3: error: cannot convert type `{.d: Cpp.DeletedDestructor}` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+8]]:3: error: cannot convert type `{.d: Cpp.DeletedDestructor}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct.carbon:[[@LINE-29]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithNestedDeletedDestructor(x: {.a: {.d: Cpp.DeletedDestructor}}) {
  // CHECK:STDERR: fail_struct.carbon:[[@LINE+8]]:3: error: cannot convert type `{.a: {.d: Cpp.DeletedDestructor}}` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct.carbon:[[@LINE-41]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+8]]:3: error: cannot convert type `{.a: {.d: Cpp.DeletedDestructor}}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct.carbon:[[@LINE-51]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithTrivialAndDeletedDestructor(
    x: {.a: i32, .d: Cpp.DeletedDestructor}) {
  // CHECK:STDERR: fail_struct.carbon:[[@LINE+8]]:3: error: cannot convert type `{.a: i32, .d: Cpp.DeletedDestructor}` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct.carbon:[[@LINE-64]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+8]]:3: error: cannot convert type `{.a: i32, .d: Cpp.DeletedDestructor}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct.carbon:[[@LINE-74]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithDeletedDestructorAndTrivial(
    x: {.d: Cpp.DeletedDestructor, .a: i32}) {
  // CHECK:STDERR: fail_struct.carbon:[[@LINE+8]]:3: error: cannot convert type `{.d: Cpp.DeletedDestructor, .a: i32}` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct.carbon:[[@LINE-87]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+8]]:3: error: cannot convert type `{.d: Cpp.DeletedDestructor, .a: i32}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct.carbon:[[@LINE-97]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_tuple.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestTupleWithDeletedDestructor(x: (Cpp.DeletedDestructor,)) {
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+20]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    28 |   CheckHasDestruction(x);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+8]]:3: error: cannot convert type `(Cpp.DeletedDestructor,)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+8]]:3: error: cannot convert type `(Cpp.DeletedDestructor,)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE-29]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithNestedDeletedDestructor(x: ((Cpp.DeletedDestructor,),)) {
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+8]]:3: error: cannot convert type `((Cpp.DeletedDestructor,),)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE-41]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+8]]:3: error: cannot convert type `((Cpp.DeletedDestructor,),)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE-51]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithTrivialAndDeletedDestructor(x: (i32, Cpp.DeletedDestructor)) {
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+8]]:3: error: cannot convert type `(i32, Cpp.DeletedDestructor)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE-63]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+8]]:3: error: cannot convert type `(i32, Cpp.DeletedDestructor)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE-73]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithDeletedDestructorAndTrivial(x: (Cpp.DeletedDestructor, i32)) {
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+8]]:3: error: cannot convert type `(Cpp.DeletedDestructor, i32)` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE-85]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+8]]:3: error: cannot convert type `(Cpp.DeletedDestructor, i32)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE-95]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](_: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}
