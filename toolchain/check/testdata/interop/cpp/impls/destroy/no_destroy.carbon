// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy/cpp_no_destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy/cpp_no_destroy.carbon

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
  // CHECK:STDERR: fail_cpp_overload_set.carbon:[[@LINE+4]]:3-35: error: cannot convert type `<type of Cpp.Function>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_cpp_overload_set.carbon:[[@LINE-5]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(Cpp.Function);

  // CHECK:STDERR: fail_cpp_overload_set.carbon:[[@LINE+4]]:3-42: error: cannot convert type `<type of Cpp.Function>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_cpp_overload_set.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(Cpp.Function);
}

// --- fail_cpp_template_name.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestCppTemplateName() {
  // CHECK:STDERR: fail_cpp_template_name.carbon:[[@LINE+4]]:3-35: error: cannot convert type `<type of Cpp.Template>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_cpp_template_name.carbon:[[@LINE-5]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(Cpp.Template);

  // CHECK:STDERR: fail_cpp_template_name.carbon:[[@LINE+4]]:3-42: error: cannot convert type `<type of Cpp.Template>` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_cpp_template_name.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(Cpp.Template);
}

// --- fail_array.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestArrayOfDeletedDestructor(x: array(Cpp.DeletedDestructor, 2)) {
  // CHECK:STDERR: fail_array.carbon:[[@LINE+10]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_array.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_array.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_array.carbon:[[@LINE+4]]:3-24: error: cannot convert type `array(Cpp.DeletedDestructor, 2)` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_array.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_array.carbon:[[@LINE+4]]:3-31: error: cannot convert type `array(Cpp.DeletedDestructor, 2)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_array.carbon:[[@LINE-17]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestArrayOfArrayOfDeletedDestructor(
    x: array(array(Cpp.DeletedDestructor, 2), 3)) {
  // CHECK:STDERR: fail_array.carbon:[[@LINE+4]]:3-24: error: cannot convert type `array(array(Cpp.DeletedDestructor, 2), 3)` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_array.carbon:[[@LINE-26]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_array.carbon:[[@LINE+4]]:3-31: error: cannot convert type `array(array(Cpp.DeletedDestructor, 2), 3)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_array.carbon:[[@LINE-32]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_class.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "types.h";

fn TestDeletedDestructor(x: Cpp.DeletedDestructor) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+10]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_class.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-24: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-12]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-18]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestHasDeletedMember(x: Cpp.HasDeletedMember) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+11]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_class.carbon:[[@LINE-24]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:18:21: note: destructor of 'HasDeletedMember' is implicitly deleted because field 'd' has a deleted destructor [CppInteropParseNote]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-29]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-24: error: cannot convert type `Cpp.HasDeletedMember` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-33]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `Cpp.HasDeletedMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-39]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDeletedDestructorField(x: DeletedDestructorField) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-24: error: cannot convert type `DeletedDestructorField` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-47]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `DeletedDestructorField` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-53]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestNestedDeletedDestructorMember(x: NestedDeletedDestructorMember) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-24: error: cannot convert type `NestedDeletedDestructorMember` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-61]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `NestedDeletedDestructorMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-67]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDeletedDestructorAndTrivialMembers(
    x: DeletedDestructorAndTrivialMembers) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-24: error: cannot convert type `DeletedDestructorAndTrivialMembers` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-76]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `DeletedDestructorAndTrivialMembers` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-82]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTrivialAndDeletedDestructorMembers(
    x: TrivialAndDeletedDestructorMembers) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-24: error: cannot convert type `TrivialAndDeletedDestructorMembers` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-91]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `TrivialAndDeletedDestructorMembers` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-97]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestDeletedDestructorBaseAndTrivialMember(
    x: DeletedDestructorBaseAndTrivialMember) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-24: error: cannot convert type `DeletedDestructorBaseAndTrivialMember` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-106]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `DeletedDestructorBaseAndTrivialMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-112]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTrivialBaseAndDeletedDestructorMember(
    x: TrivialBaseAndDeletedDestructorMember) {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-24: error: cannot convert type `TrivialBaseAndDeletedDestructorMember` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-121]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:3-31: error: cannot convert type `TrivialBaseAndDeletedDestructorMember` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_class.carbon:[[@LINE-127]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
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
  // CHECK:STDERR: fail_todo_class_adapter.carbon:[[@LINE+10]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_todo_class_adapter.carbon:[[@LINE-6]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_todo_class_adapter.carbon:[[@LINE-10]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_class_adapter.carbon:[[@LINE+4]]:3-24: error: cannot convert type `AdaptDeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_todo_class_adapter.carbon:[[@LINE-14]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_todo_class_adapter.carbon:[[@LINE+4]]:3-31: error: cannot convert type `AdaptDeletedDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_todo_class_adapter.carbon:[[@LINE-20]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_todo_class_cpp_carbon_cpp.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: imported from: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+5]]:1 [InImport]
// CHECK:STDERR: types.carbon:46:1-36: error: cannot convert type `CarbonHoldsDeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
// CHECK:STDERR: {{.*}}/prelude/destroy.carbon:35:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:
// CHECK:STDERR: types.carbon:46:36: error: attempt to use a deleted function [CppInteropParseError]
import library "check_destroy_format";
import library "types";
// CHECK:STDERR: included from: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
// CHECK:STDERR: {{.*}}/prelude/destroy.carbon:35:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:
import Cpp library "types.h";

// A C++ type that holds a Carbon type that holds a C++ type.
inline Cpp '''
struct CppHoldsCarbonHoldsDeletedDestructor {
  Carbon::CarbonHoldsDeletedDestructor c;
};
''';

// TODO: `CheckHasDestruction` should fail too, because
// `CarbonHoldsDeletedDestructor` can't be destroyed. This should be
// determined by reverse-interop, not Carbon-C++ interop because C++
// needs to be able to tell that its types are not destructible.
fn TestCppHoldsCarbonHoldsDeletedDestructor(
    x: Cpp.CppHoldsCarbonHoldsDeletedDestructor) {
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+4]]:3-31: error: cannot convert type `Cpp.CppHoldsCarbonHoldsDeletedDestructor` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE-24]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_struct.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestStructWithDeletedDestructor(x: {.d: Cpp.DeletedDestructor}) {
  // CHECK:STDERR: fail_struct.carbon:[[@LINE+10]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_struct.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-24: error: cannot convert type `{.d: Cpp.DeletedDestructor}` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-31: error: cannot convert type `{.d: Cpp.DeletedDestructor}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-17]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithNestedDeletedDestructor(x: {.a: {.d: Cpp.DeletedDestructor}}) {
  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-24: error: cannot convert type `{.a: {.d: Cpp.DeletedDestructor}}` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-25]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-31: error: cannot convert type `{.a: {.d: Cpp.DeletedDestructor}}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-31]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithTrivialAndDeletedDestructor(
    x: {.a: i32, .d: Cpp.DeletedDestructor}) {
  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-24: error: cannot convert type `{.a: i32, .d: Cpp.DeletedDestructor}` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-40]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-31: error: cannot convert type `{.a: i32, .d: Cpp.DeletedDestructor}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-46]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestStructWithDeletedDestructorAndTrivial(
    x: {.d: Cpp.DeletedDestructor, .a: i32}) {
  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-24: error: cannot convert type `{.d: Cpp.DeletedDestructor, .a: i32}` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-55]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_struct.carbon:[[@LINE+4]]:3-31: error: cannot convert type `{.d: Cpp.DeletedDestructor, .a: i32}` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_struct.carbon:[[@LINE-61]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

// --- fail_tuple.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "types.h";

fn TestTupleWithDeletedDestructor(x: (Cpp.DeletedDestructor,)) {
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+10]]:24: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_tuple.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-24: error: cannot convert type `(Cpp.DeletedDestructor,)` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-31: error: cannot convert type `(Cpp.DeletedDestructor,)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-17]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithNestedDeletedDestructor(x: ((Cpp.DeletedDestructor,),)) {
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-24: error: cannot convert type `((Cpp.DeletedDestructor,),)` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-25]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-31: error: cannot convert type `((Cpp.DeletedDestructor,),)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-31]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithTrivialAndDeletedDestructor(x: (i32, Cpp.DeletedDestructor)) {
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-24: error: cannot convert type `(i32, Cpp.DeletedDestructor)` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-39]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-31: error: cannot convert type `(i32, Cpp.DeletedDestructor)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-45]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}

fn TestTupleWithDeletedDestructorAndTrivial(x: (Cpp.DeletedDestructor, i32)) {
  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-24: error: cannot convert type `(Cpp.DeletedDestructor, i32)` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-53]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:1-47: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasDestruction(x);

  // CHECK:STDERR: fail_tuple.carbon:[[@LINE+4]]:3-31: error: cannot convert type `(Cpp.DeletedDestructor, i32)` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_tuple.carbon:[[@LINE-59]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:1-61: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
}
