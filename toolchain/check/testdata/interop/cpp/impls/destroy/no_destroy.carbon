// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/destroy/no_destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/destroy/no_destroy.carbon

// --- core.carbon

package Core library "core";

export import library "prelude/parts/array";

// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a vaule's type can be destroyed.
fn CheckImplsDestroy[T: Core.Destroy](_: T) {}

// --- types.carbon

library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- cpp_types.h

struct DeletedDestructor {
  ~DeletedDestructor() = delete;
};

struct DestructorImplicitlyDeleted {
  DeletedDestructor field;
};

// --- fail_cpp_overload_set.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

import Cpp inline '''c++
void Function();
''';

fn TestCppOverloadSet() {
  // CHECK:STDERR: fail_cpp_overload_set.carbon:[[@LINE+4]]:3-33: error: cannot convert type `<type of Cpp.Function>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_cpp_overload_set.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(Cpp.Function);

  // TODO: add CheckImplsTrivialDestroy(Cpp.Function);

  // TODO: add CheckImplsDynamicDestroy(Cpp.Function);
}

// --- fail_cpp_template_name.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

import Cpp inline '''c++
template<class>
struct ClassTemplate {};
''';

fn TestCppTemplateName() {
  // CHECK:STDERR: fail_cpp_template_name.carbon:[[@LINE+4]]:3-38: error: cannot convert type `<type of Cpp.ClassTemplate>` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_cpp_template_name.carbon:[[@LINE-9]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(Cpp.ClassTemplate);

  // TODO: add CheckImplsTrivialDestroy(Cpp.ClassTemplate);

  // TODO: add CheckImplsDynamicDestroy(Cpp.ClassTemplate);
}

// --- fail_array.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

// CHECK:STDERR: fail_array.carbon:[[@LINE+2]]:36-66: error: name `Core.Array` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR:
fn TestArrayOfDeletedDestructor(x: array(Cpp.DeletedDestructor, 2)) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// CHECK:STDERR: fail_array.carbon:[[@LINE+2]]:49-79: error: name `Core.Array` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR:
fn TestArrayOfArrayOfDeletedDestructor(x: array(array(Cpp.DeletedDestructor, 2), 3)) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_cpp_class_with_deleted_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestClassWithDestructorExplicitlyDeleted(x: Cpp.DeletedDestructor) {
  // CHECK:STDERR: fail_cpp_class_with_deleted_destructor.carbon:[[@LINE+10]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_cpp_class_with_deleted_destructor.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_cpp_class_with_deleted_destructor.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_cpp_class_with_deleted_destructor.carbon:[[@LINE+4]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_cpp_class_with_deleted_destructor.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

fn TestClassWithDestructorImplicitlyDeleted(x: Cpp.DestructorImplicitlyDeleted) {
  // CHECK:STDERR: fail_cpp_class_with_deleted_destructor.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_cpp_class_with_deleted_destructor.carbon:[[@LINE-22]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:7:21: note: destructor of 'DestructorImplicitlyDeleted' is implicitly deleted because field 'field' has a deleted destructor [CppInteropParseNote]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_cpp_class_with_deleted_destructor.carbon:[[@LINE-26]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_cpp_class_with_deleted_destructor.carbon:[[@LINE+4]]:3-22: error: cannot convert type `Cpp.DestructorImplicitlyDeleted` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_cpp_class_with_deleted_destructor.carbon:[[@LINE-30]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_carbon_class_with_cpp_field_that_deletes_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class FieldExplicitlyDeletesDestructor {
  var d: Cpp.DeletedDestructor;
}

fn TestDeletedDestructorField(x: FieldExplicitlyDeletesDestructor) {
  // CHECK:STDERR: fail_carbon_class_with_cpp_field_that_deletes_destructor.carbon:[[@LINE+16]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_carbon_class_with_cpp_field_that_deletes_destructor.carbon:[[@LINE-8]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_carbon_class_with_cpp_field_that_deletes_destructor.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_carbon_class_with_cpp_field_that_deletes_destructor.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_carbon_class_with_cpp_field_that_deletes_destructor.carbon:[[@LINE-16]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_carbon_class_with_cpp_field_that_deletes_destructor.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_carbon_class_with_cpp_field_that_deletes_destructor.carbon:[[@LINE-21]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

class FieldImplicitlyDeletesDestructor {
  // CHECK:STDERR: fail_carbon_class_with_cpp_field_that_deletes_destructor.carbon:[[@LINE+2]]:14-40: error: name `DestructorImplicitlyDeleted` not found [NameNotFound]
  // CHECK:STDERR:
  var inner: DestructorImplicitlyDeleted;
}

// CHECK:STDERR: fail_carbon_class_with_cpp_field_that_deletes_destructor.carbon:[[@LINE+2]]:41-69: error: name `NestedDeletedDestructorMember` not found [NameNotFound]
// CHECK:STDERR:
fn TestNestedDeletedDestructorMember(x: NestedDeletedDestructorMember) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

class NoDestroyMemberFirst {
  var a: Cpp.DeletedDestructor;
  var b: HasTrivialDestruction;
}

fn TestNoDestroyMemberFirst(x: NoDestroyMemberFirst) {
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE+16]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE-9]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE-18]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE-23]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

class NoDestroyMemberSecond {
  var a: HasTrivialDestruction;
  var b: Cpp.DeletedDestructor;
}

fn TestNoDestroyMemberSecond(x: NoDestroyMemberSecond) {
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE-41]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE-46]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

class NoDestroyMemberFirst {
  var a: Cpp.DeletedDestructor;
  var b: HasNonTrivialDestruction;
}

fn TestNoDestroyMemberFirst(x: NoDestroyMemberFirst) {
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE+16]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE-9]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE-18]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE-23]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

class NoDestroyMemberSecond {
  var a: HasNonTrivialDestruction;
  var b: Cpp.DeletedDestructor;
}

fn TestNoDestroyMemberSecond(x: NoDestroyMemberSecond) {
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE-41]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE-46]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_deleted_destructor_field.base

// CHECK:STDERR: fail_deleted_destructor_field.base:[[@LINE+2]]:1-35: error: file extension of `.carbon` required for api [IncorrectExtension]
// CHECK:STDERR:
library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

base class DeletedDestructorFieldBase {
  var d: Cpp.DeletedDestructor;
}

class DeletedDestructorBaseAndTrivialMember {
  extend base: DeletedDestructorFieldBase;
  // CHECK:STDERR: fail_deleted_destructor_field.base:[[@LINE+2]]:10-12: error: name `Core.Int` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:
  var a: i32;
}

fn TestDeletedDestructorBaseAndTrivialMember(x: DeletedDestructorBaseAndTrivialMember) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_field_trivial_base_and_deleted_destructor_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

base class TrivialBase {}

class TrivialBaseAndDeletedDestructorMember {
  extend base: TrivialBase;
  var d: Cpp.DeletedDestructor;
}

fn TestTrivialBaseAndDeletedDestructorMember(x: TrivialBaseAndDeletedDestructorMember) {
  // CHECK:STDERR: fail_field_trivial_base_and_deleted_destructor_member.carbon:[[@LINE+16]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_field_trivial_base_and_deleted_destructor_member.carbon:[[@LINE-11]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_field_trivial_base_and_deleted_destructor_member.carbon:[[@LINE-14]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_field_trivial_base_and_deleted_destructor_member.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_field_trivial_base_and_deleted_destructor_member.carbon:[[@LINE-19]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_field_trivial_base_and_deleted_destructor_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_field_trivial_base_and_deleted_destructor_member.carbon:[[@LINE-24]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- todo_fail_class_inaccessible_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp;

inline Cpp '''c++
struct ProtectedDestructor {
protected:
  ~ProtectedDestructor() = default;
};
''';

// TODO: `Core.Destroy` should fail, because the destructor isn't public.
fn TestProtectedDestructor(x: Cpp.ProtectedDestructor) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

inline Cpp '''c++
struct PrivateDestructor {
private:
  ~PrivateDestructor() = default;
};
''';

// TODO: `Core.Destroy` should fail, because the destructor isn't public.
fn TestPrivateDestructor(x: Cpp.PrivateDestructor) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_todo_class_adapter.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class AdaptDeletedDestructor {
  adapt Cpp.DeletedDestructor;
}

fn TestAdaptDeletedDestructor(x: AdaptDeletedDestructor) {
  // CHECK:STDERR: fail_todo_class_adapter.carbon:[[@LINE+6]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_todo_class_adapter.carbon:[[@LINE-8]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_todo_class_adapter.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_todo_class_cpp_carbon_cpp.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

// CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+17]]:36: error: attempt to use a deleted function [CppInteropParseError]
// CHECK:STDERR: included from: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE-3]]:10 [InCppInclude]
// CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
// CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+12]]:1-36: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
// CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+8]]:1-36: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
// CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+4]]:36: error: attempt to use a deleted function [CppInteropParseError]
// CHECK:STDERR: included from: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE-16]]:10 [InCppInclude]
// CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
// CHECK:STDERR:
class CarbonHoldsDeletedDestructor {
  var d: Cpp.DeletedDestructor;
}

// A C++ type that holds a Carbon type that holds a C++ type.
inline Cpp '''
struct BidirectionalInteropClassWithDeletedDestructor {
  Carbon::CarbonHoldsDeletedDestructor c;
};
''';

// TODO: `CheckHasDestruction` should fail too, because
// `CarbonHoldsDeletedDestructor` can't be destroyed. This should be
// determined by reverse-interop, not Carbon-C++ interop because C++
// needs to be able to tell that its types are not destructible.
fn TestBidirectionalInteropClassWithDeletedDestructor(x: Cpp.BidirectionalInteropClassWithDeletedDestructor) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_struct_with_member_that_deletes_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestStructWithDeletedDestructor(x: {.d: Cpp.DeletedDestructor}) {
  // CHECK:STDERR: fail_struct_with_member_that_deletes_destructor.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_struct_with_member_that_deletes_destructor.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_struct_with_member_that_deletes_destructor.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_struct_with_member_that_deletes_destructor.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_struct_with_member_that_deletes_destructor.carbon:[[@LINE-12]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_struct_with_no_destroy_member_and_trivially_destroyable_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

fn TestNoDestroyFirst(x: {.d: Cpp.DeletedDestructor, .a: HasTrivialDestruction}) {
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_struct_with_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

fn TestNoDestroySecond(x: {.a: HasTrivialDestruction, .d: Cpp.DeletedDestructor}) {
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_trivially_destroyable_member.carbon:[[@LINE-26]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_struct_with_no_destroy_member_and_non_trivial_destroy_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

fn TestNoDestroyFirst(x: {.d: Cpp.DeletedDestructor, .a: HasNonTrivialDestruction}) {
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

fn TestNoDestroySecond(x: {.a: HasNonTrivialDestruction, .d: Cpp.DeletedDestructor}) {
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member.carbon:[[@LINE-26]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_tuple_with_no_destroy_element.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestNoDestroyElement(x: (Cpp.DeletedDestructor,)) {
  // CHECK:STDERR: fail_tuple_with_no_destroy_element.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_tuple_with_no_destroy_element.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_tuple_with_no_destroy_element.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_tuple_with_no_destroy_element.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_tuple_with_no_destroy_element.carbon:[[@LINE-12]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_tuple_with_no_destroy_element_and_trivially_destroyable_element.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

// CHECK:STDERR: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element.carbon:[[@LINE+2]]:50-70: error: name `HasTrivialDestruction` not found [NameNotFound]
// CHECK:STDERR:
fn TestNoDestroyFirst(x: (Cpp.DeletedDestructor, HasTrivialDestruction)) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// CHECK:STDERR: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element.carbon:[[@LINE+2]]:28-48: error: name `HasTrivialDestruction` not found [NameNotFound]
// CHECK:STDERR:
fn TestNoDestroySecond(x: (HasTrivialDestruction, Cpp.DeletedDestructor)) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

// CHECK:STDERR: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element.carbon:[[@LINE+2]]:50-73: error: name `HasNonTrivialDestruction` not found [NameNotFound]
// CHECK:STDERR:
fn TestNoDestroyFirst(x: (Cpp.DeletedDestructor, HasNonTrivialDestruction)) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// CHECK:STDERR: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element.carbon:[[@LINE+2]]:28-51: error: name `HasNonTrivialDestruction` not found [NameNotFound]
// CHECK:STDERR:
fn TestNoDestroySecond(x: (HasNonTrivialDestruction, Cpp.DeletedDestructor)) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}
