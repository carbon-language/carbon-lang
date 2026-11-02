// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// EXTRA-ARGS: --clang-arg=--std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/destroy/no_destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/destroy/no_destroy.carbon

// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a value's type can be destroyed.
fn CheckImplsDestroy[T: Core.Destroy](_: T) {}

// Identifies whether a type can be destroyed.
fn CheckTypeImplsDestroy(generic _: Core.Destroy) {}

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

struct TrivialPrivateDestructor {
private:
  ~TrivialPrivateDestructor() = default;
};

struct PrivateDestructor {
private:
  ~PrivateDestructor();
};

struct DestructorImplicitlyDeleted {
  DeletedDestructor field;
};

template<int N = 0>
class AmbiguousDestructorT {
 public:
  ~AmbiguousDestructorT() requires (N != 1);
  ~AmbiguousDestructorT() requires (N != 2);
};

using AmbiguousDestructor = AmbiguousDestructorT<>;

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

// --- fail_void.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp;

fn TestVoid() {
  // CHECK:STDERR: fail_void.carbon:[[@LINE+4]]:3-33: error: cannot convert type `void` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_void.carbon:[[@LINE-5]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:8:34-48: note: initializing generic parameter `_` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  CheckTypeImplsDestroy(Cpp.void);

  // TODO: add CheckImplsTrivialDestroy(Cpp.void);

  // TODO: add CheckImplsDynamicDestroy(Cpp.void);
}

// --- fail_unbounded_array.carbon

library "[[@TEST_NAME]]";

// Note: unbounded array types aren't filtered as non-destructible yet.
// The toolchain produces an unrelated diagnostic because we don't support
// unbounded arrays yet.

import library "check_destroy_format";
import Cpp inline '''c++
// CHECK:STDERR: fail_unbounded_array.carbon:[[@LINE+1]]:7: error: semantics TODO: `Unsupported: Type declaration: UnboundedArray` [SemanticsTodo]
using UnboundedArray = int[];
''';

// CHECK:STDERR: fail_unbounded_array.carbon:[[@LINE+2]]:26-43: note: in `Cpp` name lookup for `UnboundedArray` [InCppNameLookup]
// CHECK:STDERR:
fn TestUnboundedArray(x: Cpp.UnboundedArray) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
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

// --- fail_cpp_array.carbon

import library "check_destroy_format";
import Cpp library "cpp_types.h";

inline Cpp '''c++
using Array = DeletedDestructor[4];
''';

fn TestCppArray(x: Cpp.Array) {
  // CHECK:STDERR: fail_cpp_array.carbon:[[@LINE+26]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_cpp_array.carbon:[[@LINE-8]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_cpp_array.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_cpp_array.carbon:[[@LINE+20]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_cpp_array.carbon:[[@LINE-16]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_cpp_array.carbon:[[@LINE+15]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_cpp_array.carbon:[[@LINE-21]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_cpp_array.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_cpp_array.carbon:[[@LINE-26]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_cpp_array.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_cpp_array.carbon:[[@LINE-31]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_carbon_array.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestCarbonArray(x: array(Cpp.DeletedDestructor, 2)) {
  // CHECK:STDERR: fail_carbon_array.carbon:[[@LINE+16]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_carbon_array.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_carbon_array.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_carbon_array.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_carbon_array.carbon:[[@LINE-12]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_carbon_array.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_carbon_array.carbon:[[@LINE-17]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_carbon_array_of_array.carbon

// This test is separate from `fail_array.carbon` because all of the errors
// in this split are suppressed when combined.

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestArrayOfArrayOfDeletedDestructor(x: array(array(Cpp.DeletedDestructor, 2), 3)) {
  // CHECK:STDERR: fail_carbon_array_of_array.carbon:[[@LINE+16]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_carbon_array_of_array.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_carbon_array_of_array.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_carbon_array_of_array.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_carbon_array_of_array.carbon:[[@LINE-12]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_carbon_array_of_array.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_carbon_array_of_array.carbon:[[@LINE-17]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_cpp_class_with_destructor_explicitly_deleted.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestClassWithDestructorExplicitlyDeleted(x: Cpp.DeletedDestructor) {
  // CHECK:STDERR: fail_cpp_class_with_destructor_explicitly_deleted.carbon:[[@LINE+10]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_cpp_class_with_destructor_explicitly_deleted.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_cpp_class_with_destructor_explicitly_deleted.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_cpp_class_with_destructor_explicitly_deleted.carbon:[[@LINE+4]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_cpp_class_with_destructor_explicitly_deleted.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_cpp_class_with_destructor_implicitly_deleted.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestClassWithDestructorImplicitlyDeleted(x: Cpp.DestructorImplicitlyDeleted) {
  // CHECK:STDERR: fail_cpp_class_with_destructor_implicitly_deleted.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_cpp_class_with_destructor_implicitly_deleted.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:17:21: note: destructor of 'DestructorImplicitlyDeleted' is implicitly deleted because field 'field' has a deleted destructor [CppInteropParseNote]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_cpp_class_with_destructor_implicitly_deleted.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_cpp_class_with_destructor_implicitly_deleted.carbon:[[@LINE+4]]:3-22: error: cannot convert type `Cpp.DestructorImplicitlyDeleted` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_cpp_class_with_destructor_implicitly_deleted.carbon:[[@LINE-12]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_carbon_class_with_cpp_field_that_deletes_destructor_implicit.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class FieldExplicitlyDeletesDestructor {
  var d: Cpp.DeletedDestructor;
}

fn TestDeletedDestructorField(x: FieldExplicitlyDeletesDestructor) {
  // CHECK:STDERR: fail_carbon_class_with_cpp_field_that_deletes_destructor_implicit.carbon:[[@LINE+16]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_carbon_class_with_cpp_field_that_deletes_destructor_implicit.carbon:[[@LINE-8]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_carbon_class_with_cpp_field_that_deletes_destructor_implicit.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_carbon_class_with_cpp_field_that_deletes_destructor_implicit.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_carbon_class_with_cpp_field_that_deletes_destructor_implicit.carbon:[[@LINE-16]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_carbon_class_with_cpp_field_that_deletes_destructor_implicit.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_carbon_class_with_cpp_field_that_deletes_destructor_implicit.carbon:[[@LINE-21]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_carbon_class_with_cpp_field_that_deletes_destructor_explicit.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class FieldImplicitlyDeletesDestructor {
  var inner: Cpp.DestructorImplicitlyDeleted;
}

fn TestFieldImplicitlyDeletesDestructor(x: FieldImplicitlyDeletesDestructor) {
  // CHECK:STDERR: fail_carbon_class_with_cpp_field_that_deletes_destructor_explicit.carbon:[[@LINE+17]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_carbon_class_with_cpp_field_that_deletes_destructor_explicit.carbon:[[@LINE-8]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:17:21: note: destructor of 'DestructorImplicitlyDeleted' is implicitly deleted because field 'field' has a deleted destructor [CppInteropParseNote]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_carbon_class_with_cpp_field_that_deletes_destructor_explicit.carbon:[[@LINE-12]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_carbon_class_with_cpp_field_that_deletes_destructor_explicit.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DestructorImplicitlyDeleted` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_carbon_class_with_cpp_field_that_deletes_destructor_explicit.carbon:[[@LINE-17]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_carbon_class_with_cpp_field_that_deletes_destructor_explicit.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DestructorImplicitlyDeleted` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_carbon_class_with_cpp_field_that_deletes_destructor_explicit.carbon:[[@LINE-22]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_class_contains_no_destroy_member_and_trivially_destroyable_member_first.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

class NoDestroyMemberFirst {
  var a: Cpp.DeletedDestructor;
  var b: HasTrivialDestruction;
}

fn TestNoDestroyMemberFirst(x: NoDestroyMemberFirst) {
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE+16]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE-9]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE-18]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE-23]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_class_contains_no_destroy_member_and_trivially_destroyable_member_second.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

class NoDestroyMemberSecond {
  var a: HasTrivialDestruction;
  var b: Cpp.DeletedDestructor;
}

fn TestNoDestroyMemberSecond(x: NoDestroyMemberSecond) {
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE+16]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE-9]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE-18]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE-23]]:1 [InImport]
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
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE-18]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
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
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE-41]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_class_contains_no_destroy_member_and_non_trivially_destroyable_member.carbon:[[@LINE-46]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_deleted_destructor_field.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

base class DeletedDestructorFieldBase {
  var d: Cpp.DeletedDestructor;
}

class DeletedDestructorBaseAndTrivialMember {
  extend base: DeletedDestructorFieldBase;
  var a: HasTrivialDestruction;
}

fn TestDeletedDestructorBaseAndTrivialMember(x: DeletedDestructorBaseAndTrivialMember) {
  // CHECK:STDERR: fail_deleted_destructor_field.carbon:[[@LINE+16]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_deleted_destructor_field.carbon:[[@LINE-13]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_deleted_destructor_field.carbon:[[@LINE-17]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_deleted_destructor_field.carbon:[[@LINE+10]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_deleted_destructor_field.carbon:[[@LINE-22]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_deleted_destructor_field.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_deleted_destructor_field.carbon:[[@LINE-27]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
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
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_field_trivial_base_and_deleted_destructor_member.carbon:[[@LINE-19]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_field_trivial_base_and_deleted_destructor_member.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_field_trivial_base_and_deleted_destructor_member.carbon:[[@LINE-24]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- todo_fail_class_trivial_protected_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp inline '''c++
struct TrivialProtectedDestructor {
protected:
  ~TrivialProtectedDestructor() = default;
};
''';

// TODO: `Core.Destroy` should fail, because the destructor isn't public.
fn TestTrivialProtectedDestructor(x: Cpp.TrivialProtectedDestructor) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_class_protected_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp inline '''c++
struct ProtectedDestructor {
protected:
  ~ProtectedDestructor();
};
''';

fn TestProtectedDestructor(x: Cpp.ProtectedDestructor) {
  // CHECK:STDERR: fail_class_protected_destructor.carbon:[[@LINE+5]]:3-22: error: cannot access protected member `<C++ destructor>` of type `Cpp.ProtectedDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_class_protected_destructor.carbon:[[@LINE-6]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: imported from: fail_class_protected_destructor.carbon:[[@LINE-11]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- todo_fail_trivial_private_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

// TODO: `Core.Destroy` should fail, because the destructor isn't public.
fn TestTrivialPrivateDestructor(x: Cpp.TrivialPrivateDestructor) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_class_private_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestPrivateDestructor(x: Cpp.PrivateDestructor) {
  // CHECK:STDERR: fail_class_private_destructor.carbon:[[@LINE+6]]:3-22: error: cannot access private member `<C++ destructor>` of type `Cpp.PrivateDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_class_private_destructor.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:13:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: imported from: fail_class_private_destructor.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- todo_fail_derived_from_trivial_private_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class Derived {
  extend base: Cpp.TrivialPrivateDestructor;
}

// TODO: `Core.Destroy` should fail, because the destructor isn't public.
fn TestPrivateDestructor(x: Derived) {
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_derived_from_private_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class Derived {
  extend base: Cpp.PrivateDestructor;
}

// TODO: `Core.Destroy` should fail, because the destructor isn't public.
fn TestPrivateDestructor(x: Derived) {
  // CHECK:STDERR: fail_derived_from_private_destructor.carbon:[[@LINE+6]]:3-22: error: cannot access private member `<C++ destructor>` of type `Cpp.PrivateDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_derived_from_private_destructor.carbon:[[@LINE-9]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:13:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: imported from: fail_derived_from_private_destructor.carbon:[[@LINE-12]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
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
// CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+12]]:1-36: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
// CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+8]]:1-36: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
// CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
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

// TODO: `CheckImplsDestroy` should fail too, because
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
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_struct_with_member_that_deletes_destructor.carbon:[[@LINE-12]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_struct_with_no_destroy_member_and_trivially_destroyable_member_first.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

fn TestNoDestroyFirst(x: {.d: Cpp.DeletedDestructor, .a: HasTrivialDestruction}) {
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_struct_with_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_trivially_destroyable_member_first.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_struct_with_no_destroy_member_and_trivially_destroyable_member_second.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";


fn TestNoDestroySecond(x: {.a: HasTrivialDestruction, .d: Cpp.DeletedDestructor}) {
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_struct_with_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE-5]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE-9]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_trivially_destroyable_member_second.carbon:[[@LINE-14]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_first.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

fn TestNoDestroyFirst(x: {.d: Cpp.DeletedDestructor, .a: HasNonTrivialDestruction}) {
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_first.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_first.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_first.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_first.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_first.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_second.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

fn TestNoDestroySecond(x: {.a: HasNonTrivialDestruction, .d: Cpp.DeletedDestructor}) {
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_second.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_second.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_second.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_second.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_struct_with_no_destroy_member_and_non_trivial_destroy_member_second.carbon:[[@LINE-13]]:1 [InImport]
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
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_tuple_with_no_destroy_element.carbon:[[@LINE-12]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_first.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

fn TestNoDestroyFirst(x: (Cpp.DeletedDestructor, HasTrivialDestruction)) {
  // CHECK:STDERR: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_first.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_first.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_first.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_first.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_first.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_second.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";


fn TestNoDestroySecond(x: (HasTrivialDestruction, Cpp.DeletedDestructor)) {
  // CHECK:STDERR: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_second.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_second.carbon:[[@LINE-5]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_second.carbon:[[@LINE-9]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_second.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_tuple_with_no_destroy_element_and_trivially_destroyable_element_second.carbon:[[@LINE-14]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_first.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

fn TestNoDestroyFirst(x: (Cpp.DeletedDestructor, HasNonTrivialDestruction)) {
  // CHECK:STDERR: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_first.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_first.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_first.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_first.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_first.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_second.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";
import Cpp library "cpp_types.h";

fn TestNoDestroySecond(x: (HasNonTrivialDestruction, Cpp.DeletedDestructor)) {
  // CHECK:STDERR: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_second.carbon:[[@LINE+11]]:22: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_second.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: imported from: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_second.carbon:[[@LINE-8]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_second.carbon:[[@LINE+5]]:3-22: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: {{.*}}/prelude/destroy.carbon:36:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: imported from: fail_tuple_with_no_destroy_element_and_non_trivially_destroyable_element_second.carbon:[[@LINE-13]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_ambiguous_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

// CHECK:STDERR: fail_ambiguous_destructor.carbon:[[@LINE+9]]:28-53: error: parameter has incomplete type `Cpp.AmbiguousDestructorT` in function definition [IncompleteTypeInFunctionParam]
// CHECK:STDERR: included from: fail_ambiguous_destructor.carbon:[[@LINE-3]]:10 [InCppInclude]
// CHECK:STDERR: ./cpp_types.h:21:7: note: destructor of class 'AmbiguousDestructorT<>' is ambiguous [CppInteropParseError]
// CHECK:STDERR: ./cpp_types.h:21:7: note: in instantiation of template class 'AmbiguousDestructorT<>' requested here [CppInteropParseNote]
// CHECK:STDERR: ./cpp_types.h:23:3: note: candidate function [CppInteropParseNote]
// CHECK:STDERR: ./cpp_types.h:24:3: note: candidate function [CppInteropParseNote]
// CHECK:STDERR:
// CHECK:STDERR: fail_ambiguous_destructor.carbon:[[@LINE+2]]:28-53: warning: binding `x` unused [UnusedBinding]
// CHECK:STDERR:
fn TestAmbiguousDestructor(x: Cpp.AmbiguousDestructor) {}

// --- fail_member_has_ambiguous_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class MemberHasAmbiguousDestructor {
  // CHECK:STDERR: fail_member_has_ambiguous_destructor.carbon:[[@LINE+7]]:10-32: error: binding pattern has incomplete type `AmbiguousDestructor` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR: included from: fail_member_has_ambiguous_destructor.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:21:7: note: destructor of class 'AmbiguousDestructorT<>' is ambiguous [CppInteropParseError]
  // CHECK:STDERR: ./cpp_types.h:21:7: note: in instantiation of template class 'AmbiguousDestructorT<>' requested here [CppInteropParseNote]
  // CHECK:STDERR: ./cpp_types.h:23:3: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR: ./cpp_types.h:24:3: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:
  var x: Cpp.AmbiguousDestructor;
}

// --- fail_derived_from_ambiguous_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class DerivedFromAmbiguousDestructor {
  // CHECK:STDERR: fail_derived_from_ambiguous_destructor.carbon:[[@LINE+7]]:16-38: error: base `AmbiguousDestructor` is an incomplete type [IncompleteTypeInBaseDecl]
  // CHECK:STDERR: included from: fail_derived_from_ambiguous_destructor.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./cpp_types.h:21:7: note: destructor of class 'AmbiguousDestructorT<>' is ambiguous [CppInteropParseError]
  // CHECK:STDERR: ./cpp_types.h:21:7: note: in instantiation of template class 'AmbiguousDestructorT<>' requested here [CppInteropParseNote]
  // CHECK:STDERR: ./cpp_types.h:23:3: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR: ./cpp_types.h:24:3: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:
  extend base: Cpp.AmbiguousDestructor;
}
