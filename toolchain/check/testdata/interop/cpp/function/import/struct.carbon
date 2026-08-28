// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/struct.carbon

// ============================================================================
// Forward-declared struct as parameter type
// ============================================================================

// --- decl_value_param_type.h

struct ForwardDeclaredStruct;

auto foo1(ForwardDeclaredStruct) -> void;
auto foo2(ForwardDeclaredStruct) -> void;

// --- fail_import_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:4:6-9: error: argument type 'ForwardDeclaredStruct' is incomplete [CppInteropParseError]
// CHECK:STDERR: ./decl_value_param_type.h:2:8-28: note: forward declaration of 'ForwardDeclaredStruct' [CppInteropParseNote]
import Cpp library "decl_value_param_type.h";

fn F(p: Cpp.ForwardDeclaredStruct*) {
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+5]]:3-14: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  // CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE-5]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:5:6-9: error: argument type 'ForwardDeclaredStruct' is incomplete [CppInteropParseError]
  // CHECK:STDERR: ./decl_value_param_type.h:2:8-28: note: forward declaration of 'ForwardDeclaredStruct' [CppInteropParseNote]
  Cpp.foo1(*p);
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+12]]:3-14: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-3]]:12-13: error: forming value of incomplete type `Cpp.ForwardDeclaredStruct` [IncompleteTypeInValueConversion]
  // CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE-12]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:8: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: ./decl_value_param_type.h:4:32: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+5]]:12-13: error: forming value of incomplete type `Cpp.ForwardDeclaredStruct` [IncompleteTypeInValueConversion]
  // CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE-17]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:8: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: ./decl_value_param_type.h:5:32: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.foo2(*p);
}

// ============================================================================
// Defined struct without data members as parameter type
// ============================================================================

// --- definition_no_data_members_value_param_type.h

struct EmptyStruct {};

auto foo_empty(EmptyStruct) -> void;

// --- import_definition_no_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_no_data_members_value_param_type.h";

fn F(s: Cpp.EmptyStruct) {
  //@dump-sem-ir-begin
  Cpp.foo_empty(s);
  //@dump-sem-ir-end
}

// ============================================================================
// Non copyable struct as parameter type
// ============================================================================

// --- non_copyable_param_type.h

struct NonCopyableStruct {
  NonCopyableStruct(const NonCopyableStruct&) = delete;
};

auto foo_non_copyable(NonCopyableStruct) -> void;

// --- fail_import_non_copyable_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_non_copyable_param_type.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./non_copyable_param_type.h:6:6-21: error: call to deleted constructor of 'NonCopyableStruct' [CppInteropParseError]
// CHECK:STDERR: ./non_copyable_param_type.h:3:3-19: note: 'NonCopyableStruct' has been explicitly marked deleted here [CppInteropParseNote]
// CHECK:STDERR: ./non_copyable_param_type.h:6:40: note: passing argument to parameter here [CppInteropParseNote]
import Cpp library "non_copyable_param_type.h";

fn F(s: Cpp.NonCopyableStruct) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_import_non_copyable_param_type.carbon:[[@LINE+2]]:3-25: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  Cpp.foo_non_copyable(s);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct with a single data member as parameter type
// ============================================================================

// --- definition_single_data_member_value_param_type.h

struct SingleMemberD {};

struct SingleMemberStruct {
  SingleMemberD d;
};

auto foo_single_member(SingleMemberStruct) -> void;

// --- definition_single_data_member_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_single_data_member_value_param_type.h";

fn F(s: Cpp.SingleMemberStruct) {
  //@dump-sem-ir-begin
  Cpp.foo_single_member(s);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct with multiple data members as parameter type
// ============================================================================

// --- definition_multiple_data_members_value_param_type.h

struct MultipleMembersD {};

struct MultipleMembersStruct {
  MultipleMembersD d1;
  MultipleMembersD d2;
  MultipleMembersD d3;
};

auto foo_multiple_members(MultipleMembersStruct) -> void;

// --- definition_multiple_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_multiple_data_members_value_param_type.h";

fn F(s: Cpp.MultipleMembersStruct) {
  //@dump-sem-ir-begin
  Cpp.foo_multiple_members(s);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct in namespace
// ============================================================================

// --- definition_in_namespace_value_param_type.h

namespace N { struct InNamespaceStruct {}; }

auto foo_in_namespace(N::InNamespaceStruct) -> void;

// --- import_definition_in_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_namespace_value_param_type.h";

fn F(s: Cpp.N.InNamespaceStruct) {
  //@dump-sem-ir-begin
  Cpp.foo_in_namespace(s);
  // Check that the parameter type was imported correctly.
  var unused x: Cpp.N.InNamespaceStruct;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct in relative namespace
// ============================================================================

// --- definition_in_relative_namespace_value_param_type.h

namespace N1 {
  namespace N2 { struct InRelativeNamespaceStruct {}; }
  auto foo(N2::InRelativeNamespaceStruct) -> void;
}

// --- import_definition_in_relative_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_relative_namespace_value_param_type.h";

fn F(s: Cpp.N1.N2.InRelativeNamespaceStruct) {
  //@dump-sem-ir-begin
  Cpp.N1.foo(s);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct in struct
// ============================================================================

// --- definition_in_outer_definition.h

struct OuterStruct {
  struct InnerStruct {};
};

auto foo_in_outer(OuterStruct::InnerStruct) -> void;

// --- import_definition_in_outer_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_outer_definition.h";

fn F(s: Cpp.OuterStruct.InnerStruct) {
  //@dump-sem-ir-begin
  Cpp.foo_in_outer(s);
  var unused x: Cpp.OuterStruct;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct and explicitly used
// ============================================================================

// --- definition_with_static_method.h

struct StaticMethodStruct {
  static void bar();
};

auto foo_static_method(StaticMethodStruct) -> void;

// --- import_definition_and_static_method_call_before.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F(s: Cpp.StaticMethodStruct) {
  //@dump-sem-ir-begin
  Cpp.StaticMethodStruct.bar();
  Cpp.foo_static_method(s);
  //@dump-sem-ir-end
}

// --- import_definition_and_static_method_call_after.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F(s: Cpp.StaticMethodStruct) {
  //@dump-sem-ir-begin
  Cpp.foo_static_method(s);
  Cpp.StaticMethodStruct.bar();
  //@dump-sem-ir-end
}

// ============================================================================
// Incomplete struct
// ============================================================================

// --- incomplete.h

struct IncompleteStruct;

// --- fail_import_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp library "incomplete.h";

fn F() {
  // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE+4]]:3-26: error: member access into incomplete class `Cpp.IncompleteStruct` [QualifiedExprInIncompleteClassScope]
  // CHECK:STDERR: included from: fail_import_incomplete.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./incomplete.h:2:8: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  Cpp.IncompleteStruct.foo();
}

// ============================================================================
// Pointer to forward-declared struct as parameter type
// ============================================================================

// --- decl_pointer_param_type.h

struct DeclPointerParamStruct;

auto foo_decl_pointer_param(DeclPointerParamStruct* _Nonnull) -> void;

// --- import_decl_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_param_type.h";

fn F(s: Cpp.DeclPointerParamStruct*) {
  //@dump-sem-ir-begin
  Cpp.foo_decl_pointer_param(s);
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined struct as parameter type
// ============================================================================

// --- definition_pointer_param_type.h

struct DefPointerParamStruct {};

auto foo_def_pointer_param(DefPointerParamStruct* _Nonnull) -> void;

// --- import_definition_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_pointer_param_type.h";

fn F(s: Cpp.DefPointerParamStruct*) {
  //@dump-sem-ir-begin
  Cpp.foo_def_pointer_param(s);
  //@dump-sem-ir-end
}

// ============================================================================
// Forward-declared struct as return type
// ============================================================================

// --- decl_value_return_type.h

struct DeclValueReturnStruct;

auto foo_decl_value_return() -> DeclValueReturnStruct;

// --- fail_import_decl_value_return_type.carbon

library "[[@TEST_NAME]]";

// TODO: We should only produce one error here.
// CHECK:STDERR: included from: fail_import_decl_value_return_type.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:4:6-26: error: calling 'foo_decl_value_return' with incomplete return type 'DeclValueReturnStruct' [CppInteropParseError]
// CHECK:STDERR: ./decl_value_return_type.h:4:6-26: note: 'foo_decl_value_return' declared here [CppInteropParseNote]
// CHECK:STDERR: ./decl_value_return_type.h:2:8-28: note: forward declaration of 'DeclValueReturnStruct' [CppInteropParseNote]
import Cpp library "decl_value_return_type.h";

fn F() {
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+7]]:3-29: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+5]]:3-29: error: function returns incomplete type `Cpp.DeclValueReturnStruct` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR: included from: fail_import_decl_value_return_type.carbon:[[@LINE-6]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_return_type.h:2:8: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: ./decl_value_return_type.h:4:1: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR:
  Cpp.foo_decl_value_return();
}

// ============================================================================
// Defined struct as return type
// ============================================================================

// --- definition_value_return_type.h

struct DefValueReturnStruct {};

auto foo_def_value_return() -> DefValueReturnStruct;

// --- import_definition_value_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_value_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo_def_value_return();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to forward-declared struct as return type
// ============================================================================

// --- decl_pointer_return_type.h

struct DeclPointerReturnStruct;

auto foo_decl_pointer_return() -> DeclPointerReturnStruct* _Nonnull;

// --- import_decl_pointer_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo_decl_pointer_return();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined struct as return type
// ============================================================================

// --- definition_pointer_return_type.h

struct DefPointerReturnStruct {};

auto foo_def_pointer_return() -> DefPointerReturnStruct* _Nonnull;

// --- import_definition_pointer_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_pointer_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo_def_pointer_return();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_definition_no_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %EmptyStruct: type = class_type @EmptyStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_empty.cpp_overload_set.type: type = cpp_overload_set_type @foo_empty.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_empty.cpp_overload_set.value: %foo_empty.cpp_overload_set.type = cpp_overload_set_value @foo_empty.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.9e5: type = ptr_type %EmptyStruct [concrete]
// CHECK:STDOUT:   %foo_empty__carbon_thunk.type: type = fn_type @foo_empty__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_empty__carbon_thunk: %foo_empty__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .EmptyStruct = %EmptyStruct.decl
// CHECK:STDOUT:     .foo_empty = %foo_empty.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyStruct.decl: type = class_decl @EmptyStruct [concrete = constants.%EmptyStruct] {} {}
// CHECK:STDOUT:   %foo_empty.cpp_overload_set.value: %foo_empty.cpp_overload_set.type = cpp_overload_set_value @foo_empty.cpp_overload_set [concrete = constants.%foo_empty.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_empty__carbon_thunk.decl: %foo_empty__carbon_thunk.type = fn_decl @foo_empty__carbon_thunk [concrete = constants.%foo_empty__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %EmptyStruct) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_empty.ref: %foo_empty.cpp_overload_set.type = name_ref foo_empty, imports.%foo_empty.cpp_overload_set.value [concrete = constants.%foo_empty.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %EmptyStruct = name_ref s, %s
// CHECK:STDOUT:   %.loc8: ref %EmptyStruct = value_as_ref %s.ref
// CHECK:STDOUT:   %addr: %ptr.9e5 = addr_of %.loc8
// CHECK:STDOUT:   %foo_empty__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_empty__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_non_copyable_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonCopyableStruct: type = class_type @NonCopyableStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_non_copyable.cpp_overload_set.type: type = cpp_overload_set_type @foo_non_copyable.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_non_copyable.cpp_overload_set.value: %foo_non_copyable.cpp_overload_set.type = cpp_overload_set_value @foo_non_copyable.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_non_copyable.type: type = fn_type @foo_non_copyable [concrete]
// CHECK:STDOUT:   %foo_non_copyable: %foo_non_copyable.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonCopyableStruct = %NonCopyableStruct.decl
// CHECK:STDOUT:     .foo_non_copyable = %foo_non_copyable.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonCopyableStruct.decl: type = class_decl @NonCopyableStruct [concrete = constants.%NonCopyableStruct] {} {}
// CHECK:STDOUT:   %foo_non_copyable.cpp_overload_set.value: %foo_non_copyable.cpp_overload_set.type = cpp_overload_set_value @foo_non_copyable.cpp_overload_set [concrete = constants.%foo_non_copyable.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_non_copyable.decl: %foo_non_copyable.type = fn_decl @foo_non_copyable [concrete = constants.%foo_non_copyable] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %NonCopyableStruct) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_non_copyable.ref: %foo_non_copyable.cpp_overload_set.type = name_ref foo_non_copyable, imports.%foo_non_copyable.cpp_overload_set.value [concrete = constants.%foo_non_copyable.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %NonCopyableStruct = name_ref s, %s
// CHECK:STDOUT:   %foo_non_copyable.call: init %empty_tuple.type = call imports.%foo_non_copyable.decl(%s.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- definition_single_data_member_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SingleMemberStruct: type = class_type @SingleMemberStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_single_member.cpp_overload_set.type: type = cpp_overload_set_type @foo_single_member.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_single_member.cpp_overload_set.value: %foo_single_member.cpp_overload_set.type = cpp_overload_set_value @foo_single_member.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.3c4: type = ptr_type %SingleMemberStruct [concrete]
// CHECK:STDOUT:   %foo_single_member__carbon_thunk.type: type = fn_type @foo_single_member__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_single_member__carbon_thunk: %foo_single_member__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .SingleMemberStruct = %SingleMemberStruct.decl
// CHECK:STDOUT:     .foo_single_member = %foo_single_member.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SingleMemberStruct.decl: type = class_decl @SingleMemberStruct [concrete = constants.%SingleMemberStruct] {} {}
// CHECK:STDOUT:   %foo_single_member.cpp_overload_set.value: %foo_single_member.cpp_overload_set.type = cpp_overload_set_value @foo_single_member.cpp_overload_set [concrete = constants.%foo_single_member.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_single_member__carbon_thunk.decl: %foo_single_member__carbon_thunk.type = fn_decl @foo_single_member__carbon_thunk [concrete = constants.%foo_single_member__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %SingleMemberStruct) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_single_member.ref: %foo_single_member.cpp_overload_set.type = name_ref foo_single_member, imports.%foo_single_member.cpp_overload_set.value [concrete = constants.%foo_single_member.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %SingleMemberStruct = name_ref s, %s
// CHECK:STDOUT:   %.loc8: ref %SingleMemberStruct = value_as_ref %s.ref
// CHECK:STDOUT:   %addr: %ptr.3c4 = addr_of %.loc8
// CHECK:STDOUT:   %foo_single_member__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_single_member__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- definition_multiple_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MultipleMembersStruct: type = class_type @MultipleMembersStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_multiple_members.cpp_overload_set.type: type = cpp_overload_set_type @foo_multiple_members.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_multiple_members.cpp_overload_set.value: %foo_multiple_members.cpp_overload_set.type = cpp_overload_set_value @foo_multiple_members.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.f92: type = ptr_type %MultipleMembersStruct [concrete]
// CHECK:STDOUT:   %foo_multiple_members__carbon_thunk.type: type = fn_type @foo_multiple_members__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_multiple_members__carbon_thunk: %foo_multiple_members__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MultipleMembersStruct = %MultipleMembersStruct.decl
// CHECK:STDOUT:     .foo_multiple_members = %foo_multiple_members.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MultipleMembersStruct.decl: type = class_decl @MultipleMembersStruct [concrete = constants.%MultipleMembersStruct] {} {}
// CHECK:STDOUT:   %foo_multiple_members.cpp_overload_set.value: %foo_multiple_members.cpp_overload_set.type = cpp_overload_set_value @foo_multiple_members.cpp_overload_set [concrete = constants.%foo_multiple_members.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_multiple_members__carbon_thunk.decl: %foo_multiple_members__carbon_thunk.type = fn_decl @foo_multiple_members__carbon_thunk [concrete = constants.%foo_multiple_members__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %MultipleMembersStruct) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_multiple_members.ref: %foo_multiple_members.cpp_overload_set.type = name_ref foo_multiple_members, imports.%foo_multiple_members.cpp_overload_set.value [concrete = constants.%foo_multiple_members.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %MultipleMembersStruct = name_ref s, %s
// CHECK:STDOUT:   %.loc8: ref %MultipleMembersStruct = value_as_ref %s.ref
// CHECK:STDOUT:   %addr: %ptr.f92 = addr_of %.loc8
// CHECK:STDOUT:   %foo_multiple_members__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_multiple_members__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InNamespaceStruct: type = class_type @InNamespaceStruct [concrete]
// CHECK:STDOUT:   %pattern_type.d3e: type = pattern_type %InNamespaceStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_in_namespace.cpp_overload_set.type: type = cpp_overload_set_type @foo_in_namespace.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_in_namespace.cpp_overload_set.value: %foo_in_namespace.cpp_overload_set.type = cpp_overload_set_value @foo_in_namespace.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.f55: type = ptr_type %InNamespaceStruct [concrete]
// CHECK:STDOUT:   %foo_in_namespace__carbon_thunk.type: type = fn_type @foo_in_namespace__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_in_namespace__carbon_thunk: %foo_in_namespace__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d3e = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.d3e = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %InNamespaceStruct.Op.type: type = fn_type @InNamespaceStruct.Op [concrete]
// CHECK:STDOUT:   %InNamespaceStruct.Op: %InNamespaceStruct.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.85f: <witness> = custom_witness (%InNamespaceStruct.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.598: %Default.type = facet_value %InNamespaceStruct, (%custom_witness.85f) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.d48: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.162, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.598) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %InNamespaceStruct, (%DefaultOrUnformed.impl_witness.d48) [concrete]
// CHECK:STDOUT:   %InNamespaceStruct.cpp_destructor.type: type = fn_type @InNamespaceStruct.cpp_destructor [concrete]
// CHECK:STDOUT:   %InNamespaceStruct.cpp_destructor: %InNamespaceStruct.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .foo_in_namespace = %foo_in_namespace.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .InNamespaceStruct = %InNamespaceStruct.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InNamespaceStruct.decl: type = class_decl @InNamespaceStruct [concrete = constants.%InNamespaceStruct] {} {}
// CHECK:STDOUT:   %foo_in_namespace.cpp_overload_set.value: %foo_in_namespace.cpp_overload_set.type = cpp_overload_set_value @foo_in_namespace.cpp_overload_set [concrete = constants.%foo_in_namespace.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_in_namespace__carbon_thunk.decl: %foo_in_namespace__carbon_thunk.type = fn_decl @foo_in_namespace__carbon_thunk [concrete = constants.%foo_in_namespace__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.c2f: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.cfe)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.162 = impl_witness_table (%Core.import_ref.c2f), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %InNamespaceStruct) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_in_namespace.ref: %foo_in_namespace.cpp_overload_set.type = name_ref foo_in_namespace, imports.%foo_in_namespace.cpp_overload_set.value [concrete = constants.%foo_in_namespace.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %InNamespaceStruct = name_ref s, %s
// CHECK:STDOUT:   %.loc8: ref %InNamespaceStruct = value_as_ref %s.ref
// CHECK:STDOUT:   %addr: %ptr.f55 = addr_of %.loc8
// CHECK:STDOUT:   %foo_in_namespace__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_in_namespace__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %x.var: ref %InNamespaceStruct = var_storage %x.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%InNamespaceStruct, (constants.%DefaultOrUnformed.impl_witness.d48) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc10_40.1: %DefaultOrUnformed.type = converted constants.%InNamespaceStruct, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_40.1 [concrete = constants.%InNamespaceStruct]
// CHECK:STDOUT:   %.loc10_40.2: type = converted %.loc10_40.1, %as_type [concrete = constants.%InNamespaceStruct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_3: ref %InNamespaceStruct = splice_block %x.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %InNamespaceStruct to %.loc10_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %x.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc10_22: type = splice_block %InNamespaceStruct.ref.loc10 [concrete = constants.%InNamespaceStruct] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %InNamespaceStruct.ref.loc10: type = name_ref InNamespaceStruct, imports.%InNamespaceStruct.decl [concrete = constants.%InNamespaceStruct]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %InNamespaceStruct = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d3e = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d3e = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InNamespaceStruct.cpp_destructor.bound: <bound method> = bound_method %x.var, constants.%InNamespaceStruct.cpp_destructor
// CHECK:STDOUT:   %InNamespaceStruct.cpp_destructor.call: init %empty_tuple.type = call %InNamespaceStruct.cpp_destructor.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_relative_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InRelativeNamespaceStruct: type = class_type @InRelativeNamespaceStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.772: type = ptr_type %InRelativeNamespaceStruct [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N1 = %N1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .InRelativeNamespaceStruct = %InRelativeNamespaceStruct.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InRelativeNamespaceStruct.decl: type = class_decl @InRelativeNamespaceStruct [concrete = constants.%InRelativeNamespaceStruct] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %InRelativeNamespaceStruct) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc8: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %InRelativeNamespaceStruct = name_ref s, %s
// CHECK:STDOUT:   %.loc8: ref %InRelativeNamespaceStruct = value_as_ref %s.ref
// CHECK:STDOUT:   %addr: %ptr.772 = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_outer_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %OuterStruct: type = class_type @OuterStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %InnerStruct: type = class_type @InnerStruct [concrete]
// CHECK:STDOUT:   %foo_in_outer.cpp_overload_set.type: type = cpp_overload_set_type @foo_in_outer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_in_outer.cpp_overload_set.value: %foo_in_outer.cpp_overload_set.type = cpp_overload_set_value @foo_in_outer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d74: type = ptr_type %InnerStruct [concrete]
// CHECK:STDOUT:   %foo_in_outer__carbon_thunk.type: type = fn_type @foo_in_outer__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_in_outer__carbon_thunk: %foo_in_outer__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e34: type = pattern_type %OuterStruct [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.e34 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.e34 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %OuterStruct.Op.type: type = fn_type @OuterStruct.Op [concrete]
// CHECK:STDOUT:   %OuterStruct.Op: %OuterStruct.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b4: <witness> = custom_witness (%OuterStruct.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.e09: %Default.type = facet_value %OuterStruct, (%custom_witness.9b4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.219: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.162, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.e09) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %OuterStruct, (%DefaultOrUnformed.impl_witness.219) [concrete]
// CHECK:STDOUT:   %OuterStruct.cpp_destructor.type: type = fn_type @OuterStruct.cpp_destructor [concrete]
// CHECK:STDOUT:   %OuterStruct.cpp_destructor: %OuterStruct.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OuterStruct = %OuterStruct.decl
// CHECK:STDOUT:     .foo_in_outer = %foo_in_outer.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OuterStruct.decl: type = class_decl @OuterStruct [concrete = constants.%OuterStruct] {} {}
// CHECK:STDOUT:   %foo_in_outer.cpp_overload_set.value: %foo_in_outer.cpp_overload_set.type = cpp_overload_set_value @foo_in_outer.cpp_overload_set [concrete = constants.%foo_in_outer.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_in_outer__carbon_thunk.decl: %foo_in_outer__carbon_thunk.type = fn_decl @foo_in_outer__carbon_thunk [concrete = constants.%foo_in_outer__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.c2f: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.cfe)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.162 = impl_witness_table (%Core.import_ref.c2f), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %InnerStruct) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_in_outer.ref: %foo_in_outer.cpp_overload_set.type = name_ref foo_in_outer, imports.%foo_in_outer.cpp_overload_set.value [concrete = constants.%foo_in_outer.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %InnerStruct = name_ref s, %s
// CHECK:STDOUT:   %.loc8: ref %InnerStruct = value_as_ref %s.ref
// CHECK:STDOUT:   %addr: %ptr.d74 = addr_of %.loc8
// CHECK:STDOUT:   %foo_in_outer__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_in_outer__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %x.var: ref %OuterStruct = var_storage %x.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%OuterStruct, (constants.%DefaultOrUnformed.impl_witness.219) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc9_32.1: %DefaultOrUnformed.type = converted constants.%OuterStruct, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc9_32.1 [concrete = constants.%OuterStruct]
// CHECK:STDOUT:   %.loc9_32.2: type = converted %.loc9_32.1, %as_type [concrete = constants.%OuterStruct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9_3: ref %OuterStruct = splice_block %x.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %OuterStruct to %.loc9_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %x.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %OuterStruct.ref.loc9 [concrete = constants.%OuterStruct] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OuterStruct.ref.loc9: type = name_ref OuterStruct, imports.%OuterStruct.decl [concrete = constants.%OuterStruct]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %OuterStruct = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.e34 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.e34 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OuterStruct.cpp_destructor.bound: <bound method> = bound_method %x.var, constants.%OuterStruct.cpp_destructor
// CHECK:STDOUT:   %OuterStruct.cpp_destructor.call: init %empty_tuple.type = call %OuterStruct.cpp_destructor.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %StaticMethodStruct: type = class_type @StaticMethodStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %StaticMethodStruct.bar.cpp_overload_set.type: type = cpp_overload_set_type @StaticMethodStruct.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodStruct.bar.cpp_overload_set.value: %StaticMethodStruct.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodStruct.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodStruct.bar.type: type = fn_type @StaticMethodStruct.bar [concrete]
// CHECK:STDOUT:   %StaticMethodStruct.bar: %StaticMethodStruct.bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.type: type = cpp_overload_set_type @foo_static_method.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.value: %foo_static_method.cpp_overload_set.type = cpp_overload_set_value @foo_static_method.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.f77: type = ptr_type %StaticMethodStruct [concrete]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.type: type = fn_type @foo_static_method__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk: %foo_static_method__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .StaticMethodStruct = %StaticMethodStruct.decl
// CHECK:STDOUT:     .foo_static_method = %foo_static_method.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMethodStruct.decl: type = class_decl @StaticMethodStruct [concrete = constants.%StaticMethodStruct] {} {}
// CHECK:STDOUT:   %StaticMethodStruct.bar.cpp_overload_set.value: %StaticMethodStruct.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodStruct.bar.cpp_overload_set [concrete = constants.%StaticMethodStruct.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodStruct.bar.decl: %StaticMethodStruct.bar.type = fn_decl @StaticMethodStruct.bar [concrete = constants.%StaticMethodStruct.bar] {} {}
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.value: %foo_static_method.cpp_overload_set.type = cpp_overload_set_value @foo_static_method.cpp_overload_set [concrete = constants.%foo_static_method.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.decl: %foo_static_method__carbon_thunk.type = fn_decl @foo_static_method__carbon_thunk [concrete = constants.%foo_static_method__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %StaticMethodStruct) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %StaticMethodStruct.ref.loc8: type = name_ref StaticMethodStruct, imports.%StaticMethodStruct.decl [concrete = constants.%StaticMethodStruct]
// CHECK:STDOUT:   %bar.ref: %StaticMethodStruct.bar.cpp_overload_set.type = name_ref bar, imports.%StaticMethodStruct.bar.cpp_overload_set.value [concrete = constants.%StaticMethodStruct.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodStruct.bar.call: init %empty_tuple.type = call imports.%StaticMethodStruct.bar.decl()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_static_method.ref: %foo_static_method.cpp_overload_set.type = name_ref foo_static_method, imports.%foo_static_method.cpp_overload_set.value [concrete = constants.%foo_static_method.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %StaticMethodStruct = name_ref s, %s
// CHECK:STDOUT:   %.loc9: ref %StaticMethodStruct = value_as_ref %s.ref
// CHECK:STDOUT:   %addr: %ptr.f77 = addr_of %.loc9
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_static_method__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %StaticMethodStruct: type = class_type @StaticMethodStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.type: type = cpp_overload_set_type @foo_static_method.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.value: %foo_static_method.cpp_overload_set.type = cpp_overload_set_value @foo_static_method.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.f77: type = ptr_type %StaticMethodStruct [concrete]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.type: type = fn_type @foo_static_method__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk: %foo_static_method__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %StaticMethodStruct.bar.cpp_overload_set.type: type = cpp_overload_set_type @StaticMethodStruct.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodStruct.bar.cpp_overload_set.value: %StaticMethodStruct.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodStruct.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodStruct.bar.type: type = fn_type @StaticMethodStruct.bar [concrete]
// CHECK:STDOUT:   %StaticMethodStruct.bar: %StaticMethodStruct.bar.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .StaticMethodStruct = %StaticMethodStruct.decl
// CHECK:STDOUT:     .foo_static_method = %foo_static_method.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMethodStruct.decl: type = class_decl @StaticMethodStruct [concrete = constants.%StaticMethodStruct] {} {}
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.value: %foo_static_method.cpp_overload_set.type = cpp_overload_set_value @foo_static_method.cpp_overload_set [concrete = constants.%foo_static_method.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.decl: %foo_static_method__carbon_thunk.type = fn_decl @foo_static_method__carbon_thunk [concrete = constants.%foo_static_method__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMethodStruct.bar.cpp_overload_set.value: %StaticMethodStruct.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodStruct.bar.cpp_overload_set [concrete = constants.%StaticMethodStruct.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodStruct.bar.decl: %StaticMethodStruct.bar.type = fn_decl @StaticMethodStruct.bar [concrete = constants.%StaticMethodStruct.bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %StaticMethodStruct) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_static_method.ref: %foo_static_method.cpp_overload_set.type = name_ref foo_static_method, imports.%foo_static_method.cpp_overload_set.value [concrete = constants.%foo_static_method.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %StaticMethodStruct = name_ref s, %s
// CHECK:STDOUT:   %.loc8: ref %StaticMethodStruct = value_as_ref %s.ref
// CHECK:STDOUT:   %addr: %ptr.f77 = addr_of %.loc8
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_static_method__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %StaticMethodStruct.ref.loc9: type = name_ref StaticMethodStruct, imports.%StaticMethodStruct.decl [concrete = constants.%StaticMethodStruct]
// CHECK:STDOUT:   %bar.ref: %StaticMethodStruct.bar.cpp_overload_set.type = name_ref bar, imports.%StaticMethodStruct.bar.cpp_overload_set.value [concrete = constants.%StaticMethodStruct.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodStruct.bar.call: init %empty_tuple.type = call imports.%StaticMethodStruct.bar.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeclPointerParamStruct: type = class_type @DeclPointerParamStruct [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeclPointerParamStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_decl_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param.cpp_overload_set.value: %foo_decl_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_decl_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param.type: type = fn_type @foo_decl_pointer_param [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param: %foo_decl_pointer_param.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DeclPointerParamStruct = %DeclPointerParamStruct.decl
// CHECK:STDOUT:     .foo_decl_pointer_param = %foo_decl_pointer_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeclPointerParamStruct.decl: type = class_decl @DeclPointerParamStruct [concrete = constants.%DeclPointerParamStruct] {} {}
// CHECK:STDOUT:   %foo_decl_pointer_param.cpp_overload_set.value: %foo_decl_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_decl_pointer_param.cpp_overload_set [concrete = constants.%foo_decl_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_decl_pointer_param.decl: %foo_decl_pointer_param.type = fn_decl @foo_decl_pointer_param [concrete = constants.%foo_decl_pointer_param] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_decl_pointer_param.ref: %foo_decl_pointer_param.cpp_overload_set.type = name_ref foo_decl_pointer_param, imports.%foo_decl_pointer_param.cpp_overload_set.value [concrete = constants.%foo_decl_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %ptr = name_ref s, %s
// CHECK:STDOUT:   %foo_decl_pointer_param.call: init %empty_tuple.type = call imports.%foo_decl_pointer_param.decl(%s.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DefPointerParamStruct: type = class_type @DefPointerParamStruct [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DefPointerParamStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_def_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param.cpp_overload_set.value: %foo_def_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_def_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param.type: type = fn_type @foo_def_pointer_param [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param: %foo_def_pointer_param.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DefPointerParamStruct = %DefPointerParamStruct.decl
// CHECK:STDOUT:     .foo_def_pointer_param = %foo_def_pointer_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefPointerParamStruct.decl: type = class_decl @DefPointerParamStruct [concrete = constants.%DefPointerParamStruct] {} {}
// CHECK:STDOUT:   %foo_def_pointer_param.cpp_overload_set.value: %foo_def_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_def_pointer_param.cpp_overload_set [concrete = constants.%foo_def_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_def_pointer_param.decl: %foo_def_pointer_param.type = fn_decl @foo_def_pointer_param [concrete = constants.%foo_def_pointer_param] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_def_pointer_param.ref: %foo_def_pointer_param.cpp_overload_set.type = name_ref foo_def_pointer_param, imports.%foo_def_pointer_param.cpp_overload_set.value [concrete = constants.%foo_def_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %ptr = name_ref s, %s
// CHECK:STDOUT:   %foo_def_pointer_param.call: init %empty_tuple.type = call imports.%foo_def_pointer_param.decl(%s.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_value_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_def_value_return.cpp_overload_set.type: type = cpp_overload_set_type @foo_def_value_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_def_value_return.cpp_overload_set.value: %foo_def_value_return.cpp_overload_set.type = cpp_overload_set_value @foo_def_value_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %DefValueReturnStruct: type = class_type @DefValueReturnStruct [concrete]
// CHECK:STDOUT:   %ptr.b68: type = ptr_type %DefValueReturnStruct [concrete]
// CHECK:STDOUT:   %foo_def_value_return__carbon_thunk.type: type = fn_type @foo_def_value_return__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_def_value_return__carbon_thunk: %foo_def_value_return__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefValueReturnStruct.cpp_destructor.type: type = fn_type @DefValueReturnStruct.cpp_destructor [concrete]
// CHECK:STDOUT:   %DefValueReturnStruct.cpp_destructor: %DefValueReturnStruct.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_def_value_return = %foo_def_value_return.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_def_value_return.cpp_overload_set.value: %foo_def_value_return.cpp_overload_set.type = cpp_overload_set_value @foo_def_value_return.cpp_overload_set [concrete = constants.%foo_def_value_return.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_def_value_return__carbon_thunk.decl: %foo_def_value_return__carbon_thunk.type = fn_decl @foo_def_value_return__carbon_thunk [concrete = constants.%foo_def_value_return__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_def_value_return.ref: %foo_def_value_return.cpp_overload_set.type = name_ref foo_def_value_return, imports.%foo_def_value_return.cpp_overload_set.value [concrete = constants.%foo_def_value_return.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_28.1: ref %DefValueReturnStruct = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.b68 = addr_of %.loc8_28.1
// CHECK:STDOUT:   %foo_def_value_return__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_def_value_return__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_28.2: init %DefValueReturnStruct to %.loc8_28.1 = mark_in_place_init %foo_def_value_return__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_28.3: ref %DefValueReturnStruct = temporary %.loc8_28.1, %.loc8_28.2
// CHECK:STDOUT:   %DefValueReturnStruct.cpp_destructor.bound: <bound method> = bound_method %.loc8_28.3, constants.%DefValueReturnStruct.cpp_destructor
// CHECK:STDOUT:   %DefValueReturnStruct.cpp_destructor.call: init %empty_tuple.type = call %DefValueReturnStruct.cpp_destructor.bound(%.loc8_28.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo_decl_pointer_return.cpp_overload_set.type: type = cpp_overload_set_type @foo_decl_pointer_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_return.cpp_overload_set.value: %foo_decl_pointer_return.cpp_overload_set.type = cpp_overload_set_value @foo_decl_pointer_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %DeclPointerReturnStruct: type = class_type @DeclPointerReturnStruct [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeclPointerReturnStruct [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_return.type: type = fn_type @foo_decl_pointer_return [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_return: %foo_decl_pointer_return.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_decl_pointer_return = %foo_decl_pointer_return.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_decl_pointer_return.cpp_overload_set.value: %foo_decl_pointer_return.cpp_overload_set.type = cpp_overload_set_value @foo_decl_pointer_return.cpp_overload_set [concrete = constants.%foo_decl_pointer_return.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_decl_pointer_return.decl: %foo_decl_pointer_return.type = fn_decl @foo_decl_pointer_return [concrete = constants.%foo_decl_pointer_return] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_decl_pointer_return.ref: %foo_decl_pointer_return.cpp_overload_set.type = name_ref foo_decl_pointer_return, imports.%foo_decl_pointer_return.cpp_overload_set.value [concrete = constants.%foo_decl_pointer_return.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_decl_pointer_return.call: init %ptr = call imports.%foo_decl_pointer_return.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo_def_pointer_return.cpp_overload_set.type: type = cpp_overload_set_type @foo_def_pointer_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_def_pointer_return.cpp_overload_set.value: %foo_def_pointer_return.cpp_overload_set.type = cpp_overload_set_value @foo_def_pointer_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %DefPointerReturnStruct: type = class_type @DefPointerReturnStruct [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DefPointerReturnStruct [concrete]
// CHECK:STDOUT:   %foo_def_pointer_return.type: type = fn_type @foo_def_pointer_return [concrete]
// CHECK:STDOUT:   %foo_def_pointer_return: %foo_def_pointer_return.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_def_pointer_return = %foo_def_pointer_return.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_def_pointer_return.cpp_overload_set.value: %foo_def_pointer_return.cpp_overload_set.type = cpp_overload_set_value @foo_def_pointer_return.cpp_overload_set [concrete = constants.%foo_def_pointer_return.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_def_pointer_return.decl: %foo_def_pointer_return.type = fn_decl @foo_def_pointer_return [concrete = constants.%foo_def_pointer_return] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_def_pointer_return.ref: %foo_def_pointer_return.cpp_overload_set.type = name_ref foo_def_pointer_return, imports.%foo_def_pointer_return.cpp_overload_set.value [concrete = constants.%foo_def_pointer_return.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_def_pointer_return.call: init %ptr = call imports.%foo_def_pointer_return.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
