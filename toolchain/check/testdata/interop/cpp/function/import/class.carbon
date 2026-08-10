// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/class.carbon

// ============================================================================
// Forward-declared class as parameter type
// ============================================================================

// --- decl_value_param_type.h

class ForwardDeclaredClass;

auto foo_decl_param1(ForwardDeclaredClass) -> void;
auto foo_decl_param2(ForwardDeclaredClass) -> void;

// --- fail_import_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:4:6-20: error: argument type 'ForwardDeclaredClass' is incomplete [CppInteropParseError]
// CHECK:STDERR: ./decl_value_param_type.h:2:7-26: note: forward declaration of 'ForwardDeclaredClass' [CppInteropParseNote]
import Cpp library "decl_value_param_type.h";

fn F(p: Cpp.ForwardDeclaredClass*) {
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+5]]:3-25: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  // CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE-5]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:5:6-20: error: argument type 'ForwardDeclaredClass' is incomplete [CppInteropParseError]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7-26: note: forward declaration of 'ForwardDeclaredClass' [CppInteropParseNote]
  Cpp.foo_decl_param1(*p);
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+12]]:3-25: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-3]]:23-24: error: forming value of incomplete type `Cpp.ForwardDeclaredClass` [IncompleteTypeInValueConversion]
  // CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE-12]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: ./decl_value_param_type.h:4:42: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+5]]:23-24: error: forming value of incomplete type `Cpp.ForwardDeclaredClass` [IncompleteTypeInValueConversion]
  // CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE-17]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: ./decl_value_param_type.h:5:42: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.foo_decl_param2(*p);
}

// ============================================================================
// Defined class without data members as parameter type
// ============================================================================

// --- definition_no_data_members_value_param_type.h

class EmptyClass {};

auto foo_empty_param(EmptyClass) -> void;

// --- import_definition_no_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_no_data_members_value_param_type.h";

fn F(c: Cpp.EmptyClass) {
  //@dump-sem-ir-begin
  Cpp.foo_empty_param(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Non copyable class as parameter type
// ============================================================================

// --- non_copyable_param_type.h

class NonCopyableClass {
  NonCopyableClass(const NonCopyableClass&) = delete;
};

auto foo_non_copyable_param(NonCopyableClass) -> void;

// --- fail_import_non_copyable_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_non_copyable_param_type.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./non_copyable_param_type.h:6:6-27: error: call to deleted constructor of 'NonCopyableClass' [CppInteropParseError]
// CHECK:STDERR: ./non_copyable_param_type.h:3:3-18: note: 'NonCopyableClass' has been explicitly marked deleted here [CppInteropParseNote]
// CHECK:STDERR: ./non_copyable_param_type.h:6:45: note: passing argument to parameter here [CppInteropParseNote]
import Cpp library "non_copyable_param_type.h";

fn F(c: Cpp.NonCopyableClass) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_import_non_copyable_param_type.carbon:[[@LINE+2]]:3-31: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  Cpp.foo_non_copyable_param(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Move-only class as parameter type
// ============================================================================

// --- move_only_param_type.h

struct MoveOnlyClass {
  MoveOnlyClass(MoveOnlyClass&&);
};

auto foo_move_only_param(MoveOnlyClass) -> void;
auto make_move_only() -> MoveOnlyClass;
auto move_move_only(MoveOnlyClass&) -> MoveOnlyClass&&;

// --- pass_move_only_param_temporary.carbon

library "[[@TEST_NAME]]";

import Cpp library "move_only_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo_move_only_param(Cpp.make_move_only());
  //@dump-sem-ir-end
}

// --- fail_pass_move_only_param_value.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_pass_move_only_param_value.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:6:6-24: error: call to implicitly-deleted copy constructor of 'MoveOnlyClass' [CppInteropParseError]
// CHECK:STDERR: ./move_only_param_type.h:3:3-15: note: copy constructor is implicitly deleted because 'MoveOnlyClass' has a user-declared move constructor [CppInteropParseNote]
// CHECK:STDERR: ./move_only_param_type.h:6:39: note: passing argument to parameter here [CppInteropParseNote]
import Cpp library "move_only_param_type.h";

fn F(c: Cpp.MoveOnlyClass) {
  // CHECK:STDERR: fail_pass_move_only_param_value.carbon:[[@LINE+2]]:3-28: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  Cpp.foo_move_only_param(c);
}

// --- fail_pass_move_only_param_ref.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_pass_move_only_param_ref.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:6:6-24: error: call to implicitly-deleted copy constructor of 'MoveOnlyClass' [CppInteropParseError]
// CHECK:STDERR: ./move_only_param_type.h:3:3-15: note: copy constructor is implicitly deleted because 'MoveOnlyClass' has a user-declared move constructor [CppInteropParseNote]
// CHECK:STDERR: ./move_only_param_type.h:6:39: note: passing argument to parameter here [CppInteropParseNote]
import Cpp library "move_only_param_type.h";

fn F(ref c: Cpp.MoveOnlyClass) {
  // CHECK:STDERR: fail_pass_move_only_param_ref.carbon:[[@LINE+2]]:3-28: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  Cpp.foo_move_only_param(c);
}

// --- fail_todo_pass_move_only_param_ref_by_move.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_todo_pass_move_only_param_ref_by_move.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:6:6-24: error: call to implicitly-deleted copy constructor of 'MoveOnlyClass' [CppInteropParseError]
// CHECK:STDERR: ./move_only_param_type.h:3:3-15: note: copy constructor is implicitly deleted because 'MoveOnlyClass' has a user-declared move constructor [CppInteropParseNote]
// CHECK:STDERR: ./move_only_param_type.h:6:39: note: passing argument to parameter here [CppInteropParseNote]
import Cpp library "move_only_param_type.h";

fn F(ref c: Cpp.MoveOnlyClass) {
  // CHECK:STDERR: fail_todo_pass_move_only_param_ref_by_move.carbon:[[@LINE+2]]:3-52: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  Cpp.foo_move_only_param(Cpp.move_move_only(ref c));
}

// ============================================================================
// Defined class with a single data member as parameter type
// ============================================================================

// --- definition_single_data_member_value_param_type.h

class SingleMemberD {};

class SingleMemberClass {
  SingleMemberD d;
};

auto foo_single_member_param(SingleMemberClass) -> void;

// --- import_definition_single_data_member_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_single_data_member_value_param_type.h";

fn F(c: Cpp.SingleMemberClass) {
  //@dump-sem-ir-begin
  Cpp.foo_single_member_param(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined class with multiple data members as parameter type
// ============================================================================

// --- definition_multiple_data_members_value_param_type.h

class MultipleMembersD {};

class MultipleMembersClass {
  MultipleMembersD d1;
  MultipleMembersD d2;
  MultipleMembersD d3;
};

auto foo_multiple_members_param(MultipleMembersClass) -> void;

// --- import_definition_multiple_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_multiple_data_members_value_param_type.h";

fn F(c: Cpp.MultipleMembersClass) {
  //@dump-sem-ir-begin
  Cpp.foo_multiple_members_param(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined class in namespace
// ============================================================================

// --- definition_in_namespace_value_param_type.h

namespace N { class InNamespaceClass {}; }

auto foo_in_namespace_param(N::InNamespaceClass) -> void;

// --- import_definition_in_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_namespace_value_param_type.h";

fn F(c: Cpp.N.InNamespaceClass) {
  //@dump-sem-ir-begin
  Cpp.foo_in_namespace_param(c);
  // Check that the parameter type was imported correctly.
  var unused x: Cpp.N.InNamespaceClass;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined class in relative namespace
// ============================================================================

// --- definition_in_relative_namespace_value_param_type.h

namespace N1 {
  namespace N2 { class InRelativeNamespaceClass {}; }
  auto foo(N2::InRelativeNamespaceClass) -> void;
}

// --- import_definition_in_relative_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_relative_namespace_value_param_type.h";

fn F(c: Cpp.N1.N2.InRelativeNamespaceClass) {
  //@dump-sem-ir-begin
  Cpp.N1.foo(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined class in class
// ============================================================================

// --- definition_in_outer_definition.h

class OuterClass {
 public:
  class InnerClass {};
};

auto foo_in_outer_param(OuterClass::InnerClass) -> void;

// --- import_definition_in_outer_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_outer_definition.h";

fn F(c: Cpp.OuterClass.InnerClass) {
  //@dump-sem-ir-begin
  Cpp.foo_in_outer_param(c);
  var unused x: Cpp.OuterClass;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined class and explicitly used
// ============================================================================

// --- definition_with_static_method.h

class StaticMethodClass {
 public:
  static void bar();
};

auto foo_static_method_param(StaticMethodClass) -> void;

// --- import_definition_and_static_method_call_before.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F(c: Cpp.StaticMethodClass) {
  //@dump-sem-ir-begin
  Cpp.StaticMethodClass.bar();
  Cpp.foo_static_method_param(c);
  //@dump-sem-ir-end
}

// --- import_definition_and_static_method_call_after.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F(c: Cpp.StaticMethodClass) {
  //@dump-sem-ir-begin
  Cpp.foo_static_method_param(c);
  Cpp.StaticMethodClass.bar();
  //@dump-sem-ir-end
}

// ============================================================================
// Incomplete class
// ============================================================================

// --- incomplete.h

class Incomplete;

// --- fail_import_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp library "incomplete.h";

fn F() {
  // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE+4]]:3-20: error: member access into incomplete class `Cpp.Incomplete` [QualifiedExprInIncompleteClassScope]
  // CHECK:STDERR: included from: fail_import_incomplete.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./incomplete.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  Cpp.Incomplete.foo();
}

// ============================================================================
// Pointer to forward-declared class as parameter type
// ============================================================================

// --- decl_pointer_param_type.h

class DeclPointerParamClass;

auto foo_decl_pointer_param(DeclPointerParamClass* _Nonnull) -> void;

// --- import_decl_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_param_type.h";

fn F(c: Cpp.DeclPointerParamClass*) {
  //@dump-sem-ir-begin
  Cpp.foo_decl_pointer_param(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined class as parameter type
// ============================================================================

// --- definition_pointer_param_type.h

class DefPointerParamClass {};

auto foo_def_pointer_param(DefPointerParamClass* _Nonnull) -> void;

// --- import_definition_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_pointer_param_type.h";

fn F(c: Cpp.DefPointerParamClass*) {
  //@dump-sem-ir-begin
  Cpp.foo_def_pointer_param(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Forward-declared class as return type
// ============================================================================

// --- decl_value_return_type.h

class DeclValueReturnClass;

auto foo_decl_value_return() -> DeclValueReturnClass;

// --- fail_import_decl_value_return_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_decl_value_return_type.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:4:6-26: error: calling 'foo_decl_value_return' with incomplete return type 'DeclValueReturnClass' [CppInteropParseError]
// CHECK:STDERR: ./decl_value_return_type.h:4:6-26: note: 'foo_decl_value_return' declared here [CppInteropParseNote]
// CHECK:STDERR: ./decl_value_return_type.h:2:7-26: note: forward declaration of 'DeclValueReturnClass' [CppInteropParseNote]
import Cpp library "decl_value_return_type.h";

fn F() {
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+7]]:3-29: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+5]]:3-29: error: function returns incomplete type `Cpp.DeclValueReturnClass` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR: included from: fail_import_decl_value_return_type.carbon:[[@LINE-6]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_return_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: ./decl_value_return_type.h:4:1: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR:
  Cpp.foo_decl_value_return();
}

// ============================================================================
// Defined class as return type
// ============================================================================

// --- definition_value_return_type.h

class DefValueReturnClass {};

auto foo_def_value_return() -> DefValueReturnClass;

// --- import_definition_value_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_value_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo_def_value_return();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to forward-declared class as return type
// ============================================================================

// --- decl_pointer_return_type.h

class DeclPointerReturnClass;

auto foo_decl_pointer_return() -> DeclPointerReturnClass* _Nonnull;

// --- import_decl_pointer_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo_decl_pointer_return();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined class as return type
// ============================================================================

// --- definition_pointer_return_type.h

class DefPointerReturnClass {};

auto foo_def_pointer_return() -> DefPointerReturnClass* _Nonnull;

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
// CHECK:STDOUT:   %EmptyClass: type = class_type @EmptyClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_empty_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_empty_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_empty_param.cpp_overload_set.value: %foo_empty_param.cpp_overload_set.type = cpp_overload_set_value @foo_empty_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.3ee: type = ptr_type %EmptyClass [concrete]
// CHECK:STDOUT:   %foo_empty_param__carbon_thunk.type: type = fn_type @foo_empty_param__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_empty_param__carbon_thunk: %foo_empty_param__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .EmptyClass = %EmptyClass.decl
// CHECK:STDOUT:     .foo_empty_param = %foo_empty_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyClass.decl: type = class_decl @EmptyClass [concrete = constants.%EmptyClass] {} {}
// CHECK:STDOUT:   %foo_empty_param.cpp_overload_set.value: %foo_empty_param.cpp_overload_set.type = cpp_overload_set_value @foo_empty_param.cpp_overload_set [concrete = constants.%foo_empty_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_empty_param__carbon_thunk.decl: %foo_empty_param__carbon_thunk.type = fn_decl @foo_empty_param__carbon_thunk [concrete = constants.%foo_empty_param__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %EmptyClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_empty_param.ref: %foo_empty_param.cpp_overload_set.type = name_ref foo_empty_param, imports.%foo_empty_param.cpp_overload_set.value [concrete = constants.%foo_empty_param.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %EmptyClass = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %EmptyClass = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.3ee = addr_of %.loc8
// CHECK:STDOUT:   %foo_empty_param__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_empty_param__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_non_copyable_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonCopyableClass: type = class_type @NonCopyableClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_non_copyable_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_non_copyable_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_non_copyable_param.cpp_overload_set.value: %foo_non_copyable_param.cpp_overload_set.type = cpp_overload_set_value @foo_non_copyable_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_non_copyable_param.type: type = fn_type @foo_non_copyable_param [concrete]
// CHECK:STDOUT:   %foo_non_copyable_param: %foo_non_copyable_param.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonCopyableClass = %NonCopyableClass.decl
// CHECK:STDOUT:     .foo_non_copyable_param = %foo_non_copyable_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonCopyableClass.decl: type = class_decl @NonCopyableClass [concrete = constants.%NonCopyableClass] {} {}
// CHECK:STDOUT:   %foo_non_copyable_param.cpp_overload_set.value: %foo_non_copyable_param.cpp_overload_set.type = cpp_overload_set_value @foo_non_copyable_param.cpp_overload_set [concrete = constants.%foo_non_copyable_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_non_copyable_param.decl: %foo_non_copyable_param.type = fn_decl @foo_non_copyable_param [concrete = constants.%foo_non_copyable_param] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %NonCopyableClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_non_copyable_param.ref: %foo_non_copyable_param.cpp_overload_set.type = name_ref foo_non_copyable_param, imports.%foo_non_copyable_param.cpp_overload_set.value [concrete = constants.%foo_non_copyable_param.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %NonCopyableClass = name_ref c, %c
// CHECK:STDOUT:   %foo_non_copyable_param.call: init %empty_tuple.type = call imports.%foo_non_copyable_param.decl(%c.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- pass_move_only_param_temporary.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_move_only_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_move_only_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_move_only_param.cpp_overload_set.value: %foo_move_only_param.cpp_overload_set.type = cpp_overload_set_value @foo_move_only_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %make_move_only.cpp_overload_set.type: type = cpp_overload_set_type @make_move_only.cpp_overload_set [concrete]
// CHECK:STDOUT:   %make_move_only.cpp_overload_set.value: %make_move_only.cpp_overload_set.type = cpp_overload_set_value @make_move_only.cpp_overload_set [concrete]
// CHECK:STDOUT:   %MoveOnlyClass: type = class_type @MoveOnlyClass [concrete]
// CHECK:STDOUT:   %ptr.a0f: type = ptr_type %MoveOnlyClass [concrete]
// CHECK:STDOUT:   %make_move_only__carbon_thunk.type: type = fn_type @make_move_only__carbon_thunk [concrete]
// CHECK:STDOUT:   %make_move_only__carbon_thunk: %make_move_only__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo_move_only_param__carbon_thunk.type: type = fn_type @foo_move_only_param__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_move_only_param__carbon_thunk: %foo_move_only_param__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_move_only_param = %foo_move_only_param.cpp_overload_set.value
// CHECK:STDOUT:     .make_move_only = %make_move_only.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_move_only_param.cpp_overload_set.value: %foo_move_only_param.cpp_overload_set.type = cpp_overload_set_value @foo_move_only_param.cpp_overload_set [concrete = constants.%foo_move_only_param.cpp_overload_set.value]
// CHECK:STDOUT:   %make_move_only.cpp_overload_set.value: %make_move_only.cpp_overload_set.type = cpp_overload_set_value @make_move_only.cpp_overload_set [concrete = constants.%make_move_only.cpp_overload_set.value]
// CHECK:STDOUT:   %make_move_only__carbon_thunk.decl: %make_move_only__carbon_thunk.type = fn_decl @make_move_only__carbon_thunk [concrete = constants.%make_move_only__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_move_only_param__carbon_thunk.decl: %foo_move_only_param__carbon_thunk.type = fn_decl @foo_move_only_param__carbon_thunk [concrete = constants.%foo_move_only_param__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_move_only_param.ref: %foo_move_only_param.cpp_overload_set.type = name_ref foo_move_only_param, imports.%foo_move_only_param.cpp_overload_set.value [concrete = constants.%foo_move_only_param.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc8_27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %make_move_only.ref: %make_move_only.cpp_overload_set.type = name_ref make_move_only, imports.%make_move_only.cpp_overload_set.value [concrete = constants.%make_move_only.cpp_overload_set.value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc8_46: %ptr.a0f = addr_of %_.var
// CHECK:STDOUT:   %make_move_only__carbon_thunk.call: init %empty_tuple.type = call imports.%make_move_only__carbon_thunk.decl(%addr.loc8_46)
// CHECK:STDOUT:   %.loc8: init %MoveOnlyClass to %_.var = mark_in_place_init %make_move_only__carbon_thunk.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc8_47: %ptr.a0f = addr_of %_.var
// CHECK:STDOUT:   %foo_move_only_param__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_move_only_param__carbon_thunk.decl(%addr.loc8_47)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_single_data_member_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SingleMemberClass: type = class_type @SingleMemberClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_single_member_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_single_member_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_single_member_param.cpp_overload_set.value: %foo_single_member_param.cpp_overload_set.type = cpp_overload_set_value @foo_single_member_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.8ba: type = ptr_type %SingleMemberClass [concrete]
// CHECK:STDOUT:   %foo_single_member_param__carbon_thunk.type: type = fn_type @foo_single_member_param__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_single_member_param__carbon_thunk: %foo_single_member_param__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .SingleMemberClass = %SingleMemberClass.decl
// CHECK:STDOUT:     .foo_single_member_param = %foo_single_member_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SingleMemberClass.decl: type = class_decl @SingleMemberClass [concrete = constants.%SingleMemberClass] {} {}
// CHECK:STDOUT:   %foo_single_member_param.cpp_overload_set.value: %foo_single_member_param.cpp_overload_set.type = cpp_overload_set_value @foo_single_member_param.cpp_overload_set [concrete = constants.%foo_single_member_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_single_member_param__carbon_thunk.decl: %foo_single_member_param__carbon_thunk.type = fn_decl @foo_single_member_param__carbon_thunk [concrete = constants.%foo_single_member_param__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %SingleMemberClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_single_member_param.ref: %foo_single_member_param.cpp_overload_set.type = name_ref foo_single_member_param, imports.%foo_single_member_param.cpp_overload_set.value [concrete = constants.%foo_single_member_param.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %SingleMemberClass = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %SingleMemberClass = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.8ba = addr_of %.loc8
// CHECK:STDOUT:   %foo_single_member_param__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_single_member_param__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_multiple_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MultipleMembersClass: type = class_type @MultipleMembersClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_multiple_members_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_multiple_members_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_multiple_members_param.cpp_overload_set.value: %foo_multiple_members_param.cpp_overload_set.type = cpp_overload_set_value @foo_multiple_members_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.772: type = ptr_type %MultipleMembersClass [concrete]
// CHECK:STDOUT:   %foo_multiple_members_param__carbon_thunk.type: type = fn_type @foo_multiple_members_param__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_multiple_members_param__carbon_thunk: %foo_multiple_members_param__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MultipleMembersClass = %MultipleMembersClass.decl
// CHECK:STDOUT:     .foo_multiple_members_param = %foo_multiple_members_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MultipleMembersClass.decl: type = class_decl @MultipleMembersClass [concrete = constants.%MultipleMembersClass] {} {}
// CHECK:STDOUT:   %foo_multiple_members_param.cpp_overload_set.value: %foo_multiple_members_param.cpp_overload_set.type = cpp_overload_set_value @foo_multiple_members_param.cpp_overload_set [concrete = constants.%foo_multiple_members_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_multiple_members_param__carbon_thunk.decl: %foo_multiple_members_param__carbon_thunk.type = fn_decl @foo_multiple_members_param__carbon_thunk [concrete = constants.%foo_multiple_members_param__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %MultipleMembersClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_multiple_members_param.ref: %foo_multiple_members_param.cpp_overload_set.type = name_ref foo_multiple_members_param, imports.%foo_multiple_members_param.cpp_overload_set.value [concrete = constants.%foo_multiple_members_param.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %MultipleMembersClass = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %MultipleMembersClass = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.772 = addr_of %.loc8
// CHECK:STDOUT:   %foo_multiple_members_param__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_multiple_members_param__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InNamespaceClass: type = class_type @InNamespaceClass [concrete]
// CHECK:STDOUT:   %pattern_type.d8e: type = pattern_type %InNamespaceClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_in_namespace_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_in_namespace_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_in_namespace_param.cpp_overload_set.value: %foo_in_namespace_param.cpp_overload_set.type = cpp_overload_set_value @foo_in_namespace_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.b66: type = ptr_type %InNamespaceClass [concrete]
// CHECK:STDOUT:   %foo_in_namespace_param__carbon_thunk.type: type = fn_type @foo_in_namespace_param__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_in_namespace_param__carbon_thunk: %foo_in_namespace_param__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d8e = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.d8e = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %InNamespaceClass.Op.type: type = fn_type @InNamespaceClass.Op [concrete]
// CHECK:STDOUT:   %InNamespaceClass.Op: %InNamespaceClass.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.85f: <witness> = custom_witness (%InNamespaceClass.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.9a6: %Default.type = facet_value %InNamespaceClass, (%custom_witness.85f) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.4fe: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.162, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.9a6) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %InNamespaceClass, (%DefaultOrUnformed.impl_witness.4fe) [concrete]
// CHECK:STDOUT:   %InNamespaceClass.cpp_destructor.type: type = fn_type @InNamespaceClass.cpp_destructor [concrete]
// CHECK:STDOUT:   %InNamespaceClass.cpp_destructor: %InNamespaceClass.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .foo_in_namespace_param = %foo_in_namespace_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .InNamespaceClass = %InNamespaceClass.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InNamespaceClass.decl: type = class_decl @InNamespaceClass [concrete = constants.%InNamespaceClass] {} {}
// CHECK:STDOUT:   %foo_in_namespace_param.cpp_overload_set.value: %foo_in_namespace_param.cpp_overload_set.type = cpp_overload_set_value @foo_in_namespace_param.cpp_overload_set [concrete = constants.%foo_in_namespace_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_in_namespace_param__carbon_thunk.decl: %foo_in_namespace_param__carbon_thunk.type = fn_decl @foo_in_namespace_param__carbon_thunk [concrete = constants.%foo_in_namespace_param__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.c2f: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.cfe)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.162 = impl_witness_table (%Core.import_ref.c2f), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %InNamespaceClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_in_namespace_param.ref: %foo_in_namespace_param.cpp_overload_set.type = name_ref foo_in_namespace_param, imports.%foo_in_namespace_param.cpp_overload_set.value [concrete = constants.%foo_in_namespace_param.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %InNamespaceClass = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %InNamespaceClass = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.b66 = addr_of %.loc8
// CHECK:STDOUT:   %foo_in_namespace_param__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_in_namespace_param__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %x.var: ref %InNamespaceClass = var_storage %x.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%InNamespaceClass, (constants.%DefaultOrUnformed.impl_witness.4fe) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc10_39.1: %DefaultOrUnformed.type = converted constants.%InNamespaceClass, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_39.1 [concrete = constants.%InNamespaceClass]
// CHECK:STDOUT:   %.loc10_39.2: type = converted %.loc10_39.1, %as_type [concrete = constants.%InNamespaceClass]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_3: ref %InNamespaceClass = splice_block %x.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %InNamespaceClass to %.loc10_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %x.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc10_22: type = splice_block %InNamespaceClass.ref.loc10 [concrete = constants.%InNamespaceClass] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %InNamespaceClass.ref.loc10: type = name_ref InNamespaceClass, imports.%InNamespaceClass.decl [concrete = constants.%InNamespaceClass]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %InNamespaceClass = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d8e = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d8e = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InNamespaceClass.cpp_destructor.bound: <bound method> = bound_method %x.var, constants.%InNamespaceClass.cpp_destructor
// CHECK:STDOUT:   %InNamespaceClass.cpp_destructor.call: init %empty_tuple.type = call %InNamespaceClass.cpp_destructor.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_relative_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InRelativeNamespaceClass: type = class_type @InRelativeNamespaceClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.e6f: type = ptr_type %InRelativeNamespaceClass [concrete]
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
// CHECK:STDOUT:     .InRelativeNamespaceClass = %InRelativeNamespaceClass.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InRelativeNamespaceClass.decl: type = class_decl @InRelativeNamespaceClass [concrete = constants.%InRelativeNamespaceClass] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %InRelativeNamespaceClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc8: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %InRelativeNamespaceClass = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %InRelativeNamespaceClass = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.e6f = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_outer_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %OuterClass: type = class_type @OuterClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %InnerClass: type = class_type @InnerClass [concrete]
// CHECK:STDOUT:   %foo_in_outer_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_in_outer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_in_outer_param.cpp_overload_set.value: %foo_in_outer_param.cpp_overload_set.type = cpp_overload_set_value @foo_in_outer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.5e5: type = ptr_type %InnerClass [concrete]
// CHECK:STDOUT:   %foo_in_outer_param__carbon_thunk.type: type = fn_type @foo_in_outer_param__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_in_outer_param__carbon_thunk: %foo_in_outer_param__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dd5: type = pattern_type %OuterClass [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.dd5 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.dd5 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %OuterClass.Op.type: type = fn_type @OuterClass.Op [concrete]
// CHECK:STDOUT:   %OuterClass.Op: %OuterClass.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b4: <witness> = custom_witness (%OuterClass.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.3fb: %Default.type = facet_value %OuterClass, (%custom_witness.9b4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.436: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.162, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.3fb) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %OuterClass, (%DefaultOrUnformed.impl_witness.436) [concrete]
// CHECK:STDOUT:   %OuterClass.cpp_destructor.type: type = fn_type @OuterClass.cpp_destructor [concrete]
// CHECK:STDOUT:   %OuterClass.cpp_destructor: %OuterClass.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OuterClass = %OuterClass.decl
// CHECK:STDOUT:     .foo_in_outer_param = %foo_in_outer_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OuterClass.decl: type = class_decl @OuterClass [concrete = constants.%OuterClass] {} {}
// CHECK:STDOUT:   %foo_in_outer_param.cpp_overload_set.value: %foo_in_outer_param.cpp_overload_set.type = cpp_overload_set_value @foo_in_outer_param.cpp_overload_set [concrete = constants.%foo_in_outer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_in_outer_param__carbon_thunk.decl: %foo_in_outer_param__carbon_thunk.type = fn_decl @foo_in_outer_param__carbon_thunk [concrete = constants.%foo_in_outer_param__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.c2f: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.cfe)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.162 = impl_witness_table (%Core.import_ref.c2f), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %InnerClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_in_outer_param.ref: %foo_in_outer_param.cpp_overload_set.type = name_ref foo_in_outer_param, imports.%foo_in_outer_param.cpp_overload_set.value [concrete = constants.%foo_in_outer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %InnerClass = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %InnerClass = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.5e5 = addr_of %.loc8
// CHECK:STDOUT:   %foo_in_outer_param__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_in_outer_param__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %x.var: ref %OuterClass = var_storage %x.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%OuterClass, (constants.%DefaultOrUnformed.impl_witness.436) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc9_31.1: %DefaultOrUnformed.type = converted constants.%OuterClass, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc9_31.1 [concrete = constants.%OuterClass]
// CHECK:STDOUT:   %.loc9_31.2: type = converted %.loc9_31.1, %as_type [concrete = constants.%OuterClass]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9_3: ref %OuterClass = splice_block %x.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %OuterClass to %.loc9_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %x.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %OuterClass.ref.loc9 [concrete = constants.%OuterClass] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OuterClass.ref.loc9: type = name_ref OuterClass, imports.%OuterClass.decl [concrete = constants.%OuterClass]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %OuterClass = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.dd5 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.dd5 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OuterClass.cpp_destructor.bound: <bound method> = bound_method %x.var, constants.%OuterClass.cpp_destructor
// CHECK:STDOUT:   %OuterClass.cpp_destructor.call: init %empty_tuple.type = call %OuterClass.cpp_destructor.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %StaticMethodClass: type = class_type @StaticMethodClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %StaticMethodClass.bar.cpp_overload_set.type: type = cpp_overload_set_type @StaticMethodClass.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodClass.bar.cpp_overload_set.value: %StaticMethodClass.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodClass.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodClass.bar.type: type = fn_type @StaticMethodClass.bar [concrete]
// CHECK:STDOUT:   %StaticMethodClass.bar: %StaticMethodClass.bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo_static_method_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_static_method_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_static_method_param.cpp_overload_set.value: %foo_static_method_param.cpp_overload_set.type = cpp_overload_set_value @foo_static_method_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d55: type = ptr_type %StaticMethodClass [concrete]
// CHECK:STDOUT:   %foo_static_method_param__carbon_thunk.type: type = fn_type @foo_static_method_param__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_static_method_param__carbon_thunk: %foo_static_method_param__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .StaticMethodClass = %StaticMethodClass.decl
// CHECK:STDOUT:     .foo_static_method_param = %foo_static_method_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMethodClass.decl: type = class_decl @StaticMethodClass [concrete = constants.%StaticMethodClass] {} {}
// CHECK:STDOUT:   %StaticMethodClass.bar.cpp_overload_set.value: %StaticMethodClass.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodClass.bar.cpp_overload_set [concrete = constants.%StaticMethodClass.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodClass.bar.decl: %StaticMethodClass.bar.type = fn_decl @StaticMethodClass.bar [concrete = constants.%StaticMethodClass.bar] {} {}
// CHECK:STDOUT:   %foo_static_method_param.cpp_overload_set.value: %foo_static_method_param.cpp_overload_set.type = cpp_overload_set_value @foo_static_method_param.cpp_overload_set [concrete = constants.%foo_static_method_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_static_method_param__carbon_thunk.decl: %foo_static_method_param__carbon_thunk.type = fn_decl @foo_static_method_param__carbon_thunk [concrete = constants.%foo_static_method_param__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %StaticMethodClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %StaticMethodClass.ref.loc8: type = name_ref StaticMethodClass, imports.%StaticMethodClass.decl [concrete = constants.%StaticMethodClass]
// CHECK:STDOUT:   %bar.ref: %StaticMethodClass.bar.cpp_overload_set.type = name_ref bar, imports.%StaticMethodClass.bar.cpp_overload_set.value [concrete = constants.%StaticMethodClass.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodClass.bar.call: init %empty_tuple.type = call imports.%StaticMethodClass.bar.decl()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_static_method_param.ref: %foo_static_method_param.cpp_overload_set.type = name_ref foo_static_method_param, imports.%foo_static_method_param.cpp_overload_set.value [concrete = constants.%foo_static_method_param.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %StaticMethodClass = name_ref c, %c
// CHECK:STDOUT:   %.loc9: ref %StaticMethodClass = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.d55 = addr_of %.loc9
// CHECK:STDOUT:   %foo_static_method_param__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_static_method_param__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %StaticMethodClass: type = class_type @StaticMethodClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_static_method_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_static_method_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_static_method_param.cpp_overload_set.value: %foo_static_method_param.cpp_overload_set.type = cpp_overload_set_value @foo_static_method_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d55: type = ptr_type %StaticMethodClass [concrete]
// CHECK:STDOUT:   %foo_static_method_param__carbon_thunk.type: type = fn_type @foo_static_method_param__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_static_method_param__carbon_thunk: %foo_static_method_param__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %StaticMethodClass.bar.cpp_overload_set.type: type = cpp_overload_set_type @StaticMethodClass.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodClass.bar.cpp_overload_set.value: %StaticMethodClass.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodClass.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodClass.bar.type: type = fn_type @StaticMethodClass.bar [concrete]
// CHECK:STDOUT:   %StaticMethodClass.bar: %StaticMethodClass.bar.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .StaticMethodClass = %StaticMethodClass.decl
// CHECK:STDOUT:     .foo_static_method_param = %foo_static_method_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMethodClass.decl: type = class_decl @StaticMethodClass [concrete = constants.%StaticMethodClass] {} {}
// CHECK:STDOUT:   %foo_static_method_param.cpp_overload_set.value: %foo_static_method_param.cpp_overload_set.type = cpp_overload_set_value @foo_static_method_param.cpp_overload_set [concrete = constants.%foo_static_method_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_static_method_param__carbon_thunk.decl: %foo_static_method_param__carbon_thunk.type = fn_decl @foo_static_method_param__carbon_thunk [concrete = constants.%foo_static_method_param__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMethodClass.bar.cpp_overload_set.value: %StaticMethodClass.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodClass.bar.cpp_overload_set [concrete = constants.%StaticMethodClass.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodClass.bar.decl: %StaticMethodClass.bar.type = fn_decl @StaticMethodClass.bar [concrete = constants.%StaticMethodClass.bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %StaticMethodClass) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_static_method_param.ref: %foo_static_method_param.cpp_overload_set.type = name_ref foo_static_method_param, imports.%foo_static_method_param.cpp_overload_set.value [concrete = constants.%foo_static_method_param.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %StaticMethodClass = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %StaticMethodClass = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.d55 = addr_of %.loc8
// CHECK:STDOUT:   %foo_static_method_param__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_static_method_param__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %StaticMethodClass.ref.loc9: type = name_ref StaticMethodClass, imports.%StaticMethodClass.decl [concrete = constants.%StaticMethodClass]
// CHECK:STDOUT:   %bar.ref: %StaticMethodClass.bar.cpp_overload_set.type = name_ref bar, imports.%StaticMethodClass.bar.cpp_overload_set.value [concrete = constants.%StaticMethodClass.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodClass.bar.call: init %empty_tuple.type = call imports.%StaticMethodClass.bar.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeclPointerParamClass: type = class_type @DeclPointerParamClass [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeclPointerParamClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_decl_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param.cpp_overload_set.value: %foo_decl_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_decl_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param.type: type = fn_type @foo_decl_pointer_param [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param: %foo_decl_pointer_param.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DeclPointerParamClass = %DeclPointerParamClass.decl
// CHECK:STDOUT:     .foo_decl_pointer_param = %foo_decl_pointer_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeclPointerParamClass.decl: type = class_decl @DeclPointerParamClass [concrete = constants.%DeclPointerParamClass] {} {}
// CHECK:STDOUT:   %foo_decl_pointer_param.cpp_overload_set.value: %foo_decl_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_decl_pointer_param.cpp_overload_set [concrete = constants.%foo_decl_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_decl_pointer_param.decl: %foo_decl_pointer_param.type = fn_decl @foo_decl_pointer_param [concrete = constants.%foo_decl_pointer_param] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_decl_pointer_param.ref: %foo_decl_pointer_param.cpp_overload_set.type = name_ref foo_decl_pointer_param, imports.%foo_decl_pointer_param.cpp_overload_set.value [concrete = constants.%foo_decl_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %ptr = name_ref c, %c
// CHECK:STDOUT:   %foo_decl_pointer_param.call: init %empty_tuple.type = call imports.%foo_decl_pointer_param.decl(%c.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DefPointerParamClass: type = class_type @DefPointerParamClass [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DefPointerParamClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_def_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param.cpp_overload_set.value: %foo_def_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_def_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param.type: type = fn_type @foo_def_pointer_param [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param: %foo_def_pointer_param.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DefPointerParamClass = %DefPointerParamClass.decl
// CHECK:STDOUT:     .foo_def_pointer_param = %foo_def_pointer_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefPointerParamClass.decl: type = class_decl @DefPointerParamClass [concrete = constants.%DefPointerParamClass] {} {}
// CHECK:STDOUT:   %foo_def_pointer_param.cpp_overload_set.value: %foo_def_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_def_pointer_param.cpp_overload_set [concrete = constants.%foo_def_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_def_pointer_param.decl: %foo_def_pointer_param.type = fn_decl @foo_def_pointer_param [concrete = constants.%foo_def_pointer_param] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_def_pointer_param.ref: %foo_def_pointer_param.cpp_overload_set.type = name_ref foo_def_pointer_param, imports.%foo_def_pointer_param.cpp_overload_set.value [concrete = constants.%foo_def_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %ptr = name_ref c, %c
// CHECK:STDOUT:   %foo_def_pointer_param.call: init %empty_tuple.type = call imports.%foo_def_pointer_param.decl(%c.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_value_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_def_value_return.cpp_overload_set.type: type = cpp_overload_set_type @foo_def_value_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_def_value_return.cpp_overload_set.value: %foo_def_value_return.cpp_overload_set.type = cpp_overload_set_value @foo_def_value_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %DefValueReturnClass: type = class_type @DefValueReturnClass [concrete]
// CHECK:STDOUT:   %ptr.8c3: type = ptr_type %DefValueReturnClass [concrete]
// CHECK:STDOUT:   %foo_def_value_return__carbon_thunk.type: type = fn_type @foo_def_value_return__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_def_value_return__carbon_thunk: %foo_def_value_return__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefValueReturnClass.cpp_destructor.type: type = fn_type @DefValueReturnClass.cpp_destructor [concrete]
// CHECK:STDOUT:   %DefValueReturnClass.cpp_destructor: %DefValueReturnClass.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc8_28.1: ref %DefValueReturnClass = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.8c3 = addr_of %.loc8_28.1
// CHECK:STDOUT:   %foo_def_value_return__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_def_value_return__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_28.2: init %DefValueReturnClass to %.loc8_28.1 = mark_in_place_init %foo_def_value_return__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_28.3: ref %DefValueReturnClass = temporary %.loc8_28.1, %.loc8_28.2
// CHECK:STDOUT:   %DefValueReturnClass.cpp_destructor.bound: <bound method> = bound_method %.loc8_28.3, constants.%DefValueReturnClass.cpp_destructor
// CHECK:STDOUT:   %DefValueReturnClass.cpp_destructor.call: init %empty_tuple.type = call %DefValueReturnClass.cpp_destructor.bound(%.loc8_28.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo_decl_pointer_return.cpp_overload_set.type: type = cpp_overload_set_type @foo_decl_pointer_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_return.cpp_overload_set.value: %foo_decl_pointer_return.cpp_overload_set.type = cpp_overload_set_value @foo_decl_pointer_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %DeclPointerReturnClass: type = class_type @DeclPointerReturnClass [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeclPointerReturnClass [concrete]
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
// CHECK:STDOUT:   %DefPointerReturnClass: type = class_type @DefPointerReturnClass [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DefPointerReturnClass [concrete]
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
