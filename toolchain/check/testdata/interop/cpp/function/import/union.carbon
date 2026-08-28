// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/union.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/union.carbon

// ============================================================================
// Forward-declared union as parameter type
// ============================================================================

// --- decl_value_param_type.h

union ForwardDeclaredUnion;

auto foo1(ForwardDeclaredUnion) -> void;
auto foo2(ForwardDeclaredUnion) -> void;

// --- fail_import_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:4:6-9: error: argument type 'ForwardDeclaredUnion' is incomplete [CppInteropParseError]
// CHECK:STDERR: ./decl_value_param_type.h:2:7-26: note: forward declaration of 'ForwardDeclaredUnion' [CppInteropParseNote]
import Cpp library "decl_value_param_type.h";

fn F(p: Cpp.ForwardDeclaredUnion*) {
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+5]]:3-14: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  // CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE-5]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:5:6-9: error: argument type 'ForwardDeclaredUnion' is incomplete [CppInteropParseError]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7-26: note: forward declaration of 'ForwardDeclaredUnion' [CppInteropParseNote]
  Cpp.foo1(*p);
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+12]]:3-14: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-3]]:12-13: error: forming value of incomplete type `Cpp.ForwardDeclaredUnion` [IncompleteTypeInValueConversion]
  // CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE-12]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: ./decl_value_param_type.h:4:31: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+5]]:12-13: error: forming value of incomplete type `Cpp.ForwardDeclaredUnion` [IncompleteTypeInValueConversion]
  // CHECK:STDERR: included from: fail_import_decl_value_param_type.carbon:[[@LINE-17]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: ./decl_value_param_type.h:5:31: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.foo2(*p);
}

// ============================================================================
// Defined union without data members as parameter type
// ============================================================================

// --- definition_no_data_members_value_param_type.h

union EmptyUnion {};

auto foo_empty(EmptyUnion) -> void;

// --- import_definition_no_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_no_data_members_value_param_type.h";

fn F(u: Cpp.EmptyUnion) {
  //@dump-sem-ir-begin
  Cpp.foo_empty(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union with a single data member as parameter type
// ============================================================================

// --- definition_single_data_member_value_param_type.h

struct SingleMemberS {};

union SingleMemberUnion {
  SingleMemberS s;
};

auto foo_single_member(SingleMemberUnion) -> void;

// --- import_definition_single_data_member_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_single_data_member_value_param_type.h";

fn F(u: Cpp.SingleMemberUnion) {
  //@dump-sem-ir-begin
  Cpp.foo_single_member(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union with multiple data members as parameter type
// ============================================================================

// --- definition_multiple_data_members_value_param_type.h

struct MultipleMembersS {};

union MultipleMembersUnion {
  MultipleMembersS s1;
  MultipleMembersS s2;
  MultipleMembersS s3;
};

auto foo_multiple_members(MultipleMembersUnion) -> void;

// --- import_definition_multiple_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_multiple_data_members_value_param_type.h";

fn F(u: Cpp.MultipleMembersUnion) {
  //@dump-sem-ir-begin
  Cpp.foo_multiple_members(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union in namespace
// ============================================================================

// --- definition_in_namespace_value_param_type.h

namespace N { union InNamespaceUnion {}; }

auto foo_in_namespace(N::InNamespaceUnion) -> void;

// --- import_definition_in_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_namespace_value_param_type.h";

fn F(u: Cpp.N.InNamespaceUnion) {
  //@dump-sem-ir-begin
  Cpp.foo_in_namespace(u);
  // Check that the parameter type was imported correctly.
  var unused x: Cpp.N.InNamespaceUnion;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union in relative namespace
// ============================================================================

// --- definition_in_relative_namespace_value_param_type.h

namespace N1 {
  namespace N2 { union InRelativeNamespaceUnion {}; }
  auto foo(N2::InRelativeNamespaceUnion) -> void;
}

// --- import_definition_in_relative_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_relative_namespace_value_param_type.h";

fn F(u: Cpp.N1.N2.InRelativeNamespaceUnion) {
  //@dump-sem-ir-begin
  Cpp.N1.foo(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union in union
// ============================================================================

// --- definition_in_outer_definition.h

union OuterUnion {
 public:
  union InnerUnion {};
};

auto foo_in_outer(OuterUnion::InnerUnion) -> void;

// --- import_definition_in_outer_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_outer_definition.h";

fn F(u: Cpp.OuterUnion.InnerUnion) {
  //@dump-sem-ir-begin
  Cpp.foo_in_outer(u);
  var unused x: Cpp.OuterUnion;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union and explicitly used
// ============================================================================

// --- definition_with_static_method.h

union StaticMethodUnion {
  static void bar();
};

auto foo_static_method(StaticMethodUnion) -> void;

// --- import_definition_and_static_method_call_before.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F(u: Cpp.StaticMethodUnion) {
  //@dump-sem-ir-begin
  Cpp.StaticMethodUnion.bar();
  Cpp.foo_static_method(u);
  //@dump-sem-ir-end
}

// --- import_definition_and_static_method_call_after.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F(u: Cpp.StaticMethodUnion) {
  //@dump-sem-ir-begin
  Cpp.foo_static_method(u);
  Cpp.StaticMethodUnion.bar();
  //@dump-sem-ir-end
}

// ============================================================================
// Incomplete union
// ============================================================================

// --- incomplete.h

union IncompleteUnion;

// --- fail_import_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp library "incomplete.h";

fn F() {
  // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE+4]]:3-25: error: member access into incomplete class `Cpp.IncompleteUnion` [QualifiedExprInIncompleteClassScope]
  // CHECK:STDERR: included from: fail_import_incomplete.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./incomplete.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  Cpp.IncompleteUnion.foo();
}

// ============================================================================
// Pointer to forward-declared union as parameter type
// ============================================================================

// --- decl_pointer_param_type.h

union DeclPointerParamUnion;

auto foo_decl_pointer_param(DeclPointerParamUnion* _Nonnull) -> void;

// --- import_decl_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_param_type.h";

fn F(u: Cpp.DeclPointerParamUnion*) {
  //@dump-sem-ir-begin
  Cpp.foo_decl_pointer_param(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined union as parameter type
// ============================================================================

// --- definition_pointer_param_type.h

union DefPointerParamUnion {};

auto foo_def_pointer_param(DefPointerParamUnion* _Nonnull) -> void;

// --- import_definition_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_pointer_param_type.h";

fn F(u: Cpp.DefPointerParamUnion*) {
  //@dump-sem-ir-begin
  Cpp.foo_def_pointer_param(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Forward-declared union as return type
// ============================================================================

// --- decl_value_return_type.h

union DeclValueReturnUnion;

auto foo_decl_value_return() -> DeclValueReturnUnion;

// --- fail_import_decl_value_return_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_decl_value_return_type.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:4:6-26: error: calling 'foo_decl_value_return' with incomplete return type 'DeclValueReturnUnion' [CppInteropParseError]
// CHECK:STDERR: ./decl_value_return_type.h:4:6-26: note: 'foo_decl_value_return' declared here [CppInteropParseNote]
// CHECK:STDERR: ./decl_value_return_type.h:2:7-26: note: forward declaration of 'DeclValueReturnUnion' [CppInteropParseNote]
import Cpp library "decl_value_return_type.h";

fn F() {
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+7]]:3-29: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+5]]:3-29: error: function returns incomplete type `Cpp.DeclValueReturnUnion` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR: included from: fail_import_decl_value_return_type.carbon:[[@LINE-6]]:10 [InCppInclude]
  // CHECK:STDERR: ./decl_value_return_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: ./decl_value_return_type.h:4:1: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR:
  Cpp.foo_decl_value_return();
}

// ============================================================================
// Defined union as return type
// ============================================================================

// --- definition_value_return_type.h

union DefValueReturnUnion {};

auto foo_def_value_return() -> DefValueReturnUnion;

// --- import_definition_value_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_value_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo_def_value_return();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to forward-declared union as return type
// ============================================================================

// --- decl_pointer_return_type.h

union DeclPointerReturnUnion;

auto foo_decl_pointer_return() -> DeclPointerReturnUnion* _Nonnull;

// --- import_decl_pointer_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo_decl_pointer_return();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined union as return type
// ============================================================================

// --- definition_pointer_return_type.h

union DefPointerReturnUnion {};

auto foo_def_pointer_return() -> DefPointerReturnUnion* _Nonnull;

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
// CHECK:STDOUT:   %EmptyUnion: type = class_type @EmptyUnion [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_empty.cpp_overload_set.type: type = cpp_overload_set_type @foo_empty.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_empty.cpp_overload_set.value: %foo_empty.cpp_overload_set.type = cpp_overload_set_value @foo_empty.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.73e: type = ptr_type %EmptyUnion [concrete]
// CHECK:STDOUT:   %foo_empty__carbon_thunk.type: type = fn_type @foo_empty__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_empty__carbon_thunk: %foo_empty__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .EmptyUnion = %EmptyUnion.decl
// CHECK:STDOUT:     .foo_empty = %foo_empty.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyUnion.decl: type = class_decl @EmptyUnion [concrete = constants.%EmptyUnion] {} {}
// CHECK:STDOUT:   %foo_empty.cpp_overload_set.value: %foo_empty.cpp_overload_set.type = cpp_overload_set_value @foo_empty.cpp_overload_set [concrete = constants.%foo_empty.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_empty__carbon_thunk.decl: %foo_empty__carbon_thunk.type = fn_decl @foo_empty__carbon_thunk [concrete = constants.%foo_empty__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %EmptyUnion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_empty.ref: %foo_empty.cpp_overload_set.type = name_ref foo_empty, imports.%foo_empty.cpp_overload_set.value [concrete = constants.%foo_empty.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %EmptyUnion = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %EmptyUnion = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.73e = addr_of %.loc8
// CHECK:STDOUT:   %foo_empty__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_empty__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_single_data_member_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SingleMemberUnion: type = class_type @SingleMemberUnion [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_single_member.cpp_overload_set.type: type = cpp_overload_set_type @foo_single_member.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_single_member.cpp_overload_set.value: %foo_single_member.cpp_overload_set.type = cpp_overload_set_value @foo_single_member.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.6f8: type = ptr_type %SingleMemberUnion [concrete]
// CHECK:STDOUT:   %foo_single_member__carbon_thunk.type: type = fn_type @foo_single_member__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_single_member__carbon_thunk: %foo_single_member__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .SingleMemberUnion = %SingleMemberUnion.decl
// CHECK:STDOUT:     .foo_single_member = %foo_single_member.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SingleMemberUnion.decl: type = class_decl @SingleMemberUnion [concrete = constants.%SingleMemberUnion] {} {}
// CHECK:STDOUT:   %foo_single_member.cpp_overload_set.value: %foo_single_member.cpp_overload_set.type = cpp_overload_set_value @foo_single_member.cpp_overload_set [concrete = constants.%foo_single_member.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_single_member__carbon_thunk.decl: %foo_single_member__carbon_thunk.type = fn_decl @foo_single_member__carbon_thunk [concrete = constants.%foo_single_member__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %SingleMemberUnion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_single_member.ref: %foo_single_member.cpp_overload_set.type = name_ref foo_single_member, imports.%foo_single_member.cpp_overload_set.value [concrete = constants.%foo_single_member.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %SingleMemberUnion = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %SingleMemberUnion = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.6f8 = addr_of %.loc8
// CHECK:STDOUT:   %foo_single_member__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_single_member__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_multiple_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MultipleMembersUnion: type = class_type @MultipleMembersUnion [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_multiple_members.cpp_overload_set.type: type = cpp_overload_set_type @foo_multiple_members.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_multiple_members.cpp_overload_set.value: %foo_multiple_members.cpp_overload_set.type = cpp_overload_set_value @foo_multiple_members.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.f0d: type = ptr_type %MultipleMembersUnion [concrete]
// CHECK:STDOUT:   %foo_multiple_members__carbon_thunk.type: type = fn_type @foo_multiple_members__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_multiple_members__carbon_thunk: %foo_multiple_members__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MultipleMembersUnion = %MultipleMembersUnion.decl
// CHECK:STDOUT:     .foo_multiple_members = %foo_multiple_members.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MultipleMembersUnion.decl: type = class_decl @MultipleMembersUnion [concrete = constants.%MultipleMembersUnion] {} {}
// CHECK:STDOUT:   %foo_multiple_members.cpp_overload_set.value: %foo_multiple_members.cpp_overload_set.type = cpp_overload_set_value @foo_multiple_members.cpp_overload_set [concrete = constants.%foo_multiple_members.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_multiple_members__carbon_thunk.decl: %foo_multiple_members__carbon_thunk.type = fn_decl @foo_multiple_members__carbon_thunk [concrete = constants.%foo_multiple_members__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %MultipleMembersUnion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_multiple_members.ref: %foo_multiple_members.cpp_overload_set.type = name_ref foo_multiple_members, imports.%foo_multiple_members.cpp_overload_set.value [concrete = constants.%foo_multiple_members.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %MultipleMembersUnion = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %MultipleMembersUnion = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.f0d = addr_of %.loc8
// CHECK:STDOUT:   %foo_multiple_members__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_multiple_members__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InNamespaceUnion: type = class_type @InNamespaceUnion [concrete]
// CHECK:STDOUT:   %pattern_type.007: type = pattern_type %InNamespaceUnion [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_in_namespace.cpp_overload_set.type: type = cpp_overload_set_type @foo_in_namespace.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_in_namespace.cpp_overload_set.value: %foo_in_namespace.cpp_overload_set.type = cpp_overload_set_value @foo_in_namespace.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.e1a: type = ptr_type %InNamespaceUnion [concrete]
// CHECK:STDOUT:   %foo_in_namespace__carbon_thunk.type: type = fn_type @foo_in_namespace__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_in_namespace__carbon_thunk: %foo_in_namespace__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.007 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.007 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %InNamespaceUnion.Op.type: type = fn_type @InNamespaceUnion.Op [concrete]
// CHECK:STDOUT:   %InNamespaceUnion.Op: %InNamespaceUnion.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.85f: <witness> = custom_witness (%InNamespaceUnion.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.c4a: %Default.type = facet_value %InNamespaceUnion, (%custom_witness.85f) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.ee1: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.162, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.c4a) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %InNamespaceUnion, (%DefaultOrUnformed.impl_witness.ee1) [concrete]
// CHECK:STDOUT:   %InNamespaceUnion.cpp_destructor.type: type = fn_type @InNamespaceUnion.cpp_destructor [concrete]
// CHECK:STDOUT:   %InNamespaceUnion.cpp_destructor: %InNamespaceUnion.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .foo_in_namespace = %foo_in_namespace.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .InNamespaceUnion = %InNamespaceUnion.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InNamespaceUnion.decl: type = class_decl @InNamespaceUnion [concrete = constants.%InNamespaceUnion] {} {}
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
// CHECK:STDOUT: fn @F(%u.param: %InNamespaceUnion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_in_namespace.ref: %foo_in_namespace.cpp_overload_set.type = name_ref foo_in_namespace, imports.%foo_in_namespace.cpp_overload_set.value [concrete = constants.%foo_in_namespace.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %InNamespaceUnion = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %InNamespaceUnion = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.e1a = addr_of %.loc8
// CHECK:STDOUT:   %foo_in_namespace__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_in_namespace__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %x.var: ref %InNamespaceUnion = var_storage %x.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%InNamespaceUnion, (constants.%DefaultOrUnformed.impl_witness.ee1) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc10_39.1: %DefaultOrUnformed.type = converted constants.%InNamespaceUnion, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_39.1 [concrete = constants.%InNamespaceUnion]
// CHECK:STDOUT:   %.loc10_39.2: type = converted %.loc10_39.1, %as_type [concrete = constants.%InNamespaceUnion]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_3: ref %InNamespaceUnion = splice_block %x.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %InNamespaceUnion to %.loc10_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %x.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc10_22: type = splice_block %InNamespaceUnion.ref.loc10 [concrete = constants.%InNamespaceUnion] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %InNamespaceUnion.ref.loc10: type = name_ref InNamespaceUnion, imports.%InNamespaceUnion.decl [concrete = constants.%InNamespaceUnion]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %InNamespaceUnion = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.007 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.007 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InNamespaceUnion.cpp_destructor.bound: <bound method> = bound_method %x.var, constants.%InNamespaceUnion.cpp_destructor
// CHECK:STDOUT:   %InNamespaceUnion.cpp_destructor.call: init %empty_tuple.type = call %InNamespaceUnion.cpp_destructor.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_relative_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InRelativeNamespaceUnion: type = class_type @InRelativeNamespaceUnion [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.cfd: type = ptr_type %InRelativeNamespaceUnion [concrete]
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
// CHECK:STDOUT:     .InRelativeNamespaceUnion = %InRelativeNamespaceUnion.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InRelativeNamespaceUnion.decl: type = class_decl @InRelativeNamespaceUnion [concrete = constants.%InRelativeNamespaceUnion] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %InRelativeNamespaceUnion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc8: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %InRelativeNamespaceUnion = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %InRelativeNamespaceUnion = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.cfd = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_outer_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %OuterUnion: type = class_type @OuterUnion [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %InnerUnion: type = class_type @InnerUnion [concrete]
// CHECK:STDOUT:   %foo_in_outer.cpp_overload_set.type: type = cpp_overload_set_type @foo_in_outer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_in_outer.cpp_overload_set.value: %foo_in_outer.cpp_overload_set.type = cpp_overload_set_value @foo_in_outer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.63b: type = ptr_type %InnerUnion [concrete]
// CHECK:STDOUT:   %foo_in_outer__carbon_thunk.type: type = fn_type @foo_in_outer__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_in_outer__carbon_thunk: %foo_in_outer__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.89d: type = pattern_type %OuterUnion [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.89d = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.89d = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %OuterUnion.Op.type: type = fn_type @OuterUnion.Op [concrete]
// CHECK:STDOUT:   %OuterUnion.Op: %OuterUnion.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b4: <witness> = custom_witness (%OuterUnion.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.a9b: %Default.type = facet_value %OuterUnion, (%custom_witness.9b4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.661: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.162, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.a9b) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %OuterUnion, (%DefaultOrUnformed.impl_witness.661) [concrete]
// CHECK:STDOUT:   %OuterUnion.cpp_destructor.type: type = fn_type @OuterUnion.cpp_destructor [concrete]
// CHECK:STDOUT:   %OuterUnion.cpp_destructor: %OuterUnion.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OuterUnion = %OuterUnion.decl
// CHECK:STDOUT:     .foo_in_outer = %foo_in_outer.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OuterUnion.decl: type = class_decl @OuterUnion [concrete = constants.%OuterUnion] {} {}
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
// CHECK:STDOUT: fn @F(%u.param: %InnerUnion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_in_outer.ref: %foo_in_outer.cpp_overload_set.type = name_ref foo_in_outer, imports.%foo_in_outer.cpp_overload_set.value [concrete = constants.%foo_in_outer.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %InnerUnion = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %InnerUnion = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.63b = addr_of %.loc8
// CHECK:STDOUT:   %foo_in_outer__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_in_outer__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %x.var: ref %OuterUnion = var_storage %x.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%OuterUnion, (constants.%DefaultOrUnformed.impl_witness.661) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc9_31.1: %DefaultOrUnformed.type = converted constants.%OuterUnion, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc9_31.1 [concrete = constants.%OuterUnion]
// CHECK:STDOUT:   %.loc9_31.2: type = converted %.loc9_31.1, %as_type [concrete = constants.%OuterUnion]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9_3: ref %OuterUnion = splice_block %x.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %OuterUnion to %.loc9_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %x.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %OuterUnion.ref.loc9 [concrete = constants.%OuterUnion] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OuterUnion.ref.loc9: type = name_ref OuterUnion, imports.%OuterUnion.decl [concrete = constants.%OuterUnion]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %OuterUnion = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.89d = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.89d = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OuterUnion.cpp_destructor.bound: <bound method> = bound_method %x.var, constants.%OuterUnion.cpp_destructor
// CHECK:STDOUT:   %OuterUnion.cpp_destructor.call: init %empty_tuple.type = call %OuterUnion.cpp_destructor.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %StaticMethodUnion: type = class_type @StaticMethodUnion [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %StaticMethodUnion.bar.cpp_overload_set.type: type = cpp_overload_set_type @StaticMethodUnion.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodUnion.bar.cpp_overload_set.value: %StaticMethodUnion.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodUnion.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodUnion.bar.type: type = fn_type @StaticMethodUnion.bar [concrete]
// CHECK:STDOUT:   %StaticMethodUnion.bar: %StaticMethodUnion.bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.type: type = cpp_overload_set_type @foo_static_method.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.value: %foo_static_method.cpp_overload_set.type = cpp_overload_set_value @foo_static_method.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.9f6: type = ptr_type %StaticMethodUnion [concrete]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.type: type = fn_type @foo_static_method__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk: %foo_static_method__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .StaticMethodUnion = %StaticMethodUnion.decl
// CHECK:STDOUT:     .foo_static_method = %foo_static_method.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMethodUnion.decl: type = class_decl @StaticMethodUnion [concrete = constants.%StaticMethodUnion] {} {}
// CHECK:STDOUT:   %StaticMethodUnion.bar.cpp_overload_set.value: %StaticMethodUnion.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodUnion.bar.cpp_overload_set [concrete = constants.%StaticMethodUnion.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodUnion.bar.decl: %StaticMethodUnion.bar.type = fn_decl @StaticMethodUnion.bar [concrete = constants.%StaticMethodUnion.bar] {} {}
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.value: %foo_static_method.cpp_overload_set.type = cpp_overload_set_value @foo_static_method.cpp_overload_set [concrete = constants.%foo_static_method.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.decl: %foo_static_method__carbon_thunk.type = fn_decl @foo_static_method__carbon_thunk [concrete = constants.%foo_static_method__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %StaticMethodUnion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %StaticMethodUnion.ref.loc8: type = name_ref StaticMethodUnion, imports.%StaticMethodUnion.decl [concrete = constants.%StaticMethodUnion]
// CHECK:STDOUT:   %bar.ref: %StaticMethodUnion.bar.cpp_overload_set.type = name_ref bar, imports.%StaticMethodUnion.bar.cpp_overload_set.value [concrete = constants.%StaticMethodUnion.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodUnion.bar.call: init %empty_tuple.type = call imports.%StaticMethodUnion.bar.decl()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_static_method.ref: %foo_static_method.cpp_overload_set.type = name_ref foo_static_method, imports.%foo_static_method.cpp_overload_set.value [concrete = constants.%foo_static_method.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %StaticMethodUnion = name_ref u, %u
// CHECK:STDOUT:   %.loc9: ref %StaticMethodUnion = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.9f6 = addr_of %.loc9
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_static_method__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %StaticMethodUnion: type = class_type @StaticMethodUnion [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.type: type = cpp_overload_set_type @foo_static_method.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.value: %foo_static_method.cpp_overload_set.type = cpp_overload_set_value @foo_static_method.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.9f6: type = ptr_type %StaticMethodUnion [concrete]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.type: type = fn_type @foo_static_method__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk: %foo_static_method__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %StaticMethodUnion.bar.cpp_overload_set.type: type = cpp_overload_set_type @StaticMethodUnion.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodUnion.bar.cpp_overload_set.value: %StaticMethodUnion.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodUnion.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMethodUnion.bar.type: type = fn_type @StaticMethodUnion.bar [concrete]
// CHECK:STDOUT:   %StaticMethodUnion.bar: %StaticMethodUnion.bar.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .StaticMethodUnion = %StaticMethodUnion.decl
// CHECK:STDOUT:     .foo_static_method = %foo_static_method.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMethodUnion.decl: type = class_decl @StaticMethodUnion [concrete = constants.%StaticMethodUnion] {} {}
// CHECK:STDOUT:   %foo_static_method.cpp_overload_set.value: %foo_static_method.cpp_overload_set.type = cpp_overload_set_value @foo_static_method.cpp_overload_set [concrete = constants.%foo_static_method.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.decl: %foo_static_method__carbon_thunk.type = fn_decl @foo_static_method__carbon_thunk [concrete = constants.%foo_static_method__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMethodUnion.bar.cpp_overload_set.value: %StaticMethodUnion.bar.cpp_overload_set.type = cpp_overload_set_value @StaticMethodUnion.bar.cpp_overload_set [concrete = constants.%StaticMethodUnion.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodUnion.bar.decl: %StaticMethodUnion.bar.type = fn_decl @StaticMethodUnion.bar [concrete = constants.%StaticMethodUnion.bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %StaticMethodUnion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_static_method.ref: %foo_static_method.cpp_overload_set.type = name_ref foo_static_method, imports.%foo_static_method.cpp_overload_set.value [concrete = constants.%foo_static_method.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %StaticMethodUnion = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %StaticMethodUnion = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.9f6 = addr_of %.loc8
// CHECK:STDOUT:   %foo_static_method__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_static_method__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %StaticMethodUnion.ref.loc9: type = name_ref StaticMethodUnion, imports.%StaticMethodUnion.decl [concrete = constants.%StaticMethodUnion]
// CHECK:STDOUT:   %bar.ref: %StaticMethodUnion.bar.cpp_overload_set.type = name_ref bar, imports.%StaticMethodUnion.bar.cpp_overload_set.value [concrete = constants.%StaticMethodUnion.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMethodUnion.bar.call: init %empty_tuple.type = call imports.%StaticMethodUnion.bar.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeclPointerParamUnion: type = class_type @DeclPointerParamUnion [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeclPointerParamUnion [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_decl_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param.cpp_overload_set.value: %foo_decl_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_decl_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param.type: type = fn_type @foo_decl_pointer_param [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_param: %foo_decl_pointer_param.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DeclPointerParamUnion = %DeclPointerParamUnion.decl
// CHECK:STDOUT:     .foo_decl_pointer_param = %foo_decl_pointer_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeclPointerParamUnion.decl: type = class_decl @DeclPointerParamUnion [concrete = constants.%DeclPointerParamUnion] {} {}
// CHECK:STDOUT:   %foo_decl_pointer_param.cpp_overload_set.value: %foo_decl_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_decl_pointer_param.cpp_overload_set [concrete = constants.%foo_decl_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_decl_pointer_param.decl: %foo_decl_pointer_param.type = fn_decl @foo_decl_pointer_param [concrete = constants.%foo_decl_pointer_param] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_decl_pointer_param.ref: %foo_decl_pointer_param.cpp_overload_set.type = name_ref foo_decl_pointer_param, imports.%foo_decl_pointer_param.cpp_overload_set.value [concrete = constants.%foo_decl_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %ptr = name_ref u, %u
// CHECK:STDOUT:   %foo_decl_pointer_param.call: init %empty_tuple.type = call imports.%foo_decl_pointer_param.decl(%u.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DefPointerParamUnion: type = class_type @DefPointerParamUnion [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DefPointerParamUnion [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_def_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param.cpp_overload_set.value: %foo_def_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_def_pointer_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param.type: type = fn_type @foo_def_pointer_param [concrete]
// CHECK:STDOUT:   %foo_def_pointer_param: %foo_def_pointer_param.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DefPointerParamUnion = %DefPointerParamUnion.decl
// CHECK:STDOUT:     .foo_def_pointer_param = %foo_def_pointer_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefPointerParamUnion.decl: type = class_decl @DefPointerParamUnion [concrete = constants.%DefPointerParamUnion] {} {}
// CHECK:STDOUT:   %foo_def_pointer_param.cpp_overload_set.value: %foo_def_pointer_param.cpp_overload_set.type = cpp_overload_set_value @foo_def_pointer_param.cpp_overload_set [concrete = constants.%foo_def_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_def_pointer_param.decl: %foo_def_pointer_param.type = fn_decl @foo_def_pointer_param [concrete = constants.%foo_def_pointer_param] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_def_pointer_param.ref: %foo_def_pointer_param.cpp_overload_set.type = name_ref foo_def_pointer_param, imports.%foo_def_pointer_param.cpp_overload_set.value [concrete = constants.%foo_def_pointer_param.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %ptr = name_ref u, %u
// CHECK:STDOUT:   %foo_def_pointer_param.call: init %empty_tuple.type = call imports.%foo_def_pointer_param.decl(%u.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_value_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_def_value_return.cpp_overload_set.type: type = cpp_overload_set_type @foo_def_value_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_def_value_return.cpp_overload_set.value: %foo_def_value_return.cpp_overload_set.type = cpp_overload_set_value @foo_def_value_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %DefValueReturnUnion: type = class_type @DefValueReturnUnion [concrete]
// CHECK:STDOUT:   %ptr.5e5: type = ptr_type %DefValueReturnUnion [concrete]
// CHECK:STDOUT:   %foo_def_value_return__carbon_thunk.type: type = fn_type @foo_def_value_return__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_def_value_return__carbon_thunk: %foo_def_value_return__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefValueReturnUnion.cpp_destructor.type: type = fn_type @DefValueReturnUnion.cpp_destructor [concrete]
// CHECK:STDOUT:   %DefValueReturnUnion.cpp_destructor: %DefValueReturnUnion.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc8_28.1: ref %DefValueReturnUnion = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.5e5 = addr_of %.loc8_28.1
// CHECK:STDOUT:   %foo_def_value_return__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_def_value_return__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_28.2: init %DefValueReturnUnion to %.loc8_28.1 = mark_in_place_init %foo_def_value_return__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_28.3: ref %DefValueReturnUnion = temporary %.loc8_28.1, %.loc8_28.2
// CHECK:STDOUT:   %DefValueReturnUnion.cpp_destructor.bound: <bound method> = bound_method %.loc8_28.3, constants.%DefValueReturnUnion.cpp_destructor
// CHECK:STDOUT:   %DefValueReturnUnion.cpp_destructor.call: init %empty_tuple.type = call %DefValueReturnUnion.cpp_destructor.bound(%.loc8_28.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo_decl_pointer_return.cpp_overload_set.type: type = cpp_overload_set_type @foo_decl_pointer_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_decl_pointer_return.cpp_overload_set.value: %foo_decl_pointer_return.cpp_overload_set.type = cpp_overload_set_value @foo_decl_pointer_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %DeclPointerReturnUnion: type = class_type @DeclPointerReturnUnion [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeclPointerReturnUnion [concrete]
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
// CHECK:STDOUT:   %DefPointerReturnUnion: type = class_type @DefPointerReturnUnion [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DefPointerReturnUnion [concrete]
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
