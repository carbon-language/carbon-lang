// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/union.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/union.carbon

// ============================================================================
// Forward-declared union as parameter type
// ============================================================================

// --- decl_value_param_type.h

union U;

auto foo1(U) -> void;
auto foo2(U) -> void;

// --- fail_import_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:4:6: error: argument type 'U' is incomplete [CppInteropParseError]
// CHECK:STDERR:     4 | auto foo1(U) -> void;
// CHECK:STDERR:       |      ^~~~
// CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:2:7: note: forward declaration of 'U' [CppInteropParseNote]
// CHECK:STDERR:     2 | union U;
// CHECK:STDERR:       |       ^
import Cpp library "decl_value_param_type.h";

fn F(p: Cpp.U*) {
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+12]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo1(*p);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:5:6: error: argument type 'U' is incomplete [CppInteropParseError]
  // CHECK:STDERR:     5 | auto foo2(U) -> void;
  // CHECK:STDERR:       |      ^~~~
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: forward declaration of 'U' [CppInteropParseNote]
  // CHECK:STDERR:     2 | union U;
  // CHECK:STDERR:       |       ^
  Cpp.foo1(*p);
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+28]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo2(*p);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-5]]:12: error: forming value of incomplete type `Cpp.U` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   Cpp.foo1(*p);
  // CHECK:STDERR:            ^~
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-23]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: union U;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-27]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:4:12: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto foo1(U) -> void;
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+12]]:12: error: forming value of incomplete type `Cpp.U` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   Cpp.foo2(*p);
  // CHECK:STDERR:            ^~
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-35]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: union U;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-39]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:5:12: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto foo2(U) -> void;
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  Cpp.foo2(*p);
}

// ============================================================================
// Defined union without data members as parameter type
// ============================================================================

// --- definition_no_data_members_value_param_type.h

union U {};

auto foo(U) -> void;

// --- import_definition_no_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_no_data_members_value_param_type.h";

fn F(u: Cpp.U) {
  //@dump-sem-ir-begin
  Cpp.foo(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union with a single data member as parameter type
// ============================================================================

// --- definition_single_data_member_value_param_type.h

struct S {};

union U {
  S s;
};

auto foo(U) -> void;

// --- import_definition_single_data_member_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_single_data_member_value_param_type.h";

fn F(u: Cpp.U) {
  //@dump-sem-ir-begin
  Cpp.foo(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union with multiple data members as parameter type
// ============================================================================

// --- definition_multiple_data_members_value_param_type.h

struct S {};

union U {
  S s1;
  S s2;
  S s3;
};

auto foo(U) -> void;

// --- import_definition_multiple_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_multiple_data_members_value_param_type.h";

fn F(u: Cpp.U) {
  //@dump-sem-ir-begin
  Cpp.foo(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union in namespace
// ============================================================================

// --- definition_in_namespace_value_param_type.h

namespace N { union U {}; }

auto foo(N::U) -> void;

// --- import_definition_in_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_namespace_value_param_type.h";

fn F(u: Cpp.N.U) {
  //@dump-sem-ir-begin
  Cpp.foo(u);
  // Check that the parameter type was imported correctly.
  var unused x: Cpp.N.U;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union in relative namespace
// ============================================================================

// --- definition_in_relative_namespace_value_param_type.h

namespace N1 {
  namespace N2 { union U {}; }
  auto foo(N2::U) -> void;
}

// --- import_definition_in_relative_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_relative_namespace_value_param_type.h";

fn F(u: Cpp.N1.N2.U) {
  //@dump-sem-ir-begin
  Cpp.N1.foo(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union in union
// ============================================================================

// --- definition_in_outer_definition.h

union O {
 public:
  union U {};
};

auto foo(O::U) -> void;

// --- import_definition_in_outer_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_outer_definition.h";

fn F(u: Cpp.O.U) {
  //@dump-sem-ir-begin
  Cpp.foo(u);
  var unused x: Cpp.O;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined union and explicitly used
// ============================================================================

// --- definition_with_static_method.h

union U {
  static void bar();
};

auto foo(U) -> void;

// --- import_definition_and_static_method_call_before.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F(u: Cpp.U) {
  //@dump-sem-ir-begin
  Cpp.U.bar();
  Cpp.foo(u);
  //@dump-sem-ir-end
}

// --- import_definition_and_static_method_call_after.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F(u: Cpp.U) {
  //@dump-sem-ir-begin
  Cpp.foo(u);
  Cpp.U.bar();
  //@dump-sem-ir-end
}

// ============================================================================
// Incomplete union
// ============================================================================

// --- incomplete.h

union Incomplete;

// --- fail_import_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp library "incomplete.h";

fn F() {
  // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE+8]]:3: error: member access into incomplete class `Cpp.Incomplete` [QualifiedExprInIncompleteClassScope]
  // CHECK:STDERR:   Cpp.Incomplete.foo();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./incomplete.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: union Incomplete;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  Cpp.Incomplete.foo();
}

// ============================================================================
// Pointer to forward-declared union as parameter type
// ============================================================================

// --- decl_pointer_param_type.h

union U;

auto foo(U* _Nonnull) -> void;

// --- import_decl_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_param_type.h";

fn F(u: Cpp.U*) {
  //@dump-sem-ir-begin
  Cpp.foo(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined union as parameter type
// ============================================================================

// --- definition_pointer_param_type.h

union U {};

auto foo(U* _Nonnull) -> void;

// --- import_definition_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_pointer_param_type.h";

fn F(u: Cpp.U*) {
  //@dump-sem-ir-begin
  Cpp.foo(u);
  //@dump-sem-ir-end
}

// ============================================================================
// Forward-declared union as return type
// ============================================================================

// --- decl_value_return_type.h

union U;

auto foo() -> U;

// --- fail_import_decl_value_return_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+12]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:4:6: error: calling 'foo' with incomplete return type 'U' [CppInteropParseError]
// CHECK:STDERR:     4 | auto foo() -> U;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:4:6: note: 'foo' declared here [CppInteropParseNote]
// CHECK:STDERR:     4 | auto foo() -> U;
// CHECK:STDERR:       |      ^
// CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:2:7: note: forward declaration of 'U' [CppInteropParseNote]
// CHECK:STDERR:     2 | union U;
// CHECK:STDERR:       |       ^
import Cpp library "decl_value_return_type.h";

fn F() {
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+16]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+12]]:3: error: function returns incomplete type `Cpp.U` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_return_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: union U;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE-14]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_return_type.h:4:1: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: auto foo() -> U;
  // CHECK:STDERR: ^
  // CHECK:STDERR:
  Cpp.foo();
}

// ============================================================================
// Defined union as return type
// ============================================================================

// --- definition_value_return_type.h

union U {};

auto foo() -> U;

// --- import_definition_value_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_value_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to forward-declared union as return type
// ============================================================================

// --- decl_pointer_return_type.h

union U;

auto foo() -> U* _Nonnull;

// --- import_decl_pointer_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined union as return type
// ============================================================================

// --- definition_pointer_return_type.h

union U {};

auto foo() -> U* _Nonnull;

// --- import_definition_pointer_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_pointer_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_definition_no_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %U) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %U = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %U = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_single_data_member_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %U) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %U = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %U = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_multiple_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %U) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %U = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %U = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %pattern_type.eb9: type = pattern_type %U [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.87e: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.50c: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.DefaultOrUnformed.impl.Op.type.8d1: type = fn_type @T.binding.as_type.as.DefaultOrUnformed.impl.Op, @T.binding.as_type.as.DefaultOrUnformed.impl(%T.50c) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.DefaultOrUnformed.impl.Op.185: %T.binding.as_type.as.DefaultOrUnformed.impl.Op.type.8d1 = struct_value () [symbolic]
// CHECK:STDOUT:   %U.Op.type: type = fn_type @U.Op [concrete]
// CHECK:STDOUT:   %U.Op: %U.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cb9: <witness> = custom_witness (%U.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.916: %Default.type = facet_value %U, (%custom_witness.cb9) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.225: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.608, @T.binding.as_type.as.DefaultOrUnformed.impl(%Default.facet.916) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %U, (%DefaultOrUnformed.impl_witness.225) [concrete]
// CHECK:STDOUT:   %U.cpp_destructor.type: type = fn_type @U.cpp_destructor [concrete]
// CHECK:STDOUT:   %U.cpp_destructor: %U.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.9c6: @T.binding.as_type.as.DefaultOrUnformed.impl.%T.binding.as_type.as.DefaultOrUnformed.impl.Op.type (%T.binding.as_type.as.DefaultOrUnformed.impl.Op.type.8d1) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.DefaultOrUnformed.impl.%T.binding.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.binding.as_type.as.DefaultOrUnformed.impl.Op.185)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.608 = impl_witness_table (%Core.import_ref.9c6), @T.binding.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %U) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %U = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %U = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.87e = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.eb9 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.eb9 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %U = var %x.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%U, (constants.%DefaultOrUnformed.impl_witness.225) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc10_24.1: %DefaultOrUnformed.type = converted constants.%U, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_24.1 [concrete = constants.%U]
// CHECK:STDOUT:   %.loc10_24.2: type = converted %.loc10_24.1, %as_type [concrete = constants.%U]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_3: ref %U = splice_block %x.var {}
// CHECK:STDOUT:   %T.binding.as_type.as.DefaultOrUnformed.impl.Op.call: init %U to %.loc10_3 = call %T.binding.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %x.var, %T.binding.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc10_22: type = splice_block %U.ref.loc10 [concrete = constants.%U] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %U.ref.loc10: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %U = ref_binding x, %x.var
// CHECK:STDOUT:   %U.cpp_destructor.bound: <bound method> = bound_method %x.var, constants.%U.cpp_destructor
// CHECK:STDOUT:   %U.cpp_destructor.call: init %empty_tuple.type = call %U.cpp_destructor.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_relative_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.8c1: type = ptr_type %U [concrete]
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
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %U) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc8: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %U = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %U = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.8c1 = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_outer_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %O: type = class_type @O [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.a6c: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cff: type = pattern_type %O [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.50c: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.DefaultOrUnformed.impl.Op.type.8d1: type = fn_type @T.binding.as_type.as.DefaultOrUnformed.impl.Op, @T.binding.as_type.as.DefaultOrUnformed.impl(%T.50c) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.DefaultOrUnformed.impl.Op.185: %T.binding.as_type.as.DefaultOrUnformed.impl.Op.type.8d1 = struct_value () [symbolic]
// CHECK:STDOUT:   %O.Op.type: type = fn_type @O.Op [concrete]
// CHECK:STDOUT:   %O.Op: %O.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8f6: <witness> = custom_witness (%O.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.d7b: %Default.type = facet_value %O, (%custom_witness.8f6) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.e05: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.608, @T.binding.as_type.as.DefaultOrUnformed.impl(%Default.facet.d7b) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %O, (%DefaultOrUnformed.impl_witness.e05) [concrete]
// CHECK:STDOUT:   %O.cpp_destructor.type: type = fn_type @O.cpp_destructor [concrete]
// CHECK:STDOUT:   %O.cpp_destructor: %O.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %O.decl: type = class_decl @O [concrete = constants.%O] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.9c6: @T.binding.as_type.as.DefaultOrUnformed.impl.%T.binding.as_type.as.DefaultOrUnformed.impl.Op.type (%T.binding.as_type.as.DefaultOrUnformed.impl.Op.type.8d1) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.DefaultOrUnformed.impl.%T.binding.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.binding.as_type.as.DefaultOrUnformed.impl.Op.185)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.608 = impl_witness_table (%Core.import_ref.9c6), @T.binding.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %U) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %U = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %U = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.a6c = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cff = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.cff = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %O = var %x.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%O, (constants.%DefaultOrUnformed.impl_witness.e05) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc9_22.1: %DefaultOrUnformed.type = converted constants.%O, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc9_22.1 [concrete = constants.%O]
// CHECK:STDOUT:   %.loc9_22.2: type = converted %.loc9_22.1, %as_type [concrete = constants.%O]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9_3: ref %O = splice_block %x.var {}
// CHECK:STDOUT:   %T.binding.as_type.as.DefaultOrUnformed.impl.Op.call: init %O to %.loc9_3 = call %T.binding.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %x.var, %T.binding.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %O.ref.loc9 [concrete = constants.%O] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc9: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %O = ref_binding x, %x.var
// CHECK:STDOUT:   %O.cpp_destructor.bound: <bound method> = bound_method %x.var, constants.%O.cpp_destructor
// CHECK:STDOUT:   %O.cpp_destructor.call: init %empty_tuple.type = call %O.cpp_destructor.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %U.bar.cpp_overload_set.type: type = cpp_overload_set_type @U.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %U.bar.cpp_overload_set.value: %U.bar.cpp_overload_set.type = cpp_overload_set_value @U.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %U.bar.type: type = fn_type @U.bar [concrete]
// CHECK:STDOUT:   %U.bar: %U.bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %U.bar.cpp_overload_set.value: %U.bar.cpp_overload_set.type = cpp_overload_set_value @U.bar.cpp_overload_set [concrete = constants.%U.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %U.bar.decl: %U.bar.type = fn_decl @U.bar [concrete = constants.%U.bar] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %U) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %U.ref.loc8: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %bar.ref: %U.bar.cpp_overload_set.type = name_ref bar, imports.%U.bar.cpp_overload_set.value [concrete = constants.%U.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %U.bar.call: init %empty_tuple.type = call imports.%U.bar.decl()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %U = name_ref u, %u
// CHECK:STDOUT:   %.loc9: ref %U = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of %.loc9
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.bar.cpp_overload_set.type: type = cpp_overload_set_type @U.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %U.bar.cpp_overload_set.value: %U.bar.cpp_overload_set.type = cpp_overload_set_value @U.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %U.bar.type: type = fn_type @U.bar [concrete]
// CHECK:STDOUT:   %U.bar: %U.bar.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.bar.cpp_overload_set.value: %U.bar.cpp_overload_set.type = cpp_overload_set_value @U.bar.cpp_overload_set [concrete = constants.%U.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %U.bar.decl: %U.bar.type = fn_decl @U.bar [concrete = constants.%U.bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %U) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %U = name_ref u, %u
// CHECK:STDOUT:   %.loc8: ref %U = value_as_ref %u.ref
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %U.ref.loc9: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %bar.ref: %U.bar.cpp_overload_set.type = name_ref bar, imports.%U.bar.cpp_overload_set.value [concrete = constants.%U.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %U.bar.call: init %empty_tuple.type = call imports.%U.bar.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %ptr = name_ref u, %u
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%u.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: %ptr = name_ref u, %u
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%u.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_value_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.cpp_destructor.type: type = fn_type @U.cpp_destructor [concrete]
// CHECK:STDOUT:   %U.cpp_destructor: %U.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_11.1: ref %U = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of %.loc8_11.1
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_11.2: init %U to %.loc8_11.1 = mark_in_place_init %foo__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_11.3: ref %U = temporary %.loc8_11.1, %.loc8_11.2
// CHECK:STDOUT:   %U.cpp_destructor.bound: <bound method> = bound_method %.loc8_11.3, constants.%U.cpp_destructor
// CHECK:STDOUT:   %U.cpp_destructor.call: init %empty_tuple.type = call %U.cpp_destructor.bound(%.loc8_11.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
