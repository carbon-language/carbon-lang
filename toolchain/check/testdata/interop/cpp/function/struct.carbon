// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/struct.carbon

// ============================================================================
// Forward-declared struct as parameter type
// ============================================================================

// --- decl_value_param_type.h

struct S;

auto foo(S) -> void;

// --- fail_import_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:4:6: error: argument type 'S' is incomplete [CppInteropParseError]
// CHECK:STDERR:     4 | auto foo(S) -> void;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:2:8: note: forward declaration of 'S' [CppInteropParseNote]
// CHECK:STDERR:     2 | struct S;
// CHECK:STDERR:       |        ^
import Cpp library "decl_value_param_type.h";

fn F() {
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+12]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.S);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+8]]:11: error: invalid use of incomplete type `Cpp.S` [IncompleteTypeInConversion]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.S);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:8: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: struct S;
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  Cpp.foo({} as Cpp.S);
}

// --- fail_import_decl_value_param_type_previously_imported.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:4:6: error: argument type 'S' is incomplete [CppInteropParseError]
// CHECK:STDERR:     4 | auto foo(S) -> void;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:2:8: note: forward declaration of 'S' [CppInteropParseNote]
// CHECK:STDERR:     2 | struct S;
// CHECK:STDERR:       |        ^
import Cpp library "decl_value_param_type.h";

fn F() {
  let s: Cpp.S;
  // CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE+16]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo(s);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE-5]]:10: error: binding pattern has incomplete type `S` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:   let s: Cpp.S;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:8: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: struct S;
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE-13]]:15: error: expected `=`; `let` declaration must have an initializer [ExpectedInitializerAfterLet]
  // CHECK:STDERR:   let s: Cpp.S;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  Cpp.foo(s);
}

// ============================================================================
// Forward-declared struct as parameter type imported twice
// ============================================================================

// --- double_decl_value_param_type.h

struct S;

auto foo1(S) -> void;
auto foo2(S) -> void;

// --- fail_import_double_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./double_decl_value_param_type.h:4:6: error: argument type 'S' is incomplete [CppInteropParseError]
// CHECK:STDERR:     4 | auto foo1(S) -> void;
// CHECK:STDERR:       |      ^~~~
// CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./double_decl_value_param_type.h:2:8: note: forward declaration of 'S' [CppInteropParseNote]
// CHECK:STDERR:     2 | struct S;
// CHECK:STDERR:       |        ^
import Cpp library "double_decl_value_param_type.h";

fn F() {
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+12]]:3: note: in `Cpp` name lookup for `foo1` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo1({} as Cpp.S);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./double_decl_value_param_type.h:5:6: error: argument type 'S' is incomplete [CppInteropParseError]
  // CHECK:STDERR:     5 | auto foo2(S) -> void;
  // CHECK:STDERR:       |      ^~~~
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./double_decl_value_param_type.h:2:8: note: forward declaration of 'S' [CppInteropParseNote]
  // CHECK:STDERR:     2 | struct S;
  // CHECK:STDERR:       |        ^
  Cpp.foo1({} as Cpp.S);
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+20]]:3: note: in `Cpp` name lookup for `foo2` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo2({} as Cpp.S);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-5]]:12: error: invalid use of incomplete type `Cpp.S` [IncompleteTypeInConversion]
  // CHECK:STDERR:   Cpp.foo1({} as Cpp.S);
  // CHECK:STDERR:            ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-23]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./double_decl_value_param_type.h:2:8: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: struct S;
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+8]]:12: error: invalid use of incomplete type `Cpp.S` [IncompleteTypeInConversion]
  // CHECK:STDERR:   Cpp.foo2({} as Cpp.S);
  // CHECK:STDERR:            ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-31]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./double_decl_value_param_type.h:2:8: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: struct S;
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  Cpp.foo2({} as Cpp.S);
}

// ============================================================================
// Defined struct without data members as parameter type
// ============================================================================

// --- definition_no_data_members_value_param_type.h

struct S {};

auto foo(S) -> void;

// --- import_definition_no_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_no_data_members_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.S);
  //@dump-sem-ir-end
}

// ============================================================================
// Non copyable struct as parameter type
// ============================================================================

// --- non_copyable_param_type.h

struct S {
  S(const S&) = delete;
};

auto foo(S) -> void;

// --- fail_import_non_copyable_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_non_copyable_param_type.carbon:[[@LINE+12]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./non_copyable_param_type.h:6:6: error: call to deleted constructor of 'S' [CppInteropParseError]
// CHECK:STDERR:     6 | auto foo(S) -> void;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_import_non_copyable_param_type.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./non_copyable_param_type.h:3:3: note: 'S' has been explicitly marked deleted here [CppInteropParseNote]
// CHECK:STDERR:     3 |   S(const S&) = delete;
// CHECK:STDERR:       |   ^
// CHECK:STDERR: fail_import_non_copyable_param_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./non_copyable_param_type.h:6:11: note: passing argument to parameter here [CppInteropParseNote]
// CHECK:STDERR:     6 | auto foo(S) -> void;
// CHECK:STDERR:       |           ^
import Cpp library "non_copyable_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_import_non_copyable_param_type.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.S);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.foo({} as Cpp.S);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct with a single data member as parameter type
// ============================================================================

// --- definition_single_data_member_value_param_type.h

struct D {};

struct S {
  D d;
};

auto foo(S) -> void;

// --- fail_todo_import_definition_single_data_member_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_single_data_member_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_definition_single_data_member_value_param_type.carbon:[[@LINE+7]]:11: error: cannot convert expression of type `{}` to `Cpp.S` with `as` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.S);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_definition_single_data_member_value_param_type.carbon:[[@LINE+4]]:11: note: type `{}` does not implement interface `Core.As(Cpp.S)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.S);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo({} as Cpp.S);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct with multiple data members as parameter type
// ============================================================================

// --- definition_multiple_data_members_value_param_type.h

struct D {};

struct S {
  D d1;
  D d2;
  D d3;
};

auto foo(S) -> void;

// --- fail_todo_import_definition_multiple_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_multiple_data_members_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_definition_multiple_data_members_value_param_type.carbon:[[@LINE+7]]:11: error: cannot convert expression of type `{}` to `Cpp.S` with `as` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.S);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_definition_multiple_data_members_value_param_type.carbon:[[@LINE+4]]:11: note: type `{}` does not implement interface `Core.As(Cpp.S)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.S);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo({} as Cpp.S);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct in namespace
// ============================================================================

// --- definition_in_namespace_value_param_type.h

namespace N { struct S {}; }

auto foo(N::S) -> void;

// --- import_definition_in_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_namespace_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.N.S);
  // Check that the parameter type was imported correctly.
  var x: Cpp.N.S;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct in relative namespace
// ============================================================================

// --- definition_in_relative_namespace_value_param_type.h

namespace N1 {
  namespace N2 { struct S {}; }
  auto foo(N2::S) -> void;
}

// --- import_definition_in_relative_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_relative_namespace_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.N1.foo({} as Cpp.N1.N2.S);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct in struct
// ============================================================================

// --- definition_in_outer_definition.h

struct O {
  struct S {};
};

auto foo(O::S) -> void;

// --- import_definition_in_outer_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_outer_definition.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.O.S);
  var x: Cpp.O;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined struct and explicitly used
// ============================================================================

// --- definition_with_static_method.h

struct S {
  static void bar();
};

auto foo(S) -> void;

// --- import_definition_and_static_method_call_before.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.S.bar();
  Cpp.foo({} as Cpp.S);
  //@dump-sem-ir-end
}

// --- import_definition_and_static_method_call_after.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.S);
  Cpp.S.bar();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to forward-declared struct as parameter type
// ============================================================================

// --- decl_pointer_param_type.h

struct S;

auto foo(S* _Nonnull) -> void;

// --- import_decl_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_param_type.h";

fn F(s: Cpp.S*) {
  //@dump-sem-ir-begin
  Cpp.foo(s);
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined struct as parameter type
// ============================================================================

// --- definition_pointer_param_type.h

struct S {};

auto foo(S* _Nonnull) -> void;

// --- import_definition_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_pointer_param_type.h";

fn F(s: Cpp.S*) {
  //@dump-sem-ir-begin
  Cpp.foo(s);
  //@dump-sem-ir-end
}

// ============================================================================
// Forward-declared struct as return type
// ============================================================================

// --- decl_value_return_type.h

struct S;

auto foo() -> S;

// --- fail_import_decl_value_return_type.carbon

library "[[@TEST_NAME]]";

// TODO: We should only produce one error here.
// CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+12]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:4:6: error: calling 'foo' with incomplete return type 'S' [CppInteropParseError]
// CHECK:STDERR:     4 | auto foo() -> S;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:4:6: note: 'foo' declared here [CppInteropParseNote]
// CHECK:STDERR:     4 | auto foo() -> S;
// CHECK:STDERR:       |      ^
// CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:2:8: note: forward declaration of 'S' [CppInteropParseNote]
// CHECK:STDERR:     2 | struct S;
// CHECK:STDERR:       |        ^
import Cpp library "decl_value_return_type.h";

fn F() {
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+13]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+9]]:3: error: function returns incomplete type `Cpp.S` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_return_type.h:2:8: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: struct S;
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR:
  Cpp.foo();
}

// ============================================================================
// Defined struct as return type
// ============================================================================

// --- definition_value_return_type.h

struct S {};

auto foo() -> S;

// --- import_definition_value_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_value_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to forward-declared struct as return type
// ============================================================================

// --- decl_pointer_return_type.h

struct S;

auto foo() -> S* _Nonnull;

// --- import_decl_pointer_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined struct as return type
// ============================================================================

// --- definition_pointer_return_type.h

struct S {};

auto foo() -> S* _Nonnull;

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
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.642: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.ab5: %T.as.Destroy.impl.Op.type.642 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc8_12.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %S = class_init (), %.loc8_12.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %S = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %S = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %S = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %S = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_22: %ptr.5c7 = addr_of %.loc8_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_22)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_12.4, constants.%T.as.Destroy.impl.Op.ab5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.4, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_12: %ptr.5c7 = addr_of %.loc8_12.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_non_copyable_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.642: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.ab5: %T.as.Destroy.impl.Op.type.642 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc24_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc24_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc24_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc24_12.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc24_12.3: init %S = class_init (), %.loc24_12.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc24_12.4: ref %S = temporary %.loc24_12.2, %.loc24_12.3
// CHECK:STDOUT:   %.loc24_14.1: ref %S = converted %.loc24_12.1, %.loc24_12.4
// CHECK:STDOUT:   %.loc24_14.2: %S = bind_value %.loc24_14.1
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc24_14.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc24_12.4, constants.%T.as.Destroy.impl.Op.ab5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc24_12.4, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of %.loc24_12.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_definition_single_data_member_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc15_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc15_12: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc15_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc15_14: %S = converted %.loc15_12, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_definition_multiple_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc15_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc15_12: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc15_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc15_14: %S = converted %.loc15_12, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %pattern_type.cd8: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.edf: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.2b5: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.9b3: %T.as.Destroy.impl.Op.type.2b5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc8: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc8_12.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %S = class_init (), %.loc8_12.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %S = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %S = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %S = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %S = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_24: %ptr.edf = addr_of %.loc8_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_24)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cd8 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.cd8 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %S = var %x.var_patt
// CHECK:STDOUT:   %.loc10: type = splice_block %S.ref.loc10 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %S.ref.loc10: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %S = bind_name x, %x.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %x.var, constants.%T.as.Destroy.impl.Op.9b3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %x.var, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10: %ptr.edf = addr_of %x.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_12.4, constants.%T.as.Destroy.impl.Op.9b3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_12.4, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_12: %ptr.edf = addr_of %.loc8_12.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_relative_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.887: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.17f: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.463: %T.as.Destroy.impl.Op.type.17f = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N1 = %N1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc8_6: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc8_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc8_23: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %N2.ref: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:   %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc8_15.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc8_15.3: init %S = class_init (), %.loc8_15.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_15.4: ref %S = temporary %.loc8_15.2, %.loc8_15.3
// CHECK:STDOUT:   %.loc8_17.1: ref %S = converted %.loc8_15.1, %.loc8_15.4
// CHECK:STDOUT:   %.loc8_17.2: %S = bind_value %.loc8_17.1
// CHECK:STDOUT:   %.loc8_17.3: ref %S = value_as_ref %.loc8_17.2
// CHECK:STDOUT:   %addr.loc8_31: %ptr.887 = addr_of %.loc8_17.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_31)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_15.4, constants.%T.as.Destroy.impl.Op.463
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_15.4, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_15: %ptr.887 = addr_of %.loc8_15.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_outer_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %O: type = class_type @O [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.149: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cff: type = pattern_type %O [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.1b8: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%O) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.2df: %T.as.Destroy.impl.Op.type.1b8 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.820: type = ptr_type %O [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.23f: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.952: %T.as.Destroy.impl.Op.type.23f = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %O.decl: type = class_decl @O [concrete = constants.%O] {} {}
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc8: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc8_12.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %S = class_init (), %.loc8_12.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %S = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %S = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %S = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %S = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_24: %ptr.149 = addr_of %.loc8_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_24)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cff = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.cff = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %O = var %x.var_patt
// CHECK:STDOUT:   %.loc9: type = splice_block %O.ref.loc9 [concrete = constants.%O] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc9: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %O = bind_name x, %x.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %x.var, constants.%T.as.Destroy.impl.Op.2df
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %x.var, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc9: %ptr.820 = addr_of %x.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_12.4, constants.%T.as.Destroy.impl.Op.952
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_12.4, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_12: %ptr.149 = addr_of %.loc8_12.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %S.bar.type: type = fn_type @S.bar [concrete]
// CHECK:STDOUT:   %S.bar: %S.bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.642: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.ab5: %T.as.Destroy.impl.Op.type.642 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %S.bar.decl: %S.bar.type = fn_decl @S.bar [concrete = constants.%S.bar] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %bar.ref: %S.bar.type = name_ref bar, imports.%S.bar.decl [concrete = constants.%S.bar]
// CHECK:STDOUT:   %S.bar.call: init %empty_tuple.type = call %bar.ref()
// CHECK:STDOUT:   %Cpp.ref.loc9_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc9_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc9_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc9_12.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc9_12.3: init %S = class_init (), %.loc9_12.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc9_12.4: ref %S = temporary %.loc9_12.2, %.loc9_12.3
// CHECK:STDOUT:   %.loc9_14.1: ref %S = converted %.loc9_12.1, %.loc9_12.4
// CHECK:STDOUT:   %.loc9_14.2: %S = bind_value %.loc9_14.1
// CHECK:STDOUT:   %.loc9_14.3: ref %S = value_as_ref %.loc9_14.2
// CHECK:STDOUT:   %addr.loc9_22: %ptr.5c7 = addr_of %.loc9_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc9_22)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc9_12.4, constants.%T.as.Destroy.impl.Op.ab5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc9_12.4, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc9_12: %ptr.5c7 = addr_of %.loc9_12.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc9_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %S.bar.type: type = fn_type @S.bar [concrete]
// CHECK:STDOUT:   %S.bar: %S.bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.642: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.ab5: %T.as.Destroy.impl.Op.type.642 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.bar.decl: %S.bar.type = fn_decl @S.bar [concrete = constants.%S.bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc8_12.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %S = class_init (), %.loc8_12.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %S = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %S = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %S = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %S = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_22: %ptr.5c7 = addr_of %.loc8_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_22)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %bar.ref: %S.bar.type = name_ref bar, imports.%S.bar.decl [concrete = constants.%S.bar]
// CHECK:STDOUT:   %S.bar.call: init %empty_tuple.type = call %bar.ref()
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_12.4, constants.%T.as.Destroy.impl.Op.ab5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.4, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_12: %ptr.5c7 = addr_of %.loc8_12.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %s.ref: %ptr = name_ref s, %s
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%s.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %s.ref: %ptr = name_ref s, %s
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%s.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_value_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.642: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.ab5: %T.as.Destroy.impl.Op.type.642 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc8_11.1: ref %S = temporary_storage
// CHECK:STDOUT:   %addr.loc8_11.1: %ptr.5c7 = addr_of %.loc8_11.1
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_11.1)
// CHECK:STDOUT:   %.loc8_11.2: init %S = in_place_init %foo__carbon_thunk.call, %.loc8_11.1
// CHECK:STDOUT:   %.loc8_11.3: ref %S = temporary %.loc8_11.1, %.loc8_11.2
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.3, constants.%T.as.Destroy.impl.Op.ab5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_11.3, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11.2: %ptr.5c7 = addr_of %.loc8_11.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_11.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call: init %ptr = call %foo.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call: init %ptr = call %foo.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
