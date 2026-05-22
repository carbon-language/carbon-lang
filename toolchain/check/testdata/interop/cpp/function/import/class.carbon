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

class C;

auto foo1(C) -> void;
auto foo2(C) -> void;

// --- fail_import_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:4:6: error: argument type 'C' is incomplete [CppInteropParseError]
// CHECK:STDERR:     4 | auto foo1(C) -> void;
// CHECK:STDERR:       |      ^~~~
// CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:2:7: note: forward declaration of 'C' [CppInteropParseNote]
// CHECK:STDERR:     2 | class C;
// CHECK:STDERR:       |       ^
import Cpp library "decl_value_param_type.h";

fn F(p: Cpp.C*) {
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+12]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo1(*p);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:5:6: error: argument type 'C' is incomplete [CppInteropParseError]
  // CHECK:STDERR:     5 | auto foo2(C) -> void;
  // CHECK:STDERR:       |      ^~~~
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: forward declaration of 'C' [CppInteropParseNote]
  // CHECK:STDERR:     2 | class C;
  // CHECK:STDERR:       |       ^
  Cpp.foo1(*p);
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+28]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo2(*p);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-5]]:12: error: forming value of incomplete type `Cpp.C` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   Cpp.foo1(*p);
  // CHECK:STDERR:            ^~
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-23]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-27]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:4:12: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto foo1(C) -> void;
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+12]]:12: error: forming value of incomplete type `Cpp.C` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   Cpp.foo2(*p);
  // CHECK:STDERR:            ^~
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-35]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-39]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:5:12: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto foo2(C) -> void;
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  Cpp.foo2(*p);
}

// ============================================================================
// Defined class without data members as parameter type
// ============================================================================

// --- definition_no_data_members_value_param_type.h

class C {};

auto foo(C) -> void;

// --- import_definition_no_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_no_data_members_value_param_type.h";

fn F(c: Cpp.C) {
  //@dump-sem-ir-begin
  Cpp.foo(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Non copyable class as parameter type
// ============================================================================

// --- non_copyable_param_type.h

class C {
  C(const C&) = delete;
};

auto foo(C) -> void;

// --- fail_import_non_copyable_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_non_copyable_param_type.carbon:[[@LINE+12]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./non_copyable_param_type.h:6:6: error: call to deleted constructor of 'C' [CppInteropParseError]
// CHECK:STDERR:     6 | auto foo(C) -> void;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_import_non_copyable_param_type.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./non_copyable_param_type.h:3:3: note: 'C' has been explicitly marked deleted here [CppInteropParseNote]
// CHECK:STDERR:     3 |   C(const C&) = delete;
// CHECK:STDERR:       |   ^
// CHECK:STDERR: fail_import_non_copyable_param_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./non_copyable_param_type.h:6:11: note: passing argument to parameter here [CppInteropParseNote]
// CHECK:STDERR:     6 | auto foo(C) -> void;
// CHECK:STDERR:       |           ^
import Cpp library "non_copyable_param_type.h";

fn F(c: Cpp.C) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_import_non_copyable_param_type.carbon:[[@LINE+4]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo(c);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Move-only class as parameter type
// ============================================================================

// --- move_only_param_type.h

struct C {
  C(C&&);
};

auto foo(C) -> void;
auto make() -> C;
auto move(C&) -> C&&;

// --- pass_move_only_param_temporary.carbon

library "[[@TEST_NAME]]";

import Cpp library "move_only_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(Cpp.make());
  //@dump-sem-ir-end
}

// --- fail_pass_move_only_param_value.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_pass_move_only_param_value.carbon:[[@LINE+12]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:6:6: error: call to implicitly-deleted copy constructor of 'C' [CppInteropParseError]
// CHECK:STDERR:     6 | auto foo(C) -> void;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_pass_move_only_param_value.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:3:3: note: copy constructor is implicitly deleted because 'C' has a user-declared move constructor [CppInteropParseNote]
// CHECK:STDERR:     3 |   C(C&&);
// CHECK:STDERR:       |   ^
// CHECK:STDERR: fail_pass_move_only_param_value.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:6:11: note: passing argument to parameter here [CppInteropParseNote]
// CHECK:STDERR:     6 | auto foo(C) -> void;
// CHECK:STDERR:       |           ^
import Cpp library "move_only_param_type.h";

fn F(c: Cpp.C) {
  // CHECK:STDERR: fail_pass_move_only_param_value.carbon:[[@LINE+4]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo(c);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(c);
}

// --- fail_pass_move_only_param_ref.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_pass_move_only_param_ref.carbon:[[@LINE+12]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:6:6: error: call to implicitly-deleted copy constructor of 'C' [CppInteropParseError]
// CHECK:STDERR:     6 | auto foo(C) -> void;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_pass_move_only_param_ref.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:3:3: note: copy constructor is implicitly deleted because 'C' has a user-declared move constructor [CppInteropParseNote]
// CHECK:STDERR:     3 |   C(C&&);
// CHECK:STDERR:       |   ^
// CHECK:STDERR: fail_pass_move_only_param_ref.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:6:11: note: passing argument to parameter here [CppInteropParseNote]
// CHECK:STDERR:     6 | auto foo(C) -> void;
// CHECK:STDERR:       |           ^
import Cpp library "move_only_param_type.h";

fn F(ref c: Cpp.C) {
  // CHECK:STDERR: fail_pass_move_only_param_ref.carbon:[[@LINE+4]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo(c);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(c);
}

// --- fail_todo_pass_move_only_param_ref_by_move.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_pass_move_only_param_ref_by_move.carbon:[[@LINE+12]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:6:6: error: call to implicitly-deleted copy constructor of 'C' [CppInteropParseError]
// CHECK:STDERR:     6 | auto foo(C) -> void;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_todo_pass_move_only_param_ref_by_move.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:3:3: note: copy constructor is implicitly deleted because 'C' has a user-declared move constructor [CppInteropParseNote]
// CHECK:STDERR:     3 |   C(C&&);
// CHECK:STDERR:       |   ^
// CHECK:STDERR: fail_todo_pass_move_only_param_ref_by_move.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./move_only_param_type.h:6:11: note: passing argument to parameter here [CppInteropParseNote]
// CHECK:STDERR:     6 | auto foo(C) -> void;
// CHECK:STDERR:       |           ^
import Cpp library "move_only_param_type.h";

fn F(ref c: Cpp.C) {
  // CHECK:STDERR: fail_todo_pass_move_only_param_ref_by_move.carbon:[[@LINE+4]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo(Cpp.move(ref c));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(Cpp.move(ref c));
}

// ============================================================================
// Defined class with a single data member as parameter type
// ============================================================================

// --- definition_single_data_member_value_param_type.h

class D {};

class C {
  D d;
};

auto foo(C) -> void;

// --- import_definition_single_data_member_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_single_data_member_value_param_type.h";

fn F(c: Cpp.C) {
  //@dump-sem-ir-begin
  Cpp.foo(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined class with multiple data members as parameter type
// ============================================================================

// --- definition_multiple_data_members_value_param_type.h

class D {};

class C {
  D d1;
  D d2;
  D d3;
};

auto foo(C) -> void;

// --- import_definition_multiple_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_multiple_data_members_value_param_type.h";

fn F(c: Cpp.C) {
  //@dump-sem-ir-begin
  Cpp.foo(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined class in namespace
// ============================================================================

// --- definition_in_namespace_value_param_type.h

namespace N { class C {}; }

auto foo(N::C) -> void;

// --- import_definition_in_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_namespace_value_param_type.h";

fn F(c: Cpp.N.C) {
  //@dump-sem-ir-begin
  Cpp.foo(c);
  // Check that the parameter type was imported correctly.
  var unused x: Cpp.N.C;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined class in relative namespace
// ============================================================================

// --- definition_in_relative_namespace_value_param_type.h

namespace N1 {
  namespace N2 { class C {}; }
  auto foo(N2::C) -> void;
}

// --- import_definition_in_relative_namespace_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_relative_namespace_value_param_type.h";

fn F(c: Cpp.N1.N2.C) {
  //@dump-sem-ir-begin
  Cpp.N1.foo(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Defined class in class
// ============================================================================

// --- definition_in_outer_definition.h

class O {
 public:
  class C {};
};

auto foo(O::C) -> void;

// --- import_definition_in_outer_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_in_outer_definition.h";

fn F(c: Cpp.O.C) {
  //@dump-sem-ir-begin
  Cpp.foo(c);
  var unused x: Cpp.O;
  //@dump-sem-ir-end
}

// ============================================================================
// Defined class and explicitly used
// ============================================================================

// --- definition_with_static_method.h

class C {
 public:
  static void bar();
};

auto foo(C) -> void;

// --- import_definition_and_static_method_call_before.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F(c: Cpp.C) {
  //@dump-sem-ir-begin
  Cpp.C.bar();
  Cpp.foo(c);
  //@dump-sem-ir-end
}

// --- import_definition_and_static_method_call_after.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F(c: Cpp.C) {
  //@dump-sem-ir-begin
  Cpp.foo(c);
  Cpp.C.bar();
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
  // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE+8]]:3: error: member access into incomplete class `Cpp.Incomplete` [QualifiedExprInIncompleteClassScope]
  // CHECK:STDERR:   Cpp.Incomplete.foo();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./incomplete.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Incomplete;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  Cpp.Incomplete.foo();
}

// ============================================================================
// Pointer to forward-declared class as parameter type
// ============================================================================

// --- decl_pointer_param_type.h

class C;

auto foo(C* _Nonnull) -> void;

// --- import_decl_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_param_type.h";

fn F(c: Cpp.C*) {
  //@dump-sem-ir-begin
  Cpp.foo(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined class as parameter type
// ============================================================================

// --- definition_pointer_param_type.h

class C {};

auto foo(C* _Nonnull) -> void;

// --- import_definition_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_pointer_param_type.h";

fn F(c: Cpp.C*) {
  //@dump-sem-ir-begin
  Cpp.foo(c);
  //@dump-sem-ir-end
}

// ============================================================================
// Forward-declared class as return type
// ============================================================================

// --- decl_value_return_type.h

class C;

auto foo() -> C;

// --- fail_import_decl_value_return_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+12]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:4:6: error: calling 'foo' with incomplete return type 'C' [CppInteropParseError]
// CHECK:STDERR:     4 | auto foo() -> C;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:4:6: note: 'foo' declared here [CppInteropParseNote]
// CHECK:STDERR:     4 | auto foo() -> C;
// CHECK:STDERR:       |      ^
// CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_return_type.h:2:7: note: forward declaration of 'C' [CppInteropParseNote]
// CHECK:STDERR:     2 | class C;
// CHECK:STDERR:       |       ^
import Cpp library "decl_value_return_type.h";

fn F() {
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+16]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+12]]:3: error: function returns incomplete type `Cpp.C` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_return_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE-14]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_return_type.h:4:1: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: auto foo() -> C;
  // CHECK:STDERR: ^
  // CHECK:STDERR:
  Cpp.foo();
}

// ============================================================================
// Defined class as return type
// ============================================================================

// --- definition_value_return_type.h

class C {};

auto foo() -> C;

// --- import_definition_value_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_value_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to forward-declared class as return type
// ============================================================================

// --- decl_pointer_return_type.h

class C;

auto foo() -> C* _Nonnull;

// --- import_decl_pointer_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to defined class as return type
// ============================================================================

// --- definition_pointer_return_type.h

class C {};

auto foo() -> C* _Nonnull;

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
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dbb: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %C = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.dbb = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_non_copyable_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%c.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- pass_move_only_param_temporary.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %make.cpp_overload_set.type: type = cpp_overload_set_type @make.cpp_overload_set [concrete]
// CHECK:STDOUT:   %make.cpp_overload_set.value: %make.cpp_overload_set.type = cpp_overload_set_value @make.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.dbb: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %make__carbon_thunk.type: type = fn_type @make__carbon_thunk [concrete]
// CHECK:STDOUT:   %make__carbon_thunk: %make__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     .make = %make.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %make.cpp_overload_set.value: %make.cpp_overload_set.type = cpp_overload_set_value @make.cpp_overload_set [concrete = constants.%make.cpp_overload_set.value]
// CHECK:STDOUT:   %make__carbon_thunk.decl: %make__carbon_thunk.type = fn_decl @make__carbon_thunk [concrete = constants.%make__carbon_thunk] {
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
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %make.ref: %make.cpp_overload_set.type = name_ref make, imports.%make.cpp_overload_set.value [concrete = constants.%make.cpp_overload_set.value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc8_20: %ptr.dbb = addr_of %_.var
// CHECK:STDOUT:   %make__carbon_thunk.call: init %empty_tuple.type = call imports.%make__carbon_thunk.decl(%addr.loc8_20)
// CHECK:STDOUT:   %.loc8: init %C to %_.var = mark_in_place_init %make__carbon_thunk.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc8_21: %ptr.dbb = addr_of %_.var
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_21)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_single_data_member_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dbb: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %C = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.dbb = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_multiple_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dbb: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %C = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.dbb = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.8ab: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d15: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.4ca: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.47c: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.4ca) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.25a: %T.as_type.as.DefaultOrUnformed.impl.Op.type.47c = struct_value () [symbolic]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3d0: <witness> = custom_witness (%C.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.8ae: %Default.type = facet_value %C, (%custom_witness.3d0) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.a4e: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.b72, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.8ae) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %C, (%DefaultOrUnformed.impl_witness.a4e) [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dfa: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.47c) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.25a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.b72 = impl_witness_table (%Core.import_ref.dfa), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %C = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.d15 = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.8ab = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.8ab = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %C = var %x.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%C, (constants.%DefaultOrUnformed.impl_witness.a4e) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc10_24.1: %DefaultOrUnformed.type = converted constants.%C, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_24.1 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_24.2: type = converted %.loc10_24.1, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_3: ref %C = splice_block %x.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %C to %.loc10_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %x.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc10_22: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %C = ref_binding x, %x.var
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %x.var, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_relative_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.4c9: type = ptr_type %C [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc8: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %C = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.4c9 = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_outer_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %O: type = class_type @O [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.714: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.194: type = pattern_type %O [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.4ca: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.47c: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.4ca) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.25a: %T.as_type.as.DefaultOrUnformed.impl.Op.type.47c = struct_value () [symbolic]
// CHECK:STDOUT:   %O.Op.type: type = fn_type @O.Op [concrete]
// CHECK:STDOUT:   %O.Op: %O.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.213: <witness> = custom_witness (%O.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.15f: %Default.type = facet_value %O, (%custom_witness.213) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.424: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.b72, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.15f) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %O, (%DefaultOrUnformed.impl_witness.424) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.dfa: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.47c) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.25a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.b72 = impl_witness_table (%Core.import_ref.dfa), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %C = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.714 = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.194 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.194 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %O = var %x.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%O, (constants.%DefaultOrUnformed.impl_witness.424) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc9_22.1: %DefaultOrUnformed.type = converted constants.%O, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc9_22.1 [concrete = constants.%O]
// CHECK:STDOUT:   %.loc9_22.2: type = converted %.loc9_22.1, %as_type [concrete = constants.%O]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9_3: ref %O = splice_block %x.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %O to %.loc9_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %x.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
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
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.bar.cpp_overload_set.type: type = cpp_overload_set_type @C.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.bar.cpp_overload_set.value: %C.bar.cpp_overload_set.type = cpp_overload_set_value @C.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.bar.type: type = fn_type @C.bar [concrete]
// CHECK:STDOUT:   %C.bar: %C.bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dbb: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.bar.cpp_overload_set.value: %C.bar.cpp_overload_set.type = cpp_overload_set_value @C.bar.cpp_overload_set [concrete = constants.%C.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %C.bar.decl: %C.bar.type = fn_decl @C.bar [concrete = constants.%C.bar] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %bar.ref: %C.bar.cpp_overload_set.type = name_ref bar, imports.%C.bar.cpp_overload_set.value [concrete = constants.%C.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %C.bar.call: init %empty_tuple.type = call imports.%C.bar.decl()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc9: ref %C = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.dbb = addr_of %.loc9
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dbb: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.bar.cpp_overload_set.type: type = cpp_overload_set_type @C.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.bar.cpp_overload_set.value: %C.bar.cpp_overload_set.type = cpp_overload_set_value @C.bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.bar.type: type = fn_type @C.bar [concrete]
// CHECK:STDOUT:   %C.bar: %C.bar.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.bar.cpp_overload_set.value: %C.bar.cpp_overload_set.type = cpp_overload_set_value @C.bar.cpp_overload_set [concrete = constants.%C.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %C.bar.decl: %C.bar.type = fn_decl @C.bar [concrete = constants.%C.bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc8: ref %C = value_as_ref %c.ref
// CHECK:STDOUT:   %addr: %ptr.dbb = addr_of %.loc8
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %bar.ref: %C.bar.cpp_overload_set.type = name_ref bar, imports.%C.bar.cpp_overload_set.value [concrete = constants.%C.bar.cpp_overload_set.value]
// CHECK:STDOUT:   %C.bar.call: init %empty_tuple.type = call imports.%C.bar.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %ptr = name_ref c, %c
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%c.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %c.ref: %ptr = name_ref c, %c
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%c.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_value_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.dbb: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc8_11.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.dbb = addr_of %.loc8_11.1
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_11.2: init %C to %.loc8_11.1 = mark_in_place_init %foo__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_11.3: ref %C = temporary %.loc8_11.1, %.loc8_11.2
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc8_11.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc8_11.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [concrete]
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
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [concrete]
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
