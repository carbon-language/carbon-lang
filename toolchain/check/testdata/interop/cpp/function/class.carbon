// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/class.carbon

// ============================================================================
// Forward-declared class as parameter type
// ============================================================================

// --- decl_value_param_type.h

class C;

auto foo(C) -> void;

// --- fail_import_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_value_param_type.h";

fn F() {
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+8]]:11: error: invalid use of incomplete type `Cpp.C` [IncompleteTypeInConversion]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.C);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  Cpp.foo({} as Cpp.C);
}

// --- fail_import_decl_value_param_type_previously_imported.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_value_param_type.h";

fn F() {
  // CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE+12]]:10: error: binding pattern has incomplete type `C` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:   let c: Cpp.C;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE+4]]:15: error: expected `=`; `let` declaration must have an initializer [ExpectedInitializerAfterLet]
  // CHECK:STDERR:   let c: Cpp.C;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  let c: Cpp.C;
  Cpp.foo(c);
}

// ============================================================================
// Forward-declared class as parameter type imported twice
// ============================================================================

// --- double_decl_value_param_type.h

class C;

auto foo1(C) -> void;
auto foo2(C) -> void;

// --- fail_import_double_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "double_decl_value_param_type.h";

fn F() {
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+8]]:12: error: invalid use of incomplete type `Cpp.C` [IncompleteTypeInConversion]
  // CHECK:STDERR:   Cpp.foo1({} as Cpp.C);
  // CHECK:STDERR:            ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./double_decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  Cpp.foo1({} as Cpp.C);
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+8]]:12: error: invalid use of incomplete type `Cpp.C` [IncompleteTypeInConversion]
  // CHECK:STDERR:   Cpp.foo2({} as Cpp.C);
  // CHECK:STDERR:            ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-15]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./double_decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  Cpp.foo2({} as Cpp.C);
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

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.C);
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

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_import_non_copyable_param_type.carbon:[[@LINE+4]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo({} as Cpp.C);
  //@dump-sem-ir-end
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

// --- fail_todo_import_definition_single_data_member_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_single_data_member_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_definition_single_data_member_value_param_type.carbon:[[@LINE+7]]:11: error: cannot convert expression of type `{}` to `Cpp.C` with `as` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.C);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_definition_single_data_member_value_param_type.carbon:[[@LINE+4]]:11: note: type `{}` does not implement interface `Core.As(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.C);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo({} as Cpp.C);
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

// --- fail_todo_import_definition_multiple_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_multiple_data_members_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_definition_multiple_data_members_value_param_type.carbon:[[@LINE+7]]:11: error: cannot convert expression of type `{}` to `Cpp.C` with `as` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.C);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_definition_multiple_data_members_value_param_type.carbon:[[@LINE+4]]:11: note: type `{}` does not implement interface `Core.As(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.C);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo({} as Cpp.C);
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

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.N.C);
  // Check that the parameter type was imported correctly.
  var x: Cpp.N.C;
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

fn F() {
  //@dump-sem-ir-begin
  Cpp.N1.foo({} as Cpp.N1.N2.C);
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

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.O.C);
  var x: Cpp.O;
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

fn F() {
  //@dump-sem-ir-begin
  Cpp.C.bar();
  Cpp.foo({} as Cpp.C);
  //@dump-sem-ir-end
}

// --- import_definition_and_static_method_call_after.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.C);
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
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+13]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+9]]:3: error: function returns incomplete type `Cpp.C` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_return_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon: note: return type declared here [IncompleteReturnTypeHere]
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.fc1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.6b9: %AggregateT.as_type.as.Destroy.impl.Op.type.fc1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
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
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc8_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %C = class_init (), %.loc8_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %C = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %C = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %C = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %C = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_22: %ptr.d9e = addr_of %.loc8_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_22)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_12.5: %type_where = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_12.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.6b9
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_12: %ptr.d9e = addr_of %.loc8_12.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_non_copyable_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.fc1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.6b9: %AggregateT.as_type.as.Destroy.impl.Op.type.fc1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
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
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc24_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc24_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc24_12.3: init %C = class_init (), %.loc24_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc24_12.4: ref %C = temporary %.loc24_12.2, %.loc24_12.3
// CHECK:STDOUT:   %.loc24_14.1: ref %C = converted %.loc24_12.1, %.loc24_12.4
// CHECK:STDOUT:   %.loc24_14.2: %C = bind_value %.loc24_14.1
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc24_14.2)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc24_12.5: %type_where = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc24_12.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.6b9
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc24_12.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc24_12.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_definition_single_data_member_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc15_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_12: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc15_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc15_14: %C = converted %.loc15_12, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_definition_multiple_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc15_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_12: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc15_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc15_14: %C = converted %.loc15_12, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.69f: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.523: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.1f8: %AggregateT.as_type.as.Destroy.impl.Op.type.523 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
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
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc8: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc8_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %C = class_init (), %.loc8_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %C = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %C = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %C = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %C = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_24: %ptr.838 = addr_of %.loc8_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_24)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.69f = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.69f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %C = var %x.var_patt
// CHECK:STDOUT:   %.loc10_15: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %C = bind_name x, %x.var
// CHECK:STDOUT:   %facet_value.loc10: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc10_3: %type_where = converted constants.%C, %facet_value.loc10 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %x.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.1f8
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %x.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10: %ptr.838 = addr_of %x.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10)
// CHECK:STDOUT:   %facet_value.loc8: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_12.5: %type_where = converted constants.%C, %facet_value.loc8 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_12.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.1f8
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_12.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_12: %ptr.838 = addr_of %.loc8_12.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_relative_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.92e: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.92e = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c0c: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.532: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.52b: %AggregateT.as_type.as.Destroy.impl.Op.type.532 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N1 = %N1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo = %.218
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.218: %.92e = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %N2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
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
// CHECK:STDOUT:   %foo.ref: %.92e = name_ref foo, imports.%.218 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc8_23: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %N2.ref: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc8_15.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_15.3: init %C = class_init (), %.loc8_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_15.4: ref %C = temporary %.loc8_15.2, %.loc8_15.3
// CHECK:STDOUT:   %.loc8_17.1: ref %C = converted %.loc8_15.1, %.loc8_15.4
// CHECK:STDOUT:   %.loc8_17.2: %C = bind_value %.loc8_17.1
// CHECK:STDOUT:   %.loc8_17.3: ref %C = value_as_ref %.loc8_17.2
// CHECK:STDOUT:   %addr.loc8_31: %ptr.c0c = addr_of %.loc8_17.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_31)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_15.5: %type_where = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_15.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.52b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_15.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_15: %ptr.c0c = addr_of %.loc8_15.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_outer_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %O: type = class_type @O [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.de2: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cff: type = pattern_type %O [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.568: %type_where = facet_value %O, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.a17: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.568) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.c17: %AggregateT.as_type.as.Destroy.impl.Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.820: type = ptr_type %O [concrete]
// CHECK:STDOUT:   %facet_value.e34: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.fac: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.e34) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.0e1: %AggregateT.as_type.as.Destroy.impl.Op.type.fac = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %O.decl: type = class_decl @O [concrete = constants.%O] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
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
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc8: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc8_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %C = class_init (), %.loc8_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %C = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %C = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %C = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %C = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_24: %ptr.de2 = addr_of %.loc8_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_24)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cff = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.cff = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %O = var %x.var_patt
// CHECK:STDOUT:   %.loc9_13: type = splice_block %O.ref.loc9 [concrete = constants.%O] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc9: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %O = bind_name x, %x.var
// CHECK:STDOUT:   %facet_value.loc9: %type_where = facet_value constants.%O, () [concrete = constants.%facet_value.568]
// CHECK:STDOUT:   %.loc9_3: %type_where = converted constants.%O, %facet_value.loc9 [concrete = constants.%facet_value.568]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %x.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.c17
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %x.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc9: %ptr.820 = addr_of %x.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9)
// CHECK:STDOUT:   %facet_value.loc8: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value.e34]
// CHECK:STDOUT:   %.loc8_12.5: %type_where = converted constants.%C, %facet_value.loc8 [concrete = constants.%facet_value.e34]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_12.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.0e1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_12.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_12: %ptr.de2 = addr_of %.loc8_12.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.442: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct.87a: %.442 = struct_value () [concrete]
// CHECK:STDOUT:   %C.bar.type: type = fn_type @C.bar [concrete]
// CHECK:STDOUT:   %C.bar: %C.bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %empty_struct.109: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.fc1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.6b9: %AggregateT.as_type.as.Destroy.impl.Op.type.fc1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %.4a4: %.442 = cpp_overload_set_value @foo [concrete = constants.%empty_struct.87a]
// CHECK:STDOUT:   %C.bar.decl: %C.bar.type = fn_decl @C.bar [concrete = constants.%C.bar] {} {}
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo__carbon_thunk [concrete = constants.%empty_struct.109]
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
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %bar.ref: %.442 = name_ref bar, imports.%.4a4 [concrete = constants.%empty_struct.87a]
// CHECK:STDOUT:   %C.bar.call: init %empty_tuple.type = call imports.%C.bar.decl()
// CHECK:STDOUT:   %Cpp.ref.loc9_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct.109]
// CHECK:STDOUT:   %.loc9_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc9_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc9_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc9_12.3: init %C = class_init (), %.loc9_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_12.4: ref %C = temporary %.loc9_12.2, %.loc9_12.3
// CHECK:STDOUT:   %.loc9_14.1: ref %C = converted %.loc9_12.1, %.loc9_12.4
// CHECK:STDOUT:   %.loc9_14.2: %C = bind_value %.loc9_14.1
// CHECK:STDOUT:   %.loc9_14.3: ref %C = value_as_ref %.loc9_14.2
// CHECK:STDOUT:   %addr.loc9_22: %ptr.d9e = addr_of %.loc9_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc9_22)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc9_12.5: %type_where = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc9_12.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.6b9
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc9_12.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc9_12: %ptr.d9e = addr_of %.loc9_12.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc9_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %empty_struct.109: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.442: type = cpp_overload_set_type @C.bar [concrete]
// CHECK:STDOUT:   %empty_struct.87a: %.442 = struct_value () [concrete]
// CHECK:STDOUT:   %C.bar.type: type = fn_type @C.bar [concrete]
// CHECK:STDOUT:   %C.bar: %C.bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.fc1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.6b9: %AggregateT.as_type.as.Destroy.impl.Op.type.fc1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo__carbon_thunk [concrete = constants.%empty_struct.109]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.4a4: %.442 = cpp_overload_set_value @C.bar [concrete = constants.%empty_struct.87a]
// CHECK:STDOUT:   %C.bar.decl: %C.bar.type = fn_decl @C.bar [concrete = constants.%C.bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct.109]
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc8_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %C = class_init (), %.loc8_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %C = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %C = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %C = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %C = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_22: %ptr.d9e = addr_of %.loc8_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_22)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %bar.ref: %.442 = name_ref bar, imports.%.4a4 [concrete = constants.%empty_struct.87a]
// CHECK:STDOUT:   %C.bar.call: init %empty_tuple.type = call imports.%C.bar.decl()
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_12.5: %type_where = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_12.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.6b9
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_12: %ptr.d9e = addr_of %.loc8_12.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %c.ref: %ptr = name_ref c, %c
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%c.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_value_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.fc1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.6b9: %AggregateT.as_type.as.Destroy.impl.Op.type.fc1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_11.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8_11.1: %ptr.d9e = addr_of %.loc8_11.1
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_11.1)
// CHECK:STDOUT:   %.loc8_11.2: init %C = in_place_init %foo__carbon_thunk.call, %.loc8_11.1
// CHECK:STDOUT:   %.loc8_11.3: ref %C = temporary %.loc8_11.1, %.loc8_11.2
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_11.4: %type_where = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.3, constants.%AggregateT.as_type.as.Destroy.impl.Op.6b9
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_11.3, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11.2: %ptr.d9e = addr_of %.loc8_11.3
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_11.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
