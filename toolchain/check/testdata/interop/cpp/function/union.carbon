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

auto foo(U) -> void;

// --- fail_import_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:4:6: error: argument type 'U' is incomplete [CppInteropParseError]
// CHECK:STDERR:     4 | auto foo(U) -> void;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:2:7: note: forward declaration of 'U' [CppInteropParseNote]
// CHECK:STDERR:     2 | union U;
// CHECK:STDERR:       |       ^
import Cpp library "decl_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+12]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.U);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE+8]]:11: error: invalid use of incomplete type `Cpp.U` [IncompleteTypeInConversion]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.U);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_decl_value_param_type.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: union U;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  Cpp.foo({} as Cpp.U);
  //@dump-sem-ir-end
}

// --- fail_import_decl_value_param_type_previously_imported.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:4:6: error: argument type 'U' is incomplete [CppInteropParseError]
// CHECK:STDERR:     4 | auto foo(U) -> void;
// CHECK:STDERR:       |      ^~~
// CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./decl_value_param_type.h:2:7: note: forward declaration of 'U' [CppInteropParseNote]
// CHECK:STDERR:     2 | union U;
// CHECK:STDERR:       |       ^
import Cpp library "decl_value_param_type.h";

fn F() {
  let u: Cpp.U;
  // CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE+16]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo(u);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE-5]]:10: error: binding pattern has incomplete type `U` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:   let u: Cpp.U;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: union U;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_param_type_previously_imported.carbon:[[@LINE-13]]:15: error: expected `=`; `let` declaration must have an initializer [ExpectedInitializerAfterLet]
  // CHECK:STDERR:   let u: Cpp.U;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  Cpp.foo(u);
}

// ============================================================================
// Forward-declared union as parameter type imported twice
// ============================================================================

// --- double_decl_value_param_type.h

union U;

auto foo1(U) -> void;
auto foo2(U) -> void;

// --- fail_import_double_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./double_decl_value_param_type.h:4:6: error: argument type 'U' is incomplete [CppInteropParseError]
// CHECK:STDERR:     4 | auto foo1(U) -> void;
// CHECK:STDERR:       |      ^~~~
// CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./double_decl_value_param_type.h:2:7: note: forward declaration of 'U' [CppInteropParseNote]
// CHECK:STDERR:     2 | union U;
// CHECK:STDERR:       |       ^
import Cpp library "double_decl_value_param_type.h";

fn F() {
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+12]]:3: note: in `Cpp` name lookup for `foo1` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo1({} as Cpp.U);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./double_decl_value_param_type.h:5:6: error: argument type 'U' is incomplete [CppInteropParseError]
  // CHECK:STDERR:     5 | auto foo2(U) -> void;
  // CHECK:STDERR:       |      ^~~~
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./double_decl_value_param_type.h:2:7: note: forward declaration of 'U' [CppInteropParseNote]
  // CHECK:STDERR:     2 | union U;
  // CHECK:STDERR:       |       ^
  Cpp.foo1({} as Cpp.U);
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+20]]:3: note: in `Cpp` name lookup for `foo2` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo2({} as Cpp.U);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-5]]:12: error: invalid use of incomplete type `Cpp.U` [IncompleteTypeInConversion]
  // CHECK:STDERR:   Cpp.foo1({} as Cpp.U);
  // CHECK:STDERR:            ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-23]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./double_decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: union U;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE+8]]:12: error: invalid use of incomplete type `Cpp.U` [IncompleteTypeInConversion]
  // CHECK:STDERR:   Cpp.foo2({} as Cpp.U);
  // CHECK:STDERR:            ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_double_decl_value_param_type.carbon:[[@LINE-31]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./double_decl_value_param_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: union U;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  Cpp.foo2({} as Cpp.U);
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

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.U);
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

// --- fail_import_definition_single_data_member_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_single_data_member_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_import_definition_single_data_member_value_param_type.carbon:[[@LINE+7]]:11: error: cannot convert expression of type `{}` to `Cpp.U` with `as` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.U);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_definition_single_data_member_value_param_type.carbon:[[@LINE+4]]:11: note: type `{}` does not implement interface `Core.As(Cpp.U)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.U);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo({} as Cpp.U);
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

// --- fail_import_definition_multiple_data_members_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_multiple_data_members_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_import_definition_multiple_data_members_value_param_type.carbon:[[@LINE+7]]:11: error: cannot convert expression of type `{}` to `Cpp.U` with `as` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.U);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_definition_multiple_data_members_value_param_type.carbon:[[@LINE+4]]:11: note: type `{}` does not implement interface `Core.As(Cpp.U)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Cpp.foo({} as Cpp.U);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo({} as Cpp.U);
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

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.N.U);
  // Check that the parameter type was imported correctly.
  var x: Cpp.N.U;
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

fn F() {
  //@dump-sem-ir-begin
  Cpp.N1.foo({} as Cpp.N1.N2.U);
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

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.O.U);
  var x: Cpp.O;
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

fn F() {
  //@dump-sem-ir-begin
  Cpp.U.bar();
  Cpp.foo({} as Cpp.U);
  //@dump-sem-ir-end
}

// --- import_definition_and_static_method_call_after.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.U);
  Cpp.U.bar();
  //@dump-sem-ir-end
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
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+13]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE+9]]:3: error: function returns incomplete type `Cpp.U` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./decl_value_return_type.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: union U;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_import_decl_value_return_type.carbon: note: return type declared here [IncompleteReturnTypeHere]
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

// CHECK:STDOUT: --- fail_import_decl_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc28_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc28_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %U.ref: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_no_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %U.val: %U = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.0b7: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%U) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.2fa: %T.as.Destroy.impl.Op.type.0b7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
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
// CHECK:STDOUT:   %U.ref: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %.loc8_12.2: ref %U = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %U = class_init (), %.loc8_12.2 [concrete = constants.%U.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %U = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %U = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %U = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %U = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_22: %ptr.86f = addr_of %.loc8_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_22)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_12.4, constants.%T.as.Destroy.impl.Op.2fa
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.4, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_12: %ptr.86f = addr_of %.loc8_12.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_definition_single_data_member_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
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
// CHECK:STDOUT:   %U.ref: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %.loc15_14: %U = converted %.loc15_12, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_definition_multiple_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
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
// CHECK:STDOUT:   %U.ref: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %.loc15_14: %U = converted %.loc15_12, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of <error> [concrete = <error>]
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
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.87e: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %U.val: %U = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.2e1: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%U) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.d5d: %T.as.Destroy.impl.Op.type.2e1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
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
// CHECK:STDOUT:   %U.ref.loc8: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %.loc8_12.2: ref %U = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %U = class_init (), %.loc8_12.2 [concrete = constants.%U.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %U = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %U = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %U = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %U = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_24: %ptr.87e = addr_of %.loc8_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_24)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.eb9 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.eb9 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %U = var %x.var_patt
// CHECK:STDOUT:   %.loc10: type = splice_block %U.ref.loc10 [concrete = constants.%U] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %U.ref.loc10: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %U = bind_name x, %x.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %x.var, constants.%T.as.Destroy.impl.Op.d5d
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %x.var, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10: %ptr.87e = addr_of %x.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_12.4, constants.%T.as.Destroy.impl.Op.d5d
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_12.4, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_12: %ptr.87e = addr_of %.loc8_12.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_relative_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.8c1: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %U.val: %U = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.ee1: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%U) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.28c: %T.as.Destroy.impl.Op.type.ee1 = struct_value () [concrete]
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
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
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
// CHECK:STDOUT:   %U.ref: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %.loc8_15.2: ref %U = temporary_storage
// CHECK:STDOUT:   %.loc8_15.3: init %U = class_init (), %.loc8_15.2 [concrete = constants.%U.val]
// CHECK:STDOUT:   %.loc8_15.4: ref %U = temporary %.loc8_15.2, %.loc8_15.3
// CHECK:STDOUT:   %.loc8_17.1: ref %U = converted %.loc8_15.1, %.loc8_15.4
// CHECK:STDOUT:   %.loc8_17.2: %U = bind_value %.loc8_17.1
// CHECK:STDOUT:   %.loc8_17.3: ref %U = value_as_ref %.loc8_17.2
// CHECK:STDOUT:   %addr.loc8_31: %ptr.8c1 = addr_of %.loc8_17.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_31)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_15.4, constants.%T.as.Destroy.impl.Op.28c
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_15.4, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_15: %ptr.8c1 = addr_of %.loc8_15.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_outer_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %O: type = class_type @O [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.a6c: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %U.val: %U = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cff: type = pattern_type %O [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.1b8: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%O) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.2df: %T.as.Destroy.impl.Op.type.1b8 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.820: type = ptr_type %O [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.8f9: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%U) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.f3b: %T.as.Destroy.impl.Op.type.8f9 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %O.decl: type = class_decl @O [concrete = constants.%O] {} {}
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
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
// CHECK:STDOUT:   %U.ref: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %.loc8_12.2: ref %U = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %U = class_init (), %.loc8_12.2 [concrete = constants.%U.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %U = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %U = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %U = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %U = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_24: %ptr.a6c = addr_of %.loc8_14.3
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
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_12.4, constants.%T.as.Destroy.impl.Op.f3b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_12.4, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_12: %ptr.a6c = addr_of %.loc8_12.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %U.bar.type: type = fn_type @U.bar [concrete]
// CHECK:STDOUT:   %U.bar: %U.bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.val: %U = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.0b7: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%U) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.2fa: %T.as.Destroy.impl.Op.type.0b7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %U.bar.decl: %U.bar.type = fn_decl @U.bar [concrete = constants.%U.bar] {} {}
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
// CHECK:STDOUT:   %U.ref.loc8: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %bar.ref: %U.bar.type = name_ref bar, imports.%U.bar.decl [concrete = constants.%U.bar]
// CHECK:STDOUT:   %U.bar.call: init %empty_tuple.type = call %bar.ref()
// CHECK:STDOUT:   %Cpp.ref.loc9_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc9_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc9_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %U.ref.loc9: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %.loc9_12.2: ref %U = temporary_storage
// CHECK:STDOUT:   %.loc9_12.3: init %U = class_init (), %.loc9_12.2 [concrete = constants.%U.val]
// CHECK:STDOUT:   %.loc9_12.4: ref %U = temporary %.loc9_12.2, %.loc9_12.3
// CHECK:STDOUT:   %.loc9_14.1: ref %U = converted %.loc9_12.1, %.loc9_12.4
// CHECK:STDOUT:   %.loc9_14.2: %U = bind_value %.loc9_14.1
// CHECK:STDOUT:   %.loc9_14.3: ref %U = value_as_ref %.loc9_14.2
// CHECK:STDOUT:   %addr.loc9_22: %ptr.86f = addr_of %.loc9_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc9_22)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc9_12.4, constants.%T.as.Destroy.impl.Op.2fa
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc9_12.4, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc9_12: %ptr.86f = addr_of %.loc9_12.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc9_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %U.val: %U = struct_value () [concrete]
// CHECK:STDOUT:   %U.bar.type: type = fn_type @U.bar [concrete]
// CHECK:STDOUT:   %U.bar: %U.bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.0b7: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%U) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.2fa: %T.as.Destroy.impl.Op.type.0b7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
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
// CHECK:STDOUT:   %U.bar.decl: %U.bar.type = fn_decl @U.bar [concrete = constants.%U.bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %U.ref.loc8: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %.loc8_12.2: ref %U = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %U = class_init (), %.loc8_12.2 [concrete = constants.%U.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %U = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_14.1: ref %U = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_14.2: %U = bind_value %.loc8_14.1
// CHECK:STDOUT:   %.loc8_14.3: ref %U = value_as_ref %.loc8_14.2
// CHECK:STDOUT:   %addr.loc8_22: %ptr.86f = addr_of %.loc8_14.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_22)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %U.ref.loc9: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %bar.ref: %U.bar.type = name_ref bar, imports.%U.bar.decl [concrete = constants.%U.bar]
// CHECK:STDOUT:   %U.bar.call: init %empty_tuple.type = call %bar.ref()
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_12.4, constants.%T.as.Destroy.impl.Op.2fa
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.4, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_12: %ptr.86f = addr_of %.loc8_12.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
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
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %u.ref: %ptr = name_ref u, %u
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%u.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
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
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %u.ref: %ptr = name_ref u, %u
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%u.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_value_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.0b7: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%U) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.2fa: %T.as.Destroy.impl.Op.type.0b7 = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc8_11.1: ref %U = temporary_storage
// CHECK:STDOUT:   %addr.loc8_11.1: %ptr.86f = addr_of %.loc8_11.1
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_11.1)
// CHECK:STDOUT:   %.loc8_11.2: init %U = in_place_init %foo__carbon_thunk.call, %.loc8_11.1
// CHECK:STDOUT:   %.loc8_11.3: ref %U = temporary %.loc8_11.1, %.loc8_11.2
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.3, constants.%T.as.Destroy.impl.Op.2fa
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_11.3, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11.2: %ptr.86f = addr_of %.loc8_11.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_11.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_decl_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %U [concrete]
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
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %U [concrete]
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
