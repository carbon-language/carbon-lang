// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
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

// --- fail_todo_import_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_decl_value_param_type.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo({});
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_import_decl_value_param_type.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo({});
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.foo({});
  //@dump-sem-ir-end
}

// --- fail_todo_import_decl_value_param_type_previously_imported.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_value_param_type.h";

fn F() {
  // CHECK:STDERR: fail_todo_import_decl_value_param_type_previously_imported.carbon:[[@LINE+11]]:10: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
  // CHECK:STDERR:   let u: Cpp.U;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_todo_import_decl_value_param_type_previously_imported.carbon:[[@LINE+8]]:10: note: in `Cpp` name lookup for `U` [InCppNameLookup]
  // CHECK:STDERR:   let u: Cpp.U;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_decl_value_param_type_previously_imported.carbon:[[@LINE+4]]:15: error: expected `=`; `let` declaration must have an initializer [ExpectedInitializerAfterLet]
  // CHECK:STDERR:   let u: Cpp.U;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  let u: Cpp.U;
  Cpp.foo(u);
}

// ============================================================================
// Forward-declared union as parameter type imported twice
// ============================================================================

// --- double_decl_value_param_type.h

union U;

auto foo1(U) -> void;
auto foo2(U) -> void;

// --- fail_todo_import_double_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "double_decl_value_param_type.h";

fn F() {
  // CHECK:STDERR: fail_todo_import_double_decl_value_param_type.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo1({});
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_import_double_decl_value_param_type.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo1` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo1({});
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  Cpp.foo1({});
  Cpp.foo2({});
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
  Cpp.foo({});
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
  // CHECK:STDERR: fail_import_definition_single_data_member_value_param_type.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Non-empty union` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo({});
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_import_definition_single_data_member_value_param_type.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo({});
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.foo({});
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
  // CHECK:STDERR: fail_import_definition_multiple_data_members_value_param_type.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Non-empty union` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo({});
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_import_definition_multiple_data_members_value_param_type.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo({});
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.foo({});
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
  Cpp.foo({});
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
  Cpp.N1.foo({});
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
  Cpp.foo({});
  //@dump-sem-ir-end
}

// --- import_definition_and_static_method_call_after.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({});
  Cpp.U.bar();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to forward-declared union as parameter type
// ============================================================================

// --- decl_pointer_param_type.h

union U;

auto foo(U* _Nonnull) -> void;

// --- fail_todo_import_decl_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_param_type.h";

// CHECK:STDERR: fail_todo_import_decl_pointer_param_type.carbon:[[@LINE+7]]:9: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
// CHECK:STDERR: fn F(u: Cpp.U*) {
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_todo_import_decl_pointer_param_type.carbon:[[@LINE+4]]:9: note: in `Cpp` name lookup for `U` [InCppNameLookup]
// CHECK:STDERR: fn F(u: Cpp.U*) {
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR:
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

// --- fail_todo_import_decl_value_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_value_return_type.h";

fn F() {
  // CHECK:STDERR: fail_todo_import_decl_value_return_type.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_import_decl_value_return_type.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~
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

auto foo() -> U*;

// --- fail_todo_import_decl_pointer_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_decl_pointer_return_type.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: nullable pointer: U *` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_import_decl_pointer_return_type.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
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

// CHECK:STDOUT: --- fail_todo_import_decl_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
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
// CHECK:STDOUT:   %.loc15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_no_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.val: %U = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.0b7: type = fn_type @Op.2, @impl(%U) [concrete]
// CHECK:STDOUT:   %Op.2fa: %Op.type.0b7 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
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
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_12.2: ref %U = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %U = class_init (), %.loc8_12.2 [concrete = constants.%U.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %U = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_12.5: ref %U = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_12.6: %U = bind_value %.loc8_12.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_12.6)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_12.2, constants.%Op.2fa
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of %.loc8_12.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_definition_single_data_member_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
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
// CHECK:STDOUT:   %.loc15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_definition_multiple_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
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
// CHECK:STDOUT:   %.loc15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.val: %U = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.2e1: type = fn_type @Op.2, @impl(%U) [concrete]
// CHECK:STDOUT:   %Op.d5d: %Op.type.2e1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.87e: type = ptr_type %U [concrete]
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
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_12.2: ref %U = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %U = class_init (), %.loc8_12.2 [concrete = constants.%U.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %U = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_12.5: ref %U = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_12.6: %U = bind_value %.loc8_12.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_12.6)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_12.2, constants.%Op.d5d
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.87e = addr_of %.loc8_12.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_relative_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.val: %U = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.ee1: type = fn_type @Op.2, @impl(%U) [concrete]
// CHECK:STDOUT:   %Op.28c: %Op.type.ee1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.8c1: type = ptr_type %U [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N1 = %N1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1: <namespace> = namespace [concrete] {
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
// CHECK:STDOUT:   %N1.ref: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc8_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_15.2: ref %U = temporary_storage
// CHECK:STDOUT:   %.loc8_15.3: init %U = class_init (), %.loc8_15.2 [concrete = constants.%U.val]
// CHECK:STDOUT:   %.loc8_15.4: ref %U = temporary %.loc8_15.2, %.loc8_15.3
// CHECK:STDOUT:   %.loc8_15.5: ref %U = converted %.loc8_15.1, %.loc8_15.4
// CHECK:STDOUT:   %.loc8_15.6: %U = bind_value %.loc8_15.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_15.6)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_15.2, constants.%Op.28c
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_15.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.8c1 = addr_of %.loc8_15.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %bar.type: type = fn_type @bar [concrete]
// CHECK:STDOUT:   %bar: %bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.val: %U = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.0b7: type = fn_type @Op.2, @impl(%U) [concrete]
// CHECK:STDOUT:   %Op.2fa: %Op.type.0b7 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = %U.decl
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U.decl: type = class_decl @U [concrete = constants.%U] {} {}
// CHECK:STDOUT:   %bar.decl: %bar.type = fn_decl @bar [concrete = constants.%bar] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %U.ref: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %bar.ref: %bar.type = name_ref bar, imports.%bar.decl [concrete = constants.%bar]
// CHECK:STDOUT:   %bar.call: init %empty_tuple.type = call %bar.ref()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc9_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_12.2: ref %U = temporary_storage
// CHECK:STDOUT:   %.loc9_12.3: init %U = class_init (), %.loc9_12.2 [concrete = constants.%U.val]
// CHECK:STDOUT:   %.loc9_12.4: ref %U = temporary %.loc9_12.2, %.loc9_12.3
// CHECK:STDOUT:   %.loc9_12.5: ref %U = converted %.loc9_12.1, %.loc9_12.4
// CHECK:STDOUT:   %.loc9_12.6: %U = bind_value %.loc9_12.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc9_12.6)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc9_12.2, constants.%Op.2fa
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc9_12.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of %.loc9_12.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %U: type = class_type @U [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.val: %U = struct_value () [concrete]
// CHECK:STDOUT:   %bar.type: type = fn_type @bar [concrete]
// CHECK:STDOUT:   %bar: %bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.0b7: type = fn_type @Op.2, @impl(%U) [concrete]
// CHECK:STDOUT:   %Op.2fa: %Op.type.0b7 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
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
// CHECK:STDOUT:   %bar.decl: %bar.type = fn_decl @bar [concrete = constants.%bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc8_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_12.2: ref %U = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %U = class_init (), %.loc8_12.2 [concrete = constants.%U.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %U = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_12.5: ref %U = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_12.6: %U = bind_value %.loc8_12.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_12.6)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %U.ref: type = name_ref U, imports.%U.decl [concrete = constants.%U]
// CHECK:STDOUT:   %bar.ref: %bar.type = name_ref bar, imports.%bar.decl [concrete = constants.%bar]
// CHECK:STDOUT:   %bar.call: init %empty_tuple.type = call %bar.ref()
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_12.2, constants.%Op.2fa
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of %.loc8_12.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_decl_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .U = <error>
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
// CHECK:STDOUT: fn @F(%u.param: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %u.ref: <error> = name_ref u, %u
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(<error>)
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
// CHECK:STDOUT:   %Op.type.0b7: type = fn_type @Op.2, @impl(%U) [concrete]
// CHECK:STDOUT:   %Op.2fa: %Op.type.0b7 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.86f: type = ptr_type %U [concrete]
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
// CHECK:STDOUT:   %.loc8_11.1: ref %U = temporary_storage
// CHECK:STDOUT:   %foo.call: init %U = call %foo.ref() to %.loc8_11.1
// CHECK:STDOUT:   %.loc8_11.2: ref %U = temporary %.loc8_11.1, %foo.call
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_11.1, constants.%Op.2fa
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_11.1, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.86f = addr_of %.loc8_11.1
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_decl_pointer_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
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
// CHECK:STDOUT:   %foo.call: init <error> = call %foo.ref()
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
