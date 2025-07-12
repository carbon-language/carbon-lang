// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
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

// --- fail_todo_import_decl_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_value_param_type.h";

fn F() {
  // CHECK:STDERR: fail_todo_import_decl_value_param_type.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo({});
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_import_decl_value_param_type.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo({});
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.foo({});
}

// --- fail_todo_import_decl_value_param_type_previously_imported.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_value_param_type.h";

fn F() {
  // CHECK:STDERR: fail_todo_import_decl_value_param_type_previously_imported.carbon:[[@LINE+11]]:10: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
  // CHECK:STDERR:   let c: Cpp.C;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_todo_import_decl_value_param_type_previously_imported.carbon:[[@LINE+8]]:10: note: in `Cpp` name lookup for `C` [InCppNameLookup]
  // CHECK:STDERR:   let c: Cpp.C;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_decl_value_param_type_previously_imported.carbon:[[@LINE+4]]:15: error: expected `=`; `let` declaration must have an initializer [ExpectedInitializerAfterLet]
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
  Cpp.foo({});
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

// --- import_definition_single_data_member_value_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_single_data_member_value_param_type.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({});
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

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({});
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
  Cpp.foo({});
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
  Cpp.N1.foo({});
  //@dump-sem-ir-end
}

// ============================================================================
// Defined class and explicitly used
// ============================================================================

// --- definition_with_static_method.h

class C {
  static void bar();
};

auto foo(C) -> void;

// --- import_definition_and_static_method_call_before.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.C.bar();
  Cpp.foo({});
  //@dump-sem-ir-end
}

// --- import_definition_and_static_method_call_after.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition_with_static_method.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo({});
  Cpp.C.bar();
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer to forward-declared class as parameter type
// ============================================================================

// --- decl_pointer_param_type.h

class C;

auto foo(C* _Nonnull) -> void;

// --- fail_todo_import_decl_pointer_param_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_param_type.h";

// CHECK:STDERR: fail_todo_import_decl_pointer_param_type.carbon:[[@LINE+7]]:9: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
// CHECK:STDERR: fn F(c: Cpp.C*) {
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_todo_import_decl_pointer_param_type.carbon:[[@LINE+4]]:9: note: in `Cpp` name lookup for `C` [InCppNameLookup]
// CHECK:STDERR: fn F(c: Cpp.C*) {
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR:
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

// --- fail_todo_import_decl_pointer_return_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "decl_pointer_return_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_decl_pointer_return_type.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
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
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.1b3: type = fn_type @Op.2, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.21b: %Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
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
// CHECK:STDOUT:   %.loc8_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %C = class_init (), %.loc8_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %C = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_12.5: ref %C = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_12.6: %C = bind_value %.loc8_12.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_12.6)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_12.2, constants.%Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc8_12.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_single_data_member_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.1b3: type = fn_type @Op.2, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.21b: %Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
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
// CHECK:STDOUT:   %.loc8_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %C = class_init (), %.loc8_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %C = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_12.5: ref %C = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_12.6: %C = bind_value %.loc8_12.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_12.6)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_12.2, constants.%Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc8_12.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_multiple_data_members_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.1b3: type = fn_type @Op.2, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.21b: %Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
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
// CHECK:STDOUT:   %.loc8_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %C = class_init (), %.loc8_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %C = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_12.5: ref %C = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_12.6: %C = bind_value %.loc8_12.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_12.6)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_12.2, constants.%Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc8_12.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.9ae: type = fn_type @Op.2, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.dbb: %Op.type.9ae = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %C [concrete]
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
// CHECK:STDOUT:   %.loc8_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %C = class_init (), %.loc8_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %C = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_12.5: ref %C = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_12.6: %C = bind_value %.loc8_12.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_12.6)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_12.2, constants.%Op.dbb
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.838 = addr_of %.loc8_12.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_in_relative_namespace_value_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.b28: type = fn_type @Op.2, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.f48: %Op.type.b28 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c0c: type = ptr_type %C [concrete]
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
// CHECK:STDOUT:   %.loc8_15.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_15.3: init %C = class_init (), %.loc8_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_15.4: ref %C = temporary %.loc8_15.2, %.loc8_15.3
// CHECK:STDOUT:   %.loc8_15.5: ref %C = converted %.loc8_15.1, %.loc8_15.4
// CHECK:STDOUT:   %.loc8_15.6: %C = bind_value %.loc8_15.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_15.6)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_15.2, constants.%Op.f48
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_15.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.c0c = addr_of %.loc8_15.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %bar.type: type = fn_type @bar [concrete]
// CHECK:STDOUT:   %bar: %bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.1b3: type = fn_type @Op.2, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.21b: %Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
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
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %bar.ref: %bar.type = name_ref bar, imports.%bar.decl [concrete = constants.%bar]
// CHECK:STDOUT:   %bar.call: init %empty_tuple.type = call %bar.ref()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %.loc9_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc9_12.3: init %C = class_init (), %.loc9_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_12.4: ref %C = temporary %.loc9_12.2, %.loc9_12.3
// CHECK:STDOUT:   %.loc9_12.5: ref %C = converted %.loc9_12.1, %.loc9_12.4
// CHECK:STDOUT:   %.loc9_12.6: %C = bind_value %.loc9_12.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc9_12.6)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc9_12.2, constants.%Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc9_12.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc9_12.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_and_static_method_call_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %bar.type: type = fn_type @bar [concrete]
// CHECK:STDOUT:   %bar: %bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.1b3: type = fn_type @Op.2, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.21b: %Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
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
// CHECK:STDOUT:   %.loc8_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_12.3: init %C = class_init (), %.loc8_12.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_12.4: ref %C = temporary %.loc8_12.2, %.loc8_12.3
// CHECK:STDOUT:   %.loc8_12.5: ref %C = converted %.loc8_12.1, %.loc8_12.4
// CHECK:STDOUT:   %.loc8_12.6: %C = bind_value %.loc8_12.5
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_12.6)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %bar.ref: %bar.type = name_ref bar, imports.%bar.decl [concrete = constants.%bar]
// CHECK:STDOUT:   %bar.call: init %empty_tuple.type = call %bar.ref()
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_12.2, constants.%Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_12.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc8_12.2
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
// CHECK:STDOUT:     .C = <error>
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
// CHECK:STDOUT: fn @F(%c.param: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %c.ref: <error> = name_ref c, %c
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_pointer_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
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
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %c.ref: %ptr = name_ref c, %c
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%c.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition_value_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.1b3: type = fn_type @Op.2, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.21b: %Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
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
// CHECK:STDOUT:   %.loc8_11.1: ref %C = temporary_storage
// CHECK:STDOUT:   %foo.call: init %C = call %foo.ref() to %.loc8_11.1
// CHECK:STDOUT:   %.loc8_11.2: ref %C = temporary %.loc8_11.1, %foo.call
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8_11.1, constants.%Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_11.1, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc8_11.1
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
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [concrete]
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
