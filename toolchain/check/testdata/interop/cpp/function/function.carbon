// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/function.carbon

// ============================================================================
// Global
// ============================================================================

// --- global.h

auto foo() -> void;

// --- import_global.carbon

library "[[@TEST_NAME]]";

import Cpp library "global.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// --- fail_import_global_use_different_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "global.h";

fn MyF() {
  // CHECK:STDERR: fail_import_global_use_different_name.carbon:[[@LINE+4]]:3: error: member name `bar` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.bar();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.bar();
}

// ============================================================================
// Carbon keyword name
// ============================================================================

// --- special_name.h

auto base() -> void;

// --- fail_import_special_name_call_unescaped.carbon

library "[[@TEST_NAME]]";

import Cpp library "special_name.h";

fn MyF() {
  // CHECK:STDERR: fail_import_special_name_call_unescaped.carbon:[[@LINE+4]]:3: error: member name `base` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.base();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  Cpp.base();
}

// --- import_special_name_call_escaped.carbon

library "[[@TEST_NAME]]";

import Cpp library "special_name.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.r#base();
  //@dump-sem-ir-end
}

// ============================================================================
// Overloaded
// ============================================================================

// --- overloaded.h

auto foo() -> void;
auto foo(int value) -> void;

// --- import_overloaded.carbon

library "[[@TEST_NAME]]";

import Cpp library "overloaded.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Variadic
// ============================================================================

// --- variadic.h

auto foo(int...) -> void;

// --- fail_todo_import_variadic.carbon

library "[[@TEST_NAME]]";

import Cpp library "variadic.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_variadic.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Variadic function` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_variadic.carbon:[[@LINE+4]]:3: note: in call to a Cpp (overloaded) function here [InCallToOverloadedCppFunction]
  // CHECK:STDERR:   Cpp.foo(1);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(1);
  //@dump-sem-ir-end
}

// ============================================================================
// Static
// ============================================================================

// --- static.h

static auto foo() -> void;

// --- import_static.carbon

library "[[@TEST_NAME]]";

import Cpp library "static.h";

fn F() {
  //@dump-sem-ir-begin
  // TODO: We should eventually warn or error that `Cpp.foo` has internal
  // linkage but is not defined.
  Cpp.foo();
  //@dump-sem-ir-end
}

// TODO: Test that template functions are unsupported.
//       This is not tested because template functions are not considered a single result when doing lookup.

// TODO: Add a test per unsupported param type. See https://github.com/carbon-language/carbon-lang/pull/5477/files/4321e21ed27d987fd71be182d292973fd9849df8#r2094655176

// TODO: Add a test per unsupported return type. See https://github.com/carbon-language/carbon-lang/pull/5477/files/4321e21ed27d987fd71be182d292973fd9849df8#r2094655176

// CHECK:STDOUT: --- import_global.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.273: type = overloaded_fn_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.273 = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.f88
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.f88: %.273 = overloaded_fn_decl @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.273 = name_ref foo, imports.%.f88 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_special_name_call_escaped.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.be0: type = overloaded_fn_type @base [concrete]
// CHECK:STDOUT:   %empty_struct: %.be0 = struct_value () [concrete]
// CHECK:STDOUT:   %base.type: type = fn_type @base [concrete]
// CHECK:STDOUT:   %base: %base.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .r#base = %.3e8
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.3e8: %.be0 = overloaded_fn_decl @base [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %base.decl: %base.type = fn_decl @base [concrete = constants.%base] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %base.ref: %.be0 = name_ref r#base, imports.%.3e8 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %base.call: init %empty_tuple.type = call imports.%base.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.273: type = overloaded_fn_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.273 = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.f88
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.f88: %.273 = overloaded_fn_decl @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.273 = name_ref foo, imports.%.f88 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_variadic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.273: type = overloaded_fn_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct: %.273 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.f88
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.f88: %.273 = overloaded_fn_decl @<null name> [concrete = constants.%empty_struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.273 = name_ref foo, imports.%.f88 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_static.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.273: type = overloaded_fn_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.273 = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.f88
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.f88: %.273 = overloaded_fn_decl @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.273 = name_ref foo, imports.%.f88 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
