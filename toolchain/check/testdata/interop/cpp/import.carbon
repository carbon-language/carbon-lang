// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/import.carbon

// ============================================================================
// Import C++ namespace indirectly
// ============================================================================

// --- namespace.h

namespace MyNamespace {

class MyClass {};

}  // namespace MyNamespace

// --- namespace_api.carbon

library "[[@TEST_NAME]]";

import Cpp library "namespace.h";

alias MyNamespaceAlias = Cpp.MyNamespace;

// --- fail_todo_import_namespace_api.carbon

library "[[@TEST_NAME]]";

import library "namespace_api";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_namespace_api.carbon:[[@LINE+4]]:10: error: member name `MyClass` not found in `Cpp.MyNamespace` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   var x: MyNamespaceAlias.MyClass;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var x: MyNamespaceAlias.MyClass;
  //@dump-sem-ir-end
}

// ============================================================================
// Import C++ function indirectly
// ============================================================================

// --- function.h

auto foo_short(short x) -> void;
auto foo_int(int x) -> void;

// --- function_api.carbon

library "[[@TEST_NAME]]";

import Cpp library "function.h";

alias FooShort = Cpp.foo_short;
alias FooInt = Cpp.foo_int;

// --- todo_import_function_api.carbon

library "[[@TEST_NAME]]";

import library "function_api";

// TODO: Fix this test as a follow-up of https://github.com/carbon-language/carbon-lang/pull/5891.
fn F() {
  //@dump-sem-ir-begin
  // FooShort(8 as i16);
  // FooInt(9);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_todo_import_namespace_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: <error> = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref <error> = var %x.var_patt [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x: <error> = bind_name x, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_import_function_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
