// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/unsupported_decl_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/unsupported_decl_type.carbon

// --- structured_binding.h

struct X { int a, b; };
auto [a, b] = X{};

// --- fail_todo_use_structured_binding.carbon

library "[[@TEST_NAME]]";

import Cpp library "structured_binding.h";

fn F() -> i32 {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_use_structured_binding.carbon:[[@LINE+4]]:17: error: member name `C` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   var unused c: Cpp.C({});
  // CHECK:STDERR:                 ^~~~~
  // CHECK:STDERR:
  var unused c: Cpp.C({});
  //@dump-sem-ir-end
// CHECK:STDERR: fail_todo_use_structured_binding.carbon:[[@LINE+4]]:1: error: missing `return` at end of function with declared return type [MissingReturnStatement]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

// CHECK:STDOUT: --- fail_todo_use_structured_binding.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: <error> = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: <error> = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref <error> = var %c.var_patt [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %c: ref <error> = ref_binding c, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
