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

// CHECK:STDERR: fail_todo_use_structured_binding.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./structured_binding.h:3:7: error: semantics TODO: `Unsupported: Declaration type Binding` [SemanticsTodo]
// CHECK:STDERR: auto [a, b] = X{};
// CHECK:STDERR:       ^
import Cpp library "structured_binding.h";

fn F() -> i32 {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_use_structured_binding.carbon:[[@LINE+4]]:10: note: in `Cpp` name lookup for `a` [InCppNameLookup]
  // CHECK:STDERR:   return Cpp.a;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return Cpp.a;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_todo_use_structured_binding.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .a = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %a.ref: <error> = name_ref a, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
