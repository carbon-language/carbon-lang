// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/import/indirect.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/import/indirect.carbon

// --- shared.h

namespace A {
  struct X {
    X(int);
    int x;
  };
}

// --- direct_import.carbon

library "[[@TEST_NAME]]";

import Cpp library "shared.h";

fn F() -> Cpp.A.X { return 1; }

// --- indirect_import.carbon
// CHECK:STDERR: indirect_import.carbon: error: semantics TODO: `Unsupported: Importing C++ indirectly` [SemanticsTodo]
// CHECK:STDERR:

library "[[@TEST_NAME]]";

import Cpp library "shared.h";
import library "direct_import";

fn G() -> i32 {
  //@dump-sem-ir-begin
  return F().x;
  //@dump-sem-ir-end
}

// --- fail_todo_not_included.carbon
// CHECK:STDERR: fail_todo_not_included.carbon: error: semantics TODO: `Unsupported: Importing C++ indirectly` [SemanticsTodo]
// CHECK:STDERR:

library "[[@TEST_NAME]]";

import library "direct_import";

fn G() -> i32 {
  // TODO: This should eventually work, by importing the C++ AST from
  // `direct_import`.
  return F().x;
}

// CHECK:STDOUT: --- indirect_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//direct_import, F, loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: <error> = call %F.ref()
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
