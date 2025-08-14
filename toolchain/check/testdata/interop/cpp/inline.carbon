// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/inline.carbon

// --- use_inline_function_decl.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''

// A C++ function.
inline void func() {}

''';

fn Run() {
  //@dump-sem-ir-begin
  Cpp.func();
  //@dump-sem-ir-end
}

// --- with_language_marker.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++

// A C++ function.
inline void another_func() {}

''';

fn Run() {
  //@dump-sem-ir-begin
  Cpp.another_func();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- use_inline_function_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %func.type: type = fn_type @func [concrete]
// CHECK:STDOUT:   %func: %func.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .func = %func.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %func.decl: %func.type = fn_decl @func [concrete = constants.%func] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %func.ref: %func.type = name_ref func, imports.%func.decl [concrete = constants.%func]
// CHECK:STDOUT:   %func.call: init %empty_tuple.type = call %func.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- with_language_marker.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %another_func.type: type = fn_type @another_func [concrete]
// CHECK:STDOUT:   %another_func: %another_func.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .another_func = %another_func.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %another_func.decl: %another_func.type = fn_decl @another_func [concrete = constants.%another_func] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %another_func.ref: %another_func.type = name_ref another_func, imports.%another_func.decl [concrete = constants.%another_func]
// CHECK:STDOUT:   %another_func.call: init %empty_tuple.type = call %another_func.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
