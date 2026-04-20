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

//@dump-sem-ir-begin
import Cpp inline '''

// A C++ function.
inline void func() {}

''';
//@dump-sem-ir-end

fn Run() {
  //@dump-sem-ir-begin
  Cpp.func();
  //@dump-sem-ir-end
}

// --- with_language_marker.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
import Cpp inline '''c++

// A C++ function.
inline void another_func() {}

''';
//@dump-sem-ir-end

fn Run() {
  //@dump-sem-ir-begin
  Cpp.another_func();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- use_inline_function_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %func.cpp_overload_set.type: type = cpp_overload_set_type @func.cpp_overload_set [concrete]
// CHECK:STDOUT:   %func.cpp_overload_set.value: %func.cpp_overload_set.type = cpp_overload_set_value @func.cpp_overload_set [concrete]
// CHECK:STDOUT:   %func.type: type = fn_type @func [concrete]
// CHECK:STDOUT:   %func: %func.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .func = %func.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %func.cpp_overload_set.value: %func.cpp_overload_set.type = cpp_overload_set_value @func.cpp_overload_set [concrete = constants.%func.cpp_overload_set.value]
// CHECK:STDOUT:   %func.decl: %func.type = fn_decl @func [concrete = constants.%func] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %func.ref: %func.cpp_overload_set.type = name_ref func, imports.%func.cpp_overload_set.value [concrete = constants.%func.cpp_overload_set.value]
// CHECK:STDOUT:   %func.call: init %empty_tuple.type = call imports.%func.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- with_language_marker.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %another_func.cpp_overload_set.type: type = cpp_overload_set_type @another_func.cpp_overload_set [concrete]
// CHECK:STDOUT:   %another_func.cpp_overload_set.value: %another_func.cpp_overload_set.type = cpp_overload_set_value @another_func.cpp_overload_set [concrete]
// CHECK:STDOUT:   %another_func.type: type = fn_type @another_func [concrete]
// CHECK:STDOUT:   %another_func: %another_func.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .another_func = %another_func.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %another_func.cpp_overload_set.value: %another_func.cpp_overload_set.type = cpp_overload_set_value @another_func.cpp_overload_set [concrete = constants.%another_func.cpp_overload_set.value]
// CHECK:STDOUT:   %another_func.decl: %another_func.type = fn_decl @another_func [concrete = constants.%another_func] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %another_func.ref: %another_func.cpp_overload_set.type = name_ref another_func, imports.%another_func.cpp_overload_set.value [concrete = constants.%another_func.cpp_overload_set.value]
// CHECK:STDOUT:   %another_func.call: init %empty_tuple.type = call imports.%another_func.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
