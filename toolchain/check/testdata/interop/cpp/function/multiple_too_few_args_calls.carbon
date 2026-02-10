// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/multiple_too_few_args_calls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/multiple_too_few_args_calls.carbon

// --- fail_call_too_few_args.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct X {};
void F1(X x);
void F2(X x);
''';

fn Call() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+7]]:10: error: no matching function for call to 'F1' [CppInteropParseError]
  // CHECK:STDERR:    19 |   Cpp.F1();
  // CHECK:STDERR:       |          ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-9]]:6: note: candidate function not viable: requires single argument 'x', but no arguments were provided [CppInteropParseNote]
  // CHECK:STDERR:     6 | void F1(X x);
  // CHECK:STDERR:       |      ^  ~~~
  // CHECK:STDERR:
  Cpp.F1();
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+7]]:10: error: no matching function for call to 'F2' [CppInteropParseError]
  // CHECK:STDERR:    27 |   Cpp.F2();
  // CHECK:STDERR:       |          ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-16]]:6: note: candidate function not viable: requires single argument 'x', but no arguments were provided [CppInteropParseNote]
  // CHECK:STDERR:     7 | void F2(X x);
  // CHECK:STDERR:       |      ^  ~~~
  // CHECK:STDERR:
  Cpp.F2();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_call_too_few_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F1.cpp_overload_set.type: type = cpp_overload_set_type @F1.cpp_overload_set [concrete]
// CHECK:STDOUT:   %F1.cpp_overload_set.value: %F1.cpp_overload_set.type = cpp_overload_set_value @F1.cpp_overload_set [concrete]
// CHECK:STDOUT:   %F2.cpp_overload_set.type: type = cpp_overload_set_type @F2.cpp_overload_set [concrete]
// CHECK:STDOUT:   %F2.cpp_overload_set.value: %F2.cpp_overload_set.type = cpp_overload_set_value @F2.cpp_overload_set [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .F1 = %F1.cpp_overload_set.value
// CHECK:STDOUT:     .F2 = %F2.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.cpp_overload_set.value: %F1.cpp_overload_set.type = cpp_overload_set_value @F1.cpp_overload_set [concrete = constants.%F1.cpp_overload_set.value]
// CHECK:STDOUT:   %F2.cpp_overload_set.value: %F2.cpp_overload_set.type = cpp_overload_set_value @F2.cpp_overload_set [concrete = constants.%F2.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %F1.ref: %F1.cpp_overload_set.type = name_ref F1, imports.%F1.cpp_overload_set.value [concrete = constants.%F1.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %F2.ref: %F2.cpp_overload_set.type = name_ref F2, imports.%F2.cpp_overload_set.value [concrete = constants.%F2.cpp_overload_set.value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
