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
// CHECK:STDOUT:   %.082: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.e7d: %.082 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %.152: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.172: %.152 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .F1 = %.e7d
// CHECK:STDOUT:     .F2 = %.172
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.e7d: %.082 = cpp_overload_set_value @<null name> [concrete = constants.%.e7d]
// CHECK:STDOUT:   %.172: %.152 = cpp_overload_set_value @<null name> [concrete = constants.%.172]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %F1.ref: %.082 = name_ref F1, imports.%.e7d [concrete = constants.%.e7d]
// CHECK:STDOUT:   %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %F2.ref: %.152 = name_ref F2, imports.%.172 [concrete = constants.%.172]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
