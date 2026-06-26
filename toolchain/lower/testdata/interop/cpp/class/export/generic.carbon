// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/export/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/export/generic.carbon

// --- generic_class.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C(T: type) {
  var t: T;
}

inline Cpp '''
class A {};
class B {};

void F() {
  // CHECK:STDERR: generic_class.carbon:[[@LINE+15]]:11: error: no member named 'C' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    28 |   Carbon::C<A> c1;
  // CHECK:STDERR:       |           ^
  // CHECK:STDERR:
  // CHECK:STDERR: generic_class.carbon:[[@LINE+11]]:13: error: 'A' does not refer to a value [CppInteropParseError]
  // CHECK:STDERR:    28 |   Carbon::C<A> c1;
  // CHECK:STDERR:       |             ^
  // CHECK:STDERR: generic_class.carbon:[[@LINE-11]]:7: note: declared here [CppInteropParseNote]
  // CHECK:STDERR:     9 | class A {};
  // CHECK:STDERR:       |       ^
  // CHECK:STDERR:
  // CHECK:STDERR: generic_class.carbon:[[@LINE+4]]:16: error: use of undeclared identifier 'c1' [CppInteropParseError]
  // CHECK:STDERR:    28 |   Carbon::C<A> c1;
  // CHECK:STDERR:       |                ^~
  // CHECK:STDERR:
  Carbon::C<A> c1;
  // CHECK:STDERR: generic_class.carbon:[[@LINE+4]]:3: error: use of undeclared identifier 'c1' [CppInteropParseError]
  // CHECK:STDERR:    33 |   c1.t = A();
  // CHECK:STDERR:       |   ^~
  // CHECK:STDERR:
  c1.t = A();
  // CHECK:STDERR: generic_class.carbon:[[@LINE+15]]:11: error: no member named 'C' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    49 |   Carbon::C<B> c2;
  // CHECK:STDERR:       |           ^
  // CHECK:STDERR:
  // CHECK:STDERR: generic_class.carbon:[[@LINE+11]]:13: error: 'B' does not refer to a value [CppInteropParseError]
  // CHECK:STDERR:    49 |   Carbon::C<B> c2;
  // CHECK:STDERR:       |             ^
  // CHECK:STDERR: generic_class.carbon:[[@LINE-31]]:7: note: declared here [CppInteropParseNote]
  // CHECK:STDERR:    10 | class B {};
  // CHECK:STDERR:       |       ^
  // CHECK:STDERR:
  // CHECK:STDERR: generic_class.carbon:[[@LINE+4]]:16: error: use of undeclared identifier 'c2' [CppInteropParseError]
  // CHECK:STDERR:    49 |   Carbon::C<B> c2;
  // CHECK:STDERR:       |                ^~
  // CHECK:STDERR:
  Carbon::C<B> c2;
  // CHECK:STDERR: generic_class.carbon:[[@LINE+4]]:3: error: use of undeclared identifier 'c2' [CppInteropParseError]
  // CHECK:STDERR:    54 |   c2.t = B();
  // CHECK:STDERR:       |   ^~
  // CHECK:STDERR:
  c2.t = B();
}
''';

fn Run() {
  Cpp.F();
}

