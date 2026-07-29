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
  // CHECK:STDERR: generic_class.carbon:[[@LINE+7]]:16: error: implicit instantiation of undefined template 'Carbon::C<A>' [CppInteropParseError]
  // CHECK:STDERR:    20 |   Carbon::C<A> c1;
  // CHECK:STDERR:       |                ^
  // CHECK:STDERR: generic_class.carbon:[[@LINE-12]]:18: note: template is declared here [CppInteropParseNote]
  // CHECK:STDERR:     4 | class C(T: type) {
  // CHECK:STDERR:       |                  ^
  // CHECK:STDERR:
  Carbon::C<A> c1;
  c1.t = A();
  // CHECK:STDERR: generic_class.carbon:[[@LINE+7]]:16: error: implicit instantiation of undefined template 'Carbon::C<B>' [CppInteropParseError]
  // CHECK:STDERR:    29 |   Carbon::C<B> c2;
  // CHECK:STDERR:       |                ^
  // CHECK:STDERR: generic_class.carbon:[[@LINE-21]]:18: note: template is declared here [CppInteropParseNote]
  // CHECK:STDERR:     4 | class C(T: type) {
  // CHECK:STDERR:       |                  ^
  // CHECK:STDERR:
  Carbon::C<B> c2;
  c2.t = B();
}
''';

fn Run() {
  Cpp.F();
}

