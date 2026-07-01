// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/export/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/export/generic.carbon

// --- generic_type_impls_interface.carbon
library "[[@TEST_NAME]]";
import Cpp;

interface I {
  fn Doit(self);
}

class A {
  impl as I {
    fn Doit(unused self) {}
  }
}
class B {
  impl as I {
    fn Doit(unused self) {}
  }
}

// CHECK:STDERR: generic_type_impls_interface.carbon:[[@LINE+4]]:1: error: semantics TODO: `unsupported: C++ calling a Carbon function with generic parameters` [SemanticsTodo]
// CHECK:STDERR: fn F[T:! I](t: T) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F[T:! I](t: T) {
  t.Doit();
}

inline Cpp '''
void G() {
  Carbon::A a;
  Carbon::B b;
  // CHECK:STDERR: generic_type_impls_interface.carbon:[[@LINE+4]]:11: error: no type named 'F' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    35 |   Carbon::F(a);
  // CHECK:STDERR:       |   ~~~~~~~~^
  // CHECK:STDERR:
  Carbon::F(a);
  // CHECK:STDERR: generic_type_impls_interface.carbon:[[@LINE+4]]:11: error: no type named 'F' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    40 |   Carbon::F(b);
  // CHECK:STDERR:       |   ~~~~~~~~^
  // CHECK:STDERR:
  Carbon::F(b);
}
''';

fn Run() {
  Cpp.G();
}

