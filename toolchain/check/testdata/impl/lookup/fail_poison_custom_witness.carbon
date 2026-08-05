// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/fail_poison_custom_witness.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/fail_poison_custom_witness.carbon

// --- destroy.carbon

package Core library "destroy";


interface Destroy {
  fn Op(ref self);
}

impl forall [T: type] T as Destroy {
  fn Op(unused ref self) {}
}

// --- const_ref_param.carbon
library "[[@TEST_NAME]]";
import Core library "destroy";
import Cpp;

inline Cpp '''
struct X {};

struct A {
  virtual void f(const X& x) = 0;
};
''';

// CHECK:STDERR: const_ref_param.carbon:[[@LINE+8]]:1: error: found `impl` for Cpp.A as Destroy, which has a custom witness [PoisonedImplLookupCustomResult]
// CHECK:STDERR: class B {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: const_ref_param.carbon:[[@LINE-14]]:1: in import [InImport]
// CHECK:STDERR: destroy.carbon:9:1: note: the use tried to select the `impl` here, but a custom witness was already chosen [PoisonedImplLookupCustomResultNoteBadImpl]
// CHECK:STDERR: impl forall [T: type] T as Destroy {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class B {
  extend base: Cpp.A;
  override fn f(unused self, unused x: Cpp.X) {}
}

inline Cpp '''
Carbon::B b;
''';
