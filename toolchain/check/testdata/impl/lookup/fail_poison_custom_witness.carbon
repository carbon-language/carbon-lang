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

// TODO: uncomment when `require impls` adds a witness table entry.
// interface SubobjectDestroy {
//   final fn Op(ref self) = "subobject.destroy";
// }

interface Destroy {
  // TODO: uncomment when `require impls` adds a witness table entry.
  // require impls SubobjectDestroy;

  // TODO: change `Self` to `partial Self`.
  fn Op(ref self);

  // TODO: remove when `require impls` adds a witness table entry.
  final fn SubobjectDestroy(unused ref self) {
    // This function body is ignored by the toolchain. We can't use
    // a built-in because the compiler treats it as an error. We
    // should address this at some point, but it's not a high
    // priority right now.
  }

  final fn SelfDestruct(ref self) {
    self.Op();
    self.SubobjectDestroy();
  }
}

// TODO: uncomment when `require impls` adds a witness table entry.
// impl forall [T: SubobjectDestroy] partial T as Destroy {
//   alias Op = T.Op;
// }

impl forall [T: type] T as Destroy {
  fn Op(unused ref self) {}
}

// --- fail_poison_custom_witness.carbon

library "[[@TEST_NAME]]";
import Core library "destroy";
import Cpp;

inline Cpp '''
struct X {};

struct A {
  virtual void f(const X& x) = 0;
};
''';

// CHECK:STDERR: fail_poison_custom_witness.carbon:[[@LINE+8]]:1: error: found `impl` for Cpp.A as Destroy, which has a custom witness [PoisonedImplLookupCustomResult]
// CHECK:STDERR: class B {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_poison_custom_witness.carbon:[[@LINE-14]]:1: in import [InImport]
// CHECK:STDERR: destroy.carbon:35:1: note: the use tried to select the `impl` here, but a custom witness was already chosen [PoisonedImplLookupCustomResultNoteBadImpl]
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
