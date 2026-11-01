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

interface SubobjectDestroy {
  fn Op(ref self);
}

impl forall [T: type] T as SubobjectDestroy {
  fn Op(unused ref self) {}
}

// TODO: rename to `Destroy` when `require impls` adds a witness table entry.
interface DestroyImpl {
  // TODO: uncomment when `require impls` adds a witness table entry.
  // require impls SubobjectDestroy;

  // TODO: change `Self` to `partial Self`.
  fn Op(ref self);
}

impl forall [T: SubobjectDestroy] T as DestroyImpl {
  fn Op(ref self) = "no_op";
}

// TODO: uncomment when `require impls` adds a witness table entry.
// impl forall [T: SubobjectDestroy] partial T as Destroy {
//   alias Op = T.Op;
// }

fn SelfDestruct[T: DestroyImpl](unused ref t: T) {}

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

// CHECK:STDERR: fail_poison_custom_witness.carbon:[[@LINE+3]]:1-9: error: found `impl` for Cpp.A as SubobjectDestroy, which has a custom witness [PoisonedImplLookupCustomResult]
// CHECK:STDERR: destroy.carbon:8:1-45: note: the use tried to select the `impl` here, but a custom witness was already chosen [PoisonedImplLookupCustomResultNoteBadImpl]
// CHECK:STDERR:
class B {
  extend base: Cpp.A;
  override fn f(unused self, unused x: Cpp.X) {}
}

inline Cpp '''
Carbon::B b;
''';
