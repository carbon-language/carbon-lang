// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/fail_todo_require_impls_for_impl_in_class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/fail_todo_require_impls_for_impl_in_class.carbon

// --- destroy.carbon

package Core library "prelude/destroy";

interface SubobjectDestroy {
  fn Op(ref self);
}

interface DestroyImpl {
  require impls SubobjectDestroy;

  fn Op(ref self);
}

impl forall [T: SubobjectDestroy] T as DestroyImpl {
  fn Op(ref self) = "no_op";
}

// --- fail_todo_require_impls_for_impl_in_class.carbon

import Core library "prelude/destroy";

class NonTrivialDestroy {
  // CHECK:STDERR: fail_todo_require_impls_for_impl_in_class.carbon:[[@LINE+4]]:3: error: interface `DestroyImpl` being implemented requires that `NonTrivialDestroy` implements `Core.SubobjectDestroy` [InterfaceRequireImplsNotImplemented]
  // CHECK:STDERR:   impl as Core.DestroyImpl {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}
