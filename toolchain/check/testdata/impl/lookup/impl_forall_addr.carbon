// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/impl_forall_addr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/impl_forall_addr.carbon

// --- impl_forall.carbon
library "[[@TEST_NAME]]";

class A(T:! type) {
  var n: T;
}

interface I(U:! type) {
  // CHECK:STDERR: impl_forall.carbon:[[@LINE+8]]:13: error: expected `:` or `:!` in binding pattern [ExpectedBindingPattern]
  // CHECK:STDERR:   fn F[addr self: Self*]() -> U*;
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  // CHECK:STDERR: impl_forall.carbon:[[@LINE+4]]:13: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   fn F[addr self: Self*]() -> U*;
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  fn F[addr self: Self*]() -> U*;
}

//@dump-sem-ir-begin
impl forall [V:! type] A(V) as I(V) {
  // CHECK:STDERR: impl_forall.carbon:[[@LINE+4]]:13: error: expected `:` or `:!` in binding pattern [ExpectedBindingPattern]
  // CHECK:STDERR:   fn F[addr self: Self*]() -> V* {
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  fn F[addr self: Self*]() -> V* {
    return &self->n;
  }
}
//@dump-sem-ir-end

fn TestGeneric[W:! type](a: A(W)*) -> W* {
  //@dump-sem-ir-begin
  return a->(I(W).F)();
  //@dump-sem-ir-end
}

fn TestSpecific(a: A({})*) -> {}* {
  //@dump-sem-ir-begin
  return a->(I({}).F)();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
