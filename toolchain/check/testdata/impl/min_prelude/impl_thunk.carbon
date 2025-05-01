// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
// EXTRA-ARGS: --no-dump-sem-ir --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/min_prelude/impl_thunk.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/min_prelude/impl_thunk.carbon

// --- convert_in_class.carbon

library "[[@TEST_NAME]]";

interface X(T:! type, U:! type) {
  fn F(t: T) -> U;
}

class ConvertsToA {}
class ConvertsToB {}

// Check that we don't try to define a thunk for `A.B.(as X).F` until we reach
// the end of `A`. If we tried earlier, we wouldn't find a conversion from
// `ConvertsToA` to `A` or from `ConvertsToB` to `B`.
class A {
  class B {
    impl as X(ConvertsToA, B) {
      fn F(a: A) -> ConvertsToB;
    }

    impl ConvertsToB as Core.ImplicitAs(B) {
      fn Convert[self: Self]() -> B { return {}; }
    }
  }

  impl ConvertsToA as Core.ImplicitAs(A) {
    fn Convert[self: Self]() -> A { return {}; }
  }
}
