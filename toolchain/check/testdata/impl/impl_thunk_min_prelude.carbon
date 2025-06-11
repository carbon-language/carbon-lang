// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/impl_thunk_min_prelude.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/impl_thunk_min_prelude.carbon

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

// --- fail_todo_out_of_line_thunk.carbon

library "[[@TEST_NAME]]";

class Wrap(T:! type) {}

interface OpWith(U:! type) {
  fn Op[self: Self](u: U);
}

impl forall [T:! type, U:! Core.ImplicitAs(Wrap(T))] Wrap(T) as OpWith(U) {
  // CHECK:STDERR: fail_todo_out_of_line_thunk.carbon:[[@LINE+7]]:3: error: use of undefined generic function [MissingGenericFunctionDefinition]
  // CHECK:STDERR:   fn Op[self: Self](other: Self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_out_of_line_thunk.carbon:[[@LINE+4]]:3: note: generic function declared here [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR:   fn Op[self: Self](other: Self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn Op[self: Self](other: Self);
}

// TODO: Once we support the syntax for defining impl members out of line,
// define the above function here.
// fn (forall [T:! type, U:! Core.ImplicitAs(Wrap(T))] Wrap(T) as OpWith(U)).Op[self: Self](other: Self) {}

// --- builtin_thunk.carbon

library "[[@TEST_NAME]]";

class Wrap(T:! type) {}

interface OpWith(U:! type) {
  fn Op[self: Self](u: U);
}

impl forall [T:! type, U:! Core.ImplicitAs(Wrap(T))] Wrap(T) as OpWith(U) {
  fn Op[self: Self](other: Self) = "no_op";
}
