// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/observe.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/observe.carbon

// --- base.carbon

library "[[@TEST_NAME]]";

interface P { fn InP[self: Self](); }
interface Q { fn InQ[self: Self](); }
interface R { fn InR[self: Self](); }

interface Transitive {
  let A:! P;
  let B:! Q where .Self == A;
  let C:! R where .Self == B;

  fn GetA[self: Self]() -> A;
  fn TakesC[self: Self](c: C);

  observe A == B == C;
}

fn F[T:! Transitive](t: T) {
  var a: T.A = t.GetA();

  // CHECK:STDERR: base.carbon:[[@LINE+7]]:3: error: cannot convert type `T.(Transitive.A)` that implements `P` into type implementing `R where...` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   t.TakesC(a);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: base.carbon:[[@LINE-11]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:   fn TakesC[self: Self](c: C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  t.TakesC(a);

  // CHECK:STDERR: base.carbon:[[@LINE+8]]:4: error: `Core.As` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   (a as T.C).InR();
  // CHECK:STDERR:    ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: base.carbon:[[@LINE-15]]:3: error: `Core.Destroy` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   var a: T.A = t.GetA();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  (a as T.C).InR();
}

// --- impls.carbon

library "[[@TEST_NAME]]";

interface A { }
interface B { require impls A; }
interface C { require impls B; }
interface D { require impls C; }

fn RequiresA[T:! A](x: T);
fn RequiresC[T:! C](x: T);
fn RequiresD[T:! D](x: T) {
  // CHECK:STDERR: impls.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `D` into type implementing `C` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresC(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: impls.carbon:[[@LINE-5]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresC[T:! C](x: T);
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresC(x);

  // CHECK:STDERR: impls.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `D` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: impls.carbon:[[@LINE-15]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T:! A](x: T);
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresA(x);

  observe T impls C;

  observe T impls B;

  // CHECK:STDERR: impls.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `D` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: impls.carbon:[[@LINE-28]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T:! A](x: T);
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresA(x);
}
