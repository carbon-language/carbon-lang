// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/symbolic_facets.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/symbolic_facets.carbon

// By placing each interface inside a generic class, a facet type refering to
// the interface becomes symbolic. Normally they would be concrete. This can
// affect decisions in deduce which unwraps symbolic values, but still needs to
// ensure they convert correctly.

// --- fail_missing_interface.carbon
library "[[@TEST_NAME]]";

class C(CC:! type) {
  interface A {}
  fn F(T:! A) {}
}

class D(DD:! type) {
  interface B {}
  fn G(T:! B) {
    // T only implements D(DD).B, so should not convert to a facet value
    // implementing C(()).A.
    //
    // CHECK:STDERR: fail_missing_interface.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `B` into type implementing `A` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     C(()).F(T);
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_missing_interface.carbon:[[@LINE-12]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(T:! A) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~
    // CHECK:STDERR:
    C(()).F(T);
  }
}

// --- fail_interface_wrong_generic_param.carbon
library "[[@TEST_NAME]]";

class C(CC:! type) {
  interface A {}
  fn F(T:! A) {}
}

class D(DD:! type) {
  interface B {}
  fn G(T:! B & C({}).A) {
    // T implements C({}).A and D(DD).B, so should not convert to a facet value
    // implementing C(()).A.
    //
    // CHECK:STDERR: fail_interface_wrong_generic_param.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `A & B` into type implementing `A` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     C(()).F(T);
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_interface_wrong_generic_param.carbon:[[@LINE-12]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(T:! A) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~
    // CHECK:STDERR:
    C(()).F(T);
  }
}

// --- compatible_deduce.carbon
library "[[@TEST_NAME]]";

class C(CC:! type) {
  interface A {}
  fn F(T:! A) {}
}

class D(DD:! type) {
  interface B {}
  fn G(T:! B & C(()).A) {
    C(()).F(T);
  }
}
