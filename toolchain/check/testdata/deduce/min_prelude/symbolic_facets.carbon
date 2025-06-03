// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/min_prelude/symbolic_facets.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/min_prelude/symbolic_facets.carbon

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

fn H(T:! C({}).A & D({}).B) {
  D({}).G(T);
}

// --- fail_interface_wrong_generic_param_using_generic_arg.carbon
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
    // CHECK:STDERR: fail_interface_wrong_generic_param_using_generic_arg.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `A & B` into type implementing `A` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     C(DD).F(T);
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_interface_wrong_generic_param_using_generic_arg.carbon:[[@LINE-12]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(T:! A) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~
    // CHECK:STDERR:
    C(DD).F(T);
  }
}

fn H(T:! C({}).A & D({}).B) {
  D({}).G(T);
}

// --- fail_interface_wrong_generic_param_with_rewrite.carbon
library "[[@TEST_NAME]]";

class C(CC:! type) {
  interface A { let X:! type; }
  fn F(T:! A where .X = CC) {}
}

class D(DD:! type) {
  interface B {}
  fn G(T:! B & C({}).A where .X = {}) {
    // T implements C({}).A and D(DD).B, so should not convert to a facet value
    // implementing C(()).A.
    //
    // CHECK:STDERR: fail_interface_wrong_generic_param_with_rewrite.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `A & B where .(A.X) = {}` into type implementing `A where .(A.X) = ()` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     C(()).F(T);
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_interface_wrong_generic_param_with_rewrite.carbon:[[@LINE-12]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(T:! A where .X = CC) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    C(()).F(T);
  }
}

fn H(T:! C({}).A & D({}).B where .X = {}) {
  D({}).G(T);
}

// --- fail_interface_wrong_generic_param_using_generic_arg_with_rewrite.carbon
library "[[@TEST_NAME]]";

class C(CC:! type) {
  interface A { let X:! type; }
  fn F(T:! A where .X = CC) {}
}

class D(DD:! type) {
  interface B {}
  fn G(T:! B & C({}).A where .X = {}) {
    // T implements C({}).A and D(DD).B, so should not convert to a facet value
    // implementing C(()).A.
    //
    // CHECK:STDERR: fail_interface_wrong_generic_param_using_generic_arg_with_rewrite.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `A & B where .(A.X) = {}` into type implementing `A where .(A.X) = DD` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     C(DD).F(T);
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_interface_wrong_generic_param_using_generic_arg_with_rewrite.carbon:[[@LINE-12]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(T:! A where .X = CC) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    C(DD).F(T);
  }
}

fn H(T:! C({}).A & D({}).B where .X = {}) {
  D({}).G(T);
}

// --- compatible_deduce.carbon
library "[[@TEST_NAME]]";

class C(CC:! type) {
  interface A { let X:! type; }
  fn F(T:! A where .X = CC) {}
}

class D(DD:! type) {
  interface B {}
  fn G(T:! B & C(()).A where .X = ()) {
    C(()).F(T);
  }
}

fn H(T:! C(()).A & D({}).B where .X = ()) {
  D({}).G(T);
}
