// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/use_impl_from_definition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/use_impl_from_definition.carbon

// --- construct_facet_value_inside_definition.carbon
library "[[@TEST_NAME]]";

interface I {
  let U:! type;
  fn F() -> U;
}

class C { adapt {}; }

impl forall [T:! type] T as I where .U = C {
  // `T as I` does an impl lookup to form a FacetValue, which must find the impl
  // being defined and use that. The witness must be a final witness in order to
  // know that `U` is `C`.
  fn F() -> (T as I).U { return {} as C; }
}

// --- construct_facet_value_from_self_inside_definition.carbon
library "[[@TEST_NAME]]";

interface I {
  let U:! type;
  fn F() -> U;
}

class C { adapt {}; }

impl forall [T:! type] T as I where .U = C {
  // `Self as I` is a symbolic facet value specified in the interface-with-self
  // specific, and then monomorphized with the self type of the impl.
  fn F() -> (Self as I).U { return {} as C; }
}
