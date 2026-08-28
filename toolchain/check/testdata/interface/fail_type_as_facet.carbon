// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_type_as_facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_type_as_facet.carbon

interface I {
  let T: type;
  // CHECK:STDERR: fail_type_as_facet.carbon:[[@LINE+3]]:17: error: cannot implicitly convert non-type value of type `Self.(TODO: element 0 in incomplete Self as I)` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR: fail_type_as_facet.carbon:[[@LINE+2]]:17: note: type `Self.(TODO: element 0 in incomplete Self as I)` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  fn F[X: T](x: X);
}
