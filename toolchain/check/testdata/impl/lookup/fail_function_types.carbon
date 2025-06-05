// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/fail_function_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/fail_function_types.carbon

// --- fail_function_type_with_self_type.carbon
library "[[@TEST_NAME]]";

interface Z {
  fn G();
}

impl forall [T:! type] T as Z {
  fn G() {}
}

fn F() {
  // CHECK:STDERR: fail_function_type_with_self_type.carbon:[[@LINE+11]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &(().(Z.G)) as type;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_function_type_with_self_type.carbon:[[@LINE+7]]:3: error: cannot convert non-type value of type `<type of G in () as Z>*` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   &(().(Z.G)) as type;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_function_type_with_self_type.carbon:[[@LINE+4]]:3: note: type `<type of G in () as Z>*` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   &(().(Z.G)) as type;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  &(().(Z.G)) as type;
}

// --- fail_function_type.carbon
library "[[@TEST_NAME]]";

fn F() {
  // This does an impl lookup for `As` with `FunctionType` in the self type.
  //
  // CHECK:STDERR: fail_function_type.carbon:[[@LINE+11]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &F as type;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_function_type.carbon:[[@LINE+7]]:3: error: cannot convert non-type value of type `<type of F>*` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   &F as type;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_function_type.carbon:[[@LINE+4]]:3: note: type `<type of F>*` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   &F as type;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  &F as type;
}
