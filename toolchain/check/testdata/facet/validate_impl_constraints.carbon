// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/validate_impl_constraints.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/validate_impl_constraints.carbon

// --- fail_todo_where_impls_tests_associated_constant_of_generic_type.carbon

class C(U:! type) {}

// C(U) impls M if U impls L.
interface L {}
interface M {}
impl forall [U:! L] C(U) as M {}

// U requires that C(.Self) impls M.
// - C(.Self) impls M can be rewritten as C(U) impls M.
// - C(U) impls M if U impls L => Requires U impls L.
fn F(U:! type where C(.Self) impls M) {}

fn G(T:! L) {
  // CHECK:STDERR: fail_todo_where_impls_tests_associated_constant_of_generic_type.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `L` into type implementing `type where...` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_todo_where_impls_tests_associated_constant_of_generic_type.carbon:[[@LINE-6]]:6: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(U:! type where C(.Self) impls M) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(T);
}
