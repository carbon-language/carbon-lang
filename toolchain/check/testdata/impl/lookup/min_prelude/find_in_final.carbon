// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
// EXTRA-ARGS: --no-dump-sem-ir --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/find_in_final.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/find_in_final.carbon

// --- fail_todo_keep_looking_for_final_impl.carbon
library "[[@TEST_NAME]]";
interface I {
  let T:! type;
}

final impl forall [U:! type] U as I where .T = () {}

fn F(V:! I) -> V.T {
  // TODO: Even though we have a witness that `V impls I` from the constraint
  // on `I`, we should do an impl lookup to see if any effectively final impl
  // applies when we find an unknown value in that witness. In this case, that
  // lookup would find an impl with more specific values for associated
  // constants that we should merge.

  // CHECK:STDERR: fail_todo_keep_looking_for_final_impl.carbon:[[@LINE+7]]:3: error: member access into undefined facet type `Core.ImplicitAs(V.(I.T))` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return ();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: include_files/convert.carbon:12:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return ();
}
