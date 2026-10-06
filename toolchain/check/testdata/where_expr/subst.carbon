// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/subst.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/subst.carbon

// --- subst_into_named_constraint.carbon
library "[[@TEST_NAME]]";

interface A { let T: type; }
constraint B {
  require impls A;
  alias U = Self.(A.T);
}

// When type-checking this, we will thaw the `.Self` in the `.U = ()`
// constraint. That performs a substitution of a SpecificConstant that refers
// into the definition region of constraint B's NamedConstraintWithSelf generic.
// For this to work, we need to resolve the definition region of that specific
// as part of the substitution process.
fn F(generic _: B where .U = ()) {}

class C { impl as A where .T = () {} }
fn Test() { F(C); }
