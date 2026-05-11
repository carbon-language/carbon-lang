// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/access.carbon

// --- impl_witness_access_in_impl_type_structure.carbon
library "[[@TEST_NAME]]";

interface X(T:! type) {}
interface Y {
  let Y1:! type;
}
interface Z {
  let Z1:! type;
}

// There's 2 ImplWitnessAccess instructions in the type, but they together
// resolve to a symbolic type value, so the type structure is: `?C(?)`
impl forall [U:! Z where .Z1 impls Y] U as X(U.Z1.(Y.Y1)) {}

class C {
  impl as Y where .Y1 = {} {}
}

fn F(V:! Z where .Z1 = C) {
  // The type stucture is `?C(C)` which will match the impl's less specific
  // `?C(?)`, then the impl will deduce the parameter of `X` to be `{}` from the
  // type of `V`. This would fail if ImplWitnessAccess instructions were treated
  // as Concrete in the type structure, since the impl would have a different
  // concrete value (an ImplWitnessAccess) than the query (a StructValue).
  V as X({});
}

fn G(V:! Z where .Z1 impls Y) {
  // The type structure is `?C(?)`, also built from ImplWitnessAccess insts,
  // which will match the impl's `?C(?)`.
  V as X(V.(Z.Z1).(Y.Y1));
}

// TODO: This creates an infinite loop in `.Self` substitution.
// fn H(U:! type, V:! Z where .Z1 impls (Y where .Y1 = U)) {
//   // The type structure is `?C(?)`, without using an ImplWitnessAccess, which
//   // will match the impl's `?C(?)`. This would fail if ImplWitnessAccess
//   // instructions were treated as Concrete in the type structure, since the
//   // impl's type structure would be more specific than the query's.
//   V as X(U);
// }
