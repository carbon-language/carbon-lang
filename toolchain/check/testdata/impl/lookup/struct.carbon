// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/struct.carbon

interface Z {
  let X:! type;
  fn ZZ() -> X;
}

class C { adapt (); }
class D { adapt (); }

impl {.a: (), .b: ()} as Z where .X = C {
  fn ZZ() -> C { return () as C; }
}

// A struct with different field names. Structs with different field names are
// different types and can be matched by an impl lookup with matching field
// names.
impl {.aa: (), .bb: ()} as Z where .X = D {
  fn ZZ() -> D { return () as D; }
}

// Structs with different field orders are different types too, for impl
// lookup. See https://github.com/carbon-language/carbon-lang/issues/5413.
impl {.b: (), .a: ()} as Z where .X = D {
  fn ZZ() -> D { return () as D; }
}

fn F(T:! Z) -> T.X {
  return T.ZZ();
}

fn G() {
  // Check which impl is selected for struct literals with different field
  // names.
  let c1: C = F({.a: (), .b: ()});
  let d1: D = F({.aa: (), .bb: ()});
  let d2: D = F({.b: (), .a: ()});
}
