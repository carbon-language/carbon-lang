// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/struct.carbon

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

// TODO: Are structs with different field orders different types too, for impl
// lookup? Or should this impl be diagnosed as overlapping with the impl on
// `{.a, .b}`? Raised in:
// https://github.com/carbon-language/carbon-lang/issues/5413
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

  // TODO: It is unclear if `F` should return `C` or `D`, since it has the same
  // field names but different order as the impl that would give us `C` here.
  let d2: D = F({.b: (), .a: ()});
}
