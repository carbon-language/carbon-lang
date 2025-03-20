// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --custom-core --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_to_facet_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_to_facet_value.carbon

// --- convert_concrete_facet_value_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface Y {}

class C {}
impl C as Z {}
impl C as Y {}

fn G(z:! Z) {}

fn F() {
  G(C);
  G(C as Y);
  G(C as Z);
  G(C as (Y & Z));

  G(((((C as Y) as type) as Y) as type) as Y);
  G((((((C as Y) as type) as Y) as type) as Y) as type);
}

// --- convert_symbolic_facet_value_to_facet_value.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {}
impl forall [T:! Y] T as Z {}

fn F(U:! Y) {
  U as Z;
}
