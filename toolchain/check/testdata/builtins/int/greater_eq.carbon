// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/greater_eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/greater_eq.carbon

// --- int_greater_eq.carbon

library "[[@TEST_NAME]]";

fn GreaterEq(a: i32, b: i32) -> bool = "int.greater_eq";
fn Negate(a: i32) -> i32 = "int.snegate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  false_ as (if GreaterEq(1, 2) then True else False);
  true_ as (if GreaterEq(1, 1) then True else False);
  true_ as (if GreaterEq(1, 0) then True else False);
  false_ as (if GreaterEq(Negate(1), 0) then True else False);
  true_ as (if GreaterEq(0, Negate(1)) then True else False);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> bool {
  return GreaterEq(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn GreaterEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.greater_eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(GreaterEq(5, 5)) as Expect(true);
  Test(GreaterEq(5, 6)) as Expect(false);
  Test(GreaterEq(6, 5)) as Expect(true);
  Test(GreaterEq(-1, -1)) as Expect(true);
  Test(GreaterEq(-1, 1)) as Expect(false);
  Test(GreaterEq(1, -1)) as Expect(true);
}

// --- mixed.carbon

library "[[@TEST_NAME]]";

fn GreaterEq(a: Core.IntLiteral(), b: i32) -> bool = "int.greater_eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(GreaterEq(5, 5)) as Expect(true);
  Test(GreaterEq(5, 6)) as Expect(false);
  Test(GreaterEq(6, 5)) as Expect(true);
  Test(GreaterEq(-1, -1)) as Expect(true);
  Test(GreaterEq(-1, 1)) as Expect(false);
  Test(GreaterEq(1, -1)) as Expect(true);
}
