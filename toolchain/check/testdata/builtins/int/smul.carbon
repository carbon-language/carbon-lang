// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/smul.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/smul.carbon

// --- i32.carbon

library "[[@TEST_NAME]]";

fn Mul(a: i32, b: i32) -> i32 = "int.smul";

class Expect(N:! i32) {}
fn Test(N:! i32) -> Expect(N) { return {}; }

fn F() {
  Test(Mul(0, 0)) as Expect(0);
  Test(Mul(0, 3)) as Expect(0);
  Test(Mul(1, 2)) as Expect(2);
  Test(Mul(3, 2)) as Expect(6);
  Test(Mul(0x7FFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  Test(Mul(0x7FFF_FFFF, -1)) as Expect(-0x7FFF_FFFF);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return Mul(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Mul(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.smul";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Mul(0, 0)) as Expect(0);
  Test(Mul(0, 3)) as Expect(0);
  Test(Mul(1, 2)) as Expect(2);
  Test(Mul(3, 2)) as Expect(6);
  Test(Mul(0x7FFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  Test(Mul(0x7FFF_FFFF, -1)) as Expect(-0x7FFF_FFFF);

  // Test some cases that might -- but shouldn't -- overflow.
  Test(Mul(0x8000_0000_0000_0000, 1)) as Expect(0x8000_0000_0000_0000);
  Test(Mul(0x8000_0000_0000_0000, -1)) as Expect(-0x8000_0000_0000_0000);
  Test(Mul(-0x8000_0000_0000_0000, 1)) as Expect(-0x8000_0000_0000_0000);
  Test(Mul(-0x8000_0000_0000_0000, -1)) as Expect(0x8000_0000_0000_0000);
  Test(Mul(-0x8000_0000_0000_0000, -1)) as Expect(0x8000_0000_0000_0000);
  Test(Mul(-0x8000_0000_0000_0000, -0x8000_0000_0000_0000))
    as Expect(0x4000_0000_0000_0000_0000_0000_0000_0000);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Mul(a: i32, b: i32) -> i32 = "int.smul";

let a: i32 = Mul(0x7FFF, 0x10000);
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: error: integer overflow in calculation `32768 * 65536` [CompileTimeIntegerOverflow]
// CHECK:STDERR: let b: i32 = Mul(0x8000, 0x10000);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = Mul(0x8000, 0x10000);
