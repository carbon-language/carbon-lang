// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/make_type_signed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/make_type_signed.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(n: IntLiteral()) -> type = "int.make_type_signed";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(n: Int(64)) -> Int(64) {
  return n;
}

fn G(n: Int(13)) -> Int(13) {
  return n;
}

fn Symbolic(N:! IntLiteral(), x: Int(N)) -> Int(N) {
  return x;
}

// --- import_types.carbon

library "[[@TEST_NAME]]";

import library "types";
import library "use_types";

fn UseF(n: Int(64)) -> Int(64) {
  return F(n);
}

fn UseG(n: Int(13)) -> Int(13) {
  return G(n);
}

fn UseSymbolic(n: Int(24)) -> Int(24) {
  return Symbolic(24, n);
}

// --- fail_zero_size.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_zero_size.carbon:[[@LINE+4]]:8: error: integer type width of 0 is not positive [IntWidthNotPositive]
// CHECK:STDERR: var n: Int(0);
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR:
var n: Int(0);

// --- fail_negative_size.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Negate(n: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_negative_size.carbon:[[@LINE+4]]:8: error: integer type width of -1 is not positive [IntWidthNotPositive]
// CHECK:STDERR: var n: Int(Negate(1));
// CHECK:STDERR:        ^~~~~~~~~~~~~~
// CHECK:STDERR:
var n: Int(Negate(1));

// --- fail_oversized.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_oversized.carbon:[[@LINE+4]]:8: error: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR: var m: Int(1000000000);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~
// CHECK:STDERR:
var m: Int(1000000000);
