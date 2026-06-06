// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/operators/builtin_operators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/operators/builtin_operators.carbon

library "builtin_operators";

import Cpp inline '''c++
struct ConvertibleToInt {
  operator int() const;
};
struct ConvertibleToFloat {
  operator float() const;
};
''';

fn TestInteger(a: Cpp.ConvertibleToInt, b: Cpp.ConvertibleToInt) {
  // Test arithmetic.
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `add` unused [UnusedBinding]
  // CHECK:STDERR:   let add: i32 = a + b;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let add: i32 = a + b;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `sub` unused [UnusedBinding]
  // CHECK:STDERR:   let sub: i32 = a - b;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let sub: i32 = a - b;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `mul` unused [UnusedBinding]
  // CHECK:STDERR:   let mul: i32 = a * b;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let mul: i32 = a * b;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `div` unused [UnusedBinding]
  // CHECK:STDERR:   let div: i32 = a / b;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let div: i32 = a / b;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `mod` unused [UnusedBinding]
  // CHECK:STDERR:   let mod: i32 = a % b;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let mod: i32 = a % b;

  // Test bitwise.
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `and_val` unused [UnusedBinding]
  // CHECK:STDERR:   let and_val: i32 = a & b;
  // CHECK:STDERR:       ^~~~~~~~~~~~
  // CHECK:STDERR:
  let and_val: i32 = a & b;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `or_val` unused [UnusedBinding]
  // CHECK:STDERR:   let or_val: i32 = a | b;
  // CHECK:STDERR:       ^~~~~~~~~~~
  // CHECK:STDERR:
  let or_val: i32 = a | b;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `xor_val` unused [UnusedBinding]
  // CHECK:STDERR:   let xor_val: i32 = a ^ b;
  // CHECK:STDERR:       ^~~~~~~~~~~~
  // CHECK:STDERR:
  let xor_val: i32 = a ^ b;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `shl` unused [UnusedBinding]
  // CHECK:STDERR:   let shl: i32 = a << b;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let shl: i32 = a << b;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `shr` unused [UnusedBinding]
  // CHECK:STDERR:   let shr: i32 = a >> b;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let shr: i32 = a >> b;

  // Test comparison.
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `eq` unused [UnusedBinding]
  // CHECK:STDERR:   let eq: bool = (a == b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let eq: bool = (a == b);
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `ne` unused [UnusedBinding]
  // CHECK:STDERR:   let ne: bool = (a != b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let ne: bool = (a != b);
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `lt` unused [UnusedBinding]
  // CHECK:STDERR:   let lt: bool = (a < b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let lt: bool = (a < b);
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `le` unused [UnusedBinding]
  // CHECK:STDERR:   let le: bool = (a <= b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let le: bool = (a <= b);
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `gt` unused [UnusedBinding]
  // CHECK:STDERR:   let gt: bool = (a > b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let gt: bool = (a > b);
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `ge` unused [UnusedBinding]
  // CHECK:STDERR:   let ge: bool = (a >= b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let ge: bool = (a >= b);

  // Test unary.
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `neg` unused [UnusedBinding]
  // CHECK:STDERR:   let neg: i32 = -a;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let neg: i32 = -a;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `comp` unused [UnusedBinding]
  // CHECK:STDERR:   let comp: i32 = ^a;
  // CHECK:STDERR:       ^~~~~~~~~
  // CHECK:STDERR:
  let comp: i32 = ^a;
}

fn TestFloat(a: Cpp.ConvertibleToFloat, b: Cpp.ConvertibleToFloat) {
  // Test arithmetic.
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `add` unused [UnusedBinding]
  // CHECK:STDERR:   let add: f32 = a + b;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let add: f32 = a + b;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `sub` unused [UnusedBinding]
  // CHECK:STDERR:   let sub: f32 = a - b;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let sub: f32 = a - b;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `mul` unused [UnusedBinding]
  // CHECK:STDERR:   let mul: f32 = a * b;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let mul: f32 = a * b;
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `div` unused [UnusedBinding]
  // CHECK:STDERR:   let div: f32 = a / b;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let div: f32 = a / b;

  // Test comparison.
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `eq` unused [UnusedBinding]
  // CHECK:STDERR:   let eq: bool = (a == b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let eq: bool = (a == b);
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `ne` unused [UnusedBinding]
  // CHECK:STDERR:   let ne: bool = (a != b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let ne: bool = (a != b);
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `lt` unused [UnusedBinding]
  // CHECK:STDERR:   let lt: bool = (a < b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let lt: bool = (a < b);
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `le` unused [UnusedBinding]
  // CHECK:STDERR:   let le: bool = (a <= b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let le: bool = (a <= b);
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `gt` unused [UnusedBinding]
  // CHECK:STDERR:   let gt: bool = (a > b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let gt: bool = (a > b);
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `ge` unused [UnusedBinding]
  // CHECK:STDERR:   let ge: bool = (a >= b);
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let ge: bool = (a >= b);

  // Test unary.
  // CHECK:STDERR: builtin_operators.carbon:[[@LINE+4]]:7: warning: binding `neg` unused [UnusedBinding]
  // CHECK:STDERR:   let neg: f32 = -a;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let neg: f32 = -a;
}
