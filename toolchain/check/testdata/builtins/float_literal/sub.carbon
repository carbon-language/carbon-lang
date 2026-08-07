// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float_literal/sub.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float_literal/sub.carbon

// --- float_literal_sub.carbon

library "[[@TEST_NAME]]";

fn Negate(a: Core.FloatLiteral) -> Core.FloatLiteral = "float.negate";
fn Sub(a: Core.FloatLiteral, b: Core.FloatLiteral) -> Core.FloatLiteral = "float.sub";

// Since FloatLiterals are not canonical we need to compare using sized float.
class Expect(N: f128) {}
fn Test(generic N: Core.FloatLiteral) -> Expect(N) { return {}; }

fn F() {
    Test(Sub(2.0, 1.0)) as Expect(1.0);
    Test(Sub(0x1.8p0, 2.5)) as Expect(Negate(1.0));
    Test(Sub(2.5, 0x1.8p0)) as Expect(1.0);
    Test(Sub(1.0e20, 1.0e-20)) as Expect(0.9999999999999999999999999999999999999999e20);
    Test(Sub(1.0e-20, 1.0e20)) as Expect(Negate(0.9999999999999999999999999999999999999999e20));
    Test(Sub(1.0, 1.000000000000000000000000000001)) as Expect(Negate(0.000000000000000000000000000001));
    Test(Sub(1.0000000000000000000000000000001, 1.0)) as Expect(0.0000000000000000000000000000001);
}

// --- fail_too_large.carbon

library "[[@TEST_NAME]]";

fn Sub(a: Core.FloatLiteral, b: Core.FloatLiteral) -> Core.FloatLiteral = "float.sub";

// CHECK:STDERR: fail_too_large.carbon:[[@LINE+4]]:29: error: binary calculation `16*2^1023999999996 - 1*10^-19` would exceed the maximum supported integer width of 8388608 [CompileTimeFloatLiteralBinaryOperationOutOfRange]
// CHECK:STDERR: let v1: Core.FloatLiteral = Sub(0x1.0p1024000000000, 0.0000000000000000001);
// CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let v1: Core.FloatLiteral = Sub(0x1.0p1024000000000, 0.0000000000000000001);
