// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float_literal/mul.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float_literal/mul.carbon

// --- float_literal_mul.carbon

library "[[@TEST_NAME]]";

fn Mul(a: Core.FloatLiteral, b: Core.FloatLiteral) -> Core.FloatLiteral = "float.mul";

fn Negate(a: Core.FloatLiteral) -> Core.FloatLiteral = "float.negate";

// Since FloatLiterals are not canonical we need to compare using sized float.
class Expect(N: f128) {}
fn Test(generic N: Core.FloatLiteral) -> Expect(N) { return {}; }

fn F() {
    Test(Mul(2.0, 3.0)) as Expect(6.0);
    Test(Mul(0x2.0p0, 1.5)) as Expect(3.0);
    Test(Mul(0x1.8p0, 2.0)) as Expect(3.0);
    Test(Mul(2.0, 0x1.8p0)) as Expect(3.0);
    Test(Mul(Negate(2.0), 3.0)) as Expect(Negate(6.0));
    Test(Mul(1.5, 1.5)) as Expect(2.25);
    Test(Mul(0.1, 0.1)) as Expect(0.01);
    Test(Mul(1.0e20, 1.0e-20)) as Expect(1.0);
    Test(Mul(1.0e2000000000, 1.0e-2000000000)) as Expect(1.0);
    Test(Mul(1.0e39999999999999999993, 1.0e-39999999999999999993)) as Expect(1.0);
    Test(Mul(0x1.0p4, 0x1.0p-4)) as Expect(1.0);
    Test(Mul(0.0, Negate(3.0))) as Expect(0.0);
    Test(Mul(1.0, 1.000000000000000000000000000000)) as Expect(1.0);
}

// --- fail_too_large.carbon

library "[[@TEST_NAME]]";

fn Mul(a: Core.FloatLiteral, b: Core.FloatLiteral) -> Core.FloatLiteral = "float.mul";

fn F(){
     // CHECK:STDERR: fail_too_large.carbon:[[@LINE+4]]:6: error: binary calculation `10*10^1999999999 * 10*10^1999999999` would exceed the maximum supported integer width of 8388608 [CompileTimeFloatLiteralBinaryOperationOutOfRange]
     // CHECK:STDERR:      Mul(1.0e2000000000, 1.0e2000000000);
     // CHECK:STDERR:      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     // CHECK:STDERR:
     Mul(1.0e2000000000, 1.0e2000000000);
     // CHECK:STDERR: fail_too_large.carbon:[[@LINE+4]]:6: error: binary calculation `10*10^9223372036854775806 * 10*10^9223372036854775806` would exceed the maximum supported integer width of 8388608 [CompileTimeFloatLiteralBinaryOperationOutOfRange]
     // CHECK:STDERR:      Mul(1.0e9223372036854775807, 1.0e9223372036854775807);
     // CHECK:STDERR:      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     // CHECK:STDERR:
     Mul(1.0e9223372036854775807, 1.0e9223372036854775807);
     // CHECK:STDERR: fail_too_large.carbon:[[@LINE+4]]:6: error: binary calculation `10*10^39999999999999999992 * 10*10^-1` would exceed the maximum supported integer width of 8388608 [CompileTimeFloatLiteralBinaryOperationOutOfRange]
     // CHECK:STDERR:      Mul(1.0e39999999999999999993, 1.0);
     // CHECK:STDERR:      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     // CHECK:STDERR:
     Mul(1.0e39999999999999999993, 1.0);
     // CHECK:STDERR: fail_too_large.carbon:[[@LINE+4]]:6: error: binary calculation `16*2^-2147483648 * 10*10^-1` would exceed the maximum supported integer width of 8388608 [CompileTimeFloatLiteralBinaryOperationOutOfRange]
     // CHECK:STDERR:      Mul(0x1.0p-2147483644, 1.0);
     // CHECK:STDERR:      ^~~~~~~~~~~~~~~~~~~~~~~~~~~
     // CHECK:STDERR:
     Mul(0x1.0p-2147483644, 1.0);
}
