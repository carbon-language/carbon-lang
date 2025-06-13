// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  //@dump-sem-ir-begin
  return Mul(a, b);
  //@dump-sem-ir-end
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

library "[[@TEST_NAME]]";

fn Mul(a: i32, b: i32) -> i32 = "int.smul";

let a: i32 = Mul(0x7FFF, 0x10000);
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: error: integer overflow in calculation `32768 * 65536` [CompileTimeIntegerOverflow]
// CHECK:STDERR: let b: i32 = Mul(0x8000, 0x10000);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = Mul(0x8000, 0x10000);

// CHECK:STDOUT: --- i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Mul.type: type = fn_type @Mul [concrete]
// CHECK:STDOUT:   %Mul: %Mul.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mul.ref: %Mul.type = name_ref Mul, file.%Mul.decl [concrete = constants.%Mul]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.smul: init %i32 = call %Mul.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc20_19.1: %i32 = value_of_initializer %int.smul
// CHECK:STDOUT:   %.loc20_19.2: %i32 = converted %int.smul, %.loc20_19.1
// CHECK:STDOUT:   return %.loc20_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
