// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/sadd.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/sadd.carbon

// --- i32.carbon

library "[[@TEST_NAME]]";

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

class Expect(N:! i32) {}
fn Test(N:! i32) -> Expect(N) { return {}; }

fn F() {
  Test(Add(0, 0)) as Expect(0);
  Test(Add(1, 2)) as Expect(3);
  Test(Add(0x7FFF_FFFE, 1)) as Expect(0x7FFF_FFFF);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  //@dump-sem-ir-begin
  return Add(a, b);
  //@dump-sem-ir-end
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Add(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.sadd";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Add(0, 0)) as Expect(0);
  Test(Add(1, 2)) as Expect(3);

  // Test some cases that might -- but shouldn't -- overflow.
  Test(Add(0x7FFF_FFFE, 1)) as Expect(0x7FFF_FFFF);
  Test(Add(0x7FFF_FFFF, 1)) as Expect(0x8000_0000);
  Test(Add(0x7FFF_FFFF_FFFF_FFFF, 1)) as Expect(0x8000_0000_0000_0000);
  Test(Add(0xFFFF_FFFF_FFFF_FFFF, 1)) as Expect(0x1_0000_0000_0000_0000);
  Test(Add(-0x8000_0000_0000_0000, -1)) as Expect(-0x8000_0000_0000_0001);
  Test(Add(-0x8000_0000_0000_0000, -0x8000_0000_0000_0000)) as Expect(-0x1_0000_0000_0000_0000);
}

// --- fail_too_few.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew(a: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: i32) -> i32 = "int.sadd";

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+4]]:26: error: name `TooMany` not found [NameNotFound]
// CHECK:STDERR: var too_many: array(i32, TooMany(1, 2, 3));
// CHECK:STDERR:                          ^~~~~~~
// CHECK:STDERR:
var too_many: array(i32, TooMany(1, 2, 3));

fn RuntimeCallIsValidTooFew(a: i32) -> i32 {
  return TooFew(a);
}

// --- fail_too_many.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.sadd";

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+4]]:33: error: name `BadReturnType` not found [NameNotFound]
// CHECK:STDERR: var bad_return_type: array(i32, BadReturnType(1, 2));
// CHECK:STDERR:                                 ^~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: array(i32, BadReturnType(1, 2));

fn RuntimeCallIsValidTooMany(a: i32, b: i32, c: i32) -> i32 {
  return TooMany(a, b, c);
}

// --- fail_bad_return_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_return_type.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: i32, b: i32) -> bool = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32, b: i32) -> bool = "int.sadd";

// CHECK:STDERR: fail_bad_return_type.carbon:[[@LINE+4]]:25: error: name `TooFew` not found [NameNotFound]
// CHECK:STDERR: var too_few: array(i32, TooFew(1));
// CHECK:STDERR:                         ^~~~~~
// CHECK:STDERR:
var too_few: array(i32, TooFew(1));

fn RuntimeCallIsValidBadReturnType(a: i32, b: i32) -> bool {
  return BadReturnType(a, b);
}

// --- fail_bad_call.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_call.carbon:[[@LINE+4]]:26: error: name `JustRight` not found [NameNotFound]
// CHECK:STDERR: var bad_call: array(i32, JustRight(1, 2, 3));
// CHECK:STDERR:                          ^~~~~~~~~
// CHECK:STDERR:
var bad_call: array(i32, JustRight(1, 2, 3));

// --- fail_mixed_add.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_mixed_add.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAdd1(a: i32, b: Core.IntLiteral()) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAdd1(a: i32, b: Core.IntLiteral()) -> i32 = "int.sadd";
// CHECK:STDERR: fail_mixed_add.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAdd2(a: Core.IntLiteral(), b: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAdd2(a: Core.IntLiteral(), b: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_mixed_add.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAdd3(a: i32, b: Core.IntLiteral()) -> Core.IntLiteral() = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAdd3(a: i32, b: Core.IntLiteral()) -> Core.IntLiteral() = "int.sadd";
// CHECK:STDERR: fail_mixed_add.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAdd4(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAdd4(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.sadd";

// --- fail_overflow.carbon

library "[[@TEST_NAME]]";

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

let a: i32 = Add(0x7FFFFFFF, 0);
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: error: integer overflow in calculation `2147483647 + 1` [CompileTimeIntegerOverflow]
// CHECK:STDERR: let b: i32 = Add(0x7FFFFFFF, 1);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = Add(0x7FFFFFFF, 1);

// CHECK:STDOUT: --- i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Add.type.b1f: type = fn_type @Add.1 [concrete]
// CHECK:STDOUT:   %Add: %Add.type.b1f = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref: %Add.type.b1f = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.sadd: init %i32 = call %Add.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc17_19.1: %i32 = value_of_initializer %int.sadd
// CHECK:STDOUT:   %.loc17_19.2: %i32 = converted %int.sadd, %.loc17_19.1
// CHECK:STDOUT:   return %.loc17_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
