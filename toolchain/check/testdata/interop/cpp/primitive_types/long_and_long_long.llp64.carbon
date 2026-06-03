// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-pc-windows-msvc --clang-arg=-fno-PIE
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/primitive_types/long_and_long_long.llp64.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/primitive_types/long_and_long_long.llp64.carbon

// --- long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  let cpp_long_long: Cpp.long_long = 1;
  let unused carbon_long_long: i64 = cpp_long_long;

  var unused a: array(f32, 1 as Cpp.long_long) = (1.0,);
  //@dump-sem-ir-end
}

// --- unsigned_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  let cpp_unsigned_long_long: Cpp.unsigned_long_long = 1;
  let unused carbon_unsigned_long_long: u64 = cpp_unsigned_long_long;

  var unused a: array(f32, 1 as Cpp.unsigned_long_long) = (1.0,);
  //@dump-sem-ir-end
}

// --- long.carbon

library "[[@TEST_NAME]]";

import Cpp inline
'''
class LongResult {};
auto PassLong(long x) -> LongResult;

class IntResult {};
auto PassLong(int x) -> IntResult;
''';

fn F() {
  //@dump-sem-ir-begin
  let cpp_long: Cpp.long = 1;
  let unused cpp_long_result: Cpp.LongResult = Cpp.PassLong(cpp_long);

  let cpp_compat_long: Core.CppCompat.Long32 = cpp_long;
  let unused cpp_compat_long_result: Cpp.LongResult = Cpp.PassLong(cpp_compat_long);

  let carbon_i32: i32 = cpp_long as i32;
  let unused carbon_i32_result: Cpp.IntResult = Cpp.PassLong(carbon_i32);

  let unused carbon_i64: i64 = cpp_long;

  var unused a: array(f32, 1 as Cpp.long) = (1.0,);
  //@dump-sem-ir-end
}

// --- fail_implicit_conversion_long_to_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  let cpp_long: Cpp.long = 1;
  // CHECK:STDERR: fail_implicit_conversion_long_to_i32.carbon:[[@LINE+7]]:32: error: cannot implicitly convert expression of type `Cpp.long` to `i32` [ConversionFailure]
  // CHECK:STDERR:   let unused carbon_i32: i32 = cpp_long;
  // CHECK:STDERR:                                ^~~~~~~~
  // CHECK:STDERR: fail_implicit_conversion_long_to_i32.carbon:[[@LINE+4]]:32: note: type `Cpp.long` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused carbon_i32: i32 = cpp_long;
  // CHECK:STDERR:                                ^~~~~~~~
  // CHECK:STDERR:
  let unused carbon_i32: i32 = cpp_long;
}

// --- copy_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CopyLong() {
  //@dump-sem-ir-begin
  var a: Cpp.long = 1;
  var unused b: Cpp.long = a;
  //@dump-sem-ir-end
}

// --- comparisons_homogeneous_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHomogeneousLong() {
  //@dump-sem-ir-begin
  let a: Cpp.long = 1;
  let b: Cpp.long = 1;
  a == b;
  a != b;
  a > b;
  a < b;
  a >= b;
  a <= b;
  //@dump-sem-ir-end
}

// --- comparisons_heterogeneous_long_left_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongLeftSide() {
  let b: i32 = 1;
  //@dump-sem-ir-begin
  let a: Cpp.long = 1;
  a == b;
  a != b;
  a > b;
  a < b;
  a >= b;
  a <= b;

  a == 1;
  a != 1;
  a > 1;
  a < 1;
  a >= 1;
  a <= 1;
  //@dump-sem-ir-end
}

// --- comparisons_heterogeneous_long_right_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongRightSide() {
  let b: i32 = 1;
  //@dump-sem-ir-begin
  let a: Cpp.long = 1;
  b == a;
  b != a;
  b > a;
  b < a;
  b >= a;
  b <= a;

  1 == a;
  1 != a;
  1 > a;
  1 < a;
  1 >= a;
  1 <= a;
  //@dump-sem-ir-end
}

// --- comparisons_heterogeneous_long_and_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongAndI64() {
  let a: Cpp.long = 1;
  let b: i64 = 1;
  a == b;
  b == a;
}

// --- fail_todo_comparisons_heterogeneous_long_and_i16.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongAndI16() {
  let a: Cpp.long = 1;
  let b: i16 = 1;
  // CHECK:STDERR: fail_todo_comparisons_heterogeneous_long_and_i16.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.EqWith(i16)` in type `Cpp.long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a == b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  a == b;
  // CHECK:STDERR: fail_todo_comparisons_heterogeneous_long_and_i16.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.EqWith(Cpp.long)` in type `i16` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   b == a;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  b == a;
}

// --- fail_todo_comparisons_heterogeneous_long_and_i128.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongAndI128() {
  let a: Cpp.long = 1;
  let b: i128 = 1;
  // CHECK:STDERR: fail_todo_comparisons_heterogeneous_long_and_i128.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.EqWith(i128)` in type `Cpp.long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a == b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  a == b;
  // CHECK:STDERR: fail_todo_comparisons_heterogeneous_long_and_i128.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.EqWith(Cpp.long)` in type `i128` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   b == a;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  b == a;
}

// operators

// --- arithmetic_homogeneous_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ArithmeticHomogeneousLong() {
  //@dump-sem-ir-begin
  let x: Cpp.long = 1;
  let y: Cpp.long = 1;

  let unused a: Cpp.long = -x;
  let unused b: Cpp.long = x + y;
  let unused c: Cpp.long = x - y;
  let unused d: Cpp.long = x * y;
  let unused e: Cpp.long = x / y;
  let unused f: Cpp.long = x % y;
  //@dump-sem-ir-end
}

// --- arithmetic_heterogeneous_long_left_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongLeftSide() {
  let b: i32 = 1;
  let x: Cpp.long = 1;

  AssertSameType(x + b, x);
  AssertSameType(x - b, x);
  AssertSameType(x * b, x);
  AssertSameType(x / b, x);
  AssertSameType(x % b, x);

  AssertSameType(x + 1, x);
  AssertSameType(x - 1, x);
  AssertSameType(x * 1, x);
  AssertSameType(x / 1, x);
  AssertSameType(x % 1, x);
}

// --- arithmetic_heterogeneous_long_right_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongRightSide() {
  let b: i32 = 1;
  let x: Cpp.long = 1;
  AssertSameType(b + x, x);
  AssertSameType(b - x, x);
  AssertSameType(b * x, x);
  AssertSameType(b / x, x);
  AssertSameType(b % x, x);

  AssertSameType(1 + x, x);
  AssertSameType(1 - x, x);
  AssertSameType(1 * x, x);
  AssertSameType(1 / x, x);
  AssertSameType(1 % x, x);
}

// --- arithmetic_heterogeneous_long_and_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongAndI64() {
  let x: Cpp.long = 1;
  let y: i64 = 1;
  AssertSameType(x + y, y);
  AssertSameType(y + x, y);
}

// --- fail_todo_arithmetic_heterogeneous_long_and_i16.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongAndI16() {
  let x: Cpp.long = 1;
  let y: i16 = 1;
  // CHECK:STDERR: fail_todo_arithmetic_heterogeneous_long_and_i16.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.AddWith(i16)` in type `Cpp.long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(x + y, x);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(x + y, x);

  // CHECK:STDERR: fail_todo_arithmetic_heterogeneous_long_and_i16.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.AddWith(Cpp.long)` in type `i16` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(y + x, x);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(y + x, x);
}

// --- fail_todo_arithmetic_heterogeneous_long_and_i128.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongAndI128() {
  let x: Cpp.long = 1;
  let y: i128 = 1;
  // CHECK:STDERR: fail_todo_arithmetic_heterogeneous_long_and_i128.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.AddWith(i128)` in type `Cpp.long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(x + y, y);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(x + y, y);

  // CHECK:STDERR: fail_todo_arithmetic_heterogeneous_long_and_i128.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.AddWith(Cpp.long)` in type `i128` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(y + x, y);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(y + x, y);
}

// --- bitwise_homogeneous_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn BitWiseHomogeneousLong() {
  let a: Cpp.long = 1;
  let b: Cpp.long = 1;
  let c: Cpp.long = 1;

  AssertSameType(^a, c);
  AssertSameType(a & b, c);
  AssertSameType(a | b, c);
  AssertSameType(a ^ b, c);
  AssertSameType(a << b, c);
  AssertSameType(a >> b, c);
}

// --- bitwise_heterogeneous_long_left_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongLeftSide() {
  let b: i32 = 1;
  let a: Cpp.long = 1;

  AssertSameType(a & b, a);
  AssertSameType(a | b, a);
  AssertSameType(a ^ b, a);
  AssertSameType(a << b, a);
  AssertSameType(a >> b, a);

  AssertSameType(a & 1, a);
  AssertSameType(a | 1, a);
  AssertSameType(a ^ 1, a);
  AssertSameType(a << 1, a);
  AssertSameType(a >> 1, a);
}

// --- bitwise_heterogeneous_long_right_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongRightSide() {
  let b: i32 = 1;
  let a: Cpp.long = 1;

  AssertSameType(b & a, a);
  AssertSameType(b | a, a);
  AssertSameType(b ^ a, a);
  AssertSameType(b << a, a);
  AssertSameType(b >> a, a);

  AssertSameType(1 & a, a);
  AssertSameType(1 | a, a);
  AssertSameType(1 ^ a, a);
  AssertSameType(1 << a, a);
  AssertSameType(1 >> a, a);
}

// --- bitwise_heterogeneous_long_and_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongAndI64() {
  let a: Cpp.long = 1;
  let b: i64 = 1;

  AssertSameType(a & b, b);
  AssertSameType(b & a, b);
}

// --- fail_todo_bitwise_heterogeneous_long_and_i16.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongAndI16() {
  let a: Cpp.long = 1;
  let b: i16 = 1;

  // CHECK:STDERR: fail_todo_bitwise_heterogeneous_long_and_i16.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.BitAndWith(i16)` in type `Cpp.long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(a & b, a);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(a & b, a);
  // CHECK:STDERR: fail_todo_bitwise_heterogeneous_long_and_i16.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.BitAndWith(Cpp.long)` in type `i16` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(b & a, a);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(b & a, a);
}

// --- fail_todo_bitwise_heterogeneous_long_and_i128.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongAndI128() {
  let a: Cpp.long = 1;
  let b: i128 = 1;

  // CHECK:STDERR: fail_todo_bitwise_heterogeneous_long_and_i128.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.BitAndWith(i128)` in type `Cpp.long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(a & b, b);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(a & b, b);
  // CHECK:STDERR: fail_todo_bitwise_heterogeneous_long_and_i128.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.BitAndWith(Cpp.long)` in type `i128` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(b & a, b);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(b & a, b);
}

// --- compound_assignment_homogeneous_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentHomogeneousLong() {
  //@dump-sem-ir-begin
  var a: Cpp.long = 1;
  var b: Cpp.long = 1;

  a += b;
  a -= b;
  a *= b;
  a /= b;
  a %= b;

  a &= b;
  a |= b;
  a ^= b;
  a <<= b;
  a >>= b;
  //@dump-sem-ir-end
}

// --- compound_assignment_hereogeneous_long_and_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentLongAndI32() {
  let b: i32 = 1;
  //@dump-sem-ir-begin
  var a: Cpp.long = 1;
  a += b;
  a -= b;
  a *= b;
  a /= b;
  a %= b;

  a &= b;
  a |= b;
  a ^= b;
  a <<= b;
  a >>= b;
  //@dump-sem-ir-end
}

// --- compound_assignment_heterogeneous_long_and_int_literal.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentLongAndIntLiteral() {
  //@dump-sem-ir-begin
  var a: Cpp.long = 1;
  a += 1;
  //@dump-sem-ir-end
}

// --- compound_assignment_heterogeneous_long_and_runtime_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentLongAndRuntimeI32() {
  //@dump-sem-ir-begin
  var a: Cpp.long = 1;
  let b: i32 = 1;
  a += b;
  //@dump-sem-ir-end
}

// --- fail_todo_compound_assignment_heterogeneous_long_and_i16.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentHeterogeneousI16Long() {
  var a: Cpp.long = 1;
  // CHECK:STDERR: fail_todo_compound_assignment_heterogeneous_long_and_i16.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.AddAssignWith(i16)` in type `Cpp.long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a += (1 as i16);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  a += (1 as i16);
}

// --- fail_compound_assignment_heterogeneous_long_and_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentHeterogeneousI64Long() {
  var a: Cpp.long = 1;
  // CHECK:STDERR: fail_compound_assignment_heterogeneous_long_and_i64.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.AddAssignWith(i64)` in type `Cpp.long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a += (1 as i64);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  a += (1 as i64);
}

// --- increment_decrement_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn IncrementDecrementLong() {
  //@dump-sem-ir-begin
  var a: Cpp.long = 1;
  ++a;
  --a;
  //@dump-sem-ir-end
}

// --- unsigned_long.carbon

library "[[@TEST_NAME]]";

import Cpp inline
'''
class ULongResult {};
auto PassULong(unsigned long x) -> ULongResult;

class UIntResult {};
auto PassULong(unsigned int x) -> UIntResult;
''';

fn F() {
  //@dump-sem-ir-begin
  let cpp_unsigned_long: Cpp.unsigned_long = 1;
  let unused cpp_unsigned_long_result: Cpp.ULongResult = Cpp.PassULong(cpp_unsigned_long);

  let cpp_compat_ulong: Core.CppCompat.ULong32 = cpp_unsigned_long;
  let unused cpp_compat_ulong_result: Cpp.ULongResult = Cpp.PassULong(cpp_compat_ulong);

  let carbon_u32: u32 = 1;
  let unused carbon_u32_result: Cpp.UIntResult = Cpp.PassULong(carbon_u32);

  var unused a: array(f32, 1 as Cpp.unsigned_long) = (1.0,);
  //@dump-sem-ir-end
}

// --- fail_no_ulong_to_u32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  let cpp_unsigned_long: Cpp.unsigned_long = 1;

  // For portability reasons, this is not allowed even though `unsigned long`
  // happens to fit in `u32` on this target.
  // CHECK:STDERR: fail_no_ulong_to_u32.carbon:[[@LINE+11]]:25: error: cannot implicitly convert expression of type `Cpp.unsigned_long` to `u32` [ConversionFailure]
  // CHECK:STDERR:   let carbon_u32: u32 = cpp_unsigned_long;
  // CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_no_ulong_to_u32.carbon:[[@LINE+8]]:25: note: type `Cpp.unsigned_long` does not implement interface `Core.ImplicitAs(u32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let carbon_u32: u32 = cpp_unsigned_long;
  // CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_no_ulong_to_u32.carbon:[[@LINE+4]]:7: warning: binding `carbon_u32` unused [UnusedBinding]
  // CHECK:STDERR:   let carbon_u32: u32 = cpp_unsigned_long;
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let carbon_u32: u32 = cpp_unsigned_long;
}

// --- copy_unsigned_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CopyUnsignedLong() {
  //@dump-sem-ir-begin
  var a: Cpp.unsigned_long = 1;
  var unused b: Cpp.unsigned_long = a;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fdd: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.08a: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.4e8(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.ace: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.f73: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.34b: %Int.as.ImplicitAs.impl.Convert.type.f73 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.35c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl.4e8(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.fc2: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.4e8(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.fc5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.fc2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.8a1: %ImplicitAs.type.08a = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.35c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.744: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.8a1) [concrete]
// CHECK:STDOUT:   %.d4c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.744, %ImplicitAs.facet.8a1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.fc5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.fc5, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.072: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0f3: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.a26: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.b69: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.f55: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.44f: %Core.IntLiteral.as.As.impl.Convert.type.f55 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.0e9: <witness> = impl_witness imports.%As.impl_witness_table.2b9, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8db: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.fb1: %Core.IntLiteral.as.As.impl.Convert.type.8db = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b69 = facet_value Core.IntLiteral, (%As.impl_witness.0e9) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.553: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i64, %As.facet) [concrete]
// CHECK:STDOUT:   %.9b9: type = fn_type_with_self_type %As.WithSelf.Convert.type.553, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.fb1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.fb1, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.b27: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4e4: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.eac, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.382: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.63c: %Int.as.ImplicitAs.impl.Convert.type.382 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.643: %ImplicitAs.type.ace = facet_value %i64, (%ImplicitAs.impl_witness.4e4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.aac: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.643) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.aac, %ImplicitAs.facet.643 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.0f3, %Int.as.ImplicitAs.impl.Convert.63c [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.63c, @Int.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.e78: <bound method> = bound_method %int_1.0f3, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.a26 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.aab: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d0d: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.a26)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.539: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.65d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.234, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.82a: %ImplicitAs.type.d0d = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.65d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c87: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.a26, %ImplicitAs.facet.82a) [concrete]
// CHECK:STDOUT:   %.3a8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c87, %ImplicitAs.facet.82a [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.aaa: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.2ec: %f32.a26 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.2ec) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = @F.%i64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.4e8.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.4e8.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl.4e8 [concrete]
// CHECK:STDOUT:   %Core.import_ref.835: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.f73) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.34b)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.eac = impl_witness_table (%Core.import_ref.835), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.fd7: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.f55) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.44f)]
// CHECK:STDOUT:   %As.impl_witness_table.2b9 = impl_witness_table (%Core.import_ref.fd7), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.63cb: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.539)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.234 = impl_witness_table (%Core.import_ref.63cb), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.fdd = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_25: type = splice_block %long_long.ref.loc8 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.d4c = impl_witness_access constants.%ImplicitAs.impl_witness.35c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.fc5]
// CHECK:STDOUT:   %bound_method.loc8_38.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_38.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.072]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc8_38.2(%int_1.loc8) [concrete = constants.%int_1.0f3]
// CHECK:STDOUT:   %.loc8_38.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0f3]
// CHECK:STDOUT:   %.loc8_38.2: %i64 = converted %int_1.loc8, %.loc8_38.1 [concrete = constants.%int_1.0f3]
// CHECK:STDOUT:   %cpp_long_long: %i64 = value_binding cpp_long_long, %.loc8_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_long.patt: %pattern_type.fdd = value_binding_pattern carbon_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref: %i64 = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %i64.loc9: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long_long: %i64 = value_binding carbon_long_long, %cpp_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.aab = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.aab = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_55: %.3a8 = impl_witness_access constants.%ImplicitAs.impl_witness.65d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd]
// CHECK:STDOUT:   %bound_method.loc11_55.1: <bound method> = bound_method %float, %impl.elem0.loc11_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_55: <specific function> = specific_function %impl.elem0.loc11_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_55.2: <bound method> = bound_method %float, %specific_fn.loc11_55 [concrete = constants.%bound_method.aaa]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.a26 = call %bound_method.loc11_55.2(%float) [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc11_55.2: init %f32.a26 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_55.3: ref %f32.a26 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_55.4: init %f32.a26 to %.loc11_55.3 = in_place_init %.loc11_55.2 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc11_55.5: init %array_type to %a.var = array_init (%.loc11_55.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_55.1, %.loc11_55.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_46: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc11: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.9b9 = impl_witness_access constants.%As.impl_witness.0e9, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.fb1]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.b27]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.0f3]
// CHECK:STDOUT:     %.loc11_30.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.0f3]
// CHECK:STDOUT:     %.loc11_30.2: %i64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.0f3]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.737 = impl_witness_access constants.%ImplicitAs.impl_witness.4e4, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.63c]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.e78]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.2(%self.param: ref %f32.a26) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.3(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsigned_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.597: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9b6: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.eb6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.501(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.498: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.eb6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.ace: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.594: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.791(%From.fe9) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.141: %UInt.as.ImplicitAs.impl.Convert.type.594 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.248: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0c6, @Core.IntLiteral.as.ImplicitAs.impl.501(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.dbe: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.501(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.b79: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.dbe = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b50: %ImplicitAs.type.9b6 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.248) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a9d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.b50) [concrete]
// CHECK:STDOUT:   %.200: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a9d, %ImplicitAs.facet.b50 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.b79 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.b79, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.231: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.8a8: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.a26: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.8af: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.150: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.b6a: %Core.IntLiteral.as.As.impl.Convert.type.150 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.042: <witness> = impl_witness imports.%As.impl_witness_table.613, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.72b: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.113: %Core.IntLiteral.as.As.impl.Convert.type.72b = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.8af = facet_value Core.IntLiteral, (%As.impl_witness.042) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.ed7: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u64, %As.facet) [concrete]
// CHECK:STDOUT:   %.528: type = fn_type_with_self_type %As.WithSelf.Convert.type.ed7, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.113 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.113, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.e13: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.569: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1b2, @UInt.as.ImplicitAs.impl.791(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.fea: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.791(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.7f4: %UInt.as.ImplicitAs.impl.Convert.type.fea = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f07: %ImplicitAs.type.ace = facet_value %u64, (%ImplicitAs.impl_witness.569) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.85c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.f07) [concrete]
// CHECK:STDOUT:   %.71b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.85c, %ImplicitAs.facet.f07 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.8a8, %UInt.as.ImplicitAs.impl.Convert.7f4 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.7f4, @UInt.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.fb5: <bound method> = bound_method %int_1.8a8, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.a26 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.aab: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d0d: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.a26)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.539: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.65d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.234, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.82a: %ImplicitAs.type.d0d = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.65d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c87: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.a26, %ImplicitAs.facet.82a) [concrete]
// CHECK:STDOUT:   %.3a8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c87, %ImplicitAs.facet.82a [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.aaa: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.2ec: %f32.a26 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.2ec) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long_long = @F.%u64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.e9f: @Core.IntLiteral.as.ImplicitAs.impl.501.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.eb6) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.501.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.498)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0c6 = impl_witness_table (%Core.import_ref.e9f), @Core.IntLiteral.as.ImplicitAs.impl.501 [concrete]
// CHECK:STDOUT:   %Core.import_ref.96b: @UInt.as.ImplicitAs.impl.791.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.594) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.791.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.141)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1b2 = impl_witness_table (%Core.import_ref.96b), @UInt.as.ImplicitAs.impl.791 [concrete]
// CHECK:STDOUT:   %Core.import_ref.8db: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.150) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.b6a)]
// CHECK:STDOUT:   %As.impl_witness_table.613 = impl_witness_table (%Core.import_ref.8db), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.63cb: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.539)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.234 = impl_witness_table (%Core.import_ref.63cb), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.597 = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_34: type = splice_block %unsigned_long_long.ref.loc8 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long_long.ref.loc8: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.200 = impl_witness_access constants.%ImplicitAs.impl_witness.248, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.b79]
// CHECK:STDOUT:   %bound_method.loc8_56.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_56.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.231]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc8_56.2(%int_1.loc8) [concrete = constants.%int_1.8a8]
// CHECK:STDOUT:   %.loc8_56.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.8a8]
// CHECK:STDOUT:   %.loc8_56.2: %u64 = converted %int_1.loc8, %.loc8_56.1 [concrete = constants.%int_1.8a8]
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = value_binding cpp_unsigned_long_long, %.loc8_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long_long.patt: %pattern_type.597 = value_binding_pattern carbon_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref: %u64 = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %u64.loc9: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long_long: %u64 = value_binding carbon_unsigned_long_long, %cpp_unsigned_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.aab = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.aab = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_64.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_64: %.3a8 = impl_witness_access constants.%ImplicitAs.impl_witness.65d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd]
// CHECK:STDOUT:   %bound_method.loc11_64.1: <bound method> = bound_method %float, %impl.elem0.loc11_64 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_64: <specific function> = specific_function %impl.elem0.loc11_64, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_64.2: <bound method> = bound_method %float, %specific_fn.loc11_64 [concrete = constants.%bound_method.aaa]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.a26 = call %bound_method.loc11_64.2(%float) [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc11_64.2: init %f32.a26 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_64.3: ref %f32.a26 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_64.4: init %f32.a26 to %.loc11_64.3 = in_place_init %.loc11_64.2 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc11_64.5: init %array_type to %a.var = array_init (%.loc11_64.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_64.1, %.loc11_64.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_55: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc11: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.528 = impl_witness_access constants.%As.impl_witness.042, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.113]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.e13]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.8a8]
// CHECK:STDOUT:     %.loc11_30.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.8a8]
// CHECK:STDOUT:     %.loc11_30.2: %u64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.8a8]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.71b = impl_witness_access constants.%ImplicitAs.impl_witness.569, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.7f4]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @UInt.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.fb5]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.2(%self.param: ref %f32.a26) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.3(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.754: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.754 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6fa: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ace: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.08a: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e41: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.293 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c09: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %.f84: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93b, %ImplicitAs.facet.c09 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.3d0: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %LongResult: type = class_type @LongResult [concrete]
// CHECK:STDOUT:   %pattern_type.676: type = pattern_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLong.cpp_overload_set.type: type = cpp_overload_set_type @PassLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassLong.cpp_overload_set.value: %PassLong.cpp_overload_set.type = cpp_overload_set_value @PassLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.c5a: type = ptr_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.type.972681.1: type = fn_type @PassLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.b57a85.1: %PassLong__carbon_thunk.type.972681.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %IntResult: type = class_type @IntResult [concrete]
// CHECK:STDOUT:   %pattern_type.47e: type = pattern_type %IntResult [concrete]
// CHECK:STDOUT:   %ptr.ff0: type = ptr_type %IntResult [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.type.972681.2: type = fn_type @PassLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.b57a85.2: %PassLong__carbon_thunk.type.972681.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.fdd: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e25: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0d0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.07b: %ImplicitAs.type.08a = facet_value %Cpp.long, (%ImplicitAs.impl_witness.e25) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.bbd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.07b) [concrete]
// CHECK:STDOUT:   %.ed6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.bbd, %ImplicitAs.facet.07b [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.a26: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.909: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.909 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.4f6: type = facet_type <@As, @As(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.f62: <witness> = impl_witness imports.%As.impl_witness_table.737 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.4f6 = facet_value Core.IntLiteral, (%As.impl_witness.f62) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.fe0: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long, %As.facet) [concrete]
// CHECK:STDOUT:   %.2c9: type = fn_type_with_self_type %As.WithSelf.Convert.type.fe0, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b4e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d05: %ImplicitAs.type.ace = facet_value %Cpp.long, (%ImplicitAs.impl_witness.bc1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6fa: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.d05) [concrete]
// CHECK:STDOUT:   %.d01: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6fa, %ImplicitAs.facet.d05 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.3d0, %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.a26 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.aab: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d0d: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.a26)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.539: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.65d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.234, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.82a: %ImplicitAs.type.d0d = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.65d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c87: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.a26, %ImplicitAs.facet.82a) [concrete]
// CHECK:STDOUT:   %.3a8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c87, %ImplicitAs.facet.82a [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.2ec: %f32.a26 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.2ec) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc26_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT:   %IntResult.cpp_destructor.type: type = fn_type @IntResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %IntResult.cpp_destructor: %IntResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor.type: type = fn_type @LongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor: %LongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .CppCompat = %CppCompat.463205.1
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     .LongResult = %LongResult.decl
// CHECK:STDOUT:     .PassLong = %PassLong.cpp_overload_set.value
// CHECK:STDOUT:     .IntResult = %IntResult.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.463205.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .Long32 = %Core.Long32
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Long32: type = import_ref Core//prelude/types/cpp/int, Long32, loaded [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.754 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.adef5f.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.293 = impl_witness_table (%Core.import_ref.adef5f.1), @Core.IntLiteral.as.ImplicitAs.impl.ec9 [concrete]
// CHECK:STDOUT:   %LongResult.decl: type = class_decl @LongResult [concrete = constants.%LongResult] {} {}
// CHECK:STDOUT:   %PassLong.cpp_overload_set.value: %PassLong.cpp_overload_set.type = cpp_overload_set_value @PassLong.cpp_overload_set [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassLong__carbon_thunk.decl.f47c99.1: %PassLong__carbon_thunk.type.972681.1 = fn_decl @PassLong__carbon_thunk.1 [concrete = constants.%PassLong__carbon_thunk.b57a85.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %IntResult.decl: type = class_decl @IntResult [concrete = constants.%IntResult] {} {}
// CHECK:STDOUT:   %PassLong__carbon_thunk.decl.f47c99.2: %PassLong__carbon_thunk.type.972681.2 = fn_decl @PassLong__carbon_thunk.2 [concrete = constants.%PassLong__carbon_thunk.b57a85.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.adef5f.2: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0d0 = impl_witness_table (%Core.import_ref.adef5f.2), @Cpp.long.as.ImplicitAs.impl.6e3 [concrete]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.909 = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.adef5f.3: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.737 = impl_witness_table (%Core.import_ref.adef5f.3), @Core.IntLiteral.as.As.impl.846 [concrete]
// CHECK:STDOUT:   %Core.import_ref.adef5f.4: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b4e = impl_witness_table (%Core.import_ref.adef5f.4), @Cpp.long.as.ImplicitAs.impl.6b0 [concrete]
// CHECK:STDOUT:   %Core.import_ref.63cb: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.539)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.234 = impl_witness_table (%Core.import_ref.63cb), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.83e = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_20: type = splice_block %long.ref.loc15 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc15: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc15_28.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc15_28.2: %Cpp.long = converted %int_1.loc15, %.loc15_28.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %cpp_long: %Cpp.long = value_binding cpp_long, %.loc15_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_result.patt: %pattern_type.676 = value_binding_pattern cpp_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc16: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long.ref.loc16: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc16_69.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.c5a = addr_of %.loc16_69.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.f47c99.1(%cpp_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_69.2: init %LongResult to %.loc16_69.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_34: type = splice_block %LongResult.ref.loc16 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc16: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_69.3: ref %LongResult = temporary %.loc16_69.1, %.loc16_69.2
// CHECK:STDOUT:   %.loc16_69.4: %LongResult = acquire_value %.loc16_69.3
// CHECK:STDOUT:   %cpp_long_result: %LongResult = value_binding cpp_long_result, %.loc16_69.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long.patt: %pattern_type.83e = value_binding_pattern cpp_compat_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc18: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc18: type = splice_block %Long32.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.463205.1 [concrete = imports.%CppCompat.463205.1]
// CHECK:STDOUT:     %Long32.ref: type = name_ref Long32, imports.%Core.Long32 [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long: %Cpp.long = value_binding cpp_compat_long, %cpp_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_result.patt: %pattern_type.676 = value_binding_pattern cpp_compat_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_55: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc19: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long.ref: %Cpp.long = name_ref cpp_compat_long, %cpp_compat_long
// CHECK:STDOUT:   %.loc19_83.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.c5a = addr_of %.loc19_83.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.f47c99.1(%cpp_compat_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_83.2: init %LongResult to %.loc19_83.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_41: type = splice_block %LongResult.ref.loc19 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc19: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_83.3: ref %LongResult = temporary %.loc19_83.1, %.loc19_83.2
// CHECK:STDOUT:   %.loc19_83.4: %LongResult = acquire_value %.loc19_83.3
// CHECK:STDOUT:   %cpp_compat_long_result: %LongResult = value_binding cpp_compat_long_result, %.loc19_83.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32.patt: %pattern_type.104 = value_binding_pattern carbon_i32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc21: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i32.loc21_37: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc21_34.1: %i32 = as_compatible %cpp_long.ref.loc21
// CHECK:STDOUT:   %.loc21_34.2: %i32 = converted %cpp_long.ref.loc21, %.loc21_34.1
// CHECK:STDOUT:   %i32.loc21_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %carbon_i32: %i32 = value_binding carbon_i32, %.loc21_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32_result.patt: %pattern_type.47e = value_binding_pattern carbon_i32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_49: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc22: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i32.ref: %i32 = name_ref carbon_i32, %carbon_i32
// CHECK:STDOUT:   %.loc22_72.1: ref %IntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.ff0 = addr_of %.loc22_72.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.f47c99.2(%carbon_i32.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_72.2: init %IntResult to %.loc22_72.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_36: type = splice_block %IntResult.ref [concrete = constants.%IntResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntResult.ref: type = name_ref IntResult, imports.%IntResult.decl [concrete = constants.%IntResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_72.3: ref %IntResult = temporary %.loc22_72.1, %.loc22_72.2
// CHECK:STDOUT:   %.loc22_72.4: %IntResult = acquire_value %.loc22_72.3
// CHECK:STDOUT:   %carbon_i32_result: %IntResult = value_binding carbon_i32_result, %.loc22_72.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.fdd = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc24: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc24: %.ed6 = impl_witness_access constants.%ImplicitAs.impl_witness.e25, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %cpp_long.ref.loc24, %impl.elem0.loc24
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc24: init %i64 = call %bound_method.loc24(%cpp_long.ref.loc24)
// CHECK:STDOUT:   %.loc24_32.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc24
// CHECK:STDOUT:   %.loc24_32.2: %i64 = converted %cpp_long.ref.loc24, %.loc24_32.1
// CHECK:STDOUT:   %carbon_i64: %i64 = value_binding carbon_i64, %.loc24_32.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.aab = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.aab = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc26_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc26_50: %.3a8 = impl_witness_access constants.%ImplicitAs.impl_witness.65d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd]
// CHECK:STDOUT:   %bound_method.loc26_50.1: <bound method> = bound_method %float, %impl.elem0.loc26_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc26_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_50.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.a26 = call %bound_method.loc26_50.2(%float) [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc26_50.2: init %f32.a26 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc26_50.3: ref %f32.a26 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc26_50.4: init %f32.a26 to %.loc26_50.3 = in_place_init %.loc26_50.2 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc26_50.5: init %array_type to %a.var = array_init (%.loc26_50.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc26_3: init %array_type = converted %.loc26_50.1, %.loc26_50.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc26_3
// CHECK:STDOUT:   %.loc26_41: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:     %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc26: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:     %impl.elem0.loc26_30.1: %.2c9 = impl_witness_access constants.%As.impl_witness.f62, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc26_30.1: <bound method> = bound_method %int_1.loc26, %impl.elem0.loc26_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long = call %bound_method.loc26_30.1(%int_1.loc26) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:     %.loc26_30.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:     %.loc26_30.2: %Cpp.long = converted %int_1.loc26, %.loc26_30.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:     %impl.elem0.loc26_30.2: %.d01 = impl_witness_access constants.%ImplicitAs.impl_witness.bc1, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:     %bound_method.loc26_30.2: <bound method> = bound_method %.loc26_30.2, %impl.elem0.loc26_30.2 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long.as.ImplicitAs.impl.Convert.call.loc26: init Core.IntLiteral = call %bound_method.loc26_30.2(%.loc26_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc26_30.3: Core.IntLiteral = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc26 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc26_30.4: Core.IntLiteral = converted %.loc26_30.2, %.loc26_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc26_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   %IntResult.cpp_destructor.bound: <bound method> = bound_method %.loc22_72.3, constants.%IntResult.cpp_destructor
// CHECK:STDOUT:   %IntResult.cpp_destructor.call: init %empty_tuple.type = call %IntResult.cpp_destructor.bound(%.loc22_72.3)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_83.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call.loc19: init %empty_tuple.type = call %LongResult.cpp_destructor.bound.loc19(%.loc19_83.3)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound.loc16: <bound method> = bound_method %.loc16_69.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call.loc16: init %empty_tuple.type = call %LongResult.cpp_destructor.bound.loc16(%.loc16_69.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_3.2(%self.param: ref %f32.a26) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_3.3(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6fa: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e41: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.293 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f84: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93b, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.3d0: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.553: <witness> = impl_witness imports.%Copy.impl_witness_table.256 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.553) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.b77: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.253: type = fn_type_with_self_type %Copy.WithSelf.Op.type.b77, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ade: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.293 = impl_witness_table (%Core.import_ref.ade), @Core.IntLiteral.as.ImplicitAs.impl.ec9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.f63: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.256 = impl_witness_table (%Core.import_ref.f63), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.83e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.83e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.83e = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.83e = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_28: %Cpp.long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.253 = impl_witness_access constants.%Copy.impl_witness.553, element0 [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_28, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.call: init %Cpp.long = call %bound_method.loc9(%.loc9_28)
// CHECK:STDOUT:   assign %b.var, %Cpp.long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_homogeneous_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6fa: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e41: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.293 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f84: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93b, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.3d0: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.0be: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.455: <witness> = impl_witness imports.%EqWith.impl_witness_table.d73 [concrete]
// CHECK:STDOUT:   %EqWith.facet.2bd: %EqWith.type.0be = facet_value %Cpp.long, (%EqWith.impl_witness.455) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.f74: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.2bd) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.c21: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.2bd) [concrete]
// CHECK:STDOUT:   %.fa7: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.f74, %EqWith.facet.2bd [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.78c: type = fn_type @Cpp.long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.1ca: %Cpp.long.as.EqWith.impl.Equal.type.78c = struct_value () [concrete]
// CHECK:STDOUT:   %.d35: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.c21, %EqWith.facet.2bd [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.8e5: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.a6a: %Cpp.long.as.EqWith.impl.NotEqual.type.8e5 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.f75: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.f4f: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.497 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.410: %OrderedWith.type.f75 = facet_value %Cpp.long, (%OrderedWith.impl_witness.f4f) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.4c6: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.410) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.6a3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.410) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.607: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.410) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.7f3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.410) [concrete]
// CHECK:STDOUT:   %.fd6: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.607, %OrderedWith.facet.410 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.725: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.7c8: %Cpp.long.as.OrderedWith.impl.Greater.type.725 = struct_value () [concrete]
// CHECK:STDOUT:   %.fd0: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.4c6, %OrderedWith.facet.410 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.335: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.606: %Cpp.long.as.OrderedWith.impl.Less.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %.3e2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.7f3, %OrderedWith.facet.410 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = struct_value () [concrete]
// CHECK:STDOUT:   %.cc3: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.6a3, %OrderedWith.facet.410 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.42c: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.42c = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ade: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.293 = impl_witness_table (%Core.import_ref.ade), @Core.IntLiteral.as.ImplicitAs.impl.ec9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.b5b: %Cpp.long.as.EqWith.impl.Equal.type.78c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %Core.import_ref.3bc: %Cpp.long.as.EqWith.impl.NotEqual.type.8e5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %EqWith.impl_witness_table.d73 = impl_witness_table (%Core.import_ref.b5b, %Core.import_ref.3bc), @Cpp.long.as.EqWith.impl.554 [concrete]
// CHECK:STDOUT:   %Core.import_ref.746: %Cpp.long.as.OrderedWith.impl.Less.type.335 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %Core.import_ref.202: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.42c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %Core.import_ref.cc2: %Cpp.long.as.OrderedWith.impl.Greater.type.725 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %Core.import_ref.4c7: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.497 = impl_witness_table (%Core.import_ref.746, %Core.import_ref.202, %Core.import_ref.cc2, %Core.import_ref.4c7), @Cpp.long.as.OrderedWith.impl.439 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.83e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc8_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc8_21.2: %Cpp.long = converted %int_1.loc8, %.loc8_21.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc8_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.83e = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc9_21.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.fa7 = impl_witness_access constants.%EqWith.impl_witness.455, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.d35 = impl_witness_access constants.%EqWith.impl_witness.455, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.fd6 = impl_witness_access constants.%OrderedWith.impl_witness.f4f, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.fd0 = impl_witness_access constants.%OrderedWith.impl_witness.f4f, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.3e2 = impl_witness_access constants.%OrderedWith.impl_witness.f4f, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.cc3 = impl_witness_access constants.%OrderedWith.impl_witness.f4f, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15(%a.ref.loc15, %b.ref.loc15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_heterogeneous_long_left_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6fa: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e41: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.293 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c09: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %.f84: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93b, %ImplicitAs.facet.c09 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.3d0: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.536: type = facet_type <@EqWith, @EqWith(%i32)> [concrete]
// CHECK:STDOUT:   %T.887: %ImplicitAs.type.6fa = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.9e873c.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.e2f(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.fc9fc1.1: %Cpp.long.as.EqWith.impl.NotEqual.type.9e873c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.9e873c.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.e2f(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.fc9fc1.2: %Cpp.long.as.EqWith.impl.NotEqual.type.9e873c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.d45cab.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.e2f(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.c9f6e2.1: %Cpp.long.as.EqWith.impl.Equal.type.d45cab.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.d45cab.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.e2f(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.c9f6e2.2: %Cpp.long.as.EqWith.impl.Equal.type.d45cab.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.70b: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.72d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.29c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0fb: %ImplicitAs.type.6fa = facet_value %i32, (%ImplicitAs.impl_witness.72d) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.23c: <witness> = impl_witness imports.%EqWith.impl_witness_table.f94, @Cpp.long.as.EqWith.impl.e2f(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.17afc6.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.e2f(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.e1af36.1: %Cpp.long.as.EqWith.impl.Equal.type.17afc6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.5255ce.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.e2f(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.ab5731.1: %Cpp.long.as.EqWith.impl.NotEqual.type.5255ce.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.17afc6.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.e2f(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.e1af36.2: %Cpp.long.as.EqWith.impl.Equal.type.17afc6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.5255ce.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.e2f(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.ab5731.2: %Cpp.long.as.EqWith.impl.NotEqual.type.5255ce.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.757: %EqWith.type.536 = facet_value %Cpp.long, (%EqWith.impl_witness.23c) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.3d6: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i32, %EqWith.facet.757) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.966: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i32, %EqWith.facet.757) [concrete]
// CHECK:STDOUT:   %.c27: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.3d6, %EqWith.facet.757 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.fa6f78.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.e1af36.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.fa6f78.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.e1af36.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.db6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %.9d6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.db6, %ImplicitAs.facet.0fb [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.475: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.966, %EqWith.facet.757 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.3a486e.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.ab5731.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.3a486e.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.ab5731.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.58a: type = facet_type <@OrderedWith, @OrderedWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.0ba330.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.786(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.d190f3.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.0ba330.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.0ba330.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.786(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.d190f3.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.0ba330.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.a01d2b.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.786(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.1365fc.1: %Cpp.long.as.OrderedWith.impl.Greater.type.a01d2b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.a01d2b.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.786(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.1365fc.2: %Cpp.long.as.OrderedWith.impl.Greater.type.a01d2b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8f5c1d.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.786(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.b8ff7f.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8f5c1d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8f5c1d.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.786(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.b8ff7f.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8f5c1d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.77ead7.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.786(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.912b7a.1: %Cpp.long.as.OrderedWith.impl.Less.type.77ead7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.77ead7.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.786(%T.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.912b7a.2: %Cpp.long.as.OrderedWith.impl.Less.type.77ead7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.0e2: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.da4: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.e4e, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.a2f55b.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.335d62.1: %Cpp.long.as.OrderedWith.impl.Less.type.a2f55b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8589a2.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1dbdf6.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8589a2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.375c4b.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.d25f2f.1: %Cpp.long.as.OrderedWith.impl.Greater.type.375c4b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.549ba3.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f6b944.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.549ba3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.a2f55b.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.335d62.2: %Cpp.long.as.OrderedWith.impl.Less.type.a2f55b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8589a2.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1dbdf6.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8589a2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.375c4b.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.d25f2f.2: %Cpp.long.as.OrderedWith.impl.Greater.type.375c4b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.549ba3.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f6b944.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.549ba3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.35f: %OrderedWith.type.58a = facet_value %Cpp.long, (%OrderedWith.impl_witness.da4) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.c2b: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.35f) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.728: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.35f) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.62f: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.35f) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.9e0: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.35f) [concrete]
// CHECK:STDOUT:   %.1dc: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.62f, %OrderedWith.facet.35f [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.624957.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.d25f2f.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.624957.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.d25f2f.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %.e70: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.c2b, %OrderedWith.facet.35f [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.87c251.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.335d62.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.87c251.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.335d62.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %.fa8: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.9e0, %OrderedWith.facet.35f [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e9f17d.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f6b944.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e9f17d.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f6b944.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %.7b7: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.728, %OrderedWith.facet.35f [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.1318bc.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1dbdf6.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.1318bc.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1dbdf6.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.475: <witness> = impl_witness imports.%EqWith.impl_witness_table.f94, @Cpp.long.as.EqWith.impl.e2f(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.46b9a6.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.e2f(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.7dbd0a.1: %Cpp.long.as.EqWith.impl.Equal.type.46b9a6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.f0dc65.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.e2f(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.dd0676.1: %Cpp.long.as.EqWith.impl.NotEqual.type.f0dc65.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.46b9a6.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.e2f(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.7dbd0a.2: %Cpp.long.as.EqWith.impl.Equal.type.46b9a6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.f0dc65.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.e2f(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.dd0676.2: %Cpp.long.as.EqWith.impl.NotEqual.type.f0dc65.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.4c4: %EqWith.type.70b = facet_value %Cpp.long, (%EqWith.impl_witness.475) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b4c: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.4c4) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.f25: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.4c4) [concrete]
// CHECK:STDOUT:   %.f43: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.b4c, %EqWith.facet.4c4 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.5752dd.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.7dbd0a.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.5752dd.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.7dbd0a.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %.e6b: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.f25, %EqWith.facet.4c4 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.e03a04.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.dd0676.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.e03a04.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.dd0676.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.70e: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.e4e, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.cc8d23.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bc9b37.1: %Cpp.long.as.OrderedWith.impl.Less.type.cc8d23.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4875b2.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.26f675.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4875b2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.4d8ecd.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.5b50d7.1: %Cpp.long.as.OrderedWith.impl.Greater.type.4d8ecd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.5183ad.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3fc9f4.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.5183ad.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.cc8d23.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bc9b37.2: %Cpp.long.as.OrderedWith.impl.Less.type.cc8d23.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4875b2.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.26f675.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4875b2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.4d8ecd.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.5b50d7.2: %Cpp.long.as.OrderedWith.impl.Greater.type.4d8ecd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.5183ad.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.786(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3fc9f4.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.5183ad.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.58d: %OrderedWith.type.0e2 = facet_value %Cpp.long, (%OrderedWith.impl_witness.70e) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.f3a: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.58d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.dc4: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.58d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.bb8: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.58d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.b7d: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.58d) [concrete]
// CHECK:STDOUT:   %.384: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.bb8, %OrderedWith.facet.58d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.ea58ad.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.5b50d7.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.ea58ad.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.5b50d7.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %.28e: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.f3a, %OrderedWith.facet.58d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.b7bcad.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.bc9b37.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.b7bcad.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.bc9b37.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %.532: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.b7d, %OrderedWith.facet.58d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.660f9a.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3fc9f4.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.660f9a.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3fc9f4.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %.484: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.dc4, %OrderedWith.facet.58d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.813265.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.26f675.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.813265.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.26f675.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.adef5f.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.293 = impl_witness_table (%Core.import_ref.adef5f.1), @Core.IntLiteral.as.ImplicitAs.impl.ec9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.42d: @Cpp.long.as.EqWith.impl.e2f.%Cpp.long.as.EqWith.impl.Equal.type.2 (%Cpp.long.as.EqWith.impl.Equal.type.d45cab.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2f.%Cpp.long.as.EqWith.impl.Equal.2 (constants.%Cpp.long.as.EqWith.impl.Equal.c9f6e2.1)]
// CHECK:STDOUT:   %Core.import_ref.fd8: @Cpp.long.as.EqWith.impl.e2f.%Cpp.long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long.as.EqWith.impl.NotEqual.type.9e873c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2f.%Cpp.long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long.as.EqWith.impl.NotEqual.fc9fc1.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.f94 = impl_witness_table (%Core.import_ref.42d, %Core.import_ref.fd8), @Cpp.long.as.EqWith.impl.e2f [concrete]
// CHECK:STDOUT:   %Core.NotEqual.3e4: @Cpp.long.as.EqWith.impl.e2f.%Cpp.long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long.as.EqWith.impl.NotEqual.type.9e873c.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2f.%Cpp.long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long.as.EqWith.impl.NotEqual.fc9fc1.2)]
// CHECK:STDOUT:   %Core.Equal.de5: @Cpp.long.as.EqWith.impl.e2f.%Cpp.long.as.EqWith.impl.Equal.type.1 (%Cpp.long.as.EqWith.impl.Equal.type.d45cab.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long.as.EqWith.impl.e2f.%Cpp.long.as.EqWith.impl.Equal.1 (constants.%Cpp.long.as.EqWith.impl.Equal.c9f6e2.2)]
// CHECK:STDOUT:   %Core.import_ref.adef5f.2: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.29c = impl_witness_table (%Core.import_ref.adef5f.2), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.de7: @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.Less.type.2 (%Cpp.long.as.OrderedWith.impl.Less.type.77ead7.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long.as.OrderedWith.impl.Less.912b7a.1)]
// CHECK:STDOUT:   %Core.import_ref.caf: @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8f5c1d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.b8ff7f.1)]
// CHECK:STDOUT:   %Core.import_ref.f56: @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long.as.OrderedWith.impl.Greater.type.a01d2b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long.as.OrderedWith.impl.Greater.1365fc.1)]
// CHECK:STDOUT:   %Core.import_ref.60f: @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.0ba330.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.d190f3.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.e4e = impl_witness_table (%Core.import_ref.de7, %Core.import_ref.caf, %Core.import_ref.f56, %Core.import_ref.60f), @Cpp.long.as.OrderedWith.impl.786 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.6a2: @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.0ba330.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.d190f3.2)]
// CHECK:STDOUT:   %Core.Greater.421: @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long.as.OrderedWith.impl.Greater.type.a01d2b.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long.as.OrderedWith.impl.Greater.1365fc.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.b05: @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8f5c1d.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.b8ff7f.2)]
// CHECK:STDOUT:   %Core.Less.158: @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.Less.type.1 (%Cpp.long.as.OrderedWith.impl.Less.type.77ead7.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.786.%Cpp.long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long.as.OrderedWith.impl.Less.912b7a.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.83e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc9_21.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.c27 = impl_witness_access constants.%EqWith.impl_witness.23c, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.e1af36.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.fa6f78.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.EqWith.impl.Equal.type.17afc6.1 = specific_constant imports.%Core.Equal.de5, @Cpp.long.as.EqWith.impl.e2f(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.e1af36.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Cpp.long.as.EqWith.impl.Equal.type.17afc6.1 = name_ref Equal, %.loc10_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.e1af36.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.fa6f78.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.475 = impl_witness_access constants.%EqWith.impl_witness.23c, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.ab5731.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.3a486e.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.5255ce.1 = specific_constant imports.%Core.NotEqual.3e4, @Cpp.long.as.EqWith.impl.e2f(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.ab5731.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Cpp.long.as.EqWith.impl.NotEqual.type.5255ce.1 = name_ref NotEqual, %.loc11_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.ab5731.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.3a486e.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.1dc = impl_witness_access constants.%OrderedWith.impl_witness.da4, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.d25f2f.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.624957.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.OrderedWith.impl.Greater.type.375c4b.1 = specific_constant imports.%Core.Greater.421, @Cpp.long.as.OrderedWith.impl.786(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.d25f2f.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Cpp.long.as.OrderedWith.impl.Greater.type.375c4b.1 = name_ref Greater, %.loc12_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.d25f2f.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.624957.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12_7(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_7.2: %Cpp.long = converted %b.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.e70 = impl_witness_access constants.%OrderedWith.impl_witness.da4, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.335d62.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.87c251.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.OrderedWith.impl.Less.type.a2f55b.1 = specific_constant imports.%Core.Less.158, @Cpp.long.as.OrderedWith.impl.786(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.335d62.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Cpp.long.as.OrderedWith.impl.Less.type.a2f55b.1 = name_ref Less, %.loc13_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.335d62.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.87c251.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long = call %bound_method.loc13_7(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_7.2: %Cpp.long = converted %b.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.fa8 = impl_witness_access constants.%OrderedWith.impl_witness.da4, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f6b944.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e9f17d.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.549ba3.1 = specific_constant imports.%Core.GreaterOrEquivalent.6a2, @Cpp.long.as.OrderedWith.impl.786(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f6b944.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.549ba3.1 = name_ref GreaterOrEquivalent, %.loc14_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f6b944.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e9f17d.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.7b7 = impl_witness_access constants.%OrderedWith.impl_witness.da4, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1dbdf6.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.1318bc.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8589a2.1 = specific_constant imports.%Core.LessOrEquivalent.b05, @Cpp.long.as.OrderedWith.impl.786(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1dbdf6.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.8589a2.1 = name_ref LessOrEquivalent, %.loc15_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1dbdf6.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.1318bc.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long = call %bound_method.loc15_8(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_8.2: %Cpp.long = converted %b.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.f43 = impl_witness_access constants.%EqWith.impl_witness.475, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.7dbd0a.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.e41) [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.6fa = converted Core.IntLiteral, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.5752dd.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.EqWith.impl.Equal.type.46b9a6.1 = specific_constant imports.%Core.Equal.de5, @Cpp.long.as.EqWith.impl.e2f(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.7dbd0a.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long.as.EqWith.impl.Equal.type.46b9a6.1 = name_ref Equal, %.loc17_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.7dbd0a.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.5752dd.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.e6b = impl_witness_access constants.%EqWith.impl_witness.475, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.dd0676.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.e41) [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.6fa = converted Core.IntLiteral, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.e03a04.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.f0dc65.1 = specific_constant imports.%Core.NotEqual.3e4, @Cpp.long.as.EqWith.impl.e2f(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.dd0676.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long.as.EqWith.impl.NotEqual.type.f0dc65.1 = name_ref NotEqual, %.loc18_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.dd0676.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.e03a04.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.384 = impl_witness_access constants.%OrderedWith.impl_witness.70e, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.5b50d7.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.e41) [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.6fa = converted Core.IntLiteral, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.ea58ad.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.OrderedWith.impl.Greater.type.4d8ecd.1 = specific_constant imports.%Core.Greater.421, @Cpp.long.as.OrderedWith.impl.786(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.5b50d7.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long.as.OrderedWith.impl.Greater.type.4d8ecd.1 = name_ref Greater, %.loc19_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.5b50d7.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.ea58ad.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.28e = impl_witness_access constants.%OrderedWith.impl_witness.70e, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.bc9b37.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.e41) [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.6fa = converted Core.IntLiteral, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.b7bcad.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.OrderedWith.impl.Less.type.cc8d23.1 = specific_constant imports.%Core.Less.158, @Cpp.long.as.OrderedWith.impl.786(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.bc9b37.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long.as.OrderedWith.impl.Less.type.cc8d23.1 = name_ref Less, %.loc20_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.bc9b37.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.b7bcad.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.532 = impl_witness_access constants.%OrderedWith.impl_witness.70e, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3fc9f4.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.e41) [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.6fa = converted Core.IntLiteral, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.660f9a.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.5183ad.1 = specific_constant imports.%Core.GreaterOrEquivalent.6a2, @Cpp.long.as.OrderedWith.impl.786(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3fc9f4.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.5183ad.1 = name_ref GreaterOrEquivalent, %.loc21_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3fc9f4.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.660f9a.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.484 = impl_witness_access constants.%OrderedWith.impl_witness.70e, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.26f675.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.e41) [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.6fa = converted Core.IntLiteral, %ImplicitAs.facet.loc22 [concrete = constants.%ImplicitAs.facet.c09]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.813265.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4875b2.1 = specific_constant imports.%Core.LessOrEquivalent.b05, @Cpp.long.as.OrderedWith.impl.786(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.26f675.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4875b2.1 = name_ref LessOrEquivalent, %.loc22_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.26f675.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.813265.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %a.ref.loc22, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_8: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long = call %bound_method.loc22_8(%int_1.loc22) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc22_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc22_8.2: %Cpp.long = converted %int_1.loc22, %.loc22_8.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc22: init bool = call %bound_method.loc22_5.3(%a.ref.loc22, %.loc22_8.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_heterogeneous_long_right_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6fa: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e41: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.293 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c09: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %.f84: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93b, %ImplicitAs.facet.c09 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.3d0: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.0be: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.887: %ImplicitAs.type.6fa = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.182979.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.1a3(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.ac4f99.1: %T.as_type.as.EqWith.impl.NotEqual.type.182979.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.182979.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.1a3(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.ac4f99.2: %T.as_type.as.EqWith.impl.NotEqual.type.182979.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.0e80bf.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.1a3(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.a11a61.1: %T.as_type.as.EqWith.impl.Equal.type.0e80bf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.0e80bf.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.1a3(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.a11a61.2: %T.as_type.as.EqWith.impl.Equal.type.0e80bf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.72d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.29c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0fb: %ImplicitAs.type.6fa = facet_value %i32, (%ImplicitAs.impl_witness.72d) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.37d: <witness> = impl_witness imports.%EqWith.impl_witness_table.360, @T.as_type.as.EqWith.impl.1a3(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.50eb4b.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.1a3(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.33d0bf.1: %T.as_type.as.EqWith.impl.Equal.type.50eb4b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.989cc9.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.1a3(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.30ef12.1: %T.as_type.as.EqWith.impl.NotEqual.type.989cc9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.50eb4b.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.1a3(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.33d0bf.2: %T.as_type.as.EqWith.impl.Equal.type.50eb4b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.989cc9.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.1a3(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.30ef12.2: %T.as_type.as.EqWith.impl.NotEqual.type.989cc9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.32b: %EqWith.type.0be = facet_value %i32, (%EqWith.impl_witness.37d) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.3ac: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.32b) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.2b7: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.32b) [concrete]
// CHECK:STDOUT:   %.367: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.3ac, %EqWith.facet.32b [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.c2fb87.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.33d0bf.2, @T.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.c2fb87.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.33d0bf.1, @T.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.db6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %.9d6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.db6, %ImplicitAs.facet.0fb [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.50c: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.2b7, %EqWith.facet.32b [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.ed888b.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.30ef12.2, @T.as_type.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.ed888b.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.30ef12.1, @T.as_type.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.f75: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.42dbc4.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d3b(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.112484.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.42dbc4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.42dbc4.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d3b(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.112484.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.42dbc4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.d820b7.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.d3b(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.803b09.1: %T.as_type.as.OrderedWith.impl.Greater.type.d820b7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.d820b7.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.d3b(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.803b09.2: %T.as_type.as.OrderedWith.impl.Greater.type.d820b7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f0b1d3.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d3b(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.1f6607.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f0b1d3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f0b1d3.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d3b(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.1f6607.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f0b1d3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.8e479f.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.d3b(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.6822d0.1: %T.as_type.as.OrderedWith.impl.Less.type.8e479f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.8e479f.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.d3b(%T.887) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.6822d0.2: %T.as_type.as.OrderedWith.impl.Less.type.8e479f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.f27: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.dda, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.a0db57.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.3482c5.1: %T.as_type.as.OrderedWith.impl.Less.type.a0db57.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.82d2f0.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.510b4c.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.82d2f0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.d192c2.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.b157e1.1: %T.as_type.as.OrderedWith.impl.Greater.type.d192c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.288fa6.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8e9946.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.288fa6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.a0db57.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.3482c5.2: %T.as_type.as.OrderedWith.impl.Less.type.a0db57.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.82d2f0.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.510b4c.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.82d2f0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.d192c2.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.b157e1.2: %T.as_type.as.OrderedWith.impl.Greater.type.d192c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.288fa6.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8e9946.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.288fa6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.99a: %OrderedWith.type.f75 = facet_value %i32, (%OrderedWith.impl_witness.f27) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.ce0: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.99a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.edb: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.99a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.a8e: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.99a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.d2b: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.99a) [concrete]
// CHECK:STDOUT:   %.685: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.a8e, %OrderedWith.facet.99a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.df8542.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.b157e1.2, @T.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.df8542.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.b157e1.1, @T.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %.f18: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.ce0, %OrderedWith.facet.99a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.33d5cc.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.3482c5.2, @T.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.33d5cc.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.3482c5.1, @T.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %.de3: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.d2b, %OrderedWith.facet.99a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ed31ba.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8e9946.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ed31ba.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8e9946.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.edb, %OrderedWith.facet.99a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.71880f.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.510b4c.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.71880f.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.510b4c.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.95f: <witness> = impl_witness imports.%EqWith.impl_witness_table.360, @T.as_type.as.EqWith.impl.1a3(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.4eea73.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.1a3(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.e65d61.1: %T.as_type.as.EqWith.impl.Equal.type.4eea73.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.728f73.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.1a3(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.a73f2e.1: %T.as_type.as.EqWith.impl.NotEqual.type.728f73.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.4eea73.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.1a3(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.e65d61.2: %T.as_type.as.EqWith.impl.Equal.type.4eea73.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.728f73.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.1a3(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.a73f2e.2: %T.as_type.as.EqWith.impl.NotEqual.type.728f73.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.f3d: %EqWith.type.0be = facet_value Core.IntLiteral, (%EqWith.impl_witness.95f) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.027: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.f3d) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.4c3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.f3d) [concrete]
// CHECK:STDOUT:   %.2c2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.027, %EqWith.facet.f3d [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.c5052e.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.e65d61.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.2c5f20.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.e65d61.2, @T.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.bcadad.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.2c5f20.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.c5052e.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.e65d61.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.2c5f20.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.e65d61.1, @T.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.bcadad.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.2c5f20.2 [concrete]
// CHECK:STDOUT:   %.cdc: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.4c3, %EqWith.facet.f3d [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.77ff22.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.a73f2e.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.4dc8f8.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.a73f2e.2, @T.as_type.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.a142d4.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.4dc8f8.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.77ff22.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.a73f2e.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.4dc8f8.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.a73f2e.1, @T.as_type.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.a142d4.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.4dc8f8.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.eca: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.dda, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.1b517c.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.2d3dbd.1: %T.as_type.as.OrderedWith.impl.Less.type.1b517c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b1c747.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.a60f6a.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b1c747.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.4df4af.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.892f9f.1: %T.as_type.as.OrderedWith.impl.Greater.type.4df4af.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.10b2de.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2af6e8.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.10b2de.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.1b517c.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.2d3dbd.2: %T.as_type.as.OrderedWith.impl.Less.type.1b517c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b1c747.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.a60f6a.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b1c747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.4df4af.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.892f9f.2: %T.as_type.as.OrderedWith.impl.Greater.type.4df4af.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.10b2de.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d3b(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2af6e8.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.10b2de.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.b8e: %OrderedWith.type.f75 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.eca) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.170: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.b8e) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.df6: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.b8e) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.24c: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.b8e) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.310: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.b8e) [concrete]
// CHECK:STDOUT:   %.356: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.24c, %OrderedWith.facet.b8e [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.0cb14b.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.892f9f.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.793e8a.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.892f9f.2, @T.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.1e56b3.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.793e8a.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.0cb14b.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.892f9f.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.793e8a.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.892f9f.1, @T.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.1e56b3.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.793e8a.2 [concrete]
// CHECK:STDOUT:   %.b2a: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.170, %OrderedWith.facet.b8e [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.ea8939.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.2d3dbd.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.de2542.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.2d3dbd.2, @T.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.a1115a.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.de2542.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.ea8939.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.2d3dbd.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.de2542.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.2d3dbd.1, @T.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.a1115a.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.de2542.2 [concrete]
// CHECK:STDOUT:   %.e96: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.310, %OrderedWith.facet.b8e [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.46e31a.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2af6e8.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.17bde9.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2af6e8.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.ff8bc5.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.17bde9.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.46e31a.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2af6e8.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.17bde9.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2af6e8.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.ff8bc5.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.17bde9.2 [concrete]
// CHECK:STDOUT:   %.93c: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.df6, %OrderedWith.facet.b8e [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.bd23ae.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.a60f6a.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.36fde0.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.a60f6a.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.fbe496.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.36fde0.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.bd23ae.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.a60f6a.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.36fde0.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.a60f6a.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %bound_method.fbe496.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.36fde0.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.adef5f.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.293 = impl_witness_table (%Core.import_ref.adef5f.1), @Core.IntLiteral.as.ImplicitAs.impl.ec9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.566: @T.as_type.as.EqWith.impl.1a3.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.0e80bf.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.1a3.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.a11a61.1)]
// CHECK:STDOUT:   %Core.import_ref.b57: @T.as_type.as.EqWith.impl.1a3.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.182979.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.1a3.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.ac4f99.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.360 = impl_witness_table (%Core.import_ref.566, %Core.import_ref.b57), @T.as_type.as.EqWith.impl.1a3 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.b04: @T.as_type.as.EqWith.impl.1a3.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.182979.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.1a3.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.ac4f99.2)]
// CHECK:STDOUT:   %Core.Equal.883: @T.as_type.as.EqWith.impl.1a3.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.0e80bf.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.1a3.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.a11a61.2)]
// CHECK:STDOUT:   %Core.import_ref.adef5f.2: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.29c = impl_witness_table (%Core.import_ref.adef5f.2), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ae9: @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.8e479f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.6822d0.1)]
// CHECK:STDOUT:   %Core.import_ref.069: @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f0b1d3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1f6607.1)]
// CHECK:STDOUT:   %Core.import_ref.0b2: @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.d820b7.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.803b09.1)]
// CHECK:STDOUT:   %Core.import_ref.d11: @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.42dbc4.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.112484.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.dda = impl_witness_table (%Core.import_ref.ae9, %Core.import_ref.069, %Core.import_ref.0b2, %Core.import_ref.d11), @T.as_type.as.OrderedWith.impl.d3b [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.e59: @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.42dbc4.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.112484.2)]
// CHECK:STDOUT:   %Core.Greater.0f9: @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.d820b7.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.803b09.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.fc1: @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.f0b1d3.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1f6607.2)]
// CHECK:STDOUT:   %Core.Less.add: @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.8e479f.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d3b.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.6822d0.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.83e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc9_21.2
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.367 = impl_witness_access constants.%EqWith.impl_witness.37d, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.33d0bf.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.c2fb87.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.as_type.as.EqWith.impl.Equal.type.50eb4b.1 = specific_constant imports.%Core.Equal.883, @T.as_type.as.EqWith.impl.1a3(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.33d0bf.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.50eb4b.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.33d0bf.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.c2fb87.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long = call %bound_method.loc10_3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %Cpp.long = converted %b.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %a.ref.loc10)
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.50c = impl_witness_access constants.%EqWith.impl_witness.37d, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.30ef12.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.ed888b.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.as_type.as.EqWith.impl.NotEqual.type.989cc9.1 = specific_constant imports.%Core.NotEqual.b04, @T.as_type.as.EqWith.impl.1a3(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.30ef12.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.989cc9.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.30ef12.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.ed888b.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11_3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %Cpp.long = converted %b.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %a.ref.loc11)
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.685 = impl_witness_access constants.%OrderedWith.impl_witness.f27, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.b157e1.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.df8542.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.as_type.as.OrderedWith.impl.Greater.type.d192c2.1 = specific_constant imports.%Core.Greater.0f9, @T.as_type.as.OrderedWith.impl.d3b(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.b157e1.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.d192c2.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.b157e1.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.df8542.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12_3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %Cpp.long = converted %b.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %a.ref.loc12)
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.f18 = impl_witness_access constants.%OrderedWith.impl_witness.f27, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.3482c5.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.33d5cc.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.as_type.as.OrderedWith.impl.Less.type.a0db57.1 = specific_constant imports.%Core.Less.add, @T.as_type.as.OrderedWith.impl.d3b(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.3482c5.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.a0db57.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.3482c5.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.33d5cc.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long = call %bound_method.loc13_3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %Cpp.long = converted %b.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %a.ref.loc13)
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.de3 = impl_witness_access constants.%OrderedWith.impl_witness.f27, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8e9946.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ed31ba.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.288fa6.1 = specific_constant imports.%Core.GreaterOrEquivalent.e59, @T.as_type.as.OrderedWith.impl.d3b(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8e9946.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.288fa6.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8e9946.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ed31ba.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long = call %bound_method.loc14_3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %Cpp.long = converted %b.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %a.ref.loc14)
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.8e2 = impl_witness_access constants.%OrderedWith.impl_witness.f27, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.510b4c.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.71880f.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.82d2f0.1 = specific_constant imports.%Core.LessOrEquivalent.fc1, @T.as_type.as.OrderedWith.impl.d3b(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.510b4c.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.82d2f0.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.510b4c.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.71880f.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long = call %bound_method.loc15_3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %Cpp.long = converted %b.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %a.ref.loc15)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.2c2 = impl_witness_access constants.%EqWith.impl_witness.95f, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.e65d61.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.c5052e.1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.2c5f20.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.bcadad.1]
// CHECK:STDOUT:   %.loc17_5: %T.as_type.as.EqWith.impl.Equal.type.4eea73.1 = specific_constant imports.%Core.Equal.883, @T.as_type.as.EqWith.impl.1a3(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.e65d61.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.4eea73.1 = name_ref Equal, %.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.e65d61.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.c5052e.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.2c5f20.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17 [concrete = constants.%bound_method.bcadad.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.cdc = impl_witness_access constants.%EqWith.impl_witness.95f, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.a73f2e.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.77ff22.1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.4dc8f8.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.a142d4.1]
// CHECK:STDOUT:   %.loc18_5: %T.as_type.as.EqWith.impl.NotEqual.type.728f73.1 = specific_constant imports.%Core.NotEqual.b04, @T.as_type.as.EqWith.impl.1a3(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.a73f2e.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.728f73.1 = name_ref NotEqual, %.loc18_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.a73f2e.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.77ff22.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.4dc8f8.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18 [concrete = constants.%bound_method.a142d4.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.356 = impl_witness_access constants.%OrderedWith.impl_witness.eca, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.892f9f.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.0cb14b.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.793e8a.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.1e56b3.1]
// CHECK:STDOUT:   %.loc19_5: %T.as_type.as.OrderedWith.impl.Greater.type.4df4af.1 = specific_constant imports.%Core.Greater.0f9, @T.as_type.as.OrderedWith.impl.d3b(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.892f9f.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.4df4af.1 = name_ref Greater, %.loc19_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.892f9f.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.0cb14b.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.793e8a.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19 [concrete = constants.%bound_method.1e56b3.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.b2a = impl_witness_access constants.%OrderedWith.impl_witness.eca, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.2d3dbd.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.ea8939.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.de2542.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.a1115a.1]
// CHECK:STDOUT:   %.loc20_5: %T.as_type.as.OrderedWith.impl.Less.type.1b517c.1 = specific_constant imports.%Core.Less.add, @T.as_type.as.OrderedWith.impl.d3b(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.2d3dbd.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.1b517c.1 = name_ref Less, %.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.2d3dbd.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.ea8939.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.de2542.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20 [concrete = constants.%bound_method.a1115a.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.e96 = impl_witness_access constants.%OrderedWith.impl_witness.eca, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2af6e8.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.46e31a.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.17bde9.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.ff8bc5.1]
// CHECK:STDOUT:   %.loc21_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.10b2de.1 = specific_constant imports.%Core.GreaterOrEquivalent.e59, @T.as_type.as.OrderedWith.impl.d3b(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2af6e8.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.10b2de.1 = name_ref GreaterOrEquivalent, %.loc21_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2af6e8.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.46e31a.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.17bde9.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21 [concrete = constants.%bound_method.ff8bc5.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.93c = impl_witness_access constants.%OrderedWith.impl_witness.eca, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.a60f6a.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.bd23ae.1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.36fde0.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.fbe496.1]
// CHECK:STDOUT:   %.loc22_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b1c747.1 = specific_constant imports.%Core.LessOrEquivalent.fc1, @T.as_type.as.OrderedWith.impl.d3b(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.a60f6a.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.b1c747.1 = name_ref LessOrEquivalent, %.loc22_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.a60f6a.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.bd23ae.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.c09) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.36fde0.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22 [concrete = constants.%bound_method.fbe496.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long = call %bound_method.loc22_3(%int_1.loc22) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc22_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc22_3.2: %Cpp.long = converted %int_1.loc22, %.loc22_3.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc22: init bool = call %bound_method.loc22_5.3(%.loc22_3.2, %a.ref.loc22)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- arithmetic_homogeneous_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6fa: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e41: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.293 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f84: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93b, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.3d0: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %Cpp.long, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.5e8: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.597: type = fn_type_with_self_type %Negate.WithSelf.Op.type.5e8, %Negate.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op.type: type = fn_type @Cpp.long.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op: %Cpp.long.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.a0e: type = facet_type <@AddWith, @AddWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.4d0: <witness> = impl_witness imports.%AddWith.impl_witness_table.56c [concrete]
// CHECK:STDOUT:   %AddWith.facet.714: %AddWith.type.a0e = facet_value %Cpp.long, (%AddWith.impl_witness.4d0) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.1fc: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Cpp.long, %AddWith.facet.714) [concrete]
// CHECK:STDOUT:   %.ffe: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.1fc, %AddWith.facet.714 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.type.43d: type = fn_type @Cpp.long.as.AddWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.cc7: %Cpp.long.as.AddWith.impl.Op.type.43d = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.type.9eb: type = facet_type <@SubWith, @SubWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness.a6a: <witness> = impl_witness imports.%SubWith.impl_witness_table.c71 [concrete]
// CHECK:STDOUT:   %SubWith.facet.629: %SubWith.type.9eb = facet_value %Cpp.long, (%SubWith.impl_witness.a6a) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.af8: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Cpp.long, %SubWith.facet.629) [concrete]
// CHECK:STDOUT:   %.942: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.af8, %SubWith.facet.629 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.type.43d: type = fn_type @Cpp.long.as.SubWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.cc7: %Cpp.long.as.SubWith.impl.Op.type.43d = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.type.4b2: type = facet_type <@MulWith, @MulWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness.6fd: <witness> = impl_witness imports.%MulWith.impl_witness_table.2cc [concrete]
// CHECK:STDOUT:   %MulWith.facet.1d5: %MulWith.type.4b2 = facet_value %Cpp.long, (%MulWith.impl_witness.6fd) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.5fb: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Cpp.long, %MulWith.facet.1d5) [concrete]
// CHECK:STDOUT:   %.46b: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.5fb, %MulWith.facet.1d5 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.type.43d: type = fn_type @Cpp.long.as.MulWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.cc7: %Cpp.long.as.MulWith.impl.Op.type.43d = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.type.8f6: type = facet_type <@DivWith, @DivWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness.49c: <witness> = impl_witness imports.%DivWith.impl_witness_table.de3 [concrete]
// CHECK:STDOUT:   %DivWith.facet.453: %DivWith.type.8f6 = facet_value %Cpp.long, (%DivWith.impl_witness.49c) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.fd4: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Cpp.long, %DivWith.facet.453) [concrete]
// CHECK:STDOUT:   %.8d5: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.fd4, %DivWith.facet.453 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.type.43d: type = fn_type @Cpp.long.as.DivWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.cc7: %Cpp.long.as.DivWith.impl.Op.type.43d = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.type.35c: type = facet_type <@ModWith, @ModWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness.d7f: <witness> = impl_witness imports.%ModWith.impl_witness_table.5be [concrete]
// CHECK:STDOUT:   %ModWith.facet.a96: %ModWith.type.35c = facet_value %Cpp.long, (%ModWith.impl_witness.d7f) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.49d: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Cpp.long, %ModWith.facet.a96) [concrete]
// CHECK:STDOUT:   %.d7b: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.49d, %ModWith.facet.a96 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.type.43d: type = fn_type @Cpp.long.as.ModWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.cc7: %Cpp.long.as.ModWith.impl.Op.type.43d = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ade: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.293 = impl_witness_table (%Core.import_ref.ade), @Core.IntLiteral.as.ImplicitAs.impl.ec9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.11dd96.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f63b02.1: %Cpp.long.as.Negate.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.11dd96.1, %Core.import_ref.f63b02.1), @Cpp.long.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.11dd96.4 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f63b02.2: %Cpp.long.as.AddWith.impl.Op.type.43d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.AddWith.impl.Op.cc7]
// CHECK:STDOUT:   %AddWith.impl_witness_table.56c = impl_witness_table (%Core.import_ref.11dd96.4, %Core.import_ref.f63b02.2), @Cpp.long.as.AddWith.impl.a94 [concrete]
// CHECK:STDOUT:   %Core.import_ref.11dd96.7 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f63b02.3: %Cpp.long.as.SubWith.impl.Op.type.43d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.SubWith.impl.Op.cc7]
// CHECK:STDOUT:   %SubWith.impl_witness_table.c71 = impl_witness_table (%Core.import_ref.11dd96.7, %Core.import_ref.f63b02.3), @Cpp.long.as.SubWith.impl.35e [concrete]
// CHECK:STDOUT:   %Core.import_ref.11dd96.10 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f63b02.4: %Cpp.long.as.MulWith.impl.Op.type.43d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.MulWith.impl.Op.cc7]
// CHECK:STDOUT:   %MulWith.impl_witness_table.2cc = impl_witness_table (%Core.import_ref.11dd96.10, %Core.import_ref.f63b02.4), @Cpp.long.as.MulWith.impl.6fa [concrete]
// CHECK:STDOUT:   %Core.import_ref.11dd96.13 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f63b02.5: %Cpp.long.as.DivWith.impl.Op.type.43d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.DivWith.impl.Op.cc7]
// CHECK:STDOUT:   %DivWith.impl_witness_table.de3 = impl_witness_table (%Core.import_ref.11dd96.13, %Core.import_ref.f63b02.5), @Cpp.long.as.DivWith.impl.42b [concrete]
// CHECK:STDOUT:   %Core.import_ref.11dd96.16 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f63b02.6: %Cpp.long.as.ModWith.impl.Op.type.43d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ModWith.impl.Op.cc7]
// CHECK:STDOUT:   %ModWith.impl_witness_table.5be = impl_witness_table (%Core.import_ref.11dd96.16, %Core.import_ref.f63b02.6), @Cpp.long.as.ModWith.impl.d0a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ArithmeticHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.83e = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc8_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc8_21.2: %Cpp.long = converted %int_1.loc8, %.loc8_21.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %x: %Cpp.long = value_binding x, %.loc8_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.83e = value_binding_pattern y [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %y: %Cpp.long = value_binding y, %.loc9_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.83e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc11: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc11: %.597 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Cpp.long.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %x.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op.call: init %Cpp.long = call %bound_method.loc11(%x.ref.loc11)
// CHECK:STDOUT:   %.loc11_20: type = splice_block %long.ref.loc11 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc11: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_28.1: %Cpp.long = value_of_initializer %Cpp.long.as.Negate.impl.Op.call
// CHECK:STDOUT:   %.loc11_28.2: %Cpp.long = converted %Cpp.long.as.Negate.impl.Op.call, %.loc11_28.1
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc11_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.83e = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc12: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc12: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc12: %.ffe = impl_witness_access constants.%AddWith.impl_witness.4d0, element1 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.cc7]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %x.ref.loc12, %impl.elem1.loc12
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.call: init %Cpp.long = call %bound_method.loc12(%x.ref.loc12, %y.ref.loc12)
// CHECK:STDOUT:   %.loc12_20: type = splice_block %long.ref.loc12 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc12: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_30.1: %Cpp.long = value_of_initializer %Cpp.long.as.AddWith.impl.Op.call
// CHECK:STDOUT:   %.loc12_30.2: %Cpp.long = converted %Cpp.long.as.AddWith.impl.Op.call, %.loc12_30.1
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc12_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.83e = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc13: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc13: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc13: %.942 = impl_witness_access constants.%SubWith.impl_witness.a6a, element1 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.cc7]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %x.ref.loc13, %impl.elem1.loc13
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.call: init %Cpp.long = call %bound_method.loc13(%x.ref.loc13, %y.ref.loc13)
// CHECK:STDOUT:   %.loc13_20: type = splice_block %long.ref.loc13 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc13: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_30.1: %Cpp.long = value_of_initializer %Cpp.long.as.SubWith.impl.Op.call
// CHECK:STDOUT:   %.loc13_30.2: %Cpp.long = converted %Cpp.long.as.SubWith.impl.Op.call, %.loc13_30.1
// CHECK:STDOUT:   %c: %Cpp.long = value_binding c, %.loc13_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.83e = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc14: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc14: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc14: %.46b = impl_witness_access constants.%MulWith.impl_witness.6fd, element1 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.cc7]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %x.ref.loc14, %impl.elem1.loc14
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.call: init %Cpp.long = call %bound_method.loc14(%x.ref.loc14, %y.ref.loc14)
// CHECK:STDOUT:   %.loc14_20: type = splice_block %long.ref.loc14 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc14: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_30.1: %Cpp.long = value_of_initializer %Cpp.long.as.MulWith.impl.Op.call
// CHECK:STDOUT:   %.loc14_30.2: %Cpp.long = converted %Cpp.long.as.MulWith.impl.Op.call, %.loc14_30.1
// CHECK:STDOUT:   %d: %Cpp.long = value_binding d, %.loc14_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.83e = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc15: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc15: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc15: %.8d5 = impl_witness_access constants.%DivWith.impl_witness.49c, element1 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.cc7]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %x.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.call: init %Cpp.long = call %bound_method.loc15(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:   %.loc15_20: type = splice_block %long.ref.loc15 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc15: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_30.1: %Cpp.long = value_of_initializer %Cpp.long.as.DivWith.impl.Op.call
// CHECK:STDOUT:   %.loc15_30.2: %Cpp.long = converted %Cpp.long.as.DivWith.impl.Op.call, %.loc15_30.1
// CHECK:STDOUT:   %e: %Cpp.long = value_binding e, %.loc15_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.83e = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc16: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc16: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc16: %.d7b = impl_witness_access constants.%ModWith.impl_witness.d7f, element1 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.cc7]
// CHECK:STDOUT:   %bound_method.loc16: <bound method> = bound_method %x.ref.loc16, %impl.elem1.loc16
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.call: init %Cpp.long = call %bound_method.loc16(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:   %.loc16_20: type = splice_block %long.ref.loc16 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc16: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_30.1: %Cpp.long = value_of_initializer %Cpp.long.as.ModWith.impl.Op.call
// CHECK:STDOUT:   %.loc16_30.2: %Cpp.long = converted %Cpp.long.as.ModWith.impl.Op.call, %.loc16_30.1
// CHECK:STDOUT:   %f: %Cpp.long = value_binding f, %.loc16_30.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_homogeneous_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6fa: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f49: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.5ec: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.f49) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.921: %T.as_type.as.ImplicitAs.impl.Convert.type.5ec = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e41: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.293 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c09: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %.f84: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93b, %ImplicitAs.facet.c09 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.3d0: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.ecc: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %U.887: %ImplicitAs.type.6fa = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c37e49.1: %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c37e49.2: %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.553: <witness> = impl_witness imports.%Copy.impl_witness_table.256 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.553) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.998: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c42, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3e9: %ImplicitAs.type.6fa = facet_value %Cpp.long, (%ImplicitAs.impl_witness.998) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.830: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.d4d, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.ba770b.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.ce7855.1: %Cpp.long.as.AddAssignWith.impl.Op.type.ba770b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.ba770b.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.ce7855.2: %Cpp.long.as.AddAssignWith.impl.Op.type.ba770b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.fc6: %AddAssignWith.type.ecc = facet_value %Cpp.long, (%AddAssignWith.impl_witness.830) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.874: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet.fc6) [concrete]
// CHECK:STDOUT:   %.5f0: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.874, %AddAssignWith.facet.fc6 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.544ffe.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.ce7855.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.544ffe.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.ce7855.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.b0c: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.d98f7d.1: %Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.d98f7d.2: %Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.d79: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.c70, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.19fb65.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.498a54.1: %Cpp.long.as.SubAssignWith.impl.Op.type.19fb65.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.19fb65.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.498a54.2: %Cpp.long.as.SubAssignWith.impl.Op.type.19fb65.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.d27: %SubAssignWith.type.b0c = facet_value %Cpp.long, (%SubAssignWith.impl_witness.d79) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.ebd: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long, %SubAssignWith.facet.d27) [concrete]
// CHECK:STDOUT:   %.700: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.ebd, %SubAssignWith.facet.d27 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.e00b17.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.498a54.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.e00b17.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.498a54.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.6f5: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.9be123.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.4d4f44.1: %Cpp.long.as.MulAssignWith.impl.Op.type.9be123.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.9be123.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.4d4f44.2: %Cpp.long.as.MulAssignWith.impl.Op.type.9be123.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.e21: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.511, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.23c32e.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.9be979.1: %Cpp.long.as.MulAssignWith.impl.Op.type.23c32e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.23c32e.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.9be979.2: %Cpp.long.as.MulAssignWith.impl.Op.type.23c32e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.457: %MulAssignWith.type.6f5 = facet_value %Cpp.long, (%MulAssignWith.impl_witness.e21) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.c2c: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long, %MulAssignWith.facet.457) [concrete]
// CHECK:STDOUT:   %.756: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.c2c, %MulAssignWith.facet.457 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.6c2d47.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.9be979.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.6c2d47.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.9be979.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.9e4: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.b4447c.1: %Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.b4447c.2: %Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.9e8: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.b74, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.a9acbb.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.60ccc0.1: %Cpp.long.as.DivAssignWith.impl.Op.type.a9acbb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.a9acbb.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.60ccc0.2: %Cpp.long.as.DivAssignWith.impl.Op.type.a9acbb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.810: %DivAssignWith.type.9e4 = facet_value %Cpp.long, (%DivAssignWith.impl_witness.9e8) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.53b: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long, %DivAssignWith.facet.810) [concrete]
// CHECK:STDOUT:   %.b84: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.53b, %DivAssignWith.facet.810 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.04e150.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.60ccc0.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.04e150.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.60ccc0.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.20c: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.ff452e.1: %Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.ff452e.2: %Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.574: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.fb8, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.837c3c.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.401e7e.1: %Cpp.long.as.ModAssignWith.impl.Op.type.837c3c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.837c3c.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.401e7e.2: %Cpp.long.as.ModAssignWith.impl.Op.type.837c3c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.d6b: %ModAssignWith.type.20c = facet_value %Cpp.long, (%ModAssignWith.impl_witness.574) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.948: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long, %ModAssignWith.facet.d6b) [concrete]
// CHECK:STDOUT:   %.e3a: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.948, %ModAssignWith.facet.d6b [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.cb024b.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.401e7e.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.cb024b.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.401e7e.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.0cc: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.b72f56.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.b72f56.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.32b: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.38e, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.f1429d.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.efe73c.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.f1429d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.f1429d.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.efe73c.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.f1429d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.c27: %BitAndAssignWith.type.0cc = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.32b) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.3d7: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long, %BitAndAssignWith.facet.c27) [concrete]
// CHECK:STDOUT:   %.521: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.3d7, %BitAndAssignWith.facet.c27 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.bd1e6a.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.efe73c.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.bd1e6a.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.efe73c.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.d61: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.169685.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.169685.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.a4d: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.033, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.fbf7ed.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.732095.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fbf7ed.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.fbf7ed.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.732095.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fbf7ed.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.03e: %BitOrAssignWith.type.d61 = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.a4d) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.964: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long, %BitOrAssignWith.facet.03e) [concrete]
// CHECK:STDOUT:   %.7dc: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.964, %BitOrAssignWith.facet.03e [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.67bf91.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.732095.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.67bf91.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.732095.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.f85: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.583c20.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.583c20.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.07c: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.443, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.6a0f8a.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.9bad15.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6a0f8a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.6a0f8a.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.9bad15.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6a0f8a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.b38: %BitXorAssignWith.type.f85 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.07c) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.959: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long, %BitXorAssignWith.facet.b38) [concrete]
// CHECK:STDOUT:   %.9c9: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.959, %BitXorAssignWith.facet.b38 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.db7227.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.9bad15.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.db7227.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.9bad15.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.cb1: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.2764b8.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.2764b8.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.695: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.89e, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.809b36.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e1af3f.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.809b36.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.809b36.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e1af3f.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.809b36.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.2ff: %LeftShiftAssignWith.type.cb1 = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.695) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.ccd: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long, %LeftShiftAssignWith.facet.2ff) [concrete]
// CHECK:STDOUT:   %.55f: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.ccd, %LeftShiftAssignWith.facet.2ff [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.176c12.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.e1af3f.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.176c12.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.e1af3f.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.eeb: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.723f9d.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.723f9d.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.52b: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.eba, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7abe8a.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.d21be7.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7abe8a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7abe8a.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.d21be7.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7abe8a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.a7c: %RightShiftAssignWith.type.eeb = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.52b) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.de5: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long, %RightShiftAssignWith.facet.a7c) [concrete]
// CHECK:STDOUT:   %.9cb: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.de5, %RightShiftAssignWith.facet.a7c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.576b1b.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.d21be7.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.576b1b.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.d21be7.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.3e9) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.3b1: @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert.type (%T.as_type.as.ImplicitAs.impl.Convert.type.5ec) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert (constants.%T.as_type.as.ImplicitAs.impl.Convert.921)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c42 = impl_witness_table (%Core.import_ref.3b1), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ade: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.293 = impl_witness_table (%Core.import_ref.ade), @Core.IntLiteral.as.ImplicitAs.impl.ec9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d2: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.c37e49.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.d4d = impl_witness_table (%Core.import_ref.1d2), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.e4e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.c37e49.2)]
// CHECK:STDOUT:   %Core.import_ref.f63: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.256 = impl_witness_table (%Core.import_ref.f63), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1bc: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.d98f7d.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.c70 = impl_witness_table (%Core.import_ref.1bc), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.792: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.d98f7d.2)]
// CHECK:STDOUT:   %Core.import_ref.ae2: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.9be123.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.4d4f44.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.511 = impl_witness_table (%Core.import_ref.ae2), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b40: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.9be123.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.4d4f44.2)]
// CHECK:STDOUT:   %Core.import_ref.49d: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.b4447c.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.b74 = impl_witness_table (%Core.import_ref.49d), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.6ac: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.b4447c.2)]
// CHECK:STDOUT:   %Core.import_ref.faa: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.ff452e.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.fb8 = impl_witness_table (%Core.import_ref.faa), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.264: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.ff452e.2)]
// CHECK:STDOUT:   %Core.import_ref.df6: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.b72f56.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.38e = impl_witness_table (%Core.import_ref.df6), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.27c: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.b72f56.2)]
// CHECK:STDOUT:   %Core.import_ref.c64: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.169685.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.033 = impl_witness_table (%Core.import_ref.c64), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.dda: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.169685.2)]
// CHECK:STDOUT:   %Core.import_ref.ac7: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.583c20.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.443 = impl_witness_table (%Core.import_ref.ac7), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c2e: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.583c20.2)]
// CHECK:STDOUT:   %Core.import_ref.0b9: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2764b8.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.89e = impl_witness_table (%Core.import_ref.0b9), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.e49: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2764b8.2)]
// CHECK:STDOUT:   %Core.import_ref.8a6: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.723f9d.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.eba = impl_witness_table (%Core.import_ref.8a6), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d53: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.723f9d.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.83e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.83e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.83e = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.83e = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var %b.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   assign %b.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = ref_binding b, %b.var
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11: %.5f0 = impl_witness_access constants.%AddAssignWith.impl_witness.830, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.ce7855.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.6fa = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.998) [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.6fa = converted constants.%Cpp.long, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.544ffe.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.ba770b.1 = specific_constant imports.%Core.Op.e4e, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.ce7855.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.AddAssignWith.impl.Op.type.ba770b.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.ce7855.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.544ffe.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.700 = impl_witness_access constants.%SubAssignWith.impl_witness.d79, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.498a54.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.6fa = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.998) [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.6fa = converted constants.%Cpp.long, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.e00b17.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.SubAssignWith.impl.Op.type.19fb65.1 = specific_constant imports.%Core.Op.792, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.498a54.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.SubAssignWith.impl.Op.type.19fb65.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.498a54.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.e00b17.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.756 = impl_witness_access constants.%MulAssignWith.impl_witness.e21, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.9be979.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.6fa = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.998) [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.6fa = converted constants.%Cpp.long, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.6c2d47.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.23c32e.1 = specific_constant imports.%Core.Op.b40, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.9be979.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.MulAssignWith.impl.Op.type.23c32e.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.9be979.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.6c2d47.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.b84 = impl_witness_access constants.%DivAssignWith.impl_witness.9e8, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.60ccc0.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.6fa = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.998) [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.6fa = converted constants.%Cpp.long, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.04e150.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.DivAssignWith.impl.Op.type.a9acbb.1 = specific_constant imports.%Core.Op.6ac, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.60ccc0.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.DivAssignWith.impl.Op.type.a9acbb.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.60ccc0.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.04e150.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.e3a = impl_witness_access constants.%ModAssignWith.impl_witness.574, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.401e7e.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.6fa = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.998) [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.6fa = converted constants.%Cpp.long, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.cb024b.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.837c3c.1 = specific_constant imports.%Core.Op.264, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.401e7e.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long.as.ModAssignWith.impl.Op.type.837c3c.1 = name_ref Op, %.loc15_5.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.401e7e.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.cb024b.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.521 = impl_witness_access constants.%BitAndAssignWith.impl_witness.32b, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.efe73c.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.6fa = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.998) [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.6fa = converted constants.%Cpp.long, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.bd1e6a.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.f1429d.1 = specific_constant imports.%Core.Op.27c, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.efe73c.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitAndAssignWith.impl.Op.type.f1429d.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.efe73c.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.bd1e6a.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.7dc = impl_witness_access constants.%BitOrAssignWith.impl_witness.a4d, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.732095.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.6fa = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.998) [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.6fa = converted constants.%Cpp.long, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.67bf91.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fbf7ed.1 = specific_constant imports.%Core.Op.dda, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.732095.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fbf7ed.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.732095.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.67bf91.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.9c9 = impl_witness_access constants.%BitXorAssignWith.impl_witness.07c, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.9bad15.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.6fa = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.998) [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.6fa = converted constants.%Cpp.long, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.db7227.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6a0f8a.1 = specific_constant imports.%Core.Op.c2e, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.9bad15.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6a0f8a.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.9bad15.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.db7227.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.55f = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.695, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e1af3f.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.6fa = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.998) [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.6fa = converted constants.%Cpp.long, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.176c12.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.809b36.1 = specific_constant imports.%Core.Op.e49, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e1af3f.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.809b36.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e1af3f.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.176c12.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.9cb = impl_witness_access constants.%RightShiftAssignWith.impl_witness.52b, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.d21be7.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.6fa = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.998) [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.6fa = converted constants.%Cpp.long, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.3e9]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.576b1b.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7abe8a.1 = specific_constant imports.%Core.Op.d53, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.d21be7.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7abe8a.1 = name_ref Op, %.loc21_5.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.d21be7.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3e9) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.576b1b.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_9)
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_hereogeneous_long_and_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6fa: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e41: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.293 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c09: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %.f84: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93b, %ImplicitAs.facet.c09 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.3d0: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.e37: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %U.887: %ImplicitAs.type.6fa = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c37e49.1: %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c37e49.2: %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.72d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.29c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0fb: %ImplicitAs.type.6fa = facet_value %i32, (%ImplicitAs.impl_witness.72d) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.129: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.d4d, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.1: %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.2: %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.627: %AddAssignWith.type.e37 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.129) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.839: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.627) [concrete]
// CHECK:STDOUT:   %.72d: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.839, %AddAssignWith.facet.627 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.548e6a.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.548e6a.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.db6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %.9d6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.db6, %ImplicitAs.facet.0fb [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.227: type = facet_type <@SubAssignWith, @SubAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.d98f7d.1: %Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.d98f7d.2: %Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.f18: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.c70, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.df5f14.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.0d59b4.1: %Cpp.long.as.SubAssignWith.impl.Op.type.df5f14.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.df5f14.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.0d59b4.2: %Cpp.long.as.SubAssignWith.impl.Op.type.df5f14.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.a58: %SubAssignWith.type.227 = facet_value %Cpp.long, (%SubAssignWith.impl_witness.f18) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.5bc: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i32, %SubAssignWith.facet.a58) [concrete]
// CHECK:STDOUT:   %.1e1: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.5bc, %SubAssignWith.facet.a58 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.de7e80.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.0d59b4.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.de7e80.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.0d59b4.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.e08: type = facet_type <@MulAssignWith, @MulAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.9be123.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.4d4f44.1: %Cpp.long.as.MulAssignWith.impl.Op.type.9be123.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.9be123.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.4d4f44.2: %Cpp.long.as.MulAssignWith.impl.Op.type.9be123.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.19a: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.511, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.779f22.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.5c5e76.1: %Cpp.long.as.MulAssignWith.impl.Op.type.779f22.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.779f22.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.5c5e76.2: %Cpp.long.as.MulAssignWith.impl.Op.type.779f22.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.b4d: %MulAssignWith.type.e08 = facet_value %Cpp.long, (%MulAssignWith.impl_witness.19a) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.5e3: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i32, %MulAssignWith.facet.b4d) [concrete]
// CHECK:STDOUT:   %.cc6: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.5e3, %MulAssignWith.facet.b4d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.d36f7a.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.5c5e76.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.d36f7a.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.5c5e76.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.1e4: type = facet_type <@DivAssignWith, @DivAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.b4447c.1: %Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.b4447c.2: %Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.5ea: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.b74, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.a25461.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.898613.1: %Cpp.long.as.DivAssignWith.impl.Op.type.a25461.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.a25461.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.898613.2: %Cpp.long.as.DivAssignWith.impl.Op.type.a25461.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.7f9: %DivAssignWith.type.1e4 = facet_value %Cpp.long, (%DivAssignWith.impl_witness.5ea) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.597: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i32, %DivAssignWith.facet.7f9) [concrete]
// CHECK:STDOUT:   %.880: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.597, %DivAssignWith.facet.7f9 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.bac757.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.898613.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.bac757.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.898613.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.4f9: type = facet_type <@ModAssignWith, @ModAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.ff452e.1: %Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.ff452e.2: %Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.866: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.fb8, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.fbd100.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.970296.1: %Cpp.long.as.ModAssignWith.impl.Op.type.fbd100.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.fbd100.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.970296.2: %Cpp.long.as.ModAssignWith.impl.Op.type.fbd100.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.159: %ModAssignWith.type.4f9 = facet_value %Cpp.long, (%ModAssignWith.impl_witness.866) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.031: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i32, %ModAssignWith.facet.159) [concrete]
// CHECK:STDOUT:   %.3ec: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.031, %ModAssignWith.facet.159 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.acff4e.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.970296.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.acff4e.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.970296.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.e52: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.b72f56.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.b72f56.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.a7a: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.38e, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.cbeec5.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.116220.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.cbeec5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.cbeec5.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.116220.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.cbeec5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.241: %BitAndAssignWith.type.e52 = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.a7a) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.45a: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i32, %BitAndAssignWith.facet.241) [concrete]
// CHECK:STDOUT:   %.345: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.45a, %BitAndAssignWith.facet.241 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.324142.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.116220.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.324142.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.116220.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.db1: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.169685.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.169685.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.4c0: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.033, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.1d56d4.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.9f5cee.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.1d56d4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.1d56d4.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.9f5cee.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.1d56d4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.a3f: %BitOrAssignWith.type.db1 = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.4c0) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.02c: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i32, %BitOrAssignWith.facet.a3f) [concrete]
// CHECK:STDOUT:   %.490: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.02c, %BitOrAssignWith.facet.a3f [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.e27b36.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.9f5cee.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.e27b36.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.9f5cee.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.fae: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.583c20.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.583c20.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.f5d: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.443, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.ef9e9f.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.fb73fa.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.ef9e9f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.ef9e9f.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.fb73fa.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.ef9e9f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.947: %BitXorAssignWith.type.fae = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.f5d) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.14b: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i32, %BitXorAssignWith.facet.947) [concrete]
// CHECK:STDOUT:   %.87e: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.14b, %BitXorAssignWith.facet.947 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.6900f6.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.fb73fa.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.6900f6.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.fb73fa.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.8b2: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.2764b8.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.2764b8.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.aae: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.89e, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bed071.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.a3f1fe.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bed071.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bed071.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.a3f1fe.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bed071.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.518: %LeftShiftAssignWith.type.8b2 = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.aae) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.386: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i32, %LeftShiftAssignWith.facet.518) [concrete]
// CHECK:STDOUT:   %.2ae: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.386, %LeftShiftAssignWith.facet.518 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.207d23.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.a3f1fe.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.207d23.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.a3f1fe.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.62d: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.723f9d.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.723f9d.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.20f: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.eba, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.40d73d.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.07491d.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.40d73d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.40d73d.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.07491d.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.40d73d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.a755: %RightShiftAssignWith.type.62d = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.20f) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.318: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i32, %RightShiftAssignWith.facet.a755) [concrete]
// CHECK:STDOUT:   %.585: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.318, %RightShiftAssignWith.facet.a755 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.d8b850.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.07491d.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.d8b850.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.07491d.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.adef5f.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.293 = impl_witness_table (%Core.import_ref.adef5f.1), @Core.IntLiteral.as.ImplicitAs.impl.ec9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d2: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.c37e49.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.d4d = impl_witness_table (%Core.import_ref.1d2), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.e4e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.c37e49.2)]
// CHECK:STDOUT:   %Core.import_ref.adef5f.2: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.29c = impl_witness_table (%Core.import_ref.adef5f.2), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1bc: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.d98f7d.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.c70 = impl_witness_table (%Core.import_ref.1bc), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.792: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.e6fddb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.d98f7d.2)]
// CHECK:STDOUT:   %Core.import_ref.ae2: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.9be123.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.4d4f44.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.511 = impl_witness_table (%Core.import_ref.ae2), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b40: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.9be123.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.4d4f44.2)]
// CHECK:STDOUT:   %Core.import_ref.49d: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.b4447c.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.b74 = impl_witness_table (%Core.import_ref.49d), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.6ac: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.109ac5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.b4447c.2)]
// CHECK:STDOUT:   %Core.import_ref.faa: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.ff452e.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.fb8 = impl_witness_table (%Core.import_ref.faa), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.264: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.e961bf.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.ff452e.2)]
// CHECK:STDOUT:   %Core.import_ref.df6: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.b72f56.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.38e = impl_witness_table (%Core.import_ref.df6), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.27c: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.f954e6.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.b72f56.2)]
// CHECK:STDOUT:   %Core.import_ref.c64: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.169685.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.033 = impl_witness_table (%Core.import_ref.c64), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.dda: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.186a1d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.169685.2)]
// CHECK:STDOUT:   %Core.import_ref.ac7: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.583c20.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.443 = impl_witness_table (%Core.import_ref.ac7), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c2e: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.cdf2c9.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.583c20.2)]
// CHECK:STDOUT:   %Core.import_ref.0b9: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2764b8.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.89e = impl_witness_table (%Core.import_ref.0b9), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.e49: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.50a853.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2764b8.2)]
// CHECK:STDOUT:   %Core.import_ref.8a6: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.723f9d.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.eba = impl_witness_table (%Core.import_ref.8a6), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d53: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.053d7a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.723f9d.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndI32() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.83e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.83e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.72d = impl_witness_access constants.%AddAssignWith.impl_witness.129, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.548e6a.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.1 = specific_constant imports.%Core.Op.e4e, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.548e6a.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.1e1 = impl_witness_access constants.%SubAssignWith.impl_witness.f18, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.0d59b4.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.de7e80.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.SubAssignWith.impl.Op.type.df5f14.1 = specific_constant imports.%Core.Op.792, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.0d59b4.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.SubAssignWith.impl.Op.type.df5f14.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.0d59b4.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.de7e80.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.cc6 = impl_witness_access constants.%MulAssignWith.impl_witness.19a, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.5c5e76.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.d36f7a.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.779f22.1 = specific_constant imports.%Core.Op.b40, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.5c5e76.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.MulAssignWith.impl.Op.type.779f22.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.5c5e76.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.d36f7a.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_8.2: %Cpp.long = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.880 = impl_witness_access constants.%DivAssignWith.impl_witness.5ea, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.898613.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.bac757.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.DivAssignWith.impl.Op.type.a25461.1 = specific_constant imports.%Core.Op.6ac, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.898613.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.DivAssignWith.impl.Op.type.a25461.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.898613.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.bac757.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_8.2: %Cpp.long = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.3ec = impl_witness_access constants.%ModAssignWith.impl_witness.866, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.970296.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.acff4e.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.fbd100.1 = specific_constant imports.%Core.Op.264, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.970296.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.ModAssignWith.impl.Op.type.fbd100.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.970296.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.acff4e.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5: %.345 = impl_witness_access constants.%BitAndAssignWith.impl_witness.a7a, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.116220.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc16: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc16 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.324142.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.cbeec5.1 = specific_constant imports.%Core.Op.27c, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.116220.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long.as.BitAndAssignWith.impl.Op.type.cbeec5.1 = name_ref Op, %.loc16_5.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.116220.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.324142.2]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %a.ref.loc16, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16: init %Cpp.long = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.3(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.490 = impl_witness_access constants.%BitOrAssignWith.impl_witness.4c0, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9f5cee.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.e27b36.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.1d56d4.1 = specific_constant imports.%Core.Op.dda, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9f5cee.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitOrAssignWith.impl.Op.type.1d56d4.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9f5cee.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.e27b36.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.87e = impl_witness_access constants.%BitXorAssignWith.impl_witness.f5d, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.fb73fa.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.6900f6.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.ef9e9f.1 = specific_constant imports.%Core.Op.c2e, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.fb73fa.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitXorAssignWith.impl.Op.type.ef9e9f.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.fb73fa.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.6900f6.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc18
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.2ae = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.aae, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.a3f1fe.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.207d23.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bed071.1 = specific_constant imports.%Core.Op.e49, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.a3f1fe.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bed071.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.a3f1fe.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.207d23.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc19
// CHECK:STDOUT:   %.loc19_9.2: %Cpp.long = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.585 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.20f, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.07491d.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.d8b850.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.40d73d.1 = specific_constant imports.%Core.Op.d53, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.07491d.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.40d73d.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.07491d.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.d8b850.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_9: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_9
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_9(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_9.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc20
// CHECK:STDOUT:   %.loc20_9.2: %Cpp.long = converted %b.ref.loc20, %.loc20_9.1
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_and_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6fa: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e41: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.293 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f84: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93b, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.3d0: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.d09: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.887: %ImplicitAs.type.6fa = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c37e49.1: %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c37e49.2: %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.378: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.d4d, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.e519c0.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.043422.1: %Cpp.long.as.AddAssignWith.impl.Op.type.e519c0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.e519c0.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.043422.2: %Cpp.long.as.AddAssignWith.impl.Op.type.e519c0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.b24: %AddAssignWith.type.d09 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.378) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.697: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet.b24) [concrete]
// CHECK:STDOUT:   %.f6c: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.697, %AddAssignWith.facet.b24 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c44639.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.043422.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c44639.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.043422.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ade: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.293 = impl_witness_table (%Core.import_ref.ade), @Core.IntLiteral.as.ImplicitAs.impl.ec9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d2: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.c37e49.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.d4d = impl_witness_table (%Core.import_ref.1d2), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.e4e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.c37e49.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.83e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.83e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9_5: %.f6c = impl_witness_access constants.%AddAssignWith.impl_witness.378, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.043422.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.e41) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.6fa = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c44639.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.e519c0.1 = specific_constant imports.%Core.Op.e4e, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.043422.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.e519c0.1 = name_ref Op, %.loc9_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.043422.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c44639.2]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_8: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9_8(%int_1.loc9) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc9_8.2: %Cpp.long = converted %int_1.loc9, %.loc9_8.1 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_5.3(%a.ref, %.loc9_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_and_runtime_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6fa: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e41: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.293 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c09: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.c09) [concrete]
// CHECK:STDOUT:   %.f84: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93b, %ImplicitAs.facet.c09 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.3d0: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.4e8(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl.4e8(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.4e8(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.840: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.840) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet.840 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d9c: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.e37: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %U.887: %ImplicitAs.type.6fa = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c37e49.1: %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.887) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c37e49.2: %Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.72d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.29c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0fb: %ImplicitAs.type.6fa = facet_value %i32, (%ImplicitAs.impl_witness.72d) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.129: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.d4d, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.1: %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.2: %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.627: %AddAssignWith.type.e37 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.129) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.839: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.627) [concrete]
// CHECK:STDOUT:   %.72d: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.839, %AddAssignWith.facet.627 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.548e6a.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.548e6a.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.db6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.0fb) [concrete]
// CHECK:STDOUT:   %.9d6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.db6, %ImplicitAs.facet.0fb [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.adef5f.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.293 = impl_witness_table (%Core.import_ref.adef5f.1), @Core.IntLiteral.as.ImplicitAs.impl.ec9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.4e8.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.4e8.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl.4e8 [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d2: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.c37e49.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.d4d = impl_witness_table (%Core.import_ref.1d2), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.e4e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.ee98fc.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.c37e49.2)]
// CHECK:STDOUT:   %Core.import_ref.adef5f.2: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.29c = impl_witness_table (%Core.import_ref.adef5f.2), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndRuntimeI32() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.83e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.83e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.104 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc9: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_16.2(%int_1.loc9) [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc9_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc9_16.2: %i32 = converted %int_1.loc9, %.loc9_16.1 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %b: %i32 = value_binding b, %.loc9_16.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.72d = impl_witness_access constants.%AddAssignWith.impl_witness.129, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6fa = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.72d) [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.6fa = converted constants.%i32, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet.0fb]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.548e6a.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.1 = specific_constant imports.%Core.Op.e4e, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.69179c.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.4ab9dd.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.0fb) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.548e6a.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.9d6 = impl_witness_access constants.%ImplicitAs.impl_witness.72d, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc10_8(%b.ref)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref, %.loc10_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- increment_decrement_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6fa: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e41: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.293 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6fa = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f84: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93b, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.3d0: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.c20: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.b8b: type = fn_type_with_self_type %Inc.WithSelf.Op.type.c20, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.type: type = fn_type @Cpp.long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op: %Cpp.long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.de7: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.f14: type = fn_type_with_self_type %Dec.WithSelf.Op.type.de7, %Dec.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.type: type = fn_type @Cpp.long.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op: %Cpp.long.as.Dec.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ade: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.293 = impl_witness_table (%Core.import_ref.ade), @Core.IntLiteral.as.ImplicitAs.impl.ec9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.f63b02.1: %Cpp.long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.f63b02.1), @Cpp.long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f63b02.2: %Cpp.long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.f63b02.2), @Cpp.long.as.Dec.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IncrementDecrementLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.83e = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.83e = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.f84 = impl_witness_access constants.%ImplicitAs.impl_witness.e41, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.3d0]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc9: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc9: %.b8b = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.f14 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10(%a.ref.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %Cpp.long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsigned_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long: type = class_type @ULong32 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.9f4: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.754: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.754 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ae2: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ace: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1e5: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a9a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5ba: %ImplicitAs.type.ae2 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.1e5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.57e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet.5ba) [concrete]
// CHECK:STDOUT:   %.c47: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.57e, %ImplicitAs.facet.5ba [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.3ef: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongResult: type = class_type @ULongResult [concrete]
// CHECK:STDOUT:   %pattern_type.38e: type = pattern_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULong.cpp_overload_set.type: type = cpp_overload_set_type @PassULong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassULong.cpp_overload_set.value: %PassULong.cpp_overload_set.type = cpp_overload_set_value @PassULong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.54a: type = ptr_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.type.f4cf84.1: type = fn_type @PassULong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.8dc497.1: %PassULong__carbon_thunk.type.f4cf84.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.126: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b0a: type = facet_type <@ImplicitAs, @ImplicitAs(%u32)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.eb6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.501(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.498: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.eb6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.040: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0c6, @Core.IntLiteral.as.ImplicitAs.impl.501(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.34d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.501(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.09f: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.34d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.1fc: %ImplicitAs.type.b0a = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.040) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9e6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u32, %ImplicitAs.facet.1fc) [concrete]
// CHECK:STDOUT:   %.955: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9e6, %ImplicitAs.facet.1fc [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.be4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.09f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.09f, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.591: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.ce2: %u32 = int_value 1 [concrete]
// CHECK:STDOUT:   %UIntResult: type = class_type @UIntResult [concrete]
// CHECK:STDOUT:   %pattern_type.5a5: type = pattern_type %UIntResult [concrete]
// CHECK:STDOUT:   %ptr.c09: type = ptr_type %UIntResult [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.type.f4cf84.2: type = fn_type @PassULong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.8dc497.2: %PassULong__carbon_thunk.type.f4cf84.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.a26: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.909: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.909 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.6d7: type = facet_type <@As, @As(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.314: <witness> = impl_witness imports.%As.impl_witness_table.e82 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.6d7 = facet_value Core.IntLiteral, (%As.impl_witness.314) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.981: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.f45: type = fn_type_with_self_type %As.WithSelf.Convert.type.981, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6d1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.245 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.677: %ImplicitAs.type.ace = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.6d1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a12: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.677) [concrete]
// CHECK:STDOUT:   %.165: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a12, %ImplicitAs.facet.677 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.3ef, %Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.a26 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.aab: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d0d: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.a26)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.539: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.65d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.234, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.82a: %ImplicitAs.type.d0d = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.65d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c87: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.a26, %ImplicitAs.facet.82a) [concrete]
// CHECK:STDOUT:   %.3a8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c87, %ImplicitAs.facet.82a [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.aaa: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.2ec: %f32.a26 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.2ec) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc24_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT:   %UIntResult.cpp_destructor.type: type = fn_type @UIntResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %UIntResult.cpp_destructor: %UIntResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor.type: type = fn_type @ULongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor: %ULongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .CppCompat = %CppCompat.463205.1
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     .ULongResult = %ULongResult.decl
// CHECK:STDOUT:     .PassULong = %PassULong.cpp_overload_set.value
// CHECK:STDOUT:     .UIntResult = %UIntResult.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.463205.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .ULong32 = %Core.ULong32
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ULong32: type = import_ref Core//prelude/types/cpp/int, ULong32, loaded [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.754 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.adef5f.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a9a = impl_witness_table (%Core.import_ref.adef5f.1), @Core.IntLiteral.as.ImplicitAs.impl.160 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULong.cpp_overload_set.value: %PassULong.cpp_overload_set.type = cpp_overload_set_value @PassULong.cpp_overload_set [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULong__carbon_thunk.decl.43aa98.1: %PassULong__carbon_thunk.type.f4cf84.1 = fn_decl @PassULong__carbon_thunk.1 [concrete = constants.%PassULong__carbon_thunk.8dc497.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.e9f: @Core.IntLiteral.as.ImplicitAs.impl.501.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.eb6) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.501.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.498)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0c6 = impl_witness_table (%Core.import_ref.e9f), @Core.IntLiteral.as.ImplicitAs.impl.501 [concrete]
// CHECK:STDOUT:   %UIntResult.decl: type = class_decl @UIntResult [concrete = constants.%UIntResult] {} {}
// CHECK:STDOUT:   %PassULong__carbon_thunk.decl.43aa98.2: %PassULong__carbon_thunk.type.f4cf84.2 = fn_decl @PassULong__carbon_thunk.2 [concrete = constants.%PassULong__carbon_thunk.8dc497.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.909 = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.adef5f.2: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.e82 = impl_witness_table (%Core.import_ref.adef5f.2), @Core.IntLiteral.as.As.impl.51f [concrete]
// CHECK:STDOUT:   %Core.import_ref.adef5f.3: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.245 = impl_witness_table (%Core.import_ref.adef5f.3), @Cpp.unsigned_long.as.ImplicitAs.impl.b42 [concrete]
// CHECK:STDOUT:   %Core.import_ref.63cb: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.539)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.234 = impl_witness_table (%Core.import_ref.63cb), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.9f4 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_29: type = splice_block %unsigned_long.ref.loc15 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc15: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.c47 = impl_witness_access constants.%ImplicitAs.impl_witness.1e5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.unsigned_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.3ef]
// CHECK:STDOUT:   %.loc15_46.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.3ef]
// CHECK:STDOUT:   %.loc15_46.2: %Cpp.unsigned_long = converted %int_1.loc15, %.loc15_46.1 [concrete = constants.%int_1.3ef]
// CHECK:STDOUT:   %cpp_unsigned_long: %Cpp.unsigned_long = value_binding cpp_unsigned_long, %.loc15_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_result.patt: %pattern_type.38e = value_binding_pattern cpp_unsigned_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_58: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc16: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc16: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc16_89.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.54a = addr_of %.loc16_89.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.43aa98.1(%cpp_unsigned_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_89.2: init %ULongResult to %.loc16_89.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_43: type = splice_block %ULongResult.ref.loc16 [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_40: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref.loc16: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_89.3: ref %ULongResult = temporary %.loc16_89.1, %.loc16_89.2
// CHECK:STDOUT:   %.loc16_89.4: %ULongResult = acquire_value %.loc16_89.3
// CHECK:STDOUT:   %cpp_unsigned_long_result: %ULongResult = value_binding cpp_unsigned_long_result, %.loc16_89.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong.patt: %pattern_type.9f4 = value_binding_pattern cpp_compat_ulong [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc18: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc18: type = splice_block %ULong32.ref [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.463205.1 [concrete = imports.%CppCompat.463205.1]
// CHECK:STDOUT:     %ULong32.ref: type = name_ref ULong32, imports.%Core.ULong32 [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_ulong: %Cpp.unsigned_long = value_binding cpp_compat_ulong, %cpp_unsigned_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_result.patt: %pattern_type.38e = value_binding_pattern cpp_compat_ulong_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc19: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong.ref: %Cpp.unsigned_long = name_ref cpp_compat_ulong, %cpp_compat_ulong
// CHECK:STDOUT:   %.loc19_87.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.54a = addr_of %.loc19_87.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.43aa98.1(%cpp_compat_ulong.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_87.2: init %ULongResult to %.loc19_87.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_42: type = splice_block %ULongResult.ref.loc19 [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_39: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref.loc19: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_87.3: ref %ULongResult = temporary %.loc19_87.1, %.loc19_87.2
// CHECK:STDOUT:   %.loc19_87.4: %ULongResult = acquire_value %.loc19_87.3
// CHECK:STDOUT:   %cpp_compat_ulong_result: %ULongResult = value_binding cpp_compat_ulong_result, %.loc19_87.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u32.patt: %pattern_type.126 = value_binding_pattern carbon_u32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u32: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %impl.elem0.loc21: %.955 = impl_witness_access constants.%ImplicitAs.impl_witness.040, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.09f]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.be4]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.591]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %u32 = call %bound_method.loc21_25.2(%int_1.loc21) [concrete = constants.%int_1.ce2]
// CHECK:STDOUT:   %.loc21_25.1: %u32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.ce2]
// CHECK:STDOUT:   %.loc21_25.2: %u32 = converted %int_1.loc21, %.loc21_25.1 [concrete = constants.%int_1.ce2]
// CHECK:STDOUT:   %carbon_u32: %u32 = value_binding carbon_u32, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u32_result.patt: %pattern_type.5a5 = value_binding_pattern carbon_u32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc22: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u32.ref: %u32 = name_ref carbon_u32, %carbon_u32
// CHECK:STDOUT:   %.loc22_74.1: ref %UIntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.c09 = addr_of %.loc22_74.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.43aa98.2(%carbon_u32.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_74.2: init %UIntResult to %.loc22_74.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_36: type = splice_block %UIntResult.ref [concrete = constants.%UIntResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %UIntResult.ref: type = name_ref UIntResult, imports.%UIntResult.decl [concrete = constants.%UIntResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_74.3: ref %UIntResult = temporary %.loc22_74.1, %.loc22_74.2
// CHECK:STDOUT:   %.loc22_74.4: %UIntResult = acquire_value %.loc22_74.3
// CHECK:STDOUT:   %carbon_u32_result: %UIntResult = value_binding carbon_u32_result, %.loc22_74.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.aab = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.aab = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_59: %.3a8 = impl_witness_access constants.%ImplicitAs.impl_witness.65d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd]
// CHECK:STDOUT:   %bound_method.loc24_59.1: <bound method> = bound_method %float, %impl.elem0.loc24_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_59.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.aaa]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.a26 = call %bound_method.loc24_59.2(%float) [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc24_59.2: init %f32.a26 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_59.3: ref %f32.a26 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_59.4: init %f32.a26 to %.loc24_59.3 = in_place_init %.loc24_59.2 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc24_59.5: init %array_type to %a.var = array_init (%.loc24_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_59.1, %.loc24_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_50: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc24: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.f45 = impl_witness_access constants.%As.impl_witness.314, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.3ef]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.3ef]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.3ef]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.165 = impl_witness_access constants.%ImplicitAs.impl_witness.6d1, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   %UIntResult.cpp_destructor.bound: <bound method> = bound_method %.loc22_74.3, constants.%UIntResult.cpp_destructor
// CHECK:STDOUT:   %UIntResult.cpp_destructor.call: init %empty_tuple.type = call %UIntResult.cpp_destructor.bound(%.loc22_74.3)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_87.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call.loc19: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound.loc19(%.loc19_87.3)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound.loc16: <bound method> = bound_method %.loc16_89.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call.loc16: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound.loc16(%.loc16_89.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.2(%self.param: ref %f32.a26) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.3(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_unsigned_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long: type = class_type @ULong32 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32.builtin: type = int_type unsigned, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.9f4: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ae2: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1e5: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a9a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.ae2 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.1e5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.57e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.c47: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.57e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.3ef: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.d5d: <witness> = impl_witness imports.%Copy.impl_witness_table.916 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long, (%Copy.impl_witness.d5d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.9f4: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.f33: type = fn_type_with_self_type %Copy.WithSelf.Op.type.9f4, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op: %Cpp.unsigned_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ade: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a9a = impl_witness_table (%Core.import_ref.ade), @Core.IntLiteral.as.ImplicitAs.impl.160 [concrete]
// CHECK:STDOUT:   %Core.import_ref.f63: %Cpp.unsigned_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.916 = impl_witness_table (%Core.import_ref.f63), @Cpp.unsigned_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyUnsignedLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9f4 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.9f4 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.unsigned_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.c47 = impl_witness_access constants.%ImplicitAs.impl_witness.1e5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.3ef]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.unsigned_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.3ef]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %unsigned_long.ref.loc8 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.unsigned_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.9f4 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.9f4 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.unsigned_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.unsigned_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_37: %Cpp.unsigned_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.f33 = impl_witness_access constants.%Copy.impl_witness.d5d, element0 [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_37, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long = call %bound_method.loc9(%.loc9_37)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %unsigned_long.ref.loc9 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc9: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %u32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %Cpp.unsigned_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
