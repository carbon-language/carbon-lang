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
// CHECK:STDOUT:   %pattern_type.c8a: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e82: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4bd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2f8: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.7c2: %Int.as.ImplicitAs.impl.Convert.type.2f8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a33: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1be, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d03: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bb6: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d03 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b11: %ImplicitAs.type.e82 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a33) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.486: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.b11) [concrete]
// CHECK:STDOUT:   %.036: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.486, %ImplicitAs.facet.b11 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.bb6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.bb6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.3b9: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.9e7: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.750: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.66a: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.fc5: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3a8: %Core.IntLiteral.as.As.impl.Convert.type.fc5 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.32d: <witness> = impl_witness imports.%As.impl_witness_table.913, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.faf: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.370: %Core.IntLiteral.as.As.impl.Convert.type.faf = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.66a = facet_value Core.IntLiteral, (%As.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.ee4: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i64, %As.facet) [concrete]
// CHECK:STDOUT:   %.feb: type = fn_type_with_self_type %As.WithSelf.Convert.type.ee4, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.370 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.370, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.c14: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e8a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9df, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.35f: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.f2c: %Int.as.ImplicitAs.impl.Convert.type.35f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.607: %ImplicitAs.type.4bd = facet_value %i64, (%ImplicitAs.impl_witness.e8a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.374: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.607) [concrete]
// CHECK:STDOUT:   %.d1c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.374, %ImplicitAs.facet.607 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.9e7, %Int.as.ImplicitAs.impl.Convert.f2c [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.f2c, @Int.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.181: <bound method> = bound_method %int_1.9e7, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.750 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.ee6: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1dd: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.750)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bfc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.3c0, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b89: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b89 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9f5: %ImplicitAs.type.1dd = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bfc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.68e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.750, %ImplicitAs.facet.9f5) [concrete]
// CHECK:STDOUT:   %.28b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.68e, %ImplicitAs.facet.9f5 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.5a8: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.467: %f32.750 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.467) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.3: %Destroy.Op.type.60afa5.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = @F.%i64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.7f3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1be = impl_witness_table (%Core.import_ref.7f3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.321: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2f8) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.7c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9df = impl_witness_table (%Core.import_ref.321), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.4af: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.fc5) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.3a8)]
// CHECK:STDOUT:   %As.impl_witness_table.913 = impl_witness_table (%Core.import_ref.4af), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.831: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3c0 = impl_witness_table (%Core.import_ref.831), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.c8a = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_25: type = splice_block %long_long.ref.loc8 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.036 = impl_witness_access constants.%ImplicitAs.impl_witness.a33, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bb6]
// CHECK:STDOUT:   %bound_method.loc8_38.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_38.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.3b9]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc8_38.2(%int_1.loc8) [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:   %.loc8_38.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:   %.loc8_38.2: %i64 = converted %int_1.loc8, %.loc8_38.1 [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:   %cpp_long_long: %i64 = value_binding cpp_long_long, %.loc8_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_long.patt: %pattern_type.c8a = value_binding_pattern carbon_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref: %i64 = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %i64.loc9: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long_long: %i64 = value_binding carbon_long_long, %cpp_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ee6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ee6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_55: %.28b = impl_witness_access constants.%ImplicitAs.impl_witness.bfc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05]
// CHECK:STDOUT:   %bound_method.loc11_55.1: <bound method> = bound_method %float, %impl.elem0.loc11_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_55: <specific function> = specific_function %impl.elem0.loc11_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_55.2: <bound method> = bound_method %float, %specific_fn.loc11_55 [concrete = constants.%bound_method.5a8]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.750 = call %bound_method.loc11_55.2(%float) [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc11_55.2: init %f32.750 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.467]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_55.3: ref %f32.750 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_55.4: init %f32.750 to %.loc11_55.3 = in_place_init %.loc11_55.2 [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc11_55.5: init %array_type to %a.var = array_init (%.loc11_55.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_55.1, %.loc11_55.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_46: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.750 [concrete = constants.%f32.750]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc11: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.feb = impl_witness_access constants.%As.impl_witness.32d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.370]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.c14]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:     %.loc11_30.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:     %.loc11_30.2: %i64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.9e7]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.d1c = impl_witness_access constants.%ImplicitAs.impl_witness.e8a, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.f2c]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.181]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.2(%self.param: ref %f32.750) {
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
// CHECK:STDOUT:   %pattern_type.bb8: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.704: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a01: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.d3a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a01 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4bd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.dce: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.1b7(%From.fe9) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.60d: %UInt.as.ImplicitAs.impl.Convert.type.dce = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.093: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c97, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.de8: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.de8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.338: %ImplicitAs.type.704 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.093) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.25c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.338) [concrete]
// CHECK:STDOUT:   %.f05: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.25c, %ImplicitAs.facet.338 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.818: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.21a: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.750: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.689: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.841: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.94d: %Core.IntLiteral.as.As.impl.Convert.type.841 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.03f: <witness> = impl_witness imports.%As.impl_witness_table.db7, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c59: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.c9b: %Core.IntLiteral.as.As.impl.Convert.type.c59 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.689 = facet_value Core.IntLiteral, (%As.impl_witness.03f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.479: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u64, %As.facet) [concrete]
// CHECK:STDOUT:   %.37a: type = fn_type_with_self_type %As.WithSelf.Convert.type.479, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.c9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.c9b, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.eb6: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c34: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.3d8, @UInt.as.ImplicitAs.impl.1b7(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.22d: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.1b7(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.8c3: %UInt.as.ImplicitAs.impl.Convert.type.22d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2d5: %ImplicitAs.type.4bd = facet_value %u64, (%ImplicitAs.impl_witness.c34) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b9f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.2d5) [concrete]
// CHECK:STDOUT:   %.511: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b9f, %ImplicitAs.facet.2d5 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.21a, %UInt.as.ImplicitAs.impl.Convert.8c3 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.8c3, @UInt.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.699: <bound method> = bound_method %int_1.21a, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.750 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.ee6: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1dd: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.750)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bfc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.3c0, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b89: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b89 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9f5: %ImplicitAs.type.1dd = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bfc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.68e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.750, %ImplicitAs.facet.9f5) [concrete]
// CHECK:STDOUT:   %.28b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.68e, %ImplicitAs.facet.9f5 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.5a8: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.467: %f32.750 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.467) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.3: %Destroy.Op.type.60afa5.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long_long = @F.%u64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.0ae: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a01) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.d3a)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c97 = impl_witness_table (%Core.import_ref.0ae), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.767: @UInt.as.ImplicitAs.impl.1b7.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.dce) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.1b7.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.60d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3d8 = impl_witness_table (%Core.import_ref.767), @UInt.as.ImplicitAs.impl.1b7 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c1c: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.841) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.94d)]
// CHECK:STDOUT:   %As.impl_witness_table.db7 = impl_witness_table (%Core.import_ref.c1c), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.831: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3c0 = impl_witness_table (%Core.import_ref.831), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.bb8 = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_34: type = splice_block %unsigned_long_long.ref.loc8 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long_long.ref.loc8: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.f05 = impl_witness_access constants.%ImplicitAs.impl_witness.093, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3e4]
// CHECK:STDOUT:   %bound_method.loc8_56.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_56.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.818]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc8_56.2(%int_1.loc8) [concrete = constants.%int_1.21a]
// CHECK:STDOUT:   %.loc8_56.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.21a]
// CHECK:STDOUT:   %.loc8_56.2: %u64 = converted %int_1.loc8, %.loc8_56.1 [concrete = constants.%int_1.21a]
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = value_binding cpp_unsigned_long_long, %.loc8_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long_long.patt: %pattern_type.bb8 = value_binding_pattern carbon_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref: %u64 = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %u64.loc9: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long_long: %u64 = value_binding carbon_unsigned_long_long, %cpp_unsigned_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ee6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ee6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_64.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_64: %.28b = impl_witness_access constants.%ImplicitAs.impl_witness.bfc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05]
// CHECK:STDOUT:   %bound_method.loc11_64.1: <bound method> = bound_method %float, %impl.elem0.loc11_64 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_64: <specific function> = specific_function %impl.elem0.loc11_64, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_64.2: <bound method> = bound_method %float, %specific_fn.loc11_64 [concrete = constants.%bound_method.5a8]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.750 = call %bound_method.loc11_64.2(%float) [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc11_64.2: init %f32.750 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.467]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_64.3: ref %f32.750 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_64.4: init %f32.750 to %.loc11_64.3 = in_place_init %.loc11_64.2 [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc11_64.5: init %array_type to %a.var = array_init (%.loc11_64.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_64.1, %.loc11_64.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_55: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.750 [concrete = constants.%f32.750]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc11: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.37a = impl_witness_access constants.%As.impl_witness.03f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.c9b]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.eb6]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.21a]
// CHECK:STDOUT:     %.loc11_30.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.21a]
// CHECK:STDOUT:     %.loc11_30.2: %u64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.21a]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.511 = impl_witness_access constants.%ImplicitAs.impl_witness.c34, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.8c3]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @UInt.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.699]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.2(%self.param: ref %f32.750) {
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
// CHECK:STDOUT:   %pattern_type.a99: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.57f: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4bd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e82: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.69e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6fd: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.337: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.337, %ImplicitAs.facet.6fd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.dfc: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %LongResult: type = class_type @LongResult [concrete]
// CHECK:STDOUT:   %pattern_type.5f3: type = pattern_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLong.cpp_overload_set.type: type = cpp_overload_set_type @PassLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassLong.cpp_overload_set.value: %PassLong.cpp_overload_set.type = cpp_overload_set_value @PassLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.618: type = ptr_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.type.972681.1: type = fn_type @PassLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.b57a85.1: %PassLong__carbon_thunk.type.972681.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %IntResult: type = class_type @IntResult [concrete]
// CHECK:STDOUT:   %pattern_type.e33: type = pattern_type %IntResult [concrete]
// CHECK:STDOUT:   %ptr.837: type = ptr_type %IntResult [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.type.972681.2: type = fn_type @PassLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.b57a85.2: %PassLong__carbon_thunk.type.972681.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c8a: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.773: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c30: %ImplicitAs.type.e82 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.773) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.d34: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.c30) [concrete]
// CHECK:STDOUT:   %.c5b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.d34, %ImplicitAs.facet.c30 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.750: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.fa8: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.fa8 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.d2d: type = facet_type <@As, @As(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.2e4: <witness> = impl_witness imports.%As.impl_witness_table.c0a [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d2d = facet_value Core.IntLiteral, (%As.impl_witness.2e4) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.d3b: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long, %As.facet) [concrete]
// CHECK:STDOUT:   %.b8d: type = fn_type_with_self_type %As.WithSelf.Convert.type.d3b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bcb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.3f68 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f79: %ImplicitAs.type.4bd = facet_value %Cpp.long, (%ImplicitAs.impl_witness.bcb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f8e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.f79) [concrete]
// CHECK:STDOUT:   %.6e7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f8e, %ImplicitAs.facet.f79 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.dfc, %Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.750 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.ee6: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1dd: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.750)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bfc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.3c0, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b89: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b89 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9f5: %ImplicitAs.type.1dd = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bfc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.68e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.750, %ImplicitAs.facet.9f5) [concrete]
// CHECK:STDOUT:   %.28b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.68e, %ImplicitAs.facet.9f5 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.467: %f32.750 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.467) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.3: type = fn_type @Destroy.Op.loc26_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.3: %Destroy.Op.type.60afa5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %IntResult.cpp_destructor.type: type = fn_type @IntResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %IntResult.cpp_destructor: %IntResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor.type: type = fn_type @LongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor: %LongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .CppCompat = %CppCompat.bd5028.1
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
// CHECK:STDOUT:   %CppCompat.bd5028.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .Long32 = %Core.Long32
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Long32: type = import_ref Core//prelude/types/cpp/int, Long32, loaded [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.69e = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.8d5 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82b = impl_witness_table (%Core.import_ref.67b339.2), @Cpp.long.as.ImplicitAs.impl.71e [concrete]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.fa8 = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.67b339.3: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.c0a = impl_witness_table (%Core.import_ref.67b339.3), @Core.IntLiteral.as.As.impl.a27 [concrete]
// CHECK:STDOUT:   %Core.import_ref.67b339.4: %Cpp.long.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3f68 = impl_witness_table (%Core.import_ref.67b339.4), @Cpp.long.as.ImplicitAs.impl.81b [concrete]
// CHECK:STDOUT:   %Core.import_ref.831: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3c0 = impl_witness_table (%Core.import_ref.831), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.a99 = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_20: type = splice_block %long.ref.loc15 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc15: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc15_28.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc15_28.2: %Cpp.long = converted %int_1.loc15, %.loc15_28.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %cpp_long: %Cpp.long = value_binding cpp_long, %.loc15_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_result.patt: %pattern_type.5f3 = value_binding_pattern cpp_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc16: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long.ref.loc16: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc16_69.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.618 = addr_of %.loc16_69.1
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
// CHECK:STDOUT:     %cpp_compat_long.patt: %pattern_type.a99 = value_binding_pattern cpp_compat_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc18: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc18: type = splice_block %Long32.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.bd5028.1 [concrete = imports.%CppCompat.bd5028.1]
// CHECK:STDOUT:     %Long32.ref: type = name_ref Long32, imports.%Core.Long32 [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long: %Cpp.long = value_binding cpp_compat_long, %cpp_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_result.patt: %pattern_type.5f3 = value_binding_pattern cpp_compat_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_55: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc19: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long.ref: %Cpp.long = name_ref cpp_compat_long, %cpp_compat_long
// CHECK:STDOUT:   %.loc19_83.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.618 = addr_of %.loc19_83.1
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
// CHECK:STDOUT:     %carbon_i32.patt: %pattern_type.c6b = value_binding_pattern carbon_i32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc21: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i32.loc21_37: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc21_34.1: %i32 = as_compatible %cpp_long.ref.loc21
// CHECK:STDOUT:   %.loc21_34.2: %i32 = converted %cpp_long.ref.loc21, %.loc21_34.1
// CHECK:STDOUT:   %i32.loc21_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %carbon_i32: %i32 = value_binding carbon_i32, %.loc21_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32_result.patt: %pattern_type.e33 = value_binding_pattern carbon_i32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_49: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc22: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i32.ref: %i32 = name_ref carbon_i32, %carbon_i32
// CHECK:STDOUT:   %.loc22_72.1: ref %IntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.837 = addr_of %.loc22_72.1
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
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.c8a = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc24: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc24: %.c5b = impl_witness_access constants.%ImplicitAs.impl_witness.773, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %cpp_long.ref.loc24, %impl.elem0.loc24
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc24: init %i64 = call %bound_method.loc24(%cpp_long.ref.loc24)
// CHECK:STDOUT:   %.loc24_32.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc24
// CHECK:STDOUT:   %.loc24_32.2: %i64 = converted %cpp_long.ref.loc24, %.loc24_32.1
// CHECK:STDOUT:   %carbon_i64: %i64 = value_binding carbon_i64, %.loc24_32.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ee6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ee6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc26_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc26_50: %.28b = impl_witness_access constants.%ImplicitAs.impl_witness.bfc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05]
// CHECK:STDOUT:   %bound_method.loc26_50.1: <bound method> = bound_method %float, %impl.elem0.loc26_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc26_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_50.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.750 = call %bound_method.loc26_50.2(%float) [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc26_50.2: init %f32.750 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.467]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc26_50.3: ref %f32.750 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc26_50.4: init %f32.750 to %.loc26_50.3 = in_place_init %.loc26_50.2 [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc26_50.5: init %array_type to %a.var = array_init (%.loc26_50.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc26_3: init %array_type = converted %.loc26_50.1, %.loc26_50.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc26_3
// CHECK:STDOUT:   %.loc26_41: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.750 [concrete = constants.%f32.750]
// CHECK:STDOUT:     %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc26: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:     %impl.elem0.loc26_30.1: %.b8d = impl_witness_access constants.%As.impl_witness.2e4, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc26_30.1: <bound method> = bound_method %int_1.loc26, %impl.elem0.loc26_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long = call %bound_method.loc26_30.1(%int_1.loc26) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:     %.loc26_30.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:     %.loc26_30.2: %Cpp.long = converted %int_1.loc26, %.loc26_30.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:     %impl.elem0.loc26_30.2: %.6e7 = impl_witness_access constants.%ImplicitAs.impl_witness.bcb, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:     %bound_method.loc26_30.2: <bound method> = bound_method %.loc26_30.2, %impl.elem0.loc26_30.2 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long.as.ImplicitAs.impl.Convert.call.loc26: init Core.IntLiteral = call %bound_method.loc26_30.2(%.loc26_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc26_30.3: Core.IntLiteral = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc26 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc26_30.4: Core.IntLiteral = converted %.loc26_30.2, %.loc26_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc26_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.3
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
// CHECK:STDOUT: fn @Destroy.Op.loc26_3.2(%self.param: ref %f32.750) {
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
// CHECK:STDOUT:   %pattern_type.a99: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.57f: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.69e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.337: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.337, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.dfc: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.430: <witness> = impl_witness imports.%Copy.impl_witness_table.455 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.430) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e35: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.56f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e35, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.69e = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.8d5 [concrete]
// CHECK:STDOUT:   %Core.import_ref.59e: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.455 = impl_witness_table (%Core.import_ref.59e), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a99 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a99 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.a99 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.a99 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_28: %Cpp.long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.56f = impl_witness_access constants.%Copy.impl_witness.430, element0 [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_28, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.call: init %Cpp.long = call %bound_method.loc9(%.loc9_28)
// CHECK:STDOUT:   assign %b.var, %Cpp.long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.b63e49.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.a99: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.57f: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.69e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.337: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.337, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.dfc: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.38b: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.cbf: <witness> = impl_witness imports.%EqWith.impl_witness_table.dcd [concrete]
// CHECK:STDOUT:   %EqWith.facet.d30: %EqWith.type.38b = facet_value %Cpp.long, (%EqWith.impl_witness.cbf) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b7f: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.d30) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.181: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.d30) [concrete]
// CHECK:STDOUT:   %.2ab: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.b7f, %EqWith.facet.d30 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.78c: type = fn_type @Cpp.long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.1ca: %Cpp.long.as.EqWith.impl.Equal.type.78c = struct_value () [concrete]
// CHECK:STDOUT:   %.09b: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.181, %EqWith.facet.d30 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.8e5: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.a6a: %Cpp.long.as.EqWith.impl.NotEqual.type.8e5 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.d86: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.358: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.2dc [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.6e2: %OrderedWith.type.d86 = facet_value %Cpp.long, (%OrderedWith.impl_witness.358) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.e00: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.6e2) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.67a: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.6e2) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.cf9: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.6e2) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.7e6: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.6e2) [concrete]
// CHECK:STDOUT:   %.955: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.cf9, %OrderedWith.facet.6e2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.725: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.7c8: %Cpp.long.as.OrderedWith.impl.Greater.type.725 = struct_value () [concrete]
// CHECK:STDOUT:   %.ce1: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.e00, %OrderedWith.facet.6e2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.335: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.606: %Cpp.long.as.OrderedWith.impl.Less.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %.e4e: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.7e6, %OrderedWith.facet.6e2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = struct_value () [concrete]
// CHECK:STDOUT:   %.ba5: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.67a, %OrderedWith.facet.6e2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.42c: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.42c = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.69e = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.8d5 [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: %Cpp.long.as.EqWith.impl.Equal.type.78c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %Core.import_ref.206: %Cpp.long.as.EqWith.impl.NotEqual.type.8e5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %EqWith.impl_witness_table.dcd = impl_witness_table (%Core.import_ref.74d, %Core.import_ref.206), @Cpp.long.as.EqWith.impl.e54 [concrete]
// CHECK:STDOUT:   %Core.import_ref.6ae: %Cpp.long.as.OrderedWith.impl.Less.type.335 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %Core.import_ref.dfe: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.42c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %Core.import_ref.148: %Cpp.long.as.OrderedWith.impl.Greater.type.725 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %Core.import_ref.3cc: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.2dc = impl_witness_table (%Core.import_ref.6ae, %Core.import_ref.dfe, %Core.import_ref.148, %Core.import_ref.3cc), @Cpp.long.as.OrderedWith.impl.e32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a99 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc8_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc8_21.2: %Cpp.long = converted %int_1.loc8, %.loc8_21.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc8_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.a99 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc9_21.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.2ab = impl_witness_access constants.%EqWith.impl_witness.cbf, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.09b = impl_witness_access constants.%EqWith.impl_witness.cbf, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.955 = impl_witness_access constants.%OrderedWith.impl_witness.358, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.ce1 = impl_witness_access constants.%OrderedWith.impl_witness.358, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.e4e = impl_witness_access constants.%OrderedWith.impl_witness.358, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.ba5 = impl_witness_access constants.%OrderedWith.impl_witness.358, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.0c8]
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
// CHECK:STDOUT:   %pattern_type.a99: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.57f: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.69e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6fd: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.337: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.337, %ImplicitAs.facet.6fd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.dfc: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.694: type = facet_type <@EqWith, @EqWith(%i32)> [concrete]
// CHECK:STDOUT:   %T.7ce: %ImplicitAs.type.57f = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.a30b9f.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.676(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.f8664f.1: %Cpp.long.as.EqWith.impl.NotEqual.type.a30b9f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.a30b9f.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.676(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.f8664f.2: %Cpp.long.as.EqWith.impl.NotEqual.type.a30b9f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.436f75.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.676(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.990e2a.1: %Cpp.long.as.EqWith.impl.Equal.type.436f75.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.436f75.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.676(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.990e2a.2: %Cpp.long.as.EqWith.impl.Equal.type.436f75.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.290: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ef6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.968 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f1f: %ImplicitAs.type.57f = facet_value %i32, (%ImplicitAs.impl_witness.ef6) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.536: <witness> = impl_witness imports.%EqWith.impl_witness_table.ac9, @Cpp.long.as.EqWith.impl.676(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.ab456c.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.676(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.779d45.1: %Cpp.long.as.EqWith.impl.Equal.type.ab456c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.b12970.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.676(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.52fc9d.1: %Cpp.long.as.EqWith.impl.NotEqual.type.b12970.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.ab456c.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.676(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.779d45.2: %Cpp.long.as.EqWith.impl.Equal.type.ab456c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.b12970.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.676(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.52fc9d.2: %Cpp.long.as.EqWith.impl.NotEqual.type.b12970.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.2d3: %EqWith.type.694 = facet_value %Cpp.long, (%EqWith.impl_witness.536) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.174: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i32, %EqWith.facet.2d3) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.f62: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i32, %EqWith.facet.2d3) [concrete]
// CHECK:STDOUT:   %.5dc: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.174, %EqWith.facet.2d3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.f5a1df.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.779d45.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.47d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %.94f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.47d, %ImplicitAs.facet.f1f [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.f5a1df.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.779d45.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %.722: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.f62, %EqWith.facet.2d3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.7a9cc0.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.52fc9d.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.7a9cc0.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.52fc9d.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.9fe: type = facet_type <@OrderedWith, @OrderedWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.505ca2.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.70e(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.fc14fe.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.505ca2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.505ca2.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.70e(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.fc14fe.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.505ca2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.570e66.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.70e(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.a04d4b.1: %Cpp.long.as.OrderedWith.impl.Greater.type.570e66.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.570e66.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.70e(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.a04d4b.2: %Cpp.long.as.OrderedWith.impl.Greater.type.570e66.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.922a3d.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.70e(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.6e2f49.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.922a3d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.922a3d.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.70e(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.6e2f49.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.922a3d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.700ac0.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.70e(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.1b2d38.1: %Cpp.long.as.OrderedWith.impl.Less.type.700ac0.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.700ac0.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.70e(%T.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.1b2d38.2: %Cpp.long.as.OrderedWith.impl.Less.type.700ac0.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.d0c: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.ecb: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.3cd, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.d3d2f9.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.da08f7.1: %Cpp.long.as.OrderedWith.impl.Less.type.d3d2f9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e8c820.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9cadd4.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e8c820.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.aa6543.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.32071c.1: %Cpp.long.as.OrderedWith.impl.Greater.type.aa6543.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ed2e76.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8259bb.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ed2e76.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.d3d2f9.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.da08f7.2: %Cpp.long.as.OrderedWith.impl.Less.type.d3d2f9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e8c820.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9cadd4.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e8c820.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.aa6543.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.32071c.2: %Cpp.long.as.OrderedWith.impl.Greater.type.aa6543.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ed2e76.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8259bb.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ed2e76.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.fdf: %OrderedWith.type.9fe = facet_value %Cpp.long, (%OrderedWith.impl_witness.ecb) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.17d: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.fdf) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.e32: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.fdf) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.10a: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.fdf) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.5ae: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.fdf) [concrete]
// CHECK:STDOUT:   %.6c0: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.10a, %OrderedWith.facet.fdf [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.589731.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.32071c.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.589731.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.32071c.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %.0c9: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.17d, %OrderedWith.facet.fdf [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.8b31cc.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.da08f7.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.8b31cc.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.da08f7.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %.323: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.5ae, %OrderedWith.facet.fdf [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.dbca5b.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8259bb.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.dbca5b.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8259bb.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %.1e4: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.e32, %OrderedWith.facet.fdf [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.858623.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9cadd4.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.858623.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9cadd4.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.4c6: <witness> = impl_witness imports.%EqWith.impl_witness_table.ac9, @Cpp.long.as.EqWith.impl.676(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.2dfac6.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.676(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.11872d.1: %Cpp.long.as.EqWith.impl.Equal.type.2dfac6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.eca941.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.676(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.17e525.1: %Cpp.long.as.EqWith.impl.NotEqual.type.eca941.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.2dfac6.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.676(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.11872d.2: %Cpp.long.as.EqWith.impl.Equal.type.2dfac6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.eca941.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.676(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.17e525.2: %Cpp.long.as.EqWith.impl.NotEqual.type.eca941.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.7f3: %EqWith.type.290 = facet_value %Cpp.long, (%EqWith.impl_witness.4c6) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b74: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.7f3) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.d4e: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.7f3) [concrete]
// CHECK:STDOUT:   %.624: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.b74, %EqWith.facet.7f3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.c8c541.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.11872d.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.c8c541.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.11872d.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %.d78: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.d4e, %EqWith.facet.7f3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.06067e.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.17e525.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.06067e.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.17e525.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.c0d: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.3cd, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.73307c.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.93a121.1: %Cpp.long.as.OrderedWith.impl.Less.type.73307c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.561a98.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.829d6c.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.561a98.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.133b47.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.45dc8e.1: %Cpp.long.as.OrderedWith.impl.Greater.type.133b47.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.bc966f.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.864755.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.bc966f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.73307c.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.93a121.2: %Cpp.long.as.OrderedWith.impl.Less.type.73307c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.561a98.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.829d6c.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.561a98.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.133b47.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.45dc8e.2: %Cpp.long.as.OrderedWith.impl.Greater.type.133b47.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.bc966f.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.70e(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.864755.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.bc966f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.49c: %OrderedWith.type.d0c = facet_value %Cpp.long, (%OrderedWith.impl_witness.c0d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.887: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.49c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.509: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.49c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.a06: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.49c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.d09: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.49c) [concrete]
// CHECK:STDOUT:   %.25c: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.a06, %OrderedWith.facet.49c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.35e919.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.45dc8e.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.35e919.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.45dc8e.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %.8f3: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.887, %OrderedWith.facet.49c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.90ec60.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.93a121.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.90ec60.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.93a121.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %.f9f: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.d09, %OrderedWith.facet.49c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5f06c7.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.864755.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5f06c7.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.864755.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %.e0b: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.509, %OrderedWith.facet.49c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.1f97e5.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.829d6c.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.1f97e5.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.829d6c.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.69e = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.8d5 [concrete]
// CHECK:STDOUT:   %Core.import_ref.5c2: @Cpp.long.as.EqWith.impl.676.%Cpp.long.as.EqWith.impl.Equal.type.2 (%Cpp.long.as.EqWith.impl.Equal.type.436f75.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.676.%Cpp.long.as.EqWith.impl.Equal.2 (constants.%Cpp.long.as.EqWith.impl.Equal.990e2a.1)]
// CHECK:STDOUT:   %Core.import_ref.b46: @Cpp.long.as.EqWith.impl.676.%Cpp.long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long.as.EqWith.impl.NotEqual.type.a30b9f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.676.%Cpp.long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long.as.EqWith.impl.NotEqual.f8664f.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.ac9 = impl_witness_table (%Core.import_ref.5c2, %Core.import_ref.b46), @Cpp.long.as.EqWith.impl.676 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.540: @Cpp.long.as.EqWith.impl.676.%Cpp.long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long.as.EqWith.impl.NotEqual.type.a30b9f.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long.as.EqWith.impl.676.%Cpp.long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long.as.EqWith.impl.NotEqual.f8664f.2)]
// CHECK:STDOUT:   %Core.Equal.db0: @Cpp.long.as.EqWith.impl.676.%Cpp.long.as.EqWith.impl.Equal.type.1 (%Cpp.long.as.EqWith.impl.Equal.type.436f75.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long.as.EqWith.impl.676.%Cpp.long.as.EqWith.impl.Equal.1 (constants.%Cpp.long.as.EqWith.impl.Equal.990e2a.2)]
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.968 = impl_witness_table (%Core.import_ref.67b339.2), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ff4: @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.Less.type.2 (%Cpp.long.as.OrderedWith.impl.Less.type.700ac0.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long.as.OrderedWith.impl.Less.1b2d38.1)]
// CHECK:STDOUT:   %Core.import_ref.a9b: @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.922a3d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.6e2f49.1)]
// CHECK:STDOUT:   %Core.import_ref.670: @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long.as.OrderedWith.impl.Greater.type.570e66.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long.as.OrderedWith.impl.Greater.a04d4b.1)]
// CHECK:STDOUT:   %Core.import_ref.1550: @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.505ca2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.fc14fe.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.3cd = impl_witness_table (%Core.import_ref.ff4, %Core.import_ref.a9b, %Core.import_ref.670, %Core.import_ref.1550), @Cpp.long.as.OrderedWith.impl.70e [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.8c9: @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.505ca2.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.fc14fe.2)]
// CHECK:STDOUT:   %Core.Greater.737: @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long.as.OrderedWith.impl.Greater.type.570e66.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long.as.OrderedWith.impl.Greater.a04d4b.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.f5f: @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.922a3d.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.6e2f49.2)]
// CHECK:STDOUT:   %Core.Less.ca7: @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.Less.type.1 (%Cpp.long.as.OrderedWith.impl.Less.type.700ac0.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.70e.%Cpp.long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long.as.OrderedWith.impl.Less.1b2d38.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a99 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc9_21.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.5dc = impl_witness_access constants.%EqWith.impl_witness.536, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.779d45.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.f5a1df.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.EqWith.impl.Equal.type.ab456c.1 = specific_constant imports.%Core.Equal.db0, @Cpp.long.as.EqWith.impl.676(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.779d45.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Cpp.long.as.EqWith.impl.Equal.type.ab456c.1 = name_ref Equal, %.loc10_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.779d45.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.f5a1df.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.722 = impl_witness_access constants.%EqWith.impl_witness.536, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.52fc9d.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.7a9cc0.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.b12970.1 = specific_constant imports.%Core.NotEqual.540, @Cpp.long.as.EqWith.impl.676(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.52fc9d.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Cpp.long.as.EqWith.impl.NotEqual.type.b12970.1 = name_ref NotEqual, %.loc11_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.52fc9d.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.7a9cc0.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.6c0 = impl_witness_access constants.%OrderedWith.impl_witness.ecb, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.32071c.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.589731.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.OrderedWith.impl.Greater.type.aa6543.1 = specific_constant imports.%Core.Greater.737, @Cpp.long.as.OrderedWith.impl.70e(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.32071c.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Cpp.long.as.OrderedWith.impl.Greater.type.aa6543.1 = name_ref Greater, %.loc12_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.32071c.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.589731.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_7: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_7: init %Cpp.long = call %bound_method.loc12_7(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_7
// CHECK:STDOUT:   %.loc12_7.2: %Cpp.long = converted %b.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.0c9 = impl_witness_access constants.%OrderedWith.impl_witness.ecb, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.da08f7.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.8b31cc.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.OrderedWith.impl.Less.type.d3d2f9.1 = specific_constant imports.%Core.Less.ca7, @Cpp.long.as.OrderedWith.impl.70e(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.da08f7.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Cpp.long.as.OrderedWith.impl.Less.type.d3d2f9.1 = name_ref Less, %.loc13_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.da08f7.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.8b31cc.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_7: init %Cpp.long = call %bound_method.loc13_7(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_7
// CHECK:STDOUT:   %.loc13_7.2: %Cpp.long = converted %b.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.323 = impl_witness_access constants.%OrderedWith.impl_witness.ecb, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8259bb.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.dbca5b.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ed2e76.1 = specific_constant imports.%Core.GreaterOrEquivalent.8c9, @Cpp.long.as.OrderedWith.impl.70e(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8259bb.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ed2e76.1 = name_ref GreaterOrEquivalent, %.loc14_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.8259bb.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.dbca5b.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.1e4 = impl_witness_access constants.%OrderedWith.impl_witness.ecb, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9cadd4.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.858623.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e8c820.1 = specific_constant imports.%Core.LessOrEquivalent.f5f, @Cpp.long.as.OrderedWith.impl.70e(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9cadd4.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e8c820.1 = name_ref LessOrEquivalent, %.loc15_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9cadd4.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_5: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_5: init %Cpp.long = call %bound_method.loc15_5.3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_5
// CHECK:STDOUT:   %.loc15_5.4: %Cpp.long = converted %b.ref.loc15, %.loc15_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.858623.2]
// CHECK:STDOUT:   %bound_method.loc15_5.4: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_8: init %Cpp.long = call %bound_method.loc15_8(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_8
// CHECK:STDOUT:   %.loc15_8.2: %Cpp.long = converted %b.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.4(%a.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.624 = impl_witness_access constants.%EqWith.impl_witness.4c6, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.11872d.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.300) [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.57f = converted Core.IntLiteral, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.c8c541.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.EqWith.impl.Equal.type.2dfac6.1 = specific_constant imports.%Core.Equal.db0, @Cpp.long.as.EqWith.impl.676(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.11872d.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long.as.EqWith.impl.Equal.type.2dfac6.1 = name_ref Equal, %.loc17_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.11872d.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long = call %bound_method.loc17_5.3(%int_1.loc17) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long = converted %int_1.loc17, %.loc17_5.3 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.c8c541.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.d78 = impl_witness_access constants.%EqWith.impl_witness.4c6, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.17e525.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.300) [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.57f = converted Core.IntLiteral, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.06067e.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.eca941.1 = specific_constant imports.%Core.NotEqual.540, @Cpp.long.as.EqWith.impl.676(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.17e525.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long.as.EqWith.impl.NotEqual.type.eca941.1 = name_ref NotEqual, %.loc18_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.17e525.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long = call %bound_method.loc18_5.3(%int_1.loc18) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long = converted %int_1.loc18, %.loc18_5.3 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.06067e.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.25c = impl_witness_access constants.%OrderedWith.impl_witness.c0d, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.45dc8e.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.300) [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.57f = converted Core.IntLiteral, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.35e919.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.OrderedWith.impl.Greater.type.133b47.1 = specific_constant imports.%Core.Greater.737, @Cpp.long.as.OrderedWith.impl.70e(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.45dc8e.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long.as.OrderedWith.impl.Greater.type.133b47.1 = name_ref Greater, %.loc19_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.45dc8e.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long = call %bound_method.loc19_5.3(%int_1.loc19) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long = converted %int_1.loc19, %.loc19_5.3 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.35e919.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_7: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7: init %Cpp.long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.8f3 = impl_witness_access constants.%OrderedWith.impl_witness.c0d, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.93a121.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.300) [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.57f = converted Core.IntLiteral, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.90ec60.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.OrderedWith.impl.Less.type.73307c.1 = specific_constant imports.%Core.Less.ca7, @Cpp.long.as.OrderedWith.impl.70e(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.93a121.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long.as.OrderedWith.impl.Less.type.73307c.1 = name_ref Less, %.loc20_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.93a121.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long = call %bound_method.loc20_5.3(%int_1.loc20) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long = converted %int_1.loc20, %.loc20_5.3 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.90ec60.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7: init %Cpp.long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.f9f = impl_witness_access constants.%OrderedWith.impl_witness.c0d, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.864755.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.300) [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.57f = converted Core.IntLiteral, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5f06c7.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.bc966f.1 = specific_constant imports.%Core.GreaterOrEquivalent.8c9, @Cpp.long.as.OrderedWith.impl.70e(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.864755.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.bc966f.1 = name_ref GreaterOrEquivalent, %.loc21_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.864755.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_5: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5: init %Cpp.long = call %bound_method.loc21_5.3(%int_1.loc21) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc21_5.4: %Cpp.long = converted %int_1.loc21, %.loc21_5.3 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.5f06c7.2]
// CHECK:STDOUT:   %bound_method.loc21_5.4: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8: init %Cpp.long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.4(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.e0b = impl_witness_access constants.%OrderedWith.impl_witness.c0d, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.829d6c.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.300) [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.57f = converted Core.IntLiteral, %ImplicitAs.facet.loc22 [concrete = constants.%ImplicitAs.facet.6fd]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.1f97e5.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.561a98.1 = specific_constant imports.%Core.LessOrEquivalent.f5f, @Cpp.long.as.OrderedWith.impl.70e(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.829d6c.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.561a98.1 = name_ref LessOrEquivalent, %.loc22_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.829d6c.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_5: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5: init %Cpp.long = call %bound_method.loc22_5.3(%int_1.loc22) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc22_5.4: %Cpp.long = converted %int_1.loc22, %.loc22_5.3 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.1f97e5.2]
// CHECK:STDOUT:   %bound_method.loc22_5.4: <bound method> = bound_method %a.ref.loc22, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_8: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8: init %Cpp.long = call %bound_method.loc22_8(%int_1.loc22) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc22_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc22_8.2: %Cpp.long = converted %int_1.loc22, %.loc22_8.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc22: init bool = call %bound_method.loc22_5.4(%a.ref.loc22, %.loc22_8.2)
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
// CHECK:STDOUT:   %pattern_type.a99: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.57f: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.69e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6fd: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.337: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.337, %ImplicitAs.facet.6fd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.dfc: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.38b: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.7ce: %ImplicitAs.type.57f = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.a2c520.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.260(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.5e8303.1: %T.as_type.as.EqWith.impl.NotEqual.type.a2c520.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.a2c520.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.260(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.5e8303.2: %T.as_type.as.EqWith.impl.NotEqual.type.a2c520.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.6be74e.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.260(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.7dc567.1: %T.as_type.as.EqWith.impl.Equal.type.6be74e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.6be74e.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.260(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.7dc567.2: %T.as_type.as.EqWith.impl.Equal.type.6be74e.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ef6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.968 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f1f: %ImplicitAs.type.57f = facet_value %i32, (%ImplicitAs.impl_witness.ef6) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.7cf: <witness> = impl_witness imports.%EqWith.impl_witness_table.98e, @T.as_type.as.EqWith.impl.260(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.d28795.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.260(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.2662d9.1: %T.as_type.as.EqWith.impl.Equal.type.d28795.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.c103b8.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.260(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.93f64a.1: %T.as_type.as.EqWith.impl.NotEqual.type.c103b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.d28795.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.260(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.2662d9.2: %T.as_type.as.EqWith.impl.Equal.type.d28795.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.c103b8.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.260(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.93f64a.2: %T.as_type.as.EqWith.impl.NotEqual.type.c103b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.5cb: %EqWith.type.38b = facet_value %i32, (%EqWith.impl_witness.7cf) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.adf: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.5cb) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.def: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.5cb) [concrete]
// CHECK:STDOUT:   %.54f: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.adf, %EqWith.facet.5cb [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.565271.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.2662d9.2, @T.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.565271.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.2662d9.1, @T.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.47d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %.94f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.47d, %ImplicitAs.facet.f1f [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b2c: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.def, %EqWith.facet.5cb [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.60cb9d.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.93f64a.2, @T.as_type.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.60cb9d.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.93f64a.1, @T.as_type.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.d86: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.a7dc42.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.050(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c139f1.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.a7dc42.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.a7dc42.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.050(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c139f1.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.a7dc42.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.d5a434.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.050(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.6f211f.1: %T.as_type.as.OrderedWith.impl.Greater.type.d5a434.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.d5a434.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.050(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.6f211f.2: %T.as_type.as.OrderedWith.impl.Greater.type.d5a434.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1deb45.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.050(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e84bd2.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1deb45.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1deb45.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.050(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e84bd2.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1deb45.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.2da6ff.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.050(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.569e06.1: %T.as_type.as.OrderedWith.impl.Less.type.2da6ff.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.2da6ff.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.050(%T.7ce) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.569e06.2: %T.as_type.as.OrderedWith.impl.Less.type.2da6ff.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.643: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.aba, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.219b92.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.03602f.1: %T.as_type.as.OrderedWith.impl.Less.type.219b92.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8084e8.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.da85f5.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8084e8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.cf1df1.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.cb94c9.1: %T.as_type.as.OrderedWith.impl.Greater.type.cf1df1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.56a165.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.a3225f.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.56a165.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.219b92.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.03602f.2: %T.as_type.as.OrderedWith.impl.Less.type.219b92.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8084e8.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.da85f5.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8084e8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.cf1df1.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.cb94c9.2: %T.as_type.as.OrderedWith.impl.Greater.type.cf1df1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.56a165.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.a3225f.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.56a165.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.306: %OrderedWith.type.d86 = facet_value %i32, (%OrderedWith.impl_witness.643) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.710: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.306) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3f1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.306) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.b82: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.306) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.72e: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.306) [concrete]
// CHECK:STDOUT:   %.538: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.b82, %OrderedWith.facet.306 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.682a73.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.cb94c9.2, @T.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.682a73.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.cb94c9.1, @T.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %.42a: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.710, %OrderedWith.facet.306 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.bdada8.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.03602f.2, @T.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.bdada8.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.03602f.1, @T.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %.256: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.72e, %OrderedWith.facet.306 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3e0cfa.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.a3225f.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3e0cfa.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.a3225f.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %.418: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.3f1, %OrderedWith.facet.306 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bddad0.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.da85f5.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bddad0.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.da85f5.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.883: <witness> = impl_witness imports.%EqWith.impl_witness_table.98e, @T.as_type.as.EqWith.impl.260(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.a7a3c5.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.260(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.6eaed8.1: %T.as_type.as.EqWith.impl.Equal.type.a7a3c5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.f95cd1.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.260(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.a78465.1: %T.as_type.as.EqWith.impl.NotEqual.type.f95cd1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.a7a3c5.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.260(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.6eaed8.2: %T.as_type.as.EqWith.impl.Equal.type.a7a3c5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.f95cd1.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.260(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.a78465.2: %T.as_type.as.EqWith.impl.NotEqual.type.f95cd1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.078: %EqWith.type.38b = facet_value Core.IntLiteral, (%EqWith.impl_witness.883) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.cee: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.078) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.b75: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.078) [concrete]
// CHECK:STDOUT:   %.005: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.cee, %EqWith.facet.078 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.b59f5c.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.6eaed8.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.3c90cf.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.6eaed8.2, @T.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.a7c141.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.3c90cf.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.b59f5c.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.6eaed8.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.3c90cf.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.6eaed8.1, @T.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.a7c141.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.3c90cf.2 [concrete]
// CHECK:STDOUT:   %.207: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.b75, %EqWith.facet.078 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.01d382.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.a78465.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.ad1407.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.a78465.2, @T.as_type.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.470dd1.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.ad1407.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.01d382.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.a78465.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.ad1407.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.a78465.1, @T.as_type.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.470dd1.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.ad1407.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.ad0: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.aba, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.7af86b.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.e770c6.1: %T.as_type.as.OrderedWith.impl.Less.type.7af86b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e6472d.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.9e33ca.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e6472d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.1b3a32.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.05e776.1: %T.as_type.as.OrderedWith.impl.Greater.type.1b3a32.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9d3a77.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d0540f.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9d3a77.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.7af86b.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.e770c6.2: %T.as_type.as.OrderedWith.impl.Less.type.7af86b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e6472d.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.9e33ca.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e6472d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.1b3a32.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.05e776.2: %T.as_type.as.OrderedWith.impl.Greater.type.1b3a32.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9d3a77.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.050(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d0540f.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9d3a77.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.6b0: %OrderedWith.type.d86 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.ad0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.03f: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.6b0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.a84: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.6b0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.548: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.6b0) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.9ac: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.6b0) [concrete]
// CHECK:STDOUT:   %.56c: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.548, %OrderedWith.facet.6b0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.280660.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.05e776.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.ebe525.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.05e776.2, @T.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.b30b9e.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.ebe525.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.280660.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.05e776.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.ebe525.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.05e776.1, @T.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.b30b9e.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.ebe525.2 [concrete]
// CHECK:STDOUT:   %.b7b: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.03f, %OrderedWith.facet.6b0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.b7c085.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.e770c6.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.963d4b.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.e770c6.2, @T.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.448940.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.963d4b.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.b7c085.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.e770c6.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.963d4b.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.e770c6.1, @T.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.448940.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.963d4b.2 [concrete]
// CHECK:STDOUT:   %.58d: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.9ac, %OrderedWith.facet.6b0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.ac7ebe.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d0540f.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ef79a4.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d0540f.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.e1e0f1.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ef79a4.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.ac7ebe.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d0540f.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ef79a4.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d0540f.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.e1e0f1.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ef79a4.2 [concrete]
// CHECK:STDOUT:   %.f96: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.a84, %OrderedWith.facet.6b0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.997ef1.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.9e33ca.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2b87e3.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.9e33ca.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.05815c.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2b87e3.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.997ef1.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.9e33ca.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2b87e3.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.9e33ca.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %bound_method.05815c.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2b87e3.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.69e = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.8d5 [concrete]
// CHECK:STDOUT:   %Core.import_ref.bf6: @T.as_type.as.EqWith.impl.260.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.6be74e.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.260.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.7dc567.1)]
// CHECK:STDOUT:   %Core.import_ref.b16: @T.as_type.as.EqWith.impl.260.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.a2c520.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.260.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.5e8303.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.98e = impl_witness_table (%Core.import_ref.bf6, %Core.import_ref.b16), @T.as_type.as.EqWith.impl.260 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.c18: @T.as_type.as.EqWith.impl.260.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.a2c520.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.260.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.5e8303.2)]
// CHECK:STDOUT:   %Core.Equal.e88: @T.as_type.as.EqWith.impl.260.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.6be74e.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.260.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.7dc567.2)]
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.968 = impl_witness_table (%Core.import_ref.67b339.2), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.abf: @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.2da6ff.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.569e06.1)]
// CHECK:STDOUT:   %Core.import_ref.c48: @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1deb45.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.e84bd2.1)]
// CHECK:STDOUT:   %Core.import_ref.807: @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.d5a434.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.6f211f.1)]
// CHECK:STDOUT:   %Core.import_ref.91b: @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.a7dc42.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c139f1.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.aba = impl_witness_table (%Core.import_ref.abf, %Core.import_ref.c48, %Core.import_ref.807, %Core.import_ref.91b), @T.as_type.as.OrderedWith.impl.050 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.151: @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.a7dc42.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c139f1.2)]
// CHECK:STDOUT:   %Core.Greater.94c: @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.d5a434.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.6f211f.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.90a: @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1deb45.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.e84bd2.2)]
// CHECK:STDOUT:   %Core.Less.a6d: @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.2da6ff.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.050.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.569e06.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a99 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc9_21.2
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.54f = impl_witness_access constants.%EqWith.impl_witness.7cf, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.2662d9.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.565271.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.as_type.as.EqWith.impl.Equal.type.d28795.1 = specific_constant imports.%Core.Equal.e88, @T.as_type.as.EqWith.impl.260(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.2662d9.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.d28795.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.2662d9.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.565271.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long = call %bound_method.loc10_3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %Cpp.long = converted %b.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %a.ref.loc10)
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.b2c = impl_witness_access constants.%EqWith.impl_witness.7cf, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.93f64a.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.60cb9d.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.as_type.as.EqWith.impl.NotEqual.type.c103b8.1 = specific_constant imports.%Core.NotEqual.c18, @T.as_type.as.EqWith.impl.260(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.93f64a.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.c103b8.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.93f64a.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.60cb9d.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11_3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %Cpp.long = converted %b.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %a.ref.loc11)
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.538 = impl_witness_access constants.%OrderedWith.impl_witness.643, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.cb94c9.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.682a73.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.as_type.as.OrderedWith.impl.Greater.type.cf1df1.1 = specific_constant imports.%Core.Greater.94c, @T.as_type.as.OrderedWith.impl.050(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.cb94c9.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.cf1df1.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.cb94c9.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.682a73.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12_3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %Cpp.long = converted %b.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %a.ref.loc12)
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.42a = impl_witness_access constants.%OrderedWith.impl_witness.643, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.03602f.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.bdada8.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.as_type.as.OrderedWith.impl.Less.type.219b92.1 = specific_constant imports.%Core.Less.a6d, @T.as_type.as.OrderedWith.impl.050(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.03602f.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.219b92.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.03602f.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.bdada8.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long = call %bound_method.loc13_3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %Cpp.long = converted %b.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %a.ref.loc13)
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.256 = impl_witness_access constants.%OrderedWith.impl_witness.643, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.a3225f.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3e0cfa.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.56a165.1 = specific_constant imports.%Core.GreaterOrEquivalent.151, @T.as_type.as.OrderedWith.impl.050(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.a3225f.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.56a165.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.a3225f.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3e0cfa.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long = call %bound_method.loc14_3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %Cpp.long = converted %b.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %a.ref.loc14)
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.418 = impl_witness_access constants.%OrderedWith.impl_witness.643, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.da85f5.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bddad0.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8084e8.1 = specific_constant imports.%Core.LessOrEquivalent.90a, @T.as_type.as.OrderedWith.impl.050(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.da85f5.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8084e8.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.da85f5.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.bddad0.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long = call %bound_method.loc15_3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %Cpp.long = converted %b.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %a.ref.loc15)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.005 = impl_witness_access constants.%EqWith.impl_witness.883, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6eaed8.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.b59f5c.1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.3c90cf.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.a7c141.1]
// CHECK:STDOUT:   %.loc17_5: %T.as_type.as.EqWith.impl.Equal.type.a7a3c5.1 = specific_constant imports.%Core.Equal.e88, @T.as_type.as.EqWith.impl.260(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6eaed8.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.a7a3c5.1 = name_ref Equal, %.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6eaed8.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.b59f5c.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.3c90cf.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17 [concrete = constants.%bound_method.a7c141.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.207 = impl_witness_access constants.%EqWith.impl_witness.883, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.a78465.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.01d382.1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.ad1407.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.470dd1.1]
// CHECK:STDOUT:   %.loc18_5: %T.as_type.as.EqWith.impl.NotEqual.type.f95cd1.1 = specific_constant imports.%Core.NotEqual.c18, @T.as_type.as.EqWith.impl.260(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.a78465.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.f95cd1.1 = name_ref NotEqual, %.loc18_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.a78465.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.01d382.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.ad1407.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18 [concrete = constants.%bound_method.470dd1.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.56c = impl_witness_access constants.%OrderedWith.impl_witness.ad0, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.05e776.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.280660.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.ebe525.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.b30b9e.1]
// CHECK:STDOUT:   %.loc19_5: %T.as_type.as.OrderedWith.impl.Greater.type.1b3a32.1 = specific_constant imports.%Core.Greater.94c, @T.as_type.as.OrderedWith.impl.050(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.05e776.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.1b3a32.1 = name_ref Greater, %.loc19_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.05e776.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.280660.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.ebe525.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19 [concrete = constants.%bound_method.b30b9e.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.b7b = impl_witness_access constants.%OrderedWith.impl_witness.ad0, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.e770c6.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.b7c085.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.963d4b.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.448940.1]
// CHECK:STDOUT:   %.loc20_5: %T.as_type.as.OrderedWith.impl.Less.type.7af86b.1 = specific_constant imports.%Core.Less.a6d, @T.as_type.as.OrderedWith.impl.050(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.e770c6.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.7af86b.1 = name_ref Less, %.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.e770c6.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.b7c085.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.963d4b.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20 [concrete = constants.%bound_method.448940.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.58d = impl_witness_access constants.%OrderedWith.impl_witness.ad0, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d0540f.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.ac7ebe.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ef79a4.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.e1e0f1.1]
// CHECK:STDOUT:   %.loc21_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9d3a77.1 = specific_constant imports.%Core.GreaterOrEquivalent.151, @T.as_type.as.OrderedWith.impl.050(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d0540f.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9d3a77.1 = name_ref GreaterOrEquivalent, %.loc21_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.d0540f.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.ac7ebe.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.ef79a4.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21 [concrete = constants.%bound_method.e1e0f1.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.f96 = impl_witness_access constants.%OrderedWith.impl_witness.ad0, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.9e33ca.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.997ef1.1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2b87e3.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.05815c.1]
// CHECK:STDOUT:   %.loc22_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e6472d.1 = specific_constant imports.%Core.LessOrEquivalent.90a, @T.as_type.as.OrderedWith.impl.050(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.9e33ca.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e6472d.1 = name_ref LessOrEquivalent, %.loc22_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.9e33ca.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.997ef1.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.6fd) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.2b87e3.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22 [concrete = constants.%bound_method.05815c.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long = call %bound_method.loc22_3(%int_1.loc22) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc22_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc22_3.2: %Cpp.long = converted %int_1.loc22, %.loc22_3.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc22: init bool = call %bound_method.loc22_5.3(%.loc22_3.2, %a.ref.loc22)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- arithmetic_homogeneous_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.a99: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.57f: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.69e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.337: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.337, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.dfc: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %Cpp.long, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.4c1: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.88b: type = fn_type_with_self_type %Negate.WithSelf.Op.type.4c1, %Negate.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op.type: type = fn_type @Cpp.long.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op: %Cpp.long.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.a50: type = facet_type <@AddWith, @AddWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.381: <witness> = impl_witness imports.%AddWith.impl_witness_table.4b3 [concrete]
// CHECK:STDOUT:   %AddWith.facet.b3e: %AddWith.type.a50 = facet_value %Cpp.long, (%AddWith.impl_witness.381) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.129: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Cpp.long, %AddWith.facet.b3e) [concrete]
// CHECK:STDOUT:   %.7e7: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.129, %AddWith.facet.b3e [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.type.43d: type = fn_type @Cpp.long.as.AddWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.cc7: %Cpp.long.as.AddWith.impl.Op.type.43d = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.type.ae0: type = facet_type <@SubWith, @SubWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness.04b: <witness> = impl_witness imports.%SubWith.impl_witness_table.b87 [concrete]
// CHECK:STDOUT:   %SubWith.facet.1ae: %SubWith.type.ae0 = facet_value %Cpp.long, (%SubWith.impl_witness.04b) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.fd6: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Cpp.long, %SubWith.facet.1ae) [concrete]
// CHECK:STDOUT:   %.02e: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.fd6, %SubWith.facet.1ae [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.type.43d: type = fn_type @Cpp.long.as.SubWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.cc7: %Cpp.long.as.SubWith.impl.Op.type.43d = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.type.806: type = facet_type <@MulWith, @MulWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness.5a2: <witness> = impl_witness imports.%MulWith.impl_witness_table.b1b [concrete]
// CHECK:STDOUT:   %MulWith.facet.057: %MulWith.type.806 = facet_value %Cpp.long, (%MulWith.impl_witness.5a2) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.a88: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Cpp.long, %MulWith.facet.057) [concrete]
// CHECK:STDOUT:   %.fa6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.a88, %MulWith.facet.057 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.type.43d: type = fn_type @Cpp.long.as.MulWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.cc7: %Cpp.long.as.MulWith.impl.Op.type.43d = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.type.9c6: type = facet_type <@DivWith, @DivWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness.e73: <witness> = impl_witness imports.%DivWith.impl_witness_table.bbf [concrete]
// CHECK:STDOUT:   %DivWith.facet.ed4: %DivWith.type.9c6 = facet_value %Cpp.long, (%DivWith.impl_witness.e73) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.0fb: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Cpp.long, %DivWith.facet.ed4) [concrete]
// CHECK:STDOUT:   %.37a: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.0fb, %DivWith.facet.ed4 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.type.43d: type = fn_type @Cpp.long.as.DivWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.cc7: %Cpp.long.as.DivWith.impl.Op.type.43d = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.type.bcd: type = facet_type <@ModWith, @ModWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness.e52: <witness> = impl_witness imports.%ModWith.impl_witness_table.59a [concrete]
// CHECK:STDOUT:   %ModWith.facet.a0e: %ModWith.type.bcd = facet_value %Cpp.long, (%ModWith.impl_witness.e52) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.78c: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Cpp.long, %ModWith.facet.a0e) [concrete]
// CHECK:STDOUT:   %.ca9: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.78c, %ModWith.facet.a0e [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.type.43d: type = fn_type @Cpp.long.as.ModWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.cc7: %Cpp.long.as.ModWith.impl.Op.type.43d = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.69e = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.8d5 [concrete]
// CHECK:STDOUT:   %Core.import_ref.43490e.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.59e3cf.1: %Cpp.long.as.Negate.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.43490e.1, %Core.import_ref.59e3cf.1), @Cpp.long.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.43490e.4 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.59e3cf.2: %Cpp.long.as.AddWith.impl.Op.type.43d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.AddWith.impl.Op.cc7]
// CHECK:STDOUT:   %AddWith.impl_witness_table.4b3 = impl_witness_table (%Core.import_ref.43490e.4, %Core.import_ref.59e3cf.2), @Cpp.long.as.AddWith.impl.656 [concrete]
// CHECK:STDOUT:   %Core.import_ref.43490e.7 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.59e3cf.3: %Cpp.long.as.SubWith.impl.Op.type.43d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.SubWith.impl.Op.cc7]
// CHECK:STDOUT:   %SubWith.impl_witness_table.b87 = impl_witness_table (%Core.import_ref.43490e.7, %Core.import_ref.59e3cf.3), @Cpp.long.as.SubWith.impl.0b4 [concrete]
// CHECK:STDOUT:   %Core.import_ref.43490e.10 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.59e3cf.4: %Cpp.long.as.MulWith.impl.Op.type.43d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.MulWith.impl.Op.cc7]
// CHECK:STDOUT:   %MulWith.impl_witness_table.b1b = impl_witness_table (%Core.import_ref.43490e.10, %Core.import_ref.59e3cf.4), @Cpp.long.as.MulWith.impl.aef [concrete]
// CHECK:STDOUT:   %Core.import_ref.43490e.13 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.59e3cf.5: %Cpp.long.as.DivWith.impl.Op.type.43d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.DivWith.impl.Op.cc7]
// CHECK:STDOUT:   %DivWith.impl_witness_table.bbf = impl_witness_table (%Core.import_ref.43490e.13, %Core.import_ref.59e3cf.5), @Cpp.long.as.DivWith.impl.2ed [concrete]
// CHECK:STDOUT:   %Core.import_ref.43490e.16 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.59e3cf.6: %Cpp.long.as.ModWith.impl.Op.type.43d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ModWith.impl.Op.cc7]
// CHECK:STDOUT:   %ModWith.impl_witness_table.59a = impl_witness_table (%Core.import_ref.43490e.16, %Core.import_ref.59e3cf.6), @Cpp.long.as.ModWith.impl.897 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ArithmeticHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a99 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc8_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc8_21.2: %Cpp.long = converted %int_1.loc8, %.loc8_21.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %x: %Cpp.long = value_binding x, %.loc8_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.a99 = value_binding_pattern y [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %y: %Cpp.long = value_binding y, %.loc9_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a99 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc11: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc11: %.88b = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Cpp.long.as.Negate.impl.Op]
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
// CHECK:STDOUT:     %b.patt: %pattern_type.a99 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc12: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc12: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc12: %.7e7 = impl_witness_access constants.%AddWith.impl_witness.381, element1 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.cc7]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.a99 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc13: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc13: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc13: %.02e = impl_witness_access constants.%SubWith.impl_witness.04b, element1 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.cc7]
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
// CHECK:STDOUT:     %d.patt: %pattern_type.a99 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc14: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc14: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc14: %.fa6 = impl_witness_access constants.%MulWith.impl_witness.5a2, element1 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.cc7]
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
// CHECK:STDOUT:     %e.patt: %pattern_type.a99 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc15: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc15: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc15: %.37a = impl_witness_access constants.%DivWith.impl_witness.e73, element1 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.cc7]
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
// CHECK:STDOUT:     %f.patt: %pattern_type.a99 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc16: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc16: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc16: %.ca9 = impl_witness_access constants.%ModWith.impl_witness.e52, element1 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.cc7]
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
// CHECK:STDOUT:   %pattern_type.a99: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.57f: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.228: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.b99: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.228) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.56c: %T.as_type.as.ImplicitAs.impl.Convert.type.b99 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.69e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6fd: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.337: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.337, %ImplicitAs.facet.6fd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.dfc: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.df5: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %U.7ce: %ImplicitAs.type.57f = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.8de486.1: %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.8de486.2: %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.430: <witness> = impl_witness imports.%Copy.impl_witness_table.455 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.430) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.73d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.eef, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.998: %ImplicitAs.type.57f = facet_value %Cpp.long, (%ImplicitAs.impl_witness.73d) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.689: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.770, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.6d6faf.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.326ab9.1: %Cpp.long.as.AddAssignWith.impl.Op.type.6d6faf.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.6d6faf.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.326ab9.2: %Cpp.long.as.AddAssignWith.impl.Op.type.6d6faf.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.083: %AddAssignWith.type.df5 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.689) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.62a: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet.083) [concrete]
// CHECK:STDOUT:   %.e31: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.62a, %AddAssignWith.facet.083 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.8bb75f.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.326ab9.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.8bb75f.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.326ab9.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.465: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.67383f.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.3ad8a2.1: %Cpp.long.as.SubAssignWith.impl.Op.type.67383f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.67383f.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.3ad8a2.2: %Cpp.long.as.SubAssignWith.impl.Op.type.67383f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.369: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.b10, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.00e861.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.dc470f.1: %Cpp.long.as.SubAssignWith.impl.Op.type.00e861.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.00e861.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.dc470f.2: %Cpp.long.as.SubAssignWith.impl.Op.type.00e861.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.1b8: %SubAssignWith.type.465 = facet_value %Cpp.long, (%SubAssignWith.impl_witness.369) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.f93: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long, %SubAssignWith.facet.1b8) [concrete]
// CHECK:STDOUT:   %.5f3: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.f93, %SubAssignWith.facet.1b8 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.18a04a.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.dc470f.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.18a04a.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.dc470f.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.cf3: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.e4aeeb.1: %Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.e4aeeb.2: %Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.029: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.005, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.2fcd52.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.0c06cb.1: %Cpp.long.as.MulAssignWith.impl.Op.type.2fcd52.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.2fcd52.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.0c06cb.2: %Cpp.long.as.MulAssignWith.impl.Op.type.2fcd52.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.015: %MulAssignWith.type.cf3 = facet_value %Cpp.long, (%MulAssignWith.impl_witness.029) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.065: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long, %MulAssignWith.facet.015) [concrete]
// CHECK:STDOUT:   %.3ab: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.065, %MulAssignWith.facet.015 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c6b84b.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.0c06cb.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c6b84b.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.0c06cb.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.3ba: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.b103d3.1: %Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.b103d3.2: %Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.cd3: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.eaa, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.bbd427.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.12a66a.1: %Cpp.long.as.DivAssignWith.impl.Op.type.bbd427.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.bbd427.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.12a66a.2: %Cpp.long.as.DivAssignWith.impl.Op.type.bbd427.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.93e: %DivAssignWith.type.3ba = facet_value %Cpp.long, (%DivAssignWith.impl_witness.cd3) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.5c1: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long, %DivAssignWith.facet.93e) [concrete]
// CHECK:STDOUT:   %.479: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.5c1, %DivAssignWith.facet.93e [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.8d8562.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.12a66a.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.8d8562.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.12a66a.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.67b: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.911741.1: %Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.911741.2: %Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.756: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.923, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.bbb658.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.7b2715.1: %Cpp.long.as.ModAssignWith.impl.Op.type.bbb658.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.bbb658.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.7b2715.2: %Cpp.long.as.ModAssignWith.impl.Op.type.bbb658.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.8bb: %ModAssignWith.type.67b = facet_value %Cpp.long, (%ModAssignWith.impl_witness.756) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.439: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long, %ModAssignWith.facet.8bb) [concrete]
// CHECK:STDOUT:   %.306: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.439, %ModAssignWith.facet.8bb [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.b501ff.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.7b2715.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.b501ff.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.7b2715.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.39a: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.372d33.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.372d33.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.ad4: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.1f9, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.148bb0.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.7758c7.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.148bb0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.148bb0.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.7758c7.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.148bb0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.da9: %BitAndAssignWith.type.39a = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.ad4) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.b9d: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long, %BitAndAssignWith.facet.da9) [concrete]
// CHECK:STDOUT:   %.558: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.b9d, %BitAndAssignWith.facet.da9 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.7f08a6.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.7758c7.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.7f08a6.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.7758c7.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.0e9: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.eb672e.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.eb672e.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.ba3: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.a24, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.0ecca5.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.eaf675.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.0ecca5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.0ecca5.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.eaf675.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.0ecca5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.4db: %BitOrAssignWith.type.0e9 = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.ba3) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.186: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long, %BitOrAssignWith.facet.4db) [concrete]
// CHECK:STDOUT:   %.463: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.186, %BitOrAssignWith.facet.4db [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.d62b94.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.eaf675.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.d62b94.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.eaf675.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.fe1: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.afec29.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.afec29.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.dfd: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.a33, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.e77d85.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.996e2b.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.e77d85.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.e77d85.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.996e2b.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.e77d85.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.32e: %BitXorAssignWith.type.fe1 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.dfd) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.eab: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long, %BitXorAssignWith.facet.32e) [concrete]
// CHECK:STDOUT:   %.722: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.eab, %BitXorAssignWith.facet.32e [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.5f8731.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.996e2b.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.5f8731.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.996e2b.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.fad: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e4f634.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e4f634.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.4fd: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.d41, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ccd19c.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.98b3be.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ccd19c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ccd19c.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.98b3be.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ccd19c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.99d: %LeftShiftAssignWith.type.fad = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.4fd) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.5f4: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long, %LeftShiftAssignWith.facet.99d) [concrete]
// CHECK:STDOUT:   %.e26: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.5f4, %LeftShiftAssignWith.facet.99d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.6bb2fa.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.98b3be.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.6bb2fa.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.98b3be.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.f8d: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.737e64.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.737e64.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.b43: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.f5e, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dc66b8.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.41dc5e.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dc66b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dc66b8.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.41dc5e.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dc66b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.6f3: %RightShiftAssignWith.type.f8d = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.b43) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.ba1: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long, %RightShiftAssignWith.facet.6f3) [concrete]
// CHECK:STDOUT:   %.32c: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.ba1, %RightShiftAssignWith.facet.6f3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1b6d78.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.41dc5e.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1b6d78.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.41dc5e.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.998) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5e1: @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert.type (%T.as_type.as.ImplicitAs.impl.Convert.type.b99) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert (constants.%T.as_type.as.ImplicitAs.impl.Convert.56c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.eef = impl_witness_table (%Core.import_ref.5e1), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.69e = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.8d5 [concrete]
// CHECK:STDOUT:   %Core.import_ref.dd6: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.8de486.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.770 = impl_witness_table (%Core.import_ref.dd6), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c23: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.8de486.2)]
// CHECK:STDOUT:   %Core.import_ref.59e: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.455 = impl_witness_table (%Core.import_ref.59e), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.a72: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.67383f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.3ad8a2.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.b10 = impl_witness_table (%Core.import_ref.a72), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7c9: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.67383f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.3ad8a2.2)]
// CHECK:STDOUT:   %Core.import_ref.0f8: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.e4aeeb.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.005 = impl_witness_table (%Core.import_ref.0f8), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.70a: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.e4aeeb.2)]
// CHECK:STDOUT:   %Core.import_ref.ef9: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.b103d3.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.eaa = impl_witness_table (%Core.import_ref.ef9), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a20: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.b103d3.2)]
// CHECK:STDOUT:   %Core.import_ref.a59: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.911741.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.923 = impl_witness_table (%Core.import_ref.a59), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.ecd: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.911741.2)]
// CHECK:STDOUT:   %Core.import_ref.304: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.372d33.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.1f9 = impl_witness_table (%Core.import_ref.304), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a36: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.372d33.2)]
// CHECK:STDOUT:   %Core.import_ref.7d3: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.eb672e.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.a24 = impl_witness_table (%Core.import_ref.7d3), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.334: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.eb672e.2)]
// CHECK:STDOUT:   %Core.import_ref.add: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.afec29.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.a33 = impl_witness_table (%Core.import_ref.add), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fe2: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.afec29.2)]
// CHECK:STDOUT:   %Core.import_ref.563: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e4f634.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.d41 = impl_witness_table (%Core.import_ref.563), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.4fd: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e4f634.2)]
// CHECK:STDOUT:   %Core.import_ref.a1d: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.737e64.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.f5e = impl_witness_table (%Core.import_ref.a1d), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.e29: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.737e64.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a99 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a99 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.a99 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.a99 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var %b.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   assign %b.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = ref_binding b, %b.var
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11: %.e31 = impl_witness_access constants.%AddAssignWith.impl_witness.689, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.326ab9.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.57f = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.73d) [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.57f = converted constants.%Cpp.long, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.8bb75f.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.6d6faf.1 = specific_constant imports.%Core.Op.c23, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.326ab9.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.AddAssignWith.impl.Op.type.6d6faf.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.326ab9.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.8bb75f.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.5f3 = impl_witness_access constants.%SubAssignWith.impl_witness.369, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.dc470f.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.57f = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.73d) [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.57f = converted constants.%Cpp.long, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.18a04a.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.SubAssignWith.impl.Op.type.00e861.1 = specific_constant imports.%Core.Op.7c9, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.dc470f.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.SubAssignWith.impl.Op.type.00e861.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.dc470f.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.18a04a.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.3ab = impl_witness_access constants.%MulAssignWith.impl_witness.029, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.0c06cb.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.57f = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.73d) [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.57f = converted constants.%Cpp.long, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c6b84b.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.2fcd52.1 = specific_constant imports.%Core.Op.70a, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.0c06cb.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.MulAssignWith.impl.Op.type.2fcd52.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.0c06cb.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c6b84b.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.479 = impl_witness_access constants.%DivAssignWith.impl_witness.cd3, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.12a66a.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.57f = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.73d) [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.57f = converted constants.%Cpp.long, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.8d8562.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.DivAssignWith.impl.Op.type.bbd427.1 = specific_constant imports.%Core.Op.a20, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.12a66a.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.DivAssignWith.impl.Op.type.bbd427.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.12a66a.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.8d8562.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.306 = impl_witness_access constants.%ModAssignWith.impl_witness.756, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.7b2715.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.57f = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.73d) [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.57f = converted constants.%Cpp.long, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.b501ff.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.bbb658.1 = specific_constant imports.%Core.Op.ecd, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.7b2715.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long.as.ModAssignWith.impl.Op.type.bbb658.1 = name_ref Op, %.loc15_5.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.7b2715.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.b501ff.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.558 = impl_witness_access constants.%BitAndAssignWith.impl_witness.ad4, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.7758c7.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.57f = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.73d) [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.57f = converted constants.%Cpp.long, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.7f08a6.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.148bb0.1 = specific_constant imports.%Core.Op.a36, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.7758c7.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitAndAssignWith.impl.Op.type.148bb0.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.7758c7.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.7f08a6.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.463 = impl_witness_access constants.%BitOrAssignWith.impl_witness.ba3, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.eaf675.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.57f = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.73d) [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.57f = converted constants.%Cpp.long, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.d62b94.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.0ecca5.1 = specific_constant imports.%Core.Op.334, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.eaf675.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitOrAssignWith.impl.Op.type.0ecca5.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.eaf675.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.d62b94.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.722 = impl_witness_access constants.%BitXorAssignWith.impl_witness.dfd, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.996e2b.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.57f = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.73d) [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.57f = converted constants.%Cpp.long, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.5f8731.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.e77d85.1 = specific_constant imports.%Core.Op.fe2, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.996e2b.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.BitXorAssignWith.impl.Op.type.e77d85.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.996e2b.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.5f8731.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.e26 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.4fd, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.98b3be.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.57f = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.73d) [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.57f = converted constants.%Cpp.long, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.6bb2fa.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ccd19c.1 = specific_constant imports.%Core.Op.4fd, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.98b3be.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.ccd19c.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.98b3be.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.6bb2fa.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.32c = impl_witness_access constants.%RightShiftAssignWith.impl_witness.b43, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.41dc5e.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.57f = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.73d) [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.57f = converted constants.%Cpp.long, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.998]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1b6d78.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dc66b8.1 = specific_constant imports.%Core.Op.e29, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.41dc5e.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.dc66b8.1 = name_ref Op, %.loc21_5.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.41dc5e.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.998) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1b6d78.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_9)
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.b63e49.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.a99: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.57f: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.69e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6fd: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.337: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.337, %ImplicitAs.facet.6fd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.dfc: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.e75: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %U.7ce: %ImplicitAs.type.57f = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.8de486.1: %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.8de486.2: %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ef6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.968 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f1f: %ImplicitAs.type.57f = facet_value %i32, (%ImplicitAs.impl_witness.ef6) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.626: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.770, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.d09936.1: %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.d09936.2: %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.771: %AddAssignWith.type.e75 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.626) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.5ac: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.771) [concrete]
// CHECK:STDOUT:   %.6ba: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.5ac, %AddAssignWith.facet.771 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.85b3b1.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d09936.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.47d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %.94f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.47d, %ImplicitAs.facet.f1f [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.85b3b1.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d09936.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.78a: type = facet_type <@SubAssignWith, @SubAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.67383f.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.3ad8a2.1: %Cpp.long.as.SubAssignWith.impl.Op.type.67383f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.67383f.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.3ad8a2.2: %Cpp.long.as.SubAssignWith.impl.Op.type.67383f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.524: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.b10, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.40dc98.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.01117d.1: %Cpp.long.as.SubAssignWith.impl.Op.type.40dc98.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.40dc98.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.01117d.2: %Cpp.long.as.SubAssignWith.impl.Op.type.40dc98.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.2da: %SubAssignWith.type.78a = facet_value %Cpp.long, (%SubAssignWith.impl_witness.524) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.346: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i32, %SubAssignWith.facet.2da) [concrete]
// CHECK:STDOUT:   %.f40: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.346, %SubAssignWith.facet.2da [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.0c7f41.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.01117d.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.0c7f41.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.01117d.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.946: type = facet_type <@MulAssignWith, @MulAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.e4aeeb.1: %Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.e4aeeb.2: %Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.37d: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.005, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.4cd6f0.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.535568.1: %Cpp.long.as.MulAssignWith.impl.Op.type.4cd6f0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.4cd6f0.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.535568.2: %Cpp.long.as.MulAssignWith.impl.Op.type.4cd6f0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.aea: %MulAssignWith.type.946 = facet_value %Cpp.long, (%MulAssignWith.impl_witness.37d) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.d91: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i32, %MulAssignWith.facet.aea) [concrete]
// CHECK:STDOUT:   %.bec: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.d91, %MulAssignWith.facet.aea [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.0537f2.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.535568.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.0537f2.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.535568.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.4c8: type = facet_type <@DivAssignWith, @DivAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.b103d3.1: %Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.b103d3.2: %Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.5a6: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.eaa, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.89611c.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.c3847d.1: %Cpp.long.as.DivAssignWith.impl.Op.type.89611c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.89611c.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.c3847d.2: %Cpp.long.as.DivAssignWith.impl.Op.type.89611c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.542: %DivAssignWith.type.4c8 = facet_value %Cpp.long, (%DivAssignWith.impl_witness.5a6) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.e26: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i32, %DivAssignWith.facet.542) [concrete]
// CHECK:STDOUT:   %.58c: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.e26, %DivAssignWith.facet.542 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.f2594c.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.c3847d.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.f2594c.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.c3847d.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.20c: type = facet_type <@ModAssignWith, @ModAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.911741.1: %Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.911741.2: %Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.f17: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.923, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.8ab66f.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.489743.1: %Cpp.long.as.ModAssignWith.impl.Op.type.8ab66f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.8ab66f.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.489743.2: %Cpp.long.as.ModAssignWith.impl.Op.type.8ab66f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.c89: %ModAssignWith.type.20c = facet_value %Cpp.long, (%ModAssignWith.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.b62: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i32, %ModAssignWith.facet.c89) [concrete]
// CHECK:STDOUT:   %.c27: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.b62, %ModAssignWith.facet.c89 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.064e32.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.489743.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.064e32.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.489743.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.824: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.372d33.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.372d33.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.7a8: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.1f9, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.984c6b.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.8a3264.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.984c6b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.984c6b.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.8a3264.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.984c6b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.85a: %BitAndAssignWith.type.824 = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.7a8) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.566: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i32, %BitAndAssignWith.facet.85a) [concrete]
// CHECK:STDOUT:   %.aaf: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.566, %BitAndAssignWith.facet.85a [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.ecbb06.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.8a3264.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.ecbb06.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.8a3264.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.d21: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.eb672e.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.eb672e.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.bdd: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.a24, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.b1a3bf.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.0a291e.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.b1a3bf.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.b1a3bf.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.0a291e.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.b1a3bf.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.f73: %BitOrAssignWith.type.d21 = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.bdd) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.c42: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i32, %BitOrAssignWith.facet.f73) [concrete]
// CHECK:STDOUT:   %.23c: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.c42, %BitOrAssignWith.facet.f73 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.58683b.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.0a291e.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.58683b.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.0a291e.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.345: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.afec29.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.afec29.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.774: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.a33, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.54e198.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.ae6273.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.54e198.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.54e198.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.ae6273.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.54e198.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.9b5: %BitXorAssignWith.type.345 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.774) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.afe: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i32, %BitXorAssignWith.facet.9b5) [concrete]
// CHECK:STDOUT:   %.d9a: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.afe, %BitXorAssignWith.facet.9b5 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.a5b152.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.ae6273.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.a5b152.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.ae6273.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.ec7: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e4f634.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e4f634.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.fa2: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.d41, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9b7685.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.cd19ad.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9b7685.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9b7685.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.cd19ad.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9b7685.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.b5b: %LeftShiftAssignWith.type.ec7 = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.fa2) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.7a6: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i32, %LeftShiftAssignWith.facet.b5b) [concrete]
// CHECK:STDOUT:   %.3c2: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.7a6, %LeftShiftAssignWith.facet.b5b [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0df81a.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.cd19ad.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0df81a.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.cd19ad.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.dce: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.737e64.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.737e64.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.7a9: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.f5e, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2ffed.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.fdbfaf.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2ffed.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2ffed.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.fdbfaf.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2ffed.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.718: %RightShiftAssignWith.type.dce = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.7a9) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.0ec: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i32, %RightShiftAssignWith.facet.718) [concrete]
// CHECK:STDOUT:   %.c1e: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.0ec, %RightShiftAssignWith.facet.718 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.3d33c2.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.fdbfaf.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.3d33c2.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.fdbfaf.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.69e = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.8d5 [concrete]
// CHECK:STDOUT:   %Core.import_ref.dd6: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.8de486.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.770 = impl_witness_table (%Core.import_ref.dd6), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c23: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.8de486.2)]
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.968 = impl_witness_table (%Core.import_ref.67b339.2), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.a72: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.67383f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.3ad8a2.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.b10 = impl_witness_table (%Core.import_ref.a72), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7c9: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.67383f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.3ad8a2.2)]
// CHECK:STDOUT:   %Core.import_ref.0f8: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.e4aeeb.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.005 = impl_witness_table (%Core.import_ref.0f8), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.70a: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.ea8a2c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.e4aeeb.2)]
// CHECK:STDOUT:   %Core.import_ref.ef9: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.b103d3.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.eaa = impl_witness_table (%Core.import_ref.ef9), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a20: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.f5a8cb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.b103d3.2)]
// CHECK:STDOUT:   %Core.import_ref.a59: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.911741.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.923 = impl_witness_table (%Core.import_ref.a59), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.ecd: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.dbdc12.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.911741.2)]
// CHECK:STDOUT:   %Core.import_ref.304: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.372d33.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.1f9 = impl_witness_table (%Core.import_ref.304), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.a36: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.a90161.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.372d33.2)]
// CHECK:STDOUT:   %Core.import_ref.7d3: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.eb672e.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.a24 = impl_witness_table (%Core.import_ref.7d3), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.334: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.076f77.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.eb672e.2)]
// CHECK:STDOUT:   %Core.import_ref.add: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.afec29.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.a33 = impl_witness_table (%Core.import_ref.add), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fe2: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8f0564.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.afec29.2)]
// CHECK:STDOUT:   %Core.import_ref.563: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e4f634.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.d41 = impl_witness_table (%Core.import_ref.563), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.4fd: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.6f9c65.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e4f634.2)]
// CHECK:STDOUT:   %Core.import_ref.a1d: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.737e64.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.f5e = impl_witness_table (%Core.import_ref.a1d), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.e29: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.30e989.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.737e64.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndI32() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a99 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a99 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.6ba = impl_witness_access constants.%AddAssignWith.impl_witness.626, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d09936.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.85b3b1.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.1 = specific_constant imports.%Core.Op.c23, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d09936.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d09936.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.85b3b1.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5.1: %.f40 = impl_witness_access constants.%SubAssignWith.impl_witness.524, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.01117d.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5.1, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.0c7f41.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.SubAssignWith.impl.Op.type.40dc98.1 = specific_constant imports.%Core.Op.7c9, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.01117d.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.SubAssignWith.impl.Op.type.40dc98.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.01117d.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.0c7f41.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5.1: %.bec = impl_witness_access constants.%MulAssignWith.impl_witness.37d, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.535568.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5.1, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.0537f2.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.4cd6f0.1 = specific_constant imports.%Core.Op.70a, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.535568.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.MulAssignWith.impl.Op.type.4cd6f0.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.535568.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.0537f2.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_8: init %Cpp.long = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_8
// CHECK:STDOUT:   %.loc12_8.2: %Cpp.long = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.58c = impl_witness_access constants.%DivAssignWith.impl_witness.5a6, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.c3847d.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.f2594c.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.DivAssignWith.impl.Op.type.89611c.1 = specific_constant imports.%Core.Op.a20, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.c3847d.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.DivAssignWith.impl.Op.type.89611c.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.c3847d.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.f2594c.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_8: init %Cpp.long = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_8
// CHECK:STDOUT:   %.loc13_8.2: %Cpp.long = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5.1: %.c27 = impl_witness_access constants.%ModAssignWith.impl_witness.f17, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.489743.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5.1, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.064e32.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.8ab66f.1 = specific_constant imports.%Core.Op.ecd, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.489743.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.ModAssignWith.impl.Op.type.8ab66f.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.489743.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.064e32.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5.1: %.aaf = impl_witness_access constants.%BitAndAssignWith.impl_witness.7a8, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.8a3264.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc16: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc16 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5.1, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.ecbb06.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.984c6b.1 = specific_constant imports.%Core.Op.a36, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.8a3264.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long.as.BitAndAssignWith.impl.Op.type.984c6b.1 = name_ref Op, %.loc16_5.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.8a3264.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_5: init %Cpp.long = call %bound_method.loc16_5.3(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_5
// CHECK:STDOUT:   %.loc16_5.4: %Cpp.long = converted %b.ref.loc16, %.loc16_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.ecbb06.2]
// CHECK:STDOUT:   %bound_method.loc16_5.4: <bound method> = bound_method %a.ref.loc16, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_8: init %Cpp.long = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_8
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.4(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.23c = impl_witness_access constants.%BitOrAssignWith.impl_witness.bdd, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.0a291e.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.58683b.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.b1a3bf.1 = specific_constant imports.%Core.Op.334, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.0a291e.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitOrAssignWith.impl.Op.type.b1a3bf.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.0a291e.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long = call %bound_method.loc17_5.3(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_5
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long = converted %b.ref.loc17, %.loc17_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.58683b.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_8
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5.1: %.d9a = impl_witness_access constants.%BitXorAssignWith.impl_witness.774, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.ae6273.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5.1, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.a5b152.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.54e198.1 = specific_constant imports.%Core.Op.fe2, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.ae6273.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitXorAssignWith.impl.Op.type.54e198.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.ae6273.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long = call %bound_method.loc18_5.3(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc18_5
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long = converted %b.ref.loc18, %.loc18_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.a5b152.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc18_8
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5.1: %.3c2 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.fa2, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.cd19ad.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0df81a.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9b7685.1 = specific_constant imports.%Core.Op.4fd, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.cd19ad.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9b7685.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.cd19ad.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long = call %bound_method.loc19_5.3(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc19_5
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long = converted %b.ref.loc19, %.loc19_5.3
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0df81a.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc19_9: init %Cpp.long = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc19_9
// CHECK:STDOUT:   %.loc19_9.2: %Cpp.long = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.c1e = impl_witness_access constants.%RightShiftAssignWith.impl_witness.7a9, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.fdbfaf.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.3d33c2.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2ffed.1 = specific_constant imports.%Core.Op.e29, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.fdbfaf.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2ffed.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.fdbfaf.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long = call %bound_method.loc20_5.3(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc20_5
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long = converted %b.ref.loc20, %.loc20_5.3
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.3d33c2.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_9: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_9
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc20_9: init %Cpp.long = call %bound_method.loc20_9(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_9.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc20_9
// CHECK:STDOUT:   %.loc20_9.2: %Cpp.long = converted %b.ref.loc20, %.loc20_9.1
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_9.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.a99: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.57f: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.69e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.337: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.337, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.dfc: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.17a: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.7ce: %ImplicitAs.type.57f = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.8de486.1: %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.8de486.2: %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.0a2: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.770, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.c5f543.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.0ff1ab.1: %Cpp.long.as.AddAssignWith.impl.Op.type.c5f543.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.c5f543.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.0ff1ab.2: %Cpp.long.as.AddAssignWith.impl.Op.type.c5f543.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.8af: %AddAssignWith.type.17a = facet_value %Cpp.long, (%AddAssignWith.impl_witness.0a2) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.317: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet.8af) [concrete]
// CHECK:STDOUT:   %.d00: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.317, %AddAssignWith.facet.8af [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.045e18.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.0ff1ab.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.045e18.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.0ff1ab.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.69e = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.8d5 [concrete]
// CHECK:STDOUT:   %Core.import_ref.dd6: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.8de486.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.770 = impl_witness_table (%Core.import_ref.dd6), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c23: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.8de486.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a99 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a99 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9_5.1: %.d00 = impl_witness_access constants.%AddAssignWith.impl_witness.0a2, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.0ff1ab.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.300) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.57f = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.045e18.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.c5f543.1 = specific_constant imports.%Core.Op.c23, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.0ff1ab.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.c5f543.1 = name_ref Op, %.loc9_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.0ff1ab.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc9_5.2: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5: init %Cpp.long = call %bound_method.loc9_5.3(%int_1.loc9) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_5.4: %Cpp.long = converted %int_1.loc9, %.loc9_5.3 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.045e18.2]
// CHECK:STDOUT:   %bound_method.loc9_5.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_8: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_8: init %Cpp.long = call %bound_method.loc9_8(%int_1.loc9) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_8 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc9_8.2: %Cpp.long = converted %int_1.loc9, %.loc9_8.1 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_5.4(%a.ref, %.loc9_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.a99: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.57f: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.69e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6fd: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.337: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.6fd) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.337, %ImplicitAs.facet.6fd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.dfc: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.0da(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ddb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1be, @Core.IntLiteral.as.ImplicitAs.impl.0da(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.0da(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5ea: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ddb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.724: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.5ea) [concrete]
// CHECK:STDOUT:   %.fa3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.724, %ImplicitAs.facet.5ea [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.e75: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %U.7ce: %ImplicitAs.type.57f = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.8de486.1: %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.7ce) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.8de486.2: %Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ef6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.968 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f1f: %ImplicitAs.type.57f = facet_value %i32, (%ImplicitAs.impl_witness.ef6) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.626: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.770, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.d09936.1: %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.d09936.2: %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.771: %AddAssignWith.type.e75 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.626) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.5ac: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.771) [concrete]
// CHECK:STDOUT:   %.6ba: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.5ac, %AddAssignWith.facet.771 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.85b3b1.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d09936.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.47d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %.94f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.47d, %ImplicitAs.facet.f1f [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.85b3b1.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d09936.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.f1f) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.69e = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.8d5 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7f3: @Core.IntLiteral.as.ImplicitAs.impl.0da.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0da.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1be = impl_witness_table (%Core.import_ref.7f3), @Core.IntLiteral.as.ImplicitAs.impl.0da [concrete]
// CHECK:STDOUT:   %Core.import_ref.dd6: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.8de486.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.770 = impl_witness_table (%Core.import_ref.dd6), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c23: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.9b7323.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.8de486.2)]
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.968 = impl_witness_table (%Core.import_ref.67b339.2), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndRuntimeI32() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a99 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a99 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.c6b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc9: %.fa3 = impl_witness_access constants.%ImplicitAs.impl_witness.ddb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_16.2(%int_1.loc9) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc9_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc9_16.2: %i32 = converted %int_1.loc9, %.loc9_16.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %b: %i32 = value_binding b, %.loc9_16.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.6ba = impl_witness_access constants.%AddAssignWith.impl_witness.626, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d09936.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.57f = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.ef6) [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.57f = converted constants.%i32, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet.f1f]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.85b3b1.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.1 = specific_constant imports.%Core.Op.c23, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d09936.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.e55ec8.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d09936.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = converted %b.ref, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.f1f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.85b3b1.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.ef6, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long = call %bound_method.loc10_8(%b.ref)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref, %.loc10_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.a99: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.57f: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.69e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.57f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.337: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.337, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.dfc: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.d83: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.eef: type = fn_type_with_self_type %Inc.WithSelf.Op.type.d83, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.type: type = fn_type @Cpp.long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op: %Cpp.long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.ed1: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.e44: type = fn_type_with_self_type %Dec.WithSelf.Op.type.ed1, %Dec.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.type: type = fn_type @Cpp.long.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op: %Cpp.long.as.Dec.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.69e = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.8d5 [concrete]
// CHECK:STDOUT:   %Core.import_ref.59e3cf.1: %Cpp.long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.59e3cf.1), @Cpp.long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.59e3cf.2: %Cpp.long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.59e3cf.2), @Cpp.long.as.Dec.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IncrementDecrementLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a99 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a99 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.73b = impl_witness_access constants.%ImplicitAs.impl_witness.300, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.dfc]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc9: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc9: %.eef = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.e44 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10(%a.ref.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
// CHECK:STDOUT:   %pattern_type.6a9: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7f1: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4bd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fa6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ae5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.753: %ImplicitAs.type.7f1 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fa6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.d2d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet.753) [concrete]
// CHECK:STDOUT:   %.4ee: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.d2d, %ImplicitAs.facet.753 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.510: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongResult: type = class_type @ULongResult [concrete]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULong.cpp_overload_set.type: type = cpp_overload_set_type @PassULong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassULong.cpp_overload_set.value: %PassULong.cpp_overload_set.type = cpp_overload_set_value @PassULong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.cd3: type = ptr_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.type.f4cf84.1: type = fn_type @PassULong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.8dc497.1: %PassULong__carbon_thunk.type.f4cf84.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.632: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.79f: type = facet_type <@ImplicitAs, @ImplicitAs(%u32)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a01: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.f1e(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.d3a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a01 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0aa: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c97, @Core.IntLiteral.as.ImplicitAs.impl.f1e(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.025: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.f1e(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.376: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.025 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.09c: %ImplicitAs.type.79f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.0aa) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.715: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u32, %ImplicitAs.facet.09c) [concrete]
// CHECK:STDOUT:   %.96f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.715, %ImplicitAs.facet.09c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d64: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.376 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.376, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1f9: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.fbc: %u32 = int_value 1 [concrete]
// CHECK:STDOUT:   %UIntResult: type = class_type @UIntResult [concrete]
// CHECK:STDOUT:   %pattern_type.d4a: type = pattern_type %UIntResult [concrete]
// CHECK:STDOUT:   %ptr.c6b: type = ptr_type %UIntResult [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.type.f4cf84.2: type = fn_type @PassULong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.8dc497.2: %PassULong__carbon_thunk.type.f4cf84.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.750: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.fa8: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.fa8 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.5e7: type = facet_type <@As, @As(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.3f6: <witness> = impl_witness imports.%As.impl_witness_table.c36 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.5e7 = facet_value Core.IntLiteral, (%As.impl_witness.3f6) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e10: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.98d: type = fn_type_with_self_type %As.WithSelf.Convert.type.e10, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5ff: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.47e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3f4: %ImplicitAs.type.4bd = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.5ff) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.bd3: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.3f4) [concrete]
// CHECK:STDOUT:   %.4cd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.bd3, %ImplicitAs.facet.3f4 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.510, %Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.750 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.ee6: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1dd: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.750)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bfc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.3c0, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b89: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b89 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9f5: %ImplicitAs.type.1dd = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bfc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.68e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.750, %ImplicitAs.facet.9f5) [concrete]
// CHECK:STDOUT:   %.28b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.68e, %ImplicitAs.facet.9f5 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.5a8: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.467: %f32.750 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.467) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.3: type = fn_type @Destroy.Op.loc24_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.3: %Destroy.Op.type.60afa5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %UIntResult.cpp_destructor.type: type = fn_type @UIntResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %UIntResult.cpp_destructor: %UIntResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor.type: type = fn_type @ULongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor: %ULongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .CppCompat = %CppCompat.bd5028.1
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
// CHECK:STDOUT:   %CppCompat.bd5028.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .ULong32 = %Core.ULong32
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ULong32: type = import_ref Core//prelude/types/cpp/int, ULong32, loaded [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.67b339.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ae5 = impl_witness_table (%Core.import_ref.67b339.1), @Core.IntLiteral.as.ImplicitAs.impl.4f2 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULong.cpp_overload_set.value: %PassULong.cpp_overload_set.type = cpp_overload_set_value @PassULong.cpp_overload_set [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULong__carbon_thunk.decl.43aa98.1: %PassULong__carbon_thunk.type.f4cf84.1 = fn_decl @PassULong__carbon_thunk.1 [concrete = constants.%PassULong__carbon_thunk.8dc497.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.0ae: @Core.IntLiteral.as.ImplicitAs.impl.f1e.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a01) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.f1e.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.d3a)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c97 = impl_witness_table (%Core.import_ref.0ae), @Core.IntLiteral.as.ImplicitAs.impl.f1e [concrete]
// CHECK:STDOUT:   %UIntResult.decl: type = class_decl @UIntResult [concrete = constants.%UIntResult] {} {}
// CHECK:STDOUT:   %PassULong__carbon_thunk.decl.43aa98.2: %PassULong__carbon_thunk.type.f4cf84.2 = fn_decl @PassULong__carbon_thunk.2 [concrete = constants.%PassULong__carbon_thunk.8dc497.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.fa8 = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.67b339.2: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.c36 = impl_witness_table (%Core.import_ref.67b339.2), @Core.IntLiteral.as.As.impl.391 [concrete]
// CHECK:STDOUT:   %Core.import_ref.67b339.3: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.47e = impl_witness_table (%Core.import_ref.67b339.3), @Cpp.unsigned_long.as.ImplicitAs.impl.c2e [concrete]
// CHECK:STDOUT:   %Core.import_ref.831: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.98a) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e38)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3c0 = impl_witness_table (%Core.import_ref.831), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.6a9 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_29: type = splice_block %unsigned_long.ref.loc15 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc15: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.4ee = impl_witness_access constants.%ImplicitAs.impl_witness.fa6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.unsigned_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.510]
// CHECK:STDOUT:   %.loc15_46.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.510]
// CHECK:STDOUT:   %.loc15_46.2: %Cpp.unsigned_long = converted %int_1.loc15, %.loc15_46.1 [concrete = constants.%int_1.510]
// CHECK:STDOUT:   %cpp_unsigned_long: %Cpp.unsigned_long = value_binding cpp_unsigned_long, %.loc15_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_result.patt: %pattern_type.423 = value_binding_pattern cpp_unsigned_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_58: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc16: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc16: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc16_89.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.cd3 = addr_of %.loc16_89.1
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
// CHECK:STDOUT:     %cpp_compat_ulong.patt: %pattern_type.6a9 = value_binding_pattern cpp_compat_ulong [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc18: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc18: type = splice_block %ULong32.ref [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.bd5028.1 [concrete = imports.%CppCompat.bd5028.1]
// CHECK:STDOUT:     %ULong32.ref: type = name_ref ULong32, imports.%Core.ULong32 [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_ulong: %Cpp.unsigned_long = value_binding cpp_compat_ulong, %cpp_unsigned_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_result.patt: %pattern_type.423 = value_binding_pattern cpp_compat_ulong_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc19: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong.ref: %Cpp.unsigned_long = name_ref cpp_compat_ulong, %cpp_compat_ulong
// CHECK:STDOUT:   %.loc19_87.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.cd3 = addr_of %.loc19_87.1
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
// CHECK:STDOUT:     %carbon_u32.patt: %pattern_type.632 = value_binding_pattern carbon_u32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u32: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %impl.elem0.loc21: %.96f = impl_witness_access constants.%ImplicitAs.impl_witness.0aa, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.376]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d64]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.1f9]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %u32 = call %bound_method.loc21_25.2(%int_1.loc21) [concrete = constants.%int_1.fbc]
// CHECK:STDOUT:   %.loc21_25.1: %u32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.fbc]
// CHECK:STDOUT:   %.loc21_25.2: %u32 = converted %int_1.loc21, %.loc21_25.1 [concrete = constants.%int_1.fbc]
// CHECK:STDOUT:   %carbon_u32: %u32 = value_binding carbon_u32, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u32_result.patt: %pattern_type.d4a = value_binding_pattern carbon_u32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc22: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u32.ref: %u32 = name_ref carbon_u32, %carbon_u32
// CHECK:STDOUT:   %.loc22_74.1: ref %UIntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.c6b = addr_of %.loc22_74.1
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
// CHECK:STDOUT:     %a.patt: %pattern_type.ee6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ee6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_59: %.28b = impl_witness_access constants.%ImplicitAs.impl_witness.bfc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b05]
// CHECK:STDOUT:   %bound_method.loc24_59.1: <bound method> = bound_method %float, %impl.elem0.loc24_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_59.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.5a8]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.750 = call %bound_method.loc24_59.2(%float) [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc24_59.2: init %f32.750 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.467]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_59.3: ref %f32.750 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_59.4: init %f32.750 to %.loc24_59.3 = in_place_init %.loc24_59.2 [concrete = constants.%float.467]
// CHECK:STDOUT:   %.loc24_59.5: init %array_type to %a.var = array_init (%.loc24_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_59.1, %.loc24_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_50: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.750 [concrete = constants.%f32.750]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc24: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.98d = impl_witness_access constants.%As.impl_witness.3f6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.510]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.510]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.510]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.4cd = impl_witness_access constants.%ImplicitAs.impl_witness.5ff, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.3
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
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.2(%self.param: ref %f32.750) {
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
// CHECK:STDOUT:   %pattern_type.6a9: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7f1: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fa6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ae5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.7f1 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fa6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.d2d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.4ee: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.d2d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.510: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.285: <witness> = impl_witness imports.%Copy.impl_witness_table.5bf [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long, (%Copy.impl_witness.285) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.53a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.802: type = fn_type_with_self_type %Copy.WithSelf.Op.type.53a, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op: %Cpp.unsigned_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.67b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ae5 = impl_witness_table (%Core.import_ref.67b), @Core.IntLiteral.as.ImplicitAs.impl.4f2 [concrete]
// CHECK:STDOUT:   %Core.import_ref.59e: %Cpp.unsigned_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.5bf = impl_witness_table (%Core.import_ref.59e), @Cpp.unsigned_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyUnsignedLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6a9 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.6a9 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.unsigned_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.4ee = impl_witness_access constants.%ImplicitAs.impl_witness.fa6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.510]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.unsigned_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.510]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %unsigned_long.ref.loc8 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.unsigned_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.6a9 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.6a9 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.unsigned_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.unsigned_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_37: %Cpp.unsigned_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.802 = impl_witness_access constants.%Copy.impl_witness.285, element0 [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_37, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long = call %bound_method.loc9(%.loc9_37)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %unsigned_long.ref.loc9 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc9: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.b63e49.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.b63e49.2
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
