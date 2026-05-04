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
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2ad: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.7d6: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.81b: %Int.as.ImplicitAs.impl.Convert.type.7d6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0a6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82d, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.290: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5db: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.0a6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.261: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.5db) [concrete]
// CHECK:STDOUT:   %.61e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.261, %ImplicitAs.facet.5db [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.290 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.290, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.ec2: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.229: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.38c: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8f0: %Core.IntLiteral.as.As.impl.Convert.type.38c = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.882: <witness> = impl_witness imports.%As.impl_witness_table.2cc, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.74c: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.400: %Core.IntLiteral.as.As.impl.Convert.type.74c = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.229 = facet_value Core.IntLiteral, (%As.impl_witness.882) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.80c: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i64, %As.facet) [concrete]
// CHECK:STDOUT:   %.53e: type = fn_type_with_self_type %As.WithSelf.Convert.type.80c, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.400 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.400, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.665: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.570: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.577, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.a0c: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.5aa: %Int.as.ImplicitAs.impl.Convert.type.a0c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.714: %ImplicitAs.type.139 = facet_value %i64, (%ImplicitAs.impl_witness.570) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f63: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.714) [concrete]
// CHECK:STDOUT:   %.c89: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f63, %ImplicitAs.facet.714 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.41a, %Int.as.ImplicitAs.impl.Convert.5aa [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.5aa, @Int.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.a75: <bound method> = bound_method %int_1.41a, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.332: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c17, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f0e: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.332) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3b5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.97e, %ImplicitAs.facet.f0e) [concrete]
// CHECK:STDOUT:   %.3bc: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3b5, %ImplicitAs.facet.f0e [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6b7: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = @F.%i64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e0f: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.7d6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.81b)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.577 = impl_witness_table (%Core.import_ref.e0f), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d74: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.38c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.8f0)]
// CHECK:STDOUT:   %As.impl_witness_table.2cc = impl_witness_table (%Core.import_ref.d74), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1df: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c17 = impl_witness_table (%Core.import_ref.1df), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.95b = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_25: type = splice_block %long_long.ref.loc8 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.61e = impl_witness_access constants.%ImplicitAs.impl_witness.0a6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.290]
// CHECK:STDOUT:   %bound_method.loc8_38.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_38.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.ec2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc8_38.2(%int_1.loc8) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_38.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_38.2: %i64 = converted %int_1.loc8, %.loc8_38.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %cpp_long_long: %i64 = value_binding cpp_long_long, %.loc8_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_long.patt: %pattern_type.95b = value_binding_pattern carbon_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref: %i64 = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %i64.loc9: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long_long: %i64 = value_binding carbon_long_long, %cpp_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_55: %.3bc = impl_witness_access constants.%ImplicitAs.impl_witness.332, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d]
// CHECK:STDOUT:   %bound_method.loc11_55.1: <bound method> = bound_method %float, %impl.elem0.loc11_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_55: <specific function> = specific_function %impl.elem0.loc11_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_55.2: <bound method> = bound_method %float, %specific_fn.loc11_55 [concrete = constants.%bound_method.6b7]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc11_55.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_55.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_55.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_55.4: init %f32.97e to %.loc11_55.3 = in_place_init %.loc11_55.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_55.5: init %array_type to %a.var = array_init (%.loc11_55.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_55.1, %.loc11_55.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_46: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc11: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.53e = impl_witness_access constants.%As.impl_witness.882, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.400]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.665]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc11_30.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc11_30.2: %i64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.c89 = impl_witness_access constants.%ImplicitAs.impl_witness.570, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.5aa]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.a75]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.2(%self.param: ref %f32.97e) {
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
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.584: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.b06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.cfa: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.dd2(%From.fe9) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.1ca: %UInt.as.ImplicitAs.impl.Convert.type.cfa = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e55: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8a3, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f2f: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.34c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f2f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.32a: %ImplicitAs.type.584 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e55) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.d41: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.32a) [concrete]
// CHECK:STDOUT:   %.d53: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.d41, %ImplicitAs.facet.32a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.34c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.34c, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.4a7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f23: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.f20: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.42e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.93f: %Core.IntLiteral.as.As.impl.Convert.type.42e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.542: <witness> = impl_witness imports.%As.impl_witness_table.7ea, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.9b6: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.873: %Core.IntLiteral.as.As.impl.Convert.type.9b6 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.f20 = facet_value Core.IntLiteral, (%As.impl_witness.542) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.8c0: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u64, %As.facet) [concrete]
// CHECK:STDOUT:   %.3ab: type = fn_type_with_self_type %As.WithSelf.Convert.type.8c0, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.873 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.873, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.cd1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.260: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.166, @UInt.as.ImplicitAs.impl.dd2(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.6bc: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.dd2(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.c67: %UInt.as.ImplicitAs.impl.Convert.type.6bc = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.459: %ImplicitAs.type.139 = facet_value %u64, (%ImplicitAs.impl_witness.260) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.5a2: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.459) [concrete]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.5a2, %ImplicitAs.facet.459 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.f23, %UInt.as.ImplicitAs.impl.Convert.c67 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.c67, @UInt.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.341: <bound method> = bound_method %int_1.f23, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.332: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c17, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f0e: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.332) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3b5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.97e, %ImplicitAs.facet.f0e) [concrete]
// CHECK:STDOUT:   %.3bc: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3b5, %ImplicitAs.facet.f0e [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6b7: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long_long = @F.%u64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.2b3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.b06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8a3 = impl_witness_table (%Core.import_ref.2b3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.dec6: @UInt.as.ImplicitAs.impl.dd2.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.cfa) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.dd2.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.1ca)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.166 = impl_witness_table (%Core.import_ref.dec6), @UInt.as.ImplicitAs.impl.dd2 [concrete]
// CHECK:STDOUT:   %Core.import_ref.539: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.42e) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.93f)]
// CHECK:STDOUT:   %As.impl_witness_table.7ea = impl_witness_table (%Core.import_ref.539), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1df4: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c17 = impl_witness_table (%Core.import_ref.1df4), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.157 = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_34: type = splice_block %unsigned_long_long.ref.loc8 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long_long.ref.loc8: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.d53 = impl_witness_access constants.%ImplicitAs.impl_witness.e55, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.34c]
// CHECK:STDOUT:   %bound_method.loc8_56.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_56.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.4a7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc8_56.2(%int_1.loc8) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc8_56.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc8_56.2: %u64 = converted %int_1.loc8, %.loc8_56.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = value_binding cpp_unsigned_long_long, %.loc8_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long_long.patt: %pattern_type.157 = value_binding_pattern carbon_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref: %u64 = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %u64.loc9: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long_long: %u64 = value_binding carbon_unsigned_long_long, %cpp_unsigned_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_64.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_64: %.3bc = impl_witness_access constants.%ImplicitAs.impl_witness.332, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d]
// CHECK:STDOUT:   %bound_method.loc11_64.1: <bound method> = bound_method %float, %impl.elem0.loc11_64 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_64: <specific function> = specific_function %impl.elem0.loc11_64, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_64.2: <bound method> = bound_method %float, %specific_fn.loc11_64 [concrete = constants.%bound_method.6b7]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc11_64.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_64.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_64.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_64.4: init %f32.97e to %.loc11_64.3 = in_place_init %.loc11_64.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_64.5: init %array_type to %a.var = array_init (%.loc11_64.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_64.1, %.loc11_64.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_55: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc11: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.3ab = impl_witness_access constants.%As.impl_witness.542, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.873]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.cd1]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc11_30.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc11_30.2: %u64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.cb4 = impl_witness_access constants.%ImplicitAs.impl_witness.260, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.c67]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @UInt.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.341]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.2(%self.param: ref %f32.97e) {
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
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2ad: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.826: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2b4: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.826) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.806: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %.ed7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.806, %ImplicitAs.facet.2b4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.124: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.124 [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %LongResult: type = class_type @LongResult [concrete]
// CHECK:STDOUT:   %pattern_type.cd9: type = pattern_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLong.cpp_overload_set.type: type = cpp_overload_set_type @PassLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassLong.cpp_overload_set.value: %PassLong.cpp_overload_set.type = cpp_overload_set_value @PassLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.305: type = ptr_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.type.9d9430.1: type = fn_type @PassLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.58a8f3.1: %PassLong__carbon_thunk.type.9d9430.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %IntResult: type = class_type @IntResult [concrete]
// CHECK:STDOUT:   %pattern_type.454: type = pattern_type %IntResult [concrete]
// CHECK:STDOUT:   %ptr.3cb: type = ptr_type %IntResult [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.type.9d9430.2: type = fn_type @PassLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassLong__carbon_thunk.58a8f3.2: %PassLong__carbon_thunk.type.9d9430.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.133: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.be6 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.22c: %ImplicitAs.type.2ad = facet_value %Cpp.long, (%ImplicitAs.impl_witness.133) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.afa: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.22c) [concrete]
// CHECK:STDOUT:   %.d94: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.afa, %ImplicitAs.facet.22c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.1eb: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.476: %Cpp.long.as.ImplicitAs.impl.Convert.type.1eb = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.508: type = facet_type <@As, @As(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.ee8: <witness> = impl_witness imports.%As.impl_witness_table.6bb [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.508 = facet_value Core.IntLiteral, (%As.impl_witness.ee8) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.41e: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long, %As.facet) [concrete]
// CHECK:STDOUT:   %.8b3: type = fn_type_with_self_type %As.WithSelf.Convert.type.41e, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.48a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d05 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.78b: %ImplicitAs.type.139 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.48a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.561: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.78b) [concrete]
// CHECK:STDOUT:   %.e0b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.561, %ImplicitAs.facet.78b [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.f44: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.f90: %Cpp.long.as.ImplicitAs.impl.Convert.type.f44 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5a4, %Cpp.long.as.ImplicitAs.impl.Convert.f90 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.332: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c17, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f0e: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.332) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3b5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.97e, %ImplicitAs.facet.f0e) [concrete]
// CHECK:STDOUT:   %.3bc: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3b5, %ImplicitAs.facet.f0e [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc26_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT:   %IntResult.cpp_destructor.type: type = fn_type @IntResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %IntResult.cpp_destructor: %IntResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor.type: type = fn_type @LongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor: %LongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .CppCompat = %CppCompat.4b4
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
// CHECK:STDOUT:   %CppCompat.4b4: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .Long32 = %Core.Long32
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Long32: type = import_ref Core//prelude/types/cpp/int, Long32, loaded [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.ImplicitAs.impl.2da [concrete]
// CHECK:STDOUT:   %LongResult.decl: type = class_decl @LongResult [concrete = constants.%LongResult] {} {}
// CHECK:STDOUT:   %PassLong.cpp_overload_set.value: %PassLong.cpp_overload_set.type = cpp_overload_set_value @PassLong.cpp_overload_set [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassLong__carbon_thunk.decl.d16229.1: %PassLong__carbon_thunk.type.9d9430.1 = fn_decl @PassLong__carbon_thunk.1 [concrete = constants.%PassLong__carbon_thunk.58a8f3.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %IntResult.decl: type = class_decl @IntResult [concrete = constants.%IntResult] {} {}
// CHECK:STDOUT:   %PassLong__carbon_thunk.decl.d16229.2: %PassLong__carbon_thunk.type.9d9430.2 = fn_decl @PassLong__carbon_thunk.2 [concrete = constants.%PassLong__carbon_thunk.58a8f3.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.493: %Cpp.long.as.ImplicitAs.impl.Convert.type.1eb = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.476]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.be6 = impl_witness_table (%Core.import_ref.493), @Cpp.long.as.ImplicitAs.impl.887 [concrete]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.cac: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.6bb = impl_witness_table (%Core.import_ref.cac), @Core.IntLiteral.as.As.impl.2c7 [concrete]
// CHECK:STDOUT:   %Core.import_ref.14e: %Cpp.long.as.ImplicitAs.impl.Convert.type.f44 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.f90]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d05 = impl_witness_table (%Core.import_ref.14e), @Cpp.long.as.ImplicitAs.impl.f4b [concrete]
// CHECK:STDOUT:   %Core.import_ref.1df4: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c17 = impl_witness_table (%Core.import_ref.1df4), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.68c = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_20: type = splice_block %long.ref.loc15 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc15: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc15_28.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc15_28.2: %Cpp.long = converted %int_1.loc15, %.loc15_28.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %cpp_long: %Cpp.long = value_binding cpp_long, %.loc15_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_result.patt: %pattern_type.cd9 = value_binding_pattern cpp_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc16: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long.ref.loc16: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc16_69.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.305 = addr_of %.loc16_69.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.1(%cpp_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_69.2: init %LongResult to %.loc16_69.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_34: type = splice_block %LongResult.ref.loc16 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc16: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_69.3: ref %LongResult = temporary %.loc16_69.1, %.loc16_69.2
// CHECK:STDOUT:   %.loc16_69.4: %LongResult = acquire_value %.loc16_69.3
// CHECK:STDOUT:   %cpp_long_result: %LongResult = value_binding cpp_long_result, %.loc16_69.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long.patt: %pattern_type.68c = value_binding_pattern cpp_compat_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc18: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc18: type = splice_block %Long32.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.4b4 [concrete = imports.%CppCompat.4b4]
// CHECK:STDOUT:     %Long32.ref: type = name_ref Long32, imports.%Core.Long32 [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long: %Cpp.long = value_binding cpp_compat_long, %cpp_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_result.patt: %pattern_type.cd9 = value_binding_pattern cpp_compat_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_55: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc19: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long.ref: %Cpp.long = name_ref cpp_compat_long, %cpp_compat_long
// CHECK:STDOUT:   %.loc19_83.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.305 = addr_of %.loc19_83.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.1(%cpp_compat_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_83.2: init %LongResult to %.loc19_83.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_41: type = splice_block %LongResult.ref.loc19 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc19: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_83.3: ref %LongResult = temporary %.loc19_83.1, %.loc19_83.2
// CHECK:STDOUT:   %.loc19_83.4: %LongResult = acquire_value %.loc19_83.3
// CHECK:STDOUT:   %cpp_compat_long_result: %LongResult = value_binding cpp_compat_long_result, %.loc19_83.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32.patt: %pattern_type.7ce = value_binding_pattern carbon_i32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc21: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i32.loc21_37: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc21_34.1: %i32 = as_compatible %cpp_long.ref.loc21
// CHECK:STDOUT:   %.loc21_34.2: %i32 = converted %cpp_long.ref.loc21, %.loc21_34.1
// CHECK:STDOUT:   %i32.loc21_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %carbon_i32: %i32 = value_binding carbon_i32, %.loc21_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32_result.patt: %pattern_type.454 = value_binding_pattern carbon_i32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_49: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc22: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i32.ref: %i32 = name_ref carbon_i32, %carbon_i32
// CHECK:STDOUT:   %.loc22_72.1: ref %IntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.3cb = addr_of %.loc22_72.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.2(%carbon_i32.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_72.2: init %IntResult to %.loc22_72.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_36: type = splice_block %IntResult.ref [concrete = constants.%IntResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntResult.ref: type = name_ref IntResult, imports.%IntResult.decl [concrete = constants.%IntResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_72.3: ref %IntResult = temporary %.loc22_72.1, %.loc22_72.2
// CHECK:STDOUT:   %.loc22_72.4: %IntResult = acquire_value %.loc22_72.3
// CHECK:STDOUT:   %carbon_i32_result: %IntResult = value_binding carbon_i32_result, %.loc22_72.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.95b = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc24: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc24: %.d94 = impl_witness_access constants.%ImplicitAs.impl_witness.133, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.476]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %cpp_long.ref.loc24, %impl.elem0.loc24
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc24: init %i64 = call %bound_method.loc24(%cpp_long.ref.loc24)
// CHECK:STDOUT:   %.loc24_32.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc24
// CHECK:STDOUT:   %.loc24_32.2: %i64 = converted %cpp_long.ref.loc24, %.loc24_32.1
// CHECK:STDOUT:   %carbon_i64: %i64 = value_binding carbon_i64, %.loc24_32.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc26_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc26_50: %.3bc = impl_witness_access constants.%ImplicitAs.impl_witness.332, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d]
// CHECK:STDOUT:   %bound_method.loc26_50.1: <bound method> = bound_method %float, %impl.elem0.loc26_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc26_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_50.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc26_50.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26_50.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc26_50.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc26_50.4: init %f32.97e to %.loc26_50.3 = in_place_init %.loc26_50.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26_50.5: init %array_type to %a.var = array_init (%.loc26_50.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc26_3: init %array_type = converted %.loc26_50.1, %.loc26_50.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc26_3
// CHECK:STDOUT:   %.loc26_41: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc26: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:     %impl.elem0.loc26_30.1: %.8b3 = impl_witness_access constants.%As.impl_witness.ee8, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc26_30.1: <bound method> = bound_method %int_1.loc26, %impl.elem0.loc26_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long = call %bound_method.loc26_30.1(%int_1.loc26) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %.loc26_30.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %.loc26_30.2: %Cpp.long = converted %int_1.loc26, %.loc26_30.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %impl.elem0.loc26_30.2: %.e0b = impl_witness_access constants.%ImplicitAs.impl_witness.48a, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.f90]
// CHECK:STDOUT:     %bound_method.loc26_30.2: <bound method> = bound_method %.loc26_30.2, %impl.elem0.loc26_30.2 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long.as.ImplicitAs.impl.Convert.call.loc26: init Core.IntLiteral = call %bound_method.loc26_30.2(%.loc26_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc26_30.3: Core.IntLiteral = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc26 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc26_30.4: Core.IntLiteral = converted %.loc26_30.2, %.loc26_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc26_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.3
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
// CHECK:STDOUT: fn @Destroy.Op.loc26_3.2(%self.param: ref %f32.97e) {
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
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.826: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.826) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.806: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ed7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.806, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.fb1: <witness> = impl_witness imports.%Copy.impl_witness_table.49f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.fb1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f18: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.de4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f18, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.ImplicitAs.impl.2da [concrete]
// CHECK:STDOUT:   %Core.import_ref.a6a: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.49f = impl_witness_table (%Core.import_ref.a6a), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.68c = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.68c = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_28: %Cpp.long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.de4 = impl_witness_access constants.%Copy.impl_witness.fb1, element0 [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_28, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.call: init %Cpp.long = call %bound_method.loc9(%.loc9_28)
// CHECK:STDOUT:   assign %b.var, %Cpp.long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
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
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.826: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.826) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.806: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ed7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.806, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.494: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.060: <witness> = impl_witness imports.%EqWith.impl_witness_table.e7b [concrete]
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.494 = facet_value %Cpp.long, (%EqWith.impl_witness.060) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.17e: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.ed4: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet) [concrete]
// CHECK:STDOUT:   %.e84: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.17e, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.be1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.3d5: %Cpp.long.as.EqWith.impl.Equal.type.be1 = struct_value () [concrete]
// CHECK:STDOUT:   %.681: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.ed4, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.35d: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.cf9: %Cpp.long.as.EqWith.impl.NotEqual.type.35d = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.a39: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.109: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.a70 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.a39 = facet_value %Cpp.long, (%OrderedWith.impl_witness.109) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.2ac: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.653: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.df9: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.6f9: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %.49a: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.df9, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.9ce: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.1b4: %Cpp.long.as.OrderedWith.impl.Greater.type.9ce = struct_value () [concrete]
// CHECK:STDOUT:   %.1e2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.2ac, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.7ac: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.4e6: %Cpp.long.as.OrderedWith.impl.Less.type.7ac = struct_value () [concrete]
// CHECK:STDOUT:   %.eeb: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.6f9, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.237: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.9de: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.237 = struct_value () [concrete]
// CHECK:STDOUT:   %.770: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.653, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.945: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.530: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.945 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.ImplicitAs.impl.2da [concrete]
// CHECK:STDOUT:   %Core.import_ref.f20: %Cpp.long.as.EqWith.impl.Equal.type.be1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.3d5]
// CHECK:STDOUT:   %Core.import_ref.274: %Cpp.long.as.EqWith.impl.NotEqual.type.35d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.cf9]
// CHECK:STDOUT:   %EqWith.impl_witness_table.e7b = impl_witness_table (%Core.import_ref.f20, %Core.import_ref.274), @Cpp.long.as.EqWith.impl.46d [concrete]
// CHECK:STDOUT:   %Core.import_ref.4b5: %Cpp.long.as.OrderedWith.impl.Less.type.7ac = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.4e6]
// CHECK:STDOUT:   %Core.import_ref.111: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.945 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.530]
// CHECK:STDOUT:   %Core.import_ref.3f9: %Cpp.long.as.OrderedWith.impl.Greater.type.9ce = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.1b4]
// CHECK:STDOUT:   %Core.import_ref.818: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.237 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.9de]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.a70 = impl_witness_table (%Core.import_ref.4b5, %Core.import_ref.111, %Core.import_ref.3f9, %Core.import_ref.818), @Cpp.long.as.OrderedWith.impl.8d8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc8_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc8_21.2: %Cpp.long = converted %int_1.loc8, %.loc8_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc8_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc9_21.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.e84 = impl_witness_access constants.%EqWith.impl_witness.060, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.3d5]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.681 = impl_witness_access constants.%EqWith.impl_witness.060, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.cf9]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.49a = impl_witness_access constants.%OrderedWith.impl_witness.109, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.1b4]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.1e2 = impl_witness_access constants.%OrderedWith.impl_witness.109, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.4e6]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.eeb = impl_witness_access constants.%OrderedWith.impl_witness.109, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.9de]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.770 = impl_witness_access constants.%OrderedWith.impl_witness.109, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.530]
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
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.826: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2b4: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.826) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.806: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %.ed7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.806, %ImplicitAs.facet.2b4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.124: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.124 [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.d8f: type = facet_type <@EqWith, @EqWith(%i32)> [concrete]
// CHECK:STDOUT:   %T.57d: %ImplicitAs.type.819 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.0baadc.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.db7(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.8184d1.1: %Cpp.long.as.EqWith.impl.NotEqual.type.0baadc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.0baadc.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.db7(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.8184d1.2: %Cpp.long.as.EqWith.impl.NotEqual.type.0baadc.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.6b4678.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.db7(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.94c81e.1: %Cpp.long.as.EqWith.impl.Equal.type.6b4678.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.6b4678.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.db7(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.94c81e.2: %Cpp.long.as.EqWith.impl.Equal.type.6b4678.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.ded: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.895: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.87f: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.895) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.882: <witness> = impl_witness imports.%EqWith.impl_witness_table.d93, @Cpp.long.as.EqWith.impl.db7(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.0c1c29.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.db7(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.61b583.1: %Cpp.long.as.EqWith.impl.Equal.type.0c1c29.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.4498f4.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.db7(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.34d386.1: %Cpp.long.as.EqWith.impl.NotEqual.type.4498f4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.0c1c29.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.db7(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.61b583.2: %Cpp.long.as.EqWith.impl.Equal.type.0c1c29.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.4498f4.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.db7(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.34d386.2: %Cpp.long.as.EqWith.impl.NotEqual.type.4498f4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.e95: %EqWith.type.d8f = facet_value %Cpp.long, (%EqWith.impl_witness.882) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.d3c: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i32, %EqWith.facet.e95) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.94e: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i32, %EqWith.facet.e95) [concrete]
// CHECK:STDOUT:   %.f70: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.d3c, %EqWith.facet.e95 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.636ecb.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.61b583.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.595: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %.d59: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.595, %ImplicitAs.facet.87f [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.636ecb.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.61b583.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %.a02: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.94e, %EqWith.facet.e95 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.d90630.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.34d386.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.d90630.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.34d386.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.243: type = facet_type <@OrderedWith, @OrderedWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.31aca4.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.f52(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bf96a8.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.31aca4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.31aca4.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.f52(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bf96a8.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.31aca4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.3a0c8c.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.f52(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.389095.1: %Cpp.long.as.OrderedWith.impl.Greater.type.3a0c8c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.3a0c8c.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.f52(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.389095.2: %Cpp.long.as.OrderedWith.impl.Greater.type.3a0c8c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4726d4.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.f52(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.8b3095.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4726d4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4726d4.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.f52(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.8b3095.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4726d4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.d43a56.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.f52(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.d5bc77.1: %Cpp.long.as.OrderedWith.impl.Less.type.d43a56.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.d43a56.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.f52(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.d5bc77.2: %Cpp.long.as.OrderedWith.impl.Less.type.d43a56.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.358: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.767: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.497, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.9905e3.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.39c30a.1: %Cpp.long.as.OrderedWith.impl.Less.type.9905e3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.95c253.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.18e9de.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.95c253.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.1fe2d1.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.c97804.1: %Cpp.long.as.OrderedWith.impl.Greater.type.1fe2d1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ab7cbc.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f1825e.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ab7cbc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.9905e3.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.39c30a.2: %Cpp.long.as.OrderedWith.impl.Less.type.9905e3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.95c253.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.18e9de.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.95c253.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.1fe2d1.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.c97804.2: %Cpp.long.as.OrderedWith.impl.Greater.type.1fe2d1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ab7cbc.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f1825e.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ab7cbc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.fc8: %OrderedWith.type.243 = facet_value %Cpp.long, (%OrderedWith.impl_witness.767) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.1ca: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.fc8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.e56: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.fc8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.ccf: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.fc8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.e78: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.fc8) [concrete]
// CHECK:STDOUT:   %.ad0: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.ccf, %OrderedWith.facet.fc8 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.81ea5d.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.c97804.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.81ea5d.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.c97804.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %.787: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.1ca, %OrderedWith.facet.fc8 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.dc9710.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.39c30a.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.dc9710.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.39c30a.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %.6d3: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.e78, %OrderedWith.facet.fc8 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1aa9fc.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f1825e.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1aa9fc.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f1825e.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %.b6c: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.e56, %OrderedWith.facet.fc8 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.583828.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.18e9de.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.583828.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.18e9de.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.04e: <witness> = impl_witness imports.%EqWith.impl_witness_table.d93, @Cpp.long.as.EqWith.impl.db7(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.cd1e09.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.db7(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.a57345.1: %Cpp.long.as.EqWith.impl.Equal.type.cd1e09.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.c71dc6.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.db7(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.a2f11f.1: %Cpp.long.as.EqWith.impl.NotEqual.type.c71dc6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.cd1e09.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.db7(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.a57345.2: %Cpp.long.as.EqWith.impl.Equal.type.cd1e09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.c71dc6.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.db7(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.a2f11f.2: %Cpp.long.as.EqWith.impl.NotEqual.type.c71dc6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.caa: %EqWith.type.ded = facet_value %Cpp.long, (%EqWith.impl_witness.04e) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.ea1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.caa) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.21c: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.caa) [concrete]
// CHECK:STDOUT:   %.78d: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.ea1, %EqWith.facet.caa [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.f53006.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.a57345.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.f53006.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.a57345.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %.fbe: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.21c, %EqWith.facet.caa [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.e300fb.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.a2f11f.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.e300fb.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.a2f11f.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.565: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.497, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.cbf2b8.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.46ef5c.1: %Cpp.long.as.OrderedWith.impl.Less.type.cbf2b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e0c8d6.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.c9938f.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e0c8d6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.456c1e.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.c0ccd7.1: %Cpp.long.as.OrderedWith.impl.Greater.type.456c1e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.e6cb99.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.29183e.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.e6cb99.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.cbf2b8.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.46ef5c.2: %Cpp.long.as.OrderedWith.impl.Less.type.cbf2b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e0c8d6.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.c9938f.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e0c8d6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.456c1e.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.c0ccd7.2: %Cpp.long.as.OrderedWith.impl.Greater.type.456c1e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.e6cb99.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.f52(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.29183e.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.e6cb99.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.da6: %OrderedWith.type.358 = facet_value %Cpp.long, (%OrderedWith.impl_witness.565) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b35: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.da6) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.edc: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.da6) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.4bd: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.da6) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a79: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.da6) [concrete]
// CHECK:STDOUT:   %.e06: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.4bd, %OrderedWith.facet.da6 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.881f66.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.c0ccd7.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.881f66.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.c0ccd7.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %.0c6: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.b35, %OrderedWith.facet.da6 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.8d8bc3.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.46ef5c.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.8d8bc3.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.46ef5c.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %.982: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.a79, %OrderedWith.facet.da6 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.08a61f.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.29183e.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.08a61f.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.29183e.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %.6be: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.edc, %OrderedWith.facet.da6 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.791394.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.c9938f.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.791394.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.c9938f.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.ImplicitAs.impl.2da [concrete]
// CHECK:STDOUT:   %Core.import_ref.dac: @Cpp.long.as.EqWith.impl.db7.%Cpp.long.as.EqWith.impl.Equal.type.2 (%Cpp.long.as.EqWith.impl.Equal.type.6b4678.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.db7.%Cpp.long.as.EqWith.impl.Equal.2 (constants.%Cpp.long.as.EqWith.impl.Equal.94c81e.1)]
// CHECK:STDOUT:   %Core.import_ref.f7d: @Cpp.long.as.EqWith.impl.db7.%Cpp.long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long.as.EqWith.impl.NotEqual.type.0baadc.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.db7.%Cpp.long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long.as.EqWith.impl.NotEqual.8184d1.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.d93 = impl_witness_table (%Core.import_ref.dac, %Core.import_ref.f7d), @Cpp.long.as.EqWith.impl.db7 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.78e: @Cpp.long.as.EqWith.impl.db7.%Cpp.long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long.as.EqWith.impl.NotEqual.type.0baadc.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long.as.EqWith.impl.db7.%Cpp.long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long.as.EqWith.impl.NotEqual.8184d1.2)]
// CHECK:STDOUT:   %Core.Equal.3d5: @Cpp.long.as.EqWith.impl.db7.%Cpp.long.as.EqWith.impl.Equal.type.1 (%Cpp.long.as.EqWith.impl.Equal.type.6b4678.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long.as.EqWith.impl.db7.%Cpp.long.as.EqWith.impl.Equal.1 (constants.%Cpp.long.as.EqWith.impl.Equal.94c81e.2)]
// CHECK:STDOUT:   %Core.import_ref.990: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31b = impl_witness_table (%Core.import_ref.990), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.3f2: @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.Less.type.2 (%Cpp.long.as.OrderedWith.impl.Less.type.d43a56.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long.as.OrderedWith.impl.Less.d5bc77.1)]
// CHECK:STDOUT:   %Core.import_ref.c86b: @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4726d4.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.8b3095.1)]
// CHECK:STDOUT:   %Core.import_ref.5e8: @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long.as.OrderedWith.impl.Greater.type.3a0c8c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long.as.OrderedWith.impl.Greater.389095.1)]
// CHECK:STDOUT:   %Core.import_ref.c07: @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.31aca4.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bf96a8.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.497 = impl_witness_table (%Core.import_ref.3f2, %Core.import_ref.c86b, %Core.import_ref.5e8, %Core.import_ref.c07), @Cpp.long.as.OrderedWith.impl.f52 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.fce: @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.31aca4.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bf96a8.2)]
// CHECK:STDOUT:   %Core.Greater.d01: @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long.as.OrderedWith.impl.Greater.type.3a0c8c.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long.as.OrderedWith.impl.Greater.389095.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.0ec: @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4726d4.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.8b3095.2)]
// CHECK:STDOUT:   %Core.Less.08e: @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.Less.type.1 (%Cpp.long.as.OrderedWith.impl.Less.type.d43a56.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.f52.%Cpp.long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long.as.OrderedWith.impl.Less.d5bc77.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc9_21.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.f70 = impl_witness_access constants.%EqWith.impl_witness.882, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.61b583.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.636ecb.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.EqWith.impl.Equal.type.0c1c29.1 = specific_constant imports.%Core.Equal.3d5, @Cpp.long.as.EqWith.impl.db7(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.61b583.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Cpp.long.as.EqWith.impl.Equal.type.0c1c29.1 = name_ref Equal, %.loc10_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.61b583.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.636ecb.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.a02 = impl_witness_access constants.%EqWith.impl_witness.882, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.34d386.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.d90630.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.4498f4.1 = specific_constant imports.%Core.NotEqual.78e, @Cpp.long.as.EqWith.impl.db7(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.34d386.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Cpp.long.as.EqWith.impl.NotEqual.type.4498f4.1 = name_ref NotEqual, %.loc11_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.34d386.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.d90630.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.ad0 = impl_witness_access constants.%OrderedWith.impl_witness.767, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.c97804.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.81ea5d.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.OrderedWith.impl.Greater.type.1fe2d1.1 = specific_constant imports.%Core.Greater.d01, @Cpp.long.as.OrderedWith.impl.f52(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.c97804.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Cpp.long.as.OrderedWith.impl.Greater.type.1fe2d1.1 = name_ref Greater, %.loc12_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.c97804.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.81ea5d.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_7: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_7: init %Cpp.long = call %bound_method.loc12_7(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_7
// CHECK:STDOUT:   %.loc12_7.2: %Cpp.long = converted %b.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.787 = impl_witness_access constants.%OrderedWith.impl_witness.767, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.39c30a.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.dc9710.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.OrderedWith.impl.Less.type.9905e3.1 = specific_constant imports.%Core.Less.08e, @Cpp.long.as.OrderedWith.impl.f52(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.39c30a.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Cpp.long.as.OrderedWith.impl.Less.type.9905e3.1 = name_ref Less, %.loc13_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.39c30a.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.dc9710.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_7: init %Cpp.long = call %bound_method.loc13_7(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_7
// CHECK:STDOUT:   %.loc13_7.2: %Cpp.long = converted %b.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.6d3 = impl_witness_access constants.%OrderedWith.impl_witness.767, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f1825e.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1aa9fc.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ab7cbc.1 = specific_constant imports.%Core.GreaterOrEquivalent.fce, @Cpp.long.as.OrderedWith.impl.f52(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f1825e.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.ab7cbc.1 = name_ref GreaterOrEquivalent, %.loc14_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.f1825e.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1aa9fc.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.b6c = impl_witness_access constants.%OrderedWith.impl_witness.767, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.18e9de.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.583828.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.95c253.1 = specific_constant imports.%Core.LessOrEquivalent.0ec, @Cpp.long.as.OrderedWith.impl.f52(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.18e9de.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.95c253.1 = name_ref LessOrEquivalent, %.loc15_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.18e9de.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_5: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_5: init %Cpp.long = call %bound_method.loc15_5.3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_5
// CHECK:STDOUT:   %.loc15_5.4: %Cpp.long = converted %b.ref.loc15, %.loc15_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.583828.2]
// CHECK:STDOUT:   %bound_method.loc15_5.4: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_8: init %Cpp.long = call %bound_method.loc15_8(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_8
// CHECK:STDOUT:   %.loc15_8.2: %Cpp.long = converted %b.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.4(%a.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.78d = impl_witness_access constants.%EqWith.impl_witness.04e, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.a57345.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.826) [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.f53006.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.EqWith.impl.Equal.type.cd1e09.1 = specific_constant imports.%Core.Equal.3d5, @Cpp.long.as.EqWith.impl.db7(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.a57345.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long.as.EqWith.impl.Equal.type.cd1e09.1 = name_ref Equal, %.loc17_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.a57345.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long = call %bound_method.loc17_5.3(%int_1.loc17) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long = converted %int_1.loc17, %.loc17_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.f53006.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.fbe = impl_witness_access constants.%EqWith.impl_witness.04e, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a2f11f.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.826) [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.e300fb.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.c71dc6.1 = specific_constant imports.%Core.NotEqual.78e, @Cpp.long.as.EqWith.impl.db7(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a2f11f.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long.as.EqWith.impl.NotEqual.type.c71dc6.1 = name_ref NotEqual, %.loc18_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.a2f11f.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long = call %bound_method.loc18_5.3(%int_1.loc18) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long = converted %int_1.loc18, %.loc18_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.e300fb.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.e06 = impl_witness_access constants.%OrderedWith.impl_witness.565, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.c0ccd7.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.826) [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.881f66.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.OrderedWith.impl.Greater.type.456c1e.1 = specific_constant imports.%Core.Greater.d01, @Cpp.long.as.OrderedWith.impl.f52(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.c0ccd7.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long.as.OrderedWith.impl.Greater.type.456c1e.1 = name_ref Greater, %.loc19_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.c0ccd7.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long = call %bound_method.loc19_5.3(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long = converted %int_1.loc19, %.loc19_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.881f66.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_7: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7: init %Cpp.long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.0c6 = impl_witness_access constants.%OrderedWith.impl_witness.565, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.46ef5c.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.826) [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.8d8bc3.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.OrderedWith.impl.Less.type.cbf2b8.1 = specific_constant imports.%Core.Less.08e, @Cpp.long.as.OrderedWith.impl.f52(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.46ef5c.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long.as.OrderedWith.impl.Less.type.cbf2b8.1 = name_ref Less, %.loc20_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.46ef5c.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long = call %bound_method.loc20_5.3(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long = converted %int_1.loc20, %.loc20_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.8d8bc3.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7: init %Cpp.long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.982 = impl_witness_access constants.%OrderedWith.impl_witness.565, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.29183e.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.826) [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.08a61f.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.e6cb99.1 = specific_constant imports.%Core.GreaterOrEquivalent.fce, @Cpp.long.as.OrderedWith.impl.f52(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.29183e.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.e6cb99.1 = name_ref GreaterOrEquivalent, %.loc21_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.29183e.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_5: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5: init %Cpp.long = call %bound_method.loc21_5.3(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_5.4: %Cpp.long = converted %int_1.loc21, %.loc21_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.08a61f.2]
// CHECK:STDOUT:   %bound_method.loc21_5.4: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8: init %Cpp.long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.4(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.6be = impl_witness_access constants.%OrderedWith.impl_witness.565, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.c9938f.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.826) [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc22 [concrete = constants.%ImplicitAs.facet.2b4]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.791394.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e0c8d6.1 = specific_constant imports.%Core.LessOrEquivalent.0ec, @Cpp.long.as.OrderedWith.impl.f52(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.c9938f.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.e0c8d6.1 = name_ref LessOrEquivalent, %.loc22_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.c9938f.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_5: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5: init %Cpp.long = call %bound_method.loc22_5.3(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_5.4: %Cpp.long = converted %int_1.loc22, %.loc22_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.791394.2]
// CHECK:STDOUT:   %bound_method.loc22_5.4: <bound method> = bound_method %a.ref.loc22, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc22_8: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8: init %Cpp.long = call %bound_method.loc22_8(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_8.2: %Cpp.long = converted %int_1.loc22, %.loc22_8.1 [concrete = constants.%int_1.5a4]
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
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.826: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2b4: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.826) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.806: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %.ed7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.806, %ImplicitAs.facet.2b4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.124: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.124 [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.494: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.57d: %ImplicitAs.type.819 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.538ebe.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.600(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.08315d.1: %T.as_type.as.EqWith.impl.NotEqual.type.538ebe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.538ebe.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.600(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.08315d.2: %T.as_type.as.EqWith.impl.NotEqual.type.538ebe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.bfa2df.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.600(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.4c5205.1: %T.as_type.as.EqWith.impl.Equal.type.bfa2df.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.bfa2df.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.600(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.4c5205.2: %T.as_type.as.EqWith.impl.Equal.type.bfa2df.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.895: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.87f: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.895) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.0d0: <witness> = impl_witness imports.%EqWith.impl_witness_table.934, @T.as_type.as.EqWith.impl.600(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.a57597.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.600(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.ceaabd.1: %T.as_type.as.EqWith.impl.Equal.type.a57597.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.278edc.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.600(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.29d3c0.1: %T.as_type.as.EqWith.impl.NotEqual.type.278edc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.a57597.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.600(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.ceaabd.2: %T.as_type.as.EqWith.impl.Equal.type.a57597.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.278edc.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.600(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.29d3c0.2: %T.as_type.as.EqWith.impl.NotEqual.type.278edc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.60b: %EqWith.type.494 = facet_value %i32, (%EqWith.impl_witness.0d0) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.a3c: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.60b) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.f1a: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.60b) [concrete]
// CHECK:STDOUT:   %.7c0: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.a3c, %EqWith.facet.60b [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.d3e448.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.ceaabd.2, @T.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.d3e448.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.ceaabd.1, @T.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.595: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %.d59: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.595, %ImplicitAs.facet.87f [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b7d3: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.f1a, %EqWith.facet.60b [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.8b892c.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.29d3c0.2, @T.as_type.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.8b892c.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.29d3c0.1, @T.as_type.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.a39: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d05c2b.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.9c9(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.11eaa6.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d05c2b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d05c2b.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.9c9(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.11eaa6.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d05c2b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.fca417.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.9c9(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.71b704.1: %T.as_type.as.OrderedWith.impl.Greater.type.fca417.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.fca417.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.9c9(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.71b704.2: %T.as_type.as.OrderedWith.impl.Greater.type.fca417.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.360910.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.9c9(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.fb0920.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.360910.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.360910.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.9c9(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.fb0920.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.360910.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.bbdfd2.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.9c9(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.505b6c.1: %T.as_type.as.OrderedWith.impl.Less.type.bbdfd2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.bbdfd2.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.9c9(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.505b6c.2: %T.as_type.as.OrderedWith.impl.Less.type.bbdfd2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.44c: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.df3, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.785f15.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.68034e.1: %T.as_type.as.OrderedWith.impl.Less.type.785f15.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.855c21.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.1e14ba.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.855c21.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.400b07.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.dedd14.1: %T.as_type.as.OrderedWith.impl.Greater.type.400b07.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c4f42.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5c44ac.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c4f42.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.785f15.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.68034e.2: %T.as_type.as.OrderedWith.impl.Less.type.785f15.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.855c21.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.1e14ba.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.855c21.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.400b07.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.dedd14.2: %T.as_type.as.OrderedWith.impl.Greater.type.400b07.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c4f42.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5c44ac.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c4f42.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.5a8: %OrderedWith.type.a39 = facet_value %i32, (%OrderedWith.impl_witness.44c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.41c: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.5a8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.664: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.5a8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.fa1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.5a8) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.415: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.5a8) [concrete]
// CHECK:STDOUT:   %.b39: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.fa1, %OrderedWith.facet.5a8 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.24c510.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.dedd14.2, @T.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.24c510.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.dedd14.1, @T.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %.920: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.41c, %OrderedWith.facet.5a8 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.3b347e.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.68034e.2, @T.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.3b347e.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.68034e.1, @T.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %.53c: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.415, %OrderedWith.facet.5a8 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.728585.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5c44ac.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.728585.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5c44ac.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %.9da: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.664, %OrderedWith.facet.5a8 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.00439e.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.1e14ba.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.00439e.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.1e14ba.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.fc3: <witness> = impl_witness imports.%EqWith.impl_witness_table.934, @T.as_type.as.EqWith.impl.600(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.76caa0.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.600(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.3b534c.1: %T.as_type.as.EqWith.impl.Equal.type.76caa0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.7f7232.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.600(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.50bdbe.1: %T.as_type.as.EqWith.impl.NotEqual.type.7f7232.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.76caa0.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.600(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.3b534c.2: %T.as_type.as.EqWith.impl.Equal.type.76caa0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.7f7232.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.600(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.50bdbe.2: %T.as_type.as.EqWith.impl.NotEqual.type.7f7232.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.44a: %EqWith.type.494 = facet_value Core.IntLiteral, (%EqWith.impl_witness.fc3) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.0cc: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.44a) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.132: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.44a) [concrete]
// CHECK:STDOUT:   %.9d7: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.0cc, %EqWith.facet.44a [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.ef477d.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.3b534c.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.5238fc.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.3b534c.2, @T.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.f4cdf1.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.5238fc.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.ef477d.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.3b534c.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.5238fc.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.3b534c.1, @T.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.f4cdf1.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.5238fc.2 [concrete]
// CHECK:STDOUT:   %.802: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.132, %EqWith.facet.44a [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.1ef7f5.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.50bdbe.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.218ac5.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.50bdbe.2, @T.as_type.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.a4d1ea.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.218ac5.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.1ef7f5.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.50bdbe.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.218ac5.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.50bdbe.1, @T.as_type.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.a4d1ea.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.218ac5.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.694: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.df3, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.34c005.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.76f883.1: %T.as_type.as.OrderedWith.impl.Less.type.34c005.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.ea983f.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.52cd50.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.ea983f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.01adec.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.b936e0.1: %T.as_type.as.OrderedWith.impl.Greater.type.01adec.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.cf3682.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.506b32.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.cf3682.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.34c005.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.76f883.2: %T.as_type.as.OrderedWith.impl.Less.type.34c005.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.ea983f.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.52cd50.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.ea983f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.01adec.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.b936e0.2: %T.as_type.as.OrderedWith.impl.Greater.type.01adec.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.cf3682.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.9c9(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.506b32.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.cf3682.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.7cf: %OrderedWith.type.a39 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.694) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.a24: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.7cf) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.efd: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.7cf) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.61e: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.7cf) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.ae3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.7cf) [concrete]
// CHECK:STDOUT:   %.159: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.61e, %OrderedWith.facet.7cf [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.f32f7e.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.b936e0.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4d430c.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.b936e0.2, @T.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.1fa5ac.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4d430c.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.f32f7e.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.b936e0.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4d430c.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.b936e0.1, @T.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.1fa5ac.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.4d430c.2 [concrete]
// CHECK:STDOUT:   %.064: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.a24, %OrderedWith.facet.7cf [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.306df7.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.76f883.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.c262c8.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.76f883.2, @T.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.dca2b0.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.c262c8.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.306df7.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.76f883.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.c262c8.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.76f883.1, @T.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.dca2b0.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.c262c8.2 [concrete]
// CHECK:STDOUT:   %.41a: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.ae3, %OrderedWith.facet.7cf [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f70518.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.506b32.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3eb43c.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.506b32.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.831295.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3eb43c.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f70518.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.506b32.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3eb43c.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.506b32.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.831295.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3eb43c.2 [concrete]
// CHECK:STDOUT:   %.e98: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.efd, %OrderedWith.facet.7cf [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.a80feb.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.52cd50.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b5fa38.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.52cd50.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.3b8ed4.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b5fa38.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.a80feb.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.52cd50.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b5fa38.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.52cd50.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %bound_method.3b8ed4.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b5fa38.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.ImplicitAs.impl.2da [concrete]
// CHECK:STDOUT:   %Core.import_ref.9e3: @T.as_type.as.EqWith.impl.600.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.bfa2df.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.600.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.4c5205.1)]
// CHECK:STDOUT:   %Core.import_ref.8c9: @T.as_type.as.EqWith.impl.600.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.538ebe.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.600.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.08315d.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.934 = impl_witness_table (%Core.import_ref.9e3, %Core.import_ref.8c9), @T.as_type.as.EqWith.impl.600 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.eb3: @T.as_type.as.EqWith.impl.600.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.538ebe.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.600.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.08315d.2)]
// CHECK:STDOUT:   %Core.Equal.a27: @T.as_type.as.EqWith.impl.600.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.bfa2df.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.600.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.4c5205.2)]
// CHECK:STDOUT:   %Core.import_ref.990: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31b = impl_witness_table (%Core.import_ref.990), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72e: @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.bbdfd2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.505b6c.1)]
// CHECK:STDOUT:   %Core.import_ref.00f: @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.360910.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.fb0920.1)]
// CHECK:STDOUT:   %Core.import_ref.202: @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.fca417.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.71b704.1)]
// CHECK:STDOUT:   %Core.import_ref.420d: @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d05c2b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.11eaa6.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.df3 = impl_witness_table (%Core.import_ref.72e, %Core.import_ref.00f, %Core.import_ref.202, %Core.import_ref.420d), @T.as_type.as.OrderedWith.impl.9c9 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.ba6: @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d05c2b.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.11eaa6.2)]
// CHECK:STDOUT:   %Core.Greater.e9a: @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.fca417.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.71b704.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.39d: @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.360910.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.fb0920.2)]
// CHECK:STDOUT:   %Core.Less.daf: @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.bbdfd2.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.9c9.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.505b6c.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc9_21.2
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.7c0 = impl_witness_access constants.%EqWith.impl_witness.0d0, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.ceaabd.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.d3e448.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.as_type.as.EqWith.impl.Equal.type.a57597.1 = specific_constant imports.%Core.Equal.a27, @T.as_type.as.EqWith.impl.600(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.ceaabd.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.a57597.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.ceaabd.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.d3e448.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long = call %bound_method.loc10_3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %Cpp.long = converted %b.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %a.ref.loc10)
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.b7d3 = impl_witness_access constants.%EqWith.impl_witness.0d0, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.29d3c0.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.8b892c.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.as_type.as.EqWith.impl.NotEqual.type.278edc.1 = specific_constant imports.%Core.NotEqual.eb3, @T.as_type.as.EqWith.impl.600(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.29d3c0.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.278edc.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.29d3c0.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.8b892c.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11_3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %Cpp.long = converted %b.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %a.ref.loc11)
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.b39 = impl_witness_access constants.%OrderedWith.impl_witness.44c, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.dedd14.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.24c510.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.as_type.as.OrderedWith.impl.Greater.type.400b07.1 = specific_constant imports.%Core.Greater.e9a, @T.as_type.as.OrderedWith.impl.9c9(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.dedd14.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.400b07.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.dedd14.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.24c510.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12_3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %Cpp.long = converted %b.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %a.ref.loc12)
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.920 = impl_witness_access constants.%OrderedWith.impl_witness.44c, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.68034e.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.3b347e.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.as_type.as.OrderedWith.impl.Less.type.785f15.1 = specific_constant imports.%Core.Less.daf, @T.as_type.as.OrderedWith.impl.9c9(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.68034e.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.785f15.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.68034e.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.3b347e.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long = call %bound_method.loc13_3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %Cpp.long = converted %b.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %a.ref.loc13)
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.53c = impl_witness_access constants.%OrderedWith.impl_witness.44c, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5c44ac.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.728585.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c4f42.1 = specific_constant imports.%Core.GreaterOrEquivalent.ba6, @T.as_type.as.OrderedWith.impl.9c9(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5c44ac.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2c4f42.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.5c44ac.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.728585.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long = call %bound_method.loc14_3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %Cpp.long = converted %b.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %a.ref.loc14)
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.9da = impl_witness_access constants.%OrderedWith.impl_witness.44c, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1e14ba.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.00439e.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.855c21.1 = specific_constant imports.%Core.LessOrEquivalent.39d, @T.as_type.as.OrderedWith.impl.9c9(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1e14ba.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.855c21.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1e14ba.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.00439e.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long = call %bound_method.loc15_3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %Cpp.long = converted %b.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %a.ref.loc15)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.9d7 = impl_witness_access constants.%EqWith.impl_witness.fc3, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.3b534c.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.ef477d.1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.5238fc.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.f4cdf1.1]
// CHECK:STDOUT:   %.loc17_5: %T.as_type.as.EqWith.impl.Equal.type.76caa0.1 = specific_constant imports.%Core.Equal.a27, @T.as_type.as.EqWith.impl.600(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.3b534c.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.76caa0.1 = name_ref Equal, %.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.3b534c.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.ef477d.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.5238fc.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17 [concrete = constants.%bound_method.f4cdf1.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.802 = impl_witness_access constants.%EqWith.impl_witness.fc3, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.50bdbe.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.1ef7f5.1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.218ac5.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.a4d1ea.1]
// CHECK:STDOUT:   %.loc18_5: %T.as_type.as.EqWith.impl.NotEqual.type.7f7232.1 = specific_constant imports.%Core.NotEqual.eb3, @T.as_type.as.EqWith.impl.600(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.50bdbe.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.7f7232.1 = name_ref NotEqual, %.loc18_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.50bdbe.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.1ef7f5.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.218ac5.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18 [concrete = constants.%bound_method.a4d1ea.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.159 = impl_witness_access constants.%OrderedWith.impl_witness.694, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.b936e0.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.f32f7e.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.4d430c.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.1fa5ac.1]
// CHECK:STDOUT:   %.loc19_5: %T.as_type.as.OrderedWith.impl.Greater.type.01adec.1 = specific_constant imports.%Core.Greater.e9a, @T.as_type.as.OrderedWith.impl.9c9(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.b936e0.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.01adec.1 = name_ref Greater, %.loc19_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.b936e0.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.f32f7e.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.4d430c.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19 [concrete = constants.%bound_method.1fa5ac.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.064 = impl_witness_access constants.%OrderedWith.impl_witness.694, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.76f883.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.306df7.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.c262c8.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.dca2b0.1]
// CHECK:STDOUT:   %.loc20_5: %T.as_type.as.OrderedWith.impl.Less.type.34c005.1 = specific_constant imports.%Core.Less.daf, @T.as_type.as.OrderedWith.impl.9c9(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.76f883.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.34c005.1 = name_ref Less, %.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.76f883.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.306df7.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.c262c8.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20 [concrete = constants.%bound_method.dca2b0.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.41a = impl_witness_access constants.%OrderedWith.impl_witness.694, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.506b32.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f70518.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3eb43c.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.831295.1]
// CHECK:STDOUT:   %.loc21_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.cf3682.1 = specific_constant imports.%Core.GreaterOrEquivalent.ba6, @T.as_type.as.OrderedWith.impl.9c9(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.506b32.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.cf3682.1 = name_ref GreaterOrEquivalent, %.loc21_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.506b32.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f70518.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3eb43c.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21 [concrete = constants.%bound_method.831295.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.e98 = impl_witness_access constants.%OrderedWith.impl_witness.694, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.52cd50.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.a80feb.1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b5fa38.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.3b8ed4.1]
// CHECK:STDOUT:   %.loc22_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.ea983f.1 = specific_constant imports.%Core.LessOrEquivalent.39d, @T.as_type.as.OrderedWith.impl.9c9(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.52cd50.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.ea983f.1 = name_ref LessOrEquivalent, %.loc22_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.52cd50.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.a80feb.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.2b4) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b5fa38.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22 [concrete = constants.%bound_method.3b8ed4.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc22_3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long = call %bound_method.loc22_3(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_3.2: %Cpp.long = converted %int_1.loc22, %.loc22_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc22: init bool = call %bound_method.loc22_5.3(%.loc22_3.2, %a.ref.loc22)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- arithmetic_homogeneous_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.826: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.826) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.806: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ed7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.806, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %Cpp.long, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.bdb: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.7e5: type = fn_type_with_self_type %Negate.WithSelf.Op.type.bdb, %Negate.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op.type: type = fn_type @Cpp.long.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op: %Cpp.long.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.9c2: type = facet_type <@AddWith, @AddWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.f2b: <witness> = impl_witness imports.%AddWith.impl_witness_table.819 [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.9c2 = facet_value %Cpp.long, (%AddWith.impl_witness.f2b) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.c8c: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Cpp.long, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %.a2a: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.c8c, %AddWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.type.c8f: type = fn_type @Cpp.long.as.AddWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.c14: %Cpp.long.as.AddWith.impl.Op.type.c8f = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.type.e8e: type = facet_type <@SubWith, @SubWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness.e35: <witness> = impl_witness imports.%SubWith.impl_witness_table.02b [concrete]
// CHECK:STDOUT:   %SubWith.facet: %SubWith.type.e8e = facet_value %Cpp.long, (%SubWith.impl_witness.e35) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.bfa: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Cpp.long, %SubWith.facet) [concrete]
// CHECK:STDOUT:   %.b47: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.bfa, %SubWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.type.7d6: type = fn_type @Cpp.long.as.SubWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.f2a: %Cpp.long.as.SubWith.impl.Op.type.7d6 = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.type.a32: type = facet_type <@MulWith, @MulWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness.7ce: <witness> = impl_witness imports.%MulWith.impl_witness_table.61f [concrete]
// CHECK:STDOUT:   %MulWith.facet: %MulWith.type.a32 = facet_value %Cpp.long, (%MulWith.impl_witness.7ce) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.7fe: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Cpp.long, %MulWith.facet) [concrete]
// CHECK:STDOUT:   %.b3f: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.7fe, %MulWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.type.9a9: type = fn_type @Cpp.long.as.MulWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.f69: %Cpp.long.as.MulWith.impl.Op.type.9a9 = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.type.0c0: type = facet_type <@DivWith, @DivWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness.8f1: <witness> = impl_witness imports.%DivWith.impl_witness_table.b92 [concrete]
// CHECK:STDOUT:   %DivWith.facet: %DivWith.type.0c0 = facet_value %Cpp.long, (%DivWith.impl_witness.8f1) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.542: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Cpp.long, %DivWith.facet) [concrete]
// CHECK:STDOUT:   %.88c: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.542, %DivWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.type.adc: type = fn_type @Cpp.long.as.DivWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.9e1: %Cpp.long.as.DivWith.impl.Op.type.adc = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.type.2fc: type = facet_type <@ModWith, @ModWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness.ad5: <witness> = impl_witness imports.%ModWith.impl_witness_table.578 [concrete]
// CHECK:STDOUT:   %ModWith.facet: %ModWith.type.2fc = facet_value %Cpp.long, (%ModWith.impl_witness.ad5) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.14d: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Cpp.long, %ModWith.facet) [concrete]
// CHECK:STDOUT:   %.ec5: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.14d, %ModWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.type.d7d: type = fn_type @Cpp.long.as.ModWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.922: %Cpp.long.as.ModWith.impl.Op.type.d7d = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.ImplicitAs.impl.2da [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f35: %Cpp.long.as.Negate.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.cb912f.1, %Core.import_ref.f35), @Cpp.long.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.4 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.11e: %Cpp.long.as.AddWith.impl.Op.type.c8f = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.AddWith.impl.Op.c14]
// CHECK:STDOUT:   %AddWith.impl_witness_table.819 = impl_witness_table (%Core.import_ref.cb912f.4, %Core.import_ref.11e), @Cpp.long.as.AddWith.impl.1bc [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.7 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.394: %Cpp.long.as.SubWith.impl.Op.type.7d6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.SubWith.impl.Op.f2a]
// CHECK:STDOUT:   %SubWith.impl_witness_table.02b = impl_witness_table (%Core.import_ref.cb912f.7, %Core.import_ref.394), @Cpp.long.as.SubWith.impl.197 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.10 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.16c: %Cpp.long.as.MulWith.impl.Op.type.9a9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.MulWith.impl.Op.f69]
// CHECK:STDOUT:   %MulWith.impl_witness_table.61f = impl_witness_table (%Core.import_ref.cb912f.10, %Core.import_ref.16c), @Cpp.long.as.MulWith.impl.f64 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.13 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.146: %Cpp.long.as.DivWith.impl.Op.type.adc = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.DivWith.impl.Op.9e1]
// CHECK:STDOUT:   %DivWith.impl_witness_table.b92 = impl_witness_table (%Core.import_ref.cb912f.13, %Core.import_ref.146), @Cpp.long.as.DivWith.impl.7b6 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.16 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.67f: %Cpp.long.as.ModWith.impl.Op.type.d7d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ModWith.impl.Op.922]
// CHECK:STDOUT:   %ModWith.impl_witness_table.578 = impl_witness_table (%Core.import_ref.cb912f.16, %Core.import_ref.67f), @Cpp.long.as.ModWith.impl.deb [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ArithmeticHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.68c = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc8_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc8_21.2: %Cpp.long = converted %int_1.loc8, %.loc8_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %x: %Cpp.long = value_binding x, %.loc8_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.68c = value_binding_pattern y [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %y: %Cpp.long = value_binding y, %.loc9_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc11: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc11: %.7e5 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Cpp.long.as.Negate.impl.Op]
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
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc12: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc12: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc12: %.a2a = impl_witness_access constants.%AddWith.impl_witness.f2b, element1 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.c14]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.68c = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc13: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc13: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc13: %.b47 = impl_witness_access constants.%SubWith.impl_witness.e35, element1 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.f2a]
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
// CHECK:STDOUT:     %d.patt: %pattern_type.68c = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc14: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc14: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc14: %.b3f = impl_witness_access constants.%MulWith.impl_witness.7ce, element1 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.f69]
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
// CHECK:STDOUT:     %e.patt: %pattern_type.68c = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc15: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc15: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc15: %.88c = impl_witness_access constants.%DivWith.impl_witness.8f1, element1 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.9e1]
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
// CHECK:STDOUT:     %f.patt: %pattern_type.68c = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc16: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc16: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc16: %.ec5 = impl_witness_access constants.%ModWith.impl_witness.ad5, element1 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.922]
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
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.334: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.035) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.b8b: %T.as_type.as.ImplicitAs.impl.Convert.type.334 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.826: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2b4: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.826) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.806: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %.ed7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.806, %ImplicitAs.facet.2b4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.bef: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.7428ab.1: %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.7428ab.2: %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fb1: <witness> = impl_witness imports.%Copy.impl_witness_table.49f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.fb1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.115: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6e, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.293: %ImplicitAs.type.819 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.115) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.dfa: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.ea6, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.74c6e5.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.ae60d1.1: %Cpp.long.as.AddAssignWith.impl.Op.type.74c6e5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.74c6e5.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.ae60d1.2: %Cpp.long.as.AddAssignWith.impl.Op.type.74c6e5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.bef = facet_value %Cpp.long, (%AddAssignWith.impl_witness.dfa) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.8b3: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.152: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.8b3, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.eef335.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.ae60d1.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.eef335.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.ae60d1.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.b12: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.fc3513.1: %Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.fc3513.2: %Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.c70: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.0f0, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.658f8e.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.fd6475.1: %Cpp.long.as.SubAssignWith.impl.Op.type.658f8e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.658f8e.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.fd6475.2: %Cpp.long.as.SubAssignWith.impl.Op.type.658f8e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.b12 = facet_value %Cpp.long, (%SubAssignWith.impl_witness.c70) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.ee9: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.c23: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.ee9, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.74209d.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.fd6475.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.74209d.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.fd6475.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.59a: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.3e87f5.1: %Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.3e87f5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.7cb: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.0b8, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.792c18.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.259df5.1: %Cpp.long.as.MulAssignWith.impl.Op.type.792c18.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.792c18.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.259df5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.792c18.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.59a = facet_value %Cpp.long, (%MulAssignWith.impl_witness.7cb) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.13f: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.24b: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.13f, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.7330e2.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.259df5.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.7330e2.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.259df5.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.e6d: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.07e286.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.c9e0c0.1: %Cpp.long.as.DivAssignWith.impl.Op.type.07e286.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.07e286.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.c9e0c0.2: %Cpp.long.as.DivAssignWith.impl.Op.type.07e286.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.376: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.c92, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.524341.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.42cd1a.1: %Cpp.long.as.DivAssignWith.impl.Op.type.524341.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.524341.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.42cd1a.2: %Cpp.long.as.DivAssignWith.impl.Op.type.524341.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.e6d = facet_value %Cpp.long, (%DivAssignWith.impl_witness.376) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.220: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.2e9: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.220, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.480d87.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.42cd1a.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.480d87.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.42cd1a.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.2a3: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.6fdd26.1: %Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.6fdd26.2: %Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.474: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.501, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.5216aa.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.181cf2.1: %Cpp.long.as.ModAssignWith.impl.Op.type.5216aa.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.5216aa.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.181cf2.2: %Cpp.long.as.ModAssignWith.impl.Op.type.5216aa.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.2a3 = facet_value %Cpp.long, (%ModAssignWith.impl_witness.474) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.66a: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.b99: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.66a, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.d85655.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.181cf2.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.d85655.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.181cf2.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.efd: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.40d8b0.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.40d8b0.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.b09: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.d76, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.b86dd6.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.1f3e29.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.b86dd6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.b86dd6.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.1f3e29.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.b86dd6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.efd = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.b09) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.e5c: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.55a: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.e5c, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.32ba3e.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.1f3e29.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.32ba3e.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.1f3e29.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.b27: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.3525ae.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.3525ae.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.942: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.80e, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.f6ba26.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.e40cb6.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.f6ba26.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.f6ba26.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.e40cb6.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.f6ba26.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.b27 = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.942) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.45b: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.c11: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.45b, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.de0818.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.e40cb6.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.de0818.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.e40cb6.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.431: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.d3de9a.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.d3de9a.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.12c: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.eba, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d3edc.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.db4335.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d3edc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d3edc.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.db4335.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d3edc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.431 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.12c) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.ce9: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.99b: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.ce9, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.49f737.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.db4335.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.49f737.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.db4335.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.bed: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.c93685.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.c93685.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.d7c: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.f9a, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.42094d.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.4c9810.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.42094d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.42094d.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.4c9810.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.42094d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.bed = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.d7c) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.717: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.250: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.717, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.de5061.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.4c9810.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.de5061.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.4c9810.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.222: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.e41760.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.e41760.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.44b: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.d8d, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7121dc.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.d597c8.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7121dc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7121dc.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.d597c8.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7121dc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.222 = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.44b) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.302: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.988: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.302, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.71fc11.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.d597c8.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.71fc11.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.d597c8.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.293) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.758: @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert.type (%T.as_type.as.ImplicitAs.impl.Convert.type.334) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert (constants.%T.as_type.as.ImplicitAs.impl.Convert.b8b)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6e = impl_witness_table (%Core.import_ref.758), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.ImplicitAs.impl.2da [concrete]
// CHECK:STDOUT:   %Core.import_ref.c21: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.7428ab.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.ea6 = impl_witness_table (%Core.import_ref.c21), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7c4: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.7428ab.2)]
// CHECK:STDOUT:   %Core.import_ref.a6a: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.49f = impl_witness_table (%Core.import_ref.a6a), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f745: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.fc3513.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.0f0 = impl_witness_table (%Core.import_ref.f745), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.50c: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.fc3513.2)]
// CHECK:STDOUT:   %Core.import_ref.edb: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.3e87f5.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.0b8 = impl_witness_table (%Core.import_ref.edb), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fc4: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.3e87f5.2)]
// CHECK:STDOUT:   %Core.import_ref.e04: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.07e286.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.c9e0c0.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.c92 = impl_witness_table (%Core.import_ref.e04), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.9fb: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.07e286.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.c9e0c0.2)]
// CHECK:STDOUT:   %Core.import_ref.a40: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.6fdd26.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.501 = impl_witness_table (%Core.import_ref.a40), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3a8: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.6fdd26.2)]
// CHECK:STDOUT:   %Core.import_ref.40d: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.40d8b0.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.d76 = impl_witness_table (%Core.import_ref.40d), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.aab: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.40d8b0.2)]
// CHECK:STDOUT:   %Core.import_ref.223: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.3525ae.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.80e = impl_witness_table (%Core.import_ref.223), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.277: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.3525ae.2)]
// CHECK:STDOUT:   %Core.import_ref.603: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d3de9a.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.eba = impl_witness_table (%Core.import_ref.603), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.ea4: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d3de9a.2)]
// CHECK:STDOUT:   %Core.import_ref.58d: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.c93685.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.f9a = impl_witness_table (%Core.import_ref.58d), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.962: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.c93685.2)]
// CHECK:STDOUT:   %Core.import_ref.6ac: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.e41760.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.d8d = impl_witness_table (%Core.import_ref.6ac), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.6b7: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.e41760.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.68c = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref.loc8 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.68c = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var %b.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %b.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = ref_binding b, %b.var
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11: %.152 = impl_witness_access constants.%AddAssignWith.impl_witness.dfa, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.ae60d1.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.115) [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.eef335.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.74c6e5.1 = specific_constant imports.%Core.Op.7c4, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.ae60d1.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.AddAssignWith.impl.Op.type.74c6e5.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.ae60d1.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.eef335.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.c23 = impl_witness_access constants.%SubAssignWith.impl_witness.c70, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.fd6475.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.115) [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.74209d.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.SubAssignWith.impl.Op.type.658f8e.1 = specific_constant imports.%Core.Op.50c, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.fd6475.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.SubAssignWith.impl.Op.type.658f8e.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.fd6475.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.74209d.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.24b = impl_witness_access constants.%MulAssignWith.impl_witness.7cb, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.259df5.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.115) [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.7330e2.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.792c18.1 = specific_constant imports.%Core.Op.fc4, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.259df5.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.MulAssignWith.impl.Op.type.792c18.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.259df5.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.7330e2.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.2e9 = impl_witness_access constants.%DivAssignWith.impl_witness.376, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.42cd1a.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.115) [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.480d87.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.DivAssignWith.impl.Op.type.524341.1 = specific_constant imports.%Core.Op.9fb, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.42cd1a.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.DivAssignWith.impl.Op.type.524341.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.42cd1a.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.480d87.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.b99 = impl_witness_access constants.%ModAssignWith.impl_witness.474, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.181cf2.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.115) [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.d85655.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.5216aa.1 = specific_constant imports.%Core.Op.3a8, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.181cf2.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long.as.ModAssignWith.impl.Op.type.5216aa.1 = name_ref Op, %.loc15_5.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.181cf2.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.d85655.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.55a = impl_witness_access constants.%BitAndAssignWith.impl_witness.b09, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.1f3e29.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.115) [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.32ba3e.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.b86dd6.1 = specific_constant imports.%Core.Op.aab, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.1f3e29.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitAndAssignWith.impl.Op.type.b86dd6.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.1f3e29.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.32ba3e.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.c11 = impl_witness_access constants.%BitOrAssignWith.impl_witness.942, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.e40cb6.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.115) [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.de0818.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.f6ba26.1 = specific_constant imports.%Core.Op.277, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.e40cb6.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitOrAssignWith.impl.Op.type.f6ba26.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.e40cb6.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.de0818.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.99b = impl_witness_access constants.%BitXorAssignWith.impl_witness.12c, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.db4335.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.115) [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.49f737.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d3edc.1 = specific_constant imports.%Core.Op.ea4, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.db4335.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.BitXorAssignWith.impl.Op.type.5d3edc.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.db4335.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.49f737.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.250 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.d7c, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.4c9810.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.115) [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.de5061.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.42094d.1 = specific_constant imports.%Core.Op.962, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.4c9810.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.42094d.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.4c9810.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.de5061.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.988 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.44b, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.d597c8.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.115) [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.293]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.71fc11.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7121dc.1 = specific_constant imports.%Core.Op.6b7, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.d597c8.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.7121dc.1 = name_ref Op, %.loc21_5.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.d597c8.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.293) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.71fc11.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_9)
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
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
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.826: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2b4: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.826) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.806: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %.ed7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.806, %ImplicitAs.facet.2b4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.124: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.124 [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.794: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.7428ab.1: %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.7428ab.2: %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.895: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.87f: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.895) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.8a4: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.ea6, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.18480a.1: %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.18480a.2: %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.794 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.8a4) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.7e7: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.3bf: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.7e7, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c51759.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.18480a.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.595: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %.d59: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.595, %ImplicitAs.facet.87f [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c51759.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.18480a.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.ab0: type = facet_type <@SubAssignWith, @SubAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.fc3513.1: %Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.fc3513.2: %Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.cde: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.0f0, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.007151.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.3151e9.1: %Cpp.long.as.SubAssignWith.impl.Op.type.007151.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.007151.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.3151e9.2: %Cpp.long.as.SubAssignWith.impl.Op.type.007151.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.ab0 = facet_value %Cpp.long, (%SubAssignWith.impl_witness.cde) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.b30: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i32, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.0e9: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.b30, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.f4a251.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.3151e9.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.f4a251.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.3151e9.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.743: type = facet_type <@MulAssignWith, @MulAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.3e87f5.1: %Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.3e87f5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.10f: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.0b8, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.249066.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.3df17e.1: %Cpp.long.as.MulAssignWith.impl.Op.type.249066.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.249066.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.3df17e.2: %Cpp.long.as.MulAssignWith.impl.Op.type.249066.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.743 = facet_value %Cpp.long, (%MulAssignWith.impl_witness.10f) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.bff: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i32, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.761: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.bff, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.2a0ec5.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.3df17e.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.2a0ec5.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.3df17e.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.786: type = facet_type <@DivAssignWith, @DivAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.07e286.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.c9e0c0.1: %Cpp.long.as.DivAssignWith.impl.Op.type.07e286.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.07e286.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.c9e0c0.2: %Cpp.long.as.DivAssignWith.impl.Op.type.07e286.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.0ea: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.c92, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.f3a6e1.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.95c3cb.1: %Cpp.long.as.DivAssignWith.impl.Op.type.f3a6e1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.f3a6e1.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.95c3cb.2: %Cpp.long.as.DivAssignWith.impl.Op.type.f3a6e1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.786 = facet_value %Cpp.long, (%DivAssignWith.impl_witness.0ea) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.99d: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i32, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.d78: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.99d, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.63c0e6.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.95c3cb.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.63c0e6.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.95c3cb.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.e49: type = facet_type <@ModAssignWith, @ModAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.6fdd26.1: %Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.6fdd26.2: %Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.b4e: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.501, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.1dde31.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.a5ea74.1: %Cpp.long.as.ModAssignWith.impl.Op.type.1dde31.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.1dde31.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.a5ea74.2: %Cpp.long.as.ModAssignWith.impl.Op.type.1dde31.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.e49 = facet_value %Cpp.long, (%ModAssignWith.impl_witness.b4e) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.cae: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i32, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.7f1: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.cae, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.1bddd2.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.a5ea74.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.1bddd2.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.a5ea74.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.d65: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.40d8b0.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.40d8b0.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.84d: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.d76, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.9b8168.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.3b7601.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.9b8168.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.9b8168.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.3b7601.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.9b8168.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.d65 = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.84d) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.400: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i32, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.fd7: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.400, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.94318e.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.3b7601.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.94318e.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.3b7601.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.6e9: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.3525ae.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.3525ae.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.0b0: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.80e, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.198694.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.dcd7c4.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.198694.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.198694.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.dcd7c4.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.198694.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.6e9 = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.0b0) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.ef7: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i32, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.7ba: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.ef7, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.c12656.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.dcd7c4.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.c12656.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.dcd7c4.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.ea8: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.d3de9a.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.d3de9a.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.77c: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.eba, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.0c822c.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.1793e8.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.0c822c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.0c822c.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.1793e8.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.0c822c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.ea8 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.77c) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.a67: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i32, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.03f: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.a67, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.1d9822.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.1793e8.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.1d9822.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.1793e8.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.92b: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.c93685.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.c93685.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.aa9: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.f9a, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.8bb9f4.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.ba3cb3.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.8bb9f4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.8bb9f4.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.ba3cb3.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.8bb9f4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.92b = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.aa9) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.483: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i32, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.f43: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.483, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.dc5ccd.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.ba3cb3.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.dc5ccd.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.ba3cb3.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.8ae: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.e41760.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.e41760.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.767: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.d8d, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.9da311.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.914893.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.9da311.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.9da311.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.914893.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.9da311.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.8ae = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.767) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.b3d: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i32, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.55e: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.b3d, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.96f972.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.914893.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.96f972.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.914893.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.ImplicitAs.impl.2da [concrete]
// CHECK:STDOUT:   %Core.import_ref.c21: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.7428ab.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.ea6 = impl_witness_table (%Core.import_ref.c21), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7c4: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.7428ab.2)]
// CHECK:STDOUT:   %Core.import_ref.990: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31b = impl_witness_table (%Core.import_ref.990), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f74: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.fc3513.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.0f0 = impl_witness_table (%Core.import_ref.f74), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.50c: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.e9ae97.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.fc3513.2)]
// CHECK:STDOUT:   %Core.import_ref.edb: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.3e87f5.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.0b8 = impl_witness_table (%Core.import_ref.edb), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fc4: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.9da91f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.3e87f5.2)]
// CHECK:STDOUT:   %Core.import_ref.e04: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.07e286.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.c9e0c0.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.c92 = impl_witness_table (%Core.import_ref.e04), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.9fb: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.07e286.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.c9e0c0.2)]
// CHECK:STDOUT:   %Core.import_ref.a40: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.6fdd26.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.501 = impl_witness_table (%Core.import_ref.a40), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3a8: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.1dd956.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.6fdd26.2)]
// CHECK:STDOUT:   %Core.import_ref.40d: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.40d8b0.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.d76 = impl_witness_table (%Core.import_ref.40d), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.aab: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.fedf5f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.40d8b0.2)]
// CHECK:STDOUT:   %Core.import_ref.223: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.3525ae.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.80e = impl_witness_table (%Core.import_ref.223), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.277: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.9f94e6.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.3525ae.2)]
// CHECK:STDOUT:   %Core.import_ref.603: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d3de9a.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.eba = impl_witness_table (%Core.import_ref.603), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.ea4: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.a12c37.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d3de9a.2)]
// CHECK:STDOUT:   %Core.import_ref.58d: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.c93685.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.f9a = impl_witness_table (%Core.import_ref.58d), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.962: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.7188ed.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.c93685.2)]
// CHECK:STDOUT:   %Core.import_ref.6ac: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.e41760.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.d8d = impl_witness_table (%Core.import_ref.6ac), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.6b7: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.d554e4.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.e41760.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndI32() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.68c = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.3bf = impl_witness_access constants.%AddAssignWith.impl_witness.8a4, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.18480a.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c51759.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.1 = specific_constant imports.%Core.Op.7c4, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.18480a.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.18480a.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c51759.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5.1: %.0e9 = impl_witness_access constants.%SubAssignWith.impl_witness.cde, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.3151e9.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5.1, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.f4a251.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.SubAssignWith.impl.Op.type.007151.1 = specific_constant imports.%Core.Op.50c, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.3151e9.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.SubAssignWith.impl.Op.type.007151.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.3151e9.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.f4a251.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5.1: %.761 = impl_witness_access constants.%MulAssignWith.impl_witness.10f, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.3df17e.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5.1, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.2a0ec5.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.249066.1 = specific_constant imports.%Core.Op.fc4, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.3df17e.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.MulAssignWith.impl.Op.type.249066.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.3df17e.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.2a0ec5.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_8: init %Cpp.long = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_8
// CHECK:STDOUT:   %.loc12_8.2: %Cpp.long = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.d78 = impl_witness_access constants.%DivAssignWith.impl_witness.0ea, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.95c3cb.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.63c0e6.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.DivAssignWith.impl.Op.type.f3a6e1.1 = specific_constant imports.%Core.Op.9fb, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.95c3cb.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.DivAssignWith.impl.Op.type.f3a6e1.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.95c3cb.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.63c0e6.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_8: init %Cpp.long = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_8
// CHECK:STDOUT:   %.loc13_8.2: %Cpp.long = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5.1: %.7f1 = impl_witness_access constants.%ModAssignWith.impl_witness.b4e, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.a5ea74.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5.1, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.1bddd2.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.1dde31.1 = specific_constant imports.%Core.Op.3a8, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.a5ea74.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.ModAssignWith.impl.Op.type.1dde31.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.a5ea74.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.1bddd2.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5.1: %.fd7 = impl_witness_access constants.%BitAndAssignWith.impl_witness.84d, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.3b7601.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc16: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc16 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5.1, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.94318e.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.9b8168.1 = specific_constant imports.%Core.Op.aab, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.3b7601.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long.as.BitAndAssignWith.impl.Op.type.9b8168.1 = name_ref Op, %.loc16_5.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.3b7601.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_5: init %Cpp.long = call %bound_method.loc16_5.3(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_5
// CHECK:STDOUT:   %.loc16_5.4: %Cpp.long = converted %b.ref.loc16, %.loc16_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.94318e.2]
// CHECK:STDOUT:   %bound_method.loc16_5.4: <bound method> = bound_method %a.ref.loc16, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_8: init %Cpp.long = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_8
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.4(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.7ba = impl_witness_access constants.%BitOrAssignWith.impl_witness.0b0, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.dcd7c4.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.c12656.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.198694.1 = specific_constant imports.%Core.Op.277, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.dcd7c4.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitOrAssignWith.impl.Op.type.198694.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.dcd7c4.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long = call %bound_method.loc17_5.3(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_5
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long = converted %b.ref.loc17, %.loc17_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.c12656.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_8
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5.1: %.03f = impl_witness_access constants.%BitXorAssignWith.impl_witness.77c, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.1793e8.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5.1, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.1d9822.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.0c822c.1 = specific_constant imports.%Core.Op.ea4, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.1793e8.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitXorAssignWith.impl.Op.type.0c822c.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.1793e8.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long = call %bound_method.loc18_5.3(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc18_5
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long = converted %b.ref.loc18, %.loc18_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.1d9822.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc18_8
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5.1: %.f43 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.aa9, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.ba3cb3.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.dc5ccd.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.8bb9f4.1 = specific_constant imports.%Core.Op.962, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.ba3cb3.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.8bb9f4.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.ba3cb3.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long = call %bound_method.loc19_5.3(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc19_5
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long = converted %b.ref.loc19, %.loc19_5.3
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.dc5ccd.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc19_9: init %Cpp.long = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc19_9
// CHECK:STDOUT:   %.loc19_9.2: %Cpp.long = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.55e = impl_witness_access constants.%RightShiftAssignWith.impl_witness.767, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.914893.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.96f972.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.9da311.1 = specific_constant imports.%Core.Op.6b7, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.914893.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.9da311.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.914893.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long = call %bound_method.loc20_5.3(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc20_5
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long = converted %b.ref.loc20, %.loc20_5.3
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.96f972.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_9: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_9
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc20_9: init %Cpp.long = call %bound_method.loc20_9(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_9.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc20_9
// CHECK:STDOUT:   %.loc20_9.2: %Cpp.long = converted %b.ref.loc20, %.loc20_9.1
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_9.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
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
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.826: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.826) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.806: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ed7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.806, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.761: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.7428ab.1: %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.7428ab.2: %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.857: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.ea6, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.dedb88.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.025560.1: %Cpp.long.as.AddAssignWith.impl.Op.type.dedb88.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.dedb88.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.025560.2: %Cpp.long.as.AddAssignWith.impl.Op.type.dedb88.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.761 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.857) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.964: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.e15: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.964, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3f8ea9.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.025560.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3f8ea9.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.025560.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.ImplicitAs.impl.2da [concrete]
// CHECK:STDOUT:   %Core.import_ref.c21: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.7428ab.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.ea6 = impl_witness_table (%Core.import_ref.c21), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7c4: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.7428ab.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.68c = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9_5.1: %.e15 = impl_witness_access constants.%AddAssignWith.impl_witness.857, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.025560.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.826) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3f8ea9.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.dedb88.1 = specific_constant imports.%Core.Op.7c4, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.025560.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.dedb88.1 = name_ref Op, %.loc9_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.025560.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc9_5.2: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5: init %Cpp.long = call %bound_method.loc9_5.3(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_5.4: %Cpp.long = converted %int_1.loc9, %.loc9_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3f8ea9.2]
// CHECK:STDOUT:   %bound_method.loc9_5.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_8: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_8: init %Cpp.long = call %bound_method.loc9_8(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_8.2: %Cpp.long = converted %int_1.loc9, %.loc9_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_5.4(%a.ref, %.loc9_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
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
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.826: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2b4: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.826) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.806: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.2b4) [concrete]
// CHECK:STDOUT:   %.ed7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.806, %ImplicitAs.facet.2b4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.124: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.124 [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.257(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ac5: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82d, @Core.IntLiteral.as.ImplicitAs.impl.257(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.257(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.614: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ac5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.614) [concrete]
// CHECK:STDOUT:   %.9db: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b91, %ImplicitAs.facet.614 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d43: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.794: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.7428ab.1: %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.7428ab.2: %Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.895: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.87f: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.895) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.8a4: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.ea6, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.18480a.1: %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.18480a.2: %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.794 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.8a4) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.7e7: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.3bf: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.7e7, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c51759.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.18480a.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.595: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %.d59: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.595, %ImplicitAs.facet.87f [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c51759.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.18480a.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.87f) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.162 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.ImplicitAs.impl.2da [concrete]
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.257.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.257.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl.257 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c21: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.23f734.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.7428ab.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.ea6 = impl_witness_table (%Core.import_ref.c21), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7c4: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.23f734.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.7428ab.2)]
// CHECK:STDOUT:   %Core.import_ref.990: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31b = impl_witness_table (%Core.import_ref.990), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongAndRuntimeI32() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.68c = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.124]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d48]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc9: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d43]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_16.2(%int_1.loc9) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_16.2: %i32 = converted %int_1.loc9, %.loc9_16.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %b: %i32 = value_binding b, %.loc9_16.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.3bf = impl_witness_access constants.%AddAssignWith.impl_witness.8a4, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.18480a.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.895) [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet.87f]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c51759.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.1 = specific_constant imports.%Core.Op.7c4, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.18480a.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.88b7ef.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.18480a.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = converted %b.ref, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.87f) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.c51759.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.d59 = impl_witness_access constants.%ImplicitAs.impl_witness.895, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long = call %bound_method.loc10_8(%b.ref)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref, %.loc10_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
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
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.826: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.826) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.806: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ed7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.806, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.4d5: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.450: type = fn_type_with_self_type %Inc.WithSelf.Op.type.4d5, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.type: type = fn_type @Cpp.long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op: %Cpp.long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.2ef: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.050: type = fn_type_with_self_type %Dec.WithSelf.Op.type.2ef, %Dec.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.type: type = fn_type @Cpp.long.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op: %Cpp.long.as.Dec.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.ImplicitAs.impl.2da [concrete]
// CHECK:STDOUT:   %Core.import_ref.8ff9: %Cpp.long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.8ff9), @Cpp.long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e4c: %Cpp.long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.e4c), @Cpp.long.as.Dec.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IncrementDecrementLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.68c = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ed7 = impl_witness_access constants.%ImplicitAs.impl_witness.826, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc9: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc9: %.450 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.050 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10(%a.ref.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
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
// CHECK:STDOUT:   %pattern_type.5b7: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3c3: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.726: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f0b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.900: %ImplicitAs.type.3c3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.726) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9d9: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet.900) [concrete]
// CHECK:STDOUT:   %.1ae: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9d9, %ImplicitAs.facet.900 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.00e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ccc: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.00e = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a6d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ccc [concrete]
// CHECK:STDOUT:   %int_1.0ab: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongResult: type = class_type @ULongResult [concrete]
// CHECK:STDOUT:   %pattern_type.6f2: type = pattern_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULong.cpp_overload_set.type: type = cpp_overload_set_type @PassULong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassULong.cpp_overload_set.value: %PassULong.cpp_overload_set.type = cpp_overload_set_value @PassULong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.f5e: type = ptr_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.type.3fc2d8.1: type = fn_type @PassULong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.ff747d.1: %PassULong__carbon_thunk.type.3fc2d8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.146: type = facet_type <@ImplicitAs, @ImplicitAs(%u32)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.703(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.b06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.27a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8a3, @Core.IntLiteral.as.ImplicitAs.impl.703(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a23: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.703(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6d0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a23 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a17: %ImplicitAs.type.146 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.27a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9ff: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u32, %ImplicitAs.facet.a17) [concrete]
// CHECK:STDOUT:   %.dac: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9ff, %ImplicitAs.facet.a17 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c54: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6d0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6d0, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.df1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.c1d: %u32 = int_value 1 [concrete]
// CHECK:STDOUT:   %UIntResult: type = class_type @UIntResult [concrete]
// CHECK:STDOUT:   %pattern_type.dc7: type = pattern_type %UIntResult [concrete]
// CHECK:STDOUT:   %ptr.059: type = ptr_type %UIntResult [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.type.3fc2d8.2: type = fn_type @PassULong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.ff747d.2: %PassULong__carbon_thunk.type.3fc2d8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.950: type = facet_type <@As, @As(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.576: <witness> = impl_witness imports.%As.impl_witness_table.239 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.950 = facet_value Core.IntLiteral, (%As.impl_witness.576) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.df7: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.cbd: type = fn_type_with_self_type %As.WithSelf.Convert.type.df7, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0b4: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b0a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a59: %ImplicitAs.type.139 = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.0b4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.53e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.a59) [concrete]
// CHECK:STDOUT:   %.927: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.53e, %ImplicitAs.facet.a59 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.0ab, %Cpp.unsigned_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.332: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c17, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f0e: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.332) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3b5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.97e, %ImplicitAs.facet.f0e) [concrete]
// CHECK:STDOUT:   %.3bc: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3b5, %ImplicitAs.facet.f0e [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6b7: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc24_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT:   %UIntResult.cpp_destructor.type: type = fn_type @UIntResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %UIntResult.cpp_destructor: %UIntResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor.type: type = fn_type @ULongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor: %ULongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .CppCompat = %CppCompat.4b4
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
// CHECK:STDOUT:   %CppCompat.4b4: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .ULong32 = %Core.ULong32
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ULong32: type = import_ref Core//prelude/types/cpp/int, ULong32, loaded [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.0fb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.00e = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ccc]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f0b = impl_witness_table (%Core.import_ref.0fb), @Core.IntLiteral.as.ImplicitAs.impl.749 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULong.cpp_overload_set.value: %PassULong.cpp_overload_set.type = cpp_overload_set_value @PassULong.cpp_overload_set [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULong__carbon_thunk.decl.108234.1: %PassULong__carbon_thunk.type.3fc2d8.1 = fn_decl @PassULong__carbon_thunk.1 [concrete = constants.%PassULong__carbon_thunk.ff747d.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.2b3: @Core.IntLiteral.as.ImplicitAs.impl.703.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.703.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.b06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8a3 = impl_witness_table (%Core.import_ref.2b3), @Core.IntLiteral.as.ImplicitAs.impl.703 [concrete]
// CHECK:STDOUT:   %UIntResult.decl: type = class_decl @UIntResult [concrete = constants.%UIntResult] {} {}
// CHECK:STDOUT:   %PassULong__carbon_thunk.decl.108234.2: %PassULong__carbon_thunk.type.3fc2d8.2 = fn_decl @PassULong__carbon_thunk.2 [concrete = constants.%PassULong__carbon_thunk.ff747d.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.db1: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.239 = impl_witness_table (%Core.import_ref.db1), @Core.IntLiteral.as.As.impl.b22 [concrete]
// CHECK:STDOUT:   %Core.import_ref.afbb: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b0a = impl_witness_table (%Core.import_ref.afbb), @Cpp.unsigned_long.as.ImplicitAs.impl.4ea [concrete]
// CHECK:STDOUT:   %Core.import_ref.1df4: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c17 = impl_witness_table (%Core.import_ref.1df4), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.5b7 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_29: type = splice_block %unsigned_long.ref.loc15 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc15: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.1ae = impl_witness_access constants.%ImplicitAs.impl_witness.726, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ccc]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a6d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.unsigned_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %.loc15_46.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %.loc15_46.2: %Cpp.unsigned_long = converted %int_1.loc15, %.loc15_46.1 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %cpp_unsigned_long: %Cpp.unsigned_long = value_binding cpp_unsigned_long, %.loc15_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_result.patt: %pattern_type.6f2 = value_binding_pattern cpp_unsigned_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_58: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc16: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc16: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc16_89.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.f5e = addr_of %.loc16_89.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.1(%cpp_unsigned_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_89.2: init %ULongResult to %.loc16_89.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_43: type = splice_block %ULongResult.ref.loc16 [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_40: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref.loc16: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_89.3: ref %ULongResult = temporary %.loc16_89.1, %.loc16_89.2
// CHECK:STDOUT:   %.loc16_89.4: %ULongResult = acquire_value %.loc16_89.3
// CHECK:STDOUT:   %cpp_unsigned_long_result: %ULongResult = value_binding cpp_unsigned_long_result, %.loc16_89.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong.patt: %pattern_type.5b7 = value_binding_pattern cpp_compat_ulong [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc18: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc18: type = splice_block %ULong32.ref [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.4b4 [concrete = imports.%CppCompat.4b4]
// CHECK:STDOUT:     %ULong32.ref: type = name_ref ULong32, imports.%Core.ULong32 [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_ulong: %Cpp.unsigned_long = value_binding cpp_compat_ulong, %cpp_unsigned_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_result.patt: %pattern_type.6f2 = value_binding_pattern cpp_compat_ulong_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc19: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong.ref: %Cpp.unsigned_long = name_ref cpp_compat_ulong, %cpp_compat_ulong
// CHECK:STDOUT:   %.loc19_87.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.f5e = addr_of %.loc19_87.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.1(%cpp_compat_ulong.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_87.2: init %ULongResult to %.loc19_87.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_42: type = splice_block %ULongResult.ref.loc19 [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_39: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref.loc19: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_87.3: ref %ULongResult = temporary %.loc19_87.1, %.loc19_87.2
// CHECK:STDOUT:   %.loc19_87.4: %ULongResult = acquire_value %.loc19_87.3
// CHECK:STDOUT:   %cpp_compat_ulong_result: %ULongResult = value_binding cpp_compat_ulong_result, %.loc19_87.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u32.patt: %pattern_type.4a9 = value_binding_pattern carbon_u32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u32: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %impl.elem0.loc21: %.dac = impl_witness_access constants.%ImplicitAs.impl_witness.27a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6d0]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c54]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.df1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %u32 = call %bound_method.loc21_25.2(%int_1.loc21) [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc21_25.1: %u32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc21_25.2: %u32 = converted %int_1.loc21, %.loc21_25.1 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %carbon_u32: %u32 = value_binding carbon_u32, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u32_result.patt: %pattern_type.dc7 = value_binding_pattern carbon_u32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc22: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u32.ref: %u32 = name_ref carbon_u32, %carbon_u32
// CHECK:STDOUT:   %.loc22_74.1: ref %UIntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.059 = addr_of %.loc22_74.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.2(%carbon_u32.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_74.2: init %UIntResult to %.loc22_74.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_36: type = splice_block %UIntResult.ref [concrete = constants.%UIntResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %UIntResult.ref: type = name_ref UIntResult, imports.%UIntResult.decl [concrete = constants.%UIntResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_74.3: ref %UIntResult = temporary %.loc22_74.1, %.loc22_74.2
// CHECK:STDOUT:   %.loc22_74.4: %UIntResult = acquire_value %.loc22_74.3
// CHECK:STDOUT:   %carbon_u32_result: %UIntResult = value_binding carbon_u32_result, %.loc22_74.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_59: %.3bc = impl_witness_access constants.%ImplicitAs.impl_witness.332, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.67d]
// CHECK:STDOUT:   %bound_method.loc24_59.1: <bound method> = bound_method %float, %impl.elem0.loc24_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_59.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.6b7]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc24_59.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_59.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_59.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_59.4: init %f32.97e to %.loc24_59.3 = in_place_init %.loc24_59.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_59.5: init %array_type to %a.var = array_init (%.loc24_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_59.1, %.loc24_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_50: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc24: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.cbd = impl_witness_access constants.%As.impl_witness.576, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.927 = impl_witness_access constants.%ImplicitAs.impl_witness.0b4, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.3
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
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.2(%self.param: ref %f32.97e) {
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
// CHECK:STDOUT:   %pattern_type.5b7: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3c3: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.726: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f0b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.3c3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.726) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9d9: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1ae: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9d9, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.0ab: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.0e4: <witness> = impl_witness imports.%Copy.impl_witness_table.a75 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long, (%Copy.impl_witness.0e4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c8c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.97e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c8c, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op: %Cpp.unsigned_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.0fb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f0b = impl_witness_table (%Core.import_ref.0fb), @Core.IntLiteral.as.ImplicitAs.impl.749 [concrete]
// CHECK:STDOUT:   %Core.import_ref.d42: %Cpp.unsigned_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.a75 = impl_witness_table (%Core.import_ref.d42), @Cpp.unsigned_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyUnsignedLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.5b7 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.5b7 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.unsigned_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.1ae = impl_witness_access constants.%ImplicitAs.impl_witness.726, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.unsigned_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %unsigned_long.ref.loc8 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.unsigned_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.5b7 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.5b7 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.unsigned_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.unsigned_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_37: %Cpp.unsigned_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.97e = impl_witness_access constants.%Copy.impl_witness.0e4, element0 [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_37, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long = call %bound_method.loc9(%.loc9_37)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %unsigned_long.ref.loc9 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc9: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
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
