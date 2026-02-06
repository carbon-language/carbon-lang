// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-pc-windows-msvc --clang-arg=-fno-PIE
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/builtins.llp64.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/builtins.llp64.carbon

// --- long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  let cpp_long_long: Cpp.long_long = 1;
  let carbon_long_long: i64 = cpp_long_long;

  var a: array(f32, 1 as Cpp.long_long) = (1.0,);
  //@dump-sem-ir-end
}

// --- unsigned_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  let cpp_unsigned_long_long: Cpp.unsigned_long_long = 1;
  let carbon_unsigned_long_long: u64 = cpp_unsigned_long_long;

  var a: array(f32, 1 as Cpp.unsigned_long_long) = (1.0,);
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
  let cpp_long_result: Cpp.LongResult = Cpp.PassLong(cpp_long);

  let cpp_compat_long: Core.CppCompat.Long32 = cpp_long;
  let cpp_compat_long_result: Cpp.LongResult = Cpp.PassLong(cpp_compat_long);

  let carbon_i32: i32 = cpp_long as i32;
  let carbon_i32_result: Cpp.IntResult = Cpp.PassLong(carbon_i32);

  let carbon_i64: i64 = cpp_long;

  var a: array(f32, 1 as Cpp.long) = (1.0,);
  //@dump-sem-ir-end
}

// --- fail_implicit_conversion_long_to_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  let cpp_long: Cpp.long = 1;
  // CHECK:STDERR: fail_implicit_conversion_long_to_i32.carbon:[[@LINE+7]]:25: error: cannot implicitly convert expression of type `Cpp.long` to `i32` [ConversionFailure]
  // CHECK:STDERR:   let carbon_i32: i32 = cpp_long;
  // CHECK:STDERR:                         ^~~~~~~~
  // CHECK:STDERR: fail_implicit_conversion_long_to_i32.carbon:[[@LINE+4]]:25: note: type `Cpp.long` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let carbon_i32: i32 = cpp_long;
  // CHECK:STDERR:                         ^~~~~~~~
  // CHECK:STDERR:
  let carbon_i32: i32 = cpp_long;
}

// --- copy_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CopyLong() {
  //@dump-sem-ir-begin
  var a: Cpp.long = 1;
  var b: Cpp.long = a;
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

  a == b;
  a != b;
  a > b;
  a < b;
  a >= b;
  a <= b;
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

  b == a;
  b != a;
  b > a;
  b < a;
  b >= a;
  b <= a;
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

  let a: Cpp.long = -x;
  let b: Cpp.long = x + y;
  let c: Cpp.long = x - y;
  let d: Cpp.long = x * y;
  let e: Cpp.long = x / y;
  let f: Cpp.long = x % y;
  //@dump-sem-ir-end
}

// --- arithmetic_heterogeneous_long_left_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](a: T, b: T) {}

fn ArithmeticHeterogeneousLongLeftSide() {
  let b: i32 = 1;
  //@dump-sem-ir-begin
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

  let y: i32 = 1;
  AssertSameType(x + y, x);
  AssertSameType(x - y, x);
  AssertSameType(x * y, x);
  AssertSameType(x / y, x);
  AssertSameType(x % y, x);
  //@dump-sem-ir-end
}

// --- arithmetic_heterogeneous_long_right_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](a: T, b: T) {}

fn ArithmeticHeterogeneousLongRightSide() {
  let b: i32 = 1;
  //@dump-sem-ir-begin
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

  let y: i32 = 1;
  AssertSameType(y + x, x);
  AssertSameType(y - x, x);
  AssertSameType(y * x, x);
  AssertSameType(y / x, x);
  AssertSameType(y % x, x);
  //@dump-sem-ir-end
}

// --- arithmetic_heterogeneous_long_and_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](a: T, b: T) {}

fn ArithmeticHeterogeneousLongAndI64() {
  //@dump-sem-ir-begin
  let x: Cpp.long = 1;
  let y: i64 = 1;
  AssertSameType(x + y, y);
  AssertSameType(y + x, y);
  //@dump-sem-ir-end
}

// --- fail_todo_arithmetic_heterogeneous_long_and_i16.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](a: T, b: T) {}

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

fn AssertSameType[T:! type](a: T, b: T) {}

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

fn AssertSameType[T:! type](a: T, b: T) {}

fn BitWiseHomogeneousLong() {
  //@dump-sem-ir-begin
  let a: Cpp.long = 1;
  let b: Cpp.long = 1;
  let c: Cpp.long = 1;

  AssertSameType(^a, c);
  AssertSameType(a & b, c);
  AssertSameType(a | b, c);
  AssertSameType(a ^ b, c);
  AssertSameType(a << b, c);
  AssertSameType(a >> b, c);
  //@dump-sem-ir-end
}

// --- bitwise_heterogeneous_long_left_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](a: T, b: T) {}

fn BitWiseHeterogeneousLongLeftSide() {
  let b: i32 = 1;
  //@dump-sem-ir-begin
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

  AssertSameType(a & b, a);
  AssertSameType(a | b, a);
  AssertSameType(a ^ b, a);
  AssertSameType(a << b, a);
  AssertSameType(a >> b, a);
  //@dump-sem-ir-end
}

// --- bitwise_heterogeneous_long_right_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](a: T, b: T) {}

fn BitWiseHeterogeneousLongRightSide() {
  let b: i32 = 1;
  //@dump-sem-ir-begin
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

  AssertSameType(b & a, a);
  AssertSameType(b | a, a);
  AssertSameType(b ^ a, a);
  AssertSameType(b << a, a);
  AssertSameType(b >> a, a);
  //@dump-sem-ir-end
}

// --- bitwise_heterogeneous_long_and_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](a: T, b: T) {}

fn BitWiseHeterogeneousLongAndI64() {
  //@dump-sem-ir-begin
  let a: Cpp.long = 1;
  let b: i64 = 1;

  AssertSameType(a & b, b);
  AssertSameType(b & a, b);
  //@dump-sem-ir-end
}

// --- fail_todo_bitwise_heterogeneous_long_and_i16.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](a: T, b: T) {}

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

fn AssertSameType[T:! type](a: T, b: T) {}

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
  let cpp_unsigned_long_result: Cpp.ULongResult = Cpp.PassULong(cpp_unsigned_long);

  let cpp_compat_ulong: Core.CppCompat.ULong32 = cpp_unsigned_long;
  let cpp_compat_ulong_result: Cpp.ULongResult = Cpp.PassULong(cpp_compat_ulong);

  let carbon_u32: u32 = cpp_unsigned_long;
  let carbon_u32_result: Cpp.UIntResult = Cpp.PassULong(carbon_u32);

  var a: array(f32, 1 as Cpp.unsigned_long) = (1.0,);
  //@dump-sem-ir-end
}

// --- copy_unsigned_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CopyUnsignedLong() {
  //@dump-sem-ir-begin
  var a: Cpp.unsigned_long = 1;
  var b: Cpp.unsigned_long = a;
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.94e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i64) [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2ed: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.d29: %Int.as.ImplicitAs.impl.Convert.type.2ed = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.556: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d48: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.556) [concrete]
// CHECK:STDOUT:   %.567: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.d48 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.288: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.229: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.d57: type = fn_type @As.Convert, @As(%i64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c71: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.cee: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a54: %Core.IntLiteral.as.As.impl.Convert.type.cee = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.229 = facet_value Core.IntLiteral, (%As.impl_witness.c71) [concrete]
// CHECK:STDOUT:   %.aba: type = fn_type_with_self_type %As.Convert.type.d57, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a54 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a54, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.41b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e86: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ea2, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.ccf: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.535: %Int.as.ImplicitAs.impl.Convert.type.ccf = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.234: %ImplicitAs.type.139 = facet_value %i64, (%ImplicitAs.impl_witness.e86) [concrete]
// CHECK:STDOUT:   %.3aa: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.234 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.41a, %Int.as.ImplicitAs.impl.Convert.535 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.535, @Int.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.d1e: <bound method> = bound_method %int_1.41a, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.674) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32.97e) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %.a6f: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1e4: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = @F.%i64.1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.0bc: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2ed) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.d29)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ea2 = impl_witness_table (%Core.import_ref.0bc), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31a = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, %i64.1 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.567 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc8_38.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_38.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.288]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc8_38.2(%int_1.loc8) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_38.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc8_38.2: %i64 = converted %int_1.loc8, %.loc8_38.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %cpp_long_long: %i64 = value_binding cpp_long_long, %.loc8_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_long.patt: %pattern_type.95b = value_binding_pattern carbon_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref: %i64 = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc9: type = splice_block %i64.loc9 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc9: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc9: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_long_long: %i64 = value_binding carbon_long_long, %cpp_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %.loc11_48.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_48: %.a6f = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc11_48.1: <bound method> = bound_method %float, %impl.elem0.loc11_48 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_48: <specific function> = specific_function %impl.elem0.loc11_48, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_48.2: <bound method> = bound_method %float, %specific_fn.loc11_48 [concrete = constants.%bound_method.1e4]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc11_48.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_48.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_48.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_48.4: init %f32.97e to %.loc11_48.3 = in_place_init %.loc11_48.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_48.5: init %array_type to %a.var = array_init (%.loc11_48.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_48.1, %.loc11_48.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_39: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc11: type = name_ref long_long, %i64.1 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc11_23.1: %.aba = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:     %bound_method.loc11_23.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_23.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_23.1: <specific function> = specific_function %impl.elem0.loc11_23.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_23.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_23.1 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc11_23.2(%int_1.loc11) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc11_23.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc11_23.2: %i64 = converted %int_1.loc11, %.loc11_23.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %impl.elem0.loc11_23.2: %.3aa = impl_witness_access constants.%ImplicitAs.impl_witness.e86, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.535]
// CHECK:STDOUT:     %bound_method.loc11_23.3: <bound method> = bound_method %.loc11_23.2, %impl.elem0.loc11_23.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_23.2: <specific function> = specific_function %impl.elem0.loc11_23.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_23.4: <bound method> = bound_method %.loc11_23.2, %specific_fn.loc11_23.2 [concrete = constants.%bound_method.d1e]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_23.4(%.loc11_23.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_23.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_23.4: Core.IntLiteral = converted %.loc11_23.2, %.loc11_23.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_23.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %array_type) = "no_op";
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.c65: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%u64) [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.46e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.7f8: type = fn_type @UInt.as.ImplicitAs.impl.Convert, @UInt.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.0ea: %UInt.as.ImplicitAs.impl.Convert.type.7f8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bb3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.899, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.fb8: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.042: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.fb8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.dfe: %ImplicitAs.type.584 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bb3) [concrete]
// CHECK:STDOUT:   %.da7: type = fn_type_with_self_type %ImplicitAs.Convert.type.c65, %ImplicitAs.facet.dfe [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.042 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.042, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.fec: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f23: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.f20: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.7eb: type = fn_type @As.Convert, @As(%u64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.979: %Core.IntLiteral.as.As.impl.Convert.type.2b1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.f54: <witness> = impl_witness imports.%As.impl_witness_table.7eb, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0f2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.715: %Core.IntLiteral.as.As.impl.Convert.type.0f2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.f20 = facet_value Core.IntLiteral, (%As.impl_witness.f54) [concrete]
// CHECK:STDOUT:   %.11a: type = fn_type_with_self_type %As.Convert.type.7eb, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.715 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.715, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.9cf: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2cc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.493, @UInt.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.812: type = fn_type @UInt.as.ImplicitAs.impl.Convert, @UInt.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.309: %UInt.as.ImplicitAs.impl.Convert.type.812 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.02d: %ImplicitAs.type.139 = facet_value %u64, (%ImplicitAs.impl_witness.2cc) [concrete]
// CHECK:STDOUT:   %.a1c: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.02d [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.f23, %UInt.as.ImplicitAs.impl.Convert.309 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.309, @UInt.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.f18: <bound method> = bound_method %int_1.f23, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.674) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32.97e) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %.a6f: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1e4: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long_long = @F.%u64.1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.741: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.46e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.899 = impl_witness_table (%Core.import_ref.741), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.483: @UInt.as.ImplicitAs.impl.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.7f8) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.0ea)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.493 = impl_witness_table (%Core.import_ref.483), @UInt.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.600: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.2b1) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.979)]
// CHECK:STDOUT:   %As.impl_witness_table.7eb = impl_witness_table (%Core.import_ref.600), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31a = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %unsigned_long_long.ref.loc8: type = name_ref unsigned_long_long, %u64.1 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.da7 = impl_witness_access constants.%ImplicitAs.impl_witness.bb3, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.042]
// CHECK:STDOUT:   %bound_method.loc8_56.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_56.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.fec]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc8_56.2(%int_1.loc8) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc8_56.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc8_56.2: %u64 = converted %int_1.loc8, %.loc8_56.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = value_binding cpp_unsigned_long_long, %.loc8_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long_long.patt: %pattern_type.157 = value_binding_pattern carbon_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref: %u64 = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc9: type = splice_block %u64.loc9 [concrete = constants.%u64] {
// CHECK:STDOUT:     %int_64.loc9: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %u64.loc9: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_unsigned_long_long: %u64 = value_binding carbon_unsigned_long_long, %cpp_unsigned_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %.loc11_57.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_57: %.a6f = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc11_57.1: <bound method> = bound_method %float, %impl.elem0.loc11_57 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_57: <specific function> = specific_function %impl.elem0.loc11_57, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_57.2: <bound method> = bound_method %float, %specific_fn.loc11_57 [concrete = constants.%bound_method.1e4]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc11_57.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_57.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_57.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_57.4: init %f32.97e to %.loc11_57.3 = in_place_init %.loc11_57.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc11_57.5: init %array_type to %a.var = array_init (%.loc11_57.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_57.1, %.loc11_57.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_48: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc11: type = name_ref unsigned_long_long, %u64.1 [concrete = constants.%u64]
// CHECK:STDOUT:     %impl.elem0.loc11_23.1: %.11a = impl_witness_access constants.%As.impl_witness.f54, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.715]
// CHECK:STDOUT:     %bound_method.loc11_23.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_23.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_23.1: <specific function> = specific_function %impl.elem0.loc11_23.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_23.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_23.1 [concrete = constants.%bound_method.9cf]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc11_23.2(%int_1.loc11) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc11_23.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc11_23.2: %u64 = converted %int_1.loc11, %.loc11_23.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %impl.elem0.loc11_23.2: %.a1c = impl_witness_access constants.%ImplicitAs.impl_witness.2cc, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.309]
// CHECK:STDOUT:     %bound_method.loc11_23.3: <bound method> = bound_method %.loc11_23.2, %impl.elem0.loc11_23.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_23.2: <specific function> = specific_function %impl.elem0.loc11_23.2, @UInt.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_23.4: <bound method> = bound_method %.loc11_23.2, %specific_fn.loc11_23.2 [concrete = constants.%bound_method.f18]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_23.4(%.loc11_23.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_23.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_23.4: Core.IntLiteral = converted %.loc11_23.2, %.loc11_23.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_23.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %array_type) = "no_op";
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2ad: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.94e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bbc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cb1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ba4: %ImplicitAs.type.2ad = facet_value %Cpp.long, (%ImplicitAs.impl_witness.bbc) [concrete]
// CHECK:STDOUT:   %.8a2: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.ba4 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.431: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.bbf: %Cpp.long.as.ImplicitAs.impl.Convert.type.431 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.508: type = facet_type <@As, @As(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.229: type = fn_type @As.Convert, @As(%Cpp.long) [concrete]
// CHECK:STDOUT:   %As.impl_witness.60c: <witness> = impl_witness imports.%As.impl_witness_table.797 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.508 = facet_value Core.IntLiteral, (%As.impl_witness.60c) [concrete]
// CHECK:STDOUT:   %.040: type = fn_type_with_self_type %As.Convert.type.229, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e4d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a09 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.772: %ImplicitAs.type.139 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.e4d) [concrete]
// CHECK:STDOUT:   %.8e3: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.772 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.9ed: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.d49: %Cpp.long.as.ImplicitAs.impl.Convert.type.9ed = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5a4, %Cpp.long.as.ImplicitAs.impl.Convert.d49 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.674) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32.97e) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %.a6f: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc26 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.297: %Cpp.long.as.ImplicitAs.impl.Convert.type.431 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bbf]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cb1 = impl_witness_table (%Core.import_ref.297), @Cpp.long.as.ImplicitAs.impl.e6d [concrete]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.686: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.797 = impl_witness_table (%Core.import_ref.686), @Core.IntLiteral.as.As.impl.a00 [concrete]
// CHECK:STDOUT:   %Core.import_ref.168: %Cpp.long.as.ImplicitAs.impl.Convert.type.9ed = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.d49]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a09 = impl_witness_table (%Core.import_ref.168), @Cpp.long.as.ImplicitAs.impl.e94 [concrete]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31a = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc15: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc15_28.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc15_28.2: %Cpp.long = converted %int_1.loc15, %.loc15_28.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %cpp_long: %Cpp.long = value_binding cpp_long, %.loc15_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_result.patt: %pattern_type.cd9 = value_binding_pattern cpp_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_41: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc16: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long.ref.loc16: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc16_62.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.305 = addr_of %.loc16_62.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.1(%cpp_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_62.2: init %LongResult to %.loc16_62.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_27: type = splice_block %LongResult.ref.loc16 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc16: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_62.3: ref %LongResult = temporary %.loc16_62.1, %.loc16_62.2
// CHECK:STDOUT:   %.loc16_62.4: %LongResult = acquire_value %.loc16_62.3
// CHECK:STDOUT:   %cpp_long_result: %LongResult = value_binding cpp_long_result, %.loc16_62.4
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
// CHECK:STDOUT:   %Cpp.ref.loc19_48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc19: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long.ref: %Cpp.long = name_ref cpp_compat_long, %cpp_compat_long
// CHECK:STDOUT:   %.loc19_76.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.305 = addr_of %.loc19_76.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.1(%cpp_compat_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_76.2: init %LongResult to %.loc19_76.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_34: type = splice_block %LongResult.ref.loc19 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc19: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_76.3: ref %LongResult = temporary %.loc19_76.1, %.loc19_76.2
// CHECK:STDOUT:   %.loc19_76.4: %LongResult = acquire_value %.loc19_76.3
// CHECK:STDOUT:   %cpp_compat_long_result: %LongResult = value_binding cpp_compat_long_result, %.loc19_76.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32.patt: %pattern_type.7ce = value_binding_pattern carbon_i32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc21: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %int_32.loc21_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc21_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc21_34.1: %i32 = as_compatible %cpp_long.ref.loc21
// CHECK:STDOUT:   %.loc21_34.2: %i32 = converted %cpp_long.ref.loc21, %.loc21_34.1
// CHECK:STDOUT:   %.loc21_19: type = splice_block %i32.loc21_19 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc21_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %carbon_i32: %i32 = value_binding carbon_i32, %.loc21_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32_result.patt: %pattern_type.454 = value_binding_pattern carbon_i32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc22: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i32.ref: %i32 = name_ref carbon_i32, %carbon_i32
// CHECK:STDOUT:   %.loc22_65.1: ref %IntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.3cb = addr_of %.loc22_65.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.2(%carbon_i32.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_65.2: init %IntResult to %.loc22_65.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_29: type = splice_block %IntResult.ref [concrete = constants.%IntResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntResult.ref: type = name_ref IntResult, imports.%IntResult.decl [concrete = constants.%IntResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_65.3: ref %IntResult = temporary %.loc22_65.1, %.loc22_65.2
// CHECK:STDOUT:   %.loc22_65.4: %IntResult = acquire_value %.loc22_65.3
// CHECK:STDOUT:   %carbon_i32_result: %IntResult = value_binding carbon_i32_result, %.loc22_65.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.95b = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc24: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc24_19: type = splice_block %i64 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc24: %.8a2 = impl_witness_access constants.%ImplicitAs.impl_witness.bbc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bbf]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %cpp_long.ref.loc24, %impl.elem0.loc24
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc24: init %i64 = call %bound_method.loc24(%cpp_long.ref.loc24)
// CHECK:STDOUT:   %.loc24_25.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc24
// CHECK:STDOUT:   %.loc24_25.2: %i64 = converted %cpp_long.ref.loc24, %.loc24_25.1
// CHECK:STDOUT:   %carbon_i64: %i64 = value_binding carbon_i64, %.loc24_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %.loc26_43.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc26_43: %.a6f = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc26_43.1: <bound method> = bound_method %float, %impl.elem0.loc26_43 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc26_43, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_43.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc26_43.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26_43.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc26_43.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc26_43.4: init %f32.97e to %.loc26_43.3 = in_place_init %.loc26_43.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26_43.5: init %array_type to %a.var = array_init (%.loc26_43.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc26_3: init %array_type = converted %.loc26_43.1, %.loc26_43.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc26_3
// CHECK:STDOUT:   %.loc26_34: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc26: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:     %impl.elem0.loc26_23.1: %.040 = impl_witness_access constants.%As.impl_witness.60c, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc26_23.1: <bound method> = bound_method %int_1.loc26, %impl.elem0.loc26_23.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long = call %bound_method.loc26_23.1(%int_1.loc26) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %.loc26_23.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %.loc26_23.2: %Cpp.long = converted %int_1.loc26, %.loc26_23.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %impl.elem0.loc26_23.2: %.8e3 = impl_witness_access constants.%ImplicitAs.impl_witness.e4d, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.d49]
// CHECK:STDOUT:     %bound_method.loc26_23.2: <bound method> = bound_method %.loc26_23.2, %impl.elem0.loc26_23.2 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long.as.ImplicitAs.impl.Convert.call.loc26: init Core.IntLiteral = call %bound_method.loc26_23.2(%.loc26_23.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc26_23.3: Core.IntLiteral = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc26 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc26_23.4: Core.IntLiteral = converted %.loc26_23.2, %.loc26_23.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc26_23.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   %IntResult.cpp_destructor.bound: <bound method> = bound_method %.loc22_65.3, constants.%IntResult.cpp_destructor
// CHECK:STDOUT:   %IntResult.cpp_destructor.call: init %empty_tuple.type = call %IntResult.cpp_destructor.bound(%.loc22_65.3)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_76.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call.loc19: init %empty_tuple.type = call %LongResult.cpp_destructor.bound.loc19(%.loc19_76.3)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound.loc16: <bound method> = bound_method %.loc16_62.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call.loc16: init %empty_tuple.type = call %LongResult.cpp_destructor.bound.loc16(%.loc16_62.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc26(%self.param: %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc22(%self.param: %IntResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc19(%self.param: %LongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.e75: <witness> = impl_witness imports.%Copy.impl_witness_table.572 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.e75) [concrete]
// CHECK:STDOUT:   %.03e: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.915: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.572 = impl_witness_table (%Core.import_ref.915), @Cpp.long.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %.loc9_21: %Cpp.long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.03e = impl_witness_access constants.%Copy.impl_witness.e75, element0 [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_21, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.call: init %Cpp.long = call %bound_method.loc9(%.loc9_21)
// CHECK:STDOUT:   assign %b.var, %Cpp.long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long.ref.loc9 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc9: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = ref_binding b, %b.var
// CHECK:STDOUT:   %DestroyOp.bound.loc9: <bound method> = bound_method %b.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc9: init %empty_tuple.type = call %DestroyOp.bound.loc9(%b.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc8: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc8: init %empty_tuple.type = call %DestroyOp.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_homogeneous_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.494: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %EqWith.Equal.type.eeb: type = fn_type @EqWith.Equal, @EqWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %EqWith.NotEqual.type.d83: type = fn_type @EqWith.NotEqual, @EqWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.6db: <witness> = impl_witness imports.%EqWith.impl_witness_table.322 [concrete]
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.494 = facet_value %Cpp.long, (%EqWith.impl_witness.6db) [concrete]
// CHECK:STDOUT:   %.a8e: type = fn_type_with_self_type %EqWith.Equal.type.eeb, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.cc0: type = fn_type @Cpp.long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.1b7: %Cpp.long.as.EqWith.impl.Equal.type.cc0 = struct_value () [concrete]
// CHECK:STDOUT:   %.fbc: type = fn_type_with_self_type %EqWith.NotEqual.type.d83, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.75f: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.5de: %Cpp.long.as.EqWith.impl.NotEqual.type.75f = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.a39: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.Less.type.3c2: type = fn_type @OrderedWith.Less, @OrderedWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %OrderedWith.LessOrEquivalent.type.d58: type = fn_type @OrderedWith.LessOrEquivalent, @OrderedWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %OrderedWith.Greater.type.a51: type = fn_type @OrderedWith.Greater, @OrderedWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %OrderedWith.GreaterOrEquivalent.type.3df: type = fn_type @OrderedWith.GreaterOrEquivalent, @OrderedWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.2b3: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.e82 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.a39 = facet_value %Cpp.long, (%OrderedWith.impl_witness.2b3) [concrete]
// CHECK:STDOUT:   %.3ad: type = fn_type_with_self_type %OrderedWith.Greater.type.a51, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.da5: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.cc7: %Cpp.long.as.OrderedWith.impl.Greater.type.da5 = struct_value () [concrete]
// CHECK:STDOUT:   %.b8a: type = fn_type_with_self_type %OrderedWith.Less.type.3c2, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.d1e: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.911: %Cpp.long.as.OrderedWith.impl.Less.type.d1e = struct_value () [concrete]
// CHECK:STDOUT:   %.805: type = fn_type_with_self_type %OrderedWith.GreaterOrEquivalent.type.3df, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.9c8: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.337: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.9c8 = struct_value () [concrete]
// CHECK:STDOUT:   %.6d7: type = fn_type_with_self_type %OrderedWith.LessOrEquivalent.type.d58, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.cd7: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.187: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.cd7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.869: %Cpp.long.as.EqWith.impl.Equal.type.cc0 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1b7]
// CHECK:STDOUT:   %Core.import_ref.f17: %Cpp.long.as.EqWith.impl.NotEqual.type.75f = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.5de]
// CHECK:STDOUT:   %EqWith.impl_witness_table.322 = impl_witness_table (%Core.import_ref.869, %Core.import_ref.f17), @Cpp.long.as.EqWith.impl.7c8 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9ae: %Cpp.long.as.OrderedWith.impl.Less.type.d1e = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.911]
// CHECK:STDOUT:   %Core.import_ref.18a: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.cd7 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.187]
// CHECK:STDOUT:   %Core.import_ref.8df: %Cpp.long.as.OrderedWith.impl.Greater.type.da5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.cc7]
// CHECK:STDOUT:   %Core.import_ref.deb: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.9c8 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.337]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.e82 = impl_witness_table (%Core.import_ref.9ae, %Core.import_ref.18a, %Core.import_ref.8df, %Core.import_ref.deb), @Cpp.long.as.OrderedWith.impl.361 [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc9_21.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.a8e = impl_witness_access constants.%EqWith.impl_witness.6db, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1b7]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.fbc = impl_witness_access constants.%EqWith.impl_witness.6db, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.5de]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.3ad = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.cc7]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.b8a = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.911]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.805 = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.337]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.6d7 = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.187]
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.d8f: type = facet_type <@EqWith, @EqWith(%i32)> [concrete]
// CHECK:STDOUT:   %EqWith.Equal.type.874: type = fn_type @EqWith.Equal, @EqWith(%i32) [concrete]
// CHECK:STDOUT:   %EqWith.NotEqual.type.a9a: type = fn_type @EqWith.NotEqual, @EqWith(%i32) [concrete]
// CHECK:STDOUT:   %T.57d: %ImplicitAs.type.819 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.f13(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.c90ef2.1: %Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.f13(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.c90ef2.2: %Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.7da93b.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.f13(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.c91b06.1: %Cpp.long.as.EqWith.impl.Equal.type.7da93b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.7da93b.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.f13(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.c91b06.2: %Cpp.long.as.EqWith.impl.Equal.type.7da93b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.ded: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %EqWith.NotEqual.type.b22: type = fn_type @EqWith.NotEqual, @EqWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %EqWith.Equal.type.f75: type = fn_type @EqWith.Equal, @EqWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.15a: <witness> = impl_witness imports.%EqWith.impl_witness_table.44b, @Cpp.long.as.EqWith.impl.f13(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.f13(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.0f3e3b.1: %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.f13(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.6a28a6.1: %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.f13(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.0f3e3b.2: %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.f13(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.6a28a6.2: %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.fd1: %EqWith.type.d8f = facet_value %Cpp.long, (%EqWith.impl_witness.15a) [concrete]
// CHECK:STDOUT:   %.d7e: type = fn_type_with_self_type %EqWith.Equal.type.874, %EqWith.facet.fd1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.0f3e3b.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.c45: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.0f3e3b.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.397: type = fn_type_with_self_type %EqWith.NotEqual.type.a9a, %EqWith.facet.fd1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.6a28a6.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.6a28a6.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.243: type = facet_type <@OrderedWith, @OrderedWith(%i32)> [concrete]
// CHECK:STDOUT:   %OrderedWith.Less.type.815: type = fn_type @OrderedWith.Less, @OrderedWith(%i32) [concrete]
// CHECK:STDOUT:   %OrderedWith.LessOrEquivalent.type.257: type = fn_type @OrderedWith.LessOrEquivalent, @OrderedWith(%i32) [concrete]
// CHECK:STDOUT:   %OrderedWith.Greater.type.402: type = fn_type @OrderedWith.Greater, @OrderedWith(%i32) [concrete]
// CHECK:STDOUT:   %OrderedWith.GreaterOrEquivalent.type.6fe: type = fn_type @OrderedWith.GreaterOrEquivalent, @OrderedWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.74811b.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.15a(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.dd28ba.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.74811b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.74811b.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.15a(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.dd28ba.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.74811b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.662b24.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.15a(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.02ce94.1: %Cpp.long.as.OrderedWith.impl.Greater.type.662b24.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.662b24.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.15a(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.02ce94.2: %Cpp.long.as.OrderedWith.impl.Greater.type.662b24.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4aa4c3.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.15a(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.7e103a.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4aa4c3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4aa4c3.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.15a(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.7e103a.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4aa4c3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.a59daa.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.15a(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.954f71.1: %Cpp.long.as.OrderedWith.impl.Less.type.a59daa.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.a59daa.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.15a(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.954f71.2: %Cpp.long.as.OrderedWith.impl.Less.type.a59daa.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.358: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.GreaterOrEquivalent.type.e24: type = fn_type @OrderedWith.GreaterOrEquivalent, @OrderedWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %OrderedWith.Greater.type.30c: type = fn_type @OrderedWith.Greater, @OrderedWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %OrderedWith.LessOrEquivalent.type.776: type = fn_type @OrderedWith.LessOrEquivalent, @OrderedWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %OrderedWith.Less.type.20d: type = fn_type @OrderedWith.Less, @OrderedWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.576: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.9c6, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.5299ce.1: %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.9898ff.1: %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.5299ce.2: %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.9898ff.2: %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.19d: %OrderedWith.type.243 = facet_value %Cpp.long, (%OrderedWith.impl_witness.576) [concrete]
// CHECK:STDOUT:   %.528: type = fn_type_with_self_type %OrderedWith.Greater.type.402, %OrderedWith.facet.19d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.9898ff.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.9898ff.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.3e22: type = fn_type_with_self_type %OrderedWith.Less.type.815, %OrderedWith.facet.19d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.5299ce.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.5299ce.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.723: type = fn_type_with_self_type %OrderedWith.GreaterOrEquivalent.type.6fe, %OrderedWith.facet.19d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.d50: type = fn_type_with_self_type %OrderedWith.LessOrEquivalent.type.257, %OrderedWith.facet.19d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9548d5.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9548d5.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.8e5: <witness> = impl_witness imports.%EqWith.impl_witness_table.44b, @Cpp.long.as.EqWith.impl.f13(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.416d09.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.f13(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.6ed862.1: %Cpp.long.as.EqWith.impl.Equal.type.416d09.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.de968c.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.f13(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.f686b3.1: %Cpp.long.as.EqWith.impl.NotEqual.type.de968c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.416d09.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.f13(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.6ed862.2: %Cpp.long.as.EqWith.impl.Equal.type.416d09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.de968c.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.f13(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.f686b3.2: %Cpp.long.as.EqWith.impl.NotEqual.type.de968c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.3d4: %EqWith.type.ded = facet_value %Cpp.long, (%EqWith.impl_witness.8e5) [concrete]
// CHECK:STDOUT:   %.64d: type = fn_type_with_self_type %EqWith.Equal.type.f75, %EqWith.facet.3d4 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.334ae9.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.6ed862.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.334ae9.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.6ed862.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.2f9: type = fn_type_with_self_type %EqWith.NotEqual.type.b22, %EqWith.facet.3d4 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5aa006.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.f686b3.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5aa006.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.f686b3.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.52f: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.9c6, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.8ced31.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.0de767.1: %Cpp.long.as.OrderedWith.impl.Less.type.8ced31.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.28524d.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.28524d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.3fd1ac.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.74a807.1: %Cpp.long.as.OrderedWith.impl.Greater.type.3fd1ac.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2273d7.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2273d7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.8ced31.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.0de767.2: %Cpp.long.as.OrderedWith.impl.Less.type.8ced31.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.28524d.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.28524d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.3fd1ac.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.74a807.2: %Cpp.long.as.OrderedWith.impl.Greater.type.3fd1ac.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2273d7.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.15a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2273d7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.eef: %OrderedWith.type.358 = facet_value %Cpp.long, (%OrderedWith.impl_witness.52f) [concrete]
// CHECK:STDOUT:   %.07e: type = fn_type_with_self_type %OrderedWith.Greater.type.30c, %OrderedWith.facet.eef [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.b2149b.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.74a807.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.b2149b.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.74a807.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.66b: type = fn_type_with_self_type %OrderedWith.Less.type.20d, %OrderedWith.facet.eef [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.b6fc5e.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.0de767.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.b6fc5e.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.0de767.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.d0e: type = fn_type_with_self_type %OrderedWith.GreaterOrEquivalent.type.e24, %OrderedWith.facet.eef [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f90679.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f90679.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.4d6: type = fn_type_with_self_type %OrderedWith.LessOrEquivalent.type.776, %OrderedWith.facet.eef [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.c5c24a.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.c5c24a.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.489: @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.Equal.type.2 (%Cpp.long.as.EqWith.impl.Equal.type.7da93b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.Equal.2 (constants.%Cpp.long.as.EqWith.impl.Equal.c91b06.1)]
// CHECK:STDOUT:   %Core.import_ref.a67: @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long.as.EqWith.impl.NotEqual.c90ef2.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.44b = impl_witness_table (%Core.import_ref.489, %Core.import_ref.a67), @Cpp.long.as.EqWith.impl.f13 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.b1f: @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long.as.EqWith.impl.NotEqual.c90ef2.2)]
// CHECK:STDOUT:   %Core.Equal.0ad: @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.Equal.type.1 (%Cpp.long.as.EqWith.impl.Equal.type.7da93b.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.Equal.1 (constants.%Cpp.long.as.EqWith.impl.Equal.c91b06.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.959: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Less.type.2 (%Cpp.long.as.OrderedWith.impl.Less.type.a59daa.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long.as.OrderedWith.impl.Less.954f71.1)]
// CHECK:STDOUT:   %Core.import_ref.c231: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4aa4c3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.7e103a.1)]
// CHECK:STDOUT:   %Core.import_ref.df7b: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long.as.OrderedWith.impl.Greater.type.662b24.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long.as.OrderedWith.impl.Greater.02ce94.1)]
// CHECK:STDOUT:   %Core.import_ref.a61: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.74811b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.dd28ba.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.9c6 = impl_witness_table (%Core.import_ref.959, %Core.import_ref.c231, %Core.import_ref.df7b, %Core.import_ref.a61), @Cpp.long.as.OrderedWith.impl.15a [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.76e: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.74811b.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.dd28ba.2)]
// CHECK:STDOUT:   %Core.Greater.9da: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long.as.OrderedWith.impl.Greater.type.662b24.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long.as.OrderedWith.impl.Greater.02ce94.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.4a6: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4aa4c3.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.7e103a.2)]
// CHECK:STDOUT:   %Core.Less.c86: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Less.type.1 (%Cpp.long.as.OrderedWith.impl.Less.type.a59daa.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long.as.OrderedWith.impl.Less.954f71.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc9_21.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.d7e = impl_witness_access constants.%EqWith.impl_witness.15a, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.1 = specific_constant imports.%Core.Equal.0ad, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.1 = name_ref Equal, %.loc10_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.5: %Cpp.long = converted %b.ref.loc10, %.loc10_5.4
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.397 = impl_witness_access constants.%EqWith.impl_witness.15a, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.1 = specific_constant imports.%Core.NotEqual.b1f, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.1 = name_ref NotEqual, %.loc11_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.5: %Cpp.long = converted %b.ref.loc11, %.loc11_5.4
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.528 = impl_witness_access constants.%OrderedWith.impl_witness.576, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.1 = specific_constant imports.%Core.Greater.9da, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.1 = name_ref Greater, %.loc12_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.5: %Cpp.long = converted %b.ref.loc12, %.loc12_5.4
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_7: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_7: init %Cpp.long = call %bound_method.loc12_7(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_7
// CHECK:STDOUT:   %.loc12_7.2: %Cpp.long = converted %b.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.3e22 = impl_witness_access constants.%OrderedWith.impl_witness.576, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.1 = specific_constant imports.%Core.Less.c86, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.1 = name_ref Less, %.loc13_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.5: %Cpp.long = converted %b.ref.loc13, %.loc13_5.4
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_7: init %Cpp.long = call %bound_method.loc13_7(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_7
// CHECK:STDOUT:   %.loc13_7.2: %Cpp.long = converted %b.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.723 = impl_witness_access constants.%OrderedWith.impl_witness.576, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.1 = specific_constant imports.%Core.GreaterOrEquivalent.76e, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.1 = name_ref GreaterOrEquivalent, %.loc14_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.5: %Cpp.long = converted %b.ref.loc14, %.loc14_5.4
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.d50 = impl_witness_access constants.%OrderedWith.impl_witness.576, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc15_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc15_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc15_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9548d5.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.1 = specific_constant imports.%Core.LessOrEquivalent.4a6, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.1 = name_ref LessOrEquivalent, %.loc15_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_5: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_5: init %Cpp.long = call %bound_method.loc15_5.3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_5
// CHECK:STDOUT:   %.loc15_5.5: %Cpp.long = converted %b.ref.loc15, %.loc15_5.4
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9548d5.2]
// CHECK:STDOUT:   %bound_method.loc15_5.4: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_8: init %Cpp.long = call %bound_method.loc15_8(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_8
// CHECK:STDOUT:   %.loc15_8.2: %Cpp.long = converted %b.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.4(%a.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.64d = impl_witness_access constants.%EqWith.impl_witness.8e5, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.6ed862.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.334ae9.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long.as.EqWith.impl.Equal.type.416d09.1 = specific_constant imports.%Core.Equal.0ad, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.6ed862.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long.as.EqWith.impl.Equal.type.416d09.1 = name_ref Equal, %.loc17_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.6ed862.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long = call %bound_method.loc17_5.3(%int_1.loc17) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_5.5: %Cpp.long = converted %int_1.loc17, %.loc17_5.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.334ae9.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.2f9 = impl_witness_access constants.%EqWith.impl_witness.8e5, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.f686b3.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5aa006.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long.as.EqWith.impl.NotEqual.type.de968c.1 = specific_constant imports.%Core.NotEqual.b1f, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.f686b3.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long.as.EqWith.impl.NotEqual.type.de968c.1 = name_ref NotEqual, %.loc18_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.f686b3.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long = call %bound_method.loc18_5.3(%int_1.loc18) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_5.5: %Cpp.long = converted %int_1.loc18, %.loc18_5.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5aa006.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.07e = impl_witness_access constants.%OrderedWith.impl_witness.52f, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.74a807.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.b2149b.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long.as.OrderedWith.impl.Greater.type.3fd1ac.1 = specific_constant imports.%Core.Greater.9da, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.74a807.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long.as.OrderedWith.impl.Greater.type.3fd1ac.1 = name_ref Greater, %.loc19_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.74a807.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long = call %bound_method.loc19_5.3(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_5.5: %Cpp.long = converted %int_1.loc19, %.loc19_5.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.b2149b.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_7: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7: init %Cpp.long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.66b = impl_witness_access constants.%OrderedWith.impl_witness.52f, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.0de767.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.b6fc5e.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long.as.OrderedWith.impl.Less.type.8ced31.1 = specific_constant imports.%Core.Less.c86, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.0de767.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long.as.OrderedWith.impl.Less.type.8ced31.1 = name_ref Less, %.loc20_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.0de767.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long = call %bound_method.loc20_5.3(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_5.5: %Cpp.long = converted %int_1.loc20, %.loc20_5.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.b6fc5e.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7: init %Cpp.long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.d0e = impl_witness_access constants.%OrderedWith.impl_witness.52f, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f90679.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2273d7.1 = specific_constant imports.%Core.GreaterOrEquivalent.76e, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2273d7.1 = name_ref GreaterOrEquivalent, %.loc21_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_5: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5: init %Cpp.long = call %bound_method.loc21_5.3(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_5.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_5.5: %Cpp.long = converted %int_1.loc21, %.loc21_5.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f90679.2]
// CHECK:STDOUT:   %bound_method.loc21_5.4: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8: init %Cpp.long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.4(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.4d6 = impl_witness_access constants.%OrderedWith.impl_witness.52f, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_5.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc22_5.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_5.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.c5c24a.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.28524d.1 = specific_constant imports.%Core.LessOrEquivalent.4a6, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.28524d.1 = name_ref LessOrEquivalent, %.loc22_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_5: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5: init %Cpp.long = call %bound_method.loc22_5.3(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_5.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_5.5: %Cpp.long = converted %int_1.loc22, %.loc22_5.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.c5c24a.2]
// CHECK:STDOUT:   %bound_method.loc22_5.4: <bound method> = bound_method %a.ref.loc22, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_8: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8: init %Cpp.long = call %bound_method.loc22_8(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_8.2: %Cpp.long = converted %int_1.loc22, %.loc22_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc22: init bool = call %bound_method.loc22_5.4(%a.ref.loc22, %.loc22_8.2)
// CHECK:STDOUT:   %a.ref.loc24: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc24: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc24_5.1: %.d7e = impl_witness_access constants.%EqWith.impl_witness.15a, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.2]
// CHECK:STDOUT:   %bound_method.loc24_5.1: <bound method> = bound_method %a.ref.loc24, %impl.elem0.loc24_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc24_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc24_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc24_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc24_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc24_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc24_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_5.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.1]
// CHECK:STDOUT:   %bound_method.loc24_5.2: <bound method> = bound_method %a.ref.loc24, %specific_fn.loc24
// CHECK:STDOUT:   %.loc24_5.3: %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.1 = specific_constant imports.%Core.Equal.0ad, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.1]
// CHECK:STDOUT:   %Equal.ref.loc24: %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.1 = name_ref Equal, %.loc24_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc24: <bound method> = bound_method %a.ref.loc24, %Equal.ref.loc24
// CHECK:STDOUT:   %impl.elem0.loc24_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc24_5.3: <bound method> = bound_method %b.ref.loc24, %impl.elem0.loc24_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc24_5: init %Cpp.long = call %bound_method.loc24_5.3(%b.ref.loc24)
// CHECK:STDOUT:   %.loc24_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc24_5
// CHECK:STDOUT:   %.loc24_5.5: %Cpp.long = converted %b.ref.loc24, %.loc24_5.4
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc24: <specific function> = specific_function %Equal.ref.loc24, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.2]
// CHECK:STDOUT:   %bound_method.loc24_5.4: <bound method> = bound_method %a.ref.loc24, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc24
// CHECK:STDOUT:   %impl.elem0.loc24_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc24_8: <bound method> = bound_method %b.ref.loc24, %impl.elem0.loc24_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc24_8: init %Cpp.long = call %bound_method.loc24_8(%b.ref.loc24)
// CHECK:STDOUT:   %.loc24_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc24_8
// CHECK:STDOUT:   %.loc24_8.2: %Cpp.long = converted %b.ref.loc24, %.loc24_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc24: init bool = call %bound_method.loc24_5.4(%a.ref.loc24, %.loc24_8.2)
// CHECK:STDOUT:   %a.ref.loc25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc25: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc25: %.397 = impl_witness_access constants.%EqWith.impl_witness.15a, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.2]
// CHECK:STDOUT:   %bound_method.loc25_5.1: <bound method> = bound_method %a.ref.loc25, %impl.elem1.loc25
// CHECK:STDOUT:   %ImplicitAs.facet.loc25_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc25_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc25_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc25_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc25_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc25_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem1.loc25, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.1]
// CHECK:STDOUT:   %bound_method.loc25_5.2: <bound method> = bound_method %a.ref.loc25, %specific_fn.loc25
// CHECK:STDOUT:   %.loc25_5.3: %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.1 = specific_constant imports.%Core.NotEqual.b1f, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.1]
// CHECK:STDOUT:   %NotEqual.ref.loc25: %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.1 = name_ref NotEqual, %.loc25_5.3 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc25: <bound method> = bound_method %a.ref.loc25, %NotEqual.ref.loc25
// CHECK:STDOUT:   %impl.elem0.loc25_5: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_5.3: <bound method> = bound_method %b.ref.loc25, %impl.elem0.loc25_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc25_5: init %Cpp.long = call %bound_method.loc25_5.3(%b.ref.loc25)
// CHECK:STDOUT:   %.loc25_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc25_5
// CHECK:STDOUT:   %.loc25_5.5: %Cpp.long = converted %b.ref.loc25, %.loc25_5.4
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc25: <specific function> = specific_function %NotEqual.ref.loc25, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.2]
// CHECK:STDOUT:   %bound_method.loc25_5.4: <bound method> = bound_method %a.ref.loc25, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc25
// CHECK:STDOUT:   %impl.elem0.loc25_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_8: <bound method> = bound_method %b.ref.loc25, %impl.elem0.loc25_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc25_8: init %Cpp.long = call %bound_method.loc25_8(%b.ref.loc25)
// CHECK:STDOUT:   %.loc25_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc25_8
// CHECK:STDOUT:   %.loc25_8.2: %Cpp.long = converted %b.ref.loc25, %.loc25_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc25: init bool = call %bound_method.loc25_5.4(%a.ref.loc25, %.loc25_8.2)
// CHECK:STDOUT:   %a.ref.loc26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc26: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc26: %.528 = impl_witness_access constants.%OrderedWith.impl_witness.576, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.2]
// CHECK:STDOUT:   %bound_method.loc26_5.1: <bound method> = bound_method %a.ref.loc26, %impl.elem2.loc26
// CHECK:STDOUT:   %ImplicitAs.facet.loc26_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc26_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc26_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc26_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc26_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc26_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem2.loc26, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.1]
// CHECK:STDOUT:   %bound_method.loc26_5.2: <bound method> = bound_method %a.ref.loc26, %specific_fn.loc26
// CHECK:STDOUT:   %.loc26_5.3: %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.1 = specific_constant imports.%Core.Greater.9da, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.1]
// CHECK:STDOUT:   %Greater.ref.loc26: %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.1 = name_ref Greater, %.loc26_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc26: <bound method> = bound_method %a.ref.loc26, %Greater.ref.loc26
// CHECK:STDOUT:   %impl.elem0.loc26_5: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_5.3: <bound method> = bound_method %b.ref.loc26, %impl.elem0.loc26_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc26_5: init %Cpp.long = call %bound_method.loc26_5.3(%b.ref.loc26)
// CHECK:STDOUT:   %.loc26_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc26_5
// CHECK:STDOUT:   %.loc26_5.5: %Cpp.long = converted %b.ref.loc26, %.loc26_5.4
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc26: <specific function> = specific_function %Greater.ref.loc26, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.2]
// CHECK:STDOUT:   %bound_method.loc26_5.4: <bound method> = bound_method %a.ref.loc26, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc26
// CHECK:STDOUT:   %impl.elem0.loc26_7: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_7: <bound method> = bound_method %b.ref.loc26, %impl.elem0.loc26_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc26_7: init %Cpp.long = call %bound_method.loc26_7(%b.ref.loc26)
// CHECK:STDOUT:   %.loc26_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc26_7
// CHECK:STDOUT:   %.loc26_7.2: %Cpp.long = converted %b.ref.loc26, %.loc26_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc26: init bool = call %bound_method.loc26_5.4(%a.ref.loc26, %.loc26_7.2)
// CHECK:STDOUT:   %a.ref.loc27: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc27: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc27_5.1: %.3e22 = impl_witness_access constants.%OrderedWith.impl_witness.576, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.2]
// CHECK:STDOUT:   %bound_method.loc27_5.1: <bound method> = bound_method %a.ref.loc27, %impl.elem0.loc27_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc27_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc27_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc27_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc27_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc27_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc27_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27_5.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.1]
// CHECK:STDOUT:   %bound_method.loc27_5.2: <bound method> = bound_method %a.ref.loc27, %specific_fn.loc27
// CHECK:STDOUT:   %.loc27_5.3: %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.1 = specific_constant imports.%Core.Less.c86, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.1]
// CHECK:STDOUT:   %Less.ref.loc27: %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.1 = name_ref Less, %.loc27_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc27: <bound method> = bound_method %a.ref.loc27, %Less.ref.loc27
// CHECK:STDOUT:   %impl.elem0.loc27_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_5.3: <bound method> = bound_method %b.ref.loc27, %impl.elem0.loc27_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc27_5: init %Cpp.long = call %bound_method.loc27_5.3(%b.ref.loc27)
// CHECK:STDOUT:   %.loc27_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc27_5
// CHECK:STDOUT:   %.loc27_5.5: %Cpp.long = converted %b.ref.loc27, %.loc27_5.4
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc27: <specific function> = specific_function %Less.ref.loc27, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.2]
// CHECK:STDOUT:   %bound_method.loc27_5.4: <bound method> = bound_method %a.ref.loc27, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc27
// CHECK:STDOUT:   %impl.elem0.loc27_7: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_7: <bound method> = bound_method %b.ref.loc27, %impl.elem0.loc27_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc27_7: init %Cpp.long = call %bound_method.loc27_7(%b.ref.loc27)
// CHECK:STDOUT:   %.loc27_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc27_7
// CHECK:STDOUT:   %.loc27_7.2: %Cpp.long = converted %b.ref.loc27, %.loc27_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc27: init bool = call %bound_method.loc27_5.4(%a.ref.loc27, %.loc27_7.2)
// CHECK:STDOUT:   %a.ref.loc28: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc28: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc28: %.723 = impl_witness_access constants.%OrderedWith.impl_witness.576, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.2]
// CHECK:STDOUT:   %bound_method.loc28_5.1: <bound method> = bound_method %a.ref.loc28, %impl.elem3.loc28
// CHECK:STDOUT:   %ImplicitAs.facet.loc28_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc28_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc28_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc28_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc28_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc28_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem3.loc28, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.1]
// CHECK:STDOUT:   %bound_method.loc28_5.2: <bound method> = bound_method %a.ref.loc28, %specific_fn.loc28
// CHECK:STDOUT:   %.loc28_5.3: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.1 = specific_constant imports.%Core.GreaterOrEquivalent.76e, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc28: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.1 = name_ref GreaterOrEquivalent, %.loc28_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc28: <bound method> = bound_method %a.ref.loc28, %GreaterOrEquivalent.ref.loc28
// CHECK:STDOUT:   %impl.elem0.loc28_5: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_5.3: <bound method> = bound_method %b.ref.loc28, %impl.elem0.loc28_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc28_5: init %Cpp.long = call %bound_method.loc28_5.3(%b.ref.loc28)
// CHECK:STDOUT:   %.loc28_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc28_5
// CHECK:STDOUT:   %.loc28_5.5: %Cpp.long = converted %b.ref.loc28, %.loc28_5.4
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc28: <specific function> = specific_function %GreaterOrEquivalent.ref.loc28, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.2]
// CHECK:STDOUT:   %bound_method.loc28_5.4: <bound method> = bound_method %a.ref.loc28, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc28
// CHECK:STDOUT:   %impl.elem0.loc28_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_8: <bound method> = bound_method %b.ref.loc28, %impl.elem0.loc28_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc28_8: init %Cpp.long = call %bound_method.loc28_8(%b.ref.loc28)
// CHECK:STDOUT:   %.loc28_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc28_8
// CHECK:STDOUT:   %.loc28_8.2: %Cpp.long = converted %b.ref.loc28, %.loc28_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc28: init bool = call %bound_method.loc28_5.4(%a.ref.loc28, %.loc28_8.2)
// CHECK:STDOUT:   %a.ref.loc29: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc29: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc29: %.d50 = impl_witness_access constants.%OrderedWith.impl_witness.576, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.2]
// CHECK:STDOUT:   %bound_method.loc29_5.1: <bound method> = bound_method %a.ref.loc29, %impl.elem1.loc29
// CHECK:STDOUT:   %ImplicitAs.facet.loc29_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc29_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc29_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc29_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc29_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc29_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem1.loc29, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9548d5.1]
// CHECK:STDOUT:   %bound_method.loc29_5.2: <bound method> = bound_method %a.ref.loc29, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29_5.3: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.1 = specific_constant imports.%Core.LessOrEquivalent.4a6, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc29: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.1 = name_ref LessOrEquivalent, %.loc29_5.3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc29: <bound method> = bound_method %a.ref.loc29, %LessOrEquivalent.ref.loc29
// CHECK:STDOUT:   %impl.elem0.loc29_5: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_5.3: <bound method> = bound_method %b.ref.loc29, %impl.elem0.loc29_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29_5: init %Cpp.long = call %bound_method.loc29_5.3(%b.ref.loc29)
// CHECK:STDOUT:   %.loc29_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29_5
// CHECK:STDOUT:   %.loc29_5.5: %Cpp.long = converted %b.ref.loc29, %.loc29_5.4
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc29: <specific function> = specific_function %LessOrEquivalent.ref.loc29, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9548d5.2]
// CHECK:STDOUT:   %bound_method.loc29_5.4: <bound method> = bound_method %a.ref.loc29, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc29
// CHECK:STDOUT:   %impl.elem0.loc29_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_8: <bound method> = bound_method %b.ref.loc29, %impl.elem0.loc29_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29_8: init %Cpp.long = call %bound_method.loc29_8(%b.ref.loc29)
// CHECK:STDOUT:   %.loc29_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29_8
// CHECK:STDOUT:   %.loc29_8.2: %Cpp.long = converted %b.ref.loc29, %.loc29_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc29: init bool = call %bound_method.loc29_5.4(%a.ref.loc29, %.loc29_8.2)
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.494: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %EqWith.Equal.type.eeb: type = fn_type @EqWith.Equal, @EqWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %EqWith.NotEqual.type.d83: type = fn_type @EqWith.NotEqual, @EqWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.57d: %ImplicitAs.type.819 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.71d714.1: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.1, @T.binding.as_type.as.EqWith.impl.8ff(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.2eedb3.1: %T.binding.as_type.as.EqWith.impl.NotEqual.type.71d714.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.71d714.2: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.2, @T.binding.as_type.as.EqWith.impl.8ff(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.2eedb3.2: %T.binding.as_type.as.EqWith.impl.NotEqual.type.71d714.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.fd2c2f.1: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.1, @T.binding.as_type.as.EqWith.impl.8ff(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.c3a936.1: %T.binding.as_type.as.EqWith.impl.Equal.type.fd2c2f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.fd2c2f.2: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.2, @T.binding.as_type.as.EqWith.impl.8ff(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.c3a936.2: %T.binding.as_type.as.EqWith.impl.Equal.type.fd2c2f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.949: <witness> = impl_witness imports.%EqWith.impl_witness_table.c6d, @T.binding.as_type.as.EqWith.impl.8ff(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.1: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.2, @T.binding.as_type.as.EqWith.impl.8ff(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.b65948.1: %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.1: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.2, @T.binding.as_type.as.EqWith.impl.8ff(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.37228f.1: %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.2: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.1, @T.binding.as_type.as.EqWith.impl.8ff(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.b65948.2: %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.2: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.1, @T.binding.as_type.as.EqWith.impl.8ff(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.37228f.2: %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.57a: %EqWith.type.494 = facet_value %i32, (%EqWith.impl_witness.949) [concrete]
// CHECK:STDOUT:   %.c41: type = fn_type_with_self_type %EqWith.Equal.type.eeb, %EqWith.facet.57a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.b65948.2, @T.binding.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.b65948.1, @T.binding.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.c45: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.4a6: type = fn_type_with_self_type %EqWith.NotEqual.type.d83, %EqWith.facet.57a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.NotEqual.37228f.2, @T.binding.as_type.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.NotEqual.37228f.1, @T.binding.as_type.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.a39: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.Less.type.3c2: type = fn_type @OrderedWith.Less, @OrderedWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %OrderedWith.LessOrEquivalent.type.d58: type = fn_type @OrderedWith.LessOrEquivalent, @OrderedWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %OrderedWith.Greater.type.a51: type = fn_type @OrderedWith.Greater, @OrderedWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %OrderedWith.GreaterOrEquivalent.type.3df: type = fn_type @OrderedWith.GreaterOrEquivalent, @OrderedWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.058043.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.binding.as_type.as.OrderedWith.impl.891(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9f68cd.1: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.058043.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.058043.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.binding.as_type.as.OrderedWith.impl.891(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9f68cd.2: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.058043.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.5284c3.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.1, @T.binding.as_type.as.OrderedWith.impl.891(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.1fe718.1: %T.binding.as_type.as.OrderedWith.impl.Greater.type.5284c3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.5284c3.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.2, @T.binding.as_type.as.OrderedWith.impl.891(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.1fe718.2: %T.binding.as_type.as.OrderedWith.impl.Greater.type.5284c3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e16008.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.binding.as_type.as.OrderedWith.impl.891(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4a32de.1: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e16008.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e16008.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.binding.as_type.as.OrderedWith.impl.891(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4a32de.2: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e16008.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.7f8c78.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.1, @T.binding.as_type.as.OrderedWith.impl.891(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.a71c43.1: %T.binding.as_type.as.OrderedWith.impl.Less.type.7f8c78.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.7f8c78.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.2, @T.binding.as_type.as.OrderedWith.impl.891(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.a71c43.2: %T.binding.as_type.as.OrderedWith.impl.Less.type.7f8c78.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.f9b: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.752, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.2, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.f24104.1: %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.1: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.2, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.1: %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.1: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.1, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.f24104.2: %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.2: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.1, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.2: %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.2: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.fa3: %OrderedWith.type.a39 = facet_value %i32, (%OrderedWith.impl_witness.f9b) [concrete]
// CHECK:STDOUT:   %.2cb: type = fn_type_with_self_type %OrderedWith.Greater.type.a51, %OrderedWith.facet.fa3 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.2, @T.binding.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.1, @T.binding.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.199: type = fn_type_with_self_type %OrderedWith.Less.type.3c2, %OrderedWith.facet.fa3 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.f24104.2, @T.binding.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.f24104.1, @T.binding.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.fd0: type = fn_type_with_self_type %OrderedWith.GreaterOrEquivalent.type.3df, %OrderedWith.facet.fa3 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.2, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.1, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.184: type = fn_type_with_self_type %OrderedWith.LessOrEquivalent.type.d58, %OrderedWith.facet.fa3 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a789d4.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.2, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a789d4.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.1, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.801: <witness> = impl_witness imports.%EqWith.impl_witness_table.c6d, @T.binding.as_type.as.EqWith.impl.8ff(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.85dd3c.1: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.2, @T.binding.as_type.as.EqWith.impl.8ff(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.326379.1: %T.binding.as_type.as.EqWith.impl.Equal.type.85dd3c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.bb8d74.1: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.2, @T.binding.as_type.as.EqWith.impl.8ff(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.1: %T.binding.as_type.as.EqWith.impl.NotEqual.type.bb8d74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.type.85dd3c.2: type = fn_type @T.binding.as_type.as.EqWith.impl.Equal.1, @T.binding.as_type.as.EqWith.impl.8ff(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.326379.2: %T.binding.as_type.as.EqWith.impl.Equal.type.85dd3c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.type.bb8d74.2: type = fn_type @T.binding.as_type.as.EqWith.impl.NotEqual.1, @T.binding.as_type.as.EqWith.impl.8ff(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.2: %T.binding.as_type.as.EqWith.impl.NotEqual.type.bb8d74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.fa2: %EqWith.type.494 = facet_value Core.IntLiteral, (%EqWith.impl_witness.801) [concrete]
// CHECK:STDOUT:   %.a9d: type = fn_type_with_self_type %EqWith.Equal.type.eeb, %EqWith.facet.fa2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.62ea05.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.326379.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.326379.2, @T.binding.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.8a46b0.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.62ea05.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.326379.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.326379.1, @T.binding.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.8a46b0.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.2 [concrete]
// CHECK:STDOUT:   %.a87: type = fn_type_with_self_type %EqWith.NotEqual.type.d83, %EqWith.facet.fa2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.29032e.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.ceecb6.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.2, @T.binding.as_type.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.082399.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.ceecb6.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.29032e.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.ceecb6.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.1, @T.binding.as_type.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.082399.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.ceecb6.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.252: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.752, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.285b47.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.2, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.1: %T.binding.as_type.as.OrderedWith.impl.Less.type.285b47.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8a3347.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.1: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8a3347.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.4ddc57.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.2, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.1: %T.binding.as_type.as.OrderedWith.impl.Greater.type.4ddc57.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9cc1f0.1: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.1: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9cc1f0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.type.285b47.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Less.1, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.2: %T.binding.as_type.as.OrderedWith.impl.Less.type.285b47.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8a3347.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.2: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8a3347.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.type.4ddc57.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.Greater.1, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.2: %T.binding.as_type.as.OrderedWith.impl.Greater.type.4ddc57.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9cc1f0.2: type = fn_type @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.binding.as_type.as.OrderedWith.impl.891(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.2: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9cc1f0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.2bc: %OrderedWith.type.a39 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.252) [concrete]
// CHECK:STDOUT:   %.5df: type = fn_type_with_self_type %OrderedWith.Greater.type.a51, %OrderedWith.facet.2bc [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.8ab0dc.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.2, @T.binding.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.ea09f6.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.8ab0dc.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.1, @T.binding.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.ea09f6.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.2 [concrete]
// CHECK:STDOUT:   %.0c4: type = fn_type_with_self_type %OrderedWith.Less.type.3c2, %OrderedWith.facet.2bc [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.a0ee36.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.2, @T.binding.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.d20c63.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.a0ee36.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.1, @T.binding.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.d20c63.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.2 [concrete]
// CHECK:STDOUT:   %.9ee: type = fn_type_with_self_type %OrderedWith.GreaterOrEquivalent.type.3df, %OrderedWith.facet.2bc [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.378e29.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.2, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.212360.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.378e29.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.1, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.212360.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.2 [concrete]
// CHECK:STDOUT:   %.36d: type = fn_type_with_self_type %OrderedWith.LessOrEquivalent.type.d58, %OrderedWith.facet.2bc [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.5ef5e5.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.925a78.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.2, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.e9dda0.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.925a78.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.5ef5e5.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.925a78.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.1, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.e9dda0.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.925a78.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.534: @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.Equal.type.2 (%T.binding.as_type.as.EqWith.impl.Equal.type.fd2c2f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.Equal.2 (constants.%T.binding.as_type.as.EqWith.impl.Equal.c3a936.1)]
// CHECK:STDOUT:   %Core.import_ref.5d5: @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.NotEqual.type.2 (%T.binding.as_type.as.EqWith.impl.NotEqual.type.71d714.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.binding.as_type.as.EqWith.impl.NotEqual.2eedb3.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.c6d = impl_witness_table (%Core.import_ref.534, %Core.import_ref.5d5), @T.binding.as_type.as.EqWith.impl.8ff [concrete]
// CHECK:STDOUT:   %Core.NotEqual.a7f: @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.NotEqual.type.1 (%T.binding.as_type.as.EqWith.impl.NotEqual.type.71d714.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.binding.as_type.as.EqWith.impl.NotEqual.2eedb3.2)]
// CHECK:STDOUT:   %Core.Equal.784: @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.Equal.type.1 (%T.binding.as_type.as.EqWith.impl.Equal.type.fd2c2f.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.Equal.1 (constants.%T.binding.as_type.as.EqWith.impl.Equal.c3a936.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9e8: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Less.type.2 (%T.binding.as_type.as.OrderedWith.impl.Less.type.7f8c78.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Less.2 (constants.%T.binding.as_type.as.OrderedWith.impl.Less.a71c43.1)]
// CHECK:STDOUT:   %Core.import_ref.973: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e16008.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4a32de.1)]
// CHECK:STDOUT:   %Core.import_ref.155: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Greater.type.2 (%T.binding.as_type.as.OrderedWith.impl.Greater.type.5284c3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.binding.as_type.as.OrderedWith.impl.Greater.1fe718.1)]
// CHECK:STDOUT:   %Core.import_ref.f9eb: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.058043.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9f68cd.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.752 = impl_witness_table (%Core.import_ref.9e8, %Core.import_ref.973, %Core.import_ref.155, %Core.import_ref.f9eb), @T.binding.as_type.as.OrderedWith.impl.891 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.c46: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.058043.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9f68cd.2)]
// CHECK:STDOUT:   %Core.Greater.696: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Greater.type.1 (%T.binding.as_type.as.OrderedWith.impl.Greater.type.5284c3.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.binding.as_type.as.OrderedWith.impl.Greater.1fe718.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.947: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e16008.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4a32de.2)]
// CHECK:STDOUT:   %Core.Less.b61: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Less.type.1 (%T.binding.as_type.as.OrderedWith.impl.Less.type.7f8c78.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Less.1 (constants.%T.binding.as_type.as.OrderedWith.impl.Less.a71c43.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc9_21.2
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.c41 = impl_witness_access constants.%EqWith.impl_witness.949, element0 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.binding.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.1 = specific_constant imports.%Core.Equal.784, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @T.binding.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long = call %bound_method.loc10_3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %Cpp.long = converted %b.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %a.ref.loc10)
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.4a6 = impl_witness_access constants.%EqWith.impl_witness.949, element1 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.binding.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.1 = specific_constant imports.%Core.NotEqual.a7f, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @T.binding.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11_3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %Cpp.long = converted %b.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %a.ref.loc11)
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.2cb = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element2 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.binding.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.1 = specific_constant imports.%Core.Greater.696, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @T.binding.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12_3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %Cpp.long = converted %b.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %a.ref.loc12)
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.199 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element0 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.binding.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.1 = specific_constant imports.%Core.Less.b61, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @T.binding.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long = call %bound_method.loc13_3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %Cpp.long = converted %b.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %a.ref.loc13)
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.fd0 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element3 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.1 = specific_constant imports.%Core.GreaterOrEquivalent.c46, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long = call %bound_method.loc14_3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %Cpp.long = converted %b.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %a.ref.loc14)
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.184 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element1 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a789d4.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.1 = specific_constant imports.%Core.LessOrEquivalent.947, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a789d4.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long = call %bound_method.loc15_3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %Cpp.long = converted %b.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %a.ref.loc15)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.a9d = impl_witness_access constants.%EqWith.impl_witness.801, element0 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.326379.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.bound.62ea05.1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.binding.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.8a46b0.1]
// CHECK:STDOUT:   %.loc17_5: %T.binding.as_type.as.EqWith.impl.Equal.type.85dd3c.1 = specific_constant imports.%Core.Equal.784, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.326379.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.binding.as_type.as.EqWith.impl.Equal.type.85dd3c.1 = name_ref Equal, %.loc17_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.326379.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.bound.62ea05.2]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @T.binding.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc17 [concrete = constants.%bound_method.8a46b0.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.a87 = impl_witness_access constants.%EqWith.impl_witness.801, element1 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bound.29032e.1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.binding.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.ceecb6.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.082399.1]
// CHECK:STDOUT:   %.loc18_5: %T.binding.as_type.as.EqWith.impl.NotEqual.type.bb8d74.1 = specific_constant imports.%Core.NotEqual.a7f, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.binding.as_type.as.EqWith.impl.NotEqual.type.bb8d74.1 = name_ref NotEqual, %.loc18_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bound.29032e.2]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @T.binding.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.ceecb6.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18 [concrete = constants.%bound_method.082399.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.5df = impl_witness_access constants.%OrderedWith.impl_witness.252, element2 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.bound.8ab0dc.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.binding.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.ea09f6.1]
// CHECK:STDOUT:   %.loc19_5: %T.binding.as_type.as.OrderedWith.impl.Greater.type.4ddc57.1 = specific_constant imports.%Core.Greater.696, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.binding.as_type.as.OrderedWith.impl.Greater.type.4ddc57.1 = name_ref Greater, %.loc19_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.bound.8ab0dc.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @T.binding.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19 [concrete = constants.%bound_method.ea09f6.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.0c4 = impl_witness_access constants.%OrderedWith.impl_witness.252, element0 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.bound.a0ee36.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.binding.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.d20c63.1]
// CHECK:STDOUT:   %.loc20_5: %T.binding.as_type.as.OrderedWith.impl.Less.type.285b47.1 = specific_constant imports.%Core.Less.b61, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.binding.as_type.as.OrderedWith.impl.Less.type.285b47.1 = name_ref Less, %.loc20_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.bound.a0ee36.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @T.binding.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc20 [concrete = constants.%bound_method.d20c63.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.9ee = impl_witness_access constants.%OrderedWith.impl_witness.252, element3 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.378e29.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.212360.1]
// CHECK:STDOUT:   %.loc21_5: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9cc1f0.1 = specific_constant imports.%Core.GreaterOrEquivalent.c46, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9cc1f0.1 = name_ref GreaterOrEquivalent, %.loc21_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.378e29.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21 [concrete = constants.%bound_method.212360.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.36d = impl_witness_access constants.%OrderedWith.impl_witness.252, element1 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.5ef5e5.1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.925a78.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.e9dda0.1]
// CHECK:STDOUT:   %.loc22_5: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8a3347.1 = specific_constant imports.%Core.LessOrEquivalent.947, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8a3347.1 = name_ref LessOrEquivalent, %.loc22_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.5ef5e5.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.925a78.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22 [concrete = constants.%bound_method.e9dda0.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long = call %bound_method.loc22_3(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_3.2: %Cpp.long = converted %int_1.loc22, %.loc22_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc22: init bool = call %bound_method.loc22_5.3(%.loc22_3.2, %a.ref.loc22)
// CHECK:STDOUT:   %b.ref.loc24: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc24: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc24_5: %.c41 = impl_witness_access constants.%EqWith.impl_witness.949, element0 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.2]
// CHECK:STDOUT:   %bound_method.loc24_5.1: <bound method> = bound_method %b.ref.loc24, %impl.elem0.loc24_5
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_5, @T.binding.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.1]
// CHECK:STDOUT:   %bound_method.loc24_5.2: <bound method> = bound_method %b.ref.loc24, %specific_fn.loc24
// CHECK:STDOUT:   %.loc24_5: %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.1 = specific_constant imports.%Core.Equal.784, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.1]
// CHECK:STDOUT:   %Equal.ref.loc24: %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.1 = name_ref Equal, %.loc24_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.loc24: <bound method> = bound_method %b.ref.loc24, %Equal.ref.loc24
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc24: <specific function> = specific_function %Equal.ref.loc24, @T.binding.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.2]
// CHECK:STDOUT:   %bound_method.loc24_5.3: <bound method> = bound_method %b.ref.loc24, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc24
// CHECK:STDOUT:   %impl.elem0.loc24_3: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc24_3: <bound method> = bound_method %b.ref.loc24, %impl.elem0.loc24_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc24: init %Cpp.long = call %bound_method.loc24_3(%b.ref.loc24)
// CHECK:STDOUT:   %.loc24_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc24
// CHECK:STDOUT:   %.loc24_3.2: %Cpp.long = converted %b.ref.loc24, %.loc24_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.call.loc24: init bool = call %bound_method.loc24_5.3(%.loc24_3.2, %a.ref.loc24)
// CHECK:STDOUT:   %b.ref.loc25: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc25: %.4a6 = impl_witness_access constants.%EqWith.impl_witness.949, element1 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.2]
// CHECK:STDOUT:   %bound_method.loc25_5.1: <bound method> = bound_method %b.ref.loc25, %impl.elem1.loc25
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem1.loc25, @T.binding.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.1]
// CHECK:STDOUT:   %bound_method.loc25_5.2: <bound method> = bound_method %b.ref.loc25, %specific_fn.loc25
// CHECK:STDOUT:   %.loc25_5: %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.1 = specific_constant imports.%Core.NotEqual.a7f, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.1]
// CHECK:STDOUT:   %NotEqual.ref.loc25: %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.1 = name_ref NotEqual, %.loc25_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.loc25: <bound method> = bound_method %b.ref.loc25, %NotEqual.ref.loc25
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc25: <specific function> = specific_function %NotEqual.ref.loc25, @T.binding.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.2]
// CHECK:STDOUT:   %bound_method.loc25_5.3: <bound method> = bound_method %b.ref.loc25, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc25
// CHECK:STDOUT:   %impl.elem0.loc25: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_3: <bound method> = bound_method %b.ref.loc25, %impl.elem0.loc25
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc25: init %Cpp.long = call %bound_method.loc25_3(%b.ref.loc25)
// CHECK:STDOUT:   %.loc25_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc25
// CHECK:STDOUT:   %.loc25_3.2: %Cpp.long = converted %b.ref.loc25, %.loc25_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.call.loc25: init bool = call %bound_method.loc25_5.3(%.loc25_3.2, %a.ref.loc25)
// CHECK:STDOUT:   %b.ref.loc26: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc26: %.2cb = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element2 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.2]
// CHECK:STDOUT:   %bound_method.loc26_5.1: <bound method> = bound_method %b.ref.loc26, %impl.elem2.loc26
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem2.loc26, @T.binding.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.1]
// CHECK:STDOUT:   %bound_method.loc26_5.2: <bound method> = bound_method %b.ref.loc26, %specific_fn.loc26
// CHECK:STDOUT:   %.loc26_5: %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.1 = specific_constant imports.%Core.Greater.696, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.1]
// CHECK:STDOUT:   %Greater.ref.loc26: %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.1 = name_ref Greater, %.loc26_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.loc26: <bound method> = bound_method %b.ref.loc26, %Greater.ref.loc26
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc26: <specific function> = specific_function %Greater.ref.loc26, @T.binding.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.2]
// CHECK:STDOUT:   %bound_method.loc26_5.3: <bound method> = bound_method %b.ref.loc26, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc26
// CHECK:STDOUT:   %impl.elem0.loc26: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_3: <bound method> = bound_method %b.ref.loc26, %impl.elem0.loc26
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc26: init %Cpp.long = call %bound_method.loc26_3(%b.ref.loc26)
// CHECK:STDOUT:   %.loc26_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc26
// CHECK:STDOUT:   %.loc26_3.2: %Cpp.long = converted %b.ref.loc26, %.loc26_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.call.loc26: init bool = call %bound_method.loc26_5.3(%.loc26_3.2, %a.ref.loc26)
// CHECK:STDOUT:   %b.ref.loc27: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc27: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc27_5: %.199 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element0 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.2]
// CHECK:STDOUT:   %bound_method.loc27_5.1: <bound method> = bound_method %b.ref.loc27, %impl.elem0.loc27_5
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27_5, @T.binding.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.1]
// CHECK:STDOUT:   %bound_method.loc27_5.2: <bound method> = bound_method %b.ref.loc27, %specific_fn.loc27
// CHECK:STDOUT:   %.loc27_5: %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.1 = specific_constant imports.%Core.Less.b61, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.1]
// CHECK:STDOUT:   %Less.ref.loc27: %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.1 = name_ref Less, %.loc27_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.loc27: <bound method> = bound_method %b.ref.loc27, %Less.ref.loc27
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc27: <specific function> = specific_function %Less.ref.loc27, @T.binding.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.2]
// CHECK:STDOUT:   %bound_method.loc27_5.3: <bound method> = bound_method %b.ref.loc27, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc27
// CHECK:STDOUT:   %impl.elem0.loc27_3: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_3: <bound method> = bound_method %b.ref.loc27, %impl.elem0.loc27_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc27: init %Cpp.long = call %bound_method.loc27_3(%b.ref.loc27)
// CHECK:STDOUT:   %.loc27_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc27
// CHECK:STDOUT:   %.loc27_3.2: %Cpp.long = converted %b.ref.loc27, %.loc27_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.call.loc27: init bool = call %bound_method.loc27_5.3(%.loc27_3.2, %a.ref.loc27)
// CHECK:STDOUT:   %b.ref.loc28: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc28: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc28: %.fd0 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element3 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.2]
// CHECK:STDOUT:   %bound_method.loc28_5.1: <bound method> = bound_method %b.ref.loc28, %impl.elem3.loc28
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem3.loc28, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.1]
// CHECK:STDOUT:   %bound_method.loc28_5.2: <bound method> = bound_method %b.ref.loc28, %specific_fn.loc28
// CHECK:STDOUT:   %.loc28_5: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.1 = specific_constant imports.%Core.GreaterOrEquivalent.c46, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc28: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.1 = name_ref GreaterOrEquivalent, %.loc28_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc28: <bound method> = bound_method %b.ref.loc28, %GreaterOrEquivalent.ref.loc28
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc28: <specific function> = specific_function %GreaterOrEquivalent.ref.loc28, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.2]
// CHECK:STDOUT:   %bound_method.loc28_5.3: <bound method> = bound_method %b.ref.loc28, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc28
// CHECK:STDOUT:   %impl.elem0.loc28: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_3: <bound method> = bound_method %b.ref.loc28, %impl.elem0.loc28
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc28: init %Cpp.long = call %bound_method.loc28_3(%b.ref.loc28)
// CHECK:STDOUT:   %.loc28_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc28
// CHECK:STDOUT:   %.loc28_3.2: %Cpp.long = converted %b.ref.loc28, %.loc28_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc28: init bool = call %bound_method.loc28_5.3(%.loc28_3.2, %a.ref.loc28)
// CHECK:STDOUT:   %b.ref.loc29: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc29: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc29: %.184 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element1 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.2]
// CHECK:STDOUT:   %bound_method.loc29_5.1: <bound method> = bound_method %b.ref.loc29, %impl.elem1.loc29
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem1.loc29, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a789d4.1]
// CHECK:STDOUT:   %bound_method.loc29_5.2: <bound method> = bound_method %b.ref.loc29, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29_5: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.1 = specific_constant imports.%Core.LessOrEquivalent.947, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc29: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.1 = name_ref LessOrEquivalent, %.loc29_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc29: <bound method> = bound_method %b.ref.loc29, %LessOrEquivalent.ref.loc29
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc29: <specific function> = specific_function %LessOrEquivalent.ref.loc29, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a789d4.2]
// CHECK:STDOUT:   %bound_method.loc29_5.3: <bound method> = bound_method %b.ref.loc29, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc29
// CHECK:STDOUT:   %impl.elem0.loc29: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_3: <bound method> = bound_method %b.ref.loc29, %impl.elem0.loc29
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29: init %Cpp.long = call %bound_method.loc29_3(%b.ref.loc29)
// CHECK:STDOUT:   %.loc29_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29
// CHECK:STDOUT:   %.loc29_3.2: %Cpp.long = converted %b.ref.loc29, %.loc29_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc29: init bool = call %bound_method.loc29_5.3(%.loc29_3.2, %a.ref.loc29)
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.Op.type: type = fn_type @Negate.Op [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %Cpp.long, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %.494: type = fn_type_with_self_type %Negate.Op.type, %Negate.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op.type: type = fn_type @Cpp.long.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op: %Cpp.long.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.9c2: type = facet_type <@AddWith, @AddWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %AddWith.Op.type.9a6: type = fn_type @AddWith.Op, @AddWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.e9f: <witness> = impl_witness imports.%AddWith.impl_witness_table.2f5 [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.9c2 = facet_value %Cpp.long, (%AddWith.impl_witness.e9f) [concrete]
// CHECK:STDOUT:   %.a81: type = fn_type_with_self_type %AddWith.Op.type.9a6, %AddWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.type.094: type = fn_type @Cpp.long.as.AddWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.be0: %Cpp.long.as.AddWith.impl.Op.type.094 = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.type.e8e: type = facet_type <@SubWith, @SubWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %SubWith.Op.type.533: type = fn_type @SubWith.Op, @SubWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness.8a6: <witness> = impl_witness imports.%SubWith.impl_witness_table.e2e [concrete]
// CHECK:STDOUT:   %SubWith.facet: %SubWith.type.e8e = facet_value %Cpp.long, (%SubWith.impl_witness.8a6) [concrete]
// CHECK:STDOUT:   %.70d: type = fn_type_with_self_type %SubWith.Op.type.533, %SubWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.type.882: type = fn_type @Cpp.long.as.SubWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.37d: %Cpp.long.as.SubWith.impl.Op.type.882 = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.type.a32: type = facet_type <@MulWith, @MulWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %MulWith.Op.type.44b: type = fn_type @MulWith.Op, @MulWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness.87c: <witness> = impl_witness imports.%MulWith.impl_witness_table.20e [concrete]
// CHECK:STDOUT:   %MulWith.facet: %MulWith.type.a32 = facet_value %Cpp.long, (%MulWith.impl_witness.87c) [concrete]
// CHECK:STDOUT:   %.0a7: type = fn_type_with_self_type %MulWith.Op.type.44b, %MulWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.type.31a: type = fn_type @Cpp.long.as.MulWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.4e5: %Cpp.long.as.MulWith.impl.Op.type.31a = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.type.0c0: type = facet_type <@DivWith, @DivWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %DivWith.Op.type.494: type = fn_type @DivWith.Op, @DivWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness.4ab: <witness> = impl_witness imports.%DivWith.impl_witness_table.1ae [concrete]
// CHECK:STDOUT:   %DivWith.facet: %DivWith.type.0c0 = facet_value %Cpp.long, (%DivWith.impl_witness.4ab) [concrete]
// CHECK:STDOUT:   %.22b: type = fn_type_with_self_type %DivWith.Op.type.494, %DivWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.type.5e7: type = fn_type @Cpp.long.as.DivWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.33f: %Cpp.long.as.DivWith.impl.Op.type.5e7 = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.type.2fc: type = facet_type <@ModWith, @ModWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ModWith.Op.type.c13: type = fn_type @ModWith.Op, @ModWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness.c73: <witness> = impl_witness imports.%ModWith.impl_witness_table.ed4 [concrete]
// CHECK:STDOUT:   %ModWith.facet: %ModWith.type.2fc = facet_value %Cpp.long, (%ModWith.impl_witness.c73) [concrete]
// CHECK:STDOUT:   %.300: type = fn_type_with_self_type %ModWith.Op.type.c13, %ModWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.type.aa5: type = fn_type @Cpp.long.as.ModWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.06e: %Cpp.long.as.ModWith.impl.Op.type.aa5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.412: %Cpp.long.as.Negate.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.cb912f.1, %Core.import_ref.412), @Cpp.long.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.4 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.437: %Cpp.long.as.AddWith.impl.Op.type.094 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.AddWith.impl.Op.be0]
// CHECK:STDOUT:   %AddWith.impl_witness_table.2f5 = impl_witness_table (%Core.import_ref.cb912f.4, %Core.import_ref.437), @Cpp.long.as.AddWith.impl.8b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.7 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.51f: %Cpp.long.as.SubWith.impl.Op.type.882 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.SubWith.impl.Op.37d]
// CHECK:STDOUT:   %SubWith.impl_witness_table.e2e = impl_witness_table (%Core.import_ref.cb912f.7, %Core.import_ref.51f), @Cpp.long.as.SubWith.impl.214 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.10 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ec4: %Cpp.long.as.MulWith.impl.Op.type.31a = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.MulWith.impl.Op.4e5]
// CHECK:STDOUT:   %MulWith.impl_witness_table.20e = impl_witness_table (%Core.import_ref.cb912f.10, %Core.import_ref.ec4), @Cpp.long.as.MulWith.impl.ee0 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.13 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d85: %Cpp.long.as.DivWith.impl.Op.type.5e7 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.DivWith.impl.Op.33f]
// CHECK:STDOUT:   %DivWith.impl_witness_table.1ae = impl_witness_table (%Core.import_ref.cb912f.13, %Core.import_ref.d85), @Cpp.long.as.DivWith.impl.cb9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.16 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.394: %Cpp.long.as.ModWith.impl.Op.type.aa5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ModWith.impl.Op.06e]
// CHECK:STDOUT:   %ModWith.impl_witness_table.ed4 = impl_witness_table (%Core.import_ref.cb912f.16, %Core.import_ref.394), @Cpp.long.as.ModWith.impl.2d7 [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %y: %Cpp.long = value_binding y, %.loc9_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc11: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc11: %.494 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Cpp.long.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %x.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op.call: init %Cpp.long = call %bound_method.loc11(%x.ref.loc11)
// CHECK:STDOUT:   %.loc11_13: type = splice_block %long.ref.loc11 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc11: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_21.1: %Cpp.long = value_of_initializer %Cpp.long.as.Negate.impl.Op.call
// CHECK:STDOUT:   %.loc11_21.2: %Cpp.long = converted %Cpp.long.as.Negate.impl.Op.call, %.loc11_21.1
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc11_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc12: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc12: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc12: %.a81 = impl_witness_access constants.%AddWith.impl_witness.e9f, element1 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.be0]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %x.ref.loc12, %impl.elem1.loc12
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.call: init %Cpp.long = call %bound_method.loc12(%x.ref.loc12, %y.ref.loc12)
// CHECK:STDOUT:   %.loc12_13: type = splice_block %long.ref.loc12 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc12: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_23.1: %Cpp.long = value_of_initializer %Cpp.long.as.AddWith.impl.Op.call
// CHECK:STDOUT:   %.loc12_23.2: %Cpp.long = converted %Cpp.long.as.AddWith.impl.Op.call, %.loc12_23.1
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc12_23.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.68c = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc13: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc13: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc13: %.70d = impl_witness_access constants.%SubWith.impl_witness.8a6, element1 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.37d]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %x.ref.loc13, %impl.elem1.loc13
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.call: init %Cpp.long = call %bound_method.loc13(%x.ref.loc13, %y.ref.loc13)
// CHECK:STDOUT:   %.loc13_13: type = splice_block %long.ref.loc13 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc13: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_23.1: %Cpp.long = value_of_initializer %Cpp.long.as.SubWith.impl.Op.call
// CHECK:STDOUT:   %.loc13_23.2: %Cpp.long = converted %Cpp.long.as.SubWith.impl.Op.call, %.loc13_23.1
// CHECK:STDOUT:   %c: %Cpp.long = value_binding c, %.loc13_23.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.68c = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc14: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc14: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc14: %.0a7 = impl_witness_access constants.%MulWith.impl_witness.87c, element1 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.4e5]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %x.ref.loc14, %impl.elem1.loc14
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.call: init %Cpp.long = call %bound_method.loc14(%x.ref.loc14, %y.ref.loc14)
// CHECK:STDOUT:   %.loc14_13: type = splice_block %long.ref.loc14 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc14: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_23.1: %Cpp.long = value_of_initializer %Cpp.long.as.MulWith.impl.Op.call
// CHECK:STDOUT:   %.loc14_23.2: %Cpp.long = converted %Cpp.long.as.MulWith.impl.Op.call, %.loc14_23.1
// CHECK:STDOUT:   %d: %Cpp.long = value_binding d, %.loc14_23.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.68c = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc15: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc15: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc15: %.22b = impl_witness_access constants.%DivWith.impl_witness.4ab, element1 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.33f]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %x.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.call: init %Cpp.long = call %bound_method.loc15(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:   %.loc15_13: type = splice_block %long.ref.loc15 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc15: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_23.1: %Cpp.long = value_of_initializer %Cpp.long.as.DivWith.impl.Op.call
// CHECK:STDOUT:   %.loc15_23.2: %Cpp.long = converted %Cpp.long.as.DivWith.impl.Op.call, %.loc15_23.1
// CHECK:STDOUT:   %e: %Cpp.long = value_binding e, %.loc15_23.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.68c = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc16: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc16: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc16: %.300 = impl_witness_access constants.%ModWith.impl_witness.c73, element1 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.06e]
// CHECK:STDOUT:   %bound_method.loc16: <bound method> = bound_method %x.ref.loc16, %impl.elem1.loc16
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.call: init %Cpp.long = call %bound_method.loc16(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:   %.loc16_13: type = splice_block %long.ref.loc16 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc16: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_23.1: %Cpp.long = value_of_initializer %Cpp.long.as.ModWith.impl.Op.call
// CHECK:STDOUT:   %.loc16_23.2: %Cpp.long = converted %Cpp.long.as.ModWith.impl.Op.call, %.loc16_23.1
// CHECK:STDOUT:   %f: %Cpp.long = value_binding f, %.loc16_23.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- arithmetic_heterogeneous_long_left_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AssertSameType.type: type = fn_type @AssertSameType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AssertSameType: %AssertSameType.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %.863: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddWith.type.452: type = facet_type <@AddWith, @AddWith(%i32)> [concrete]
// CHECK:STDOUT:   %AddWith.Op.type.5eb: type = fn_type @AddWith.Op, @AddWith(%i32) [concrete]
// CHECK:STDOUT:   %T.57d: %ImplicitAs.type.819 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.type.b730f9.1: type = fn_type @Cpp.long.as.AddWith.impl.Op.1, @Cpp.long.as.AddWith.impl.c72(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.7ff34c.1: %Cpp.long.as.AddWith.impl.Op.type.b730f9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.type.b730f9.2: type = fn_type @Cpp.long.as.AddWith.impl.Op.2, @Cpp.long.as.AddWith.impl.c72(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.7ff34c.2: %Cpp.long.as.AddWith.impl.Op.type.b730f9.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.type.4c0: type = facet_type <@AddWith, @AddWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %AddWith.Op.type.0ee: type = fn_type @AddWith.Op, @AddWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.403: <witness> = impl_witness imports.%AddWith.impl_witness_table.34e, @Cpp.long.as.AddWith.impl.c72(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.type.3dbf51.1: type = fn_type @Cpp.long.as.AddWith.impl.Op.2, @Cpp.long.as.AddWith.impl.c72(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.7beae3.1: %Cpp.long.as.AddWith.impl.Op.type.3dbf51.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.type.3dbf51.2: type = fn_type @Cpp.long.as.AddWith.impl.Op.1, @Cpp.long.as.AddWith.impl.c72(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.7beae3.2: %Cpp.long.as.AddWith.impl.Op.type.3dbf51.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet.0df: %AddWith.type.452 = facet_value %Cpp.long, (%AddWith.impl_witness.403) [concrete]
// CHECK:STDOUT:   %.ef1: type = fn_type_with_self_type %AddWith.Op.type.5eb, %AddWith.facet.0df [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.specific_fn.88f499.1: <specific function> = specific_function %Cpp.long.as.AddWith.impl.Op.7beae3.2, @Cpp.long.as.AddWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.c45: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.specific_fn.88f499.2: <specific function> = specific_function %Cpp.long.as.AddWith.impl.Op.7beae3.1, @Cpp.long.as.AddWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %AssertSameType.specific_fn: <specific function> = specific_function %AssertSameType, @AssertSameType(%Cpp.long) [concrete]
// CHECK:STDOUT:   %SubWith.type.46f: type = facet_type <@SubWith, @SubWith(%i32)> [concrete]
// CHECK:STDOUT:   %SubWith.Op.type.5b4: type = fn_type @SubWith.Op, @SubWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.type.432302.1: type = fn_type @Cpp.long.as.SubWith.impl.Op.1, @Cpp.long.as.SubWith.impl.b84(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.42d8b9.1: %Cpp.long.as.SubWith.impl.Op.type.432302.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.type.432302.2: type = fn_type @Cpp.long.as.SubWith.impl.Op.2, @Cpp.long.as.SubWith.impl.b84(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.42d8b9.2: %Cpp.long.as.SubWith.impl.Op.type.432302.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.type.a89: type = facet_type <@SubWith, @SubWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %SubWith.Op.type.95d: type = fn_type @SubWith.Op, @SubWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness.bee: <witness> = impl_witness imports.%SubWith.impl_witness_table.694, @Cpp.long.as.SubWith.impl.b84(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.type.5b2a81.1: type = fn_type @Cpp.long.as.SubWith.impl.Op.2, @Cpp.long.as.SubWith.impl.b84(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.65b61c.1: %Cpp.long.as.SubWith.impl.Op.type.5b2a81.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.type.5b2a81.2: type = fn_type @Cpp.long.as.SubWith.impl.Op.1, @Cpp.long.as.SubWith.impl.b84(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.65b61c.2: %Cpp.long.as.SubWith.impl.Op.type.5b2a81.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.facet.9b8: %SubWith.type.46f = facet_value %Cpp.long, (%SubWith.impl_witness.bee) [concrete]
// CHECK:STDOUT:   %.494: type = fn_type_with_self_type %SubWith.Op.type.5b4, %SubWith.facet.9b8 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.specific_fn.8caeaa.1: <specific function> = specific_function %Cpp.long.as.SubWith.impl.Op.65b61c.2, @Cpp.long.as.SubWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.specific_fn.8caeaa.2: <specific function> = specific_function %Cpp.long.as.SubWith.impl.Op.65b61c.1, @Cpp.long.as.SubWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %MulWith.type.5e7: type = facet_type <@MulWith, @MulWith(%i32)> [concrete]
// CHECK:STDOUT:   %MulWith.Op.type.e00: type = fn_type @MulWith.Op, @MulWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.type.4f6939.1: type = fn_type @Cpp.long.as.MulWith.impl.Op.1, @Cpp.long.as.MulWith.impl.c90(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.b232c0.1: %Cpp.long.as.MulWith.impl.Op.type.4f6939.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.type.4f6939.2: type = fn_type @Cpp.long.as.MulWith.impl.Op.2, @Cpp.long.as.MulWith.impl.c90(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.b232c0.2: %Cpp.long.as.MulWith.impl.Op.type.4f6939.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.4ce: type = facet_type <@MulWith, @MulWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %MulWith.Op.type.ff6: type = fn_type @MulWith.Op, @MulWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness.ba5: <witness> = impl_witness imports.%MulWith.impl_witness_table.6e7, @Cpp.long.as.MulWith.impl.c90(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.type.ea5ff6.1: type = fn_type @Cpp.long.as.MulWith.impl.Op.2, @Cpp.long.as.MulWith.impl.c90(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.f095b7.1: %Cpp.long.as.MulWith.impl.Op.type.ea5ff6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.type.ea5ff6.2: type = fn_type @Cpp.long.as.MulWith.impl.Op.1, @Cpp.long.as.MulWith.impl.c90(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.f095b7.2: %Cpp.long.as.MulWith.impl.Op.type.ea5ff6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.facet.79b: %MulWith.type.5e7 = facet_value %Cpp.long, (%MulWith.impl_witness.ba5) [concrete]
// CHECK:STDOUT:   %.057: type = fn_type_with_self_type %MulWith.Op.type.e00, %MulWith.facet.79b [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.specific_fn.3be598.1: <specific function> = specific_function %Cpp.long.as.MulWith.impl.Op.f095b7.2, @Cpp.long.as.MulWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.specific_fn.3be598.2: <specific function> = specific_function %Cpp.long.as.MulWith.impl.Op.f095b7.1, @Cpp.long.as.MulWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %DivWith.type.469: type = facet_type <@DivWith, @DivWith(%i32)> [concrete]
// CHECK:STDOUT:   %DivWith.Op.type.c8e: type = fn_type @DivWith.Op, @DivWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.type.9ce9b6.1: type = fn_type @Cpp.long.as.DivWith.impl.Op.1, @Cpp.long.as.DivWith.impl.0e5(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.e35fb3.1: %Cpp.long.as.DivWith.impl.Op.type.9ce9b6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.type.9ce9b6.2: type = fn_type @Cpp.long.as.DivWith.impl.Op.2, @Cpp.long.as.DivWith.impl.0e5(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.e35fb3.2: %Cpp.long.as.DivWith.impl.Op.type.9ce9b6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.type.234: type = facet_type <@DivWith, @DivWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %DivWith.Op.type.c64: type = fn_type @DivWith.Op, @DivWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness.30d: <witness> = impl_witness imports.%DivWith.impl_witness_table.bcf, @Cpp.long.as.DivWith.impl.0e5(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.type.9156f2.1: type = fn_type @Cpp.long.as.DivWith.impl.Op.2, @Cpp.long.as.DivWith.impl.0e5(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.d06e88.1: %Cpp.long.as.DivWith.impl.Op.type.9156f2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.type.9156f2.2: type = fn_type @Cpp.long.as.DivWith.impl.Op.1, @Cpp.long.as.DivWith.impl.0e5(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.d06e88.2: %Cpp.long.as.DivWith.impl.Op.type.9156f2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.facet.4c5: %DivWith.type.469 = facet_value %Cpp.long, (%DivWith.impl_witness.30d) [concrete]
// CHECK:STDOUT:   %.eec: type = fn_type_with_self_type %DivWith.Op.type.c8e, %DivWith.facet.4c5 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.specific_fn.6489e2.1: <specific function> = specific_function %Cpp.long.as.DivWith.impl.Op.d06e88.2, @Cpp.long.as.DivWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.specific_fn.6489e2.2: <specific function> = specific_function %Cpp.long.as.DivWith.impl.Op.d06e88.1, @Cpp.long.as.DivWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ModWith.type.8ca: type = facet_type <@ModWith, @ModWith(%i32)> [concrete]
// CHECK:STDOUT:   %ModWith.Op.type.aee: type = fn_type @ModWith.Op, @ModWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.type.4bf837.1: type = fn_type @Cpp.long.as.ModWith.impl.Op.1, @Cpp.long.as.ModWith.impl.0a5(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.0707cb.1: %Cpp.long.as.ModWith.impl.Op.type.4bf837.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.type.4bf837.2: type = fn_type @Cpp.long.as.ModWith.impl.Op.2, @Cpp.long.as.ModWith.impl.0a5(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.0707cb.2: %Cpp.long.as.ModWith.impl.Op.type.4bf837.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.type.920: type = facet_type <@ModWith, @ModWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ModWith.Op.type.295: type = fn_type @ModWith.Op, @ModWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness.bd3: <witness> = impl_witness imports.%ModWith.impl_witness_table.9cf, @Cpp.long.as.ModWith.impl.0a5(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.type.f6cdb5.1: type = fn_type @Cpp.long.as.ModWith.impl.Op.2, @Cpp.long.as.ModWith.impl.0a5(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.6c7fa7.1: %Cpp.long.as.ModWith.impl.Op.type.f6cdb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.type.f6cdb5.2: type = fn_type @Cpp.long.as.ModWith.impl.Op.1, @Cpp.long.as.ModWith.impl.0a5(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.6c7fa7.2: %Cpp.long.as.ModWith.impl.Op.type.f6cdb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.facet.e8e: %ModWith.type.8ca = facet_value %Cpp.long, (%ModWith.impl_witness.bd3) [concrete]
// CHECK:STDOUT:   %.289: type = fn_type_with_self_type %ModWith.Op.type.aee, %ModWith.facet.e8e [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.specific_fn.127c5b.1: <specific function> = specific_function %Cpp.long.as.ModWith.impl.Op.6c7fa7.2, @Cpp.long.as.ModWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.specific_fn.127c5b.2: <specific function> = specific_function %Cpp.long.as.ModWith.impl.Op.6c7fa7.1, @Cpp.long.as.ModWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.e6c: <witness> = impl_witness imports.%AddWith.impl_witness_table.34e, @Cpp.long.as.AddWith.impl.c72(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.type.1ece0d.1: type = fn_type @Cpp.long.as.AddWith.impl.Op.2, @Cpp.long.as.AddWith.impl.c72(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.0f2d31.1: %Cpp.long.as.AddWith.impl.Op.type.1ece0d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.type.1ece0d.2: type = fn_type @Cpp.long.as.AddWith.impl.Op.1, @Cpp.long.as.AddWith.impl.c72(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.0f2d31.2: %Cpp.long.as.AddWith.impl.Op.type.1ece0d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet.993: %AddWith.type.4c0 = facet_value %Cpp.long, (%AddWith.impl_witness.e6c) [concrete]
// CHECK:STDOUT:   %.784: type = fn_type_with_self_type %AddWith.Op.type.0ee, %AddWith.facet.993 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.specific_fn.063c7e.1: <specific function> = specific_function %Cpp.long.as.AddWith.impl.Op.0f2d31.2, @Cpp.long.as.AddWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.specific_fn.063c7e.2: <specific function> = specific_function %Cpp.long.as.AddWith.impl.Op.0f2d31.1, @Cpp.long.as.AddWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness.dd6: <witness> = impl_witness imports.%SubWith.impl_witness_table.694, @Cpp.long.as.SubWith.impl.b84(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.type.ef5597.1: type = fn_type @Cpp.long.as.SubWith.impl.Op.2, @Cpp.long.as.SubWith.impl.b84(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.8a648c.1: %Cpp.long.as.SubWith.impl.Op.type.ef5597.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.type.ef5597.2: type = fn_type @Cpp.long.as.SubWith.impl.Op.1, @Cpp.long.as.SubWith.impl.b84(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.8a648c.2: %Cpp.long.as.SubWith.impl.Op.type.ef5597.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.facet.eb2: %SubWith.type.a89 = facet_value %Cpp.long, (%SubWith.impl_witness.dd6) [concrete]
// CHECK:STDOUT:   %.d5f: type = fn_type_with_self_type %SubWith.Op.type.95d, %SubWith.facet.eb2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.specific_fn.0d5da2.1: <specific function> = specific_function %Cpp.long.as.SubWith.impl.Op.8a648c.2, @Cpp.long.as.SubWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.specific_fn.0d5da2.2: <specific function> = specific_function %Cpp.long.as.SubWith.impl.Op.8a648c.1, @Cpp.long.as.SubWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness.135: <witness> = impl_witness imports.%MulWith.impl_witness_table.6e7, @Cpp.long.as.MulWith.impl.c90(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.type.8651ed.1: type = fn_type @Cpp.long.as.MulWith.impl.Op.2, @Cpp.long.as.MulWith.impl.c90(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.17c39d.1: %Cpp.long.as.MulWith.impl.Op.type.8651ed.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.type.8651ed.2: type = fn_type @Cpp.long.as.MulWith.impl.Op.1, @Cpp.long.as.MulWith.impl.c90(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.17c39d.2: %Cpp.long.as.MulWith.impl.Op.type.8651ed.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.facet.cdc: %MulWith.type.4ce = facet_value %Cpp.long, (%MulWith.impl_witness.135) [concrete]
// CHECK:STDOUT:   %.40a: type = fn_type_with_self_type %MulWith.Op.type.ff6, %MulWith.facet.cdc [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.specific_fn.fcc384.1: <specific function> = specific_function %Cpp.long.as.MulWith.impl.Op.17c39d.2, @Cpp.long.as.MulWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.specific_fn.fcc384.2: <specific function> = specific_function %Cpp.long.as.MulWith.impl.Op.17c39d.1, @Cpp.long.as.MulWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness.92c: <witness> = impl_witness imports.%DivWith.impl_witness_table.bcf, @Cpp.long.as.DivWith.impl.0e5(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.type.625951.1: type = fn_type @Cpp.long.as.DivWith.impl.Op.2, @Cpp.long.as.DivWith.impl.0e5(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.3687cc.1: %Cpp.long.as.DivWith.impl.Op.type.625951.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.type.625951.2: type = fn_type @Cpp.long.as.DivWith.impl.Op.1, @Cpp.long.as.DivWith.impl.0e5(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.3687cc.2: %Cpp.long.as.DivWith.impl.Op.type.625951.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.facet.7f1: %DivWith.type.234 = facet_value %Cpp.long, (%DivWith.impl_witness.92c) [concrete]
// CHECK:STDOUT:   %.b93: type = fn_type_with_self_type %DivWith.Op.type.c64, %DivWith.facet.7f1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.specific_fn.ecf90c.1: <specific function> = specific_function %Cpp.long.as.DivWith.impl.Op.3687cc.2, @Cpp.long.as.DivWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.specific_fn.ecf90c.2: <specific function> = specific_function %Cpp.long.as.DivWith.impl.Op.3687cc.1, @Cpp.long.as.DivWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness.334: <witness> = impl_witness imports.%ModWith.impl_witness_table.9cf, @Cpp.long.as.ModWith.impl.0a5(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.type.3cf811.1: type = fn_type @Cpp.long.as.ModWith.impl.Op.2, @Cpp.long.as.ModWith.impl.0a5(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.28b8b4.1: %Cpp.long.as.ModWith.impl.Op.type.3cf811.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.type.3cf811.2: type = fn_type @Cpp.long.as.ModWith.impl.Op.1, @Cpp.long.as.ModWith.impl.0a5(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.28b8b4.2: %Cpp.long.as.ModWith.impl.Op.type.3cf811.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.facet.52c: %ModWith.type.920 = facet_value %Cpp.long, (%ModWith.impl_witness.334) [concrete]
// CHECK:STDOUT:   %.e41: type = fn_type_with_self_type %ModWith.Op.type.295, %ModWith.facet.52c [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.specific_fn.175599.1: <specific function> = specific_function %Cpp.long.as.ModWith.impl.Op.28b8b4.2, @Cpp.long.as.ModWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.specific_fn.175599.2: <specific function> = specific_function %Cpp.long.as.ModWith.impl.Op.28b8b4.1, @Cpp.long.as.ModWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl.b2d [concrete]
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3a8: @Cpp.long.as.AddWith.impl.c72.%Cpp.long.as.AddWith.impl.Op.type.2 (%Cpp.long.as.AddWith.impl.Op.type.b730f9.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddWith.impl.c72.%Cpp.long.as.AddWith.impl.Op.2 (constants.%Cpp.long.as.AddWith.impl.Op.7ff34c.1)]
// CHECK:STDOUT:   %AddWith.impl_witness_table.34e = impl_witness_table (%Core.import_ref.cb912f.1, %Core.import_ref.3a8), @Cpp.long.as.AddWith.impl.c72 [concrete]
// CHECK:STDOUT:   %Core.Op.e41f: @Cpp.long.as.AddWith.impl.c72.%Cpp.long.as.AddWith.impl.Op.type.1 (%Cpp.long.as.AddWith.impl.Op.type.b730f9.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddWith.impl.c72.%Cpp.long.as.AddWith.impl.Op.1 (constants.%Cpp.long.as.AddWith.impl.Op.7ff34c.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.3 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.399: @Cpp.long.as.SubWith.impl.b84.%Cpp.long.as.SubWith.impl.Op.type.2 (%Cpp.long.as.SubWith.impl.Op.type.432302.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubWith.impl.b84.%Cpp.long.as.SubWith.impl.Op.2 (constants.%Cpp.long.as.SubWith.impl.Op.42d8b9.1)]
// CHECK:STDOUT:   %SubWith.impl_witness_table.694 = impl_witness_table (%Core.import_ref.cb912f.3, %Core.import_ref.399), @Cpp.long.as.SubWith.impl.b84 [concrete]
// CHECK:STDOUT:   %Core.Op.fce: @Cpp.long.as.SubWith.impl.b84.%Cpp.long.as.SubWith.impl.Op.type.1 (%Cpp.long.as.SubWith.impl.Op.type.432302.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubWith.impl.b84.%Cpp.long.as.SubWith.impl.Op.1 (constants.%Cpp.long.as.SubWith.impl.Op.42d8b9.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.674: @Cpp.long.as.MulWith.impl.c90.%Cpp.long.as.MulWith.impl.Op.type.2 (%Cpp.long.as.MulWith.impl.Op.type.4f6939.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulWith.impl.c90.%Cpp.long.as.MulWith.impl.Op.2 (constants.%Cpp.long.as.MulWith.impl.Op.b232c0.1)]
// CHECK:STDOUT:   %MulWith.impl_witness_table.6e7 = impl_witness_table (%Core.import_ref.cb912f.5, %Core.import_ref.674), @Cpp.long.as.MulWith.impl.c90 [concrete]
// CHECK:STDOUT:   %Core.Op.b1f: @Cpp.long.as.MulWith.impl.c90.%Cpp.long.as.MulWith.impl.Op.type.1 (%Cpp.long.as.MulWith.impl.Op.type.4f6939.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulWith.impl.c90.%Cpp.long.as.MulWith.impl.Op.1 (constants.%Cpp.long.as.MulWith.impl.Op.b232c0.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.7 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.228: @Cpp.long.as.DivWith.impl.0e5.%Cpp.long.as.DivWith.impl.Op.type.2 (%Cpp.long.as.DivWith.impl.Op.type.9ce9b6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivWith.impl.0e5.%Cpp.long.as.DivWith.impl.Op.2 (constants.%Cpp.long.as.DivWith.impl.Op.e35fb3.1)]
// CHECK:STDOUT:   %DivWith.impl_witness_table.bcf = impl_witness_table (%Core.import_ref.cb912f.7, %Core.import_ref.228), @Cpp.long.as.DivWith.impl.0e5 [concrete]
// CHECK:STDOUT:   %Core.Op.d43: @Cpp.long.as.DivWith.impl.0e5.%Cpp.long.as.DivWith.impl.Op.type.1 (%Cpp.long.as.DivWith.impl.Op.type.9ce9b6.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivWith.impl.0e5.%Cpp.long.as.DivWith.impl.Op.1 (constants.%Cpp.long.as.DivWith.impl.Op.e35fb3.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bd8: @Cpp.long.as.ModWith.impl.0a5.%Cpp.long.as.ModWith.impl.Op.type.2 (%Cpp.long.as.ModWith.impl.Op.type.4bf837.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModWith.impl.0a5.%Cpp.long.as.ModWith.impl.Op.2 (constants.%Cpp.long.as.ModWith.impl.Op.0707cb.1)]
// CHECK:STDOUT:   %ModWith.impl_witness_table.9cf = impl_witness_table (%Core.import_ref.cb912f.9, %Core.import_ref.bd8), @Cpp.long.as.ModWith.impl.0a5 [concrete]
// CHECK:STDOUT:   %Core.Op.4be: @Cpp.long.as.ModWith.impl.0a5.%Cpp.long.as.ModWith.impl.Op.type.1 (%Cpp.long.as.ModWith.impl.Op.type.4bf837.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModWith.impl.0a5.%Cpp.long.as.ModWith.impl.Op.1 (constants.%Cpp.long.as.ModWith.impl.Op.0707cb.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ArithmeticHeterogeneousLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.68c = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc11: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11(%int_1.loc11) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc11_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc11_21.2: %Cpp.long = converted %int_1.loc11, %.loc11_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %x: %Cpp.long = value_binding x, %.loc11_21.2
// CHECK:STDOUT:   %AssertSameType.ref.loc13: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc13_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc13: %.ef1 = impl_witness_access constants.%AddWith.impl_witness.403, element1 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.7beae3.2]
// CHECK:STDOUT:   %bound_method.loc13_20.1: <bound method> = bound_method %x.ref.loc13_18, %impl.elem1.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc13_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc13_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem1.loc13, @Cpp.long.as.AddWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddWith.impl.Op.specific_fn.88f499.1]
// CHECK:STDOUT:   %bound_method.loc13_20.2: <bound method> = bound_method %x.ref.loc13_18, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_20.3: %Cpp.long.as.AddWith.impl.Op.type.3dbf51.1 = specific_constant imports.%Core.Op.e41f, @Cpp.long.as.AddWith.impl.c72(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddWith.impl.Op.7beae3.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.AddWith.impl.Op.type.3dbf51.1 = name_ref Op, %.loc13_20.3 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.7beae3.1]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.bound.loc13: <bound method> = bound_method %x.ref.loc13_18, %Op.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_20.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_20: init %Cpp.long = call %bound_method.loc13_20.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_20
// CHECK:STDOUT:   %.loc13_20.5: %Cpp.long = converted %b.ref.loc13, %.loc13_20.4
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.specific_fn.loc13: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.AddWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddWith.impl.Op.specific_fn.88f499.2]
// CHECK:STDOUT:   %bound_method.loc13_20.4: <bound method> = bound_method %x.ref.loc13_18, %Cpp.long.as.AddWith.impl.Op.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_22: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_22: init %Cpp.long = call %bound_method.loc13_22(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_22
// CHECK:STDOUT:   %.loc13_22.2: %Cpp.long = converted %b.ref.loc13, %.loc13_22.1
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.call.loc13: init %Cpp.long = call %bound_method.loc13_20.4(%x.ref.loc13_18, %.loc13_22.2)
// CHECK:STDOUT:   %x.ref.loc13_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc13: <specific function> = specific_function %AssertSameType.ref.loc13, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc13_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.AddWith.impl.Op.call.loc13
// CHECK:STDOUT:   %.loc13_20.7: %Cpp.long = converted %Cpp.long.as.AddWith.impl.Op.call.loc13, %.loc13_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc13: init %empty_tuple.type = call %AssertSameType.specific_fn.loc13(%.loc13_20.7, %x.ref.loc13_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc14: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc14_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc14: %.494 = impl_witness_access constants.%SubWith.impl_witness.bee, element1 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.65b61c.2]
// CHECK:STDOUT:   %bound_method.loc14_20.1: <bound method> = bound_method %x.ref.loc14_18, %impl.elem1.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc14_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc14_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem1.loc14, @Cpp.long.as.SubWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubWith.impl.Op.specific_fn.8caeaa.1]
// CHECK:STDOUT:   %bound_method.loc14_20.2: <bound method> = bound_method %x.ref.loc14_18, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_20.3: %Cpp.long.as.SubWith.impl.Op.type.5b2a81.1 = specific_constant imports.%Core.Op.fce, @Cpp.long.as.SubWith.impl.b84(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubWith.impl.Op.65b61c.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.SubWith.impl.Op.type.5b2a81.1 = name_ref Op, %.loc14_20.3 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.65b61c.1]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.bound.loc14: <bound method> = bound_method %x.ref.loc14_18, %Op.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_20.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_20: init %Cpp.long = call %bound_method.loc14_20.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_20
// CHECK:STDOUT:   %.loc14_20.5: %Cpp.long = converted %b.ref.loc14, %.loc14_20.4
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.specific_fn.loc14: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.SubWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubWith.impl.Op.specific_fn.8caeaa.2]
// CHECK:STDOUT:   %bound_method.loc14_20.4: <bound method> = bound_method %x.ref.loc14_18, %Cpp.long.as.SubWith.impl.Op.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_22: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_22: init %Cpp.long = call %bound_method.loc14_22(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_22
// CHECK:STDOUT:   %.loc14_22.2: %Cpp.long = converted %b.ref.loc14, %.loc14_22.1
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.call.loc14: init %Cpp.long = call %bound_method.loc14_20.4(%x.ref.loc14_18, %.loc14_22.2)
// CHECK:STDOUT:   %x.ref.loc14_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc14: <specific function> = specific_function %AssertSameType.ref.loc14, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc14_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.SubWith.impl.Op.call.loc14
// CHECK:STDOUT:   %.loc14_20.7: %Cpp.long = converted %Cpp.long.as.SubWith.impl.Op.call.loc14, %.loc14_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc14: init %empty_tuple.type = call %AssertSameType.specific_fn.loc14(%.loc14_20.7, %x.ref.loc14_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc15: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc15_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.057 = impl_witness_access constants.%MulWith.impl_witness.ba5, element1 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.f095b7.2]
// CHECK:STDOUT:   %bound_method.loc15_20.1: <bound method> = bound_method %x.ref.loc15_18, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc15_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc15_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc15_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc15_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long.as.MulWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulWith.impl.Op.specific_fn.3be598.1]
// CHECK:STDOUT:   %bound_method.loc15_20.2: <bound method> = bound_method %x.ref.loc15_18, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_20.3: %Cpp.long.as.MulWith.impl.Op.type.ea5ff6.1 = specific_constant imports.%Core.Op.b1f, @Cpp.long.as.MulWith.impl.c90(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulWith.impl.Op.f095b7.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long.as.MulWith.impl.Op.type.ea5ff6.1 = name_ref Op, %.loc15_20.3 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.f095b7.1]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.bound.loc15: <bound method> = bound_method %x.ref.loc15_18, %Op.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_20.3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_20: init %Cpp.long = call %bound_method.loc15_20.3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_20
// CHECK:STDOUT:   %.loc15_20.5: %Cpp.long = converted %b.ref.loc15, %.loc15_20.4
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.specific_fn.loc15: <specific function> = specific_function %Op.ref.loc15, @Cpp.long.as.MulWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulWith.impl.Op.specific_fn.3be598.2]
// CHECK:STDOUT:   %bound_method.loc15_20.4: <bound method> = bound_method %x.ref.loc15_18, %Cpp.long.as.MulWith.impl.Op.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_22: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_22: init %Cpp.long = call %bound_method.loc15_22(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_22
// CHECK:STDOUT:   %.loc15_22.2: %Cpp.long = converted %b.ref.loc15, %.loc15_22.1
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.call.loc15: init %Cpp.long = call %bound_method.loc15_20.4(%x.ref.loc15_18, %.loc15_22.2)
// CHECK:STDOUT:   %x.ref.loc15_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc15: <specific function> = specific_function %AssertSameType.ref.loc15, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc15_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.MulWith.impl.Op.call.loc15
// CHECK:STDOUT:   %.loc15_20.7: %Cpp.long = converted %Cpp.long.as.MulWith.impl.Op.call.loc15, %.loc15_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc15: init %empty_tuple.type = call %AssertSameType.specific_fn.loc15(%.loc15_20.7, %x.ref.loc15_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc16: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc16_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %b.ref.loc16: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc16: %.eec = impl_witness_access constants.%DivWith.impl_witness.30d, element1 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.d06e88.2]
// CHECK:STDOUT:   %bound_method.loc16_20.1: <bound method> = bound_method %x.ref.loc16_18, %impl.elem1.loc16
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc16_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc16_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc16_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc16_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem1.loc16, @Cpp.long.as.DivWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivWith.impl.Op.specific_fn.6489e2.1]
// CHECK:STDOUT:   %bound_method.loc16_20.2: <bound method> = bound_method %x.ref.loc16_18, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_20.3: %Cpp.long.as.DivWith.impl.Op.type.9156f2.1 = specific_constant imports.%Core.Op.d43, @Cpp.long.as.DivWith.impl.0e5(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivWith.impl.Op.d06e88.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long.as.DivWith.impl.Op.type.9156f2.1 = name_ref Op, %.loc16_20.3 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.d06e88.1]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.bound.loc16: <bound method> = bound_method %x.ref.loc16_18, %Op.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_20.3: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_20: init %Cpp.long = call %bound_method.loc16_20.3(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_20
// CHECK:STDOUT:   %.loc16_20.5: %Cpp.long = converted %b.ref.loc16, %.loc16_20.4
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.specific_fn.loc16: <specific function> = specific_function %Op.ref.loc16, @Cpp.long.as.DivWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivWith.impl.Op.specific_fn.6489e2.2]
// CHECK:STDOUT:   %bound_method.loc16_20.4: <bound method> = bound_method %x.ref.loc16_18, %Cpp.long.as.DivWith.impl.Op.specific_fn.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_22: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_22: init %Cpp.long = call %bound_method.loc16_22(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_22
// CHECK:STDOUT:   %.loc16_22.2: %Cpp.long = converted %b.ref.loc16, %.loc16_22.1
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.call.loc16: init %Cpp.long = call %bound_method.loc16_20.4(%x.ref.loc16_18, %.loc16_22.2)
// CHECK:STDOUT:   %x.ref.loc16_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc16: <specific function> = specific_function %AssertSameType.ref.loc16, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc16_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.DivWith.impl.Op.call.loc16
// CHECK:STDOUT:   %.loc16_20.7: %Cpp.long = converted %Cpp.long.as.DivWith.impl.Op.call.loc16, %.loc16_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc16: init %empty_tuple.type = call %AssertSameType.specific_fn.loc16(%.loc16_20.7, %x.ref.loc16_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc17: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc17_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %b.ref.loc17: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc17: %.289 = impl_witness_access constants.%ModWith.impl_witness.bd3, element1 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.6c7fa7.2]
// CHECK:STDOUT:   %bound_method.loc17_20.1: <bound method> = bound_method %x.ref.loc17_18, %impl.elem1.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc17_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc17_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc17_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc17_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem1.loc17, @Cpp.long.as.ModWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModWith.impl.Op.specific_fn.127c5b.1]
// CHECK:STDOUT:   %bound_method.loc17_20.2: <bound method> = bound_method %x.ref.loc17_18, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_20.3: %Cpp.long.as.ModWith.impl.Op.type.f6cdb5.1 = specific_constant imports.%Core.Op.4be, @Cpp.long.as.ModWith.impl.0a5(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModWith.impl.Op.6c7fa7.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.ModWith.impl.Op.type.f6cdb5.1 = name_ref Op, %.loc17_20.3 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.6c7fa7.1]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.bound.loc17: <bound method> = bound_method %x.ref.loc17_18, %Op.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_20.3: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_20: init %Cpp.long = call %bound_method.loc17_20.3(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_20
// CHECK:STDOUT:   %.loc17_20.5: %Cpp.long = converted %b.ref.loc17, %.loc17_20.4
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.specific_fn.loc17: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.ModWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModWith.impl.Op.specific_fn.127c5b.2]
// CHECK:STDOUT:   %bound_method.loc17_20.4: <bound method> = bound_method %x.ref.loc17_18, %Cpp.long.as.ModWith.impl.Op.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_22: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_22: init %Cpp.long = call %bound_method.loc17_22(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_22
// CHECK:STDOUT:   %.loc17_22.2: %Cpp.long = converted %b.ref.loc17, %.loc17_22.1
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.call.loc17: init %Cpp.long = call %bound_method.loc17_20.4(%x.ref.loc17_18, %.loc17_22.2)
// CHECK:STDOUT:   %x.ref.loc17_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc17: <specific function> = specific_function %AssertSameType.ref.loc17, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc17_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.ModWith.impl.Op.call.loc17
// CHECK:STDOUT:   %.loc17_20.7: %Cpp.long = converted %Cpp.long.as.ModWith.impl.Op.call.loc17, %.loc17_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc17: init %empty_tuple.type = call %AssertSameType.specific_fn.loc17(%.loc17_20.7, %x.ref.loc17_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc19: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc19_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc19: %.784 = impl_witness_access constants.%AddWith.impl_witness.e6c, element1 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.0f2d31.2]
// CHECK:STDOUT:   %bound_method.loc19_20.1: <bound method> = bound_method %x.ref.loc19_18, %impl.elem1.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_20.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc19_20.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_20.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_20.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc19_20.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_20.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem1.loc19, @Cpp.long.as.AddWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.AddWith.impl.Op.specific_fn.063c7e.1]
// CHECK:STDOUT:   %bound_method.loc19_20.2: <bound method> = bound_method %x.ref.loc19_18, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_20.3: %Cpp.long.as.AddWith.impl.Op.type.1ece0d.1 = specific_constant imports.%Core.Op.e41f, @Cpp.long.as.AddWith.impl.c72(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.AddWith.impl.Op.0f2d31.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.AddWith.impl.Op.type.1ece0d.1 = name_ref Op, %.loc19_20.3 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.0f2d31.1]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.bound.loc19: <bound method> = bound_method %x.ref.loc19_18, %Op.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_20.3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_20: init %Cpp.long = call %bound_method.loc19_20.3(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_20.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_20.5: %Cpp.long = converted %int_1.loc19, %.loc19_20.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.specific_fn.loc19: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.AddWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.AddWith.impl.Op.specific_fn.063c7e.2]
// CHECK:STDOUT:   %bound_method.loc19_20.4: <bound method> = bound_method %x.ref.loc19_18, %Cpp.long.as.AddWith.impl.Op.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_22: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_22: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_22: init %Cpp.long = call %bound_method.loc19_22(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_22.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_22.2: %Cpp.long = converted %int_1.loc19, %.loc19_22.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.call.loc19: init %Cpp.long = call %bound_method.loc19_20.4(%x.ref.loc19_18, %.loc19_22.2)
// CHECK:STDOUT:   %x.ref.loc19_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc19: <specific function> = specific_function %AssertSameType.ref.loc19, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc19_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.AddWith.impl.Op.call.loc19
// CHECK:STDOUT:   %.loc19_20.7: %Cpp.long = converted %Cpp.long.as.AddWith.impl.Op.call.loc19, %.loc19_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc19: init %empty_tuple.type = call %AssertSameType.specific_fn.loc19(%.loc19_20.7, %x.ref.loc19_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc20: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc20_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc20: %.d5f = impl_witness_access constants.%SubWith.impl_witness.dd6, element1 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.8a648c.2]
// CHECK:STDOUT:   %bound_method.loc20_20.1: <bound method> = bound_method %x.ref.loc20_18, %impl.elem1.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_20.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc20_20.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_20.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_20.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc20_20.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_20.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem1.loc20, @Cpp.long.as.SubWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.SubWith.impl.Op.specific_fn.0d5da2.1]
// CHECK:STDOUT:   %bound_method.loc20_20.2: <bound method> = bound_method %x.ref.loc20_18, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_20.3: %Cpp.long.as.SubWith.impl.Op.type.ef5597.1 = specific_constant imports.%Core.Op.fce, @Cpp.long.as.SubWith.impl.b84(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.SubWith.impl.Op.8a648c.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.SubWith.impl.Op.type.ef5597.1 = name_ref Op, %.loc20_20.3 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.8a648c.1]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.bound.loc20: <bound method> = bound_method %x.ref.loc20_18, %Op.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_20.3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_20: init %Cpp.long = call %bound_method.loc20_20.3(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_20.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_20.5: %Cpp.long = converted %int_1.loc20, %.loc20_20.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.specific_fn.loc20: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.SubWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.SubWith.impl.Op.specific_fn.0d5da2.2]
// CHECK:STDOUT:   %bound_method.loc20_20.4: <bound method> = bound_method %x.ref.loc20_18, %Cpp.long.as.SubWith.impl.Op.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_22: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_22: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_22: init %Cpp.long = call %bound_method.loc20_22(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_22.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_22.2: %Cpp.long = converted %int_1.loc20, %.loc20_22.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.call.loc20: init %Cpp.long = call %bound_method.loc20_20.4(%x.ref.loc20_18, %.loc20_22.2)
// CHECK:STDOUT:   %x.ref.loc20_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc20: <specific function> = specific_function %AssertSameType.ref.loc20, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc20_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.SubWith.impl.Op.call.loc20
// CHECK:STDOUT:   %.loc20_20.7: %Cpp.long = converted %Cpp.long.as.SubWith.impl.Op.call.loc20, %.loc20_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc20: init %empty_tuple.type = call %AssertSameType.specific_fn.loc20(%.loc20_20.7, %x.ref.loc20_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc21: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc21_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc21: %.40a = impl_witness_access constants.%MulWith.impl_witness.135, element1 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.17c39d.2]
// CHECK:STDOUT:   %bound_method.loc21_20.1: <bound method> = bound_method %x.ref.loc21_18, %impl.elem1.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_20.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc21_20.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_20.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_20.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc21_20.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_20.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem1.loc21, @Cpp.long.as.MulWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.MulWith.impl.Op.specific_fn.fcc384.1]
// CHECK:STDOUT:   %bound_method.loc21_20.2: <bound method> = bound_method %x.ref.loc21_18, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_20.3: %Cpp.long.as.MulWith.impl.Op.type.8651ed.1 = specific_constant imports.%Core.Op.b1f, @Cpp.long.as.MulWith.impl.c90(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.MulWith.impl.Op.17c39d.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long.as.MulWith.impl.Op.type.8651ed.1 = name_ref Op, %.loc21_20.3 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.17c39d.1]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.bound.loc21: <bound method> = bound_method %x.ref.loc21_18, %Op.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_20.3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_20: init %Cpp.long = call %bound_method.loc21_20.3(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_20.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_20.5: %Cpp.long = converted %int_1.loc21, %.loc21_20.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.specific_fn.loc21: <specific function> = specific_function %Op.ref.loc21, @Cpp.long.as.MulWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.MulWith.impl.Op.specific_fn.fcc384.2]
// CHECK:STDOUT:   %bound_method.loc21_20.4: <bound method> = bound_method %x.ref.loc21_18, %Cpp.long.as.MulWith.impl.Op.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_22: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_22: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_22: init %Cpp.long = call %bound_method.loc21_22(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_22.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_22.2: %Cpp.long = converted %int_1.loc21, %.loc21_22.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.call.loc21: init %Cpp.long = call %bound_method.loc21_20.4(%x.ref.loc21_18, %.loc21_22.2)
// CHECK:STDOUT:   %x.ref.loc21_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc21: <specific function> = specific_function %AssertSameType.ref.loc21, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc21_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.MulWith.impl.Op.call.loc21
// CHECK:STDOUT:   %.loc21_20.7: %Cpp.long = converted %Cpp.long.as.MulWith.impl.Op.call.loc21, %.loc21_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc21: init %empty_tuple.type = call %AssertSameType.specific_fn.loc21(%.loc21_20.7, %x.ref.loc21_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc22: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc22_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.b93 = impl_witness_access constants.%DivWith.impl_witness.92c, element1 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.3687cc.2]
// CHECK:STDOUT:   %bound_method.loc22_20.1: <bound method> = bound_method %x.ref.loc22_18, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_20.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc22_20.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_20.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_20.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc22_20.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_20.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long.as.DivWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.DivWith.impl.Op.specific_fn.ecf90c.1]
// CHECK:STDOUT:   %bound_method.loc22_20.2: <bound method> = bound_method %x.ref.loc22_18, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_20.3: %Cpp.long.as.DivWith.impl.Op.type.625951.1 = specific_constant imports.%Core.Op.d43, @Cpp.long.as.DivWith.impl.0e5(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.DivWith.impl.Op.3687cc.1]
// CHECK:STDOUT:   %Op.ref.loc22: %Cpp.long.as.DivWith.impl.Op.type.625951.1 = name_ref Op, %.loc22_20.3 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.3687cc.1]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.bound.loc22: <bound method> = bound_method %x.ref.loc22_18, %Op.ref.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_20.3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_20: init %Cpp.long = call %bound_method.loc22_20.3(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_20.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_20.5: %Cpp.long = converted %int_1.loc22, %.loc22_20.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.specific_fn.loc22: <specific function> = specific_function %Op.ref.loc22, @Cpp.long.as.DivWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.DivWith.impl.Op.specific_fn.ecf90c.2]
// CHECK:STDOUT:   %bound_method.loc22_20.4: <bound method> = bound_method %x.ref.loc22_18, %Cpp.long.as.DivWith.impl.Op.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_22: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_22: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_22: init %Cpp.long = call %bound_method.loc22_22(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_22.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_22.2: %Cpp.long = converted %int_1.loc22, %.loc22_22.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.call.loc22: init %Cpp.long = call %bound_method.loc22_20.4(%x.ref.loc22_18, %.loc22_22.2)
// CHECK:STDOUT:   %x.ref.loc22_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc22: <specific function> = specific_function %AssertSameType.ref.loc22, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc22_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.DivWith.impl.Op.call.loc22
// CHECK:STDOUT:   %.loc22_20.7: %Cpp.long = converted %Cpp.long.as.DivWith.impl.Op.call.loc22, %.loc22_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc22: init %empty_tuple.type = call %AssertSameType.specific_fn.loc22(%.loc22_20.7, %x.ref.loc22_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc23: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc23_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc23: %.e41 = impl_witness_access constants.%ModWith.impl_witness.334, element1 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.28b8b4.2]
// CHECK:STDOUT:   %bound_method.loc23_20.1: <bound method> = bound_method %x.ref.loc23_18, %impl.elem1.loc23
// CHECK:STDOUT:   %ImplicitAs.facet.loc23_20.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc23_20.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc23_20.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc23_20.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc23_20.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc23_20.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem1.loc23, @Cpp.long.as.ModWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.ModWith.impl.Op.specific_fn.175599.1]
// CHECK:STDOUT:   %bound_method.loc23_20.2: <bound method> = bound_method %x.ref.loc23_18, %specific_fn.loc23
// CHECK:STDOUT:   %.loc23_20.3: %Cpp.long.as.ModWith.impl.Op.type.3cf811.1 = specific_constant imports.%Core.Op.4be, @Cpp.long.as.ModWith.impl.0a5(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.ModWith.impl.Op.28b8b4.1]
// CHECK:STDOUT:   %Op.ref.loc23: %Cpp.long.as.ModWith.impl.Op.type.3cf811.1 = name_ref Op, %.loc23_20.3 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.28b8b4.1]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.bound.loc23: <bound method> = bound_method %x.ref.loc23_18, %Op.ref.loc23
// CHECK:STDOUT:   %impl.elem0.loc23_20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc23_20.3: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23_20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_20: init %Cpp.long = call %bound_method.loc23_20.3(%int_1.loc23) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23_20.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23_20.5: %Cpp.long = converted %int_1.loc23, %.loc23_20.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.specific_fn.loc23: <specific function> = specific_function %Op.ref.loc23, @Cpp.long.as.ModWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.ModWith.impl.Op.specific_fn.175599.2]
// CHECK:STDOUT:   %bound_method.loc23_20.4: <bound method> = bound_method %x.ref.loc23_18, %Cpp.long.as.ModWith.impl.Op.specific_fn.loc23
// CHECK:STDOUT:   %impl.elem0.loc23_22: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc23_22: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_22: init %Cpp.long = call %bound_method.loc23_22(%int_1.loc23) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23_22.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23_22.2: %Cpp.long = converted %int_1.loc23, %.loc23_22.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.call.loc23: init %Cpp.long = call %bound_method.loc23_20.4(%x.ref.loc23_18, %.loc23_22.2)
// CHECK:STDOUT:   %x.ref.loc23_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc23: <specific function> = specific_function %AssertSameType.ref.loc23, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc23_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.ModWith.impl.Op.call.loc23
// CHECK:STDOUT:   %.loc23_20.7: %Cpp.long = converted %Cpp.long.as.ModWith.impl.Op.call.loc23, %.loc23_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc23: init %empty_tuple.type = call %AssertSameType.specific_fn.loc23(%.loc23_20.7, %x.ref.loc23_25)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.7ce = value_binding_pattern y [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc25: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc25_10: type = splice_block %i32.loc25 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc25: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc25_16.1: <bound method> = bound_method %int_1.loc25, %impl.elem0.loc25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_16.2: <bound method> = bound_method %int_1.loc25, %specific_fn.loc25 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25: init %i32 = call %bound_method.loc25_16.2(%int_1.loc25) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc25_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc25_16.2: %i32 = converted %int_1.loc25, %.loc25_16.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %y: %i32 = value_binding y, %.loc25_16.2
// CHECK:STDOUT:   %AssertSameType.ref.loc26: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc26_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc26: %i32 = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc26: %.ef1 = impl_witness_access constants.%AddWith.impl_witness.403, element1 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.7beae3.2]
// CHECK:STDOUT:   %bound_method.loc26_20.1: <bound method> = bound_method %x.ref.loc26_18, %impl.elem1.loc26
// CHECK:STDOUT:   %ImplicitAs.facet.loc26_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc26_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc26_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc26_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc26_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc26_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem1.loc26, @Cpp.long.as.AddWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddWith.impl.Op.specific_fn.88f499.1]
// CHECK:STDOUT:   %bound_method.loc26_20.2: <bound method> = bound_method %x.ref.loc26_18, %specific_fn.loc26
// CHECK:STDOUT:   %.loc26_20.3: %Cpp.long.as.AddWith.impl.Op.type.3dbf51.1 = specific_constant imports.%Core.Op.e41f, @Cpp.long.as.AddWith.impl.c72(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddWith.impl.Op.7beae3.1]
// CHECK:STDOUT:   %Op.ref.loc26: %Cpp.long.as.AddWith.impl.Op.type.3dbf51.1 = name_ref Op, %.loc26_20.3 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.7beae3.1]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.bound.loc26: <bound method> = bound_method %x.ref.loc26_18, %Op.ref.loc26
// CHECK:STDOUT:   %impl.elem0.loc26_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_20.3: <bound method> = bound_method %y.ref.loc26, %impl.elem0.loc26_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc26_20: init %Cpp.long = call %bound_method.loc26_20.3(%y.ref.loc26)
// CHECK:STDOUT:   %.loc26_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc26_20
// CHECK:STDOUT:   %.loc26_20.5: %Cpp.long = converted %y.ref.loc26, %.loc26_20.4
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.specific_fn.loc26: <specific function> = specific_function %Op.ref.loc26, @Cpp.long.as.AddWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddWith.impl.Op.specific_fn.88f499.2]
// CHECK:STDOUT:   %bound_method.loc26_20.4: <bound method> = bound_method %x.ref.loc26_18, %Cpp.long.as.AddWith.impl.Op.specific_fn.loc26
// CHECK:STDOUT:   %impl.elem0.loc26_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_22: <bound method> = bound_method %y.ref.loc26, %impl.elem0.loc26_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc26_22: init %Cpp.long = call %bound_method.loc26_22(%y.ref.loc26)
// CHECK:STDOUT:   %.loc26_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc26_22
// CHECK:STDOUT:   %.loc26_22.2: %Cpp.long = converted %y.ref.loc26, %.loc26_22.1
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.call.loc26: init %Cpp.long = call %bound_method.loc26_20.4(%x.ref.loc26_18, %.loc26_22.2)
// CHECK:STDOUT:   %x.ref.loc26_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc26: <specific function> = specific_function %AssertSameType.ref.loc26, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc26_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.AddWith.impl.Op.call.loc26
// CHECK:STDOUT:   %.loc26_20.7: %Cpp.long = converted %Cpp.long.as.AddWith.impl.Op.call.loc26, %.loc26_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc26: init %empty_tuple.type = call %AssertSameType.specific_fn.loc26(%.loc26_20.7, %x.ref.loc26_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc27: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc27_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc27: %i32 = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc27: %.494 = impl_witness_access constants.%SubWith.impl_witness.bee, element1 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.65b61c.2]
// CHECK:STDOUT:   %bound_method.loc27_20.1: <bound method> = bound_method %x.ref.loc27_18, %impl.elem1.loc27
// CHECK:STDOUT:   %ImplicitAs.facet.loc27_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc27_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc27_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc27_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc27_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc27_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem1.loc27, @Cpp.long.as.SubWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubWith.impl.Op.specific_fn.8caeaa.1]
// CHECK:STDOUT:   %bound_method.loc27_20.2: <bound method> = bound_method %x.ref.loc27_18, %specific_fn.loc27
// CHECK:STDOUT:   %.loc27_20.3: %Cpp.long.as.SubWith.impl.Op.type.5b2a81.1 = specific_constant imports.%Core.Op.fce, @Cpp.long.as.SubWith.impl.b84(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubWith.impl.Op.65b61c.1]
// CHECK:STDOUT:   %Op.ref.loc27: %Cpp.long.as.SubWith.impl.Op.type.5b2a81.1 = name_ref Op, %.loc27_20.3 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.65b61c.1]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.bound.loc27: <bound method> = bound_method %x.ref.loc27_18, %Op.ref.loc27
// CHECK:STDOUT:   %impl.elem0.loc27_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_20.3: <bound method> = bound_method %y.ref.loc27, %impl.elem0.loc27_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc27_20: init %Cpp.long = call %bound_method.loc27_20.3(%y.ref.loc27)
// CHECK:STDOUT:   %.loc27_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc27_20
// CHECK:STDOUT:   %.loc27_20.5: %Cpp.long = converted %y.ref.loc27, %.loc27_20.4
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.specific_fn.loc27: <specific function> = specific_function %Op.ref.loc27, @Cpp.long.as.SubWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubWith.impl.Op.specific_fn.8caeaa.2]
// CHECK:STDOUT:   %bound_method.loc27_20.4: <bound method> = bound_method %x.ref.loc27_18, %Cpp.long.as.SubWith.impl.Op.specific_fn.loc27
// CHECK:STDOUT:   %impl.elem0.loc27_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_22: <bound method> = bound_method %y.ref.loc27, %impl.elem0.loc27_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc27_22: init %Cpp.long = call %bound_method.loc27_22(%y.ref.loc27)
// CHECK:STDOUT:   %.loc27_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc27_22
// CHECK:STDOUT:   %.loc27_22.2: %Cpp.long = converted %y.ref.loc27, %.loc27_22.1
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.call.loc27: init %Cpp.long = call %bound_method.loc27_20.4(%x.ref.loc27_18, %.loc27_22.2)
// CHECK:STDOUT:   %x.ref.loc27_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc27: <specific function> = specific_function %AssertSameType.ref.loc27, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc27_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.SubWith.impl.Op.call.loc27
// CHECK:STDOUT:   %.loc27_20.7: %Cpp.long = converted %Cpp.long.as.SubWith.impl.Op.call.loc27, %.loc27_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc27: init %empty_tuple.type = call %AssertSameType.specific_fn.loc27(%.loc27_20.7, %x.ref.loc27_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc28: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc28_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc28: %i32 = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc28: %.057 = impl_witness_access constants.%MulWith.impl_witness.ba5, element1 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.f095b7.2]
// CHECK:STDOUT:   %bound_method.loc28_20.1: <bound method> = bound_method %x.ref.loc28_18, %impl.elem1.loc28
// CHECK:STDOUT:   %ImplicitAs.facet.loc28_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc28_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc28_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc28_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc28_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc28_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem1.loc28, @Cpp.long.as.MulWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulWith.impl.Op.specific_fn.3be598.1]
// CHECK:STDOUT:   %bound_method.loc28_20.2: <bound method> = bound_method %x.ref.loc28_18, %specific_fn.loc28
// CHECK:STDOUT:   %.loc28_20.3: %Cpp.long.as.MulWith.impl.Op.type.ea5ff6.1 = specific_constant imports.%Core.Op.b1f, @Cpp.long.as.MulWith.impl.c90(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulWith.impl.Op.f095b7.1]
// CHECK:STDOUT:   %Op.ref.loc28: %Cpp.long.as.MulWith.impl.Op.type.ea5ff6.1 = name_ref Op, %.loc28_20.3 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.f095b7.1]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.bound.loc28: <bound method> = bound_method %x.ref.loc28_18, %Op.ref.loc28
// CHECK:STDOUT:   %impl.elem0.loc28_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_20.3: <bound method> = bound_method %y.ref.loc28, %impl.elem0.loc28_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc28_20: init %Cpp.long = call %bound_method.loc28_20.3(%y.ref.loc28)
// CHECK:STDOUT:   %.loc28_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc28_20
// CHECK:STDOUT:   %.loc28_20.5: %Cpp.long = converted %y.ref.loc28, %.loc28_20.4
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.specific_fn.loc28: <specific function> = specific_function %Op.ref.loc28, @Cpp.long.as.MulWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulWith.impl.Op.specific_fn.3be598.2]
// CHECK:STDOUT:   %bound_method.loc28_20.4: <bound method> = bound_method %x.ref.loc28_18, %Cpp.long.as.MulWith.impl.Op.specific_fn.loc28
// CHECK:STDOUT:   %impl.elem0.loc28_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_22: <bound method> = bound_method %y.ref.loc28, %impl.elem0.loc28_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc28_22: init %Cpp.long = call %bound_method.loc28_22(%y.ref.loc28)
// CHECK:STDOUT:   %.loc28_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc28_22
// CHECK:STDOUT:   %.loc28_22.2: %Cpp.long = converted %y.ref.loc28, %.loc28_22.1
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.call.loc28: init %Cpp.long = call %bound_method.loc28_20.4(%x.ref.loc28_18, %.loc28_22.2)
// CHECK:STDOUT:   %x.ref.loc28_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc28: <specific function> = specific_function %AssertSameType.ref.loc28, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc28_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.MulWith.impl.Op.call.loc28
// CHECK:STDOUT:   %.loc28_20.7: %Cpp.long = converted %Cpp.long.as.MulWith.impl.Op.call.loc28, %.loc28_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc28: init %empty_tuple.type = call %AssertSameType.specific_fn.loc28(%.loc28_20.7, %x.ref.loc28_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc29: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc29_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc29: %i32 = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc29: %.eec = impl_witness_access constants.%DivWith.impl_witness.30d, element1 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.d06e88.2]
// CHECK:STDOUT:   %bound_method.loc29_20.1: <bound method> = bound_method %x.ref.loc29_18, %impl.elem1.loc29
// CHECK:STDOUT:   %ImplicitAs.facet.loc29_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc29_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc29_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc29_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc29_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc29_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem1.loc29, @Cpp.long.as.DivWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivWith.impl.Op.specific_fn.6489e2.1]
// CHECK:STDOUT:   %bound_method.loc29_20.2: <bound method> = bound_method %x.ref.loc29_18, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29_20.3: %Cpp.long.as.DivWith.impl.Op.type.9156f2.1 = specific_constant imports.%Core.Op.d43, @Cpp.long.as.DivWith.impl.0e5(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivWith.impl.Op.d06e88.1]
// CHECK:STDOUT:   %Op.ref.loc29: %Cpp.long.as.DivWith.impl.Op.type.9156f2.1 = name_ref Op, %.loc29_20.3 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.d06e88.1]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.bound.loc29: <bound method> = bound_method %x.ref.loc29_18, %Op.ref.loc29
// CHECK:STDOUT:   %impl.elem0.loc29_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_20.3: <bound method> = bound_method %y.ref.loc29, %impl.elem0.loc29_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29_20: init %Cpp.long = call %bound_method.loc29_20.3(%y.ref.loc29)
// CHECK:STDOUT:   %.loc29_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29_20
// CHECK:STDOUT:   %.loc29_20.5: %Cpp.long = converted %y.ref.loc29, %.loc29_20.4
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.specific_fn.loc29: <specific function> = specific_function %Op.ref.loc29, @Cpp.long.as.DivWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivWith.impl.Op.specific_fn.6489e2.2]
// CHECK:STDOUT:   %bound_method.loc29_20.4: <bound method> = bound_method %x.ref.loc29_18, %Cpp.long.as.DivWith.impl.Op.specific_fn.loc29
// CHECK:STDOUT:   %impl.elem0.loc29_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_22: <bound method> = bound_method %y.ref.loc29, %impl.elem0.loc29_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29_22: init %Cpp.long = call %bound_method.loc29_22(%y.ref.loc29)
// CHECK:STDOUT:   %.loc29_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29_22
// CHECK:STDOUT:   %.loc29_22.2: %Cpp.long = converted %y.ref.loc29, %.loc29_22.1
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.call.loc29: init %Cpp.long = call %bound_method.loc29_20.4(%x.ref.loc29_18, %.loc29_22.2)
// CHECK:STDOUT:   %x.ref.loc29_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc29: <specific function> = specific_function %AssertSameType.ref.loc29, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc29_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.DivWith.impl.Op.call.loc29
// CHECK:STDOUT:   %.loc29_20.7: %Cpp.long = converted %Cpp.long.as.DivWith.impl.Op.call.loc29, %.loc29_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc29: init %empty_tuple.type = call %AssertSameType.specific_fn.loc29(%.loc29_20.7, %x.ref.loc29_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc30: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc30_18: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc30: %i32 = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc30: %.289 = impl_witness_access constants.%ModWith.impl_witness.bd3, element1 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.6c7fa7.2]
// CHECK:STDOUT:   %bound_method.loc30_20.1: <bound method> = bound_method %x.ref.loc30_18, %impl.elem1.loc30
// CHECK:STDOUT:   %ImplicitAs.facet.loc30_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc30_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc30_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc30_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc30_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc30_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem1.loc30, @Cpp.long.as.ModWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModWith.impl.Op.specific_fn.127c5b.1]
// CHECK:STDOUT:   %bound_method.loc30_20.2: <bound method> = bound_method %x.ref.loc30_18, %specific_fn.loc30
// CHECK:STDOUT:   %.loc30_20.3: %Cpp.long.as.ModWith.impl.Op.type.f6cdb5.1 = specific_constant imports.%Core.Op.4be, @Cpp.long.as.ModWith.impl.0a5(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModWith.impl.Op.6c7fa7.1]
// CHECK:STDOUT:   %Op.ref.loc30: %Cpp.long.as.ModWith.impl.Op.type.f6cdb5.1 = name_ref Op, %.loc30_20.3 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.6c7fa7.1]
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.bound.loc30: <bound method> = bound_method %x.ref.loc30_18, %Op.ref.loc30
// CHECK:STDOUT:   %impl.elem0.loc30_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc30_20.3: <bound method> = bound_method %y.ref.loc30, %impl.elem0.loc30_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc30_20: init %Cpp.long = call %bound_method.loc30_20.3(%y.ref.loc30)
// CHECK:STDOUT:   %.loc30_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc30_20
// CHECK:STDOUT:   %.loc30_20.5: %Cpp.long = converted %y.ref.loc30, %.loc30_20.4
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.specific_fn.loc30: <specific function> = specific_function %Op.ref.loc30, @Cpp.long.as.ModWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModWith.impl.Op.specific_fn.127c5b.2]
// CHECK:STDOUT:   %bound_method.loc30_20.4: <bound method> = bound_method %x.ref.loc30_18, %Cpp.long.as.ModWith.impl.Op.specific_fn.loc30
// CHECK:STDOUT:   %impl.elem0.loc30_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc30_22: <bound method> = bound_method %y.ref.loc30, %impl.elem0.loc30_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc30_22: init %Cpp.long = call %bound_method.loc30_22(%y.ref.loc30)
// CHECK:STDOUT:   %.loc30_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc30_22
// CHECK:STDOUT:   %.loc30_22.2: %Cpp.long = converted %y.ref.loc30, %.loc30_22.1
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.call.loc30: init %Cpp.long = call %bound_method.loc30_20.4(%x.ref.loc30_18, %.loc30_22.2)
// CHECK:STDOUT:   %x.ref.loc30_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc30: <specific function> = specific_function %AssertSameType.ref.loc30, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc30_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.ModWith.impl.Op.call.loc30
// CHECK:STDOUT:   %.loc30_20.7: %Cpp.long = converted %Cpp.long.as.ModWith.impl.Op.call.loc30, %.loc30_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc30: init %empty_tuple.type = call %AssertSameType.specific_fn.loc30(%.loc30_20.7, %x.ref.loc30_25)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- arithmetic_heterogeneous_long_right_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AssertSameType.type: type = fn_type @AssertSameType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AssertSameType: %AssertSameType.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %.863: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddWith.type.9c2: type = facet_type <@AddWith, @AddWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %AddWith.Op.type.9a6: type = fn_type @AddWith.Op, @AddWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.57d: %ImplicitAs.type.819 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.type.704168.1: type = fn_type @T.binding.as_type.as.AddWith.impl.Op.1, @T.binding.as_type.as.AddWith.impl.f1a(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.399102.1: %T.binding.as_type.as.AddWith.impl.Op.type.704168.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.type.704168.2: type = fn_type @T.binding.as_type.as.AddWith.impl.Op.2, @T.binding.as_type.as.AddWith.impl.f1a(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.399102.2: %T.binding.as_type.as.AddWith.impl.Op.type.704168.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.a1d: <witness> = impl_witness imports.%AddWith.impl_witness_table.0dc, @T.binding.as_type.as.AddWith.impl.f1a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.type.5105fb.1: type = fn_type @T.binding.as_type.as.AddWith.impl.Op.2, @T.binding.as_type.as.AddWith.impl.f1a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.3fccd2.1: %T.binding.as_type.as.AddWith.impl.Op.type.5105fb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.type.5105fb.2: type = fn_type @T.binding.as_type.as.AddWith.impl.Op.1, @T.binding.as_type.as.AddWith.impl.f1a(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.3fccd2.2: %T.binding.as_type.as.AddWith.impl.Op.type.5105fb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet.704: %AddWith.type.9c2 = facet_value %i32, (%AddWith.impl_witness.a1d) [concrete]
// CHECK:STDOUT:   %.285: type = fn_type_with_self_type %AddWith.Op.type.9a6, %AddWith.facet.704 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.specific_fn.09578f.1: <specific function> = specific_function %T.binding.as_type.as.AddWith.impl.Op.3fccd2.2, @T.binding.as_type.as.AddWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.specific_fn.09578f.2: <specific function> = specific_function %T.binding.as_type.as.AddWith.impl.Op.3fccd2.1, @T.binding.as_type.as.AddWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.c45: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %AssertSameType.specific_fn: <specific function> = specific_function %AssertSameType, @AssertSameType(%Cpp.long) [concrete]
// CHECK:STDOUT:   %SubWith.type.e8e: type = facet_type <@SubWith, @SubWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %SubWith.Op.type.533: type = fn_type @SubWith.Op, @SubWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.type.d77f83.1: type = fn_type @T.binding.as_type.as.SubWith.impl.Op.1, @T.binding.as_type.as.SubWith.impl.e1e(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.600de1.1: %T.binding.as_type.as.SubWith.impl.Op.type.d77f83.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.type.d77f83.2: type = fn_type @T.binding.as_type.as.SubWith.impl.Op.2, @T.binding.as_type.as.SubWith.impl.e1e(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.600de1.2: %T.binding.as_type.as.SubWith.impl.Op.type.d77f83.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.impl_witness.702: <witness> = impl_witness imports.%SubWith.impl_witness_table.125, @T.binding.as_type.as.SubWith.impl.e1e(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.type.caca02.1: type = fn_type @T.binding.as_type.as.SubWith.impl.Op.2, @T.binding.as_type.as.SubWith.impl.e1e(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.f9b5c3.1: %T.binding.as_type.as.SubWith.impl.Op.type.caca02.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.type.caca02.2: type = fn_type @T.binding.as_type.as.SubWith.impl.Op.1, @T.binding.as_type.as.SubWith.impl.e1e(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.f9b5c3.2: %T.binding.as_type.as.SubWith.impl.Op.type.caca02.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.facet.6af: %SubWith.type.e8e = facet_value %i32, (%SubWith.impl_witness.702) [concrete]
// CHECK:STDOUT:   %.040: type = fn_type_with_self_type %SubWith.Op.type.533, %SubWith.facet.6af [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.specific_fn.b0f48e.1: <specific function> = specific_function %T.binding.as_type.as.SubWith.impl.Op.f9b5c3.2, @T.binding.as_type.as.SubWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.specific_fn.b0f48e.2: <specific function> = specific_function %T.binding.as_type.as.SubWith.impl.Op.f9b5c3.1, @T.binding.as_type.as.SubWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %MulWith.type.a32: type = facet_type <@MulWith, @MulWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %MulWith.Op.type.44b: type = fn_type @MulWith.Op, @MulWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.type.20586f.1: type = fn_type @T.binding.as_type.as.MulWith.impl.Op.1, @T.binding.as_type.as.MulWith.impl.6d9(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.036dac.1: %T.binding.as_type.as.MulWith.impl.Op.type.20586f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.type.20586f.2: type = fn_type @T.binding.as_type.as.MulWith.impl.Op.2, @T.binding.as_type.as.MulWith.impl.6d9(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.036dac.2: %T.binding.as_type.as.MulWith.impl.Op.type.20586f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.impl_witness.f85: <witness> = impl_witness imports.%MulWith.impl_witness_table.ef7, @T.binding.as_type.as.MulWith.impl.6d9(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.type.1189b3.1: type = fn_type @T.binding.as_type.as.MulWith.impl.Op.2, @T.binding.as_type.as.MulWith.impl.6d9(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.461492.1: %T.binding.as_type.as.MulWith.impl.Op.type.1189b3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.type.1189b3.2: type = fn_type @T.binding.as_type.as.MulWith.impl.Op.1, @T.binding.as_type.as.MulWith.impl.6d9(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.461492.2: %T.binding.as_type.as.MulWith.impl.Op.type.1189b3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.facet.b19: %MulWith.type.a32 = facet_value %i32, (%MulWith.impl_witness.f85) [concrete]
// CHECK:STDOUT:   %.1b4: type = fn_type_with_self_type %MulWith.Op.type.44b, %MulWith.facet.b19 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.specific_fn.475813.1: <specific function> = specific_function %T.binding.as_type.as.MulWith.impl.Op.461492.2, @T.binding.as_type.as.MulWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.specific_fn.475813.2: <specific function> = specific_function %T.binding.as_type.as.MulWith.impl.Op.461492.1, @T.binding.as_type.as.MulWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %DivWith.type.0c0: type = facet_type <@DivWith, @DivWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %DivWith.Op.type.494: type = fn_type @DivWith.Op, @DivWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.type.c7967e.1: type = fn_type @T.binding.as_type.as.DivWith.impl.Op.1, @T.binding.as_type.as.DivWith.impl.67f(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.f3fb3f.1: %T.binding.as_type.as.DivWith.impl.Op.type.c7967e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.type.c7967e.2: type = fn_type @T.binding.as_type.as.DivWith.impl.Op.2, @T.binding.as_type.as.DivWith.impl.67f(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.f3fb3f.2: %T.binding.as_type.as.DivWith.impl.Op.type.c7967e.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.impl_witness.996: <witness> = impl_witness imports.%DivWith.impl_witness_table.4bd, @T.binding.as_type.as.DivWith.impl.67f(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.type.bc0f3d.1: type = fn_type @T.binding.as_type.as.DivWith.impl.Op.2, @T.binding.as_type.as.DivWith.impl.67f(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.e5dcee.1: %T.binding.as_type.as.DivWith.impl.Op.type.bc0f3d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.type.bc0f3d.2: type = fn_type @T.binding.as_type.as.DivWith.impl.Op.1, @T.binding.as_type.as.DivWith.impl.67f(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.e5dcee.2: %T.binding.as_type.as.DivWith.impl.Op.type.bc0f3d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.facet.fbd: %DivWith.type.0c0 = facet_value %i32, (%DivWith.impl_witness.996) [concrete]
// CHECK:STDOUT:   %.0f3: type = fn_type_with_self_type %DivWith.Op.type.494, %DivWith.facet.fbd [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.specific_fn.8f9bff.1: <specific function> = specific_function %T.binding.as_type.as.DivWith.impl.Op.e5dcee.2, @T.binding.as_type.as.DivWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.specific_fn.8f9bff.2: <specific function> = specific_function %T.binding.as_type.as.DivWith.impl.Op.e5dcee.1, @T.binding.as_type.as.DivWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ModWith.type.2fc: type = facet_type <@ModWith, @ModWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ModWith.Op.type.c13: type = fn_type @ModWith.Op, @ModWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.type.302b2e.1: type = fn_type @T.binding.as_type.as.ModWith.impl.Op.1, @T.binding.as_type.as.ModWith.impl.5ca(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.02226a.1: %T.binding.as_type.as.ModWith.impl.Op.type.302b2e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.type.302b2e.2: type = fn_type @T.binding.as_type.as.ModWith.impl.Op.2, @T.binding.as_type.as.ModWith.impl.5ca(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.02226a.2: %T.binding.as_type.as.ModWith.impl.Op.type.302b2e.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.impl_witness.e1d: <witness> = impl_witness imports.%ModWith.impl_witness_table.644, @T.binding.as_type.as.ModWith.impl.5ca(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.type.9786fd.1: type = fn_type @T.binding.as_type.as.ModWith.impl.Op.2, @T.binding.as_type.as.ModWith.impl.5ca(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.c5465f.1: %T.binding.as_type.as.ModWith.impl.Op.type.9786fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.type.9786fd.2: type = fn_type @T.binding.as_type.as.ModWith.impl.Op.1, @T.binding.as_type.as.ModWith.impl.5ca(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.c5465f.2: %T.binding.as_type.as.ModWith.impl.Op.type.9786fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.facet.cf9: %ModWith.type.2fc = facet_value %i32, (%ModWith.impl_witness.e1d) [concrete]
// CHECK:STDOUT:   %.e68: type = fn_type_with_self_type %ModWith.Op.type.c13, %ModWith.facet.cf9 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.specific_fn.ee32ad.1: <specific function> = specific_function %T.binding.as_type.as.ModWith.impl.Op.c5465f.2, @T.binding.as_type.as.ModWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.specific_fn.ee32ad.2: <specific function> = specific_function %T.binding.as_type.as.ModWith.impl.Op.c5465f.1, @T.binding.as_type.as.ModWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.740: <witness> = impl_witness imports.%AddWith.impl_witness_table.0dc, @T.binding.as_type.as.AddWith.impl.f1a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.type.75226a.1: type = fn_type @T.binding.as_type.as.AddWith.impl.Op.2, @T.binding.as_type.as.AddWith.impl.f1a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.b037a1.1: %T.binding.as_type.as.AddWith.impl.Op.type.75226a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.type.75226a.2: type = fn_type @T.binding.as_type.as.AddWith.impl.Op.1, @T.binding.as_type.as.AddWith.impl.f1a(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.b037a1.2: %T.binding.as_type.as.AddWith.impl.Op.type.75226a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet.07d: %AddWith.type.9c2 = facet_value Core.IntLiteral, (%AddWith.impl_witness.740) [concrete]
// CHECK:STDOUT:   %.26f: type = fn_type_with_self_type %AddWith.Op.type.9a6, %AddWith.facet.07d [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.bound.083bd2.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.AddWith.impl.Op.b037a1.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.specific_fn.6a47c4.1: <specific function> = specific_function %T.binding.as_type.as.AddWith.impl.Op.b037a1.2, @T.binding.as_type.as.AddWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.76c7cf.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.AddWith.impl.Op.specific_fn.6a47c4.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.bound.083bd2.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.AddWith.impl.Op.b037a1.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.specific_fn.6a47c4.2: <specific function> = specific_function %T.binding.as_type.as.AddWith.impl.Op.b037a1.1, @T.binding.as_type.as.AddWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.76c7cf.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.AddWith.impl.Op.specific_fn.6a47c4.2 [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness.980: <witness> = impl_witness imports.%SubWith.impl_witness_table.125, @T.binding.as_type.as.SubWith.impl.e1e(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.type.8ada84.1: type = fn_type @T.binding.as_type.as.SubWith.impl.Op.2, @T.binding.as_type.as.SubWith.impl.e1e(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.d20ebd.1: %T.binding.as_type.as.SubWith.impl.Op.type.8ada84.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.type.8ada84.2: type = fn_type @T.binding.as_type.as.SubWith.impl.Op.1, @T.binding.as_type.as.SubWith.impl.e1e(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.d20ebd.2: %T.binding.as_type.as.SubWith.impl.Op.type.8ada84.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.facet.dc9: %SubWith.type.e8e = facet_value Core.IntLiteral, (%SubWith.impl_witness.980) [concrete]
// CHECK:STDOUT:   %.4fb: type = fn_type_with_self_type %SubWith.Op.type.533, %SubWith.facet.dc9 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.bound.09e45c.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.SubWith.impl.Op.d20ebd.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.specific_fn.8885f5.1: <specific function> = specific_function %T.binding.as_type.as.SubWith.impl.Op.d20ebd.2, @T.binding.as_type.as.SubWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.bedc54.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.SubWith.impl.Op.specific_fn.8885f5.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.bound.09e45c.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.SubWith.impl.Op.d20ebd.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.specific_fn.8885f5.2: <specific function> = specific_function %T.binding.as_type.as.SubWith.impl.Op.d20ebd.1, @T.binding.as_type.as.SubWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.bedc54.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.SubWith.impl.Op.specific_fn.8885f5.2 [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness.824: <witness> = impl_witness imports.%MulWith.impl_witness_table.ef7, @T.binding.as_type.as.MulWith.impl.6d9(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.type.2affc8.1: type = fn_type @T.binding.as_type.as.MulWith.impl.Op.2, @T.binding.as_type.as.MulWith.impl.6d9(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.075f45.1: %T.binding.as_type.as.MulWith.impl.Op.type.2affc8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.type.2affc8.2: type = fn_type @T.binding.as_type.as.MulWith.impl.Op.1, @T.binding.as_type.as.MulWith.impl.6d9(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.075f45.2: %T.binding.as_type.as.MulWith.impl.Op.type.2affc8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.facet.603: %MulWith.type.a32 = facet_value Core.IntLiteral, (%MulWith.impl_witness.824) [concrete]
// CHECK:STDOUT:   %.da9: type = fn_type_with_self_type %MulWith.Op.type.44b, %MulWith.facet.603 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.bound.9ab9cd.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.MulWith.impl.Op.075f45.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.specific_fn.785270.1: <specific function> = specific_function %T.binding.as_type.as.MulWith.impl.Op.075f45.2, @T.binding.as_type.as.MulWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.82b41e.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.MulWith.impl.Op.specific_fn.785270.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.bound.9ab9cd.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.MulWith.impl.Op.075f45.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.specific_fn.785270.2: <specific function> = specific_function %T.binding.as_type.as.MulWith.impl.Op.075f45.1, @T.binding.as_type.as.MulWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.82b41e.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.MulWith.impl.Op.specific_fn.785270.2 [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness.fea: <witness> = impl_witness imports.%DivWith.impl_witness_table.4bd, @T.binding.as_type.as.DivWith.impl.67f(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.type.b86756.1: type = fn_type @T.binding.as_type.as.DivWith.impl.Op.2, @T.binding.as_type.as.DivWith.impl.67f(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.637574.1: %T.binding.as_type.as.DivWith.impl.Op.type.b86756.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.type.b86756.2: type = fn_type @T.binding.as_type.as.DivWith.impl.Op.1, @T.binding.as_type.as.DivWith.impl.67f(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.637574.2: %T.binding.as_type.as.DivWith.impl.Op.type.b86756.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.facet.5a3: %DivWith.type.0c0 = facet_value Core.IntLiteral, (%DivWith.impl_witness.fea) [concrete]
// CHECK:STDOUT:   %.0bb: type = fn_type_with_self_type %DivWith.Op.type.494, %DivWith.facet.5a3 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.bound.53c922.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.DivWith.impl.Op.637574.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.specific_fn.332c8a.1: <specific function> = specific_function %T.binding.as_type.as.DivWith.impl.Op.637574.2, @T.binding.as_type.as.DivWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.7598a7.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.DivWith.impl.Op.specific_fn.332c8a.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.bound.53c922.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.DivWith.impl.Op.637574.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.specific_fn.332c8a.2: <specific function> = specific_function %T.binding.as_type.as.DivWith.impl.Op.637574.1, @T.binding.as_type.as.DivWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.7598a7.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.DivWith.impl.Op.specific_fn.332c8a.2 [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness.770: <witness> = impl_witness imports.%ModWith.impl_witness_table.644, @T.binding.as_type.as.ModWith.impl.5ca(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.type.84d6b7.1: type = fn_type @T.binding.as_type.as.ModWith.impl.Op.2, @T.binding.as_type.as.ModWith.impl.5ca(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.5c95af.1: %T.binding.as_type.as.ModWith.impl.Op.type.84d6b7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.type.84d6b7.2: type = fn_type @T.binding.as_type.as.ModWith.impl.Op.1, @T.binding.as_type.as.ModWith.impl.5ca(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.5c95af.2: %T.binding.as_type.as.ModWith.impl.Op.type.84d6b7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.facet.ef7: %ModWith.type.2fc = facet_value Core.IntLiteral, (%ModWith.impl_witness.770) [concrete]
// CHECK:STDOUT:   %.930: type = fn_type_with_self_type %ModWith.Op.type.c13, %ModWith.facet.ef7 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.bound.468823.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.ModWith.impl.Op.5c95af.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.specific_fn.4b7f8c.1: <specific function> = specific_function %T.binding.as_type.as.ModWith.impl.Op.5c95af.2, @T.binding.as_type.as.ModWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.83c96c.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.ModWith.impl.Op.specific_fn.4b7f8c.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.bound.468823.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.ModWith.impl.Op.5c95af.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.specific_fn.4b7f8c.2: <specific function> = specific_function %T.binding.as_type.as.ModWith.impl.Op.5c95af.1, @T.binding.as_type.as.ModWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.83c96c.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.ModWith.impl.Op.specific_fn.4b7f8c.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl.b2d [concrete]
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.2 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6ee: @T.binding.as_type.as.AddWith.impl.f1a.%T.binding.as_type.as.AddWith.impl.Op.type.2 (%T.binding.as_type.as.AddWith.impl.Op.type.704168.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.AddWith.impl.f1a.%T.binding.as_type.as.AddWith.impl.Op.2 (constants.%T.binding.as_type.as.AddWith.impl.Op.399102.1)]
// CHECK:STDOUT:   %AddWith.impl_witness_table.0dc = impl_witness_table (%Core.import_ref.cb912f.2, %Core.import_ref.6ee), @T.binding.as_type.as.AddWith.impl.f1a [concrete]
// CHECK:STDOUT:   %Core.Op.af2: @T.binding.as_type.as.AddWith.impl.f1a.%T.binding.as_type.as.AddWith.impl.Op.type.1 (%T.binding.as_type.as.AddWith.impl.Op.type.704168.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @T.binding.as_type.as.AddWith.impl.f1a.%T.binding.as_type.as.AddWith.impl.Op.1 (constants.%T.binding.as_type.as.AddWith.impl.Op.399102.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.4 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.c70: @T.binding.as_type.as.SubWith.impl.e1e.%T.binding.as_type.as.SubWith.impl.Op.type.2 (%T.binding.as_type.as.SubWith.impl.Op.type.d77f83.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.SubWith.impl.e1e.%T.binding.as_type.as.SubWith.impl.Op.2 (constants.%T.binding.as_type.as.SubWith.impl.Op.600de1.1)]
// CHECK:STDOUT:   %SubWith.impl_witness_table.125 = impl_witness_table (%Core.import_ref.cb912f.4, %Core.import_ref.c70), @T.binding.as_type.as.SubWith.impl.e1e [concrete]
// CHECK:STDOUT:   %Core.Op.fcc: @T.binding.as_type.as.SubWith.impl.e1e.%T.binding.as_type.as.SubWith.impl.Op.type.1 (%T.binding.as_type.as.SubWith.impl.Op.type.d77f83.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @T.binding.as_type.as.SubWith.impl.e1e.%T.binding.as_type.as.SubWith.impl.Op.1 (constants.%T.binding.as_type.as.SubWith.impl.Op.600de1.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.7df: @T.binding.as_type.as.MulWith.impl.6d9.%T.binding.as_type.as.MulWith.impl.Op.type.2 (%T.binding.as_type.as.MulWith.impl.Op.type.20586f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.MulWith.impl.6d9.%T.binding.as_type.as.MulWith.impl.Op.2 (constants.%T.binding.as_type.as.MulWith.impl.Op.036dac.1)]
// CHECK:STDOUT:   %MulWith.impl_witness_table.ef7 = impl_witness_table (%Core.import_ref.cb912f.6, %Core.import_ref.7df), @T.binding.as_type.as.MulWith.impl.6d9 [concrete]
// CHECK:STDOUT:   %Core.Op.6a1: @T.binding.as_type.as.MulWith.impl.6d9.%T.binding.as_type.as.MulWith.impl.Op.type.1 (%T.binding.as_type.as.MulWith.impl.Op.type.20586f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @T.binding.as_type.as.MulWith.impl.6d9.%T.binding.as_type.as.MulWith.impl.Op.1 (constants.%T.binding.as_type.as.MulWith.impl.Op.036dac.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.8 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.425: @T.binding.as_type.as.DivWith.impl.67f.%T.binding.as_type.as.DivWith.impl.Op.type.2 (%T.binding.as_type.as.DivWith.impl.Op.type.c7967e.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.DivWith.impl.67f.%T.binding.as_type.as.DivWith.impl.Op.2 (constants.%T.binding.as_type.as.DivWith.impl.Op.f3fb3f.1)]
// CHECK:STDOUT:   %DivWith.impl_witness_table.4bd = impl_witness_table (%Core.import_ref.cb912f.8, %Core.import_ref.425), @T.binding.as_type.as.DivWith.impl.67f [concrete]
// CHECK:STDOUT:   %Core.Op.ace: @T.binding.as_type.as.DivWith.impl.67f.%T.binding.as_type.as.DivWith.impl.Op.type.1 (%T.binding.as_type.as.DivWith.impl.Op.type.c7967e.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @T.binding.as_type.as.DivWith.impl.67f.%T.binding.as_type.as.DivWith.impl.Op.1 (constants.%T.binding.as_type.as.DivWith.impl.Op.f3fb3f.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.10 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0d4: @T.binding.as_type.as.ModWith.impl.5ca.%T.binding.as_type.as.ModWith.impl.Op.type.2 (%T.binding.as_type.as.ModWith.impl.Op.type.302b2e.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.ModWith.impl.5ca.%T.binding.as_type.as.ModWith.impl.Op.2 (constants.%T.binding.as_type.as.ModWith.impl.Op.02226a.1)]
// CHECK:STDOUT:   %ModWith.impl_witness_table.644 = impl_witness_table (%Core.import_ref.cb912f.10, %Core.import_ref.0d4), @T.binding.as_type.as.ModWith.impl.5ca [concrete]
// CHECK:STDOUT:   %Core.Op.c8c: @T.binding.as_type.as.ModWith.impl.5ca.%T.binding.as_type.as.ModWith.impl.Op.type.1 (%T.binding.as_type.as.ModWith.impl.Op.type.302b2e.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @T.binding.as_type.as.ModWith.impl.5ca.%T.binding.as_type.as.ModWith.impl.Op.1 (constants.%T.binding.as_type.as.ModWith.impl.Op.02226a.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ArithmeticHeterogeneousLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.68c = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc11: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11(%int_1.loc11) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc11_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc11_21.2: %Cpp.long = converted %int_1.loc11, %.loc11_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %x: %Cpp.long = value_binding x, %.loc11_21.2
// CHECK:STDOUT:   %AssertSameType.ref.loc12: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %x.ref.loc12_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc12: %.285 = impl_witness_access constants.%AddWith.impl_witness.a1d, element1 [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.3fccd2.2]
// CHECK:STDOUT:   %bound_method.loc12_20.1: <bound method> = bound_method %b.ref.loc12, %impl.elem1.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem1.loc12, @T.binding.as_type.as.AddWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.specific_fn.09578f.1]
// CHECK:STDOUT:   %bound_method.loc12_20.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_20.1: %T.binding.as_type.as.AddWith.impl.Op.type.5105fb.1 = specific_constant imports.%Core.Op.af2, @T.binding.as_type.as.AddWith.impl.f1a(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.3fccd2.1]
// CHECK:STDOUT:   %Op.ref.loc12: %T.binding.as_type.as.AddWith.impl.Op.type.5105fb.1 = name_ref Op, %.loc12_20.1 [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.3fccd2.1]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.specific_fn.loc12: <specific function> = specific_function %Op.ref.loc12, @T.binding.as_type.as.AddWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.specific_fn.09578f.2]
// CHECK:STDOUT:   %bound_method.loc12_20.3: <bound method> = bound_method %b.ref.loc12, %T.binding.as_type.as.AddWith.impl.Op.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_18: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12_18(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_18.2: %Cpp.long = converted %b.ref.loc12, %.loc12_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.call.loc12: init %Cpp.long = call %bound_method.loc12_20.3(%.loc12_18.2, %x.ref.loc12_22)
// CHECK:STDOUT:   %x.ref.loc12_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc12: <specific function> = specific_function %AssertSameType.ref.loc12, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc12_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.AddWith.impl.Op.call.loc12
// CHECK:STDOUT:   %.loc12_20.3: %Cpp.long = converted %T.binding.as_type.as.AddWith.impl.Op.call.loc12, %.loc12_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc12: init %empty_tuple.type = call %AssertSameType.specific_fn.loc12(%.loc12_20.3, %x.ref.loc12_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc13: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %x.ref.loc13_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc13: %.040 = impl_witness_access constants.%SubWith.impl_witness.702, element1 [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.f9b5c3.2]
// CHECK:STDOUT:   %bound_method.loc13_20.1: <bound method> = bound_method %b.ref.loc13, %impl.elem1.loc13
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem1.loc13, @T.binding.as_type.as.SubWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.specific_fn.b0f48e.1]
// CHECK:STDOUT:   %bound_method.loc13_20.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_20.1: %T.binding.as_type.as.SubWith.impl.Op.type.caca02.1 = specific_constant imports.%Core.Op.fcc, @T.binding.as_type.as.SubWith.impl.e1e(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.f9b5c3.1]
// CHECK:STDOUT:   %Op.ref.loc13: %T.binding.as_type.as.SubWith.impl.Op.type.caca02.1 = name_ref Op, %.loc13_20.1 [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.f9b5c3.1]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.specific_fn.loc13: <specific function> = specific_function %Op.ref.loc13, @T.binding.as_type.as.SubWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.specific_fn.b0f48e.2]
// CHECK:STDOUT:   %bound_method.loc13_20.3: <bound method> = bound_method %b.ref.loc13, %T.binding.as_type.as.SubWith.impl.Op.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_18: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long = call %bound_method.loc13_18(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_18.2: %Cpp.long = converted %b.ref.loc13, %.loc13_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.call.loc13: init %Cpp.long = call %bound_method.loc13_20.3(%.loc13_18.2, %x.ref.loc13_22)
// CHECK:STDOUT:   %x.ref.loc13_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc13: <specific function> = specific_function %AssertSameType.ref.loc13, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc13_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.SubWith.impl.Op.call.loc13
// CHECK:STDOUT:   %.loc13_20.3: %Cpp.long = converted %T.binding.as_type.as.SubWith.impl.Op.call.loc13, %.loc13_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc13: init %empty_tuple.type = call %AssertSameType.specific_fn.loc13(%.loc13_20.3, %x.ref.loc13_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc14: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %x.ref.loc14_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc14: %.1b4 = impl_witness_access constants.%MulWith.impl_witness.f85, element1 [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.461492.2]
// CHECK:STDOUT:   %bound_method.loc14_20.1: <bound method> = bound_method %b.ref.loc14, %impl.elem1.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem1.loc14, @T.binding.as_type.as.MulWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.specific_fn.475813.1]
// CHECK:STDOUT:   %bound_method.loc14_20.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_20.1: %T.binding.as_type.as.MulWith.impl.Op.type.1189b3.1 = specific_constant imports.%Core.Op.6a1, @T.binding.as_type.as.MulWith.impl.6d9(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.461492.1]
// CHECK:STDOUT:   %Op.ref.loc14: %T.binding.as_type.as.MulWith.impl.Op.type.1189b3.1 = name_ref Op, %.loc14_20.1 [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.461492.1]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.bound.loc14: <bound method> = bound_method %b.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.specific_fn.loc14: <specific function> = specific_function %Op.ref.loc14, @T.binding.as_type.as.MulWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.specific_fn.475813.2]
// CHECK:STDOUT:   %bound_method.loc14_20.3: <bound method> = bound_method %b.ref.loc14, %T.binding.as_type.as.MulWith.impl.Op.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_18: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long = call %bound_method.loc14_18(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_18.2: %Cpp.long = converted %b.ref.loc14, %.loc14_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.call.loc14: init %Cpp.long = call %bound_method.loc14_20.3(%.loc14_18.2, %x.ref.loc14_22)
// CHECK:STDOUT:   %x.ref.loc14_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc14: <specific function> = specific_function %AssertSameType.ref.loc14, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc14_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.MulWith.impl.Op.call.loc14
// CHECK:STDOUT:   %.loc14_20.3: %Cpp.long = converted %T.binding.as_type.as.MulWith.impl.Op.call.loc14, %.loc14_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc14: init %empty_tuple.type = call %AssertSameType.specific_fn.loc14(%.loc14_20.3, %x.ref.loc14_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc15: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %x.ref.loc15_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc15: %.0f3 = impl_witness_access constants.%DivWith.impl_witness.996, element1 [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.e5dcee.2]
// CHECK:STDOUT:   %bound_method.loc15_20.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.binding.as_type.as.DivWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.specific_fn.8f9bff.1]
// CHECK:STDOUT:   %bound_method.loc15_20.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_20.1: %T.binding.as_type.as.DivWith.impl.Op.type.bc0f3d.1 = specific_constant imports.%Core.Op.ace, @T.binding.as_type.as.DivWith.impl.67f(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.e5dcee.1]
// CHECK:STDOUT:   %Op.ref.loc15: %T.binding.as_type.as.DivWith.impl.Op.type.bc0f3d.1 = name_ref Op, %.loc15_20.1 [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.e5dcee.1]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.bound.loc15: <bound method> = bound_method %b.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.specific_fn.loc15: <specific function> = specific_function %Op.ref.loc15, @T.binding.as_type.as.DivWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.specific_fn.8f9bff.2]
// CHECK:STDOUT:   %bound_method.loc15_20.3: <bound method> = bound_method %b.ref.loc15, %T.binding.as_type.as.DivWith.impl.Op.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_18: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long = call %bound_method.loc15_18(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_18.2: %Cpp.long = converted %b.ref.loc15, %.loc15_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.call.loc15: init %Cpp.long = call %bound_method.loc15_20.3(%.loc15_18.2, %x.ref.loc15_22)
// CHECK:STDOUT:   %x.ref.loc15_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc15: <specific function> = specific_function %AssertSameType.ref.loc15, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc15_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.DivWith.impl.Op.call.loc15
// CHECK:STDOUT:   %.loc15_20.3: %Cpp.long = converted %T.binding.as_type.as.DivWith.impl.Op.call.loc15, %.loc15_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc15: init %empty_tuple.type = call %AssertSameType.specific_fn.loc15(%.loc15_20.3, %x.ref.loc15_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc16: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc16: %i32 = name_ref b, %b
// CHECK:STDOUT:   %x.ref.loc16_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc16: %.e68 = impl_witness_access constants.%ModWith.impl_witness.e1d, element1 [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.c5465f.2]
// CHECK:STDOUT:   %bound_method.loc16_20.1: <bound method> = bound_method %b.ref.loc16, %impl.elem1.loc16
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem1.loc16, @T.binding.as_type.as.ModWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.specific_fn.ee32ad.1]
// CHECK:STDOUT:   %bound_method.loc16_20.2: <bound method> = bound_method %b.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_20.1: %T.binding.as_type.as.ModWith.impl.Op.type.9786fd.1 = specific_constant imports.%Core.Op.c8c, @T.binding.as_type.as.ModWith.impl.5ca(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.c5465f.1]
// CHECK:STDOUT:   %Op.ref.loc16: %T.binding.as_type.as.ModWith.impl.Op.type.9786fd.1 = name_ref Op, %.loc16_20.1 [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.c5465f.1]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.bound.loc16: <bound method> = bound_method %b.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.specific_fn.loc16: <specific function> = specific_function %Op.ref.loc16, @T.binding.as_type.as.ModWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.specific_fn.ee32ad.2]
// CHECK:STDOUT:   %bound_method.loc16_20.3: <bound method> = bound_method %b.ref.loc16, %T.binding.as_type.as.ModWith.impl.Op.specific_fn.loc16
// CHECK:STDOUT:   %impl.elem0.loc16: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_18: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16: init %Cpp.long = call %bound_method.loc16_18(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16
// CHECK:STDOUT:   %.loc16_18.2: %Cpp.long = converted %b.ref.loc16, %.loc16_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.call.loc16: init %Cpp.long = call %bound_method.loc16_20.3(%.loc16_18.2, %x.ref.loc16_22)
// CHECK:STDOUT:   %x.ref.loc16_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc16: <specific function> = specific_function %AssertSameType.ref.loc16, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc16_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.ModWith.impl.Op.call.loc16
// CHECK:STDOUT:   %.loc16_20.3: %Cpp.long = converted %T.binding.as_type.as.ModWith.impl.Op.call.loc16, %.loc16_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc16: init %empty_tuple.type = call %AssertSameType.specific_fn.loc16(%.loc16_20.3, %x.ref.loc16_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc18: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %x.ref.loc18_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc18: %.26f = impl_witness_access constants.%AddWith.impl_witness.740, element1 [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.b037a1.2]
// CHECK:STDOUT:   %bound_method.loc18_20.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.bound.083bd2.1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.binding.as_type.as.AddWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.specific_fn.6a47c4.1]
// CHECK:STDOUT:   %bound_method.loc18_20.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.76c7cf.1]
// CHECK:STDOUT:   %.loc18_20.1: %T.binding.as_type.as.AddWith.impl.Op.type.75226a.1 = specific_constant imports.%Core.Op.af2, @T.binding.as_type.as.AddWith.impl.f1a(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.b037a1.1]
// CHECK:STDOUT:   %Op.ref.loc18: %T.binding.as_type.as.AddWith.impl.Op.type.75226a.1 = name_ref Op, %.loc18_20.1 [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.b037a1.1]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.bound.loc18: <bound method> = bound_method %int_1.loc18, %Op.ref.loc18 [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.bound.083bd2.2]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.specific_fn.loc18: <specific function> = specific_function %Op.ref.loc18, @T.binding.as_type.as.AddWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.specific_fn.6a47c4.2]
// CHECK:STDOUT:   %bound_method.loc18_20.3: <bound method> = bound_method %int_1.loc18, %T.binding.as_type.as.AddWith.impl.Op.specific_fn.loc18 [concrete = constants.%bound_method.76c7cf.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc18_18: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_18(%int_1.loc18) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_18.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_18.2: %Cpp.long = converted %int_1.loc18, %.loc18_18.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.call.loc18: init %Cpp.long = call %bound_method.loc18_20.3(%.loc18_18.2, %x.ref.loc18_22)
// CHECK:STDOUT:   %x.ref.loc18_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc18: <specific function> = specific_function %AssertSameType.ref.loc18, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc18_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.AddWith.impl.Op.call.loc18
// CHECK:STDOUT:   %.loc18_20.3: %Cpp.long = converted %T.binding.as_type.as.AddWith.impl.Op.call.loc18, %.loc18_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc18: init %empty_tuple.type = call %AssertSameType.specific_fn.loc18(%.loc18_20.3, %x.ref.loc18_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc19: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %x.ref.loc19_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc19: %.4fb = impl_witness_access constants.%SubWith.impl_witness.980, element1 [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.d20ebd.2]
// CHECK:STDOUT:   %bound_method.loc19_20.1: <bound method> = bound_method %int_1.loc19, %impl.elem1.loc19 [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.bound.09e45c.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem1.loc19, @T.binding.as_type.as.SubWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.specific_fn.8885f5.1]
// CHECK:STDOUT:   %bound_method.loc19_20.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.bedc54.1]
// CHECK:STDOUT:   %.loc19_20.1: %T.binding.as_type.as.SubWith.impl.Op.type.8ada84.1 = specific_constant imports.%Core.Op.fcc, @T.binding.as_type.as.SubWith.impl.e1e(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.d20ebd.1]
// CHECK:STDOUT:   %Op.ref.loc19: %T.binding.as_type.as.SubWith.impl.Op.type.8ada84.1 = name_ref Op, %.loc19_20.1 [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.d20ebd.1]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.bound.loc19: <bound method> = bound_method %int_1.loc19, %Op.ref.loc19 [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.bound.09e45c.2]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.specific_fn.loc19: <specific function> = specific_function %Op.ref.loc19, @T.binding.as_type.as.SubWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.specific_fn.8885f5.2]
// CHECK:STDOUT:   %bound_method.loc19_20.3: <bound method> = bound_method %int_1.loc19, %T.binding.as_type.as.SubWith.impl.Op.specific_fn.loc19 [concrete = constants.%bound_method.bedc54.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_18: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_18(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_18.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_18.2: %Cpp.long = converted %int_1.loc19, %.loc19_18.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.call.loc19: init %Cpp.long = call %bound_method.loc19_20.3(%.loc19_18.2, %x.ref.loc19_22)
// CHECK:STDOUT:   %x.ref.loc19_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc19: <specific function> = specific_function %AssertSameType.ref.loc19, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc19_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.SubWith.impl.Op.call.loc19
// CHECK:STDOUT:   %.loc19_20.3: %Cpp.long = converted %T.binding.as_type.as.SubWith.impl.Op.call.loc19, %.loc19_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc19: init %empty_tuple.type = call %AssertSameType.specific_fn.loc19(%.loc19_20.3, %x.ref.loc19_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc20: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %x.ref.loc20_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc20: %.da9 = impl_witness_access constants.%MulWith.impl_witness.824, element1 [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.075f45.2]
// CHECK:STDOUT:   %bound_method.loc20_20.1: <bound method> = bound_method %int_1.loc20, %impl.elem1.loc20 [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.bound.9ab9cd.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem1.loc20, @T.binding.as_type.as.MulWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.specific_fn.785270.1]
// CHECK:STDOUT:   %bound_method.loc20_20.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.82b41e.1]
// CHECK:STDOUT:   %.loc20_20.1: %T.binding.as_type.as.MulWith.impl.Op.type.2affc8.1 = specific_constant imports.%Core.Op.6a1, @T.binding.as_type.as.MulWith.impl.6d9(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.075f45.1]
// CHECK:STDOUT:   %Op.ref.loc20: %T.binding.as_type.as.MulWith.impl.Op.type.2affc8.1 = name_ref Op, %.loc20_20.1 [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.075f45.1]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.bound.loc20: <bound method> = bound_method %int_1.loc20, %Op.ref.loc20 [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.bound.9ab9cd.2]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.specific_fn.loc20: <specific function> = specific_function %Op.ref.loc20, @T.binding.as_type.as.MulWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.specific_fn.785270.2]
// CHECK:STDOUT:   %bound_method.loc20_20.3: <bound method> = bound_method %int_1.loc20, %T.binding.as_type.as.MulWith.impl.Op.specific_fn.loc20 [concrete = constants.%bound_method.82b41e.2]
// CHECK:STDOUT:   %impl.elem0.loc20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_18: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_18(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_18.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_18.2: %Cpp.long = converted %int_1.loc20, %.loc20_18.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.call.loc20: init %Cpp.long = call %bound_method.loc20_20.3(%.loc20_18.2, %x.ref.loc20_22)
// CHECK:STDOUT:   %x.ref.loc20_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc20: <specific function> = specific_function %AssertSameType.ref.loc20, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc20_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.MulWith.impl.Op.call.loc20
// CHECK:STDOUT:   %.loc20_20.3: %Cpp.long = converted %T.binding.as_type.as.MulWith.impl.Op.call.loc20, %.loc20_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc20: init %empty_tuple.type = call %AssertSameType.specific_fn.loc20(%.loc20_20.3, %x.ref.loc20_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc21: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %x.ref.loc21_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc21: %.0bb = impl_witness_access constants.%DivWith.impl_witness.fea, element1 [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.637574.2]
// CHECK:STDOUT:   %bound_method.loc21_20.1: <bound method> = bound_method %int_1.loc21, %impl.elem1.loc21 [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.bound.53c922.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem1.loc21, @T.binding.as_type.as.DivWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.specific_fn.332c8a.1]
// CHECK:STDOUT:   %bound_method.loc21_20.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.7598a7.1]
// CHECK:STDOUT:   %.loc21_20.1: %T.binding.as_type.as.DivWith.impl.Op.type.b86756.1 = specific_constant imports.%Core.Op.ace, @T.binding.as_type.as.DivWith.impl.67f(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.637574.1]
// CHECK:STDOUT:   %Op.ref.loc21: %T.binding.as_type.as.DivWith.impl.Op.type.b86756.1 = name_ref Op, %.loc21_20.1 [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.637574.1]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.bound.loc21: <bound method> = bound_method %int_1.loc21, %Op.ref.loc21 [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.bound.53c922.2]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.specific_fn.loc21: <specific function> = specific_function %Op.ref.loc21, @T.binding.as_type.as.DivWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.specific_fn.332c8a.2]
// CHECK:STDOUT:   %bound_method.loc21_20.3: <bound method> = bound_method %int_1.loc21, %T.binding.as_type.as.DivWith.impl.Op.specific_fn.loc21 [concrete = constants.%bound_method.7598a7.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_18: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_18(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_18.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_18.2: %Cpp.long = converted %int_1.loc21, %.loc21_18.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.call.loc21: init %Cpp.long = call %bound_method.loc21_20.3(%.loc21_18.2, %x.ref.loc21_22)
// CHECK:STDOUT:   %x.ref.loc21_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc21: <specific function> = specific_function %AssertSameType.ref.loc21, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc21_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.DivWith.impl.Op.call.loc21
// CHECK:STDOUT:   %.loc21_20.3: %Cpp.long = converted %T.binding.as_type.as.DivWith.impl.Op.call.loc21, %.loc21_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc21: init %empty_tuple.type = call %AssertSameType.specific_fn.loc21(%.loc21_20.3, %x.ref.loc21_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc22: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %x.ref.loc22_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc22: %.930 = impl_witness_access constants.%ModWith.impl_witness.770, element1 [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.5c95af.2]
// CHECK:STDOUT:   %bound_method.loc22_20.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.bound.468823.1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.binding.as_type.as.ModWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.specific_fn.4b7f8c.1]
// CHECK:STDOUT:   %bound_method.loc22_20.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.83c96c.1]
// CHECK:STDOUT:   %.loc22_20.1: %T.binding.as_type.as.ModWith.impl.Op.type.84d6b7.1 = specific_constant imports.%Core.Op.c8c, @T.binding.as_type.as.ModWith.impl.5ca(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.5c95af.1]
// CHECK:STDOUT:   %Op.ref.loc22: %T.binding.as_type.as.ModWith.impl.Op.type.84d6b7.1 = name_ref Op, %.loc22_20.1 [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.5c95af.1]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.bound.loc22: <bound method> = bound_method %int_1.loc22, %Op.ref.loc22 [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.bound.468823.2]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.specific_fn.loc22: <specific function> = specific_function %Op.ref.loc22, @T.binding.as_type.as.ModWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.specific_fn.4b7f8c.2]
// CHECK:STDOUT:   %bound_method.loc22_20.3: <bound method> = bound_method %int_1.loc22, %T.binding.as_type.as.ModWith.impl.Op.specific_fn.loc22 [concrete = constants.%bound_method.83c96c.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_18: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long = call %bound_method.loc22_18(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_18.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_18.2: %Cpp.long = converted %int_1.loc22, %.loc22_18.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.call.loc22: init %Cpp.long = call %bound_method.loc22_20.3(%.loc22_18.2, %x.ref.loc22_22)
// CHECK:STDOUT:   %x.ref.loc22_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc22: <specific function> = specific_function %AssertSameType.ref.loc22, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc22_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.ModWith.impl.Op.call.loc22
// CHECK:STDOUT:   %.loc22_20.3: %Cpp.long = converted %T.binding.as_type.as.ModWith.impl.Op.call.loc22, %.loc22_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc22: init %empty_tuple.type = call %AssertSameType.specific_fn.loc22(%.loc22_20.3, %x.ref.loc22_25)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.7ce = value_binding_pattern y [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc24_10: type = splice_block %i32.loc24 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc24: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc24_16.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_16.2: <bound method> = bound_method %int_1.loc24, %specific_fn.loc24 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24: init %i32 = call %bound_method.loc24_16.2(%int_1.loc24) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc24_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc24_16.2: %i32 = converted %int_1.loc24, %.loc24_16.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %y: %i32 = value_binding y, %.loc24_16.2
// CHECK:STDOUT:   %AssertSameType.ref.loc25: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %y.ref.loc25: %i32 = name_ref y, %y
// CHECK:STDOUT:   %x.ref.loc25_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc25: %.285 = impl_witness_access constants.%AddWith.impl_witness.a1d, element1 [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.3fccd2.2]
// CHECK:STDOUT:   %bound_method.loc25_20.1: <bound method> = bound_method %y.ref.loc25, %impl.elem1.loc25
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem1.loc25, @T.binding.as_type.as.AddWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.specific_fn.09578f.1]
// CHECK:STDOUT:   %bound_method.loc25_20.2: <bound method> = bound_method %y.ref.loc25, %specific_fn.loc25
// CHECK:STDOUT:   %.loc25_20.1: %T.binding.as_type.as.AddWith.impl.Op.type.5105fb.1 = specific_constant imports.%Core.Op.af2, @T.binding.as_type.as.AddWith.impl.f1a(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.3fccd2.1]
// CHECK:STDOUT:   %Op.ref.loc25: %T.binding.as_type.as.AddWith.impl.Op.type.5105fb.1 = name_ref Op, %.loc25_20.1 [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.3fccd2.1]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.bound.loc25: <bound method> = bound_method %y.ref.loc25, %Op.ref.loc25
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.specific_fn.loc25: <specific function> = specific_function %Op.ref.loc25, @T.binding.as_type.as.AddWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.specific_fn.09578f.2]
// CHECK:STDOUT:   %bound_method.loc25_20.3: <bound method> = bound_method %y.ref.loc25, %T.binding.as_type.as.AddWith.impl.Op.specific_fn.loc25
// CHECK:STDOUT:   %impl.elem0.loc25: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_18: <bound method> = bound_method %y.ref.loc25, %impl.elem0.loc25
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc25: init %Cpp.long = call %bound_method.loc25_18(%y.ref.loc25)
// CHECK:STDOUT:   %.loc25_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc25
// CHECK:STDOUT:   %.loc25_18.2: %Cpp.long = converted %y.ref.loc25, %.loc25_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.call.loc25: init %Cpp.long = call %bound_method.loc25_20.3(%.loc25_18.2, %x.ref.loc25_22)
// CHECK:STDOUT:   %x.ref.loc25_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc25: <specific function> = specific_function %AssertSameType.ref.loc25, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc25_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.AddWith.impl.Op.call.loc25
// CHECK:STDOUT:   %.loc25_20.3: %Cpp.long = converted %T.binding.as_type.as.AddWith.impl.Op.call.loc25, %.loc25_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc25: init %empty_tuple.type = call %AssertSameType.specific_fn.loc25(%.loc25_20.3, %x.ref.loc25_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc26: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %y.ref.loc26: %i32 = name_ref y, %y
// CHECK:STDOUT:   %x.ref.loc26_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc26: %.040 = impl_witness_access constants.%SubWith.impl_witness.702, element1 [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.f9b5c3.2]
// CHECK:STDOUT:   %bound_method.loc26_20.1: <bound method> = bound_method %y.ref.loc26, %impl.elem1.loc26
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem1.loc26, @T.binding.as_type.as.SubWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.specific_fn.b0f48e.1]
// CHECK:STDOUT:   %bound_method.loc26_20.2: <bound method> = bound_method %y.ref.loc26, %specific_fn.loc26
// CHECK:STDOUT:   %.loc26_20.1: %T.binding.as_type.as.SubWith.impl.Op.type.caca02.1 = specific_constant imports.%Core.Op.fcc, @T.binding.as_type.as.SubWith.impl.e1e(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.f9b5c3.1]
// CHECK:STDOUT:   %Op.ref.loc26: %T.binding.as_type.as.SubWith.impl.Op.type.caca02.1 = name_ref Op, %.loc26_20.1 [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.f9b5c3.1]
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.bound.loc26: <bound method> = bound_method %y.ref.loc26, %Op.ref.loc26
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.specific_fn.loc26: <specific function> = specific_function %Op.ref.loc26, @T.binding.as_type.as.SubWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.SubWith.impl.Op.specific_fn.b0f48e.2]
// CHECK:STDOUT:   %bound_method.loc26_20.3: <bound method> = bound_method %y.ref.loc26, %T.binding.as_type.as.SubWith.impl.Op.specific_fn.loc26
// CHECK:STDOUT:   %impl.elem0.loc26: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_18: <bound method> = bound_method %y.ref.loc26, %impl.elem0.loc26
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc26: init %Cpp.long = call %bound_method.loc26_18(%y.ref.loc26)
// CHECK:STDOUT:   %.loc26_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc26
// CHECK:STDOUT:   %.loc26_18.2: %Cpp.long = converted %y.ref.loc26, %.loc26_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.SubWith.impl.Op.call.loc26: init %Cpp.long = call %bound_method.loc26_20.3(%.loc26_18.2, %x.ref.loc26_22)
// CHECK:STDOUT:   %x.ref.loc26_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc26: <specific function> = specific_function %AssertSameType.ref.loc26, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc26_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.SubWith.impl.Op.call.loc26
// CHECK:STDOUT:   %.loc26_20.3: %Cpp.long = converted %T.binding.as_type.as.SubWith.impl.Op.call.loc26, %.loc26_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc26: init %empty_tuple.type = call %AssertSameType.specific_fn.loc26(%.loc26_20.3, %x.ref.loc26_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc27: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %y.ref.loc27: %i32 = name_ref y, %y
// CHECK:STDOUT:   %x.ref.loc27_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc27: %.1b4 = impl_witness_access constants.%MulWith.impl_witness.f85, element1 [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.461492.2]
// CHECK:STDOUT:   %bound_method.loc27_20.1: <bound method> = bound_method %y.ref.loc27, %impl.elem1.loc27
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem1.loc27, @T.binding.as_type.as.MulWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.specific_fn.475813.1]
// CHECK:STDOUT:   %bound_method.loc27_20.2: <bound method> = bound_method %y.ref.loc27, %specific_fn.loc27
// CHECK:STDOUT:   %.loc27_20.1: %T.binding.as_type.as.MulWith.impl.Op.type.1189b3.1 = specific_constant imports.%Core.Op.6a1, @T.binding.as_type.as.MulWith.impl.6d9(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.461492.1]
// CHECK:STDOUT:   %Op.ref.loc27: %T.binding.as_type.as.MulWith.impl.Op.type.1189b3.1 = name_ref Op, %.loc27_20.1 [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.461492.1]
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.bound.loc27: <bound method> = bound_method %y.ref.loc27, %Op.ref.loc27
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.specific_fn.loc27: <specific function> = specific_function %Op.ref.loc27, @T.binding.as_type.as.MulWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.MulWith.impl.Op.specific_fn.475813.2]
// CHECK:STDOUT:   %bound_method.loc27_20.3: <bound method> = bound_method %y.ref.loc27, %T.binding.as_type.as.MulWith.impl.Op.specific_fn.loc27
// CHECK:STDOUT:   %impl.elem0.loc27: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_18: <bound method> = bound_method %y.ref.loc27, %impl.elem0.loc27
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc27: init %Cpp.long = call %bound_method.loc27_18(%y.ref.loc27)
// CHECK:STDOUT:   %.loc27_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc27
// CHECK:STDOUT:   %.loc27_18.2: %Cpp.long = converted %y.ref.loc27, %.loc27_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.call.loc27: init %Cpp.long = call %bound_method.loc27_20.3(%.loc27_18.2, %x.ref.loc27_22)
// CHECK:STDOUT:   %x.ref.loc27_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc27: <specific function> = specific_function %AssertSameType.ref.loc27, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc27_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.MulWith.impl.Op.call.loc27
// CHECK:STDOUT:   %.loc27_20.3: %Cpp.long = converted %T.binding.as_type.as.MulWith.impl.Op.call.loc27, %.loc27_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc27: init %empty_tuple.type = call %AssertSameType.specific_fn.loc27(%.loc27_20.3, %x.ref.loc27_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc28: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %y.ref.loc28: %i32 = name_ref y, %y
// CHECK:STDOUT:   %x.ref.loc28_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc28: %.0f3 = impl_witness_access constants.%DivWith.impl_witness.996, element1 [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.e5dcee.2]
// CHECK:STDOUT:   %bound_method.loc28_20.1: <bound method> = bound_method %y.ref.loc28, %impl.elem1.loc28
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem1.loc28, @T.binding.as_type.as.DivWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.specific_fn.8f9bff.1]
// CHECK:STDOUT:   %bound_method.loc28_20.2: <bound method> = bound_method %y.ref.loc28, %specific_fn.loc28
// CHECK:STDOUT:   %.loc28_20.1: %T.binding.as_type.as.DivWith.impl.Op.type.bc0f3d.1 = specific_constant imports.%Core.Op.ace, @T.binding.as_type.as.DivWith.impl.67f(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.e5dcee.1]
// CHECK:STDOUT:   %Op.ref.loc28: %T.binding.as_type.as.DivWith.impl.Op.type.bc0f3d.1 = name_ref Op, %.loc28_20.1 [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.e5dcee.1]
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.bound.loc28: <bound method> = bound_method %y.ref.loc28, %Op.ref.loc28
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.specific_fn.loc28: <specific function> = specific_function %Op.ref.loc28, @T.binding.as_type.as.DivWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.DivWith.impl.Op.specific_fn.8f9bff.2]
// CHECK:STDOUT:   %bound_method.loc28_20.3: <bound method> = bound_method %y.ref.loc28, %T.binding.as_type.as.DivWith.impl.Op.specific_fn.loc28
// CHECK:STDOUT:   %impl.elem0.loc28: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_18: <bound method> = bound_method %y.ref.loc28, %impl.elem0.loc28
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc28: init %Cpp.long = call %bound_method.loc28_18(%y.ref.loc28)
// CHECK:STDOUT:   %.loc28_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc28
// CHECK:STDOUT:   %.loc28_18.2: %Cpp.long = converted %y.ref.loc28, %.loc28_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.DivWith.impl.Op.call.loc28: init %Cpp.long = call %bound_method.loc28_20.3(%.loc28_18.2, %x.ref.loc28_22)
// CHECK:STDOUT:   %x.ref.loc28_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc28: <specific function> = specific_function %AssertSameType.ref.loc28, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc28_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.DivWith.impl.Op.call.loc28
// CHECK:STDOUT:   %.loc28_20.3: %Cpp.long = converted %T.binding.as_type.as.DivWith.impl.Op.call.loc28, %.loc28_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc28: init %empty_tuple.type = call %AssertSameType.specific_fn.loc28(%.loc28_20.3, %x.ref.loc28_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc29: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %y.ref.loc29: %i32 = name_ref y, %y
// CHECK:STDOUT:   %x.ref.loc29_22: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc29: %.e68 = impl_witness_access constants.%ModWith.impl_witness.e1d, element1 [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.c5465f.2]
// CHECK:STDOUT:   %bound_method.loc29_20.1: <bound method> = bound_method %y.ref.loc29, %impl.elem1.loc29
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem1.loc29, @T.binding.as_type.as.ModWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.specific_fn.ee32ad.1]
// CHECK:STDOUT:   %bound_method.loc29_20.2: <bound method> = bound_method %y.ref.loc29, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29_20.1: %T.binding.as_type.as.ModWith.impl.Op.type.9786fd.1 = specific_constant imports.%Core.Op.c8c, @T.binding.as_type.as.ModWith.impl.5ca(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.c5465f.1]
// CHECK:STDOUT:   %Op.ref.loc29: %T.binding.as_type.as.ModWith.impl.Op.type.9786fd.1 = name_ref Op, %.loc29_20.1 [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.c5465f.1]
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.bound.loc29: <bound method> = bound_method %y.ref.loc29, %Op.ref.loc29
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.specific_fn.loc29: <specific function> = specific_function %Op.ref.loc29, @T.binding.as_type.as.ModWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.ModWith.impl.Op.specific_fn.ee32ad.2]
// CHECK:STDOUT:   %bound_method.loc29_20.3: <bound method> = bound_method %y.ref.loc29, %T.binding.as_type.as.ModWith.impl.Op.specific_fn.loc29
// CHECK:STDOUT:   %impl.elem0.loc29: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_18: <bound method> = bound_method %y.ref.loc29, %impl.elem0.loc29
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29: init %Cpp.long = call %bound_method.loc29_18(%y.ref.loc29)
// CHECK:STDOUT:   %.loc29_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29
// CHECK:STDOUT:   %.loc29_18.2: %Cpp.long = converted %y.ref.loc29, %.loc29_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.ModWith.impl.Op.call.loc29: init %Cpp.long = call %bound_method.loc29_20.3(%.loc29_18.2, %x.ref.loc29_22)
// CHECK:STDOUT:   %x.ref.loc29_25: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc29: <specific function> = specific_function %AssertSameType.ref.loc29, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc29_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.ModWith.impl.Op.call.loc29
// CHECK:STDOUT:   %.loc29_20.3: %Cpp.long = converted %T.binding.as_type.as.ModWith.impl.Op.call.loc29, %.loc29_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc29: init %empty_tuple.type = call %AssertSameType.specific_fn.loc29(%.loc29_20.3, %x.ref.loc29_25)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- arithmetic_heterogeneous_long_and_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AssertSameType.type: type = fn_type @AssertSameType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AssertSameType: %AssertSameType.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.fc6021.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2ad: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.94e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.556: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d48: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.556) [concrete]
// CHECK:STDOUT:   %.567: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.d48 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.102: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %AddWith.type.f83: type = facet_type <@AddWith, @AddWith(%i64)> [concrete]
// CHECK:STDOUT:   %AddWith.Op.type.c7f: type = fn_type @AddWith.Op, @AddWith(%i64) [concrete]
// CHECK:STDOUT:   %AddWith.type.9c2: type = facet_type <@AddWith, @AddWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %AddWith.Op.type.9a6: type = fn_type @AddWith.Op, @AddWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.39a54f.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.fc6021.1)> [symbolic]
// CHECK:STDOUT:   %U.354: %ImplicitAs.type.39a54f.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.type.188570.1: type = fn_type @Int.as.AddWith.impl.Op.2, @Int.as.AddWith.impl.b14(%N, %U.354) [symbolic]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.fecf95.1: %Int.as.AddWith.impl.Op.type.188570.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.type.188570.2: type = fn_type @Int.as.AddWith.impl.Op.3, @Int.as.AddWith.impl.b14(%N, %U.354) [symbolic]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.fecf95.2: %Int.as.AddWith.impl.Op.type.188570.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.354: %ImplicitAs.type.39a54f.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.type.4c3f59.1: type = fn_type @T.binding.as_type.as.AddWith.impl.Op.5, @T.binding.as_type.as.AddWith.impl.83e(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.363ab3.1: %T.binding.as_type.as.AddWith.impl.Op.type.4c3f59.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.type.4c3f59.2: type = fn_type @T.binding.as_type.as.AddWith.impl.Op.6, @T.binding.as_type.as.AddWith.impl.83e(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.363ab3.2: %T.binding.as_type.as.AddWith.impl.Op.type.4c3f59.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bbc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cb1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ba4: %ImplicitAs.type.2ad = facet_value %Cpp.long, (%ImplicitAs.impl_witness.bbc) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.525: <witness> = impl_witness imports.%AddWith.impl_witness_table.787, @T.binding.as_type.as.AddWith.impl.83e(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.type.e6cc8e.1: type = fn_type @T.binding.as_type.as.AddWith.impl.Op.6, @T.binding.as_type.as.AddWith.impl.83e(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.86dab9.1: %T.binding.as_type.as.AddWith.impl.Op.type.e6cc8e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.type.e6cc8e.2: type = fn_type @T.binding.as_type.as.AddWith.impl.Op.5, @T.binding.as_type.as.AddWith.impl.83e(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.86dab9.2: %T.binding.as_type.as.AddWith.impl.Op.type.e6cc8e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet.393: %AddWith.type.f83 = facet_value %Cpp.long, (%AddWith.impl_witness.525) [concrete]
// CHECK:STDOUT:   %.e8e: type = fn_type_with_self_type %AddWith.Op.type.c7f, %AddWith.facet.393 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.specific_fn.dee524.1: <specific function> = specific_function %T.binding.as_type.as.AddWith.impl.Op.86dab9.2, @T.binding.as_type.as.AddWith.impl.Op.5(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.specific_fn.dee524.2: <specific function> = specific_function %T.binding.as_type.as.AddWith.impl.Op.86dab9.1, @T.binding.as_type.as.AddWith.impl.Op.6(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %.8a2: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.ba4 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %AssertSameType.specific_fn: <specific function> = specific_function %AssertSameType, @AssertSameType(%i64) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.023: <witness> = impl_witness imports.%AddWith.impl_witness_table.c46, @Int.as.AddWith.impl.b14(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.type.0fadc0.1: type = fn_type @Int.as.AddWith.impl.Op.3, @Int.as.AddWith.impl.b14(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.f0aafc.1: %Int.as.AddWith.impl.Op.type.0fadc0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.type.0fadc0.2: type = fn_type @Int.as.AddWith.impl.Op.2, @Int.as.AddWith.impl.b14(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.f0aafc.2: %Int.as.AddWith.impl.Op.type.0fadc0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet.1c5: %AddWith.type.9c2 = facet_value %i64, (%AddWith.impl_witness.023) [concrete]
// CHECK:STDOUT:   %.f5b: type = fn_type_with_self_type %AddWith.Op.type.9a6, %AddWith.facet.1c5 [concrete]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.specific_fn.153203.1: <specific function> = specific_function %Int.as.AddWith.impl.Op.f0aafc.2, @Int.as.AddWith.impl.Op.2(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.specific_fn.153203.2: <specific function> = specific_function %Int.as.AddWith.impl.Op.f0aafc.1, @Int.as.AddWith.impl.Op.3(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl.b2d [concrete]
// CHECK:STDOUT:   %Core.import_ref.475cb3.2 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.266: @Int.as.AddWith.impl.b14.%Int.as.AddWith.impl.Op.type.2 (%Int.as.AddWith.impl.Op.type.188570.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddWith.impl.b14.%Int.as.AddWith.impl.Op.2 (constants.%Int.as.AddWith.impl.Op.fecf95.1)]
// CHECK:STDOUT:   %AddWith.impl_witness_table.c46 = impl_witness_table (%Core.import_ref.475cb3.2, %Core.import_ref.266), @Int.as.AddWith.impl.b14 [concrete]
// CHECK:STDOUT:   %Core.Op.a76: @Int.as.AddWith.impl.b14.%Int.as.AddWith.impl.Op.type.1 (%Int.as.AddWith.impl.Op.type.188570.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddWith.impl.b14.%Int.as.AddWith.impl.Op.1 (constants.%Int.as.AddWith.impl.Op.fecf95.2)]
// CHECK:STDOUT:   %Core.import_ref.475cb3.3 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bba: @T.binding.as_type.as.AddWith.impl.83e.%T.binding.as_type.as.AddWith.impl.Op.type.2 (%T.binding.as_type.as.AddWith.impl.Op.type.4c3f59.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.AddWith.impl.83e.%T.binding.as_type.as.AddWith.impl.Op.2 (constants.%T.binding.as_type.as.AddWith.impl.Op.363ab3.1)]
// CHECK:STDOUT:   %AddWith.impl_witness_table.787 = impl_witness_table (%Core.import_ref.475cb3.3, %Core.import_ref.bba), @T.binding.as_type.as.AddWith.impl.83e [concrete]
// CHECK:STDOUT:   %Core.Op.fcd: @T.binding.as_type.as.AddWith.impl.83e.%T.binding.as_type.as.AddWith.impl.Op.type.1 (%T.binding.as_type.as.AddWith.impl.Op.type.4c3f59.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @T.binding.as_type.as.AddWith.impl.83e.%T.binding.as_type.as.AddWith.impl.Op.1 (constants.%T.binding.as_type.as.AddWith.impl.Op.363ab3.2)]
// CHECK:STDOUT:   %Core.import_ref.297: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cb1 = impl_witness_table (%Core.import_ref.297), @Cpp.long.as.ImplicitAs.impl.e6d [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ArithmeticHeterogeneousLongAndI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.68c = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc10: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long = call %bound_method.loc10(%int_1.loc10) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc10_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc10_21.2: %Cpp.long = converted %int_1.loc10, %.loc10_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %x: %Cpp.long = value_binding x, %.loc10_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.95b = value_binding_pattern y [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_10: type = splice_block %i64 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc11: %.567 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc11_16.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.102]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_16.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_16.2(%int_1.loc11) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc11_16.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc11_16.2: %i64 = converted %int_1.loc11, %.loc11_16.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %y: %i64 = value_binding y, %.loc11_16.2
// CHECK:STDOUT:   %AssertSameType.ref.loc12: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %x.ref.loc12: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc12_22: %i64 = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc12: %.e8e = impl_witness_access constants.%AddWith.impl_witness.525, element1 [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.86dab9.2]
// CHECK:STDOUT:   %bound_method.loc12_20.1: <bound method> = bound_method %x.ref.loc12, %impl.elem1.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem1.loc12, @T.binding.as_type.as.AddWith.impl.Op.5(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.specific_fn.dee524.1]
// CHECK:STDOUT:   %bound_method.loc12_20.2: <bound method> = bound_method %x.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_20.1: %T.binding.as_type.as.AddWith.impl.Op.type.e6cc8e.1 = specific_constant imports.%Core.Op.fcd, @T.binding.as_type.as.AddWith.impl.83e(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.86dab9.1]
// CHECK:STDOUT:   %Op.ref.loc12: %T.binding.as_type.as.AddWith.impl.Op.type.e6cc8e.1 = name_ref Op, %.loc12_20.1 [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.86dab9.1]
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.bound: <bound method> = bound_method %x.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @T.binding.as_type.as.AddWith.impl.Op.6(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%T.binding.as_type.as.AddWith.impl.Op.specific_fn.dee524.2]
// CHECK:STDOUT:   %bound_method.loc12_20.3: <bound method> = bound_method %x.ref.loc12, %T.binding.as_type.as.AddWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12: %.8a2 = impl_witness_access constants.%ImplicitAs.impl_witness.bbc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_18: <bound method> = bound_method %x.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12: init %i64 = call %bound_method.loc12_18(%x.ref.loc12)
// CHECK:STDOUT:   %.loc12_18.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_18.2: %i64 = converted %x.ref.loc12, %.loc12_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.AddWith.impl.Op.call: init %i64 = call %bound_method.loc12_20.3(%.loc12_18.2, %y.ref.loc12_22)
// CHECK:STDOUT:   %y.ref.loc12_25: %i64 = name_ref y, %y
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc12: <specific function> = specific_function %AssertSameType.ref.loc12, @AssertSameType(constants.%i64) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc12_20.2: %i64 = value_of_initializer %T.binding.as_type.as.AddWith.impl.Op.call
// CHECK:STDOUT:   %.loc12_20.3: %i64 = converted %T.binding.as_type.as.AddWith.impl.Op.call, %.loc12_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc12: init %empty_tuple.type = call %AssertSameType.specific_fn.loc12(%.loc12_20.3, %y.ref.loc12_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc13: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %y.ref.loc13_18: %i64 = name_ref y, %y
// CHECK:STDOUT:   %x.ref.loc13: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc13: %.f5b = impl_witness_access constants.%AddWith.impl_witness.023, element1 [concrete = constants.%Int.as.AddWith.impl.Op.f0aafc.2]
// CHECK:STDOUT:   %bound_method.loc13_20.1: <bound method> = bound_method %y.ref.loc13_18, %impl.elem1.loc13
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem1.loc13, @Int.as.AddWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%Int.as.AddWith.impl.Op.specific_fn.153203.1]
// CHECK:STDOUT:   %bound_method.loc13_20.2: <bound method> = bound_method %y.ref.loc13_18, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_20.1: %Int.as.AddWith.impl.Op.type.0fadc0.1 = specific_constant imports.%Core.Op.a76, @Int.as.AddWith.impl.b14(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%Int.as.AddWith.impl.Op.f0aafc.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Int.as.AddWith.impl.Op.type.0fadc0.1 = name_ref Op, %.loc13_20.1 [concrete = constants.%Int.as.AddWith.impl.Op.f0aafc.1]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.bound: <bound method> = bound_method %y.ref.loc13_18, %Op.ref.loc13
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Int.as.AddWith.impl.Op.3(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%Int.as.AddWith.impl.Op.specific_fn.153203.2]
// CHECK:STDOUT:   %bound_method.loc13_20.3: <bound method> = bound_method %y.ref.loc13_18, %Int.as.AddWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13: %.8a2 = impl_witness_access constants.%ImplicitAs.impl_witness.bbc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_22: <bound method> = bound_method %x.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_22(%x.ref.loc13)
// CHECK:STDOUT:   %.loc13_22.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_22.2: %i64 = converted %x.ref.loc13, %.loc13_22.1
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call: init %i64 = call %bound_method.loc13_20.3(%y.ref.loc13_18, %.loc13_22.2)
// CHECK:STDOUT:   %y.ref.loc13_25: %i64 = name_ref y, %y
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc13: <specific function> = specific_function %AssertSameType.ref.loc13, @AssertSameType(constants.%i64) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc13_20.2: %i64 = value_of_initializer %Int.as.AddWith.impl.Op.call
// CHECK:STDOUT:   %.loc13_20.3: %i64 = converted %Int.as.AddWith.impl.Op.call, %.loc13_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc13: init %empty_tuple.type = call %AssertSameType.specific_fn.loc13(%.loc13_20.3, %y.ref.loc13_25)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bitwise_homogeneous_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AssertSameType.type: type = fn_type @AssertSameType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AssertSameType: %AssertSameType.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %BitComplement.type: type = facet_type <@BitComplement> [concrete]
// CHECK:STDOUT:   %BitComplement.Op.type: type = fn_type @BitComplement.Op [concrete]
// CHECK:STDOUT:   %BitComplement.impl_witness: <witness> = impl_witness imports.%BitComplement.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitComplement.facet: %BitComplement.type = facet_value %Cpp.long, (%BitComplement.impl_witness) [concrete]
// CHECK:STDOUT:   %.788: type = fn_type_with_self_type %BitComplement.Op.type, %BitComplement.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitComplement.impl.Op.type: type = fn_type @Cpp.long.as.BitComplement.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitComplement.impl.Op: %Cpp.long.as.BitComplement.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AssertSameType.specific_fn: <specific function> = specific_function %AssertSameType, @AssertSameType(%Cpp.long) [concrete]
// CHECK:STDOUT:   %BitAndWith.type.0b5: type = facet_type <@BitAndWith, @BitAndWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.e13: type = fn_type @BitAndWith.Op, @BitAndWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness.6f1: <witness> = impl_witness imports.%BitAndWith.impl_witness_table.e8c [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.0b5 = facet_value %Cpp.long, (%BitAndWith.impl_witness.6f1) [concrete]
// CHECK:STDOUT:   %.78c: type = fn_type_with_self_type %BitAndWith.Op.type.e13, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.type.a55: type = fn_type @Cpp.long.as.BitAndWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.340: %Cpp.long.as.BitAndWith.impl.Op.type.a55 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrWith.type.b22: type = facet_type <@BitOrWith, @BitOrWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %BitOrWith.Op.type.a1b: type = fn_type @BitOrWith.Op, @BitOrWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %BitOrWith.impl_witness.003: <witness> = impl_witness imports.%BitOrWith.impl_witness_table.513 [concrete]
// CHECK:STDOUT:   %BitOrWith.facet: %BitOrWith.type.b22 = facet_value %Cpp.long, (%BitOrWith.impl_witness.003) [concrete]
// CHECK:STDOUT:   %.ee7: type = fn_type_with_self_type %BitOrWith.Op.type.a1b, %BitOrWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.type.178: type = fn_type @Cpp.long.as.BitOrWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.b33: %Cpp.long.as.BitOrWith.impl.Op.type.178 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorWith.type.87f: type = facet_type <@BitXorWith, @BitXorWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %BitXorWith.Op.type.5cf: type = fn_type @BitXorWith.Op, @BitXorWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %BitXorWith.impl_witness.e60: <witness> = impl_witness imports.%BitXorWith.impl_witness_table.708 [concrete]
// CHECK:STDOUT:   %BitXorWith.facet: %BitXorWith.type.87f = facet_value %Cpp.long, (%BitXorWith.impl_witness.e60) [concrete]
// CHECK:STDOUT:   %.e9a: type = fn_type_with_self_type %BitXorWith.Op.type.5cf, %BitXorWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.type.198: type = fn_type @Cpp.long.as.BitXorWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.655: %Cpp.long.as.BitXorWith.impl.Op.type.198 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftWith.type.3f9: type = facet_type <@LeftShiftWith, @LeftShiftWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.Op.type.224: type = fn_type @LeftShiftWith.Op, @LeftShiftWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness.cd6: <witness> = impl_witness imports.%LeftShiftWith.impl_witness_table.01c [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet: %LeftShiftWith.type.3f9 = facet_value %Cpp.long, (%LeftShiftWith.impl_witness.cd6) [concrete]
// CHECK:STDOUT:   %.864: type = fn_type_with_self_type %LeftShiftWith.Op.type.224, %LeftShiftWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.type.235: type = fn_type @Cpp.long.as.LeftShiftWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.dd3: %Cpp.long.as.LeftShiftWith.impl.Op.type.235 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftWith.type.995: type = facet_type <@RightShiftWith, @RightShiftWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %RightShiftWith.Op.type.fd9: type = fn_type @RightShiftWith.Op, @RightShiftWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %RightShiftWith.impl_witness.b46: <witness> = impl_witness imports.%RightShiftWith.impl_witness_table.636 [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet: %RightShiftWith.type.995 = facet_value %Cpp.long, (%RightShiftWith.impl_witness.b46) [concrete]
// CHECK:STDOUT:   %.398: type = fn_type_with_self_type %RightShiftWith.Op.type.fd9, %RightShiftWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.type.233: type = fn_type @Cpp.long.as.RightShiftWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.f79: %Cpp.long.as.RightShiftWith.impl.Op.type.233 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.454: %Cpp.long.as.BitComplement.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.BitComplement.impl.Op]
// CHECK:STDOUT:   %BitComplement.impl_witness_table = impl_witness_table (%Core.import_ref.cb912f.1, %Core.import_ref.454), @Cpp.long.as.BitComplement.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.4 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f38: %Cpp.long.as.BitAndWith.impl.Op.type.a55 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.340]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table.e8c = impl_witness_table (%Core.import_ref.cb912f.4, %Core.import_ref.f38), @Cpp.long.as.BitAndWith.impl.7cf [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.7 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a17: %Cpp.long.as.BitOrWith.impl.Op.type.178 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.b33]
// CHECK:STDOUT:   %BitOrWith.impl_witness_table.513 = impl_witness_table (%Core.import_ref.cb912f.7, %Core.import_ref.a17), @Cpp.long.as.BitOrWith.impl.fd5 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.10 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.4fd: %Cpp.long.as.BitXorWith.impl.Op.type.198 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.655]
// CHECK:STDOUT:   %BitXorWith.impl_witness_table.708 = impl_witness_table (%Core.import_ref.cb912f.10, %Core.import_ref.4fd), @Cpp.long.as.BitXorWith.impl.ef4 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.13 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.c09: %Cpp.long.as.LeftShiftWith.impl.Op.type.235 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.dd3]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness_table.01c = impl_witness_table (%Core.import_ref.cb912f.13, %Core.import_ref.c09), @Cpp.long.as.LeftShiftWith.impl.979 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.16 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6c0: %Cpp.long.as.RightShiftWith.impl.Op.type.233 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.f79]
// CHECK:STDOUT:   %RightShiftWith.impl_witness_table.636 = impl_witness_table (%Core.import_ref.cb912f.16, %Core.import_ref.6c0), @Cpp.long.as.RightShiftWith.impl.711 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BitWiseHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_13: type = splice_block %long.ref.loc10 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc10: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc10: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long = call %bound_method.loc10(%int_1.loc10) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc10_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc10_21.2: %Cpp.long = converted %int_1.loc10, %.loc10_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc10_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_13: type = splice_block %long.ref.loc11 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc11: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc11: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11(%int_1.loc11) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc11_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc11_21.2: %Cpp.long = converted %int_1.loc11, %.loc11_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc11_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.68c = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc12_13: type = splice_block %long.ref.loc12 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc12: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc12: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12(%int_1.loc12) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc12_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc12_21.2: %Cpp.long = converted %int_1.loc12, %.loc12_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %c: %Cpp.long = value_binding c, %.loc12_21.2
// CHECK:STDOUT:   %AssertSameType.ref.loc14: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc14: %.788 = impl_witness_access constants.%BitComplement.impl_witness, element1 [concrete = constants.%Cpp.long.as.BitComplement.impl.Op]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem1.loc14
// CHECK:STDOUT:   %Cpp.long.as.BitComplement.impl.Op.call: init %Cpp.long = call %bound_method.loc14(%a.ref.loc14)
// CHECK:STDOUT:   %c.ref.loc14: %Cpp.long = name_ref c, %c
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc14: <specific function> = specific_function %AssertSameType.ref.loc14, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc14_18.1: %Cpp.long = value_of_initializer %Cpp.long.as.BitComplement.impl.Op.call
// CHECK:STDOUT:   %.loc14_18.2: %Cpp.long = converted %Cpp.long.as.BitComplement.impl.Op.call, %.loc14_18.1
// CHECK:STDOUT:   %AssertSameType.call.loc14: init %empty_tuple.type = call %AssertSameType.specific_fn.loc14(%.loc14_18.2, %c.ref.loc14)
// CHECK:STDOUT:   %AssertSameType.ref.loc15: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.78c = impl_witness_access constants.%BitAndWith.impl_witness.6f1, element1 [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.340]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.call: init %Cpp.long = call %bound_method.loc15(%a.ref.loc15, %b.ref.loc15)
// CHECK:STDOUT:   %c.ref.loc15: %Cpp.long = name_ref c, %c
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc15: <specific function> = specific_function %AssertSameType.ref.loc15, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc15_20.1: %Cpp.long = value_of_initializer %Cpp.long.as.BitAndWith.impl.Op.call
// CHECK:STDOUT:   %.loc15_20.2: %Cpp.long = converted %Cpp.long.as.BitAndWith.impl.Op.call, %.loc15_20.1
// CHECK:STDOUT:   %AssertSameType.call.loc15: init %empty_tuple.type = call %AssertSameType.specific_fn.loc15(%.loc15_20.2, %c.ref.loc15)
// CHECK:STDOUT:   %AssertSameType.ref.loc16: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc16: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc16: %.ee7 = impl_witness_access constants.%BitOrWith.impl_witness.003, element1 [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.b33]
// CHECK:STDOUT:   %bound_method.loc16: <bound method> = bound_method %a.ref.loc16, %impl.elem1.loc16
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.call: init %Cpp.long = call %bound_method.loc16(%a.ref.loc16, %b.ref.loc16)
// CHECK:STDOUT:   %c.ref.loc16: %Cpp.long = name_ref c, %c
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc16: <specific function> = specific_function %AssertSameType.ref.loc16, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc16_20.1: %Cpp.long = value_of_initializer %Cpp.long.as.BitOrWith.impl.Op.call
// CHECK:STDOUT:   %.loc16_20.2: %Cpp.long = converted %Cpp.long.as.BitOrWith.impl.Op.call, %.loc16_20.1
// CHECK:STDOUT:   %AssertSameType.call.loc16: init %empty_tuple.type = call %AssertSameType.specific_fn.loc16(%.loc16_20.2, %c.ref.loc16)
// CHECK:STDOUT:   %AssertSameType.ref.loc17: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc17: %.e9a = impl_witness_access constants.%BitXorWith.impl_witness.e60, element1 [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.655]
// CHECK:STDOUT:   %bound_method.loc17: <bound method> = bound_method %a.ref.loc17, %impl.elem1.loc17
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.call: init %Cpp.long = call %bound_method.loc17(%a.ref.loc17, %b.ref.loc17)
// CHECK:STDOUT:   %c.ref.loc17: %Cpp.long = name_ref c, %c
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc17: <specific function> = specific_function %AssertSameType.ref.loc17, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc17_20.1: %Cpp.long = value_of_initializer %Cpp.long.as.BitXorWith.impl.Op.call
// CHECK:STDOUT:   %.loc17_20.2: %Cpp.long = converted %Cpp.long.as.BitXorWith.impl.Op.call, %.loc17_20.1
// CHECK:STDOUT:   %AssertSameType.call.loc17: init %empty_tuple.type = call %AssertSameType.specific_fn.loc17(%.loc17_20.2, %c.ref.loc17)
// CHECK:STDOUT:   %AssertSameType.ref.loc18: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc18: %.864 = impl_witness_access constants.%LeftShiftWith.impl_witness.cd6, element1 [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.dd3]
// CHECK:STDOUT:   %bound_method.loc18: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.call: init %Cpp.long = call %bound_method.loc18(%a.ref.loc18, %b.ref.loc18)
// CHECK:STDOUT:   %c.ref.loc18: %Cpp.long = name_ref c, %c
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc18: <specific function> = specific_function %AssertSameType.ref.loc18, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc18_20.1: %Cpp.long = value_of_initializer %Cpp.long.as.LeftShiftWith.impl.Op.call
// CHECK:STDOUT:   %.loc18_20.2: %Cpp.long = converted %Cpp.long.as.LeftShiftWith.impl.Op.call, %.loc18_20.1
// CHECK:STDOUT:   %AssertSameType.call.loc18: init %empty_tuple.type = call %AssertSameType.specific_fn.loc18(%.loc18_20.2, %c.ref.loc18)
// CHECK:STDOUT:   %AssertSameType.ref.loc19: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc19: %.398 = impl_witness_access constants.%RightShiftWith.impl_witness.b46, element1 [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.f79]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %a.ref.loc19, %impl.elem1.loc19
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.call: init %Cpp.long = call %bound_method.loc19(%a.ref.loc19, %b.ref.loc19)
// CHECK:STDOUT:   %c.ref.loc19: %Cpp.long = name_ref c, %c
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc19: <specific function> = specific_function %AssertSameType.ref.loc19, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc19_20.1: %Cpp.long = value_of_initializer %Cpp.long.as.RightShiftWith.impl.Op.call
// CHECK:STDOUT:   %.loc19_20.2: %Cpp.long = converted %Cpp.long.as.RightShiftWith.impl.Op.call, %.loc19_20.1
// CHECK:STDOUT:   %AssertSameType.call.loc19: init %empty_tuple.type = call %AssertSameType.specific_fn.loc19(%.loc19_20.2, %c.ref.loc19)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bitwise_heterogeneous_long_left_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AssertSameType.type: type = fn_type @AssertSameType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AssertSameType: %AssertSameType.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %BitAndWith.type.0fb: type = facet_type <@BitAndWith, @BitAndWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.e35: type = fn_type @BitAndWith.Op, @BitAndWith(%i32) [concrete]
// CHECK:STDOUT:   %T.57d: %ImplicitAs.type.819 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.type.8319b2.1: type = fn_type @Cpp.long.as.BitAndWith.impl.Op.1, @Cpp.long.as.BitAndWith.impl.334(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.3828c1.1: %Cpp.long.as.BitAndWith.impl.Op.type.8319b2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.type.8319b2.2: type = fn_type @Cpp.long.as.BitAndWith.impl.Op.2, @Cpp.long.as.BitAndWith.impl.334(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.3828c1.2: %Cpp.long.as.BitAndWith.impl.Op.type.8319b2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.011: type = facet_type <@BitAndWith, @BitAndWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.4bf: type = fn_type @BitAndWith.Op, @BitAndWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness.fe3: <witness> = impl_witness imports.%BitAndWith.impl_witness_table.072, @Cpp.long.as.BitAndWith.impl.334(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.type.202298.1: type = fn_type @Cpp.long.as.BitAndWith.impl.Op.2, @Cpp.long.as.BitAndWith.impl.334(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.a9ac84.1: %Cpp.long.as.BitAndWith.impl.Op.type.202298.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.type.202298.2: type = fn_type @Cpp.long.as.BitAndWith.impl.Op.1, @Cpp.long.as.BitAndWith.impl.334(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.a9ac84.2: %Cpp.long.as.BitAndWith.impl.Op.type.202298.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.9d5: %BitAndWith.type.0fb = facet_value %Cpp.long, (%BitAndWith.impl_witness.fe3) [concrete]
// CHECK:STDOUT:   %.bdd: type = fn_type_with_self_type %BitAndWith.Op.type.e35, %BitAndWith.facet.9d5 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.specific_fn.dcd797.1: <specific function> = specific_function %Cpp.long.as.BitAndWith.impl.Op.a9ac84.2, @Cpp.long.as.BitAndWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.c45: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.specific_fn.dcd797.2: <specific function> = specific_function %Cpp.long.as.BitAndWith.impl.Op.a9ac84.1, @Cpp.long.as.BitAndWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %AssertSameType.specific_fn: <specific function> = specific_function %AssertSameType, @AssertSameType(%Cpp.long) [concrete]
// CHECK:STDOUT:   %BitOrWith.type.428: type = facet_type <@BitOrWith, @BitOrWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitOrWith.Op.type.99b: type = fn_type @BitOrWith.Op, @BitOrWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.type.9f5440.1: type = fn_type @Cpp.long.as.BitOrWith.impl.Op.1, @Cpp.long.as.BitOrWith.impl.fc3(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.81f55a.1: %Cpp.long.as.BitOrWith.impl.Op.type.9f5440.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.type.9f5440.2: type = fn_type @Cpp.long.as.BitOrWith.impl.Op.2, @Cpp.long.as.BitOrWith.impl.fc3(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.81f55a.2: %Cpp.long.as.BitOrWith.impl.Op.type.9f5440.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrWith.type.cba: type = facet_type <@BitOrWith, @BitOrWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %BitOrWith.Op.type.0e5: type = fn_type @BitOrWith.Op, @BitOrWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %BitOrWith.impl_witness.b0c: <witness> = impl_witness imports.%BitOrWith.impl_witness_table.9a6, @Cpp.long.as.BitOrWith.impl.fc3(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.type.dccbc7.1: type = fn_type @Cpp.long.as.BitOrWith.impl.Op.2, @Cpp.long.as.BitOrWith.impl.fc3(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.d9c870.1: %Cpp.long.as.BitOrWith.impl.Op.type.dccbc7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.type.dccbc7.2: type = fn_type @Cpp.long.as.BitOrWith.impl.Op.1, @Cpp.long.as.BitOrWith.impl.fc3(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.d9c870.2: %Cpp.long.as.BitOrWith.impl.Op.type.dccbc7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.acb: %BitOrWith.type.428 = facet_value %Cpp.long, (%BitOrWith.impl_witness.b0c) [concrete]
// CHECK:STDOUT:   %.faa: type = fn_type_with_self_type %BitOrWith.Op.type.99b, %BitOrWith.facet.acb [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.specific_fn.03ab4b.1: <specific function> = specific_function %Cpp.long.as.BitOrWith.impl.Op.d9c870.2, @Cpp.long.as.BitOrWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.specific_fn.03ab4b.2: <specific function> = specific_function %Cpp.long.as.BitOrWith.impl.Op.d9c870.1, @Cpp.long.as.BitOrWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitXorWith.type.1fc: type = facet_type <@BitXorWith, @BitXorWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitXorWith.Op.type.a4b: type = fn_type @BitXorWith.Op, @BitXorWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.type.876fc1.1: type = fn_type @Cpp.long.as.BitXorWith.impl.Op.1, @Cpp.long.as.BitXorWith.impl.732(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.c0eaae.1: %Cpp.long.as.BitXorWith.impl.Op.type.876fc1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.type.876fc1.2: type = fn_type @Cpp.long.as.BitXorWith.impl.Op.2, @Cpp.long.as.BitXorWith.impl.732(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.c0eaae.2: %Cpp.long.as.BitXorWith.impl.Op.type.876fc1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorWith.type.fd0: type = facet_type <@BitXorWith, @BitXorWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %BitXorWith.Op.type.1f1: type = fn_type @BitXorWith.Op, @BitXorWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %BitXorWith.impl_witness.8a0: <witness> = impl_witness imports.%BitXorWith.impl_witness_table.9f4, @Cpp.long.as.BitXorWith.impl.732(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.type.7fe348.1: type = fn_type @Cpp.long.as.BitXorWith.impl.Op.2, @Cpp.long.as.BitXorWith.impl.732(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.48a9f4.1: %Cpp.long.as.BitXorWith.impl.Op.type.7fe348.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.type.7fe348.2: type = fn_type @Cpp.long.as.BitXorWith.impl.Op.1, @Cpp.long.as.BitXorWith.impl.732(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.48a9f4.2: %Cpp.long.as.BitXorWith.impl.Op.type.7fe348.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.91f: %BitXorWith.type.1fc = facet_value %Cpp.long, (%BitXorWith.impl_witness.8a0) [concrete]
// CHECK:STDOUT:   %.f2f: type = fn_type_with_self_type %BitXorWith.Op.type.a4b, %BitXorWith.facet.91f [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.specific_fn.4ec2b6.1: <specific function> = specific_function %Cpp.long.as.BitXorWith.impl.Op.48a9f4.2, @Cpp.long.as.BitXorWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.specific_fn.4ec2b6.2: <specific function> = specific_function %Cpp.long.as.BitXorWith.impl.Op.48a9f4.1, @Cpp.long.as.BitXorWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.type.048: type = facet_type <@LeftShiftWith, @LeftShiftWith(%i32)> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.Op.type.dfc: type = fn_type @LeftShiftWith.Op, @LeftShiftWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.type.97b57f.1: type = fn_type @Cpp.long.as.LeftShiftWith.impl.Op.1, @Cpp.long.as.LeftShiftWith.impl.272(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.69ad3e.1: %Cpp.long.as.LeftShiftWith.impl.Op.type.97b57f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.type.97b57f.2: type = fn_type @Cpp.long.as.LeftShiftWith.impl.Op.2, @Cpp.long.as.LeftShiftWith.impl.272(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.69ad3e.2: %Cpp.long.as.LeftShiftWith.impl.Op.type.97b57f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.type.a1f: type = facet_type <@LeftShiftWith, @LeftShiftWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.Op.type.55b: type = fn_type @LeftShiftWith.Op, @LeftShiftWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness.982: <witness> = impl_witness imports.%LeftShiftWith.impl_witness_table.a97, @Cpp.long.as.LeftShiftWith.impl.272(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.type.1d34cb.1: type = fn_type @Cpp.long.as.LeftShiftWith.impl.Op.2, @Cpp.long.as.LeftShiftWith.impl.272(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.9105a1.1: %Cpp.long.as.LeftShiftWith.impl.Op.type.1d34cb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.type.1d34cb.2: type = fn_type @Cpp.long.as.LeftShiftWith.impl.Op.1, @Cpp.long.as.LeftShiftWith.impl.272(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.9105a1.2: %Cpp.long.as.LeftShiftWith.impl.Op.type.1d34cb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.089: %LeftShiftWith.type.048 = facet_value %Cpp.long, (%LeftShiftWith.impl_witness.982) [concrete]
// CHECK:STDOUT:   %.2be: type = fn_type_with_self_type %LeftShiftWith.Op.type.dfc, %LeftShiftWith.facet.089 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.1fda18.1: <specific function> = specific_function %Cpp.long.as.LeftShiftWith.impl.Op.9105a1.2, @Cpp.long.as.LeftShiftWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.1fda18.2: <specific function> = specific_function %Cpp.long.as.LeftShiftWith.impl.Op.9105a1.1, @Cpp.long.as.LeftShiftWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %RightShiftWith.type.bc6: type = facet_type <@RightShiftWith, @RightShiftWith(%i32)> [concrete]
// CHECK:STDOUT:   %RightShiftWith.Op.type.807: type = fn_type @RightShiftWith.Op, @RightShiftWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.type.cc50bd.1: type = fn_type @Cpp.long.as.RightShiftWith.impl.Op.1, @Cpp.long.as.RightShiftWith.impl.3d6(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.e1e575.1: %Cpp.long.as.RightShiftWith.impl.Op.type.cc50bd.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.type.cc50bd.2: type = fn_type @Cpp.long.as.RightShiftWith.impl.Op.2, @Cpp.long.as.RightShiftWith.impl.3d6(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.e1e575.2: %Cpp.long.as.RightShiftWith.impl.Op.type.cc50bd.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftWith.type.18a: type = facet_type <@RightShiftWith, @RightShiftWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %RightShiftWith.Op.type.c0c: type = fn_type @RightShiftWith.Op, @RightShiftWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %RightShiftWith.impl_witness.ccb: <witness> = impl_witness imports.%RightShiftWith.impl_witness_table.e0a, @Cpp.long.as.RightShiftWith.impl.3d6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.type.e35e06.1: type = fn_type @Cpp.long.as.RightShiftWith.impl.Op.2, @Cpp.long.as.RightShiftWith.impl.3d6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.50edfc.1: %Cpp.long.as.RightShiftWith.impl.Op.type.e35e06.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.type.e35e06.2: type = fn_type @Cpp.long.as.RightShiftWith.impl.Op.1, @Cpp.long.as.RightShiftWith.impl.3d6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.50edfc.2: %Cpp.long.as.RightShiftWith.impl.Op.type.e35e06.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.6d9: %RightShiftWith.type.bc6 = facet_value %Cpp.long, (%RightShiftWith.impl_witness.ccb) [concrete]
// CHECK:STDOUT:   %.dac: type = fn_type_with_self_type %RightShiftWith.Op.type.807, %RightShiftWith.facet.6d9 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.specific_fn.012106.1: <specific function> = specific_function %Cpp.long.as.RightShiftWith.impl.Op.50edfc.2, @Cpp.long.as.RightShiftWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.specific_fn.012106.2: <specific function> = specific_function %Cpp.long.as.RightShiftWith.impl.Op.50edfc.1, @Cpp.long.as.RightShiftWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness.0fc: <witness> = impl_witness imports.%BitAndWith.impl_witness_table.072, @Cpp.long.as.BitAndWith.impl.334(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.type.ae3ddd.1: type = fn_type @Cpp.long.as.BitAndWith.impl.Op.2, @Cpp.long.as.BitAndWith.impl.334(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.98f052.1: %Cpp.long.as.BitAndWith.impl.Op.type.ae3ddd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.type.ae3ddd.2: type = fn_type @Cpp.long.as.BitAndWith.impl.Op.1, @Cpp.long.as.BitAndWith.impl.334(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.98f052.2: %Cpp.long.as.BitAndWith.impl.Op.type.ae3ddd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.6ae: %BitAndWith.type.011 = facet_value %Cpp.long, (%BitAndWith.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %.9d7: type = fn_type_with_self_type %BitAndWith.Op.type.4bf, %BitAndWith.facet.6ae [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.specific_fn.af17b5.1: <specific function> = specific_function %Cpp.long.as.BitAndWith.impl.Op.98f052.2, @Cpp.long.as.BitAndWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.specific_fn.af17b5.2: <specific function> = specific_function %Cpp.long.as.BitAndWith.impl.Op.98f052.1, @Cpp.long.as.BitAndWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %BitOrWith.impl_witness.39f: <witness> = impl_witness imports.%BitOrWith.impl_witness_table.9a6, @Cpp.long.as.BitOrWith.impl.fc3(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.type.331453.1: type = fn_type @Cpp.long.as.BitOrWith.impl.Op.2, @Cpp.long.as.BitOrWith.impl.fc3(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.664d1e.1: %Cpp.long.as.BitOrWith.impl.Op.type.331453.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.type.331453.2: type = fn_type @Cpp.long.as.BitOrWith.impl.Op.1, @Cpp.long.as.BitOrWith.impl.fc3(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.664d1e.2: %Cpp.long.as.BitOrWith.impl.Op.type.331453.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.9a5: %BitOrWith.type.cba = facet_value %Cpp.long, (%BitOrWith.impl_witness.39f) [concrete]
// CHECK:STDOUT:   %.a50: type = fn_type_with_self_type %BitOrWith.Op.type.0e5, %BitOrWith.facet.9a5 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.specific_fn.be7132.1: <specific function> = specific_function %Cpp.long.as.BitOrWith.impl.Op.664d1e.2, @Cpp.long.as.BitOrWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.specific_fn.be7132.2: <specific function> = specific_function %Cpp.long.as.BitOrWith.impl.Op.664d1e.1, @Cpp.long.as.BitOrWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %BitXorWith.impl_witness.001: <witness> = impl_witness imports.%BitXorWith.impl_witness_table.9f4, @Cpp.long.as.BitXorWith.impl.732(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.type.97a9a2.1: type = fn_type @Cpp.long.as.BitXorWith.impl.Op.2, @Cpp.long.as.BitXorWith.impl.732(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.5b3046.1: %Cpp.long.as.BitXorWith.impl.Op.type.97a9a2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.type.97a9a2.2: type = fn_type @Cpp.long.as.BitXorWith.impl.Op.1, @Cpp.long.as.BitXorWith.impl.732(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.5b3046.2: %Cpp.long.as.BitXorWith.impl.Op.type.97a9a2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.349: %BitXorWith.type.fd0 = facet_value %Cpp.long, (%BitXorWith.impl_witness.001) [concrete]
// CHECK:STDOUT:   %.8e7: type = fn_type_with_self_type %BitXorWith.Op.type.1f1, %BitXorWith.facet.349 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.specific_fn.f1c9e6.1: <specific function> = specific_function %Cpp.long.as.BitXorWith.impl.Op.5b3046.2, @Cpp.long.as.BitXorWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.specific_fn.f1c9e6.2: <specific function> = specific_function %Cpp.long.as.BitXorWith.impl.Op.5b3046.1, @Cpp.long.as.BitXorWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness.a1c: <witness> = impl_witness imports.%LeftShiftWith.impl_witness_table.a97, @Cpp.long.as.LeftShiftWith.impl.272(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.type.d93838.1: type = fn_type @Cpp.long.as.LeftShiftWith.impl.Op.2, @Cpp.long.as.LeftShiftWith.impl.272(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.f145d8.1: %Cpp.long.as.LeftShiftWith.impl.Op.type.d93838.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.type.d93838.2: type = fn_type @Cpp.long.as.LeftShiftWith.impl.Op.1, @Cpp.long.as.LeftShiftWith.impl.272(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.f145d8.2: %Cpp.long.as.LeftShiftWith.impl.Op.type.d93838.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.5e2: %LeftShiftWith.type.a1f = facet_value %Cpp.long, (%LeftShiftWith.impl_witness.a1c) [concrete]
// CHECK:STDOUT:   %.0a1: type = fn_type_with_self_type %LeftShiftWith.Op.type.55b, %LeftShiftWith.facet.5e2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.abc44e.1: <specific function> = specific_function %Cpp.long.as.LeftShiftWith.impl.Op.f145d8.2, @Cpp.long.as.LeftShiftWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.abc44e.2: <specific function> = specific_function %Cpp.long.as.LeftShiftWith.impl.Op.f145d8.1, @Cpp.long.as.LeftShiftWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %RightShiftWith.impl_witness.5fe: <witness> = impl_witness imports.%RightShiftWith.impl_witness_table.e0a, @Cpp.long.as.RightShiftWith.impl.3d6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.type.a6dc1c.1: type = fn_type @Cpp.long.as.RightShiftWith.impl.Op.2, @Cpp.long.as.RightShiftWith.impl.3d6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.249bc3.1: %Cpp.long.as.RightShiftWith.impl.Op.type.a6dc1c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.type.a6dc1c.2: type = fn_type @Cpp.long.as.RightShiftWith.impl.Op.1, @Cpp.long.as.RightShiftWith.impl.3d6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.249bc3.2: %Cpp.long.as.RightShiftWith.impl.Op.type.a6dc1c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.1d4: %RightShiftWith.type.18a = facet_value %Cpp.long, (%RightShiftWith.impl_witness.5fe) [concrete]
// CHECK:STDOUT:   %.fab: type = fn_type_with_self_type %RightShiftWith.Op.type.c0c, %RightShiftWith.facet.1d4 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.specific_fn.c6040e.1: <specific function> = specific_function %Cpp.long.as.RightShiftWith.impl.Op.249bc3.2, @Cpp.long.as.RightShiftWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.specific_fn.c6040e.2: <specific function> = specific_function %Cpp.long.as.RightShiftWith.impl.Op.249bc3.1, @Cpp.long.as.RightShiftWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.1 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.17a: @Cpp.long.as.BitAndWith.impl.334.%Cpp.long.as.BitAndWith.impl.Op.type.2 (%Cpp.long.as.BitAndWith.impl.Op.type.8319b2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndWith.impl.334.%Cpp.long.as.BitAndWith.impl.Op.2 (constants.%Cpp.long.as.BitAndWith.impl.Op.3828c1.1)]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table.072 = impl_witness_table (%Core.import_ref.cb912f.1, %Core.import_ref.17a), @Cpp.long.as.BitAndWith.impl.334 [concrete]
// CHECK:STDOUT:   %Core.Op.c8a: @Cpp.long.as.BitAndWith.impl.334.%Cpp.long.as.BitAndWith.impl.Op.type.1 (%Cpp.long.as.BitAndWith.impl.Op.type.8319b2.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndWith.impl.334.%Cpp.long.as.BitAndWith.impl.Op.1 (constants.%Cpp.long.as.BitAndWith.impl.Op.3828c1.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.3 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3b6d: @Cpp.long.as.BitOrWith.impl.fc3.%Cpp.long.as.BitOrWith.impl.Op.type.2 (%Cpp.long.as.BitOrWith.impl.Op.type.9f5440.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrWith.impl.fc3.%Cpp.long.as.BitOrWith.impl.Op.2 (constants.%Cpp.long.as.BitOrWith.impl.Op.81f55a.1)]
// CHECK:STDOUT:   %BitOrWith.impl_witness_table.9a6 = impl_witness_table (%Core.import_ref.cb912f.3, %Core.import_ref.3b6d), @Cpp.long.as.BitOrWith.impl.fc3 [concrete]
// CHECK:STDOUT:   %Core.Op.788: @Cpp.long.as.BitOrWith.impl.fc3.%Cpp.long.as.BitOrWith.impl.Op.type.1 (%Cpp.long.as.BitOrWith.impl.Op.type.9f5440.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrWith.impl.fc3.%Cpp.long.as.BitOrWith.impl.Op.1 (constants.%Cpp.long.as.BitOrWith.impl.Op.81f55a.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.619: @Cpp.long.as.BitXorWith.impl.732.%Cpp.long.as.BitXorWith.impl.Op.type.2 (%Cpp.long.as.BitXorWith.impl.Op.type.876fc1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorWith.impl.732.%Cpp.long.as.BitXorWith.impl.Op.2 (constants.%Cpp.long.as.BitXorWith.impl.Op.c0eaae.1)]
// CHECK:STDOUT:   %BitXorWith.impl_witness_table.9f4 = impl_witness_table (%Core.import_ref.cb912f.5, %Core.import_ref.619), @Cpp.long.as.BitXorWith.impl.732 [concrete]
// CHECK:STDOUT:   %Core.Op.c61: @Cpp.long.as.BitXorWith.impl.732.%Cpp.long.as.BitXorWith.impl.Op.type.1 (%Cpp.long.as.BitXorWith.impl.Op.type.876fc1.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorWith.impl.732.%Cpp.long.as.BitXorWith.impl.Op.1 (constants.%Cpp.long.as.BitXorWith.impl.Op.c0eaae.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.7 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0ca: @Cpp.long.as.LeftShiftWith.impl.272.%Cpp.long.as.LeftShiftWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftWith.impl.Op.type.97b57f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftWith.impl.272.%Cpp.long.as.LeftShiftWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftWith.impl.Op.69ad3e.1)]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness_table.a97 = impl_witness_table (%Core.import_ref.cb912f.7, %Core.import_ref.0ca), @Cpp.long.as.LeftShiftWith.impl.272 [concrete]
// CHECK:STDOUT:   %Core.Op.2db: @Cpp.long.as.LeftShiftWith.impl.272.%Cpp.long.as.LeftShiftWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftWith.impl.Op.type.97b57f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftWith.impl.272.%Cpp.long.as.LeftShiftWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftWith.impl.Op.69ad3e.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.038: @Cpp.long.as.RightShiftWith.impl.3d6.%Cpp.long.as.RightShiftWith.impl.Op.type.2 (%Cpp.long.as.RightShiftWith.impl.Op.type.cc50bd.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftWith.impl.3d6.%Cpp.long.as.RightShiftWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftWith.impl.Op.e1e575.1)]
// CHECK:STDOUT:   %RightShiftWith.impl_witness_table.e0a = impl_witness_table (%Core.import_ref.cb912f.9, %Core.import_ref.038), @Cpp.long.as.RightShiftWith.impl.3d6 [concrete]
// CHECK:STDOUT:   %Core.Op.330: @Cpp.long.as.RightShiftWith.impl.3d6.%Cpp.long.as.RightShiftWith.impl.Op.type.1 (%Cpp.long.as.RightShiftWith.impl.Op.type.cc50bd.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftWith.impl.3d6.%Cpp.long.as.RightShiftWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftWith.impl.Op.e1e575.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BitWiseHeterogeneousLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc11: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11(%int_1.loc11) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc11_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc11_21.2: %Cpp.long = converted %int_1.loc11, %.loc11_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc11_21.2
// CHECK:STDOUT:   %AssertSameType.ref.loc13: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc13_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc13: %.bdd = impl_witness_access constants.%BitAndWith.impl_witness.fe3, element1 [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.a9ac84.2]
// CHECK:STDOUT:   %bound_method.loc13_20.1: <bound method> = bound_method %a.ref.loc13_18, %impl.elem1.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc13_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc13_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem1.loc13, @Cpp.long.as.BitAndWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.specific_fn.dcd797.1]
// CHECK:STDOUT:   %bound_method.loc13_20.2: <bound method> = bound_method %a.ref.loc13_18, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_20.3: %Cpp.long.as.BitAndWith.impl.Op.type.202298.1 = specific_constant imports.%Core.Op.c8a, @Cpp.long.as.BitAndWith.impl.334(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.a9ac84.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.BitAndWith.impl.Op.type.202298.1 = name_ref Op, %.loc13_20.3 [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.a9ac84.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.bound.loc13: <bound method> = bound_method %a.ref.loc13_18, %Op.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_20.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_20: init %Cpp.long = call %bound_method.loc13_20.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_20
// CHECK:STDOUT:   %.loc13_20.5: %Cpp.long = converted %b.ref.loc13, %.loc13_20.4
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.specific_fn.loc13: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.BitAndWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.specific_fn.dcd797.2]
// CHECK:STDOUT:   %bound_method.loc13_20.4: <bound method> = bound_method %a.ref.loc13_18, %Cpp.long.as.BitAndWith.impl.Op.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_22: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_22: init %Cpp.long = call %bound_method.loc13_22(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_22
// CHECK:STDOUT:   %.loc13_22.2: %Cpp.long = converted %b.ref.loc13, %.loc13_22.1
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.call.loc13: init %Cpp.long = call %bound_method.loc13_20.4(%a.ref.loc13_18, %.loc13_22.2)
// CHECK:STDOUT:   %a.ref.loc13_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc13: <specific function> = specific_function %AssertSameType.ref.loc13, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc13_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.BitAndWith.impl.Op.call.loc13
// CHECK:STDOUT:   %.loc13_20.7: %Cpp.long = converted %Cpp.long.as.BitAndWith.impl.Op.call.loc13, %.loc13_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc13: init %empty_tuple.type = call %AssertSameType.specific_fn.loc13(%.loc13_20.7, %a.ref.loc13_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc14: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc14_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc14: %.faa = impl_witness_access constants.%BitOrWith.impl_witness.b0c, element1 [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.d9c870.2]
// CHECK:STDOUT:   %bound_method.loc14_20.1: <bound method> = bound_method %a.ref.loc14_18, %impl.elem1.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc14_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc14_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem1.loc14, @Cpp.long.as.BitOrWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.specific_fn.03ab4b.1]
// CHECK:STDOUT:   %bound_method.loc14_20.2: <bound method> = bound_method %a.ref.loc14_18, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_20.3: %Cpp.long.as.BitOrWith.impl.Op.type.dccbc7.1 = specific_constant imports.%Core.Op.788, @Cpp.long.as.BitOrWith.impl.fc3(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.d9c870.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.BitOrWith.impl.Op.type.dccbc7.1 = name_ref Op, %.loc14_20.3 [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.d9c870.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.bound.loc14: <bound method> = bound_method %a.ref.loc14_18, %Op.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_20.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_20: init %Cpp.long = call %bound_method.loc14_20.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_20
// CHECK:STDOUT:   %.loc14_20.5: %Cpp.long = converted %b.ref.loc14, %.loc14_20.4
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.specific_fn.loc14: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.BitOrWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.specific_fn.03ab4b.2]
// CHECK:STDOUT:   %bound_method.loc14_20.4: <bound method> = bound_method %a.ref.loc14_18, %Cpp.long.as.BitOrWith.impl.Op.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_22: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_22: init %Cpp.long = call %bound_method.loc14_22(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_22
// CHECK:STDOUT:   %.loc14_22.2: %Cpp.long = converted %b.ref.loc14, %.loc14_22.1
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.call.loc14: init %Cpp.long = call %bound_method.loc14_20.4(%a.ref.loc14_18, %.loc14_22.2)
// CHECK:STDOUT:   %a.ref.loc14_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc14: <specific function> = specific_function %AssertSameType.ref.loc14, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc14_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.BitOrWith.impl.Op.call.loc14
// CHECK:STDOUT:   %.loc14_20.7: %Cpp.long = converted %Cpp.long.as.BitOrWith.impl.Op.call.loc14, %.loc14_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc14: init %empty_tuple.type = call %AssertSameType.specific_fn.loc14(%.loc14_20.7, %a.ref.loc14_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc15: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc15_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.f2f = impl_witness_access constants.%BitXorWith.impl_witness.8a0, element1 [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.48a9f4.2]
// CHECK:STDOUT:   %bound_method.loc15_20.1: <bound method> = bound_method %a.ref.loc15_18, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc15_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc15_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc15_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc15_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long.as.BitXorWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.specific_fn.4ec2b6.1]
// CHECK:STDOUT:   %bound_method.loc15_20.2: <bound method> = bound_method %a.ref.loc15_18, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_20.3: %Cpp.long.as.BitXorWith.impl.Op.type.7fe348.1 = specific_constant imports.%Core.Op.c61, @Cpp.long.as.BitXorWith.impl.732(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.48a9f4.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long.as.BitXorWith.impl.Op.type.7fe348.1 = name_ref Op, %.loc15_20.3 [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.48a9f4.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.bound.loc15: <bound method> = bound_method %a.ref.loc15_18, %Op.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_20.3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_20: init %Cpp.long = call %bound_method.loc15_20.3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_20
// CHECK:STDOUT:   %.loc15_20.5: %Cpp.long = converted %b.ref.loc15, %.loc15_20.4
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.specific_fn.loc15: <specific function> = specific_function %Op.ref.loc15, @Cpp.long.as.BitXorWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.specific_fn.4ec2b6.2]
// CHECK:STDOUT:   %bound_method.loc15_20.4: <bound method> = bound_method %a.ref.loc15_18, %Cpp.long.as.BitXorWith.impl.Op.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_22: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_22: init %Cpp.long = call %bound_method.loc15_22(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_22
// CHECK:STDOUT:   %.loc15_22.2: %Cpp.long = converted %b.ref.loc15, %.loc15_22.1
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.call.loc15: init %Cpp.long = call %bound_method.loc15_20.4(%a.ref.loc15_18, %.loc15_22.2)
// CHECK:STDOUT:   %a.ref.loc15_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc15: <specific function> = specific_function %AssertSameType.ref.loc15, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc15_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.BitXorWith.impl.Op.call.loc15
// CHECK:STDOUT:   %.loc15_20.7: %Cpp.long = converted %Cpp.long.as.BitXorWith.impl.Op.call.loc15, %.loc15_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc15: init %empty_tuple.type = call %AssertSameType.specific_fn.loc15(%.loc15_20.7, %a.ref.loc15_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc16: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc16_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc16: %.2be = impl_witness_access constants.%LeftShiftWith.impl_witness.982, element1 [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.9105a1.2]
// CHECK:STDOUT:   %bound_method.loc16_20.1: <bound method> = bound_method %a.ref.loc16_18, %impl.elem1.loc16
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc16_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc16_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc16_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc16_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem1.loc16, @Cpp.long.as.LeftShiftWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.1fda18.1]
// CHECK:STDOUT:   %bound_method.loc16_20.2: <bound method> = bound_method %a.ref.loc16_18, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_20.3: %Cpp.long.as.LeftShiftWith.impl.Op.type.1d34cb.1 = specific_constant imports.%Core.Op.2db, @Cpp.long.as.LeftShiftWith.impl.272(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.9105a1.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long.as.LeftShiftWith.impl.Op.type.1d34cb.1 = name_ref Op, %.loc16_20.3 [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.9105a1.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.bound.loc16: <bound method> = bound_method %a.ref.loc16_18, %Op.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_20.3: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_20: init %Cpp.long = call %bound_method.loc16_20.3(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_20
// CHECK:STDOUT:   %.loc16_20.5: %Cpp.long = converted %b.ref.loc16, %.loc16_20.4
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.loc16: <specific function> = specific_function %Op.ref.loc16, @Cpp.long.as.LeftShiftWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.1fda18.2]
// CHECK:STDOUT:   %bound_method.loc16_20.4: <bound method> = bound_method %a.ref.loc16_18, %Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_23: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_23: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_23
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_23: init %Cpp.long = call %bound_method.loc16_23(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_23.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_23
// CHECK:STDOUT:   %.loc16_23.2: %Cpp.long = converted %b.ref.loc16, %.loc16_23.1
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.call.loc16: init %Cpp.long = call %bound_method.loc16_20.4(%a.ref.loc16_18, %.loc16_23.2)
// CHECK:STDOUT:   %a.ref.loc16_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc16: <specific function> = specific_function %AssertSameType.ref.loc16, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc16_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.LeftShiftWith.impl.Op.call.loc16
// CHECK:STDOUT:   %.loc16_20.7: %Cpp.long = converted %Cpp.long.as.LeftShiftWith.impl.Op.call.loc16, %.loc16_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc16: init %empty_tuple.type = call %AssertSameType.specific_fn.loc16(%.loc16_20.7, %a.ref.loc16_26)
// CHECK:STDOUT:   %AssertSameType.ref.loc17: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc17_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc17: %.dac = impl_witness_access constants.%RightShiftWith.impl_witness.ccb, element1 [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.50edfc.2]
// CHECK:STDOUT:   %bound_method.loc17_20.1: <bound method> = bound_method %a.ref.loc17_18, %impl.elem1.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc17_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc17_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc17_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc17_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem1.loc17, @Cpp.long.as.RightShiftWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.specific_fn.012106.1]
// CHECK:STDOUT:   %bound_method.loc17_20.2: <bound method> = bound_method %a.ref.loc17_18, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_20.3: %Cpp.long.as.RightShiftWith.impl.Op.type.e35e06.1 = specific_constant imports.%Core.Op.330, @Cpp.long.as.RightShiftWith.impl.3d6(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.50edfc.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.RightShiftWith.impl.Op.type.e35e06.1 = name_ref Op, %.loc17_20.3 [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.50edfc.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.bound.loc17: <bound method> = bound_method %a.ref.loc17_18, %Op.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_20.3: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_20: init %Cpp.long = call %bound_method.loc17_20.3(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_20
// CHECK:STDOUT:   %.loc17_20.5: %Cpp.long = converted %b.ref.loc17, %.loc17_20.4
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.specific_fn.loc17: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.RightShiftWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.specific_fn.012106.2]
// CHECK:STDOUT:   %bound_method.loc17_20.4: <bound method> = bound_method %a.ref.loc17_18, %Cpp.long.as.RightShiftWith.impl.Op.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_23: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_23: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_23
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_23: init %Cpp.long = call %bound_method.loc17_23(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_23.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_23
// CHECK:STDOUT:   %.loc17_23.2: %Cpp.long = converted %b.ref.loc17, %.loc17_23.1
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.call.loc17: init %Cpp.long = call %bound_method.loc17_20.4(%a.ref.loc17_18, %.loc17_23.2)
// CHECK:STDOUT:   %a.ref.loc17_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc17: <specific function> = specific_function %AssertSameType.ref.loc17, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc17_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.RightShiftWith.impl.Op.call.loc17
// CHECK:STDOUT:   %.loc17_20.7: %Cpp.long = converted %Cpp.long.as.RightShiftWith.impl.Op.call.loc17, %.loc17_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc17: init %empty_tuple.type = call %AssertSameType.specific_fn.loc17(%.loc17_20.7, %a.ref.loc17_26)
// CHECK:STDOUT:   %AssertSameType.ref.loc19: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc19_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc19: %.9d7 = impl_witness_access constants.%BitAndWith.impl_witness.0fc, element1 [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.98f052.2]
// CHECK:STDOUT:   %bound_method.loc19_20.1: <bound method> = bound_method %a.ref.loc19_18, %impl.elem1.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_20.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc19_20.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_20.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_20.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc19_20.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc19_20.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem1.loc19, @Cpp.long.as.BitAndWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.specific_fn.af17b5.1]
// CHECK:STDOUT:   %bound_method.loc19_20.2: <bound method> = bound_method %a.ref.loc19_18, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_20.3: %Cpp.long.as.BitAndWith.impl.Op.type.ae3ddd.1 = specific_constant imports.%Core.Op.c8a, @Cpp.long.as.BitAndWith.impl.334(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.98f052.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.BitAndWith.impl.Op.type.ae3ddd.1 = name_ref Op, %.loc19_20.3 [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.98f052.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.bound.loc19: <bound method> = bound_method %a.ref.loc19_18, %Op.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_20.3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_20: init %Cpp.long = call %bound_method.loc19_20.3(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_20.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_20.5: %Cpp.long = converted %int_1.loc19, %.loc19_20.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.specific_fn.loc19: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.BitAndWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.specific_fn.af17b5.2]
// CHECK:STDOUT:   %bound_method.loc19_20.4: <bound method> = bound_method %a.ref.loc19_18, %Cpp.long.as.BitAndWith.impl.Op.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_22: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_22: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_22: init %Cpp.long = call %bound_method.loc19_22(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_22.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_22.2: %Cpp.long = converted %int_1.loc19, %.loc19_22.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.call.loc19: init %Cpp.long = call %bound_method.loc19_20.4(%a.ref.loc19_18, %.loc19_22.2)
// CHECK:STDOUT:   %a.ref.loc19_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc19: <specific function> = specific_function %AssertSameType.ref.loc19, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc19_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.BitAndWith.impl.Op.call.loc19
// CHECK:STDOUT:   %.loc19_20.7: %Cpp.long = converted %Cpp.long.as.BitAndWith.impl.Op.call.loc19, %.loc19_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc19: init %empty_tuple.type = call %AssertSameType.specific_fn.loc19(%.loc19_20.7, %a.ref.loc19_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc20: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc20_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc20: %.a50 = impl_witness_access constants.%BitOrWith.impl_witness.39f, element1 [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.664d1e.2]
// CHECK:STDOUT:   %bound_method.loc20_20.1: <bound method> = bound_method %a.ref.loc20_18, %impl.elem1.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_20.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc20_20.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_20.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_20.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc20_20.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc20_20.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem1.loc20, @Cpp.long.as.BitOrWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.specific_fn.be7132.1]
// CHECK:STDOUT:   %bound_method.loc20_20.2: <bound method> = bound_method %a.ref.loc20_18, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_20.3: %Cpp.long.as.BitOrWith.impl.Op.type.331453.1 = specific_constant imports.%Core.Op.788, @Cpp.long.as.BitOrWith.impl.fc3(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.664d1e.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.BitOrWith.impl.Op.type.331453.1 = name_ref Op, %.loc20_20.3 [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.664d1e.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.bound.loc20: <bound method> = bound_method %a.ref.loc20_18, %Op.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_20.3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_20: init %Cpp.long = call %bound_method.loc20_20.3(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_20.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_20.5: %Cpp.long = converted %int_1.loc20, %.loc20_20.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.specific_fn.loc20: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.BitOrWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.specific_fn.be7132.2]
// CHECK:STDOUT:   %bound_method.loc20_20.4: <bound method> = bound_method %a.ref.loc20_18, %Cpp.long.as.BitOrWith.impl.Op.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_22: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_22: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_22: init %Cpp.long = call %bound_method.loc20_22(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_22.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_22.2: %Cpp.long = converted %int_1.loc20, %.loc20_22.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.call.loc20: init %Cpp.long = call %bound_method.loc20_20.4(%a.ref.loc20_18, %.loc20_22.2)
// CHECK:STDOUT:   %a.ref.loc20_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc20: <specific function> = specific_function %AssertSameType.ref.loc20, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc20_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.BitOrWith.impl.Op.call.loc20
// CHECK:STDOUT:   %.loc20_20.7: %Cpp.long = converted %Cpp.long.as.BitOrWith.impl.Op.call.loc20, %.loc20_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc20: init %empty_tuple.type = call %AssertSameType.specific_fn.loc20(%.loc20_20.7, %a.ref.loc20_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc21: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc21_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc21: %.8e7 = impl_witness_access constants.%BitXorWith.impl_witness.001, element1 [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.5b3046.2]
// CHECK:STDOUT:   %bound_method.loc21_20.1: <bound method> = bound_method %a.ref.loc21_18, %impl.elem1.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_20.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc21_20.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_20.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_20.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc21_20.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc21_20.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem1.loc21, @Cpp.long.as.BitXorWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.specific_fn.f1c9e6.1]
// CHECK:STDOUT:   %bound_method.loc21_20.2: <bound method> = bound_method %a.ref.loc21_18, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_20.3: %Cpp.long.as.BitXorWith.impl.Op.type.97a9a2.1 = specific_constant imports.%Core.Op.c61, @Cpp.long.as.BitXorWith.impl.732(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.5b3046.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long.as.BitXorWith.impl.Op.type.97a9a2.1 = name_ref Op, %.loc21_20.3 [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.5b3046.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.bound.loc21: <bound method> = bound_method %a.ref.loc21_18, %Op.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_20.3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_20: init %Cpp.long = call %bound_method.loc21_20.3(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_20.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_20.5: %Cpp.long = converted %int_1.loc21, %.loc21_20.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.specific_fn.loc21: <specific function> = specific_function %Op.ref.loc21, @Cpp.long.as.BitXorWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.specific_fn.f1c9e6.2]
// CHECK:STDOUT:   %bound_method.loc21_20.4: <bound method> = bound_method %a.ref.loc21_18, %Cpp.long.as.BitXorWith.impl.Op.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_22: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_22: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_22: init %Cpp.long = call %bound_method.loc21_22(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_22.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_22.2: %Cpp.long = converted %int_1.loc21, %.loc21_22.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.call.loc21: init %Cpp.long = call %bound_method.loc21_20.4(%a.ref.loc21_18, %.loc21_22.2)
// CHECK:STDOUT:   %a.ref.loc21_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc21: <specific function> = specific_function %AssertSameType.ref.loc21, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc21_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.BitXorWith.impl.Op.call.loc21
// CHECK:STDOUT:   %.loc21_20.7: %Cpp.long = converted %Cpp.long.as.BitXorWith.impl.Op.call.loc21, %.loc21_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc21: init %empty_tuple.type = call %AssertSameType.specific_fn.loc21(%.loc21_20.7, %a.ref.loc21_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc22: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc22_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.0a1 = impl_witness_access constants.%LeftShiftWith.impl_witness.a1c, element1 [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.f145d8.2]
// CHECK:STDOUT:   %bound_method.loc22_20.1: <bound method> = bound_method %a.ref.loc22_18, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_20.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc22_20.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_20.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22_20.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc22_20.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc22_20.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long.as.LeftShiftWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.abc44e.1]
// CHECK:STDOUT:   %bound_method.loc22_20.2: <bound method> = bound_method %a.ref.loc22_18, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_20.3: %Cpp.long.as.LeftShiftWith.impl.Op.type.d93838.1 = specific_constant imports.%Core.Op.2db, @Cpp.long.as.LeftShiftWith.impl.272(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.f145d8.1]
// CHECK:STDOUT:   %Op.ref.loc22: %Cpp.long.as.LeftShiftWith.impl.Op.type.d93838.1 = name_ref Op, %.loc22_20.3 [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.f145d8.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.bound.loc22: <bound method> = bound_method %a.ref.loc22_18, %Op.ref.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_20.3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_20: init %Cpp.long = call %bound_method.loc22_20.3(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_20.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_20.5: %Cpp.long = converted %int_1.loc22, %.loc22_20.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.loc22: <specific function> = specific_function %Op.ref.loc22, @Cpp.long.as.LeftShiftWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.abc44e.2]
// CHECK:STDOUT:   %bound_method.loc22_20.4: <bound method> = bound_method %a.ref.loc22_18, %Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_23: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_23: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_23: init %Cpp.long = call %bound_method.loc22_23(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_23.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_23 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_23.2: %Cpp.long = converted %int_1.loc22, %.loc22_23.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.call.loc22: init %Cpp.long = call %bound_method.loc22_20.4(%a.ref.loc22_18, %.loc22_23.2)
// CHECK:STDOUT:   %a.ref.loc22_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc22: <specific function> = specific_function %AssertSameType.ref.loc22, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc22_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.LeftShiftWith.impl.Op.call.loc22
// CHECK:STDOUT:   %.loc22_20.7: %Cpp.long = converted %Cpp.long.as.LeftShiftWith.impl.Op.call.loc22, %.loc22_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc22: init %empty_tuple.type = call %AssertSameType.specific_fn.loc22(%.loc22_20.7, %a.ref.loc22_26)
// CHECK:STDOUT:   %AssertSameType.ref.loc23: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc23_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc23: %.fab = impl_witness_access constants.%RightShiftWith.impl_witness.5fe, element1 [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.249bc3.2]
// CHECK:STDOUT:   %bound_method.loc23_20.1: <bound method> = bound_method %a.ref.loc23_18, %impl.elem1.loc23
// CHECK:STDOUT:   %ImplicitAs.facet.loc23_20.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc23_20.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc23_20.1 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %ImplicitAs.facet.loc23_20.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc23_20.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc23_20.2 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem1.loc23, @Cpp.long.as.RightShiftWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.specific_fn.c6040e.1]
// CHECK:STDOUT:   %bound_method.loc23_20.2: <bound method> = bound_method %a.ref.loc23_18, %specific_fn.loc23
// CHECK:STDOUT:   %.loc23_20.3: %Cpp.long.as.RightShiftWith.impl.Op.type.a6dc1c.1 = specific_constant imports.%Core.Op.330, @Cpp.long.as.RightShiftWith.impl.3d6(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.249bc3.1]
// CHECK:STDOUT:   %Op.ref.loc23: %Cpp.long.as.RightShiftWith.impl.Op.type.a6dc1c.1 = name_ref Op, %.loc23_20.3 [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.249bc3.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.bound.loc23: <bound method> = bound_method %a.ref.loc23_18, %Op.ref.loc23
// CHECK:STDOUT:   %impl.elem0.loc23_20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc23_20.3: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23_20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_20: init %Cpp.long = call %bound_method.loc23_20.3(%int_1.loc23) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23_20.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23_20.5: %Cpp.long = converted %int_1.loc23, %.loc23_20.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.specific_fn.loc23: <specific function> = specific_function %Op.ref.loc23, @Cpp.long.as.RightShiftWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.specific_fn.c6040e.2]
// CHECK:STDOUT:   %bound_method.loc23_20.4: <bound method> = bound_method %a.ref.loc23_18, %Cpp.long.as.RightShiftWith.impl.Op.specific_fn.loc23
// CHECK:STDOUT:   %impl.elem0.loc23_23: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc23_23: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23_23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_23: init %Cpp.long = call %bound_method.loc23_23(%int_1.loc23) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23_23.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_23 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23_23.2: %Cpp.long = converted %int_1.loc23, %.loc23_23.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.call.loc23: init %Cpp.long = call %bound_method.loc23_20.4(%a.ref.loc23_18, %.loc23_23.2)
// CHECK:STDOUT:   %a.ref.loc23_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc23: <specific function> = specific_function %AssertSameType.ref.loc23, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc23_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.RightShiftWith.impl.Op.call.loc23
// CHECK:STDOUT:   %.loc23_20.7: %Cpp.long = converted %Cpp.long.as.RightShiftWith.impl.Op.call.loc23, %.loc23_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc23: init %empty_tuple.type = call %AssertSameType.specific_fn.loc23(%.loc23_20.7, %a.ref.loc23_26)
// CHECK:STDOUT:   %AssertSameType.ref.loc25: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc25_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc25: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc25: %.bdd = impl_witness_access constants.%BitAndWith.impl_witness.fe3, element1 [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.a9ac84.2]
// CHECK:STDOUT:   %bound_method.loc25_20.1: <bound method> = bound_method %a.ref.loc25_18, %impl.elem1.loc25
// CHECK:STDOUT:   %ImplicitAs.facet.loc25_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc25_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc25_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc25_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc25_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc25_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem1.loc25, @Cpp.long.as.BitAndWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.specific_fn.dcd797.1]
// CHECK:STDOUT:   %bound_method.loc25_20.2: <bound method> = bound_method %a.ref.loc25_18, %specific_fn.loc25
// CHECK:STDOUT:   %.loc25_20.3: %Cpp.long.as.BitAndWith.impl.Op.type.202298.1 = specific_constant imports.%Core.Op.c8a, @Cpp.long.as.BitAndWith.impl.334(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.a9ac84.1]
// CHECK:STDOUT:   %Op.ref.loc25: %Cpp.long.as.BitAndWith.impl.Op.type.202298.1 = name_ref Op, %.loc25_20.3 [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.a9ac84.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.bound.loc25: <bound method> = bound_method %a.ref.loc25_18, %Op.ref.loc25
// CHECK:STDOUT:   %impl.elem0.loc25_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_20.3: <bound method> = bound_method %b.ref.loc25, %impl.elem0.loc25_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc25_20: init %Cpp.long = call %bound_method.loc25_20.3(%b.ref.loc25)
// CHECK:STDOUT:   %.loc25_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc25_20
// CHECK:STDOUT:   %.loc25_20.5: %Cpp.long = converted %b.ref.loc25, %.loc25_20.4
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.specific_fn.loc25: <specific function> = specific_function %Op.ref.loc25, @Cpp.long.as.BitAndWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndWith.impl.Op.specific_fn.dcd797.2]
// CHECK:STDOUT:   %bound_method.loc25_20.4: <bound method> = bound_method %a.ref.loc25_18, %Cpp.long.as.BitAndWith.impl.Op.specific_fn.loc25
// CHECK:STDOUT:   %impl.elem0.loc25_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_22: <bound method> = bound_method %b.ref.loc25, %impl.elem0.loc25_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc25_22: init %Cpp.long = call %bound_method.loc25_22(%b.ref.loc25)
// CHECK:STDOUT:   %.loc25_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc25_22
// CHECK:STDOUT:   %.loc25_22.2: %Cpp.long = converted %b.ref.loc25, %.loc25_22.1
// CHECK:STDOUT:   %Cpp.long.as.BitAndWith.impl.Op.call.loc25: init %Cpp.long = call %bound_method.loc25_20.4(%a.ref.loc25_18, %.loc25_22.2)
// CHECK:STDOUT:   %a.ref.loc25_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc25: <specific function> = specific_function %AssertSameType.ref.loc25, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc25_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.BitAndWith.impl.Op.call.loc25
// CHECK:STDOUT:   %.loc25_20.7: %Cpp.long = converted %Cpp.long.as.BitAndWith.impl.Op.call.loc25, %.loc25_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc25: init %empty_tuple.type = call %AssertSameType.specific_fn.loc25(%.loc25_20.7, %a.ref.loc25_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc26: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc26_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc26: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc26: %.faa = impl_witness_access constants.%BitOrWith.impl_witness.b0c, element1 [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.d9c870.2]
// CHECK:STDOUT:   %bound_method.loc26_20.1: <bound method> = bound_method %a.ref.loc26_18, %impl.elem1.loc26
// CHECK:STDOUT:   %ImplicitAs.facet.loc26_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc26_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc26_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc26_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc26_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc26_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem1.loc26, @Cpp.long.as.BitOrWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.specific_fn.03ab4b.1]
// CHECK:STDOUT:   %bound_method.loc26_20.2: <bound method> = bound_method %a.ref.loc26_18, %specific_fn.loc26
// CHECK:STDOUT:   %.loc26_20.3: %Cpp.long.as.BitOrWith.impl.Op.type.dccbc7.1 = specific_constant imports.%Core.Op.788, @Cpp.long.as.BitOrWith.impl.fc3(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.d9c870.1]
// CHECK:STDOUT:   %Op.ref.loc26: %Cpp.long.as.BitOrWith.impl.Op.type.dccbc7.1 = name_ref Op, %.loc26_20.3 [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.d9c870.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.bound.loc26: <bound method> = bound_method %a.ref.loc26_18, %Op.ref.loc26
// CHECK:STDOUT:   %impl.elem0.loc26_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_20.3: <bound method> = bound_method %b.ref.loc26, %impl.elem0.loc26_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc26_20: init %Cpp.long = call %bound_method.loc26_20.3(%b.ref.loc26)
// CHECK:STDOUT:   %.loc26_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc26_20
// CHECK:STDOUT:   %.loc26_20.5: %Cpp.long = converted %b.ref.loc26, %.loc26_20.4
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.specific_fn.loc26: <specific function> = specific_function %Op.ref.loc26, @Cpp.long.as.BitOrWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrWith.impl.Op.specific_fn.03ab4b.2]
// CHECK:STDOUT:   %bound_method.loc26_20.4: <bound method> = bound_method %a.ref.loc26_18, %Cpp.long.as.BitOrWith.impl.Op.specific_fn.loc26
// CHECK:STDOUT:   %impl.elem0.loc26_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_22: <bound method> = bound_method %b.ref.loc26, %impl.elem0.loc26_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc26_22: init %Cpp.long = call %bound_method.loc26_22(%b.ref.loc26)
// CHECK:STDOUT:   %.loc26_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc26_22
// CHECK:STDOUT:   %.loc26_22.2: %Cpp.long = converted %b.ref.loc26, %.loc26_22.1
// CHECK:STDOUT:   %Cpp.long.as.BitOrWith.impl.Op.call.loc26: init %Cpp.long = call %bound_method.loc26_20.4(%a.ref.loc26_18, %.loc26_22.2)
// CHECK:STDOUT:   %a.ref.loc26_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc26: <specific function> = specific_function %AssertSameType.ref.loc26, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc26_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.BitOrWith.impl.Op.call.loc26
// CHECK:STDOUT:   %.loc26_20.7: %Cpp.long = converted %Cpp.long.as.BitOrWith.impl.Op.call.loc26, %.loc26_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc26: init %empty_tuple.type = call %AssertSameType.specific_fn.loc26(%.loc26_20.7, %a.ref.loc26_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc27: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc27_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc27: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc27: %.f2f = impl_witness_access constants.%BitXorWith.impl_witness.8a0, element1 [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.48a9f4.2]
// CHECK:STDOUT:   %bound_method.loc27_20.1: <bound method> = bound_method %a.ref.loc27_18, %impl.elem1.loc27
// CHECK:STDOUT:   %ImplicitAs.facet.loc27_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc27_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc27_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc27_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc27_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc27_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem1.loc27, @Cpp.long.as.BitXorWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.specific_fn.4ec2b6.1]
// CHECK:STDOUT:   %bound_method.loc27_20.2: <bound method> = bound_method %a.ref.loc27_18, %specific_fn.loc27
// CHECK:STDOUT:   %.loc27_20.3: %Cpp.long.as.BitXorWith.impl.Op.type.7fe348.1 = specific_constant imports.%Core.Op.c61, @Cpp.long.as.BitXorWith.impl.732(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.48a9f4.1]
// CHECK:STDOUT:   %Op.ref.loc27: %Cpp.long.as.BitXorWith.impl.Op.type.7fe348.1 = name_ref Op, %.loc27_20.3 [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.48a9f4.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.bound.loc27: <bound method> = bound_method %a.ref.loc27_18, %Op.ref.loc27
// CHECK:STDOUT:   %impl.elem0.loc27_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_20.3: <bound method> = bound_method %b.ref.loc27, %impl.elem0.loc27_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc27_20: init %Cpp.long = call %bound_method.loc27_20.3(%b.ref.loc27)
// CHECK:STDOUT:   %.loc27_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc27_20
// CHECK:STDOUT:   %.loc27_20.5: %Cpp.long = converted %b.ref.loc27, %.loc27_20.4
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.specific_fn.loc27: <specific function> = specific_function %Op.ref.loc27, @Cpp.long.as.BitXorWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorWith.impl.Op.specific_fn.4ec2b6.2]
// CHECK:STDOUT:   %bound_method.loc27_20.4: <bound method> = bound_method %a.ref.loc27_18, %Cpp.long.as.BitXorWith.impl.Op.specific_fn.loc27
// CHECK:STDOUT:   %impl.elem0.loc27_22: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_22: <bound method> = bound_method %b.ref.loc27, %impl.elem0.loc27_22
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc27_22: init %Cpp.long = call %bound_method.loc27_22(%b.ref.loc27)
// CHECK:STDOUT:   %.loc27_22.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc27_22
// CHECK:STDOUT:   %.loc27_22.2: %Cpp.long = converted %b.ref.loc27, %.loc27_22.1
// CHECK:STDOUT:   %Cpp.long.as.BitXorWith.impl.Op.call.loc27: init %Cpp.long = call %bound_method.loc27_20.4(%a.ref.loc27_18, %.loc27_22.2)
// CHECK:STDOUT:   %a.ref.loc27_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc27: <specific function> = specific_function %AssertSameType.ref.loc27, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc27_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.BitXorWith.impl.Op.call.loc27
// CHECK:STDOUT:   %.loc27_20.7: %Cpp.long = converted %Cpp.long.as.BitXorWith.impl.Op.call.loc27, %.loc27_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc27: init %empty_tuple.type = call %AssertSameType.specific_fn.loc27(%.loc27_20.7, %a.ref.loc27_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc28: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc28_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc28: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc28: %.2be = impl_witness_access constants.%LeftShiftWith.impl_witness.982, element1 [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.9105a1.2]
// CHECK:STDOUT:   %bound_method.loc28_20.1: <bound method> = bound_method %a.ref.loc28_18, %impl.elem1.loc28
// CHECK:STDOUT:   %ImplicitAs.facet.loc28_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc28_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc28_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc28_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc28_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc28_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem1.loc28, @Cpp.long.as.LeftShiftWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.1fda18.1]
// CHECK:STDOUT:   %bound_method.loc28_20.2: <bound method> = bound_method %a.ref.loc28_18, %specific_fn.loc28
// CHECK:STDOUT:   %.loc28_20.3: %Cpp.long.as.LeftShiftWith.impl.Op.type.1d34cb.1 = specific_constant imports.%Core.Op.2db, @Cpp.long.as.LeftShiftWith.impl.272(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.9105a1.1]
// CHECK:STDOUT:   %Op.ref.loc28: %Cpp.long.as.LeftShiftWith.impl.Op.type.1d34cb.1 = name_ref Op, %.loc28_20.3 [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.9105a1.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.bound.loc28: <bound method> = bound_method %a.ref.loc28_18, %Op.ref.loc28
// CHECK:STDOUT:   %impl.elem0.loc28_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_20.3: <bound method> = bound_method %b.ref.loc28, %impl.elem0.loc28_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc28_20: init %Cpp.long = call %bound_method.loc28_20.3(%b.ref.loc28)
// CHECK:STDOUT:   %.loc28_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc28_20
// CHECK:STDOUT:   %.loc28_20.5: %Cpp.long = converted %b.ref.loc28, %.loc28_20.4
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.loc28: <specific function> = specific_function %Op.ref.loc28, @Cpp.long.as.LeftShiftWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.1fda18.2]
// CHECK:STDOUT:   %bound_method.loc28_20.4: <bound method> = bound_method %a.ref.loc28_18, %Cpp.long.as.LeftShiftWith.impl.Op.specific_fn.loc28
// CHECK:STDOUT:   %impl.elem0.loc28_23: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_23: <bound method> = bound_method %b.ref.loc28, %impl.elem0.loc28_23
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc28_23: init %Cpp.long = call %bound_method.loc28_23(%b.ref.loc28)
// CHECK:STDOUT:   %.loc28_23.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc28_23
// CHECK:STDOUT:   %.loc28_23.2: %Cpp.long = converted %b.ref.loc28, %.loc28_23.1
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftWith.impl.Op.call.loc28: init %Cpp.long = call %bound_method.loc28_20.4(%a.ref.loc28_18, %.loc28_23.2)
// CHECK:STDOUT:   %a.ref.loc28_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc28: <specific function> = specific_function %AssertSameType.ref.loc28, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc28_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.LeftShiftWith.impl.Op.call.loc28
// CHECK:STDOUT:   %.loc28_20.7: %Cpp.long = converted %Cpp.long.as.LeftShiftWith.impl.Op.call.loc28, %.loc28_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc28: init %empty_tuple.type = call %AssertSameType.specific_fn.loc28(%.loc28_20.7, %a.ref.loc28_26)
// CHECK:STDOUT:   %AssertSameType.ref.loc29: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc29_18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc29: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc29: %.dac = impl_witness_access constants.%RightShiftWith.impl_witness.ccb, element1 [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.50edfc.2]
// CHECK:STDOUT:   %bound_method.loc29_20.1: <bound method> = bound_method %a.ref.loc29_18, %impl.elem1.loc29
// CHECK:STDOUT:   %ImplicitAs.facet.loc29_20.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc29_20.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc29_20.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc29_20.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc29_20.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc29_20.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem1.loc29, @Cpp.long.as.RightShiftWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.specific_fn.012106.1]
// CHECK:STDOUT:   %bound_method.loc29_20.2: <bound method> = bound_method %a.ref.loc29_18, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29_20.3: %Cpp.long.as.RightShiftWith.impl.Op.type.e35e06.1 = specific_constant imports.%Core.Op.330, @Cpp.long.as.RightShiftWith.impl.3d6(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.50edfc.1]
// CHECK:STDOUT:   %Op.ref.loc29: %Cpp.long.as.RightShiftWith.impl.Op.type.e35e06.1 = name_ref Op, %.loc29_20.3 [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.50edfc.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.bound.loc29: <bound method> = bound_method %a.ref.loc29_18, %Op.ref.loc29
// CHECK:STDOUT:   %impl.elem0.loc29_20: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_20.3: <bound method> = bound_method %b.ref.loc29, %impl.elem0.loc29_20
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29_20: init %Cpp.long = call %bound_method.loc29_20.3(%b.ref.loc29)
// CHECK:STDOUT:   %.loc29_20.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29_20
// CHECK:STDOUT:   %.loc29_20.5: %Cpp.long = converted %b.ref.loc29, %.loc29_20.4
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.specific_fn.loc29: <specific function> = specific_function %Op.ref.loc29, @Cpp.long.as.RightShiftWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftWith.impl.Op.specific_fn.012106.2]
// CHECK:STDOUT:   %bound_method.loc29_20.4: <bound method> = bound_method %a.ref.loc29_18, %Cpp.long.as.RightShiftWith.impl.Op.specific_fn.loc29
// CHECK:STDOUT:   %impl.elem0.loc29_23: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_23: <bound method> = bound_method %b.ref.loc29, %impl.elem0.loc29_23
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29_23: init %Cpp.long = call %bound_method.loc29_23(%b.ref.loc29)
// CHECK:STDOUT:   %.loc29_23.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29_23
// CHECK:STDOUT:   %.loc29_23.2: %Cpp.long = converted %b.ref.loc29, %.loc29_23.1
// CHECK:STDOUT:   %Cpp.long.as.RightShiftWith.impl.Op.call.loc29: init %Cpp.long = call %bound_method.loc29_20.4(%a.ref.loc29_18, %.loc29_23.2)
// CHECK:STDOUT:   %a.ref.loc29_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc29: <specific function> = specific_function %AssertSameType.ref.loc29, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc29_20.6: %Cpp.long = value_of_initializer %Cpp.long.as.RightShiftWith.impl.Op.call.loc29
// CHECK:STDOUT:   %.loc29_20.7: %Cpp.long = converted %Cpp.long.as.RightShiftWith.impl.Op.call.loc29, %.loc29_20.6
// CHECK:STDOUT:   %AssertSameType.call.loc29: init %empty_tuple.type = call %AssertSameType.specific_fn.loc29(%.loc29_20.7, %a.ref.loc29_26)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bitwise_heterogeneous_long_right_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AssertSameType.type: type = fn_type @AssertSameType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AssertSameType: %AssertSameType.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %BitAndWith.type.0b5: type = facet_type <@BitAndWith, @BitAndWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.e13: type = fn_type @BitAndWith.Op, @BitAndWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.57d: %ImplicitAs.type.819 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.type.b0dfa2.1: type = fn_type @T.binding.as_type.as.BitAndWith.impl.Op.1, @T.binding.as_type.as.BitAndWith.impl.2e8(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.ae0f20.1: %T.binding.as_type.as.BitAndWith.impl.Op.type.b0dfa2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.type.b0dfa2.2: type = fn_type @T.binding.as_type.as.BitAndWith.impl.Op.2, @T.binding.as_type.as.BitAndWith.impl.2e8(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.ae0f20.2: %T.binding.as_type.as.BitAndWith.impl.Op.type.b0dfa2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness.a02: <witness> = impl_witness imports.%BitAndWith.impl_witness_table.1c0, @T.binding.as_type.as.BitAndWith.impl.2e8(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.type.f24b53.1: type = fn_type @T.binding.as_type.as.BitAndWith.impl.Op.2, @T.binding.as_type.as.BitAndWith.impl.2e8(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.3425be.1: %T.binding.as_type.as.BitAndWith.impl.Op.type.f24b53.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.type.f24b53.2: type = fn_type @T.binding.as_type.as.BitAndWith.impl.Op.1, @T.binding.as_type.as.BitAndWith.impl.2e8(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.3425be.2: %T.binding.as_type.as.BitAndWith.impl.Op.type.f24b53.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.2fb: %BitAndWith.type.0b5 = facet_value %i32, (%BitAndWith.impl_witness.a02) [concrete]
// CHECK:STDOUT:   %.419: type = fn_type_with_self_type %BitAndWith.Op.type.e13, %BitAndWith.facet.2fb [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.f76196.1: <specific function> = specific_function %T.binding.as_type.as.BitAndWith.impl.Op.3425be.2, @T.binding.as_type.as.BitAndWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.f76196.2: <specific function> = specific_function %T.binding.as_type.as.BitAndWith.impl.Op.3425be.1, @T.binding.as_type.as.BitAndWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.c45: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %AssertSameType.specific_fn: <specific function> = specific_function %AssertSameType, @AssertSameType(%Cpp.long) [concrete]
// CHECK:STDOUT:   %BitOrWith.type.b22: type = facet_type <@BitOrWith, @BitOrWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %BitOrWith.Op.type.a1b: type = fn_type @BitOrWith.Op, @BitOrWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.type.f4a873.1: type = fn_type @T.binding.as_type.as.BitOrWith.impl.Op.1, @T.binding.as_type.as.BitOrWith.impl.e42(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.53d67c.1: %T.binding.as_type.as.BitOrWith.impl.Op.type.f4a873.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.type.f4a873.2: type = fn_type @T.binding.as_type.as.BitOrWith.impl.Op.2, @T.binding.as_type.as.BitOrWith.impl.e42(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.53d67c.2: %T.binding.as_type.as.BitOrWith.impl.Op.type.f4a873.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrWith.impl_witness.d0a: <witness> = impl_witness imports.%BitOrWith.impl_witness_table.946, @T.binding.as_type.as.BitOrWith.impl.e42(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.type.c5501d.1: type = fn_type @T.binding.as_type.as.BitOrWith.impl.Op.2, @T.binding.as_type.as.BitOrWith.impl.e42(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.c9149f.1: %T.binding.as_type.as.BitOrWith.impl.Op.type.c5501d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.type.c5501d.2: type = fn_type @T.binding.as_type.as.BitOrWith.impl.Op.1, @T.binding.as_type.as.BitOrWith.impl.e42(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.c9149f.2: %T.binding.as_type.as.BitOrWith.impl.Op.type.c5501d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.ea7: %BitOrWith.type.b22 = facet_value %i32, (%BitOrWith.impl_witness.d0a) [concrete]
// CHECK:STDOUT:   %.2e2: type = fn_type_with_self_type %BitOrWith.Op.type.a1b, %BitOrWith.facet.ea7 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.fbab3e.1: <specific function> = specific_function %T.binding.as_type.as.BitOrWith.impl.Op.c9149f.2, @T.binding.as_type.as.BitOrWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.fbab3e.2: <specific function> = specific_function %T.binding.as_type.as.BitOrWith.impl.Op.c9149f.1, @T.binding.as_type.as.BitOrWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitXorWith.type.87f: type = facet_type <@BitXorWith, @BitXorWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %BitXorWith.Op.type.5cf: type = fn_type @BitXorWith.Op, @BitXorWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.type.fdc423.1: type = fn_type @T.binding.as_type.as.BitXorWith.impl.Op.1, @T.binding.as_type.as.BitXorWith.impl.878(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.659e6b.1: %T.binding.as_type.as.BitXorWith.impl.Op.type.fdc423.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.type.fdc423.2: type = fn_type @T.binding.as_type.as.BitXorWith.impl.Op.2, @T.binding.as_type.as.BitXorWith.impl.878(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.659e6b.2: %T.binding.as_type.as.BitXorWith.impl.Op.type.fdc423.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorWith.impl_witness.e8e: <witness> = impl_witness imports.%BitXorWith.impl_witness_table.39f, @T.binding.as_type.as.BitXorWith.impl.878(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.type.42bd3d.1: type = fn_type @T.binding.as_type.as.BitXorWith.impl.Op.2, @T.binding.as_type.as.BitXorWith.impl.878(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.c338c8.1: %T.binding.as_type.as.BitXorWith.impl.Op.type.42bd3d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.type.42bd3d.2: type = fn_type @T.binding.as_type.as.BitXorWith.impl.Op.1, @T.binding.as_type.as.BitXorWith.impl.878(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.c338c8.2: %T.binding.as_type.as.BitXorWith.impl.Op.type.42bd3d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.19e: %BitXorWith.type.87f = facet_value %i32, (%BitXorWith.impl_witness.e8e) [concrete]
// CHECK:STDOUT:   %.24d: type = fn_type_with_self_type %BitXorWith.Op.type.5cf, %BitXorWith.facet.19e [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.125c34.1: <specific function> = specific_function %T.binding.as_type.as.BitXorWith.impl.Op.c338c8.2, @T.binding.as_type.as.BitXorWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.125c34.2: <specific function> = specific_function %T.binding.as_type.as.BitXorWith.impl.Op.c338c8.1, @T.binding.as_type.as.BitXorWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.type.3f9: type = facet_type <@LeftShiftWith, @LeftShiftWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.Op.type.224: type = fn_type @LeftShiftWith.Op, @LeftShiftWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.type.278104.1: type = fn_type @T.binding.as_type.as.LeftShiftWith.impl.Op.1, @T.binding.as_type.as.LeftShiftWith.impl.b71(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.fdafb7.1: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.278104.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.type.278104.2: type = fn_type @T.binding.as_type.as.LeftShiftWith.impl.Op.2, @T.binding.as_type.as.LeftShiftWith.impl.b71(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.fdafb7.2: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.278104.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness.fb3: <witness> = impl_witness imports.%LeftShiftWith.impl_witness_table.ce8, @T.binding.as_type.as.LeftShiftWith.impl.b71(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.type.ab52f6.1: type = fn_type @T.binding.as_type.as.LeftShiftWith.impl.Op.2, @T.binding.as_type.as.LeftShiftWith.impl.b71(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.c3e103.1: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.ab52f6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.type.ab52f6.2: type = fn_type @T.binding.as_type.as.LeftShiftWith.impl.Op.1, @T.binding.as_type.as.LeftShiftWith.impl.b71(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.c3e103.2: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.ab52f6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.702: %LeftShiftWith.type.3f9 = facet_value %i32, (%LeftShiftWith.impl_witness.fb3) [concrete]
// CHECK:STDOUT:   %.509: type = fn_type_with_self_type %LeftShiftWith.Op.type.224, %LeftShiftWith.facet.702 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.547c25.1: <specific function> = specific_function %T.binding.as_type.as.LeftShiftWith.impl.Op.c3e103.2, @T.binding.as_type.as.LeftShiftWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.547c25.2: <specific function> = specific_function %T.binding.as_type.as.LeftShiftWith.impl.Op.c3e103.1, @T.binding.as_type.as.LeftShiftWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %RightShiftWith.type.995: type = facet_type <@RightShiftWith, @RightShiftWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %RightShiftWith.Op.type.fd9: type = fn_type @RightShiftWith.Op, @RightShiftWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.type.8ef087.1: type = fn_type @T.binding.as_type.as.RightShiftWith.impl.Op.1, @T.binding.as_type.as.RightShiftWith.impl.28b(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.3f8b9d.1: %T.binding.as_type.as.RightShiftWith.impl.Op.type.8ef087.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.type.8ef087.2: type = fn_type @T.binding.as_type.as.RightShiftWith.impl.Op.2, @T.binding.as_type.as.RightShiftWith.impl.28b(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.3f8b9d.2: %T.binding.as_type.as.RightShiftWith.impl.Op.type.8ef087.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftWith.impl_witness.7d0: <witness> = impl_witness imports.%RightShiftWith.impl_witness_table.050, @T.binding.as_type.as.RightShiftWith.impl.28b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.type.2fe33f.1: type = fn_type @T.binding.as_type.as.RightShiftWith.impl.Op.2, @T.binding.as_type.as.RightShiftWith.impl.28b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.44dea2.1: %T.binding.as_type.as.RightShiftWith.impl.Op.type.2fe33f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.type.2fe33f.2: type = fn_type @T.binding.as_type.as.RightShiftWith.impl.Op.1, @T.binding.as_type.as.RightShiftWith.impl.28b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.44dea2.2: %T.binding.as_type.as.RightShiftWith.impl.Op.type.2fe33f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.4bd: %RightShiftWith.type.995 = facet_value %i32, (%RightShiftWith.impl_witness.7d0) [concrete]
// CHECK:STDOUT:   %.9db: type = fn_type_with_self_type %RightShiftWith.Op.type.fd9, %RightShiftWith.facet.4bd [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.888028.1: <specific function> = specific_function %T.binding.as_type.as.RightShiftWith.impl.Op.44dea2.2, @T.binding.as_type.as.RightShiftWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.888028.2: <specific function> = specific_function %T.binding.as_type.as.RightShiftWith.impl.Op.44dea2.1, @T.binding.as_type.as.RightShiftWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness.11b: <witness> = impl_witness imports.%BitAndWith.impl_witness_table.1c0, @T.binding.as_type.as.BitAndWith.impl.2e8(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.type.371e74.1: type = fn_type @T.binding.as_type.as.BitAndWith.impl.Op.2, @T.binding.as_type.as.BitAndWith.impl.2e8(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.657411.1: %T.binding.as_type.as.BitAndWith.impl.Op.type.371e74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.type.371e74.2: type = fn_type @T.binding.as_type.as.BitAndWith.impl.Op.1, @T.binding.as_type.as.BitAndWith.impl.2e8(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.657411.2: %T.binding.as_type.as.BitAndWith.impl.Op.type.371e74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.c73: %BitAndWith.type.0b5 = facet_value Core.IntLiteral, (%BitAndWith.impl_witness.11b) [concrete]
// CHECK:STDOUT:   %.e0e: type = fn_type_with_self_type %BitAndWith.Op.type.e13, %BitAndWith.facet.c73 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.bound.62b211.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitAndWith.impl.Op.657411.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.76feb8.1: <specific function> = specific_function %T.binding.as_type.as.BitAndWith.impl.Op.657411.2, @T.binding.as_type.as.BitAndWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.284eaa.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.76feb8.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.bound.62b211.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitAndWith.impl.Op.657411.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.76feb8.2: <specific function> = specific_function %T.binding.as_type.as.BitAndWith.impl.Op.657411.1, @T.binding.as_type.as.BitAndWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.284eaa.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.76feb8.2 [concrete]
// CHECK:STDOUT:   %BitOrWith.impl_witness.f0b: <witness> = impl_witness imports.%BitOrWith.impl_witness_table.946, @T.binding.as_type.as.BitOrWith.impl.e42(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.type.6b5126.1: type = fn_type @T.binding.as_type.as.BitOrWith.impl.Op.2, @T.binding.as_type.as.BitOrWith.impl.e42(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.552722.1: %T.binding.as_type.as.BitOrWith.impl.Op.type.6b5126.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.type.6b5126.2: type = fn_type @T.binding.as_type.as.BitOrWith.impl.Op.1, @T.binding.as_type.as.BitOrWith.impl.e42(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.552722.2: %T.binding.as_type.as.BitOrWith.impl.Op.type.6b5126.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.c28: %BitOrWith.type.b22 = facet_value Core.IntLiteral, (%BitOrWith.impl_witness.f0b) [concrete]
// CHECK:STDOUT:   %.9f3: type = fn_type_with_self_type %BitOrWith.Op.type.a1b, %BitOrWith.facet.c28 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.bound.0cc7c8.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitOrWith.impl.Op.552722.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.f5e026.1: <specific function> = specific_function %T.binding.as_type.as.BitOrWith.impl.Op.552722.2, @T.binding.as_type.as.BitOrWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.ba7a8e.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.f5e026.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.bound.0cc7c8.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitOrWith.impl.Op.552722.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.f5e026.2: <specific function> = specific_function %T.binding.as_type.as.BitOrWith.impl.Op.552722.1, @T.binding.as_type.as.BitOrWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.ba7a8e.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.f5e026.2 [concrete]
// CHECK:STDOUT:   %BitXorWith.impl_witness.a75: <witness> = impl_witness imports.%BitXorWith.impl_witness_table.39f, @T.binding.as_type.as.BitXorWith.impl.878(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.type.c3bcf6.1: type = fn_type @T.binding.as_type.as.BitXorWith.impl.Op.2, @T.binding.as_type.as.BitXorWith.impl.878(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.c5dfcc.1: %T.binding.as_type.as.BitXorWith.impl.Op.type.c3bcf6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.type.c3bcf6.2: type = fn_type @T.binding.as_type.as.BitXorWith.impl.Op.1, @T.binding.as_type.as.BitXorWith.impl.878(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.c5dfcc.2: %T.binding.as_type.as.BitXorWith.impl.Op.type.c3bcf6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.b31: %BitXorWith.type.87f = facet_value Core.IntLiteral, (%BitXorWith.impl_witness.a75) [concrete]
// CHECK:STDOUT:   %.498: type = fn_type_with_self_type %BitXorWith.Op.type.5cf, %BitXorWith.facet.b31 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.bound.8d10e0.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitXorWith.impl.Op.c5dfcc.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.9cdf74.1: <specific function> = specific_function %T.binding.as_type.as.BitXorWith.impl.Op.c5dfcc.2, @T.binding.as_type.as.BitXorWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.91b787.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.9cdf74.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.bound.8d10e0.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitXorWith.impl.Op.c5dfcc.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.9cdf74.2: <specific function> = specific_function %T.binding.as_type.as.BitXorWith.impl.Op.c5dfcc.1, @T.binding.as_type.as.BitXorWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.91b787.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.9cdf74.2 [concrete]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness.ad3: <witness> = impl_witness imports.%LeftShiftWith.impl_witness_table.ce8, @T.binding.as_type.as.LeftShiftWith.impl.b71(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.type.0d7d28.1: type = fn_type @T.binding.as_type.as.LeftShiftWith.impl.Op.2, @T.binding.as_type.as.LeftShiftWith.impl.b71(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.f0c7aa.1: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.0d7d28.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.type.0d7d28.2: type = fn_type @T.binding.as_type.as.LeftShiftWith.impl.Op.1, @T.binding.as_type.as.LeftShiftWith.impl.b71(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.f0c7aa.2: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.0d7d28.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.65b: %LeftShiftWith.type.3f9 = facet_value Core.IntLiteral, (%LeftShiftWith.impl_witness.ad3) [concrete]
// CHECK:STDOUT:   %.468: type = fn_type_with_self_type %LeftShiftWith.Op.type.224, %LeftShiftWith.facet.65b [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.bound.df9a08.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.LeftShiftWith.impl.Op.f0c7aa.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.18a0d8.1: <specific function> = specific_function %T.binding.as_type.as.LeftShiftWith.impl.Op.f0c7aa.2, @T.binding.as_type.as.LeftShiftWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.05d3ad.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.18a0d8.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.bound.df9a08.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.LeftShiftWith.impl.Op.f0c7aa.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.18a0d8.2: <specific function> = specific_function %T.binding.as_type.as.LeftShiftWith.impl.Op.f0c7aa.1, @T.binding.as_type.as.LeftShiftWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.05d3ad.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.18a0d8.2 [concrete]
// CHECK:STDOUT:   %RightShiftWith.impl_witness.269: <witness> = impl_witness imports.%RightShiftWith.impl_witness_table.050, @T.binding.as_type.as.RightShiftWith.impl.28b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.type.b3fbc2.1: type = fn_type @T.binding.as_type.as.RightShiftWith.impl.Op.2, @T.binding.as_type.as.RightShiftWith.impl.28b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.79036b.1: %T.binding.as_type.as.RightShiftWith.impl.Op.type.b3fbc2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.type.b3fbc2.2: type = fn_type @T.binding.as_type.as.RightShiftWith.impl.Op.1, @T.binding.as_type.as.RightShiftWith.impl.28b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.79036b.2: %T.binding.as_type.as.RightShiftWith.impl.Op.type.b3fbc2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.62c: %RightShiftWith.type.995 = facet_value Core.IntLiteral, (%RightShiftWith.impl_witness.269) [concrete]
// CHECK:STDOUT:   %.dd3: type = fn_type_with_self_type %RightShiftWith.Op.type.fd9, %RightShiftWith.facet.62c [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.bound.2baaff.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.RightShiftWith.impl.Op.79036b.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.fe68a8.1: <specific function> = specific_function %T.binding.as_type.as.RightShiftWith.impl.Op.79036b.2, @T.binding.as_type.as.RightShiftWith.impl.Op.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.fb5cd1.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.fe68a8.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.bound.2baaff.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.RightShiftWith.impl.Op.79036b.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.fe68a8.2: <specific function> = specific_function %T.binding.as_type.as.RightShiftWith.impl.Op.79036b.1, @T.binding.as_type.as.RightShiftWith.impl.Op.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.fb5cd1.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.fe68a8.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.2 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.eb9: @T.binding.as_type.as.BitAndWith.impl.2e8.%T.binding.as_type.as.BitAndWith.impl.Op.type.2 (%T.binding.as_type.as.BitAndWith.impl.Op.type.b0dfa2.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.BitAndWith.impl.2e8.%T.binding.as_type.as.BitAndWith.impl.Op.2 (constants.%T.binding.as_type.as.BitAndWith.impl.Op.ae0f20.1)]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table.1c0 = impl_witness_table (%Core.import_ref.cb912f.2, %Core.import_ref.eb9), @T.binding.as_type.as.BitAndWith.impl.2e8 [concrete]
// CHECK:STDOUT:   %Core.Op.36a: @T.binding.as_type.as.BitAndWith.impl.2e8.%T.binding.as_type.as.BitAndWith.impl.Op.type.1 (%T.binding.as_type.as.BitAndWith.impl.Op.type.b0dfa2.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @T.binding.as_type.as.BitAndWith.impl.2e8.%T.binding.as_type.as.BitAndWith.impl.Op.1 (constants.%T.binding.as_type.as.BitAndWith.impl.Op.ae0f20.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.4 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d41: @T.binding.as_type.as.BitOrWith.impl.e42.%T.binding.as_type.as.BitOrWith.impl.Op.type.2 (%T.binding.as_type.as.BitOrWith.impl.Op.type.f4a873.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.BitOrWith.impl.e42.%T.binding.as_type.as.BitOrWith.impl.Op.2 (constants.%T.binding.as_type.as.BitOrWith.impl.Op.53d67c.1)]
// CHECK:STDOUT:   %BitOrWith.impl_witness_table.946 = impl_witness_table (%Core.import_ref.cb912f.4, %Core.import_ref.d41), @T.binding.as_type.as.BitOrWith.impl.e42 [concrete]
// CHECK:STDOUT:   %Core.Op.e80: @T.binding.as_type.as.BitOrWith.impl.e42.%T.binding.as_type.as.BitOrWith.impl.Op.type.1 (%T.binding.as_type.as.BitOrWith.impl.Op.type.f4a873.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @T.binding.as_type.as.BitOrWith.impl.e42.%T.binding.as_type.as.BitOrWith.impl.Op.1 (constants.%T.binding.as_type.as.BitOrWith.impl.Op.53d67c.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fef: @T.binding.as_type.as.BitXorWith.impl.878.%T.binding.as_type.as.BitXorWith.impl.Op.type.2 (%T.binding.as_type.as.BitXorWith.impl.Op.type.fdc423.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.BitXorWith.impl.878.%T.binding.as_type.as.BitXorWith.impl.Op.2 (constants.%T.binding.as_type.as.BitXorWith.impl.Op.659e6b.1)]
// CHECK:STDOUT:   %BitXorWith.impl_witness_table.39f = impl_witness_table (%Core.import_ref.cb912f.6, %Core.import_ref.fef), @T.binding.as_type.as.BitXorWith.impl.878 [concrete]
// CHECK:STDOUT:   %Core.Op.a2d: @T.binding.as_type.as.BitXorWith.impl.878.%T.binding.as_type.as.BitXorWith.impl.Op.type.1 (%T.binding.as_type.as.BitXorWith.impl.Op.type.fdc423.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @T.binding.as_type.as.BitXorWith.impl.878.%T.binding.as_type.as.BitXorWith.impl.Op.1 (constants.%T.binding.as_type.as.BitXorWith.impl.Op.659e6b.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.8 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.4cb: @T.binding.as_type.as.LeftShiftWith.impl.b71.%T.binding.as_type.as.LeftShiftWith.impl.Op.type.2 (%T.binding.as_type.as.LeftShiftWith.impl.Op.type.278104.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.LeftShiftWith.impl.b71.%T.binding.as_type.as.LeftShiftWith.impl.Op.2 (constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.fdafb7.1)]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness_table.ce8 = impl_witness_table (%Core.import_ref.cb912f.8, %Core.import_ref.4cb), @T.binding.as_type.as.LeftShiftWith.impl.b71 [concrete]
// CHECK:STDOUT:   %Core.Op.79e: @T.binding.as_type.as.LeftShiftWith.impl.b71.%T.binding.as_type.as.LeftShiftWith.impl.Op.type.1 (%T.binding.as_type.as.LeftShiftWith.impl.Op.type.278104.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @T.binding.as_type.as.LeftShiftWith.impl.b71.%T.binding.as_type.as.LeftShiftWith.impl.Op.1 (constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.fdafb7.2)]
// CHECK:STDOUT:   %Core.import_ref.cb912f.10 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.172: @T.binding.as_type.as.RightShiftWith.impl.28b.%T.binding.as_type.as.RightShiftWith.impl.Op.type.2 (%T.binding.as_type.as.RightShiftWith.impl.Op.type.8ef087.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.RightShiftWith.impl.28b.%T.binding.as_type.as.RightShiftWith.impl.Op.2 (constants.%T.binding.as_type.as.RightShiftWith.impl.Op.3f8b9d.1)]
// CHECK:STDOUT:   %RightShiftWith.impl_witness_table.050 = impl_witness_table (%Core.import_ref.cb912f.10, %Core.import_ref.172), @T.binding.as_type.as.RightShiftWith.impl.28b [concrete]
// CHECK:STDOUT:   %Core.Op.f2f: @T.binding.as_type.as.RightShiftWith.impl.28b.%T.binding.as_type.as.RightShiftWith.impl.Op.type.1 (%T.binding.as_type.as.RightShiftWith.impl.Op.type.8ef087.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @T.binding.as_type.as.RightShiftWith.impl.28b.%T.binding.as_type.as.RightShiftWith.impl.Op.1 (constants.%T.binding.as_type.as.RightShiftWith.impl.Op.3f8b9d.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BitWiseHeterogeneousLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc11: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11(%int_1.loc11) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc11_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc11_21.2: %Cpp.long = converted %int_1.loc11, %.loc11_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc11_21.2
// CHECK:STDOUT:   %AssertSameType.ref.loc13: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13_22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc13: %.419 = impl_witness_access constants.%BitAndWith.impl_witness.a02, element1 [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.3425be.2]
// CHECK:STDOUT:   %bound_method.loc13_20.1: <bound method> = bound_method %b.ref.loc13, %impl.elem1.loc13
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem1.loc13, @T.binding.as_type.as.BitAndWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.f76196.1]
// CHECK:STDOUT:   %bound_method.loc13_20.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_20.1: %T.binding.as_type.as.BitAndWith.impl.Op.type.f24b53.1 = specific_constant imports.%Core.Op.36a, @T.binding.as_type.as.BitAndWith.impl.2e8(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.3425be.1]
// CHECK:STDOUT:   %Op.ref.loc13: %T.binding.as_type.as.BitAndWith.impl.Op.type.f24b53.1 = name_ref Op, %.loc13_20.1 [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.3425be.1]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.loc13: <specific function> = specific_function %Op.ref.loc13, @T.binding.as_type.as.BitAndWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.f76196.2]
// CHECK:STDOUT:   %bound_method.loc13_20.3: <bound method> = bound_method %b.ref.loc13, %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_18: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long = call %bound_method.loc13_18(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_18.2: %Cpp.long = converted %b.ref.loc13, %.loc13_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.call.loc13: init %Cpp.long = call %bound_method.loc13_20.3(%.loc13_18.2, %a.ref.loc13_22)
// CHECK:STDOUT:   %a.ref.loc13_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc13: <specific function> = specific_function %AssertSameType.ref.loc13, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc13_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.BitAndWith.impl.Op.call.loc13
// CHECK:STDOUT:   %.loc13_20.3: %Cpp.long = converted %T.binding.as_type.as.BitAndWith.impl.Op.call.loc13, %.loc13_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc13: init %empty_tuple.type = call %AssertSameType.specific_fn.loc13(%.loc13_20.3, %a.ref.loc13_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc14: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14_22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc14: %.2e2 = impl_witness_access constants.%BitOrWith.impl_witness.d0a, element1 [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.c9149f.2]
// CHECK:STDOUT:   %bound_method.loc14_20.1: <bound method> = bound_method %b.ref.loc14, %impl.elem1.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem1.loc14, @T.binding.as_type.as.BitOrWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.fbab3e.1]
// CHECK:STDOUT:   %bound_method.loc14_20.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_20.1: %T.binding.as_type.as.BitOrWith.impl.Op.type.c5501d.1 = specific_constant imports.%Core.Op.e80, @T.binding.as_type.as.BitOrWith.impl.e42(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.c9149f.1]
// CHECK:STDOUT:   %Op.ref.loc14: %T.binding.as_type.as.BitOrWith.impl.Op.type.c5501d.1 = name_ref Op, %.loc14_20.1 [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.c9149f.1]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.bound.loc14: <bound method> = bound_method %b.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.loc14: <specific function> = specific_function %Op.ref.loc14, @T.binding.as_type.as.BitOrWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.fbab3e.2]
// CHECK:STDOUT:   %bound_method.loc14_20.3: <bound method> = bound_method %b.ref.loc14, %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_18: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long = call %bound_method.loc14_18(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_18.2: %Cpp.long = converted %b.ref.loc14, %.loc14_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.call.loc14: init %Cpp.long = call %bound_method.loc14_20.3(%.loc14_18.2, %a.ref.loc14_22)
// CHECK:STDOUT:   %a.ref.loc14_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc14: <specific function> = specific_function %AssertSameType.ref.loc14, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc14_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.BitOrWith.impl.Op.call.loc14
// CHECK:STDOUT:   %.loc14_20.3: %Cpp.long = converted %T.binding.as_type.as.BitOrWith.impl.Op.call.loc14, %.loc14_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc14: init %empty_tuple.type = call %AssertSameType.specific_fn.loc14(%.loc14_20.3, %a.ref.loc14_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc15: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15_22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.24d = impl_witness_access constants.%BitXorWith.impl_witness.e8e, element1 [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.c338c8.2]
// CHECK:STDOUT:   %bound_method.loc15_20.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.binding.as_type.as.BitXorWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.125c34.1]
// CHECK:STDOUT:   %bound_method.loc15_20.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_20.1: %T.binding.as_type.as.BitXorWith.impl.Op.type.42bd3d.1 = specific_constant imports.%Core.Op.a2d, @T.binding.as_type.as.BitXorWith.impl.878(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.c338c8.1]
// CHECK:STDOUT:   %Op.ref.loc15: %T.binding.as_type.as.BitXorWith.impl.Op.type.42bd3d.1 = name_ref Op, %.loc15_20.1 [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.c338c8.1]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.bound.loc15: <bound method> = bound_method %b.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.loc15: <specific function> = specific_function %Op.ref.loc15, @T.binding.as_type.as.BitXorWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.125c34.2]
// CHECK:STDOUT:   %bound_method.loc15_20.3: <bound method> = bound_method %b.ref.loc15, %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_18: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long = call %bound_method.loc15_18(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_18.2: %Cpp.long = converted %b.ref.loc15, %.loc15_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.call.loc15: init %Cpp.long = call %bound_method.loc15_20.3(%.loc15_18.2, %a.ref.loc15_22)
// CHECK:STDOUT:   %a.ref.loc15_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc15: <specific function> = specific_function %AssertSameType.ref.loc15, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc15_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.BitXorWith.impl.Op.call.loc15
// CHECK:STDOUT:   %.loc15_20.3: %Cpp.long = converted %T.binding.as_type.as.BitXorWith.impl.Op.call.loc15, %.loc15_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc15: init %empty_tuple.type = call %AssertSameType.specific_fn.loc15(%.loc15_20.3, %a.ref.loc15_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc16: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc16: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc16_23: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc16: %.509 = impl_witness_access constants.%LeftShiftWith.impl_witness.fb3, element1 [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.c3e103.2]
// CHECK:STDOUT:   %bound_method.loc16_20.1: <bound method> = bound_method %b.ref.loc16, %impl.elem1.loc16
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem1.loc16, @T.binding.as_type.as.LeftShiftWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.547c25.1]
// CHECK:STDOUT:   %bound_method.loc16_20.2: <bound method> = bound_method %b.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_20.1: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.ab52f6.1 = specific_constant imports.%Core.Op.79e, @T.binding.as_type.as.LeftShiftWith.impl.b71(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.c3e103.1]
// CHECK:STDOUT:   %Op.ref.loc16: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.ab52f6.1 = name_ref Op, %.loc16_20.1 [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.c3e103.1]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.bound.loc16: <bound method> = bound_method %b.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.loc16: <specific function> = specific_function %Op.ref.loc16, @T.binding.as_type.as.LeftShiftWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.547c25.2]
// CHECK:STDOUT:   %bound_method.loc16_20.3: <bound method> = bound_method %b.ref.loc16, %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.loc16
// CHECK:STDOUT:   %impl.elem0.loc16: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_18: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16: init %Cpp.long = call %bound_method.loc16_18(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16
// CHECK:STDOUT:   %.loc16_18.2: %Cpp.long = converted %b.ref.loc16, %.loc16_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.call.loc16: init %Cpp.long = call %bound_method.loc16_20.3(%.loc16_18.2, %a.ref.loc16_23)
// CHECK:STDOUT:   %a.ref.loc16_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc16: <specific function> = specific_function %AssertSameType.ref.loc16, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc16_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.LeftShiftWith.impl.Op.call.loc16
// CHECK:STDOUT:   %.loc16_20.3: %Cpp.long = converted %T.binding.as_type.as.LeftShiftWith.impl.Op.call.loc16, %.loc16_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc16: init %empty_tuple.type = call %AssertSameType.specific_fn.loc16(%.loc16_20.3, %a.ref.loc16_26)
// CHECK:STDOUT:   %AssertSameType.ref.loc17: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc17: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc17_23: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc17: %.9db = impl_witness_access constants.%RightShiftWith.impl_witness.7d0, element1 [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.44dea2.2]
// CHECK:STDOUT:   %bound_method.loc17_20.1: <bound method> = bound_method %b.ref.loc17, %impl.elem1.loc17
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem1.loc17, @T.binding.as_type.as.RightShiftWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.888028.1]
// CHECK:STDOUT:   %bound_method.loc17_20.2: <bound method> = bound_method %b.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_20.1: %T.binding.as_type.as.RightShiftWith.impl.Op.type.2fe33f.1 = specific_constant imports.%Core.Op.f2f, @T.binding.as_type.as.RightShiftWith.impl.28b(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.44dea2.1]
// CHECK:STDOUT:   %Op.ref.loc17: %T.binding.as_type.as.RightShiftWith.impl.Op.type.2fe33f.1 = name_ref Op, %.loc17_20.1 [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.44dea2.1]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.bound.loc17: <bound method> = bound_method %b.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.loc17: <specific function> = specific_function %Op.ref.loc17, @T.binding.as_type.as.RightShiftWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.888028.2]
// CHECK:STDOUT:   %bound_method.loc17_20.3: <bound method> = bound_method %b.ref.loc17, %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_18: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_18(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17
// CHECK:STDOUT:   %.loc17_18.2: %Cpp.long = converted %b.ref.loc17, %.loc17_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.call.loc17: init %Cpp.long = call %bound_method.loc17_20.3(%.loc17_18.2, %a.ref.loc17_23)
// CHECK:STDOUT:   %a.ref.loc17_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc17: <specific function> = specific_function %AssertSameType.ref.loc17, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc17_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.RightShiftWith.impl.Op.call.loc17
// CHECK:STDOUT:   %.loc17_20.3: %Cpp.long = converted %T.binding.as_type.as.RightShiftWith.impl.Op.call.loc17, %.loc17_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc17: init %empty_tuple.type = call %AssertSameType.specific_fn.loc17(%.loc17_20.3, %a.ref.loc17_26)
// CHECK:STDOUT:   %AssertSameType.ref.loc19: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19_22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc19: %.e0e = impl_witness_access constants.%BitAndWith.impl_witness.11b, element1 [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.657411.2]
// CHECK:STDOUT:   %bound_method.loc19_20.1: <bound method> = bound_method %int_1.loc19, %impl.elem1.loc19 [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.bound.62b211.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem1.loc19, @T.binding.as_type.as.BitAndWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.76feb8.1]
// CHECK:STDOUT:   %bound_method.loc19_20.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.284eaa.1]
// CHECK:STDOUT:   %.loc19_20.1: %T.binding.as_type.as.BitAndWith.impl.Op.type.371e74.1 = specific_constant imports.%Core.Op.36a, @T.binding.as_type.as.BitAndWith.impl.2e8(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.657411.1]
// CHECK:STDOUT:   %Op.ref.loc19: %T.binding.as_type.as.BitAndWith.impl.Op.type.371e74.1 = name_ref Op, %.loc19_20.1 [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.657411.1]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.bound.loc19: <bound method> = bound_method %int_1.loc19, %Op.ref.loc19 [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.bound.62b211.2]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.loc19: <specific function> = specific_function %Op.ref.loc19, @T.binding.as_type.as.BitAndWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.76feb8.2]
// CHECK:STDOUT:   %bound_method.loc19_20.3: <bound method> = bound_method %int_1.loc19, %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.loc19 [concrete = constants.%bound_method.284eaa.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_18: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_18(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_18.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_18.2: %Cpp.long = converted %int_1.loc19, %.loc19_18.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.call.loc19: init %Cpp.long = call %bound_method.loc19_20.3(%.loc19_18.2, %a.ref.loc19_22)
// CHECK:STDOUT:   %a.ref.loc19_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc19: <specific function> = specific_function %AssertSameType.ref.loc19, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc19_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.BitAndWith.impl.Op.call.loc19
// CHECK:STDOUT:   %.loc19_20.3: %Cpp.long = converted %T.binding.as_type.as.BitAndWith.impl.Op.call.loc19, %.loc19_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc19: init %empty_tuple.type = call %AssertSameType.specific_fn.loc19(%.loc19_20.3, %a.ref.loc19_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc20: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20_22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc20: %.9f3 = impl_witness_access constants.%BitOrWith.impl_witness.f0b, element1 [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.552722.2]
// CHECK:STDOUT:   %bound_method.loc20_20.1: <bound method> = bound_method %int_1.loc20, %impl.elem1.loc20 [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.bound.0cc7c8.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem1.loc20, @T.binding.as_type.as.BitOrWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.f5e026.1]
// CHECK:STDOUT:   %bound_method.loc20_20.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.ba7a8e.1]
// CHECK:STDOUT:   %.loc20_20.1: %T.binding.as_type.as.BitOrWith.impl.Op.type.6b5126.1 = specific_constant imports.%Core.Op.e80, @T.binding.as_type.as.BitOrWith.impl.e42(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.552722.1]
// CHECK:STDOUT:   %Op.ref.loc20: %T.binding.as_type.as.BitOrWith.impl.Op.type.6b5126.1 = name_ref Op, %.loc20_20.1 [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.552722.1]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.bound.loc20: <bound method> = bound_method %int_1.loc20, %Op.ref.loc20 [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.bound.0cc7c8.2]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.loc20: <specific function> = specific_function %Op.ref.loc20, @T.binding.as_type.as.BitOrWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.f5e026.2]
// CHECK:STDOUT:   %bound_method.loc20_20.3: <bound method> = bound_method %int_1.loc20, %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.loc20 [concrete = constants.%bound_method.ba7a8e.2]
// CHECK:STDOUT:   %impl.elem0.loc20: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_18: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_18(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_18.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_18.2: %Cpp.long = converted %int_1.loc20, %.loc20_18.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.call.loc20: init %Cpp.long = call %bound_method.loc20_20.3(%.loc20_18.2, %a.ref.loc20_22)
// CHECK:STDOUT:   %a.ref.loc20_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc20: <specific function> = specific_function %AssertSameType.ref.loc20, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc20_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.BitOrWith.impl.Op.call.loc20
// CHECK:STDOUT:   %.loc20_20.3: %Cpp.long = converted %T.binding.as_type.as.BitOrWith.impl.Op.call.loc20, %.loc20_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc20: init %empty_tuple.type = call %AssertSameType.specific_fn.loc20(%.loc20_20.3, %a.ref.loc20_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc21: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21_22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc21: %.498 = impl_witness_access constants.%BitXorWith.impl_witness.a75, element1 [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.c5dfcc.2]
// CHECK:STDOUT:   %bound_method.loc21_20.1: <bound method> = bound_method %int_1.loc21, %impl.elem1.loc21 [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.bound.8d10e0.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem1.loc21, @T.binding.as_type.as.BitXorWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.9cdf74.1]
// CHECK:STDOUT:   %bound_method.loc21_20.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.91b787.1]
// CHECK:STDOUT:   %.loc21_20.1: %T.binding.as_type.as.BitXorWith.impl.Op.type.c3bcf6.1 = specific_constant imports.%Core.Op.a2d, @T.binding.as_type.as.BitXorWith.impl.878(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.c5dfcc.1]
// CHECK:STDOUT:   %Op.ref.loc21: %T.binding.as_type.as.BitXorWith.impl.Op.type.c3bcf6.1 = name_ref Op, %.loc21_20.1 [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.c5dfcc.1]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.bound.loc21: <bound method> = bound_method %int_1.loc21, %Op.ref.loc21 [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.bound.8d10e0.2]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.loc21: <specific function> = specific_function %Op.ref.loc21, @T.binding.as_type.as.BitXorWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.9cdf74.2]
// CHECK:STDOUT:   %bound_method.loc21_20.3: <bound method> = bound_method %int_1.loc21, %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.loc21 [concrete = constants.%bound_method.91b787.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_18: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_18(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_18.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_18.2: %Cpp.long = converted %int_1.loc21, %.loc21_18.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.call.loc21: init %Cpp.long = call %bound_method.loc21_20.3(%.loc21_18.2, %a.ref.loc21_22)
// CHECK:STDOUT:   %a.ref.loc21_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc21: <specific function> = specific_function %AssertSameType.ref.loc21, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc21_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.BitXorWith.impl.Op.call.loc21
// CHECK:STDOUT:   %.loc21_20.3: %Cpp.long = converted %T.binding.as_type.as.BitXorWith.impl.Op.call.loc21, %.loc21_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc21: init %empty_tuple.type = call %AssertSameType.specific_fn.loc21(%.loc21_20.3, %a.ref.loc21_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc22: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22_23: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.468 = impl_witness_access constants.%LeftShiftWith.impl_witness.ad3, element1 [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.f0c7aa.2]
// CHECK:STDOUT:   %bound_method.loc22_20.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.bound.df9a08.1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.binding.as_type.as.LeftShiftWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.18a0d8.1]
// CHECK:STDOUT:   %bound_method.loc22_20.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.05d3ad.1]
// CHECK:STDOUT:   %.loc22_20.1: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.0d7d28.1 = specific_constant imports.%Core.Op.79e, @T.binding.as_type.as.LeftShiftWith.impl.b71(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.f0c7aa.1]
// CHECK:STDOUT:   %Op.ref.loc22: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.0d7d28.1 = name_ref Op, %.loc22_20.1 [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.f0c7aa.1]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.bound.loc22: <bound method> = bound_method %int_1.loc22, %Op.ref.loc22 [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.bound.df9a08.2]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.loc22: <specific function> = specific_function %Op.ref.loc22, @T.binding.as_type.as.LeftShiftWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.18a0d8.2]
// CHECK:STDOUT:   %bound_method.loc22_20.3: <bound method> = bound_method %int_1.loc22, %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.loc22 [concrete = constants.%bound_method.05d3ad.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_18: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long = call %bound_method.loc22_18(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_18.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_18.2: %Cpp.long = converted %int_1.loc22, %.loc22_18.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.call.loc22: init %Cpp.long = call %bound_method.loc22_20.3(%.loc22_18.2, %a.ref.loc22_23)
// CHECK:STDOUT:   %a.ref.loc22_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc22: <specific function> = specific_function %AssertSameType.ref.loc22, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc22_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.LeftShiftWith.impl.Op.call.loc22
// CHECK:STDOUT:   %.loc22_20.3: %Cpp.long = converted %T.binding.as_type.as.LeftShiftWith.impl.Op.call.loc22, %.loc22_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc22: init %empty_tuple.type = call %AssertSameType.specific_fn.loc22(%.loc22_20.3, %a.ref.loc22_26)
// CHECK:STDOUT:   %AssertSameType.ref.loc23: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc23_23: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc23: %.dd3 = impl_witness_access constants.%RightShiftWith.impl_witness.269, element1 [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.79036b.2]
// CHECK:STDOUT:   %bound_method.loc23_20.1: <bound method> = bound_method %int_1.loc23, %impl.elem1.loc23 [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.bound.2baaff.1]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem1.loc23, @T.binding.as_type.as.RightShiftWith.impl.Op.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.fe68a8.1]
// CHECK:STDOUT:   %bound_method.loc23_20.2: <bound method> = bound_method %int_1.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.fb5cd1.1]
// CHECK:STDOUT:   %.loc23_20.1: %T.binding.as_type.as.RightShiftWith.impl.Op.type.b3fbc2.1 = specific_constant imports.%Core.Op.f2f, @T.binding.as_type.as.RightShiftWith.impl.28b(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.79036b.1]
// CHECK:STDOUT:   %Op.ref.loc23: %T.binding.as_type.as.RightShiftWith.impl.Op.type.b3fbc2.1 = name_ref Op, %.loc23_20.1 [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.79036b.1]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.bound.loc23: <bound method> = bound_method %int_1.loc23, %Op.ref.loc23 [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.bound.2baaff.2]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.loc23: <specific function> = specific_function %Op.ref.loc23, @T.binding.as_type.as.RightShiftWith.impl.Op.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.fe68a8.2]
// CHECK:STDOUT:   %bound_method.loc23_20.3: <bound method> = bound_method %int_1.loc23, %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.loc23 [concrete = constants.%bound_method.fb5cd1.2]
// CHECK:STDOUT:   %impl.elem0.loc23: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc23_18: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %Cpp.long = call %bound_method.loc23_18(%int_1.loc23) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23_18.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23_18.2: %Cpp.long = converted %int_1.loc23, %.loc23_18.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.call.loc23: init %Cpp.long = call %bound_method.loc23_20.3(%.loc23_18.2, %a.ref.loc23_23)
// CHECK:STDOUT:   %a.ref.loc23_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc23: <specific function> = specific_function %AssertSameType.ref.loc23, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc23_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.RightShiftWith.impl.Op.call.loc23
// CHECK:STDOUT:   %.loc23_20.3: %Cpp.long = converted %T.binding.as_type.as.RightShiftWith.impl.Op.call.loc23, %.loc23_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc23: init %empty_tuple.type = call %AssertSameType.specific_fn.loc23(%.loc23_20.3, %a.ref.loc23_26)
// CHECK:STDOUT:   %AssertSameType.ref.loc25: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc25: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc25_22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc25: %.419 = impl_witness_access constants.%BitAndWith.impl_witness.a02, element1 [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.3425be.2]
// CHECK:STDOUT:   %bound_method.loc25_20.1: <bound method> = bound_method %b.ref.loc25, %impl.elem1.loc25
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem1.loc25, @T.binding.as_type.as.BitAndWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.f76196.1]
// CHECK:STDOUT:   %bound_method.loc25_20.2: <bound method> = bound_method %b.ref.loc25, %specific_fn.loc25
// CHECK:STDOUT:   %.loc25_20.1: %T.binding.as_type.as.BitAndWith.impl.Op.type.f24b53.1 = specific_constant imports.%Core.Op.36a, @T.binding.as_type.as.BitAndWith.impl.2e8(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.3425be.1]
// CHECK:STDOUT:   %Op.ref.loc25: %T.binding.as_type.as.BitAndWith.impl.Op.type.f24b53.1 = name_ref Op, %.loc25_20.1 [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.3425be.1]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.bound.loc25: <bound method> = bound_method %b.ref.loc25, %Op.ref.loc25
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.loc25: <specific function> = specific_function %Op.ref.loc25, @T.binding.as_type.as.BitAndWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.f76196.2]
// CHECK:STDOUT:   %bound_method.loc25_20.3: <bound method> = bound_method %b.ref.loc25, %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.loc25
// CHECK:STDOUT:   %impl.elem0.loc25: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_18: <bound method> = bound_method %b.ref.loc25, %impl.elem0.loc25
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc25: init %Cpp.long = call %bound_method.loc25_18(%b.ref.loc25)
// CHECK:STDOUT:   %.loc25_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc25
// CHECK:STDOUT:   %.loc25_18.2: %Cpp.long = converted %b.ref.loc25, %.loc25_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.call.loc25: init %Cpp.long = call %bound_method.loc25_20.3(%.loc25_18.2, %a.ref.loc25_22)
// CHECK:STDOUT:   %a.ref.loc25_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc25: <specific function> = specific_function %AssertSameType.ref.loc25, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc25_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.BitAndWith.impl.Op.call.loc25
// CHECK:STDOUT:   %.loc25_20.3: %Cpp.long = converted %T.binding.as_type.as.BitAndWith.impl.Op.call.loc25, %.loc25_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc25: init %empty_tuple.type = call %AssertSameType.specific_fn.loc25(%.loc25_20.3, %a.ref.loc25_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc26: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc26: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc26_22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc26: %.2e2 = impl_witness_access constants.%BitOrWith.impl_witness.d0a, element1 [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.c9149f.2]
// CHECK:STDOUT:   %bound_method.loc26_20.1: <bound method> = bound_method %b.ref.loc26, %impl.elem1.loc26
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem1.loc26, @T.binding.as_type.as.BitOrWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.fbab3e.1]
// CHECK:STDOUT:   %bound_method.loc26_20.2: <bound method> = bound_method %b.ref.loc26, %specific_fn.loc26
// CHECK:STDOUT:   %.loc26_20.1: %T.binding.as_type.as.BitOrWith.impl.Op.type.c5501d.1 = specific_constant imports.%Core.Op.e80, @T.binding.as_type.as.BitOrWith.impl.e42(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.c9149f.1]
// CHECK:STDOUT:   %Op.ref.loc26: %T.binding.as_type.as.BitOrWith.impl.Op.type.c5501d.1 = name_ref Op, %.loc26_20.1 [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.c9149f.1]
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.bound.loc26: <bound method> = bound_method %b.ref.loc26, %Op.ref.loc26
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.loc26: <specific function> = specific_function %Op.ref.loc26, @T.binding.as_type.as.BitOrWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.fbab3e.2]
// CHECK:STDOUT:   %bound_method.loc26_20.3: <bound method> = bound_method %b.ref.loc26, %T.binding.as_type.as.BitOrWith.impl.Op.specific_fn.loc26
// CHECK:STDOUT:   %impl.elem0.loc26: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc26_18: <bound method> = bound_method %b.ref.loc26, %impl.elem0.loc26
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc26: init %Cpp.long = call %bound_method.loc26_18(%b.ref.loc26)
// CHECK:STDOUT:   %.loc26_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc26
// CHECK:STDOUT:   %.loc26_18.2: %Cpp.long = converted %b.ref.loc26, %.loc26_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.BitOrWith.impl.Op.call.loc26: init %Cpp.long = call %bound_method.loc26_20.3(%.loc26_18.2, %a.ref.loc26_22)
// CHECK:STDOUT:   %a.ref.loc26_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc26: <specific function> = specific_function %AssertSameType.ref.loc26, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc26_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.BitOrWith.impl.Op.call.loc26
// CHECK:STDOUT:   %.loc26_20.3: %Cpp.long = converted %T.binding.as_type.as.BitOrWith.impl.Op.call.loc26, %.loc26_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc26: init %empty_tuple.type = call %AssertSameType.specific_fn.loc26(%.loc26_20.3, %a.ref.loc26_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc27: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc27: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc27_22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc27: %.24d = impl_witness_access constants.%BitXorWith.impl_witness.e8e, element1 [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.c338c8.2]
// CHECK:STDOUT:   %bound_method.loc27_20.1: <bound method> = bound_method %b.ref.loc27, %impl.elem1.loc27
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem1.loc27, @T.binding.as_type.as.BitXorWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.125c34.1]
// CHECK:STDOUT:   %bound_method.loc27_20.2: <bound method> = bound_method %b.ref.loc27, %specific_fn.loc27
// CHECK:STDOUT:   %.loc27_20.1: %T.binding.as_type.as.BitXorWith.impl.Op.type.42bd3d.1 = specific_constant imports.%Core.Op.a2d, @T.binding.as_type.as.BitXorWith.impl.878(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.c338c8.1]
// CHECK:STDOUT:   %Op.ref.loc27: %T.binding.as_type.as.BitXorWith.impl.Op.type.42bd3d.1 = name_ref Op, %.loc27_20.1 [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.c338c8.1]
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.bound.loc27: <bound method> = bound_method %b.ref.loc27, %Op.ref.loc27
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.loc27: <specific function> = specific_function %Op.ref.loc27, @T.binding.as_type.as.BitXorWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.125c34.2]
// CHECK:STDOUT:   %bound_method.loc27_20.3: <bound method> = bound_method %b.ref.loc27, %T.binding.as_type.as.BitXorWith.impl.Op.specific_fn.loc27
// CHECK:STDOUT:   %impl.elem0.loc27: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc27_18: <bound method> = bound_method %b.ref.loc27, %impl.elem0.loc27
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc27: init %Cpp.long = call %bound_method.loc27_18(%b.ref.loc27)
// CHECK:STDOUT:   %.loc27_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc27
// CHECK:STDOUT:   %.loc27_18.2: %Cpp.long = converted %b.ref.loc27, %.loc27_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.BitXorWith.impl.Op.call.loc27: init %Cpp.long = call %bound_method.loc27_20.3(%.loc27_18.2, %a.ref.loc27_22)
// CHECK:STDOUT:   %a.ref.loc27_25: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc27: <specific function> = specific_function %AssertSameType.ref.loc27, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc27_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.BitXorWith.impl.Op.call.loc27
// CHECK:STDOUT:   %.loc27_20.3: %Cpp.long = converted %T.binding.as_type.as.BitXorWith.impl.Op.call.loc27, %.loc27_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc27: init %empty_tuple.type = call %AssertSameType.specific_fn.loc27(%.loc27_20.3, %a.ref.loc27_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc28: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc28: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc28_23: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc28: %.509 = impl_witness_access constants.%LeftShiftWith.impl_witness.fb3, element1 [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.c3e103.2]
// CHECK:STDOUT:   %bound_method.loc28_20.1: <bound method> = bound_method %b.ref.loc28, %impl.elem1.loc28
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem1.loc28, @T.binding.as_type.as.LeftShiftWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.547c25.1]
// CHECK:STDOUT:   %bound_method.loc28_20.2: <bound method> = bound_method %b.ref.loc28, %specific_fn.loc28
// CHECK:STDOUT:   %.loc28_20.1: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.ab52f6.1 = specific_constant imports.%Core.Op.79e, @T.binding.as_type.as.LeftShiftWith.impl.b71(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.c3e103.1]
// CHECK:STDOUT:   %Op.ref.loc28: %T.binding.as_type.as.LeftShiftWith.impl.Op.type.ab52f6.1 = name_ref Op, %.loc28_20.1 [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.c3e103.1]
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.bound.loc28: <bound method> = bound_method %b.ref.loc28, %Op.ref.loc28
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.loc28: <specific function> = specific_function %Op.ref.loc28, @T.binding.as_type.as.LeftShiftWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.547c25.2]
// CHECK:STDOUT:   %bound_method.loc28_20.3: <bound method> = bound_method %b.ref.loc28, %T.binding.as_type.as.LeftShiftWith.impl.Op.specific_fn.loc28
// CHECK:STDOUT:   %impl.elem0.loc28: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28_18: <bound method> = bound_method %b.ref.loc28, %impl.elem0.loc28
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc28: init %Cpp.long = call %bound_method.loc28_18(%b.ref.loc28)
// CHECK:STDOUT:   %.loc28_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc28
// CHECK:STDOUT:   %.loc28_18.2: %Cpp.long = converted %b.ref.loc28, %.loc28_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.LeftShiftWith.impl.Op.call.loc28: init %Cpp.long = call %bound_method.loc28_20.3(%.loc28_18.2, %a.ref.loc28_23)
// CHECK:STDOUT:   %a.ref.loc28_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc28: <specific function> = specific_function %AssertSameType.ref.loc28, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc28_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.LeftShiftWith.impl.Op.call.loc28
// CHECK:STDOUT:   %.loc28_20.3: %Cpp.long = converted %T.binding.as_type.as.LeftShiftWith.impl.Op.call.loc28, %.loc28_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc28: init %empty_tuple.type = call %AssertSameType.specific_fn.loc28(%.loc28_20.3, %a.ref.loc28_26)
// CHECK:STDOUT:   %AssertSameType.ref.loc29: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc29: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc29_23: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc29: %.9db = impl_witness_access constants.%RightShiftWith.impl_witness.7d0, element1 [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.44dea2.2]
// CHECK:STDOUT:   %bound_method.loc29_20.1: <bound method> = bound_method %b.ref.loc29, %impl.elem1.loc29
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem1.loc29, @T.binding.as_type.as.RightShiftWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.888028.1]
// CHECK:STDOUT:   %bound_method.loc29_20.2: <bound method> = bound_method %b.ref.loc29, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29_20.1: %T.binding.as_type.as.RightShiftWith.impl.Op.type.2fe33f.1 = specific_constant imports.%Core.Op.f2f, @T.binding.as_type.as.RightShiftWith.impl.28b(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.44dea2.1]
// CHECK:STDOUT:   %Op.ref.loc29: %T.binding.as_type.as.RightShiftWith.impl.Op.type.2fe33f.1 = name_ref Op, %.loc29_20.1 [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.44dea2.1]
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.bound.loc29: <bound method> = bound_method %b.ref.loc29, %Op.ref.loc29
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.loc29: <specific function> = specific_function %Op.ref.loc29, @T.binding.as_type.as.RightShiftWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.888028.2]
// CHECK:STDOUT:   %bound_method.loc29_20.3: <bound method> = bound_method %b.ref.loc29, %T.binding.as_type.as.RightShiftWith.impl.Op.specific_fn.loc29
// CHECK:STDOUT:   %impl.elem0.loc29: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29_18: <bound method> = bound_method %b.ref.loc29, %impl.elem0.loc29
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29: init %Cpp.long = call %bound_method.loc29_18(%b.ref.loc29)
// CHECK:STDOUT:   %.loc29_18.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29
// CHECK:STDOUT:   %.loc29_18.2: %Cpp.long = converted %b.ref.loc29, %.loc29_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.RightShiftWith.impl.Op.call.loc29: init %Cpp.long = call %bound_method.loc29_20.3(%.loc29_18.2, %a.ref.loc29_23)
// CHECK:STDOUT:   %a.ref.loc29_26: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc29: <specific function> = specific_function %AssertSameType.ref.loc29, @AssertSameType(constants.%Cpp.long) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc29_20.2: %Cpp.long = value_of_initializer %T.binding.as_type.as.RightShiftWith.impl.Op.call.loc29
// CHECK:STDOUT:   %.loc29_20.3: %Cpp.long = converted %T.binding.as_type.as.RightShiftWith.impl.Op.call.loc29, %.loc29_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc29: init %empty_tuple.type = call %AssertSameType.specific_fn.loc29(%.loc29_20.3, %a.ref.loc29_26)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bitwise_heterogeneous_long_and_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AssertSameType.type: type = fn_type @AssertSameType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AssertSameType: %AssertSameType.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.fc6021.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2ad: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.94e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.556: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d48: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.556) [concrete]
// CHECK:STDOUT:   %.567: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.d48 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.102: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %BitAndWith.type.46b: type = facet_type <@BitAndWith, @BitAndWith(%i64)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.2e7: type = fn_type @BitAndWith.Op, @BitAndWith(%i64) [concrete]
// CHECK:STDOUT:   %BitAndWith.type.0b5: type = facet_type <@BitAndWith, @BitAndWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.e13: type = fn_type @BitAndWith.Op, @BitAndWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.39a54f.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.fc6021.1)> [symbolic]
// CHECK:STDOUT:   %U.354: %ImplicitAs.type.39a54f.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.type.1b6537.1: type = fn_type @Int.as.BitAndWith.impl.Op.2, @Int.as.BitAndWith.impl.4ac(%N, %U.354) [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.b9569a.1: %Int.as.BitAndWith.impl.Op.type.1b6537.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.type.1b6537.2: type = fn_type @Int.as.BitAndWith.impl.Op.3, @Int.as.BitAndWith.impl.4ac(%N, %U.354) [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.b9569a.2: %Int.as.BitAndWith.impl.Op.type.1b6537.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.354: %ImplicitAs.type.39a54f.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.type.e95062.1: type = fn_type @T.binding.as_type.as.BitAndWith.impl.Op.5, @T.binding.as_type.as.BitAndWith.impl.96d(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.892c6d.1: %T.binding.as_type.as.BitAndWith.impl.Op.type.e95062.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.type.e95062.2: type = fn_type @T.binding.as_type.as.BitAndWith.impl.Op.6, @T.binding.as_type.as.BitAndWith.impl.96d(%N, %T.354) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.892c6d.2: %T.binding.as_type.as.BitAndWith.impl.Op.type.e95062.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bbc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cb1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ba4: %ImplicitAs.type.2ad = facet_value %Cpp.long, (%ImplicitAs.impl_witness.bbc) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness.c6a: <witness> = impl_witness imports.%BitAndWith.impl_witness_table.5e7, @T.binding.as_type.as.BitAndWith.impl.96d(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.type.7732f0.1: type = fn_type @T.binding.as_type.as.BitAndWith.impl.Op.6, @T.binding.as_type.as.BitAndWith.impl.96d(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.12fb91.1: %T.binding.as_type.as.BitAndWith.impl.Op.type.7732f0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.type.7732f0.2: type = fn_type @T.binding.as_type.as.BitAndWith.impl.Op.5, @T.binding.as_type.as.BitAndWith.impl.96d(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.12fb91.2: %T.binding.as_type.as.BitAndWith.impl.Op.type.7732f0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.bc4: %BitAndWith.type.46b = facet_value %Cpp.long, (%BitAndWith.impl_witness.c6a) [concrete]
// CHECK:STDOUT:   %.656: type = fn_type_with_self_type %BitAndWith.Op.type.2e7, %BitAndWith.facet.bc4 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.d9a57f.1: <specific function> = specific_function %T.binding.as_type.as.BitAndWith.impl.Op.12fb91.2, @T.binding.as_type.as.BitAndWith.impl.Op.5(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.d9a57f.2: <specific function> = specific_function %T.binding.as_type.as.BitAndWith.impl.Op.12fb91.1, @T.binding.as_type.as.BitAndWith.impl.Op.6(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %.8a2: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.ba4 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert: %Cpp.long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %AssertSameType.specific_fn: <specific function> = specific_function %AssertSameType, @AssertSameType(%i64) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness.c34: <witness> = impl_witness imports.%BitAndWith.impl_witness_table.ebf, @Int.as.BitAndWith.impl.4ac(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.type.871ddc.1: type = fn_type @Int.as.BitAndWith.impl.Op.3, @Int.as.BitAndWith.impl.4ac(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.740875.1: %Int.as.BitAndWith.impl.Op.type.871ddc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.type.871ddc.2: type = fn_type @Int.as.BitAndWith.impl.Op.2, @Int.as.BitAndWith.impl.4ac(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.740875.2: %Int.as.BitAndWith.impl.Op.type.871ddc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.a93: %BitAndWith.type.0b5 = facet_value %i64, (%BitAndWith.impl_witness.c34) [concrete]
// CHECK:STDOUT:   %.29c: type = fn_type_with_self_type %BitAndWith.Op.type.e13, %BitAndWith.facet.a93 [concrete]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.specific_fn.7b1efb.1: <specific function> = specific_function %Int.as.BitAndWith.impl.Op.740875.2, @Int.as.BitAndWith.impl.Op.2(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.specific_fn.7b1efb.2: <specific function> = specific_function %Int.as.BitAndWith.impl.Op.740875.1, @Int.as.BitAndWith.impl.Op.3(%int_64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl.b2d [concrete]
// CHECK:STDOUT:   %Core.import_ref.475cb3.2 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.611: @Int.as.BitAndWith.impl.4ac.%Int.as.BitAndWith.impl.Op.type.2 (%Int.as.BitAndWith.impl.Op.type.1b6537.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitAndWith.impl.4ac.%Int.as.BitAndWith.impl.Op.2 (constants.%Int.as.BitAndWith.impl.Op.b9569a.1)]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table.ebf = impl_witness_table (%Core.import_ref.475cb3.2, %Core.import_ref.611), @Int.as.BitAndWith.impl.4ac [concrete]
// CHECK:STDOUT:   %Core.Op.36f: @Int.as.BitAndWith.impl.4ac.%Int.as.BitAndWith.impl.Op.type.1 (%Int.as.BitAndWith.impl.Op.type.1b6537.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.BitAndWith.impl.4ac.%Int.as.BitAndWith.impl.Op.1 (constants.%Int.as.BitAndWith.impl.Op.b9569a.2)]
// CHECK:STDOUT:   %Core.import_ref.475cb3.3 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.29d: @T.binding.as_type.as.BitAndWith.impl.96d.%T.binding.as_type.as.BitAndWith.impl.Op.type.2 (%T.binding.as_type.as.BitAndWith.impl.Op.type.e95062.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.BitAndWith.impl.96d.%T.binding.as_type.as.BitAndWith.impl.Op.2 (constants.%T.binding.as_type.as.BitAndWith.impl.Op.892c6d.1)]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table.5e7 = impl_witness_table (%Core.import_ref.475cb3.3, %Core.import_ref.29d), @T.binding.as_type.as.BitAndWith.impl.96d [concrete]
// CHECK:STDOUT:   %Core.Op.944: @T.binding.as_type.as.BitAndWith.impl.96d.%T.binding.as_type.as.BitAndWith.impl.Op.type.1 (%T.binding.as_type.as.BitAndWith.impl.Op.type.e95062.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @T.binding.as_type.as.BitAndWith.impl.96d.%T.binding.as_type.as.BitAndWith.impl.Op.1 (constants.%T.binding.as_type.as.BitAndWith.impl.Op.892c6d.2)]
// CHECK:STDOUT:   %Core.import_ref.297: %Cpp.long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cb1 = impl_witness_table (%Core.import_ref.297), @Cpp.long.as.ImplicitAs.impl.e6d [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BitWiseHeterogeneousLongAndI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc10: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long = call %bound_method.loc10(%int_1.loc10) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc10_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc10_21.2: %Cpp.long = converted %int_1.loc10, %.loc10_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc10_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.95b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_10: type = splice_block %i64 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc11: %.567 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc11_16.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.102]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_16.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i64 = call %bound_method.loc11_16.2(%int_1.loc11) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc11_16.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc11_16.2: %i64 = converted %int_1.loc11, %.loc11_16.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %b: %i64 = value_binding b, %.loc11_16.2
// CHECK:STDOUT:   %AssertSameType.ref.loc13: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13_22: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc13: %.656 = impl_witness_access constants.%BitAndWith.impl_witness.c6a, element1 [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.12fb91.2]
// CHECK:STDOUT:   %bound_method.loc13_20.1: <bound method> = bound_method %a.ref.loc13, %impl.elem1.loc13
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem1.loc13, @T.binding.as_type.as.BitAndWith.impl.Op.5(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.d9a57f.1]
// CHECK:STDOUT:   %bound_method.loc13_20.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_20.1: %T.binding.as_type.as.BitAndWith.impl.Op.type.7732f0.1 = specific_constant imports.%Core.Op.944, @T.binding.as_type.as.BitAndWith.impl.96d(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.12fb91.1]
// CHECK:STDOUT:   %Op.ref.loc13: %T.binding.as_type.as.BitAndWith.impl.Op.type.7732f0.1 = name_ref Op, %.loc13_20.1 [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.12fb91.1]
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @T.binding.as_type.as.BitAndWith.impl.Op.6(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%T.binding.as_type.as.BitAndWith.impl.Op.specific_fn.d9a57f.2]
// CHECK:STDOUT:   %bound_method.loc13_20.3: <bound method> = bound_method %a.ref.loc13, %T.binding.as_type.as.BitAndWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13: %.8a2 = impl_witness_access constants.%ImplicitAs.impl_witness.bbc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_18: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_18(%a.ref.loc13)
// CHECK:STDOUT:   %.loc13_18.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_18.2: %i64 = converted %a.ref.loc13, %.loc13_18.1
// CHECK:STDOUT:   %T.binding.as_type.as.BitAndWith.impl.Op.call: init %i64 = call %bound_method.loc13_20.3(%.loc13_18.2, %b.ref.loc13_22)
// CHECK:STDOUT:   %b.ref.loc13_25: %i64 = name_ref b, %b
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc13: <specific function> = specific_function %AssertSameType.ref.loc13, @AssertSameType(constants.%i64) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc13_20.2: %i64 = value_of_initializer %T.binding.as_type.as.BitAndWith.impl.Op.call
// CHECK:STDOUT:   %.loc13_20.3: %i64 = converted %T.binding.as_type.as.BitAndWith.impl.Op.call, %.loc13_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc13: init %empty_tuple.type = call %AssertSameType.specific_fn.loc13(%.loc13_20.3, %b.ref.loc13_25)
// CHECK:STDOUT:   %AssertSameType.ref.loc14: %AssertSameType.type = name_ref AssertSameType, file.%AssertSameType.decl [concrete = constants.%AssertSameType]
// CHECK:STDOUT:   %b.ref.loc14_18: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc14: %.29c = impl_witness_access constants.%BitAndWith.impl_witness.c34, element1 [concrete = constants.%Int.as.BitAndWith.impl.Op.740875.2]
// CHECK:STDOUT:   %bound_method.loc14_20.1: <bound method> = bound_method %b.ref.loc14_18, %impl.elem1.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem1.loc14, @Int.as.BitAndWith.impl.Op.2(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%Int.as.BitAndWith.impl.Op.specific_fn.7b1efb.1]
// CHECK:STDOUT:   %bound_method.loc14_20.2: <bound method> = bound_method %b.ref.loc14_18, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_20.1: %Int.as.BitAndWith.impl.Op.type.871ddc.1 = specific_constant imports.%Core.Op.36f, @Int.as.BitAndWith.impl.4ac(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%Int.as.BitAndWith.impl.Op.740875.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Int.as.BitAndWith.impl.Op.type.871ddc.1 = name_ref Op, %.loc14_20.1 [concrete = constants.%Int.as.BitAndWith.impl.Op.740875.1]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %b.ref.loc14_18, %Op.ref.loc14
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Int.as.BitAndWith.impl.Op.3(constants.%int_64, constants.%ImplicitAs.facet.ba4) [concrete = constants.%Int.as.BitAndWith.impl.Op.specific_fn.7b1efb.2]
// CHECK:STDOUT:   %bound_method.loc14_20.3: <bound method> = bound_method %b.ref.loc14_18, %Int.as.BitAndWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14: %.8a2 = impl_witness_access constants.%ImplicitAs.impl_witness.bbc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_22: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_22(%a.ref.loc14)
// CHECK:STDOUT:   %.loc14_22.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_22.2: %i64 = converted %a.ref.loc14, %.loc14_22.1
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.call: init %i64 = call %bound_method.loc14_20.3(%b.ref.loc14_18, %.loc14_22.2)
// CHECK:STDOUT:   %b.ref.loc14_25: %i64 = name_ref b, %b
// CHECK:STDOUT:   %AssertSameType.specific_fn.loc14: <specific function> = specific_function %AssertSameType.ref.loc14, @AssertSameType(constants.%i64) [concrete = constants.%AssertSameType.specific_fn]
// CHECK:STDOUT:   %.loc14_20.2: %i64 = value_of_initializer %Int.as.BitAndWith.impl.Op.call
// CHECK:STDOUT:   %.loc14_20.3: %i64 = converted %Int.as.BitAndWith.impl.Op.call, %.loc14_20.2
// CHECK:STDOUT:   %AssertSameType.call.loc14: init %empty_tuple.type = call %AssertSameType.specific_fn.loc14(%.loc14_20.3, %b.ref.loc14_25)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_homogeneous_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.type.55d: type = fn_type @T.binding.as_type.as.ImplicitAs.impl.Convert, @T.binding.as_type.as.ImplicitAs.impl(%T.035) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.19b: %T.binding.as_type.as.ImplicitAs.impl.Convert.type.55d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.bef: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.Op.type.6a2: type = fn_type @AddAssignWith.Op, @AddAssignWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.78f138.1: %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.78f138.2: %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e75: <witness> = impl_witness imports.%Copy.impl_witness_table.572 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.e75) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bd4: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f1e, @T.binding.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3fe: %ImplicitAs.type.819 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.bd4) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.fb1: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.07a, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.defb93.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.8b7759.1: %Cpp.long.as.AddAssignWith.impl.Op.type.defb93.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.defb93.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.8b7759.2: %Cpp.long.as.AddAssignWith.impl.Op.type.defb93.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.bef = facet_value %Cpp.long, (%AddAssignWith.impl_witness.fb1) [concrete]
// CHECK:STDOUT:   %.dff: type = fn_type_with_self_type %AddAssignWith.Op.type.6a2, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.df00b6.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.8b7759.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.df00b6.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.8b7759.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.b12: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.Op.type.628: type = fn_type @SubAssignWith.Op, @SubAssignWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.c39240.1: %Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.c39240.2: %Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.631: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.f4b, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.7e399e.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.31eaa0.1: %Cpp.long.as.SubAssignWith.impl.Op.type.7e399e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.7e399e.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.31eaa0.2: %Cpp.long.as.SubAssignWith.impl.Op.type.7e399e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.b12 = facet_value %Cpp.long, (%SubAssignWith.impl_witness.631) [concrete]
// CHECK:STDOUT:   %.327: type = fn_type_with_self_type %SubAssignWith.Op.type.628, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.864a67.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.31eaa0.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.864a67.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.31eaa0.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.59a: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.Op.type.2a9: type = fn_type @MulAssignWith.Op, @MulAssignWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.421374.1: %Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.421374.2: %Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.0b4: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.8cc, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.c3673e.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.b067dc.1: %Cpp.long.as.MulAssignWith.impl.Op.type.c3673e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.c3673e.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.b067dc.2: %Cpp.long.as.MulAssignWith.impl.Op.type.c3673e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.59a = facet_value %Cpp.long, (%MulAssignWith.impl_witness.0b4) [concrete]
// CHECK:STDOUT:   %.4ea: type = fn_type_with_self_type %MulAssignWith.Op.type.2a9, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.a37438.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.b067dc.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.a37438.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.b067dc.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.e6d: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.Op.type.883: type = fn_type @DivAssignWith.Op, @DivAssignWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.6227df.1: %Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.6227df.2: %Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.b4e: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.6f7, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.97cfb4.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.b85335.1: %Cpp.long.as.DivAssignWith.impl.Op.type.97cfb4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.97cfb4.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.b85335.2: %Cpp.long.as.DivAssignWith.impl.Op.type.97cfb4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.e6d = facet_value %Cpp.long, (%DivAssignWith.impl_witness.b4e) [concrete]
// CHECK:STDOUT:   %.a57: type = fn_type_with_self_type %DivAssignWith.Op.type.883, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.19b8da.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.b85335.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.19b8da.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.b85335.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.2a3: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.Op.type.a01: type = fn_type @ModAssignWith.Op, @ModAssignWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.666e79.1: %Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.666e79.2: %Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.faa: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.a2a, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.24a130.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.da4005.1: %Cpp.long.as.ModAssignWith.impl.Op.type.24a130.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.24a130.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.da4005.2: %Cpp.long.as.ModAssignWith.impl.Op.type.24a130.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.2a3 = facet_value %Cpp.long, (%ModAssignWith.impl_witness.faa) [concrete]
// CHECK:STDOUT:   %.db4: type = fn_type_with_self_type %ModAssignWith.Op.type.a01, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.51594a.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.da4005.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.51594a.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.da4005.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.efd: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.Op.type.3d8: type = fn_type @BitAndAssignWith.Op, @BitAndAssignWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.d99897.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.d99897.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.a49: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.cc9, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.bdaab3.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.500716.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.bdaab3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.bdaab3.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.500716.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.bdaab3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.efd = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.a49) [concrete]
// CHECK:STDOUT:   %.1cc: type = fn_type_with_self_type %BitAndAssignWith.Op.type.3d8, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.1245aa.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.500716.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.1245aa.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.500716.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.b27: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.Op.type.99d: type = fn_type @BitOrAssignWith.Op, @BitOrAssignWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.ab4: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.54c, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.a2b279.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a2b279.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.a2b279.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a2b279.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.b27 = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.ab4) [concrete]
// CHECK:STDOUT:   %.d95: type = fn_type_with_self_type %BitOrAssignWith.Op.type.99d, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.b7967f.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.b7967f.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.431: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.Op.type.d3b: type = fn_type @BitXorAssignWith.Op, @BitXorAssignWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.4d7: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.a39, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.6f67bc.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6f67bc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.6f67bc.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6f67bc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.431 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.4d7) [concrete]
// CHECK:STDOUT:   %.797: type = fn_type_with_self_type %BitXorAssignWith.Op.type.d3b, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.04506a.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.04506a.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.bed: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.Op.type.1b3: type = fn_type @LeftShiftAssignWith.Op, @LeftShiftAssignWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.550: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.74d, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.e1513f.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.e1513f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.e1513f.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.e1513f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.bed = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.550) [concrete]
// CHECK:STDOUT:   %.98f: type = fn_type_with_self_type %LeftShiftAssignWith.Op.type.1b3, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0946f9.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0946f9.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.222: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.Op.type.400: type = fn_type @RightShiftAssignWith.Op, @RightShiftAssignWith(%Cpp.long) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.328772.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.328772.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.df5: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.b55, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.095737.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.095737.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.095737.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.095737.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.222 = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.df5) [concrete]
// CHECK:STDOUT:   %.9187: type = fn_type_with_self_type %RightShiftAssignWith.Op.type.400, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1ec453.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1ec453.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.type: type = fn_type @Cpp.long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op: %Cpp.long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.1c6: @T.binding.as_type.as.ImplicitAs.impl.%T.binding.as_type.as.ImplicitAs.impl.Convert.type (%T.binding.as_type.as.ImplicitAs.impl.Convert.type.55d) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.binding.as_type.as.ImplicitAs.impl.%T.binding.as_type.as.ImplicitAs.impl.Convert (constants.%T.binding.as_type.as.ImplicitAs.impl.Convert.19b)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f1e = impl_witness_table (%Core.import_ref.1c6), @T.binding.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.d0e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.07a = impl_witness_table (%Core.import_ref.d0e), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f81: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.2)]
// CHECK:STDOUT:   %Core.import_ref.915: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.572 = impl_witness_table (%Core.import_ref.915), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.bcb: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c39240.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.f4b = impl_witness_table (%Core.import_ref.bcb), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.739: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c39240.2)]
// CHECK:STDOUT:   %Core.import_ref.c5b: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.421374.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.8cc = impl_witness_table (%Core.import_ref.c5b), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3b8: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.421374.2)]
// CHECK:STDOUT:   %Core.import_ref.84d: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.6227df.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.6f7 = impl_witness_table (%Core.import_ref.84d), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f47: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.6227df.2)]
// CHECK:STDOUT:   %Core.import_ref.e98: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.666e79.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.a2a = impl_witness_table (%Core.import_ref.e98), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b37: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.666e79.2)]
// CHECK:STDOUT:   %Core.import_ref.325: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.d99897.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.cc9 = impl_witness_table (%Core.import_ref.325), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fe8: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.d99897.2)]
// CHECK:STDOUT:   %Core.import_ref.01a: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.54c = impl_witness_table (%Core.import_ref.01a), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.5c2: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.2)]
// CHECK:STDOUT:   %Core.import_ref.c8f: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.a39 = impl_witness_table (%Core.import_ref.c8f), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.caf: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.2)]
// CHECK:STDOUT:   %Core.import_ref.bcc: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.74d = impl_witness_table (%Core.import_ref.bcc), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.030: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.2)]
// CHECK:STDOUT:   %Core.import_ref.5f5: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.328772.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.b55 = impl_witness_table (%Core.import_ref.5f5), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3b5: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.328772.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc11: %.dff = impl_witness_access constants.%AddAssignWith.impl_witness.fb1, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.8b7759.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.df00b6.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long.as.AddAssignWith.impl.Op.type.defb93.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.8b7759.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.AddAssignWith.impl.Op.type.defb93.1 = name_ref Op, %.loc11_5.3 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.8b7759.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.df00b6.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.327 = impl_witness_access constants.%SubAssignWith.impl_witness.631, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.31eaa0.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.864a67.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long.as.SubAssignWith.impl.Op.type.7e399e.1 = specific_constant imports.%Core.Op.739, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.31eaa0.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.SubAssignWith.impl.Op.type.7e399e.1 = name_ref Op, %.loc12_5.3 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.31eaa0.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.864a67.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.4ea = impl_witness_access constants.%MulAssignWith.impl_witness.0b4, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.b067dc.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.a37438.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long.as.MulAssignWith.impl.Op.type.c3673e.1 = specific_constant imports.%Core.Op.3b8, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.b067dc.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.MulAssignWith.impl.Op.type.c3673e.1 = name_ref Op, %.loc13_5.3 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.b067dc.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.a37438.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.a57 = impl_witness_access constants.%DivAssignWith.impl_witness.b4e, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.b85335.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.19b8da.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long.as.DivAssignWith.impl.Op.type.97cfb4.1 = specific_constant imports.%Core.Op.f47, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.b85335.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.DivAssignWith.impl.Op.type.97cfb4.1 = name_ref Op, %.loc14_5.3 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.b85335.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.19b8da.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.db4 = impl_witness_access constants.%ModAssignWith.impl_witness.faa, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.da4005.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.1: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc15_5.1 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %ImplicitAs.facet.loc15_5.2: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc15_5.2: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc15_5.2 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.51594a.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long.as.ModAssignWith.impl.Op.type.24a130.1 = specific_constant imports.%Core.Op.b37, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.da4005.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long.as.ModAssignWith.impl.Op.type.24a130.1 = name_ref Op, %.loc15_5.3 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.da4005.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.51594a.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.1cc = impl_witness_access constants.%BitAndAssignWith.impl_witness.a49, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.500716.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.1245aa.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long.as.BitAndAssignWith.impl.Op.type.bdaab3.1 = specific_constant imports.%Core.Op.fe8, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.500716.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitAndAssignWith.impl.Op.type.bdaab3.1 = name_ref Op, %.loc17_5.3 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.500716.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.1245aa.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.d95 = impl_witness_access constants.%BitOrAssignWith.impl_witness.ab4, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.b7967f.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a2b279.1 = specific_constant imports.%Core.Op.5c2, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a2b279.1 = name_ref Op, %.loc18_5.3 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.b7967f.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.797 = impl_witness_access constants.%BitXorAssignWith.impl_witness.4d7, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.04506a.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6f67bc.1 = specific_constant imports.%Core.Op.caf, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6f67bc.1 = name_ref Op, %.loc19_5.3 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.04506a.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.98f = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.550, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0946f9.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.e1513f.1 = specific_constant imports.%Core.Op.030, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.e1513f.1 = name_ref Op, %.loc20_5.3 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0946f9.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.9187 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.df5, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.1: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc21_5.1 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %ImplicitAs.facet.loc21_5.2: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc21_5.2: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc21_5.2 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1ec453.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.095737.1 = specific_constant imports.%Core.Op.3b5, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.095737.1 = name_ref Op, %.loc21_5.3 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1ec453.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_9)
// CHECK:STDOUT:   %DestroyOp.bound.loc9: <bound method> = bound_method %b.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc9: init %empty_tuple.type = call %DestroyOp.bound.loc9(%b.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc8: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc8: init %empty_tuple.type = call %DestroyOp.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_hereogeneous_long_and_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.794: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.Op.type.6cd: type = fn_type @AddAssignWith.Op, @AddAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.78f138.1: %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.78f138.2: %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.ca1: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.07a, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.d19046.1: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.d19046.2: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.794 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.ca1) [concrete]
// CHECK:STDOUT:   %.5e7: type = fn_type_with_self_type %AddAssignWith.Op.type.6cd, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d19046.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.c45: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d19046.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.ab0: type = facet_type <@SubAssignWith, @SubAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.Op.type.c0c: type = fn_type @SubAssignWith.Op, @SubAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.c39240.1: %Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.c39240.2: %Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.200: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.f4b, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.f5cce2.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.22e054.1: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cce2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.f5cce2.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.22e054.2: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cce2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.ab0 = facet_value %Cpp.long, (%SubAssignWith.impl_witness.200) [concrete]
// CHECK:STDOUT:   %.f56: type = fn_type_with_self_type %SubAssignWith.Op.type.c0c, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.dd31df.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.22e054.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.dd31df.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.22e054.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.743: type = facet_type <@MulAssignWith, @MulAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.Op.type.7f1: type = fn_type @MulAssignWith.Op, @MulAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.421374.1: %Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.421374.2: %Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.c16: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.8cc, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.d2f0ca.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.1: %Cpp.long.as.MulAssignWith.impl.Op.type.d2f0ca.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.d2f0ca.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.2: %Cpp.long.as.MulAssignWith.impl.Op.type.d2f0ca.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.743 = facet_value %Cpp.long, (%MulAssignWith.impl_witness.c16) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %MulAssignWith.Op.type.7f1, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c127b2.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c127b2.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.786: type = facet_type <@DivAssignWith, @DivAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.Op.type.1bd: type = fn_type @DivAssignWith.Op, @DivAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.6227df.1: %Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.6227df.2: %Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.b4b: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.6f7, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.16271d.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.2f8f46.1: %Cpp.long.as.DivAssignWith.impl.Op.type.16271d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.16271d.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.2f8f46.2: %Cpp.long.as.DivAssignWith.impl.Op.type.16271d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.786 = facet_value %Cpp.long, (%DivAssignWith.impl_witness.b4b) [concrete]
// CHECK:STDOUT:   %.9a9: type = fn_type_with_self_type %DivAssignWith.Op.type.1bd, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ee65e7.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.2f8f46.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ee65e7.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.2f8f46.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.e49: type = facet_type <@ModAssignWith, @ModAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.Op.type.0ee: type = fn_type @ModAssignWith.Op, @ModAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.666e79.1: %Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.666e79.2: %Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.a13: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.a2a, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.e2a2e6.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.f69ad5.1: %Cpp.long.as.ModAssignWith.impl.Op.type.e2a2e6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.e2a2e6.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.f69ad5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.e2a2e6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.e49 = facet_value %Cpp.long, (%ModAssignWith.impl_witness.a13) [concrete]
// CHECK:STDOUT:   %.804: type = fn_type_with_self_type %ModAssignWith.Op.type.0ee, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.73e9bb.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.f69ad5.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.73e9bb.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.f69ad5.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.d65: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.Op.type.e4f: type = fn_type @BitAndAssignWith.Op, @BitAndAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.d99897.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.d99897.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.349: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.cc9, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.aa6f07.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.aa6f07.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.aa6f07.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.aa6f07.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.d65 = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.349) [concrete]
// CHECK:STDOUT:   %.24e: type = fn_type_with_self_type %BitAndAssignWith.Op.type.e4f, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.560221.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.560221.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.6e9: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.Op.type.9af: type = fn_type @BitOrAssignWith.Op, @BitOrAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.119: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.54c, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.fac636.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.73e223.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fac636.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.fac636.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.73e223.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fac636.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.6e9 = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.119) [concrete]
// CHECK:STDOUT:   %.b0b: type = fn_type_with_self_type %BitOrAssignWith.Op.type.9af, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.665975.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.73e223.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.665975.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.73e223.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.ea8: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.Op.type.a60: type = fn_type @BitXorAssignWith.Op, @BitXorAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.5aa: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.a39, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.53faad.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.616e38.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.53faad.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.53faad.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.616e38.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.53faad.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.ea8 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.5aa) [concrete]
// CHECK:STDOUT:   %.528: type = fn_type_with_self_type %BitXorAssignWith.Op.type.a60, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.4cd193.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.616e38.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.4cd193.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.616e38.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.92b: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.Op.type.ba8: type = fn_type @LeftShiftAssignWith.Op, @LeftShiftAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.199: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.74d, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.426a0e.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.426a0e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.426a0e.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.426a0e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.92b = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.199) [concrete]
// CHECK:STDOUT:   %.0a6: type = fn_type_with_self_type %LeftShiftAssignWith.Op.type.ba8, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.3bd66d.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.3bd66d.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.8ae: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.Op.type.011: type = fn_type @RightShiftAssignWith.Op, @RightShiftAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.328772.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.328772.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.e4e: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.b55, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2dfa9.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2dfa9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2dfa9.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2dfa9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.8ae = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.e4e) [concrete]
// CHECK:STDOUT:   %.6ef: type = fn_type_with_self_type %RightShiftAssignWith.Op.type.011, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c3db.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c3db.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.d0e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.07a = impl_witness_table (%Core.import_ref.d0e), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f81: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.bcb: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c39240.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.f4b = impl_witness_table (%Core.import_ref.bcb), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.739: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c39240.2)]
// CHECK:STDOUT:   %Core.import_ref.c5b: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.421374.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.8cc = impl_witness_table (%Core.import_ref.c5b), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3b8: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.421374.2)]
// CHECK:STDOUT:   %Core.import_ref.84d: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.6227df.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.6f7 = impl_witness_table (%Core.import_ref.84d), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f47: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.6227df.2)]
// CHECK:STDOUT:   %Core.import_ref.e98: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.666e79.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.a2a = impl_witness_table (%Core.import_ref.e98), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b37: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.666e79.2)]
// CHECK:STDOUT:   %Core.import_ref.325: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.d99897.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.cc9 = impl_witness_table (%Core.import_ref.325), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fe8: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.d99897.2)]
// CHECK:STDOUT:   %Core.import_ref.01a: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.54c = impl_witness_table (%Core.import_ref.01a), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.5c2: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.2)]
// CHECK:STDOUT:   %Core.import_ref.c8f: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.a39 = impl_witness_table (%Core.import_ref.c8f), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.caf: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.2)]
// CHECK:STDOUT:   %Core.import_ref.bcc: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.74d = impl_witness_table (%Core.import_ref.bcc), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.030: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.2)]
// CHECK:STDOUT:   %Core.import_ref.5f5: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.328772.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.b55 = impl_witness_table (%Core.import_ref.5f5), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3b5: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.328772.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.5e7 = impl_witness_access constants.%AddAssignWith.impl_witness.ca1, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = name_ref Op, %.loc10_5.3 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.5: %Cpp.long = converted %b.ref.loc10, %.loc10_5.4
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5.1: %.f56 = impl_witness_access constants.%SubAssignWith.impl_witness.200, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.22e054.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc11_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc11_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc11_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc11_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5.1, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.dd31df.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cce2.1 = specific_constant imports.%Core.Op.739, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.22e054.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cce2.1 = name_ref Op, %.loc11_5.3 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.22e054.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.5: %Cpp.long = converted %b.ref.loc11, %.loc11_5.4
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.dd31df.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5.1: %.d15 = impl_witness_access constants.%MulAssignWith.impl_witness.c16, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc12_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc12_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc12_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc12_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5.1, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c127b2.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long.as.MulAssignWith.impl.Op.type.d2f0ca.1 = specific_constant imports.%Core.Op.3b8, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.MulAssignWith.impl.Op.type.d2f0ca.1 = name_ref Op, %.loc12_5.3 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.5: %Cpp.long = converted %b.ref.loc12, %.loc12_5.4
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c127b2.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_8: init %Cpp.long = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_8
// CHECK:STDOUT:   %.loc12_8.2: %Cpp.long = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.9a9 = impl_witness_access constants.%DivAssignWith.impl_witness.b4b, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2f8f46.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc13_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc13_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ee65e7.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long.as.DivAssignWith.impl.Op.type.16271d.1 = specific_constant imports.%Core.Op.f47, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2f8f46.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.DivAssignWith.impl.Op.type.16271d.1 = name_ref Op, %.loc13_5.3 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2f8f46.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.5: %Cpp.long = converted %b.ref.loc13, %.loc13_5.4
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ee65e7.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_8: init %Cpp.long = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_8
// CHECK:STDOUT:   %.loc13_8.2: %Cpp.long = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5.1: %.804 = impl_witness_access constants.%ModAssignWith.impl_witness.a13, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.f69ad5.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc14_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc14_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5.1, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.73e9bb.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long.as.ModAssignWith.impl.Op.type.e2a2e6.1 = specific_constant imports.%Core.Op.b37, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.f69ad5.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.ModAssignWith.impl.Op.type.e2a2e6.1 = name_ref Op, %.loc14_5.3 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.f69ad5.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.5: %Cpp.long = converted %b.ref.loc14, %.loc14_5.4
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.73e9bb.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5.1: %.24e = impl_witness_access constants.%BitAndAssignWith.impl_witness.349, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc16_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc16_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc16_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc16_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5.1, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.560221.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.3: %Cpp.long.as.BitAndAssignWith.impl.Op.type.aa6f07.1 = specific_constant imports.%Core.Op.fe8, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long.as.BitAndAssignWith.impl.Op.type.aa6f07.1 = name_ref Op, %.loc16_5.3 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_5: init %Cpp.long = call %bound_method.loc16_5.3(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_5
// CHECK:STDOUT:   %.loc16_5.5: %Cpp.long = converted %b.ref.loc16, %.loc16_5.4
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.560221.2]
// CHECK:STDOUT:   %bound_method.loc16_5.4: <bound method> = bound_method %a.ref.loc16, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_8: init %Cpp.long = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_8
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.4(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.b0b = impl_witness_access constants.%BitOrAssignWith.impl_witness.119, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.73e223.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc17_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc17_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc17_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc17_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.665975.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fac636.1 = specific_constant imports.%Core.Op.5c2, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.73e223.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fac636.1 = name_ref Op, %.loc17_5.3 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.73e223.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long = call %bound_method.loc17_5.3(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_5
// CHECK:STDOUT:   %.loc17_5.5: %Cpp.long = converted %b.ref.loc17, %.loc17_5.4
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.665975.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_8
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5.1: %.528 = impl_witness_access constants.%BitXorAssignWith.impl_witness.5aa, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.616e38.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc18_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc18_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc18_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc18_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5.1, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.4cd193.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long.as.BitXorAssignWith.impl.Op.type.53faad.1 = specific_constant imports.%Core.Op.caf, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.616e38.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitXorAssignWith.impl.Op.type.53faad.1 = name_ref Op, %.loc18_5.3 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.616e38.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long = call %bound_method.loc18_5.3(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc18_5
// CHECK:STDOUT:   %.loc18_5.5: %Cpp.long = converted %b.ref.loc18, %.loc18_5.4
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.4cd193.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc18_8
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5.1: %.0a6 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.199, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc19_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc19_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc19_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc19_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.3bd66d.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.426a0e.1 = specific_constant imports.%Core.Op.030, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.426a0e.1 = name_ref Op, %.loc19_5.3 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long = call %bound_method.loc19_5.3(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc19_5
// CHECK:STDOUT:   %.loc19_5.5: %Cpp.long = converted %b.ref.loc19, %.loc19_5.4
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.3bd66d.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc19_9: init %Cpp.long = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc19_9
// CHECK:STDOUT:   %.loc19_9.2: %Cpp.long = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.6ef = impl_witness_access constants.%RightShiftAssignWith.impl_witness.e4e, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc20_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc20_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc20_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc20_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c3db.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2dfa9.1 = specific_constant imports.%Core.Op.3b5, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2dfa9.1 = name_ref Op, %.loc20_5.3 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long = call %bound_method.loc20_5.3(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc20_5
// CHECK:STDOUT:   %.loc20_5.5: %Cpp.long = converted %b.ref.loc20, %.loc20_5.4
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c3db.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_9: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_9
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc20_9: init %Cpp.long = call %bound_method.loc20_9(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_9.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc20_9
// CHECK:STDOUT:   %.loc20_9.2: %Cpp.long = converted %b.ref.loc20, %.loc20_9.1
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_9.2)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_and_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.761: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.Op.type.4fd: type = fn_type @AddAssignWith.Op, @AddAssignWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.78f138.1: %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.78f138.2: %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.623: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.07a, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.a474b7.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.5fb128.1: %Cpp.long.as.AddAssignWith.impl.Op.type.a474b7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.a474b7.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.5fb128.2: %Cpp.long.as.AddAssignWith.impl.Op.type.a474b7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.761 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.623) [concrete]
// CHECK:STDOUT:   %.4b1: type = fn_type_with_self_type %AddAssignWith.Op.type.4fd, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3a3075.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.5fb128.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3a3075.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.5fb128.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.d0e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.07a = impl_witness_table (%Core.import_ref.d0e), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f81: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9_5.1: %.4b1 = impl_witness_access constants.%AddAssignWith.impl_witness.623, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.5fb128.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.1: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_5.2: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.2: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_5.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3a3075.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long.as.AddAssignWith.impl.Op.type.a474b7.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.5fb128.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.a474b7.1 = name_ref Op, %.loc9_5.3 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.5fb128.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc9_5.2: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5: init %Cpp.long = call %bound_method.loc9_5.3(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_5.4: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_5.5: %Cpp.long = converted %int_1.loc9, %.loc9_5.4 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3a3075.2]
// CHECK:STDOUT:   %bound_method.loc9_5.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_8: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_8: init %Cpp.long = call %bound_method.loc9_8(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_8.2: %Cpp.long = converted %int_1.loc9, %.loc9_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_5.4(%a.ref, %.loc9_8.2)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_and_runtime_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %.863: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.794: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.Op.type.6cd: type = fn_type @AddAssignWith.Op, @AddAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.78f138.1: %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.78f138.2: %Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.ca1: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.07a, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.d19046.1: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.d19046.2: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.794 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.ca1) [concrete]
// CHECK:STDOUT:   %.5e7: type = fn_type_with_self_type %AddAssignWith.Op.type.6cd, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d19046.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.c45: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d19046.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl.b2d [concrete]
// CHECK:STDOUT:   %Core.import_ref.d0e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.07a = impl_witness_table (%Core.import_ref.d0e), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f81: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
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
// CHECK:STDOUT:   %.loc9_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_16.2(%int_1.loc9) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_16.2: %i32 = converted %int_1.loc9, %.loc9_16.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %b: %i32 = value_binding b, %.loc9_16.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.5e7 = impl_witness_access constants.%AddAssignWith.impl_witness.ca1, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.1: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc10_5.1 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %ImplicitAs.facet.loc10_5.2: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc10_5.2: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc10_5.2 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = name_ref Op, %.loc10_5.3 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref)
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.5: %Cpp.long = converted %b.ref, %.loc10_5.4
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.c45 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long = call %bound_method.loc10_8(%b.ref)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref, %.loc10_8.2)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- increment_decrement_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %Cpp.long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.819: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.4c2, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.Op.type: type = fn_type @Inc.Op [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %.153: type = fn_type_with_self_type %Inc.Op.type, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.type: type = fn_type @Cpp.long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op: %Cpp.long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.Op.type: type = fn_type @Dec.Op [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %.fff: type = fn_type_with_self_type %Dec.Op.type, %Dec.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.type: type = fn_type @Cpp.long.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op: %Cpp.long.as.Dec.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9d7: %Cpp.long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.9d7), @Cpp.long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d5d: %Cpp.long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.d5d), @Cpp.long.as.Dec.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.153 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.fff = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10(%a.ref.loc10)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Cpp.long) = "no_op";
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.691: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.unsigned_long) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.146: type = facet_type <@ImplicitAs, @ImplicitAs(%u32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.92a: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%u32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a19: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5b5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.66c: %ImplicitAs.type.3c3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a19) [concrete]
// CHECK:STDOUT:   %.6bf: type = fn_type_with_self_type %ImplicitAs.Convert.type.691, %ImplicitAs.facet.66c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.71f: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.14b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.71f = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.14b [concrete]
// CHECK:STDOUT:   %int_1.0ab: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongResult: type = class_type @ULongResult [concrete]
// CHECK:STDOUT:   %pattern_type.6f2: type = pattern_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULong.cpp_overload_set.type: type = cpp_overload_set_type @PassULong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassULong.cpp_overload_set.value: %PassULong.cpp_overload_set.type = cpp_overload_set_value @PassULong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.f5e: type = ptr_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.type.3fc2d8.1: type = fn_type @PassULong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassULong__carbon_thunk.ff747d.1: %PassULong__carbon_thunk.type.3fc2d8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e61: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.af7 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3c4: %ImplicitAs.type.146 = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.e61) [concrete]
// CHECK:STDOUT:   %.da1: type = fn_type_with_self_type %ImplicitAs.Convert.type.92a, %ImplicitAs.facet.3c4 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.bef: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.19d: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.bef = struct_value () [concrete]
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
// CHECK:STDOUT:   %As.Convert.type.3ff: type = fn_type @As.Convert, @As(%Cpp.unsigned_long) [concrete]
// CHECK:STDOUT:   %As.impl_witness.d5f: <witness> = impl_witness imports.%As.impl_witness_table.f58 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.950 = facet_value Core.IntLiteral, (%As.impl_witness.d5f) [concrete]
// CHECK:STDOUT:   %.80d: type = fn_type_with_self_type %As.Convert.type.3ff, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a13: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.50d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2f7: %ImplicitAs.type.139 = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.a13) [concrete]
// CHECK:STDOUT:   %.4f2: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.2f7 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.a9d: type = fn_type @Cpp.unsigned_long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.e53: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.a9d = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.0ab, %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.e53 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.674) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32.97e) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %.a6f: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc24 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.702: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.71f = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.14b]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5b5 = impl_witness_table (%Core.import_ref.702), @Core.IntLiteral.as.ImplicitAs.impl.541 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULong.cpp_overload_set.value: %PassULong.cpp_overload_set.type = cpp_overload_set_value @PassULong.cpp_overload_set [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULong__carbon_thunk.decl.108234.1: %PassULong__carbon_thunk.type.3fc2d8.1 = fn_decl @PassULong__carbon_thunk.1 [concrete = constants.%PassULong__carbon_thunk.ff747d.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.d88: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.bef = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.19d]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.af7 = impl_witness_table (%Core.import_ref.d88), @Cpp.unsigned_long.as.ImplicitAs.impl.66a [concrete]
// CHECK:STDOUT:   %UIntResult.decl: type = class_decl @UIntResult [concrete = constants.%UIntResult] {} {}
// CHECK:STDOUT:   %PassULong__carbon_thunk.decl.108234.2: %PassULong__carbon_thunk.type.3fc2d8.2 = fn_decl @PassULong__carbon_thunk.2 [concrete = constants.%PassULong__carbon_thunk.ff747d.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.190: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.f58 = impl_witness_table (%Core.import_ref.190), @Core.IntLiteral.as.As.impl.8c9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.f5e: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type.a9d = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.e53]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.50d = impl_witness_table (%Core.import_ref.f5e), @Cpp.unsigned_long.as.ImplicitAs.impl.78e [concrete]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31a = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc15: %.6bf = impl_witness_access constants.%ImplicitAs.impl_witness.a19, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.14b]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %.loc15_46.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %.loc15_46.2: %Cpp.unsigned_long = converted %int_1.loc15, %.loc15_46.1 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %cpp_unsigned_long: %Cpp.unsigned_long = value_binding cpp_unsigned_long, %.loc15_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_result.patt: %pattern_type.6f2 = value_binding_pattern cpp_unsigned_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_51: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc16: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc16: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc16_82.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.f5e = addr_of %.loc16_82.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.1(%cpp_unsigned_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_82.2: init %ULongResult to %.loc16_82.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_36: type = splice_block %ULongResult.ref.loc16 [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref.loc16: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_82.3: ref %ULongResult = temporary %.loc16_82.1, %.loc16_82.2
// CHECK:STDOUT:   %.loc16_82.4: %ULongResult = acquire_value %.loc16_82.3
// CHECK:STDOUT:   %cpp_unsigned_long_result: %ULongResult = value_binding cpp_unsigned_long_result, %.loc16_82.4
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
// CHECK:STDOUT:   %Cpp.ref.loc19_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc19: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong.ref: %Cpp.unsigned_long = name_ref cpp_compat_ulong, %cpp_compat_ulong
// CHECK:STDOUT:   %.loc19_80.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.f5e = addr_of %.loc19_80.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.1(%cpp_compat_ulong.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_80.2: init %ULongResult to %.loc19_80.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_35: type = splice_block %ULongResult.ref.loc19 [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref.loc19: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_80.3: ref %ULongResult = temporary %.loc19_80.1, %.loc19_80.2
// CHECK:STDOUT:   %.loc19_80.4: %ULongResult = acquire_value %.loc19_80.3
// CHECK:STDOUT:   %cpp_compat_ulong_result: %ULongResult = value_binding cpp_compat_ulong_result, %.loc19_80.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u32.patt: %pattern_type.4a9 = value_binding_pattern carbon_u32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc21: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc21_19: type = splice_block %u32 [concrete = constants.%u32] {
// CHECK:STDOUT:     %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc21: %.da1 = impl_witness_access constants.%ImplicitAs.impl_witness.e61, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.19d]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %cpp_unsigned_long.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc21: init %u32 = call %bound_method.loc21(%cpp_unsigned_long.ref.loc21)
// CHECK:STDOUT:   %.loc21_25.1: %u32 = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc21
// CHECK:STDOUT:   %.loc21_25.2: %u32 = converted %cpp_unsigned_long.ref.loc21, %.loc21_25.1
// CHECK:STDOUT:   %carbon_u32: %u32 = value_binding carbon_u32, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u32_result.patt: %pattern_type.dc7 = value_binding_pattern carbon_u32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc22: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u32.ref: %u32 = name_ref carbon_u32, %carbon_u32
// CHECK:STDOUT:   %.loc22_67.1: ref %UIntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.059 = addr_of %.loc22_67.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.2(%carbon_u32.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_67.2: init %UIntResult to %.loc22_67.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_29: type = splice_block %UIntResult.ref [concrete = constants.%UIntResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %UIntResult.ref: type = name_ref UIntResult, imports.%UIntResult.decl [concrete = constants.%UIntResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_67.3: ref %UIntResult = temporary %.loc22_67.1, %.loc22_67.2
// CHECK:STDOUT:   %.loc22_67.4: %UIntResult = acquire_value %.loc22_67.3
// CHECK:STDOUT:   %carbon_u32_result: %UIntResult = value_binding carbon_u32_result, %.loc22_67.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %.loc24_52.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_52: %.a6f = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc24_52.1: <bound method> = bound_method %float, %impl.elem0.loc24_52 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc24_52, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_52.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc24_52.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_52.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_52.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_52.4: init %f32.97e to %.loc24_52.3 = in_place_init %.loc24_52.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc24_52.5: init %array_type to %a.var = array_init (%.loc24_52.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_52.1, %.loc24_52.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_43: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc24: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc24_23.1: %.80d = impl_witness_access constants.%As.impl_witness.d5f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_23.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_23.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc24_23.1(%int_1.loc24) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %.loc24_23.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %.loc24_23.2: %Cpp.unsigned_long = converted %int_1.loc24, %.loc24_23.1 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %impl.elem0.loc24_23.2: %.4f2 = impl_witness_access constants.%ImplicitAs.impl_witness.a13, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.e53]
// CHECK:STDOUT:     %bound_method.loc24_23.2: <bound method> = bound_method %.loc24_23.2, %impl.elem0.loc24_23.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc24: init Core.IntLiteral = call %bound_method.loc24_23.2(%.loc24_23.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_23.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call.loc24 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_23.4: Core.IntLiteral = converted %.loc24_23.2, %.loc24_23.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_23.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   %UIntResult.cpp_destructor.bound: <bound method> = bound_method %.loc22_67.3, constants.%UIntResult.cpp_destructor
// CHECK:STDOUT:   %UIntResult.cpp_destructor.call: init %empty_tuple.type = call %UIntResult.cpp_destructor.bound(%.loc22_67.3)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_80.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call.loc19: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound.loc19(%.loc19_80.3)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound.loc16: <bound method> = bound_method %.loc16_82.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call.loc16: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound.loc16(%.loc16_82.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc24(%self.param: %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc22(%self.param: %UIntResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc19(%self.param: %ULongResult) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_unsigned_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long: type = class_type @ULong32 [concrete]
// CHECK:STDOUT:   %pattern_type.5b7: type = pattern_type %Cpp.unsigned_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3c3: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.691: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Cpp.unsigned_long) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a19: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5b5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.3c3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a19) [concrete]
// CHECK:STDOUT:   %.6bf: type = fn_type_with_self_type %ImplicitAs.Convert.type.691, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.0ab: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.ae3: <witness> = impl_witness imports.%Copy.impl_witness_table.36c [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long, (%Copy.impl_witness.ae3) [concrete]
// CHECK:STDOUT:   %.5a2: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op: %Cpp.unsigned_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.702: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5b5 = impl_witness_table (%Core.import_ref.702), @Core.IntLiteral.as.ImplicitAs.impl.541 [concrete]
// CHECK:STDOUT:   %Core.import_ref.1e0: %Cpp.unsigned_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.36c = impl_witness_table (%Core.import_ref.1e0), @Cpp.unsigned_long.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.6bf = impl_witness_access constants.%ImplicitAs.impl_witness.a19, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %.loc9_30: %Cpp.unsigned_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.5a2 = impl_witness_access constants.%Copy.impl_witness.ae3, element0 [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_30, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long = call %bound_method.loc9(%.loc9_30)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_13: type = splice_block %unsigned_long.ref.loc9 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc9: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long = ref_binding b, %b.var
// CHECK:STDOUT:   %DestroyOp.bound.loc9: <bound method> = bound_method %b.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc9: init %empty_tuple.type = call %DestroyOp.bound.loc9(%b.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc8: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc8: init %empty_tuple.type = call %DestroyOp.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Cpp.unsigned_long) = "no_op";
// CHECK:STDOUT:
