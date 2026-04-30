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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2ed: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.d29: %Int.as.ImplicitAs.impl.Convert.type.2ed = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.556: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d48: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.556) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4af: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.d48) [concrete]
// CHECK:STDOUT:   %.a64: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4af, %ImplicitAs.facet.d48 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.288: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.229: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c71: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.cee: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a54: %Core.IntLiteral.as.As.impl.Convert.type.cee = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.229 = facet_value Core.IntLiteral, (%As.impl_witness.c71) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.188: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i64, %As.facet) [concrete]
// CHECK:STDOUT:   %.fc7: type = fn_type_with_self_type %As.WithSelf.Convert.type.188, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a54 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a54, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.41b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e86: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ea2, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.ccf: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.535: %Int.as.ImplicitAs.impl.Convert.type.ccf = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.234: %ImplicitAs.type.139 = facet_value %i64, (%ImplicitAs.impl_witness.e86) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b50: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.234) [concrete]
// CHECK:STDOUT:   %.495: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b50, %ImplicitAs.facet.234 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.41a, %Int.as.ImplicitAs.impl.Convert.535 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.535, @Int.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.d1e: <bound method> = bound_method %int_1.41a, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e4d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.97e, %ImplicitAs.facet.945) [concrete]
// CHECK:STDOUT:   %.98d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e4d, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.57d: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.a64 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
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
// CHECK:STDOUT:   %i64.loc9: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long_long: %i64 = value_binding carbon_long_long, %cpp_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_55: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc11_55.1: <bound method> = bound_method %float, %impl.elem0.loc11_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_55: <specific function> = specific_function %impl.elem0.loc11_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_55.2: <bound method> = bound_method %float, %specific_fn.loc11_55 [concrete = constants.%bound_method.57d]
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
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.fc7 = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc11_30.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc11_30.2: %i64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.495 = impl_witness_access constants.%ImplicitAs.impl_witness.e86, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.535]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.d1e]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.46e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.7f8: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.607(%From.fe9) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.0ea: %UInt.as.ImplicitAs.impl.Convert.type.7f8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bb3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.899, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.fb8: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.042: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.fb8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.dfe: %ImplicitAs.type.584 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bb3) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.408: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.dfe) [concrete]
// CHECK:STDOUT:   %.f17: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.408, %ImplicitAs.facet.dfe [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.042 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.042, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.fec: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f23: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.f20: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.979: %Core.IntLiteral.as.As.impl.Convert.type.2b1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.f54: <witness> = impl_witness imports.%As.impl_witness_table.7eb, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0f2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.715: %Core.IntLiteral.as.As.impl.Convert.type.0f2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.f20 = facet_value Core.IntLiteral, (%As.impl_witness.f54) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.286: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u64, %As.facet) [concrete]
// CHECK:STDOUT:   %.904: type = fn_type_with_self_type %As.WithSelf.Convert.type.286, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.715 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.715, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.9cf: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2cc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.493, @UInt.as.ImplicitAs.impl.607(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.812: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.607(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.309: %UInt.as.ImplicitAs.impl.Convert.type.812 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.02d: %ImplicitAs.type.139 = facet_value %u64, (%ImplicitAs.impl_witness.2cc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.79a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.02d) [concrete]
// CHECK:STDOUT:   %.74a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.79a, %ImplicitAs.facet.02d [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.f23, %UInt.as.ImplicitAs.impl.Convert.309 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.309, @UInt.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.f18: <bound method> = bound_method %int_1.f23, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e4d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.97e, %ImplicitAs.facet.945) [concrete]
// CHECK:STDOUT:   %.98d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e4d, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.57d: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.741: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.46e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.899 = impl_witness_table (%Core.import_ref.741), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.483: @UInt.as.ImplicitAs.impl.607.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.7f8) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.607.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.0ea)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.493 = impl_witness_table (%Core.import_ref.483), @UInt.as.ImplicitAs.impl.607 [concrete]
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
// CHECK:STDOUT:     %unsigned_long_long.ref.loc8: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.f17 = impl_witness_access constants.%ImplicitAs.impl_witness.bb3, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.042]
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
// CHECK:STDOUT:   %u64.loc9: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long_long: %u64 = value_binding carbon_unsigned_long_long, %cpp_unsigned_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_64.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_64: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc11_64.1: <bound method> = bound_method %float, %impl.elem0.loc11_64 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_64: <specific function> = specific_function %impl.elem0.loc11_64, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_64.2: <bound method> = bound_method %float, %specific_fn.loc11_64 [concrete = constants.%bound_method.57d]
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
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.904 = impl_witness_access constants.%As.impl_witness.f54, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.715]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.9cf]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc11_30.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc11_30.2: %u64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.74a = impl_witness_access constants.%ImplicitAs.impl_witness.2cc, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.309]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @UInt.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.f18]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e8, %ImplicitAs.facet.eed [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.945: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.ba4) [concrete]
// CHECK:STDOUT:   %.12d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.945, %ImplicitAs.facet.ba4 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.431: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.bbf: %Cpp.long.as.ImplicitAs.impl.Convert.type.431 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.508: type = facet_type <@As, @As(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.60c: <witness> = impl_witness imports.%As.impl_witness_table.797 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.508 = facet_value Core.IntLiteral, (%As.impl_witness.60c) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.6c4: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long, %As.facet) [concrete]
// CHECK:STDOUT:   %.a3d: type = fn_type_with_self_type %As.WithSelf.Convert.type.6c4, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e4d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a09 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.772: %ImplicitAs.type.139 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.e4d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.1e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.772) [concrete]
// CHECK:STDOUT:   %.7f4: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.1e8, %ImplicitAs.facet.772 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.type.9ed: type = fn_type @Cpp.long.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.d49: %Cpp.long.as.ImplicitAs.impl.Convert.type.9ed = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5a4, %Cpp.long.as.ImplicitAs.impl.Convert.d49 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e4d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.97e, %ImplicitAs.facet.945) [concrete]
// CHECK:STDOUT:   %.98d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e4d, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc15: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
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
// CHECK:STDOUT:   %impl.elem0.loc24: %.12d = impl_witness_access constants.%ImplicitAs.impl_witness.bbc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bbf]
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
// CHECK:STDOUT:   %impl.elem0.loc26_50: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
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
// CHECK:STDOUT:     %impl.elem0.loc26_30.1: %.a3d = impl_witness_access constants.%As.impl_witness.60c, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc26_30.1: <bound method> = bound_method %int_1.loc26, %impl.elem0.loc26_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long = call %bound_method.loc26_30.1(%int_1.loc26) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %.loc26_30.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %.loc26_30.2: %Cpp.long = converted %int_1.loc26, %.loc26_30.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %impl.elem0.loc26_30.2: %.7f4 = impl_witness_access constants.%ImplicitAs.impl_witness.e4d, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.d49]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.e75: <witness> = impl_witness imports.%Copy.impl_witness_table.572 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.e75) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.11b: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.539: type = fn_type_with_self_type %Copy.WithSelf.Op.type.11b, %Copy.facet [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.539 = impl_witness_access constants.%Copy.impl_witness.e75, element0 [concrete = constants.%Cpp.long.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.494: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.6db: <witness> = impl_witness imports.%EqWith.impl_witness_table.322 [concrete]
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.494 = facet_value %Cpp.long, (%EqWith.impl_witness.6db) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.227: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.e56: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet) [concrete]
// CHECK:STDOUT:   %.909: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.227, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.cc0: type = fn_type @Cpp.long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.1b7: %Cpp.long.as.EqWith.impl.Equal.type.cc0 = struct_value () [concrete]
// CHECK:STDOUT:   %.163: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.e56, %EqWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.75f: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.5de: %Cpp.long.as.EqWith.impl.NotEqual.type.75f = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.a39: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.2b3: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.e82 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.a39 = facet_value %Cpp.long, (%OrderedWith.impl_witness.2b3) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.294: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.cf1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.2d5: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.8eb: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet) [concrete]
// CHECK:STDOUT:   %.fdc: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.2d5, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.da5: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.cc7: %Cpp.long.as.OrderedWith.impl.Greater.type.da5 = struct_value () [concrete]
// CHECK:STDOUT:   %.c90: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.294, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.d1e: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.911: %Cpp.long.as.OrderedWith.impl.Less.type.d1e = struct_value () [concrete]
// CHECK:STDOUT:   %.a9b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.8eb, %OrderedWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.9c8: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.337: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.9c8 = struct_value () [concrete]
// CHECK:STDOUT:   %.fad: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.cf1, %OrderedWith.facet [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc9_21.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.909 = impl_witness_access constants.%EqWith.impl_witness.6db, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1b7]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.163 = impl_witness_access constants.%EqWith.impl_witness.6db, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.5de]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.fdc = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.cc7]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.c90 = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.911]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.a9b = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.337]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.fad = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.187]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e8, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.d8f: type = facet_type <@EqWith, @EqWith(%i32)> [concrete]
// CHECK:STDOUT:   %T.57d: %ImplicitAs.type.819 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.503814.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.583(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.6454e5.1: %Cpp.long.as.EqWith.impl.NotEqual.type.503814.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.503814.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.583(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.6454e5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.503814.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.3e10d0.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.583(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.ee470e.1: %Cpp.long.as.EqWith.impl.Equal.type.3e10d0.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.3e10d0.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.583(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.ee470e.2: %Cpp.long.as.EqWith.impl.Equal.type.3e10d0.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.ded: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc39: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc39) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.495: <witness> = impl_witness imports.%EqWith.impl_witness_table.2e6, @Cpp.long.as.EqWith.impl.583(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.b2b39a.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.583(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.af4430.1: %Cpp.long.as.EqWith.impl.Equal.type.b2b39a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.e3f671.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.583(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.016eee.1: %Cpp.long.as.EqWith.impl.NotEqual.type.e3f671.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.b2b39a.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.583(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.af4430.2: %Cpp.long.as.EqWith.impl.Equal.type.b2b39a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.e3f671.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.583(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.016eee.2: %Cpp.long.as.EqWith.impl.NotEqual.type.e3f671.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.cee: %EqWith.type.d8f = facet_value %Cpp.long, (%EqWith.impl_witness.495) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.994: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i32, %EqWith.facet.cee) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.cd4: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i32, %EqWith.facet.cee) [concrete]
// CHECK:STDOUT:   %.a84: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.994, %EqWith.facet.cee [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.604db6.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.af4430.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9f0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.bb8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9f0, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.604db6.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.af4430.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.7cb: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.cd4, %EqWith.facet.cee [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.a2f319.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.016eee.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.a2f319.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.016eee.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.243: type = facet_type <@OrderedWith, @OrderedWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b86550.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.1c6(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.47ebeb.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b86550.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b86550.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.1c6(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.47ebeb.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b86550.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.b45ebb.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.1c6(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.a8f57e.1: %Cpp.long.as.OrderedWith.impl.Greater.type.b45ebb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.b45ebb.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.1c6(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.a8f57e.2: %Cpp.long.as.OrderedWith.impl.Greater.type.b45ebb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.17fc4f.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.1c6(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.85d51b.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.17fc4f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.17fc4f.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.1c6(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.85d51b.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.17fc4f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.520440.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.1c6(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.3d32d0.1: %Cpp.long.as.OrderedWith.impl.Less.type.520440.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.520440.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.1c6(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.3d32d0.2: %Cpp.long.as.OrderedWith.impl.Less.type.520440.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.358: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.e41: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.6e7, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.23d302.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.4bd542.1: %Cpp.long.as.OrderedWith.impl.Less.type.23d302.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.7a4c94.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.58c886.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.7a4c94.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.e4810a.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.fd6565.1: %Cpp.long.as.OrderedWith.impl.Greater.type.e4810a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fb6308.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.7fca0f.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fb6308.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.23d302.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.4bd542.2: %Cpp.long.as.OrderedWith.impl.Less.type.23d302.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.7a4c94.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.58c886.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.7a4c94.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.e4810a.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.fd6565.2: %Cpp.long.as.OrderedWith.impl.Greater.type.e4810a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fb6308.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.7fca0f.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fb6308.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.012: %OrderedWith.type.243 = facet_value %Cpp.long, (%OrderedWith.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.8ec: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.012) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.c79: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.012) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.fa2: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.012) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.c30: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.012) [concrete]
// CHECK:STDOUT:   %.410: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.fa2, %OrderedWith.facet.012 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.86d25e.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.fd6565.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.86d25e.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.fd6565.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.693: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.8ec, %OrderedWith.facet.012 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.098b91.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.4bd542.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.098b91.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.4bd542.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.cff: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.c30, %OrderedWith.facet.012 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.157c2e.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.7fca0f.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.157c2e.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.7fca0f.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.139: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.c79, %OrderedWith.facet.012 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.de1764.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.58c886.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.de1764.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.58c886.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.b3f: <witness> = impl_witness imports.%EqWith.impl_witness_table.2e6, @Cpp.long.as.EqWith.impl.583(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.febe5e.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.583(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.3a4fa9.1: %Cpp.long.as.EqWith.impl.Equal.type.febe5e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.582e5e.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.583(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.139b50.1: %Cpp.long.as.EqWith.impl.NotEqual.type.582e5e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.febe5e.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.583(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.3a4fa9.2: %Cpp.long.as.EqWith.impl.Equal.type.febe5e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.582e5e.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.583(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.139b50.2: %Cpp.long.as.EqWith.impl.NotEqual.type.582e5e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.1c2: %EqWith.type.ded = facet_value %Cpp.long, (%EqWith.impl_witness.b3f) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.278: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.1c2) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.937: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.1c2) [concrete]
// CHECK:STDOUT:   %.5de: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.278, %EqWith.facet.1c2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.0c655b.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.3a4fa9.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.0c655b.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.3a4fa9.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.97e: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.937, %EqWith.facet.1c2 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.1ba709.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.139b50.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.1ba709.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.139b50.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.26e: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.6e7, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.25dc38.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.2, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.cc257a.1: %Cpp.long.as.OrderedWith.impl.Less.type.25dc38.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1098b7.1: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.ef9474.1: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1098b7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.e1a7e9.1: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.2, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.afe0c2.1: %Cpp.long.as.OrderedWith.impl.Greater.type.e1a7e9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fefcb3.1: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.13ba60.1: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fefcb3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.type.25dc38.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Less.1, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.cc257a.2: %Cpp.long.as.OrderedWith.impl.Less.type.25dc38.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1098b7.2: type = fn_type @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.ef9474.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1098b7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.type.e1a7e9.2: type = fn_type @Cpp.long.as.OrderedWith.impl.Greater.1, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.afe0c2.2: %Cpp.long.as.OrderedWith.impl.Greater.type.e1a7e9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fefcb3.2: type = fn_type @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long.as.OrderedWith.impl.1c6(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.13ba60.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fefcb3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.1f7: %OrderedWith.type.358 = facet_value %Cpp.long, (%OrderedWith.impl_witness.26e) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.0f3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.1f7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.ab3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.1f7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.a37: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.1f7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.f82: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.1f7) [concrete]
// CHECK:STDOUT:   %.81e: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.a37, %OrderedWith.facet.1f7 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.d98231.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.afe0c2.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.d98231.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.afe0c2.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.8d3: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.0f3, %OrderedWith.facet.1f7 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.92e5d7.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.cc257a.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.92e5d7.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.cc257a.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.a3d: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.f82, %OrderedWith.facet.1f7 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.46a5e8.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.13ba60.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.46a5e8.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.13ba60.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.510: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.ab3, %OrderedWith.facet.1f7 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4c920e.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.ef9474.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4c920e.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.ef9474.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.93f: @Cpp.long.as.EqWith.impl.583.%Cpp.long.as.EqWith.impl.Equal.type.2 (%Cpp.long.as.EqWith.impl.Equal.type.3e10d0.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.583.%Cpp.long.as.EqWith.impl.Equal.2 (constants.%Cpp.long.as.EqWith.impl.Equal.ee470e.1)]
// CHECK:STDOUT:   %Core.import_ref.576: @Cpp.long.as.EqWith.impl.583.%Cpp.long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long.as.EqWith.impl.NotEqual.type.503814.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.EqWith.impl.583.%Cpp.long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long.as.EqWith.impl.NotEqual.6454e5.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.2e6 = impl_witness_table (%Core.import_ref.93f, %Core.import_ref.576), @Cpp.long.as.EqWith.impl.583 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.bac: @Cpp.long.as.EqWith.impl.583.%Cpp.long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long.as.EqWith.impl.NotEqual.type.503814.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long.as.EqWith.impl.583.%Cpp.long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long.as.EqWith.impl.NotEqual.6454e5.2)]
// CHECK:STDOUT:   %Core.Equal.3f1: @Cpp.long.as.EqWith.impl.583.%Cpp.long.as.EqWith.impl.Equal.type.1 (%Cpp.long.as.EqWith.impl.Equal.type.3e10d0.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long.as.EqWith.impl.583.%Cpp.long.as.EqWith.impl.Equal.1 (constants.%Cpp.long.as.EqWith.impl.Equal.ee470e.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.14c: @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.Less.type.2 (%Cpp.long.as.OrderedWith.impl.Less.type.520440.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long.as.OrderedWith.impl.Less.3d32d0.1)]
// CHECK:STDOUT:   %Core.import_ref.1ec: @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.17fc4f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.85d51b.1)]
// CHECK:STDOUT:   %Core.import_ref.005: @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long.as.OrderedWith.impl.Greater.type.b45ebb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long.as.OrderedWith.impl.Greater.a8f57e.1)]
// CHECK:STDOUT:   %Core.import_ref.a21: @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b86550.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.47ebeb.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.6e7 = impl_witness_table (%Core.import_ref.14c, %Core.import_ref.1ec, %Core.import_ref.005, %Core.import_ref.a21), @Cpp.long.as.OrderedWith.impl.1c6 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.786: @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.b86550.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.47ebeb.2)]
// CHECK:STDOUT:   %Core.Greater.64f: @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long.as.OrderedWith.impl.Greater.type.b45ebb.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long.as.OrderedWith.impl.Greater.a8f57e.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.9f4: @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.17fc4f.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.85d51b.2)]
// CHECK:STDOUT:   %Core.Less.287: @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.Less.type.1 (%Cpp.long.as.OrderedWith.impl.Less.type.520440.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.1c6.%Cpp.long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long.as.OrderedWith.impl.Less.3d32d0.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc9_21.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.a84 = impl_witness_access constants.%EqWith.impl_witness.495, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.af4430.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.604db6.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.EqWith.impl.Equal.type.b2b39a.1 = specific_constant imports.%Core.Equal.3f1, @Cpp.long.as.EqWith.impl.583(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.af4430.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Cpp.long.as.EqWith.impl.Equal.type.b2b39a.1 = name_ref Equal, %.loc10_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.af4430.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.604db6.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.7cb = impl_witness_access constants.%EqWith.impl_witness.495, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.016eee.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.a2f319.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.e3f671.1 = specific_constant imports.%Core.NotEqual.bac, @Cpp.long.as.EqWith.impl.583(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.016eee.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Cpp.long.as.EqWith.impl.NotEqual.type.e3f671.1 = name_ref NotEqual, %.loc11_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.016eee.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.a2f319.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.410 = impl_witness_access constants.%OrderedWith.impl_witness.e41, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.fd6565.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.86d25e.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.OrderedWith.impl.Greater.type.e4810a.1 = specific_constant imports.%Core.Greater.64f, @Cpp.long.as.OrderedWith.impl.1c6(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.fd6565.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Cpp.long.as.OrderedWith.impl.Greater.type.e4810a.1 = name_ref Greater, %.loc12_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.fd6565.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.86d25e.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_7: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_7: init %Cpp.long = call %bound_method.loc12_7(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_7
// CHECK:STDOUT:   %.loc12_7.2: %Cpp.long = converted %b.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.693 = impl_witness_access constants.%OrderedWith.impl_witness.e41, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.4bd542.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.098b91.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.OrderedWith.impl.Less.type.23d302.1 = specific_constant imports.%Core.Less.287, @Cpp.long.as.OrderedWith.impl.1c6(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.4bd542.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Cpp.long.as.OrderedWith.impl.Less.type.23d302.1 = name_ref Less, %.loc13_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.4bd542.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.098b91.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_7: init %Cpp.long = call %bound_method.loc13_7(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_7
// CHECK:STDOUT:   %.loc13_7.2: %Cpp.long = converted %b.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.cff = impl_witness_access constants.%OrderedWith.impl_witness.e41, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.7fca0f.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.157c2e.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fb6308.1 = specific_constant imports.%Core.GreaterOrEquivalent.786, @Cpp.long.as.OrderedWith.impl.1c6(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.7fca0f.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fb6308.1 = name_ref GreaterOrEquivalent, %.loc14_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.7fca0f.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.157c2e.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.139 = impl_witness_access constants.%OrderedWith.impl_witness.e41, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.58c886.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.de1764.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.7a4c94.1 = specific_constant imports.%Core.LessOrEquivalent.9f4, @Cpp.long.as.OrderedWith.impl.1c6(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.58c886.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.7a4c94.1 = name_ref LessOrEquivalent, %.loc15_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.58c886.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_5: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_5: init %Cpp.long = call %bound_method.loc15_5.3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_5
// CHECK:STDOUT:   %.loc15_5.4: %Cpp.long = converted %b.ref.loc15, %.loc15_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.de1764.2]
// CHECK:STDOUT:   %bound_method.loc15_5.4: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15_8: init %Cpp.long = call %bound_method.loc15_8(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15_8
// CHECK:STDOUT:   %.loc15_8.2: %Cpp.long = converted %b.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.4(%a.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.5de = impl_witness_access constants.%EqWith.impl_witness.b3f, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.3a4fa9.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.0c655b.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.EqWith.impl.Equal.type.febe5e.1 = specific_constant imports.%Core.Equal.3f1, @Cpp.long.as.EqWith.impl.583(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.3a4fa9.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long.as.EqWith.impl.Equal.type.febe5e.1 = name_ref Equal, %.loc17_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.3a4fa9.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long = call %bound_method.loc17_5.3(%int_1.loc17) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long = converted %int_1.loc17, %.loc17_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.0c655b.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.97e = impl_witness_access constants.%EqWith.impl_witness.b3f, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.139b50.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.1ba709.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.582e5e.1 = specific_constant imports.%Core.NotEqual.bac, @Cpp.long.as.EqWith.impl.583(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.139b50.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long.as.EqWith.impl.NotEqual.type.582e5e.1 = name_ref NotEqual, %.loc18_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.139b50.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long = call %bound_method.loc18_5.3(%int_1.loc18) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long = converted %int_1.loc18, %.loc18_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.1ba709.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.81e = impl_witness_access constants.%OrderedWith.impl_witness.26e, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.afe0c2.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.d98231.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.OrderedWith.impl.Greater.type.e1a7e9.1 = specific_constant imports.%Core.Greater.64f, @Cpp.long.as.OrderedWith.impl.1c6(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.afe0c2.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long.as.OrderedWith.impl.Greater.type.e1a7e9.1 = name_ref Greater, %.loc19_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.afe0c2.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long = call %bound_method.loc19_5.3(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long = converted %int_1.loc19, %.loc19_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.d98231.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_7: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7: init %Cpp.long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_7 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.8d3 = impl_witness_access constants.%OrderedWith.impl_witness.26e, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.cc257a.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.92e5d7.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.OrderedWith.impl.Less.type.25dc38.1 = specific_constant imports.%Core.Less.287, @Cpp.long.as.OrderedWith.impl.1c6(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.cc257a.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long.as.OrderedWith.impl.Less.type.25dc38.1 = name_ref Less, %.loc20_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.cc257a.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long = call %bound_method.loc20_5.3(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long = converted %int_1.loc20, %.loc20_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.92e5d7.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7: init %Cpp.long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_7 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.4(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.a3d = impl_witness_access constants.%OrderedWith.impl_witness.26e, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.13ba60.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.46a5e8.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fefcb3.1 = specific_constant imports.%Core.GreaterOrEquivalent.786, @Cpp.long.as.OrderedWith.impl.1c6(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.13ba60.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.fefcb3.1 = name_ref GreaterOrEquivalent, %.loc21_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.13ba60.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_5: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5: init %Cpp.long = call %bound_method.loc21_5.3(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_5.4: %Cpp.long = converted %int_1.loc21, %.loc21_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.46a5e8.2]
// CHECK:STDOUT:   %bound_method.loc21_5.4: <bound method> = bound_method %a.ref.loc21, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8: init %Cpp.long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.4(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.510 = impl_witness_access constants.%OrderedWith.impl_witness.26e, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.ef9474.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc22 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4c920e.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1098b7.1 = specific_constant imports.%Core.LessOrEquivalent.9f4, @Cpp.long.as.OrderedWith.impl.1c6(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.ef9474.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.1098b7.1 = name_ref LessOrEquivalent, %.loc22_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.ef9474.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_5: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5: init %Cpp.long = call %bound_method.loc22_5.3(%int_1.loc22) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22_5.4: %Cpp.long = converted %int_1.loc22, %.loc22_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.4c920e.2]
// CHECK:STDOUT:   %bound_method.loc22_5.4: <bound method> = bound_method %a.ref.loc22, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22_8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_8: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e8, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.494: type = facet_type <@EqWith, @EqWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.57d: %ImplicitAs.type.819 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.2a3875.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.243(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.96a635.1: %T.as_type.as.EqWith.impl.NotEqual.type.2a3875.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.2a3875.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.243(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.96a635.2: %T.as_type.as.EqWith.impl.NotEqual.type.2a3875.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.f439f4.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.243(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.f81ea9.1: %T.as_type.as.EqWith.impl.Equal.type.f439f4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.f439f4.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.243(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.f81ea9.2: %T.as_type.as.EqWith.impl.Equal.type.f439f4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc39: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc39) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.ee3: <witness> = impl_witness imports.%EqWith.impl_witness_table.213, @T.as_type.as.EqWith.impl.243(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.212eb2.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.243(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.6f6f00.1: %T.as_type.as.EqWith.impl.Equal.type.212eb2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.eb206f.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.243(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.635f4a.1: %T.as_type.as.EqWith.impl.NotEqual.type.eb206f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.212eb2.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.243(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.6f6f00.2: %T.as_type.as.EqWith.impl.Equal.type.212eb2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.eb206f.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.243(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.635f4a.2: %T.as_type.as.EqWith.impl.NotEqual.type.eb206f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.df3: %EqWith.type.494 = facet_value %i32, (%EqWith.impl_witness.ee3) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.655: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.df3) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.313: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.df3) [concrete]
// CHECK:STDOUT:   %.465: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.655, %EqWith.facet.df3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.9c9973.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.6f6f00.2, @T.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.9c9973.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.6f6f00.1, @T.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9f0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.bb8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9f0, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b3a: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.313, %EqWith.facet.df3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.1cadb2.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.635f4a.2, @T.as_type.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.1cadb2.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.635f4a.1, @T.as_type.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.a39: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.11d880.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d5b(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.392459.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.11d880.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.11d880.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d5b(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.392459.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.11d880.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.d4be71.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.d5b(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.df4f5e.1: %T.as_type.as.OrderedWith.impl.Greater.type.d4be71.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.d4be71.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.d5b(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.df4f5e.2: %T.as_type.as.OrderedWith.impl.Greater.type.d4be71.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43ddde.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d5b(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.881e0b.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43ddde.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43ddde.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d5b(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.881e0b.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43ddde.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.92ce55.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.d5b(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.d504b0.1: %T.as_type.as.OrderedWith.impl.Less.type.92ce55.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.92ce55.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.d5b(%T.57d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.d504b0.2: %T.as_type.as.OrderedWith.impl.Less.type.92ce55.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.f99: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.666, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.e61b26.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.11e1b5.1: %T.as_type.as.OrderedWith.impl.Less.type.e61b26.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.859fa0.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.69bd3d.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.859fa0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.ad12e9.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.26833a.1: %T.as_type.as.OrderedWith.impl.Greater.type.ad12e9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2146ac.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c5cfc4.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2146ac.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.e61b26.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.11e1b5.2: %T.as_type.as.OrderedWith.impl.Less.type.e61b26.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.859fa0.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.69bd3d.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.859fa0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.ad12e9.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.26833a.2: %T.as_type.as.OrderedWith.impl.Greater.type.ad12e9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2146ac.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c5cfc4.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2146ac.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.bc6: %OrderedWith.type.a39 = facet_value %i32, (%OrderedWith.impl_witness.f99) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.375: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.bc6) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.9be: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.bc6) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.fc7: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.bc6) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.f6d: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.bc6) [concrete]
// CHECK:STDOUT:   %.d57: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.fc7, %OrderedWith.facet.bc6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.001a30.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.26833a.2, @T.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.001a30.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.26833a.1, @T.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.400: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.375, %OrderedWith.facet.bc6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.c6ac49.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.11e1b5.2, @T.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.c6ac49.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.11e1b5.1, @T.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.3fe: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.f6d, %OrderedWith.facet.bc6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1344d6.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c5cfc4.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1344d6.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c5cfc4.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.2f8: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.9be, %OrderedWith.facet.bc6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b047b5.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.69bd3d.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b047b5.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.69bd3d.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.80a: <witness> = impl_witness imports.%EqWith.impl_witness_table.213, @T.as_type.as.EqWith.impl.243(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.2b9ebd.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.2, @T.as_type.as.EqWith.impl.243(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.5f3ba2.1: %T.as_type.as.EqWith.impl.Equal.type.2b9ebd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.9f653c.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.2, @T.as_type.as.EqWith.impl.243(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.cd4556.1: %T.as_type.as.EqWith.impl.NotEqual.type.9f653c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.2b9ebd.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.1, @T.as_type.as.EqWith.impl.243(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.5f3ba2.2: %T.as_type.as.EqWith.impl.Equal.type.2b9ebd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.9f653c.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.1, @T.as_type.as.EqWith.impl.243(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.cd4556.2: %T.as_type.as.EqWith.impl.NotEqual.type.9f653c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.a5f: %EqWith.type.494 = facet_value Core.IntLiteral, (%EqWith.impl_witness.80a) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.42c: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.a5f) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.000: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.a5f) [concrete]
// CHECK:STDOUT:   %.dc5: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.42c, %EqWith.facet.a5f [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.371af1.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.5f3ba2.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.fe3673.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.5f3ba2.2, @T.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.13fd29.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.fe3673.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.371af1.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.5f3ba2.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.fe3673.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.5f3ba2.1, @T.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.13fd29.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.fe3673.2 [concrete]
// CHECK:STDOUT:   %.627: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.000, %EqWith.facet.a5f [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.32c2a1.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.cd4556.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.5a7ce8.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.cd4556.2, @T.as_type.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.034e18.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.5a7ce8.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.32c2a1.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.cd4556.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.5a7ce8.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.cd4556.1, @T.as_type.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.034e18.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.5a7ce8.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.71d: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.666, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.4c5e47.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.2, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.780c9a.1: %T.as_type.as.OrderedWith.impl.Less.type.4c5e47.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.89841d.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e90faf.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.89841d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.b828c1.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.2, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.89db94.1: %T.as_type.as.OrderedWith.impl.Greater.type.b828c1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.53fbc2.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c1efc3.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.53fbc2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.4c5e47.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.1, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.780c9a.2: %T.as_type.as.OrderedWith.impl.Less.type.4c5e47.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.89841d.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e90faf.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.89841d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.b828c1.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.1, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.89db94.2: %T.as_type.as.OrderedWith.impl.Greater.type.b828c1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.53fbc2.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1, @T.as_type.as.OrderedWith.impl.d5b(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c1efc3.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.53fbc2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.e9c: %OrderedWith.type.a39 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.71d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b9e: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.e9c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.598: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.e9c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.ad1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.e9c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.eb5: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.e9c) [concrete]
// CHECK:STDOUT:   %.ab5: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.ad1, %OrderedWith.facet.e9c [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.eab164.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.89db94.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.d39b8b.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.89db94.2, @T.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.b3316e.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.d39b8b.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.eab164.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.89db94.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.d39b8b.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.89db94.1, @T.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.b3316e.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.d39b8b.2 [concrete]
// CHECK:STDOUT:   %.510: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.b9e, %OrderedWith.facet.e9c [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.006be7.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.780c9a.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.345a6a.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.780c9a.2, @T.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.fda4bb.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.345a6a.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.006be7.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.780c9a.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.345a6a.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.780c9a.1, @T.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.fda4bb.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.345a6a.2 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.eb5, %OrderedWith.facet.e9c [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f605d4.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c1efc3.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.26ce16.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c1efc3.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.f8c005.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.26ce16.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f605d4.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c1efc3.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.26ce16.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c1efc3.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.f8c005.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.26ce16.2 [concrete]
// CHECK:STDOUT:   %.8dc: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.598, %OrderedWith.facet.e9c [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.4b587c.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e90faf.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.f1965f.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e90faf.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.08e701.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.f1965f.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.4b587c.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e90faf.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.f1965f.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e90faf.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.08e701.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.f1965f.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.89f: @T.as_type.as.EqWith.impl.243.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.f439f4.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.243.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.f81ea9.1)]
// CHECK:STDOUT:   %Core.import_ref.ac0: @T.as_type.as.EqWith.impl.243.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.2a3875.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.243.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.96a635.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.213 = impl_witness_table (%Core.import_ref.89f, %Core.import_ref.ac0), @T.as_type.as.EqWith.impl.243 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.ffe: @T.as_type.as.EqWith.impl.243.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.2a3875.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.243.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.96a635.2)]
// CHECK:STDOUT:   %Core.Equal.e00: @T.as_type.as.EqWith.impl.243.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.f439f4.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.243.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.f81ea9.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.332: @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.92ce55.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.d504b0.1)]
// CHECK:STDOUT:   %Core.import_ref.795: @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43ddde.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.881e0b.1)]
// CHECK:STDOUT:   %Core.import_ref.8be: @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.d4be71.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.df4f5e.1)]
// CHECK:STDOUT:   %Core.import_ref.5c4: @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.11d880.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.392459.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.666 = impl_witness_table (%Core.import_ref.332, %Core.import_ref.795, %Core.import_ref.8be, %Core.import_ref.5c4), @T.as_type.as.OrderedWith.impl.d5b [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.1c9: @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.11d880.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.392459.2)]
// CHECK:STDOUT:   %Core.Greater.a1a: @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.d4be71.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.df4f5e.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.461: @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.43ddde.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.881e0b.2)]
// CHECK:STDOUT:   %Core.Less.80c: @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.92ce55.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.d5b.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.d504b0.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc9_21.2
// CHECK:STDOUT:   %b.ref.loc10: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.465 = impl_witness_access constants.%EqWith.impl_witness.ee3, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6f6f00.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.9c9973.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.as_type.as.EqWith.impl.Equal.type.212eb2.1 = specific_constant imports.%Core.Equal.e00, @T.as_type.as.EqWith.impl.243(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6f6f00.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.212eb2.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.6f6f00.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.9c9973.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long = call %bound_method.loc10_3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %Cpp.long = converted %b.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %a.ref.loc10)
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.b3a = impl_witness_access constants.%EqWith.impl_witness.ee3, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.635f4a.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.1cadb2.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.as_type.as.EqWith.impl.NotEqual.type.eb206f.1 = specific_constant imports.%Core.NotEqual.ffe, @T.as_type.as.EqWith.impl.243(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.635f4a.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.eb206f.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.635f4a.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.1cadb2.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long = call %bound_method.loc11_3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %Cpp.long = converted %b.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %a.ref.loc11)
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.d57 = impl_witness_access constants.%OrderedWith.impl_witness.f99, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.26833a.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.001a30.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.as_type.as.OrderedWith.impl.Greater.type.ad12e9.1 = specific_constant imports.%Core.Greater.a1a, @T.as_type.as.OrderedWith.impl.d5b(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.26833a.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.ad12e9.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.26833a.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.001a30.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12_3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %Cpp.long = converted %b.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %a.ref.loc12)
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.400 = impl_witness_access constants.%OrderedWith.impl_witness.f99, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.11e1b5.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.c6ac49.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.as_type.as.OrderedWith.impl.Less.type.e61b26.1 = specific_constant imports.%Core.Less.80c, @T.as_type.as.OrderedWith.impl.d5b(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.11e1b5.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.e61b26.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.11e1b5.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.c6ac49.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long = call %bound_method.loc13_3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %Cpp.long = converted %b.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %a.ref.loc13)
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.3fe = impl_witness_access constants.%OrderedWith.impl_witness.f99, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c5cfc4.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1344d6.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2146ac.1 = specific_constant imports.%Core.GreaterOrEquivalent.1c9, @T.as_type.as.OrderedWith.impl.d5b(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c5cfc4.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2146ac.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c5cfc4.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1344d6.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long = call %bound_method.loc14_3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %Cpp.long = converted %b.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %a.ref.loc14)
// CHECK:STDOUT:   %b.ref.loc15: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.2f8 = impl_witness_access constants.%OrderedWith.impl_witness.f99, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.69bd3d.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b047b5.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.859fa0.1 = specific_constant imports.%Core.LessOrEquivalent.461, @T.as_type.as.OrderedWith.impl.d5b(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.69bd3d.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.859fa0.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.69bd3d.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.b047b5.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long = call %bound_method.loc15_3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %Cpp.long = converted %b.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %a.ref.loc15)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.dc5 = impl_witness_access constants.%EqWith.impl_witness.80a, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.5f3ba2.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.371af1.1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.fe3673.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.13fd29.1]
// CHECK:STDOUT:   %.loc17_5: %T.as_type.as.EqWith.impl.Equal.type.2b9ebd.1 = specific_constant imports.%Core.Equal.e00, @T.as_type.as.EqWith.impl.243(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.5f3ba2.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.2b9ebd.1 = name_ref Equal, %.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.5f3ba2.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.371af1.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.fe3673.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17 [concrete = constants.%bound_method.13fd29.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.627 = impl_witness_access constants.%EqWith.impl_witness.80a, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.cd4556.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.32c2a1.1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.5a7ce8.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.034e18.1]
// CHECK:STDOUT:   %.loc18_5: %T.as_type.as.EqWith.impl.NotEqual.type.9f653c.1 = specific_constant imports.%Core.NotEqual.ffe, @T.as_type.as.EqWith.impl.243(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.cd4556.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.9f653c.1 = name_ref NotEqual, %.loc18_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.cd4556.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.32c2a1.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.5a7ce8.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18 [concrete = constants.%bound_method.034e18.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.ab5 = impl_witness_access constants.%OrderedWith.impl_witness.71d, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.89db94.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.eab164.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.d39b8b.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.b3316e.1]
// CHECK:STDOUT:   %.loc19_5: %T.as_type.as.OrderedWith.impl.Greater.type.b828c1.1 = specific_constant imports.%Core.Greater.a1a, @T.as_type.as.OrderedWith.impl.d5b(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.89db94.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.b828c1.1 = name_ref Greater, %.loc19_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.89db94.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.eab164.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.d39b8b.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19 [concrete = constants.%bound_method.b3316e.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.510 = impl_witness_access constants.%OrderedWith.impl_witness.71d, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.780c9a.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.006be7.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.345a6a.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.fda4bb.1]
// CHECK:STDOUT:   %.loc20_5: %T.as_type.as.OrderedWith.impl.Less.type.4c5e47.1 = specific_constant imports.%Core.Less.80c, @T.as_type.as.OrderedWith.impl.d5b(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.780c9a.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.4c5e47.1 = name_ref Less, %.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.780c9a.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.006be7.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.345a6a.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20 [concrete = constants.%bound_method.fda4bb.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.a0b = impl_witness_access constants.%OrderedWith.impl_witness.71d, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c1efc3.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f605d4.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.26ce16.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.f8c005.1]
// CHECK:STDOUT:   %.loc21_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.53fbc2.1 = specific_constant imports.%Core.GreaterOrEquivalent.1c9, @T.as_type.as.OrderedWith.impl.d5b(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c1efc3.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.53fbc2.1 = name_ref GreaterOrEquivalent, %.loc21_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.c1efc3.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.f605d4.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.26ce16.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21 [concrete = constants.%bound_method.f8c005.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.8dc = impl_witness_access constants.%OrderedWith.impl_witness.71d, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.e90faf.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.4b587c.1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.f1965f.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.08e701.1]
// CHECK:STDOUT:   %.loc22_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.89841d.1 = specific_constant imports.%Core.LessOrEquivalent.461, @T.as_type.as.OrderedWith.impl.d5b(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.e90faf.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.89841d.1 = name_ref LessOrEquivalent, %.loc22_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.e90faf.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.4b587c.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.f1965f.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22 [concrete = constants.%bound_method.08e701.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc22_3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %Cpp.long, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.b80: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.22b: type = fn_type_with_self_type %Negate.WithSelf.Op.type.b80, %Negate.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op.type: type = fn_type @Cpp.long.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op: %Cpp.long.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.9c2: type = facet_type <@AddWith, @AddWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.e9f: <witness> = impl_witness imports.%AddWith.impl_witness_table.2f5 [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.9c2 = facet_value %Cpp.long, (%AddWith.impl_witness.e9f) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.914: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Cpp.long, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %.165: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.914, %AddWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.type.094: type = fn_type @Cpp.long.as.AddWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.be0: %Cpp.long.as.AddWith.impl.Op.type.094 = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.type.e8e: type = facet_type <@SubWith, @SubWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness.8a6: <witness> = impl_witness imports.%SubWith.impl_witness_table.e2e [concrete]
// CHECK:STDOUT:   %SubWith.facet: %SubWith.type.e8e = facet_value %Cpp.long, (%SubWith.impl_witness.8a6) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.a4a: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Cpp.long, %SubWith.facet) [concrete]
// CHECK:STDOUT:   %.571: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.a4a, %SubWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.type.882: type = fn_type @Cpp.long.as.SubWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.37d: %Cpp.long.as.SubWith.impl.Op.type.882 = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.type.a32: type = facet_type <@MulWith, @MulWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness.87c: <witness> = impl_witness imports.%MulWith.impl_witness_table.20e [concrete]
// CHECK:STDOUT:   %MulWith.facet: %MulWith.type.a32 = facet_value %Cpp.long, (%MulWith.impl_witness.87c) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.038: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Cpp.long, %MulWith.facet) [concrete]
// CHECK:STDOUT:   %.7d0: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.038, %MulWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.type.31a: type = fn_type @Cpp.long.as.MulWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.4e5: %Cpp.long.as.MulWith.impl.Op.type.31a = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.type.0c0: type = facet_type <@DivWith, @DivWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness.4ab: <witness> = impl_witness imports.%DivWith.impl_witness_table.1ae [concrete]
// CHECK:STDOUT:   %DivWith.facet: %DivWith.type.0c0 = facet_value %Cpp.long, (%DivWith.impl_witness.4ab) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.d57: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Cpp.long, %DivWith.facet) [concrete]
// CHECK:STDOUT:   %.aa3: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.d57, %DivWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.type.5e7: type = fn_type @Cpp.long.as.DivWith.impl.Op.3 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.33f: %Cpp.long.as.DivWith.impl.Op.type.5e7 = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.type.2fc: type = facet_type <@ModWith, @ModWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness.c73: <witness> = impl_witness imports.%ModWith.impl_witness_table.ed4 [concrete]
// CHECK:STDOUT:   %ModWith.facet: %ModWith.type.2fc = facet_value %Cpp.long, (%ModWith.impl_witness.c73) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.b7d: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Cpp.long, %ModWith.facet) [concrete]
// CHECK:STDOUT:   %.c19: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.b7d, %ModWith.facet [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_21.2: %Cpp.long = converted %int_1.loc9, %.loc9_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %y: %Cpp.long = value_binding y, %.loc9_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc11: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc11: %.22b = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Cpp.long.as.Negate.impl.Op]
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
// CHECK:STDOUT:   %impl.elem1.loc12: %.165 = impl_witness_access constants.%AddWith.impl_witness.e9f, element1 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.be0]
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
// CHECK:STDOUT:   %impl.elem1.loc13: %.571 = impl_witness_access constants.%SubWith.impl_witness.8a6, element1 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.37d]
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
// CHECK:STDOUT:   %impl.elem1.loc14: %.7d0 = impl_witness_access constants.%MulWith.impl_witness.87c, element1 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.4e5]
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
// CHECK:STDOUT:   %impl.elem1.loc15: %.aa3 = impl_witness_access constants.%DivWith.impl_witness.4ab, element1 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.33f]
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
// CHECK:STDOUT:   %impl.elem1.loc16: %.c19 = impl_witness_access constants.%ModWith.impl_witness.c73, element1 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.06e]
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
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.e0a: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.035) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.3a8: %T.as_type.as.ImplicitAs.impl.Convert.type.e0a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e8, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.bef: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.09d746.1: %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.09d746.2: %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e75: <witness> = impl_witness imports.%Copy.impl_witness_table.572 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long, (%Copy.impl_witness.e75) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.495: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.fe1, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d75: %ImplicitAs.type.819 = facet_value %Cpp.long, (%ImplicitAs.impl_witness.495) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.8e3: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.59e, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.df4de2.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.e6e9d1.1: %Cpp.long.as.AddAssignWith.impl.Op.type.df4de2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.df4de2.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.e6e9d1.2: %Cpp.long.as.AddAssignWith.impl.Op.type.df4de2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.bef = facet_value %Cpp.long, (%AddAssignWith.impl_witness.8e3) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.177: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.051: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.177, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.949874.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.e6e9d1.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.949874.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.e6e9d1.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.b12: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.4445c4.1: %Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.4445c4.2: %Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.76e: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.714, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.f5cf5c.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.18b17c.1: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cf5c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.f5cf5c.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.18b17c.2: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cf5c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.b12 = facet_value %Cpp.long, (%SubAssignWith.impl_witness.76e) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.5bb: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.3ea: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.5bb, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.451737.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.18b17c.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.451737.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.18b17c.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.59a: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.684acb.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.6a090a.1: %Cpp.long.as.MulAssignWith.impl.Op.type.684acb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.684acb.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.6a090a.2: %Cpp.long.as.MulAssignWith.impl.Op.type.684acb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.a61: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.71c, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.de7124.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.4074b4.1: %Cpp.long.as.MulAssignWith.impl.Op.type.de7124.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.de7124.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.4074b4.2: %Cpp.long.as.MulAssignWith.impl.Op.type.de7124.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.59a = facet_value %Cpp.long, (%MulAssignWith.impl_witness.a61) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.ca0: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.c10: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.ca0, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.8626d1.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.4074b4.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.8626d1.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.4074b4.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.e6d: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.baaf86.1: %Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.baaf86.2: %Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.4cd: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.483, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.a1b2e9.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.d4ba73.1: %Cpp.long.as.DivAssignWith.impl.Op.type.a1b2e9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.a1b2e9.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.d4ba73.2: %Cpp.long.as.DivAssignWith.impl.Op.type.a1b2e9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.e6d = facet_value %Cpp.long, (%DivAssignWith.impl_witness.4cd) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.70a: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.c93: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.70a, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.c8e917.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.d4ba73.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.c8e917.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.d4ba73.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.2a3: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.206fcc.1: %Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.206fcc.2: %Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.8de: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.1fd, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.c9cf10.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.02f8e5.1: %Cpp.long.as.ModAssignWith.impl.Op.type.c9cf10.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.c9cf10.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.02f8e5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.c9cf10.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.2a3 = facet_value %Cpp.long, (%ModAssignWith.impl_witness.8de) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.405: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.1d7: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.405, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.7a7fb3.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.02f8e5.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.7a7fb3.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.02f8e5.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.efd: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.0b15fa.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.0b15fa.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.e4e: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.ac5, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.afec48.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.2d36ff.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.afec48.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.afec48.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.2d36ff.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.afec48.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.efd = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.e4e) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.249: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.4c7: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.249, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.503d46.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.2d36ff.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.503d46.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.2d36ff.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.b27: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.3b2429.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.3b2429.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.2ac: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.d80, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.a907f1.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.35a7b6.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a907f1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.a907f1.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.35a7b6.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a907f1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.b27 = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.2ac) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.1ce: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.7e5: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.1ce, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.2952dc.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.35a7b6.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.2952dc.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.35a7b6.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.431: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.3c8402.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.3c8402.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.8b2: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.fe4, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.f8190f.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.fe5efd.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.f8190f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.f8190f.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.fe5efd.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.f8190f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.431 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.8b2) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.6f3: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.754: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.6f3, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.0562d9.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.fe5efd.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.0562d9.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.fe5efd.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.bed: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.f91cfa.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.f91cfa.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.731: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.c76, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9e13fd.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.580c5b.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9e13fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9e13fd.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.580c5b.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9e13fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.bed = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.731) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.66c: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.227: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.66c, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.51b64e.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.580c5b.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.51b64e.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.580c5b.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.222: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.43d513.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.43d513.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.fa6: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.2d4, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.2a9012.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.6ebbd9.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.2a9012.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.2a9012.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.6ebbd9.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.2a9012.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.222 = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.fa6) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.ea8: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.e73: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.ea8, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c601.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.6ebbd9.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.d75) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c601.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.6ebbd9.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.d75) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.2a3: @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert.type (%T.as_type.as.ImplicitAs.impl.Convert.type.e0a) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert (constants.%T.as_type.as.ImplicitAs.impl.Convert.3a8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.fe1 = impl_witness_table (%Core.import_ref.2a3), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c07: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.09d746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.59e = impl_witness_table (%Core.import_ref.c07), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.8bb: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.09d746.2)]
// CHECK:STDOUT:   %Core.import_ref.915: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.572 = impl_witness_table (%Core.import_ref.915), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1a6: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.4445c4.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.714 = impl_witness_table (%Core.import_ref.1a6), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.889: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.4445c4.2)]
// CHECK:STDOUT:   %Core.import_ref.ab5: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.684acb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.6a090a.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.71c = impl_witness_table (%Core.import_ref.ab5), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c34: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.684acb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.6a090a.2)]
// CHECK:STDOUT:   %Core.import_ref.585: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.baaf86.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.483 = impl_witness_table (%Core.import_ref.585), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.cd4: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.baaf86.2)]
// CHECK:STDOUT:   %Core.import_ref.1ef: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.206fcc.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.1fd = impl_witness_table (%Core.import_ref.1ef), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.0d4c: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.206fcc.2)]
// CHECK:STDOUT:   %Core.import_ref.045: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.0b15fa.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.ac5 = impl_witness_table (%Core.import_ref.045), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.984: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.0b15fa.2)]
// CHECK:STDOUT:   %Core.import_ref.5e2: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.3b2429.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.d80 = impl_witness_table (%Core.import_ref.5e2), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7ac: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.3b2429.2)]
// CHECK:STDOUT:   %Core.import_ref.261: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c8402.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.fe4 = impl_witness_table (%Core.import_ref.261), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d36: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c8402.2)]
// CHECK:STDOUT:   %Core.import_ref.8d3: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.f91cfa.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.c76 = impl_witness_table (%Core.import_ref.8d3), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f28: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.f91cfa.2)]
// CHECK:STDOUT:   %Core.import_ref.559: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.43d513.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.2d4 = impl_witness_table (%Core.import_ref.559), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c63: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.43d513.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc11: %.051 = impl_witness_access constants.%AddAssignWith.impl_witness.8e3, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.e6e9d1.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.495) [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.949874.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.df4de2.1 = specific_constant imports.%Core.Op.8bb, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.e6e9d1.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.AddAssignWith.impl.Op.type.df4de2.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.e6e9d1.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.949874.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.3ea = impl_witness_access constants.%SubAssignWith.impl_witness.76e, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.18b17c.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.495) [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.451737.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cf5c.1 = specific_constant imports.%Core.Op.889, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.18b17c.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cf5c.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.18b17c.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.451737.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.c10 = impl_witness_access constants.%MulAssignWith.impl_witness.a61, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.4074b4.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.495) [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.8626d1.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.de7124.1 = specific_constant imports.%Core.Op.c34, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.4074b4.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.MulAssignWith.impl.Op.type.de7124.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.4074b4.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.8626d1.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.c93 = impl_witness_access constants.%DivAssignWith.impl_witness.4cd, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.d4ba73.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.495) [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.c8e917.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.DivAssignWith.impl.Op.type.a1b2e9.1 = specific_constant imports.%Core.Op.cd4, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.d4ba73.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.DivAssignWith.impl.Op.type.a1b2e9.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.d4ba73.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.c8e917.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.1d7 = impl_witness_access constants.%ModAssignWith.impl_witness.8de, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.02f8e5.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.495) [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.7a7fb3.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.c9cf10.1 = specific_constant imports.%Core.Op.0d4c, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.02f8e5.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long.as.ModAssignWith.impl.Op.type.c9cf10.1 = name_ref Op, %.loc15_5.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.02f8e5.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.7a7fb3.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.4c7 = impl_witness_access constants.%BitAndAssignWith.impl_witness.e4e, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.2d36ff.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.495) [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.503d46.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.afec48.1 = specific_constant imports.%Core.Op.984, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.2d36ff.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitAndAssignWith.impl.Op.type.afec48.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.2d36ff.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.503d46.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.7e5 = impl_witness_access constants.%BitOrAssignWith.impl_witness.2ac, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.35a7b6.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.495) [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.2952dc.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a907f1.1 = specific_constant imports.%Core.Op.7ac, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.35a7b6.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a907f1.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.35a7b6.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.2952dc.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.754 = impl_witness_access constants.%BitXorAssignWith.impl_witness.8b2, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.fe5efd.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.495) [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.0562d9.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.f8190f.1 = specific_constant imports.%Core.Op.d36, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.fe5efd.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.BitXorAssignWith.impl.Op.type.f8190f.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.fe5efd.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.0562d9.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.227 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.731, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.580c5b.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.495) [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.51b64e.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9e13fd.1 = specific_constant imports.%Core.Op.f28, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.580c5b.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.9e13fd.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.580c5b.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.51b64e.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.e73 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.fa6, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.6ebbd9.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.495) [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.d75]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c601.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.2a9012.1 = specific_constant imports.%Core.Op.c63, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.6ebbd9.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.2a9012.1 = name_ref Op, %.loc21_5.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.6ebbd9.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.d75) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c601.2]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e8, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.794: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.09d746.1: %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.09d746.2: %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc39: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc39) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.fe1: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.59e, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c27e6e.1: %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c27e6e.2: %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.794 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.fe1) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.45c: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.847: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.45c, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.bc1181.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.c27e6e.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9f0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.bb8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9f0, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.bc1181.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.c27e6e.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.ab0: type = facet_type <@SubAssignWith, @SubAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.4445c4.1: %Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.4445c4.2: %Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.06c: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.714, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.774eaf.1: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.2, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.ce830d.1: %Cpp.long.as.SubAssignWith.impl.Op.type.774eaf.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.type.774eaf.2: type = fn_type @Cpp.long.as.SubAssignWith.impl.Op.1, @Cpp.long.as.SubAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.ce830d.2: %Cpp.long.as.SubAssignWith.impl.Op.type.774eaf.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.ab0 = facet_value %Cpp.long, (%SubAssignWith.impl_witness.06c) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.b9c: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i32, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.0c9: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.b9c, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.d08c13.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.ce830d.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.d08c13.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.ce830d.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.743: type = facet_type <@MulAssignWith, @MulAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.684acb.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.6a090a.1: %Cpp.long.as.MulAssignWith.impl.Op.type.684acb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.684acb.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.6a090a.2: %Cpp.long.as.MulAssignWith.impl.Op.type.684acb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.eb4: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.71c, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.3a5234.1: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.2, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.848c12.1: %Cpp.long.as.MulAssignWith.impl.Op.type.3a5234.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.type.3a5234.2: type = fn_type @Cpp.long.as.MulAssignWith.impl.Op.1, @Cpp.long.as.MulAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.848c12.2: %Cpp.long.as.MulAssignWith.impl.Op.type.3a5234.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.743 = facet_value %Cpp.long, (%MulAssignWith.impl_witness.eb4) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.3fe: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i32, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.a57: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.3fe, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.646b46.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.848c12.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.646b46.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.848c12.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.786: type = facet_type <@DivAssignWith, @DivAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.baaf86.1: %Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.baaf86.2: %Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.6f8: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.483, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.aba907.1: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.2, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.a2f42c.1: %Cpp.long.as.DivAssignWith.impl.Op.type.aba907.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.type.aba907.2: type = fn_type @Cpp.long.as.DivAssignWith.impl.Op.1, @Cpp.long.as.DivAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.a2f42c.2: %Cpp.long.as.DivAssignWith.impl.Op.type.aba907.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.786 = facet_value %Cpp.long, (%DivAssignWith.impl_witness.6f8) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.3b9: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i32, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.4db: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.3b9, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.2d3f8f.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.a2f42c.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.2d3f8f.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.a2f42c.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.e49: type = facet_type <@ModAssignWith, @ModAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.206fcc.1: %Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.206fcc.2: %Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.b83: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.1fd, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.55a082.1: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.2, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.4c75b0.1: %Cpp.long.as.ModAssignWith.impl.Op.type.55a082.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.type.55a082.2: type = fn_type @Cpp.long.as.ModAssignWith.impl.Op.1, @Cpp.long.as.ModAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.4c75b0.2: %Cpp.long.as.ModAssignWith.impl.Op.type.55a082.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.e49 = facet_value %Cpp.long, (%ModAssignWith.impl_witness.b83) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.18d: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i32, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.115: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.18d, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.98db73.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.4c75b0.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.98db73.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.4c75b0.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.d65: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.0b15fa.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.0b15fa.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.ff6: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.ac5, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.4cf1e9.1: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.2, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.88cf6e.1: %Cpp.long.as.BitAndAssignWith.impl.Op.type.4cf1e9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.type.4cf1e9.2: type = fn_type @Cpp.long.as.BitAndAssignWith.impl.Op.1, @Cpp.long.as.BitAndAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.88cf6e.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.4cf1e9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.d65 = facet_value %Cpp.long, (%BitAndAssignWith.impl_witness.ff6) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.d82: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i32, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.397: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.d82, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.c9da76.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.88cf6e.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.c9da76.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.88cf6e.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.6e9: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.3b2429.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.3b2429.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.a56: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.d80, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.e80b54.1: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.2, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.ff457f.1: %Cpp.long.as.BitOrAssignWith.impl.Op.type.e80b54.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.type.e80b54.2: type = fn_type @Cpp.long.as.BitOrAssignWith.impl.Op.1, @Cpp.long.as.BitOrAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.ff457f.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.e80b54.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.6e9 = facet_value %Cpp.long, (%BitOrAssignWith.impl_witness.a56) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.1b3: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i32, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.ca1: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.1b3, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.8ea2e5.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.ff457f.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.8ea2e5.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.ff457f.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.ea8: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.3c8402.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.3c8402.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.aa9: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.fe4, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.147fea.1: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.2, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.edb545.1: %Cpp.long.as.BitXorAssignWith.impl.Op.type.147fea.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.type.147fea.2: type = fn_type @Cpp.long.as.BitXorAssignWith.impl.Op.1, @Cpp.long.as.BitXorAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.edb545.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.147fea.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.ea8 = facet_value %Cpp.long, (%BitXorAssignWith.impl_witness.aa9) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.330: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i32, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.670: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.330, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.2a5d24.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.edb545.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.2a5d24.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.edb545.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.92b: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.f91cfa.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.f91cfa.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.aec: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.c76, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.97b9eb.1: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.1ecf0e.1: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.97b9eb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.97b9eb.2: type = fn_type @Cpp.long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.1ecf0e.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.97b9eb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.92b = facet_value %Cpp.long, (%LeftShiftAssignWith.impl_witness.aec) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.7be: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i32, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.e8c: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.7be, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.9eb8bf.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.1ecf0e.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.9eb8bf.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.1ecf0e.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.8ae: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.43d513.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.43d513.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.4a7: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.2d4, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.e4a06a.1: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.b505ae.1: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.e4a06a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.type.e4a06a.2: type = fn_type @Cpp.long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.b505ae.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.e4a06a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.8ae = facet_value %Cpp.long, (%RightShiftAssignWith.impl_witness.4a7) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.af4: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i32, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.08a: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.af4, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.8c339b.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.b505ae.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.8c339b.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.b505ae.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c07: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.09d746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.59e = impl_witness_table (%Core.import_ref.c07), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.8bb: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.09d746.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1a6: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.4445c4.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.714 = impl_witness_table (%Core.import_ref.1a6), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.889: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.48a67d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.4445c4.2)]
// CHECK:STDOUT:   %Core.import_ref.ab5: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.684acb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.6a090a.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.71c = impl_witness_table (%Core.import_ref.ab5), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c34: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.684acb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.6a090a.2)]
// CHECK:STDOUT:   %Core.import_ref.585: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.baaf86.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.483 = impl_witness_table (%Core.import_ref.585), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.cd4: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.d25b13.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.baaf86.2)]
// CHECK:STDOUT:   %Core.import_ref.1ef: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.206fcc.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.1fd = impl_witness_table (%Core.import_ref.1ef), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.0d4c: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.4df61f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.206fcc.2)]
// CHECK:STDOUT:   %Core.import_ref.045: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.0b15fa.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.ac5 = impl_witness_table (%Core.import_ref.045), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.984: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.746cd7.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.0b15fa.2)]
// CHECK:STDOUT:   %Core.import_ref.5e2: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.3b2429.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.d80 = impl_witness_table (%Core.import_ref.5e2), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.7ac: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.64aae1.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.3b2429.2)]
// CHECK:STDOUT:   %Core.import_ref.261: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c8402.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.fe4 = impl_witness_table (%Core.import_ref.261), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d36: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.6347f8.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c8402.2)]
// CHECK:STDOUT:   %Core.import_ref.8d3: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.f91cfa.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.c76 = impl_witness_table (%Core.import_ref.8d3), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f28: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.bbb3f3.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.f91cfa.2)]
// CHECK:STDOUT:   %Core.import_ref.559: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.43d513.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.2d4 = impl_witness_table (%Core.import_ref.559), @Cpp.long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c63: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.d1cdba.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.43d513.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
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
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.847 = impl_witness_access constants.%AddAssignWith.impl_witness.fe1, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.c27e6e.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.bc1181.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.1 = specific_constant imports.%Core.Op.8bb, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.c27e6e.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.c27e6e.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = converted %b.ref.loc10, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.bc1181.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref.loc10, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_8: init %Cpp.long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_8
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.4(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5.1: %.0c9 = impl_witness_access constants.%SubAssignWith.impl_witness.06c, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.ce830d.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5.1, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.d08c13.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long.as.SubAssignWith.impl.Op.type.774eaf.1 = specific_constant imports.%Core.Op.889, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.ce830d.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long.as.SubAssignWith.impl.Op.type.774eaf.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.ce830d.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %impl.elem0.loc11_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_5: init %Cpp.long = call %bound_method.loc11_5.3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_5
// CHECK:STDOUT:   %.loc11_5.4: %Cpp.long = converted %b.ref.loc11, %.loc11_5.3
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.d08c13.2]
// CHECK:STDOUT:   %bound_method.loc11_5.4: <bound method> = bound_method %a.ref.loc11, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc11_8: init %Cpp.long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc11_8
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.4(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5.1: %.a57 = impl_witness_access constants.%MulAssignWith.impl_witness.eb4, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.848c12.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5.1, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.646b46.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.3a5234.1 = specific_constant imports.%Core.Op.c34, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.848c12.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long.as.MulAssignWith.impl.Op.type.3a5234.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.848c12.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_5: init %Cpp.long = call %bound_method.loc12_5.3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_5
// CHECK:STDOUT:   %.loc12_5.4: %Cpp.long = converted %b.ref.loc12, %.loc12_5.3
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.646b46.2]
// CHECK:STDOUT:   %bound_method.loc12_5.4: <bound method> = bound_method %a.ref.loc12, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc12_8: init %Cpp.long = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc12_8
// CHECK:STDOUT:   %.loc12_8.2: %Cpp.long = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.4(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5.1: %.4db = impl_witness_access constants.%DivAssignWith.impl_witness.6f8, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.a2f42c.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5.1, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.2d3f8f.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long.as.DivAssignWith.impl.Op.type.aba907.1 = specific_constant imports.%Core.Op.cd4, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.a2f42c.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long.as.DivAssignWith.impl.Op.type.aba907.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.a2f42c.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_5: init %Cpp.long = call %bound_method.loc13_5.3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_5
// CHECK:STDOUT:   %.loc13_5.4: %Cpp.long = converted %b.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.2d3f8f.2]
// CHECK:STDOUT:   %bound_method.loc13_5.4: <bound method> = bound_method %a.ref.loc13, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc13_8: init %Cpp.long = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc13_8
// CHECK:STDOUT:   %.loc13_8.2: %Cpp.long = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.4(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5.1: %.115 = impl_witness_access constants.%ModAssignWith.impl_witness.b83, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.4c75b0.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5.1, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.98db73.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.55a082.1 = specific_constant imports.%Core.Op.0d4c, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.4c75b0.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long.as.ModAssignWith.impl.Op.type.55a082.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.4c75b0.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_5: init %Cpp.long = call %bound_method.loc14_5.3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_5
// CHECK:STDOUT:   %.loc14_5.4: %Cpp.long = converted %b.ref.loc14, %.loc14_5.3
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.98db73.2]
// CHECK:STDOUT:   %bound_method.loc14_5.4: <bound method> = bound_method %a.ref.loc14, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc14_8: init %Cpp.long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc14_8
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.4(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5.1: %.397 = impl_witness_access constants.%BitAndAssignWith.impl_witness.ff6, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.88cf6e.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc16: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc16 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5.1, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.c9da76.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.4cf1e9.1 = specific_constant imports.%Core.Op.984, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.88cf6e.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long.as.BitAndAssignWith.impl.Op.type.4cf1e9.1 = name_ref Op, %.loc16_5.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.88cf6e.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_5: init %Cpp.long = call %bound_method.loc16_5.3(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_5
// CHECK:STDOUT:   %.loc16_5.4: %Cpp.long = converted %b.ref.loc16, %.loc16_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.c9da76.2]
// CHECK:STDOUT:   %bound_method.loc16_5.4: <bound method> = bound_method %a.ref.loc16, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc16_8: init %Cpp.long = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc16_8
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.4(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5.1: %.ca1 = impl_witness_access constants.%BitOrAssignWith.impl_witness.a56, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.ff457f.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5.1, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.8ea2e5.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.e80b54.1 = specific_constant imports.%Core.Op.7ac, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.ff457f.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long.as.BitOrAssignWith.impl.Op.type.e80b54.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.ff457f.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_5: init %Cpp.long = call %bound_method.loc17_5.3(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_5
// CHECK:STDOUT:   %.loc17_5.4: %Cpp.long = converted %b.ref.loc17, %.loc17_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.8ea2e5.2]
// CHECK:STDOUT:   %bound_method.loc17_5.4: <bound method> = bound_method %a.ref.loc17, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc17_8: init %Cpp.long = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc17_8
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.4(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5.1: %.670 = impl_witness_access constants.%BitXorAssignWith.impl_witness.aa9, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.edb545.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5.1, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.2a5d24.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.147fea.1 = specific_constant imports.%Core.Op.d36, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.edb545.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long.as.BitXorAssignWith.impl.Op.type.147fea.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.edb545.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc18_5: init %Cpp.long = call %bound_method.loc18_5.3(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc18_5
// CHECK:STDOUT:   %.loc18_5.4: %Cpp.long = converted %b.ref.loc18, %.loc18_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.2a5d24.2]
// CHECK:STDOUT:   %bound_method.loc18_5.4: <bound method> = bound_method %a.ref.loc18, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc18_8: init %Cpp.long = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc18_8
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.4(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5.1: %.e8c = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.aec, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1ecf0e.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.9eb8bf.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.97b9eb.1 = specific_constant imports.%Core.Op.f28, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1ecf0e.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.97b9eb.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1ecf0e.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc19_5: init %Cpp.long = call %bound_method.loc19_5.3(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc19_5
// CHECK:STDOUT:   %.loc19_5.4: %Cpp.long = converted %b.ref.loc19, %.loc19_5.3
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.9eb8bf.2]
// CHECK:STDOUT:   %bound_method.loc19_5.4: <bound method> = bound_method %a.ref.loc19, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc19_9: init %Cpp.long = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc19_9
// CHECK:STDOUT:   %.loc19_9.2: %Cpp.long = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.4(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5.1: %.08a = impl_witness_access constants.%RightShiftAssignWith.impl_witness.4a7, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.b505ae.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc39) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.8c339b.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.e4a06a.1 = specific_constant imports.%Core.Op.c63, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.b505ae.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.e4a06a.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.b505ae.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc20_5: init %Cpp.long = call %bound_method.loc20_5.3(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc20_5
// CHECK:STDOUT:   %.loc20_5.4: %Cpp.long = converted %b.ref.loc20, %.loc20_5.3
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.8c339b.2]
// CHECK:STDOUT:   %bound_method.loc20_5.4: <bound method> = bound_method %a.ref.loc20, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc39, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.761: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.09d746.1: %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.09d746.2: %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.49d: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.59e, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.793d6a.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.92c8f4.1: %Cpp.long.as.AddAssignWith.impl.Op.type.793d6a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.793d6a.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.92c8f4.2: %Cpp.long.as.AddAssignWith.impl.Op.type.793d6a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.761 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.49d) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.298: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.c39: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.298, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.7e6d5c.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.92c8f4.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.7e6d5c.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.92c8f4.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c07: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.09d746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.59e = impl_witness_table (%Core.import_ref.c07), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.8bb: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.09d746.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9_5.1: %.c39 = impl_witness_access constants.%AddAssignWith.impl_witness.49d, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.92c8f4.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.7e6d5c.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.793d6a.1 = specific_constant imports.%Core.Op.8bb, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.92c8f4.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.793d6a.1 = name_ref Op, %.loc9_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.92c8f4.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc9_5.2: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5: init %Cpp.long = call %bound_method.loc9_5.3(%int_1.loc9) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc9_5.4: %Cpp.long = converted %int_1.loc9, %.loc9_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.7e6d5c.2]
// CHECK:STDOUT:   %bound_method.loc9_5.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eed: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e8, %ImplicitAs.facet.eed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b2d(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.b94) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.794: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %U.57d: %ImplicitAs.type.819 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.09d746.1: %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%U.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.09d746.2: %Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0fc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ad [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.174: %ImplicitAs.type.819 = facet_value %i32, (%ImplicitAs.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.fe1: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.59e, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.1: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.2, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c27e6e.1: %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.2: type = fn_type @Cpp.long.as.AddAssignWith.impl.Op.1, @Cpp.long.as.AddAssignWith.impl(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.c27e6e.2: %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.794 = facet_value %Cpp.long, (%AddAssignWith.impl_witness.fe1) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.45c: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.847: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.45c, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.bc1181.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.c27e6e.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9f0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.bb8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9f0, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.bc1181.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.c27e6e.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.c07: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.09d746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.59e = impl_witness_table (%Core.import_ref.c07), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.8bb: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.1aeb96.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.09d746.2)]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
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
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc9: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_16.2(%int_1.loc9) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_16.2: %i32 = converted %int_1.loc9, %.loc9_16.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %b: %i32 = value_binding b, %.loc9_16.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5.1: %.847 = impl_witness_access constants.%AddAssignWith.impl_witness.fe1, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.c27e6e.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.bc1181.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.1 = specific_constant imports.%Core.Op.8bb, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.c27e6e.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.7007c6.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.c27e6e.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc10_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref, %impl.elem0.loc10_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc10_5: init %Cpp.long = call %bound_method.loc10_5.3(%b.ref)
// CHECK:STDOUT:   %.loc10_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc10_5
// CHECK:STDOUT:   %.loc10_5.4: %Cpp.long = converted %b.ref, %.loc10_5.3
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.bc1181.2]
// CHECK:STDOUT:   %bound_method.loc10_5.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.903 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.5a4: %Cpp.long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.633: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.314: type = fn_type_with_self_type %Inc.WithSelf.Op.type.633, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.type: type = fn_type @Cpp.long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op: %Cpp.long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.b87: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.253: type = fn_type_with_self_type %Dec.WithSelf.Op.type.b87, %Dec.facet [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.314 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.253 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long.as.Dec.impl.Op]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a19: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5b5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.66c: %ImplicitAs.type.3c3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a19) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b7d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet.66c) [concrete]
// CHECK:STDOUT:   %.541: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b7d, %ImplicitAs.facet.66c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.71f: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.14b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.71f = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.92f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.14b [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.c85(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.46e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.29e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.899, @Core.IntLiteral.as.ImplicitAs.impl.c85(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b07: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.c85(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.90e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b07 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.309: %ImplicitAs.type.146 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.29e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a88: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u32, %ImplicitAs.facet.309) [concrete]
// CHECK:STDOUT:   %.b47: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a88, %ImplicitAs.facet.309 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.412: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.90e [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.90e, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.21d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %As.impl_witness.d5f: <witness> = impl_witness imports.%As.impl_witness_table.f58 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.950 = facet_value Core.IntLiteral, (%As.impl_witness.d5f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.cab: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.59b: type = fn_type_with_self_type %As.WithSelf.Convert.type.cab, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a13: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.50d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2f7: %ImplicitAs.type.139 = facet_value %Cpp.unsigned_long, (%ImplicitAs.impl_witness.a13) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.328: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.2f7) [concrete]
// CHECK:STDOUT:   %.0db: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.328, %ImplicitAs.facet.2f7 [concrete]
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
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e4d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.97e, %ImplicitAs.facet.945) [concrete]
// CHECK:STDOUT:   %.98d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e4d, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.57d: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.741: @Core.IntLiteral.as.ImplicitAs.impl.c85.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.c85.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.46e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.899 = impl_witness_table (%Core.import_ref.741), @Core.IntLiteral.as.ImplicitAs.impl.c85 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.f5e: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc15: %.541 = impl_witness_access constants.%ImplicitAs.impl_witness.a19, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.14b]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.92f]
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
// CHECK:STDOUT:   %impl.elem0.loc21: %.b47 = impl_witness_access constants.%ImplicitAs.impl_witness.29e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.90e]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.412]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.21d]
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
// CHECK:STDOUT:   %impl.elem0.loc24_59: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc24_59.1: <bound method> = bound_method %float, %impl.elem0.loc24_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_59.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.57d]
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
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.59b = impl_witness_access constants.%As.impl_witness.d5f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.0db = impl_witness_access constants.%ImplicitAs.impl_witness.a13, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a19: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5b5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.3c3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a19) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b7d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.541: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b7d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.0ab: %Cpp.unsigned_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.ae3: <witness> = impl_witness imports.%Copy.impl_witness_table.36c [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long, (%Copy.impl_witness.ae3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.973: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.f45: type = fn_type_with_self_type %Copy.WithSelf.Op.type.973, %Copy.facet [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.541 = impl_witness_access constants.%ImplicitAs.impl_witness.a19, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %impl.elem0.loc9: %.f45 = impl_witness_access constants.%Copy.impl_witness.ae3, element0 [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
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
