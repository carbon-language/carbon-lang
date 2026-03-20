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
  // CHECK:STDERR:       ^~~~~~~~~~
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = @F.%i64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc141, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.0bc: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2ed) = import_ref Core//prelude/types/int, loc180, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.d29)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ea2 = impl_witness_table (%Core.import_ref.0bc), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/types/int, loc219, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/types/float, loc145, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31a = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.95b = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc16: type = splice_block %long_long.ref.loc16 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long_long.ref.loc16: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc19: %.a64 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc19.1: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.288]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc19.2(%int_1.loc19) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc19.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc19.2: %i64 = converted %int_1.loc19, %.loc19.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %cpp_long_long: %i64 = value_binding cpp_long_long, %.loc19.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_long.patt: %pattern_type.95b = value_binding_pattern carbon_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref: %i64 = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %i64.loc25: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long_long: %i64 = value_binding carbon_long_long, %cpp_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc47.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc47: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc47.1: <bound method> = bound_method %float, %impl.elem0.loc47 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc47: <specific function> = specific_function %impl.elem0.loc47, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc47.2: <bound method> = bound_method %float, %specific_fn.loc47 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc47.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc47.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc47.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc47.4: init %f32.97e to %.loc47.3 = in_place_init %.loc47.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc47.5: init %array_type to %a.var = array_init (%.loc47.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc29: init %array_type = converted %.loc47.1, %.loc47.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc29
// CHECK:STDOUT:   %.loc42: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc37: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc39: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc40: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc38.1: %.fc7 = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:     %bound_method.loc38.1: <bound method> = bound_method %int_1.loc37, %impl.elem0.loc38.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc38.1: <specific function> = specific_function %impl.elem0.loc38.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc38.2: <bound method> = bound_method %int_1.loc37, %specific_fn.loc38.1 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc38.2(%int_1.loc37) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc38.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc38.2: %i64 = converted %int_1.loc37, %.loc38.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %impl.elem0.loc38.2: %.495 = impl_witness_access constants.%ImplicitAs.impl_witness.e86, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.535]
// CHECK:STDOUT:     %bound_method.loc38.3: <bound method> = bound_method %.loc38.2, %impl.elem0.loc38.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc38.2: <specific function> = specific_function %impl.elem0.loc38.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc38.4: <bound method> = bound_method %.loc38.2, %specific_fn.loc38.2 [concrete = constants.%bound_method.d1e]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc38.4(%.loc38.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc38.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc38.4: Core.IntLiteral = converted %.loc38.2, %.loc38.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc38.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type) = "no_op";
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long_long = @F.%u64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.741: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6) = import_ref Core//prelude/types/uint, loc145, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.46e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.899 = impl_witness_table (%Core.import_ref.741), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.483: @UInt.as.ImplicitAs.impl.607.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.7f8) = import_ref Core//prelude/types/uint, loc184, loaded [symbolic = @UInt.as.ImplicitAs.impl.607.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.0ea)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.493 = impl_witness_table (%Core.import_ref.483), @UInt.as.ImplicitAs.impl.607 [concrete]
// CHECK:STDOUT:   %Core.import_ref.600: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.2b1) = import_ref Core//prelude/types/uint, loc223, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.979)]
// CHECK:STDOUT:   %As.impl_witness_table.7eb = impl_witness_table (%Core.import_ref.600), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/types/float, loc145, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31a = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.157 = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc16: type = splice_block %unsigned_long_long.ref.loc16 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long_long.ref.loc16: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc19: %.f17 = impl_witness_access constants.%ImplicitAs.impl_witness.bb3, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.042]
// CHECK:STDOUT:   %bound_method.loc19.1: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.fec]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc19.2(%int_1.loc19) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc19.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc19.2: %u64 = converted %int_1.loc19, %.loc19.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = value_binding cpp_unsigned_long_long, %.loc19.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long_long.patt: %pattern_type.157 = value_binding_pattern carbon_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref: %u64 = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %u64.loc25: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long_long: %u64 = value_binding carbon_unsigned_long_long, %cpp_unsigned_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc47.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc47: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc47.1: <bound method> = bound_method %float, %impl.elem0.loc47 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc47: <specific function> = specific_function %impl.elem0.loc47, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc47.2: <bound method> = bound_method %float, %specific_fn.loc47 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc47.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc47.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc47.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc47.4: init %f32.97e to %.loc47.3 = in_place_init %.loc47.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc47.5: init %array_type to %a.var = array_init (%.loc47.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc29: init %array_type = converted %.loc47.1, %.loc47.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc29
// CHECK:STDOUT:   %.loc42: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc37: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc39: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc40: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:     %impl.elem0.loc38.1: %.904 = impl_witness_access constants.%As.impl_witness.f54, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.715]
// CHECK:STDOUT:     %bound_method.loc38.1: <bound method> = bound_method %int_1.loc37, %impl.elem0.loc38.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc38.1: <specific function> = specific_function %impl.elem0.loc38.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc38.2: <bound method> = bound_method %int_1.loc37, %specific_fn.loc38.1 [concrete = constants.%bound_method.9cf]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc38.2(%int_1.loc37) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc38.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc38.2: %u64 = converted %int_1.loc37, %.loc38.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %impl.elem0.loc38.2: %.74a = impl_witness_access constants.%ImplicitAs.impl_witness.2cc, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.309]
// CHECK:STDOUT:     %bound_method.loc38.3: <bound method> = bound_method %.loc38.2, %impl.elem0.loc38.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc38.2: <specific function> = specific_function %impl.elem0.loc38.2, @UInt.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc38.4: <bound method> = bound_method %.loc38.2, %specific_fn.loc38.2 [concrete = constants.%bound_method.f18]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc38.4(%.loc38.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc38.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc38.4: Core.IntLiteral = converted %.loc38.2, %.loc38.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc38.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type) = "no_op";
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc214, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
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
// CHECK:STDOUT:   %Core.import_ref.297: %Cpp.long.as.ImplicitAs.impl.Convert.type.431 = import_ref Core//prelude/types/cpp/int, loc326, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bbf]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cb1 = impl_witness_table (%Core.import_ref.297), @Cpp.long.as.ImplicitAs.impl.e6d [concrete]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.686: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc243, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.797 = impl_witness_table (%Core.import_ref.686), @Core.IntLiteral.as.As.impl.a00 [concrete]
// CHECK:STDOUT:   %Core.import_ref.168: %Cpp.long.as.ImplicitAs.impl.Convert.type.9ed = import_ref Core//prelude/types/cpp/int, loc273, loaded [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.d49]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a09 = impl_witness_table (%Core.import_ref.168), @Cpp.long.as.ImplicitAs.impl.e94 [concrete]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/types/float, loc145, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31a = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.68c = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc18: type = splice_block %long.ref.loc18 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc18: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc21: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc21(%int_1.loc21) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21.2: %Cpp.long = converted %int_1.loc21, %.loc21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %cpp_long: %Cpp.long = value_binding cpp_long, %.loc21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_result.patt: %pattern_type.cd9 = value_binding_pattern cpp_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc32: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long.ref.loc35: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc36.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc36: %ptr.305 = addr_of %.loc36.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc36: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.1(%cpp_long.ref.loc35, %addr.loc36)
// CHECK:STDOUT:   %.loc36.2: init %LongResult to %.loc36.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc36
// CHECK:STDOUT:   %.loc28: type = splice_block %LongResult.ref.loc28 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc28: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc36.3: ref %LongResult = temporary %.loc36.1, %.loc36.2
// CHECK:STDOUT:   %.loc36.4: %LongResult = acquire_value %.loc36.3
// CHECK:STDOUT:   %cpp_long_result: %LongResult = value_binding cpp_long_result, %.loc36.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long.patt: %pattern_type.68c = value_binding_pattern cpp_compat_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc47: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc44: type = splice_block %Long32.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.4b4 [concrete = imports.%CppCompat.4b4]
// CHECK:STDOUT:     %Long32.ref: type = name_ref Long32, imports.%Core.Long32 [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long: %Cpp.long = value_binding cpp_compat_long, %cpp_long.ref.loc47
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_result.patt: %pattern_type.cd9 = value_binding_pattern cpp_compat_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc58: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long.ref: %Cpp.long = name_ref cpp_compat_long, %cpp_compat_long
// CHECK:STDOUT:   %.loc62.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc62: %ptr.305 = addr_of %.loc62.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc62: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.1(%cpp_compat_long.ref, %addr.loc62)
// CHECK:STDOUT:   %.loc62.2: init %LongResult to %.loc62.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc62
// CHECK:STDOUT:   %.loc54: type = splice_block %LongResult.ref.loc54 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc53: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc54: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc62.3: ref %LongResult = temporary %.loc62.1, %.loc62.2
// CHECK:STDOUT:   %.loc62.4: %LongResult = acquire_value %.loc62.3
// CHECK:STDOUT:   %cpp_compat_long_result: %LongResult = value_binding cpp_compat_long_result, %.loc62.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32.patt: %pattern_type.7ce = value_binding_pattern carbon_i32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc69: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i32.loc71: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc70.1: %i32 = as_compatible %cpp_long.ref.loc69
// CHECK:STDOUT:   %.loc70.2: %i32 = converted %cpp_long.ref.loc69, %.loc70.1
// CHECK:STDOUT:   %i32.loc67: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %carbon_i32: %i32 = value_binding carbon_i32, %.loc70.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32_result.patt: %pattern_type.454 = value_binding_pattern carbon_i32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc81: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc82: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i32.ref: %i32 = name_ref carbon_i32, %carbon_i32
// CHECK:STDOUT:   %.loc86.1: ref %IntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc86: %ptr.3cb = addr_of %.loc86.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc86: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.2(%carbon_i32.ref, %addr.loc86)
// CHECK:STDOUT:   %.loc86.2: init %IntResult to %.loc86.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc86
// CHECK:STDOUT:   %.loc78: type = splice_block %IntResult.ref [concrete = constants.%IntResult] {
// CHECK:STDOUT:     %Cpp.ref.loc77: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntResult.ref: type = name_ref IntResult, imports.%IntResult.decl [concrete = constants.%IntResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc86.3: ref %IntResult = temporary %.loc86.1, %.loc86.2
// CHECK:STDOUT:   %.loc86.4: %IntResult = acquire_value %.loc86.3
// CHECK:STDOUT:   %carbon_i32_result: %IntResult = value_binding carbon_i32_result, %.loc86.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.95b = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc94: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc94: %.12d = impl_witness_access constants.%ImplicitAs.impl_witness.bbc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bbf]
// CHECK:STDOUT:   %bound_method.loc94: <bound method> = bound_method %cpp_long.ref.loc94, %impl.elem0.loc94
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc94: init %i64 = call %bound_method.loc94(%cpp_long.ref.loc94)
// CHECK:STDOUT:   %.loc94.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc94
// CHECK:STDOUT:   %.loc94.2: %i64 = converted %cpp_long.ref.loc94, %.loc94.1
// CHECK:STDOUT:   %carbon_i64: %i64 = value_binding carbon_i64, %.loc94.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc114.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc114: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc114.1: <bound method> = bound_method %float, %impl.elem0.loc114 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc114, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc114.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc114.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc114.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc114.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc114.4: init %f32.97e to %.loc114.3 = in_place_init %.loc114.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc114.5: init %array_type to %a.var = array_init (%.loc114.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc96: init %array_type = converted %.loc114.1, %.loc114.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc96
// CHECK:STDOUT:   %.loc109: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc104: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc106: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc107: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:     %impl.elem0.loc105.1: %.a3d = impl_witness_access constants.%As.impl_witness.60c, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc105.1: <bound method> = bound_method %int_1.loc104, %impl.elem0.loc105.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long = call %bound_method.loc105.1(%int_1.loc104) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %.loc105.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %.loc105.2: %Cpp.long = converted %int_1.loc104, %.loc105.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %impl.elem0.loc105.2: %.7f4 = impl_witness_access constants.%ImplicitAs.impl_witness.e4d, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.d49]
// CHECK:STDOUT:     %bound_method.loc105.2: <bound method> = bound_method %.loc105.2, %impl.elem0.loc105.2 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long.as.ImplicitAs.impl.Convert.call.loc105: init Core.IntLiteral = call %bound_method.loc105.2(%.loc105.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc105.3: Core.IntLiteral = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc105 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc105.4: Core.IntLiteral = converted %.loc105.2, %.loc105.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc105.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   %IntResult.cpp_destructor.bound: <bound method> = bound_method %.loc86.3, constants.%IntResult.cpp_destructor
// CHECK:STDOUT:   %IntResult.cpp_destructor.call: init %empty_tuple.type = call %IntResult.cpp_destructor.bound(%.loc86.3)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound.loc62: <bound method> = bound_method %.loc62.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call.loc62: init %empty_tuple.type = call %LongResult.cpp_destructor.bound.loc62(%.loc62.3)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound.loc36: <bound method> = bound_method %.loc36.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call.loc36: init %empty_tuple.type = call %LongResult.cpp_destructor.bound.loc36(%.loc36.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long: type = class_type @Long32 [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc214, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.915: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc119, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %impl.elem0.loc12: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %int_1, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc12(%int_1) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc12: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc12
// CHECK:STDOUT:   %.loc16: type = splice_block %long.ref.loc16 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc16: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.68c = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %.loc29: %Cpp.long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc29: %.539 = impl_witness_access constants.%Copy.impl_witness.e75, element0 [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc29: <bound method> = bound_method %.loc29, %impl.elem0.loc29
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.call: init %Cpp.long = call %bound_method.loc29(%.loc29)
// CHECK:STDOUT:   assign %b.var, %Cpp.long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc26: type = splice_block %long.ref.loc26 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc26: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc21: <bound method> = bound_method %b.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc21: init %empty_tuple.type = call %Destroy.Op.bound.loc21(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc12: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc12: init %empty_tuple.type = call %Destroy.Op.bound.loc12(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long) = "no_op";
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
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc214, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.869: %Cpp.long.as.EqWith.impl.Equal.type.cc0 = import_ref Core//prelude/types/cpp/int, loc780, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1b7]
// CHECK:STDOUT:   %Core.import_ref.f17: %Cpp.long.as.EqWith.impl.NotEqual.type.75f = import_ref Core//prelude/types/cpp/int, loc797, loaded [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.5de]
// CHECK:STDOUT:   %EqWith.impl_witness_table.322 = impl_witness_table (%Core.import_ref.869, %Core.import_ref.f17), @Cpp.long.as.EqWith.impl.7c8 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9ae: %Cpp.long.as.OrderedWith.impl.Less.type.d1e = import_ref Core//prelude/types/cpp/int, loc828, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.911]
// CHECK:STDOUT:   %Core.import_ref.18a: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.cd7 = import_ref Core//prelude/types/cpp/int, loc845, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.187]
// CHECK:STDOUT:   %Core.import_ref.8df: %Cpp.long.as.OrderedWith.impl.Greater.type.da5 = import_ref Core//prelude/types/cpp/int, loc862, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.cc7]
// CHECK:STDOUT:   %Core.import_ref.deb: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.9c8 = import_ref Core//prelude/types/cpp/int, loc879, loaded [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.337]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.e82 = impl_witness_table (%Core.import_ref.9ae, %Core.import_ref.18a, %Core.import_ref.8df, %Core.import_ref.deb), @Cpp.long.as.OrderedWith.impl.361 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc16: type = splice_block %long.ref.loc16 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc16: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc19: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19.2: %Cpp.long = converted %int_1.loc19, %.loc19.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc19.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc28: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc25: type = splice_block %long.ref.loc25 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc25: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc28: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28: <bound method> = bound_method %int_1.loc28, %impl.elem0.loc28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28: init %Cpp.long = call %bound_method.loc28(%int_1.loc28) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc28.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc28.2: %Cpp.long = converted %int_1.loc28, %.loc28.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc28.2
// CHECK:STDOUT:   %a.ref.loc30: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc32: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc31: %.909 = impl_witness_access constants.%EqWith.impl_witness.6db, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1b7]
// CHECK:STDOUT:   %bound_method.loc31: <bound method> = bound_method %a.ref.loc30, %impl.elem0.loc31
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc31(%a.ref.loc30, %b.ref.loc32)
// CHECK:STDOUT:   %a.ref.loc34: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc36: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc35: %.163 = impl_witness_access constants.%EqWith.impl_witness.6db, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.5de]
// CHECK:STDOUT:   %bound_method.loc35: <bound method> = bound_method %a.ref.loc34, %impl.elem1.loc35
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc35(%a.ref.loc34, %b.ref.loc36)
// CHECK:STDOUT:   %a.ref.loc38: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc40: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.fdc = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.cc7]
// CHECK:STDOUT:   %bound_method.loc39: <bound method> = bound_method %a.ref.loc38, %impl.elem2
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc39(%a.ref.loc38, %b.ref.loc40)
// CHECK:STDOUT:   %a.ref.loc42: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc44: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc43: %.c90 = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.911]
// CHECK:STDOUT:   %bound_method.loc43: <bound method> = bound_method %a.ref.loc42, %impl.elem0.loc43
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc43(%a.ref.loc42, %b.ref.loc44)
// CHECK:STDOUT:   %a.ref.loc46: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc48: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.a9b = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.337]
// CHECK:STDOUT:   %bound_method.loc47: <bound method> = bound_method %a.ref.loc46, %impl.elem3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc47(%a.ref.loc46, %b.ref.loc48)
// CHECK:STDOUT:   %a.ref.loc50: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc52: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc51: %.fad = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.187]
// CHECK:STDOUT:   %bound_method.loc51: <bound method> = bound_method %a.ref.loc50, %impl.elem1.loc51
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc51(%a.ref.loc50, %b.ref.loc52)
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
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.1: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.1, @Cpp.long.as.EqWith.impl.f13(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.c90ef2.1: %Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.2: type = fn_type @Cpp.long.as.EqWith.impl.NotEqual.2, @Cpp.long.as.EqWith.impl.f13(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.c90ef2.2: %Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.7da93b.1: type = fn_type @Cpp.long.as.EqWith.impl.Equal.1, @Cpp.long.as.EqWith.impl.f13(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.c91b06.1: %Cpp.long.as.EqWith.impl.Equal.type.7da93b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.type.7da93b.2: type = fn_type @Cpp.long.as.EqWith.impl.Equal.2, @Cpp.long.as.EqWith.impl.f13(%T.57d) [symbolic]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.c91b06.2: %Cpp.long.as.EqWith.impl.Equal.type.7da93b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.ded: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
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
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.366: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i32, %EqWith.facet.fd1) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.32d: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i32, %EqWith.facet.fd1) [concrete]
// CHECK:STDOUT:   %.09a: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.366, %EqWith.facet.fd1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.0f3e3b.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9f0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.bb8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9f0, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.0f3e3b.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.1c7: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.32d, %EqWith.facet.fd1 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.6a28a6.2, @Cpp.long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.NotEqual.6a28a6.1, @Cpp.long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.243: type = facet_type <@OrderedWith, @OrderedWith(%i32)> [concrete]
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
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.280: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.19d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.c5d: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.19d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.2e4: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.19d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.353: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i32, %OrderedWith.facet.19d) [concrete]
// CHECK:STDOUT:   %.883: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.2e4, %OrderedWith.facet.19d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.9898ff.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.9898ff.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.481: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.280, %OrderedWith.facet.19d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.5299ce.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.5299ce.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.753: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.353, %OrderedWith.facet.19d [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.14a: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.c5d, %OrderedWith.facet.19d [concrete]
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
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.d3d: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.3d4) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.510: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.3d4) [concrete]
// CHECK:STDOUT:   %.a03: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.d3d, %EqWith.facet.3d4 [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.334ae9.1: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.6ed862.2, @Cpp.long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.334ae9.2: <specific function> = specific_function %Cpp.long.as.EqWith.impl.Equal.6ed862.1, @Cpp.long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.9a1: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.510, %EqWith.facet.3d4 [concrete]
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
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.af4: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.eef) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.9ec: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.eef) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.327: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.eef) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.d4c: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.eef) [concrete]
// CHECK:STDOUT:   %.5af: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.327, %OrderedWith.facet.eef [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.b2149b.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.74a807.2, @Cpp.long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.b2149b.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Greater.74a807.1, @Cpp.long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.6b7: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.af4, %OrderedWith.facet.eef [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.b6fc5e.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.0de767.2, @Cpp.long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.b6fc5e.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.Less.0de767.1, @Cpp.long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.1f9: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.d4c, %OrderedWith.facet.eef [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f90679.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.2, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f90679.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.1, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %.76b: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.9ec, %OrderedWith.facet.eef [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.c5c24a.1: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.2, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.c5c24a.2: <specific function> = specific_function %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.1, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc214, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.489: @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.Equal.type.2 (%Cpp.long.as.EqWith.impl.Equal.type.7da93b.1) = import_ref Core//prelude/types/cpp/int, loc1048, loaded [symbolic = @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.Equal.2 (constants.%Cpp.long.as.EqWith.impl.Equal.c91b06.1)]
// CHECK:STDOUT:   %Core.import_ref.a67: @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.1) = import_ref Core//prelude/types/cpp/int, loc1065, loaded [symbolic = @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long.as.EqWith.impl.NotEqual.c90ef2.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.44b = impl_witness_table (%Core.import_ref.489, %Core.import_ref.a67), @Cpp.long.as.EqWith.impl.f13 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.b1f: @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long.as.EqWith.impl.NotEqual.type.b3148d.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long.as.EqWith.impl.NotEqual.c90ef2.2)]
// CHECK:STDOUT:   %Core.Equal.0ad: @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.Equal.type.1 (%Cpp.long.as.EqWith.impl.Equal.type.7da93b.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long.as.EqWith.impl.f13.%Cpp.long.as.EqWith.impl.Equal.1 (constants.%Cpp.long.as.EqWith.impl.Equal.c91b06.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc300, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.959: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Less.type.2 (%Cpp.long.as.OrderedWith.impl.Less.type.a59daa.1) = import_ref Core//prelude/types/cpp/int, loc1104, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long.as.OrderedWith.impl.Less.954f71.1)]
// CHECK:STDOUT:   %Core.import_ref.c23: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4aa4c3.1) = import_ref Core//prelude/types/cpp/int, loc1121, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.7e103a.1)]
// CHECK:STDOUT:   %Core.import_ref.df7: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long.as.OrderedWith.impl.Greater.type.662b24.1) = import_ref Core//prelude/types/cpp/int, loc1138, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long.as.OrderedWith.impl.Greater.02ce94.1)]
// CHECK:STDOUT:   %Core.import_ref.a61: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.74811b.1) = import_ref Core//prelude/types/cpp/int, loc1155, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.dd28ba.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.9c6 = impl_witness_table (%Core.import_ref.959, %Core.import_ref.c23, %Core.import_ref.df7, %Core.import_ref.a61), @Cpp.long.as.OrderedWith.impl.15a [concrete]
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
// CHECK:STDOUT:   %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc23: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc26: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %int_1.loc26, %impl.elem0.loc26 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26: init %Cpp.long = call %bound_method.loc26(%int_1.loc26) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc26.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc26.2: %Cpp.long = converted %int_1.loc26, %.loc26.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc26.2
// CHECK:STDOUT:   %a.ref.loc28: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc30: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc29.1: %.09a = impl_witness_access constants.%EqWith.impl_witness.15a, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.2]
// CHECK:STDOUT:   %bound_method.loc29.1: <bound method> = bound_method %a.ref.loc28, %impl.elem0.loc29.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc29: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc29.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc29 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.1]
// CHECK:STDOUT:   %bound_method.loc29.2: <bound method> = bound_method %a.ref.loc28, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29.2: %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.1 = specific_constant imports.%Core.Equal.0ad, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.1]
// CHECK:STDOUT:   %Equal.ref.loc29: %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.1 = name_ref Equal, %.loc29.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc29: <bound method> = bound_method %a.ref.loc28, %Equal.ref.loc29
// CHECK:STDOUT:   %impl.elem0.loc29.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29.3: <bound method> = bound_method %b.ref.loc30, %impl.elem0.loc29.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29: init %Cpp.long = call %bound_method.loc29.3(%b.ref.loc30)
// CHECK:STDOUT:   %.loc29.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29
// CHECK:STDOUT:   %.loc29.4: %Cpp.long = converted %b.ref.loc30, %.loc29.3
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc29: <specific function> = specific_function %Equal.ref.loc29, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.2]
// CHECK:STDOUT:   %bound_method.loc29.4: <bound method> = bound_method %a.ref.loc28, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc29
// CHECK:STDOUT:   %impl.elem0.loc30: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc30: <bound method> = bound_method %b.ref.loc30, %impl.elem0.loc30
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc30: init %Cpp.long = call %bound_method.loc30(%b.ref.loc30)
// CHECK:STDOUT:   %.loc30.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc30
// CHECK:STDOUT:   %.loc30.2: %Cpp.long = converted %b.ref.loc30, %.loc30.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc29: init bool = call %bound_method.loc29.4(%a.ref.loc28, %.loc30.2)
// CHECK:STDOUT:   %a.ref.loc32: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc34: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc33: %.1c7 = impl_witness_access constants.%EqWith.impl_witness.15a, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.2]
// CHECK:STDOUT:   %bound_method.loc33.1: <bound method> = bound_method %a.ref.loc32, %impl.elem1.loc33
// CHECK:STDOUT:   %ImplicitAs.facet.loc33: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc33.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc33 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc33: <specific function> = specific_function %impl.elem1.loc33, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.1]
// CHECK:STDOUT:   %bound_method.loc33.2: <bound method> = bound_method %a.ref.loc32, %specific_fn.loc33
// CHECK:STDOUT:   %.loc33.2: %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.1 = specific_constant imports.%Core.NotEqual.b1f, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.1]
// CHECK:STDOUT:   %NotEqual.ref.loc33: %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.1 = name_ref NotEqual, %.loc33.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc33: <bound method> = bound_method %a.ref.loc32, %NotEqual.ref.loc33
// CHECK:STDOUT:   %impl.elem0.loc33: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc33.3: <bound method> = bound_method %b.ref.loc34, %impl.elem0.loc33
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc33: init %Cpp.long = call %bound_method.loc33.3(%b.ref.loc34)
// CHECK:STDOUT:   %.loc33.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc33
// CHECK:STDOUT:   %.loc33.4: %Cpp.long = converted %b.ref.loc34, %.loc33.3
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc33: <specific function> = specific_function %NotEqual.ref.loc33, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.2]
// CHECK:STDOUT:   %bound_method.loc33.4: <bound method> = bound_method %a.ref.loc32, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc33
// CHECK:STDOUT:   %impl.elem0.loc34: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc34: <bound method> = bound_method %b.ref.loc34, %impl.elem0.loc34
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc34: init %Cpp.long = call %bound_method.loc34(%b.ref.loc34)
// CHECK:STDOUT:   %.loc34.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc34
// CHECK:STDOUT:   %.loc34.2: %Cpp.long = converted %b.ref.loc34, %.loc34.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc33: init bool = call %bound_method.loc33.4(%a.ref.loc32, %.loc34.2)
// CHECK:STDOUT:   %a.ref.loc36: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc38: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc37: %.883 = impl_witness_access constants.%OrderedWith.impl_witness.576, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.2]
// CHECK:STDOUT:   %bound_method.loc37.1: <bound method> = bound_method %a.ref.loc36, %impl.elem2.loc37
// CHECK:STDOUT:   %ImplicitAs.facet.loc37: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc37.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc37 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc37: <specific function> = specific_function %impl.elem2.loc37, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.1]
// CHECK:STDOUT:   %bound_method.loc37.2: <bound method> = bound_method %a.ref.loc36, %specific_fn.loc37
// CHECK:STDOUT:   %.loc37.2: %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.1 = specific_constant imports.%Core.Greater.9da, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.1]
// CHECK:STDOUT:   %Greater.ref.loc37: %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.1 = name_ref Greater, %.loc37.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc37: <bound method> = bound_method %a.ref.loc36, %Greater.ref.loc37
// CHECK:STDOUT:   %impl.elem0.loc37: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc37.3: <bound method> = bound_method %b.ref.loc38, %impl.elem0.loc37
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc37: init %Cpp.long = call %bound_method.loc37.3(%b.ref.loc38)
// CHECK:STDOUT:   %.loc37.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc37
// CHECK:STDOUT:   %.loc37.4: %Cpp.long = converted %b.ref.loc38, %.loc37.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc37: <specific function> = specific_function %Greater.ref.loc37, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.2]
// CHECK:STDOUT:   %bound_method.loc37.4: <bound method> = bound_method %a.ref.loc36, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc37
// CHECK:STDOUT:   %impl.elem0.loc38: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc38: <bound method> = bound_method %b.ref.loc38, %impl.elem0.loc38
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc38: init %Cpp.long = call %bound_method.loc38(%b.ref.loc38)
// CHECK:STDOUT:   %.loc38.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc38
// CHECK:STDOUT:   %.loc38.2: %Cpp.long = converted %b.ref.loc38, %.loc38.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc37: init bool = call %bound_method.loc37.4(%a.ref.loc36, %.loc38.2)
// CHECK:STDOUT:   %a.ref.loc40: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc42: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc41.1: %.481 = impl_witness_access constants.%OrderedWith.impl_witness.576, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.2]
// CHECK:STDOUT:   %bound_method.loc41.1: <bound method> = bound_method %a.ref.loc40, %impl.elem0.loc41.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc41: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc41.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc41 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc41: <specific function> = specific_function %impl.elem0.loc41.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.1]
// CHECK:STDOUT:   %bound_method.loc41.2: <bound method> = bound_method %a.ref.loc40, %specific_fn.loc41
// CHECK:STDOUT:   %.loc41.2: %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.1 = specific_constant imports.%Core.Less.c86, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.1]
// CHECK:STDOUT:   %Less.ref.loc41: %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.1 = name_ref Less, %.loc41.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc41: <bound method> = bound_method %a.ref.loc40, %Less.ref.loc41
// CHECK:STDOUT:   %impl.elem0.loc41.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc41.3: <bound method> = bound_method %b.ref.loc42, %impl.elem0.loc41.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc41: init %Cpp.long = call %bound_method.loc41.3(%b.ref.loc42)
// CHECK:STDOUT:   %.loc41.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc41
// CHECK:STDOUT:   %.loc41.4: %Cpp.long = converted %b.ref.loc42, %.loc41.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc41: <specific function> = specific_function %Less.ref.loc41, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.2]
// CHECK:STDOUT:   %bound_method.loc41.4: <bound method> = bound_method %a.ref.loc40, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc41
// CHECK:STDOUT:   %impl.elem0.loc42: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc42: <bound method> = bound_method %b.ref.loc42, %impl.elem0.loc42
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc42: init %Cpp.long = call %bound_method.loc42(%b.ref.loc42)
// CHECK:STDOUT:   %.loc42.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc42
// CHECK:STDOUT:   %.loc42.2: %Cpp.long = converted %b.ref.loc42, %.loc42.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc41: init bool = call %bound_method.loc41.4(%a.ref.loc40, %.loc42.2)
// CHECK:STDOUT:   %a.ref.loc44: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc46: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc45: %.753 = impl_witness_access constants.%OrderedWith.impl_witness.576, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.2]
// CHECK:STDOUT:   %bound_method.loc45.1: <bound method> = bound_method %a.ref.loc44, %impl.elem3.loc45
// CHECK:STDOUT:   %ImplicitAs.facet.loc45: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc45.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc45 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc45: <specific function> = specific_function %impl.elem3.loc45, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.1]
// CHECK:STDOUT:   %bound_method.loc45.2: <bound method> = bound_method %a.ref.loc44, %specific_fn.loc45
// CHECK:STDOUT:   %.loc45.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.1 = specific_constant imports.%Core.GreaterOrEquivalent.76e, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc45: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.1 = name_ref GreaterOrEquivalent, %.loc45.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc45: <bound method> = bound_method %a.ref.loc44, %GreaterOrEquivalent.ref.loc45
// CHECK:STDOUT:   %impl.elem0.loc45: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc45.3: <bound method> = bound_method %b.ref.loc46, %impl.elem0.loc45
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc45: init %Cpp.long = call %bound_method.loc45.3(%b.ref.loc46)
// CHECK:STDOUT:   %.loc45.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc45
// CHECK:STDOUT:   %.loc45.4: %Cpp.long = converted %b.ref.loc46, %.loc45.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc45: <specific function> = specific_function %GreaterOrEquivalent.ref.loc45, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.2]
// CHECK:STDOUT:   %bound_method.loc45.4: <bound method> = bound_method %a.ref.loc44, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc45
// CHECK:STDOUT:   %impl.elem0.loc46: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc46: <bound method> = bound_method %b.ref.loc46, %impl.elem0.loc46
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc46: init %Cpp.long = call %bound_method.loc46(%b.ref.loc46)
// CHECK:STDOUT:   %.loc46.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc46
// CHECK:STDOUT:   %.loc46.2: %Cpp.long = converted %b.ref.loc46, %.loc46.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc45: init bool = call %bound_method.loc45.4(%a.ref.loc44, %.loc46.2)
// CHECK:STDOUT:   %a.ref.loc48: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc50: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc49: %.14a = impl_witness_access constants.%OrderedWith.impl_witness.576, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.2]
// CHECK:STDOUT:   %bound_method.loc49.1: <bound method> = bound_method %a.ref.loc48, %impl.elem1.loc49
// CHECK:STDOUT:   %ImplicitAs.facet.loc49: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc49.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc49 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc49: <specific function> = specific_function %impl.elem1.loc49, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9548d5.1]
// CHECK:STDOUT:   %bound_method.loc49.2: <bound method> = bound_method %a.ref.loc48, %specific_fn.loc49
// CHECK:STDOUT:   %.loc49.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.1 = specific_constant imports.%Core.LessOrEquivalent.4a6, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc49: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.1 = name_ref LessOrEquivalent, %.loc49.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc49: <bound method> = bound_method %a.ref.loc48, %LessOrEquivalent.ref.loc49
// CHECK:STDOUT:   %impl.elem0.loc49: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc49.3: <bound method> = bound_method %b.ref.loc50, %impl.elem0.loc49
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc49: init %Cpp.long = call %bound_method.loc49.3(%b.ref.loc50)
// CHECK:STDOUT:   %.loc49.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc49
// CHECK:STDOUT:   %.loc49.4: %Cpp.long = converted %b.ref.loc50, %.loc49.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc49: <specific function> = specific_function %LessOrEquivalent.ref.loc49, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9548d5.2]
// CHECK:STDOUT:   %bound_method.loc49.4: <bound method> = bound_method %a.ref.loc48, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc49
// CHECK:STDOUT:   %impl.elem0.loc50: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc50: <bound method> = bound_method %b.ref.loc50, %impl.elem0.loc50
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc50: init %Cpp.long = call %bound_method.loc50(%b.ref.loc50)
// CHECK:STDOUT:   %.loc50.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc50
// CHECK:STDOUT:   %.loc50.2: %Cpp.long = converted %b.ref.loc50, %.loc50.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc49: init bool = call %bound_method.loc49.4(%a.ref.loc48, %.loc50.2)
// CHECK:STDOUT:   %a.ref.loc52: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc54: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc53.1: %.a03 = impl_witness_access constants.%EqWith.impl_witness.8e5, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.6ed862.2]
// CHECK:STDOUT:   %bound_method.loc53.1: <bound method> = bound_method %a.ref.loc52, %impl.elem0.loc53.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc53: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc53.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc53 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc53: <specific function> = specific_function %impl.elem0.loc53.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.334ae9.1]
// CHECK:STDOUT:   %bound_method.loc53.2: <bound method> = bound_method %a.ref.loc52, %specific_fn.loc53
// CHECK:STDOUT:   %.loc53.2: %Cpp.long.as.EqWith.impl.Equal.type.416d09.1 = specific_constant imports.%Core.Equal.0ad, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.6ed862.1]
// CHECK:STDOUT:   %Equal.ref.loc53: %Cpp.long.as.EqWith.impl.Equal.type.416d09.1 = name_ref Equal, %.loc53.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.6ed862.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc53: <bound method> = bound_method %a.ref.loc52, %Equal.ref.loc53
// CHECK:STDOUT:   %impl.elem0.loc53.2: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc53.3: <bound method> = bound_method %int_1.loc54, %impl.elem0.loc53.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53: init %Cpp.long = call %bound_method.loc53.3(%int_1.loc54) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc53.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc53.4: %Cpp.long = converted %int_1.loc54, %.loc53.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc53: <specific function> = specific_function %Equal.ref.loc53, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.334ae9.2]
// CHECK:STDOUT:   %bound_method.loc53.4: <bound method> = bound_method %a.ref.loc52, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc53
// CHECK:STDOUT:   %impl.elem0.loc54: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc54: <bound method> = bound_method %int_1.loc54, %impl.elem0.loc54 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc54: init %Cpp.long = call %bound_method.loc54(%int_1.loc54) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc54.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc54 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc54.2: %Cpp.long = converted %int_1.loc54, %.loc54.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc53: init bool = call %bound_method.loc53.4(%a.ref.loc52, %.loc54.2)
// CHECK:STDOUT:   %a.ref.loc56: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc58: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc57: %.9a1 = impl_witness_access constants.%EqWith.impl_witness.8e5, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.f686b3.2]
// CHECK:STDOUT:   %bound_method.loc57.1: <bound method> = bound_method %a.ref.loc56, %impl.elem1.loc57
// CHECK:STDOUT:   %ImplicitAs.facet.loc57: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc57.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc57 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc57: <specific function> = specific_function %impl.elem1.loc57, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5aa006.1]
// CHECK:STDOUT:   %bound_method.loc57.2: <bound method> = bound_method %a.ref.loc56, %specific_fn.loc57
// CHECK:STDOUT:   %.loc57.2: %Cpp.long.as.EqWith.impl.NotEqual.type.de968c.1 = specific_constant imports.%Core.NotEqual.b1f, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.f686b3.1]
// CHECK:STDOUT:   %NotEqual.ref.loc57: %Cpp.long.as.EqWith.impl.NotEqual.type.de968c.1 = name_ref NotEqual, %.loc57.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.f686b3.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc57: <bound method> = bound_method %a.ref.loc56, %NotEqual.ref.loc57
// CHECK:STDOUT:   %impl.elem0.loc57: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc57.3: <bound method> = bound_method %int_1.loc58, %impl.elem0.loc57 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc57: init %Cpp.long = call %bound_method.loc57.3(%int_1.loc58) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc57.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc57 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc57.4: %Cpp.long = converted %int_1.loc58, %.loc57.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc57: <specific function> = specific_function %NotEqual.ref.loc57, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5aa006.2]
// CHECK:STDOUT:   %bound_method.loc57.4: <bound method> = bound_method %a.ref.loc56, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc57
// CHECK:STDOUT:   %impl.elem0.loc58: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc58: <bound method> = bound_method %int_1.loc58, %impl.elem0.loc58 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc58: init %Cpp.long = call %bound_method.loc58(%int_1.loc58) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc58.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc58 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc58.2: %Cpp.long = converted %int_1.loc58, %.loc58.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc57: init bool = call %bound_method.loc57.4(%a.ref.loc56, %.loc58.2)
// CHECK:STDOUT:   %a.ref.loc60: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc62: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc61: %.5af = impl_witness_access constants.%OrderedWith.impl_witness.52f, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.74a807.2]
// CHECK:STDOUT:   %bound_method.loc61.1: <bound method> = bound_method %a.ref.loc60, %impl.elem2.loc61
// CHECK:STDOUT:   %ImplicitAs.facet.loc61: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc61.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc61 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc61: <specific function> = specific_function %impl.elem2.loc61, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.b2149b.1]
// CHECK:STDOUT:   %bound_method.loc61.2: <bound method> = bound_method %a.ref.loc60, %specific_fn.loc61
// CHECK:STDOUT:   %.loc61.2: %Cpp.long.as.OrderedWith.impl.Greater.type.3fd1ac.1 = specific_constant imports.%Core.Greater.9da, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.74a807.1]
// CHECK:STDOUT:   %Greater.ref.loc61: %Cpp.long.as.OrderedWith.impl.Greater.type.3fd1ac.1 = name_ref Greater, %.loc61.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.74a807.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc61: <bound method> = bound_method %a.ref.loc60, %Greater.ref.loc61
// CHECK:STDOUT:   %impl.elem0.loc61: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc61.3: <bound method> = bound_method %int_1.loc62, %impl.elem0.loc61 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc61: init %Cpp.long = call %bound_method.loc61.3(%int_1.loc62) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc61.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc61 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc61.4: %Cpp.long = converted %int_1.loc62, %.loc61.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc61: <specific function> = specific_function %Greater.ref.loc61, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.b2149b.2]
// CHECK:STDOUT:   %bound_method.loc61.4: <bound method> = bound_method %a.ref.loc60, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc61
// CHECK:STDOUT:   %impl.elem0.loc62: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc62: <bound method> = bound_method %int_1.loc62, %impl.elem0.loc62 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc62: init %Cpp.long = call %bound_method.loc62(%int_1.loc62) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc62.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc62 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc62.2: %Cpp.long = converted %int_1.loc62, %.loc62.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc61: init bool = call %bound_method.loc61.4(%a.ref.loc60, %.loc62.2)
// CHECK:STDOUT:   %a.ref.loc64: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc66: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc65.1: %.6b7 = impl_witness_access constants.%OrderedWith.impl_witness.52f, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.0de767.2]
// CHECK:STDOUT:   %bound_method.loc65.1: <bound method> = bound_method %a.ref.loc64, %impl.elem0.loc65.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc65: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc65.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc65 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc65: <specific function> = specific_function %impl.elem0.loc65.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.b6fc5e.1]
// CHECK:STDOUT:   %bound_method.loc65.2: <bound method> = bound_method %a.ref.loc64, %specific_fn.loc65
// CHECK:STDOUT:   %.loc65.2: %Cpp.long.as.OrderedWith.impl.Less.type.8ced31.1 = specific_constant imports.%Core.Less.c86, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.0de767.1]
// CHECK:STDOUT:   %Less.ref.loc65: %Cpp.long.as.OrderedWith.impl.Less.type.8ced31.1 = name_ref Less, %.loc65.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.0de767.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc65: <bound method> = bound_method %a.ref.loc64, %Less.ref.loc65
// CHECK:STDOUT:   %impl.elem0.loc65.2: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc65.3: <bound method> = bound_method %int_1.loc66, %impl.elem0.loc65.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc65: init %Cpp.long = call %bound_method.loc65.3(%int_1.loc66) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc65.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc65 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc65.4: %Cpp.long = converted %int_1.loc66, %.loc65.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc65: <specific function> = specific_function %Less.ref.loc65, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.b6fc5e.2]
// CHECK:STDOUT:   %bound_method.loc65.4: <bound method> = bound_method %a.ref.loc64, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc65
// CHECK:STDOUT:   %impl.elem0.loc66: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc66: <bound method> = bound_method %int_1.loc66, %impl.elem0.loc66 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc66: init %Cpp.long = call %bound_method.loc66(%int_1.loc66) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc66.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc66 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc66.2: %Cpp.long = converted %int_1.loc66, %.loc66.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc65: init bool = call %bound_method.loc65.4(%a.ref.loc64, %.loc66.2)
// CHECK:STDOUT:   %a.ref.loc68: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc70: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc69: %.1f9 = impl_witness_access constants.%OrderedWith.impl_witness.52f, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.2]
// CHECK:STDOUT:   %bound_method.loc69.1: <bound method> = bound_method %a.ref.loc68, %impl.elem3.loc69
// CHECK:STDOUT:   %ImplicitAs.facet.loc69: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc69.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc69 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc69: <specific function> = specific_function %impl.elem3.loc69, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f90679.1]
// CHECK:STDOUT:   %bound_method.loc69.2: <bound method> = bound_method %a.ref.loc68, %specific_fn.loc69
// CHECK:STDOUT:   %.loc69.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2273d7.1 = specific_constant imports.%Core.GreaterOrEquivalent.76e, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc69: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2273d7.1 = name_ref GreaterOrEquivalent, %.loc69.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc69: <bound method> = bound_method %a.ref.loc68, %GreaterOrEquivalent.ref.loc69
// CHECK:STDOUT:   %impl.elem0.loc69: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc69.3: <bound method> = bound_method %int_1.loc70, %impl.elem0.loc69 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc69: init %Cpp.long = call %bound_method.loc69.3(%int_1.loc70) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc69.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc69 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc69.4: %Cpp.long = converted %int_1.loc70, %.loc69.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc69: <specific function> = specific_function %GreaterOrEquivalent.ref.loc69, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f90679.2]
// CHECK:STDOUT:   %bound_method.loc69.4: <bound method> = bound_method %a.ref.loc68, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc69
// CHECK:STDOUT:   %impl.elem0.loc70: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc70: <bound method> = bound_method %int_1.loc70, %impl.elem0.loc70 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc70: init %Cpp.long = call %bound_method.loc70(%int_1.loc70) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc70.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc70 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc70.2: %Cpp.long = converted %int_1.loc70, %.loc70.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc69: init bool = call %bound_method.loc69.4(%a.ref.loc68, %.loc70.2)
// CHECK:STDOUT:   %a.ref.loc72: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc74: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc73: %.76b = impl_witness_access constants.%OrderedWith.impl_witness.52f, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.2]
// CHECK:STDOUT:   %bound_method.loc73.1: <bound method> = bound_method %a.ref.loc72, %impl.elem1.loc73
// CHECK:STDOUT:   %ImplicitAs.facet.loc73: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc73.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc73 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc73: <specific function> = specific_function %impl.elem1.loc73, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.c5c24a.1]
// CHECK:STDOUT:   %bound_method.loc73.2: <bound method> = bound_method %a.ref.loc72, %specific_fn.loc73
// CHECK:STDOUT:   %.loc73.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.28524d.1 = specific_constant imports.%Core.LessOrEquivalent.4a6, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc73: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.28524d.1 = name_ref LessOrEquivalent, %.loc73.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc73: <bound method> = bound_method %a.ref.loc72, %LessOrEquivalent.ref.loc73
// CHECK:STDOUT:   %impl.elem0.loc73: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc73.3: <bound method> = bound_method %int_1.loc74, %impl.elem0.loc73 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc73: init %Cpp.long = call %bound_method.loc73.3(%int_1.loc74) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc73.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc73 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc73.4: %Cpp.long = converted %int_1.loc74, %.loc73.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc73: <specific function> = specific_function %LessOrEquivalent.ref.loc73, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.c5c24a.2]
// CHECK:STDOUT:   %bound_method.loc73.4: <bound method> = bound_method %a.ref.loc72, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc73
// CHECK:STDOUT:   %impl.elem0.loc74: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc74: <bound method> = bound_method %int_1.loc74, %impl.elem0.loc74 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc74: init %Cpp.long = call %bound_method.loc74(%int_1.loc74) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc74.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc74 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc74.2: %Cpp.long = converted %int_1.loc74, %.loc74.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc73: init bool = call %bound_method.loc73.4(%a.ref.loc72, %.loc74.2)
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
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.593: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.57a) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.048: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.57a) [concrete]
// CHECK:STDOUT:   %.566: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.593, %EqWith.facet.57a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.b65948.2, @T.binding.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.b65948.1, @T.binding.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9f0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.bb8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9f0, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.4d8: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.048, %EqWith.facet.57a [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.NotEqual.37228f.2, @T.binding.as_type.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.NotEqual.37228f.1, @T.binding.as_type.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.a39: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long)> [concrete]
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
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.373: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.fa3) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.43c: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.fa3) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.021: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.fa3) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a29: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.fa3) [concrete]
// CHECK:STDOUT:   %.d93: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.021, %OrderedWith.facet.fa3 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.2, @T.binding.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.1, @T.binding.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.4a1: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.373, %OrderedWith.facet.fa3 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.f24104.2, @T.binding.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.f24104.1, @T.binding.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.7fd: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.a29, %OrderedWith.facet.fa3 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.2, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.1, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.8f3: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.43c, %OrderedWith.facet.fa3 [concrete]
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
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.9e3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.fa2) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.ede: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long, %EqWith.facet.fa2) [concrete]
// CHECK:STDOUT:   %.389: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.9e3, %EqWith.facet.fa2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.62ea05.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.326379.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.1: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.326379.2, @T.binding.as_type.as.EqWith.impl.Equal.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.8a46b0.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.62ea05.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.326379.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.2: <specific function> = specific_function %T.binding.as_type.as.EqWith.impl.Equal.326379.1, @T.binding.as_type.as.EqWith.impl.Equal.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.8a46b0.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.2 [concrete]
// CHECK:STDOUT:   %.ec0: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.ede, %EqWith.facet.fa2 [concrete]
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
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.2af: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.2bc) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.916: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.2bc) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.796: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.2bc) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.e13: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long, %OrderedWith.facet.2bc) [concrete]
// CHECK:STDOUT:   %.cfa: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.796, %OrderedWith.facet.2bc [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.8ab0dc.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.2, @T.binding.as_type.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.ea09f6.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.8ab0dc.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.1, @T.binding.as_type.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.ea09f6.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.2 [concrete]
// CHECK:STDOUT:   %.ea7: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.2af, %OrderedWith.facet.2bc [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.a0ee36.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.2, @T.binding.as_type.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.d20c63.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.a0ee36.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.1, @T.binding.as_type.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.d20c63.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.2 [concrete]
// CHECK:STDOUT:   %.8f6: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.e13, %OrderedWith.facet.2bc [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.378e29.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.2 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.1: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.2, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.212360.1: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.378e29.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.1 [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.2: <specific function> = specific_function %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.1, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.eed) [concrete]
// CHECK:STDOUT:   %bound_method.212360.2: <bound method> = bound_method %int_1.5b8, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.2 [concrete]
// CHECK:STDOUT:   %.2ed: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.916, %OrderedWith.facet.2bc [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc214, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.534: @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.Equal.type.2 (%T.binding.as_type.as.EqWith.impl.Equal.type.fd2c2f.1) = import_ref Core//prelude/types/cpp/int, loc1344, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.Equal.2 (constants.%T.binding.as_type.as.EqWith.impl.Equal.c3a936.1)]
// CHECK:STDOUT:   %Core.import_ref.5d5: @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.NotEqual.type.2 (%T.binding.as_type.as.EqWith.impl.NotEqual.type.71d714.1) = import_ref Core//prelude/types/cpp/int, loc1365, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.binding.as_type.as.EqWith.impl.NotEqual.2eedb3.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.c6d = impl_witness_table (%Core.import_ref.534, %Core.import_ref.5d5), @T.binding.as_type.as.EqWith.impl.8ff [concrete]
// CHECK:STDOUT:   %Core.NotEqual.a7f: @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.NotEqual.type.1 (%T.binding.as_type.as.EqWith.impl.NotEqual.type.71d714.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.binding.as_type.as.EqWith.impl.NotEqual.2eedb3.2)]
// CHECK:STDOUT:   %Core.Equal.784: @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.Equal.type.1 (%T.binding.as_type.as.EqWith.impl.Equal.type.fd2c2f.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.binding.as_type.as.EqWith.impl.8ff.%T.binding.as_type.as.EqWith.impl.Equal.1 (constants.%T.binding.as_type.as.EqWith.impl.Equal.c3a936.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc300, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9e8: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Less.type.2 (%T.binding.as_type.as.OrderedWith.impl.Less.type.7f8c78.1) = import_ref Core//prelude/types/cpp/int, loc1408, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Less.2 (constants.%T.binding.as_type.as.OrderedWith.impl.Less.a71c43.1)]
// CHECK:STDOUT:   %Core.import_ref.973: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.e16008.1) = import_ref Core//prelude/types/cpp/int, loc1429, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.4a32de.1)]
// CHECK:STDOUT:   %Core.import_ref.155: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Greater.type.2 (%T.binding.as_type.as.OrderedWith.impl.Greater.type.5284c3.1) = import_ref Core//prelude/types/cpp/int, loc1450, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.binding.as_type.as.OrderedWith.impl.Greater.1fe718.1)]
// CHECK:STDOUT:   %Core.import_ref.f9eb: @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.058043.1) = import_ref Core//prelude/types/cpp/int, loc1471, loaded [symbolic = @T.binding.as_type.as.OrderedWith.impl.891.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.9f68cd.1)]
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
// CHECK:STDOUT:   %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc23: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc26: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %int_1.loc26, %impl.elem0.loc26 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26: init %Cpp.long = call %bound_method.loc26(%int_1.loc26) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc26.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc26.2: %Cpp.long = converted %int_1.loc26, %.loc26.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc26.2
// CHECK:STDOUT:   %b.ref.loc28: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc30: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc29: %.566 = impl_witness_access constants.%EqWith.impl_witness.949, element0 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.2]
// CHECK:STDOUT:   %bound_method.loc29.1: <bound method> = bound_method %b.ref.loc28, %impl.elem0.loc29
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @T.binding.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.1]
// CHECK:STDOUT:   %bound_method.loc29.2: <bound method> = bound_method %b.ref.loc28, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29: %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.1 = specific_constant imports.%Core.Equal.784, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.1]
// CHECK:STDOUT:   %Equal.ref.loc29: %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.1 = name_ref Equal, %.loc29 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.loc29: <bound method> = bound_method %b.ref.loc28, %Equal.ref.loc29
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc29: <specific function> = specific_function %Equal.ref.loc29, @T.binding.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.2]
// CHECK:STDOUT:   %bound_method.loc29.3: <bound method> = bound_method %b.ref.loc28, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc29
// CHECK:STDOUT:   %impl.elem0.loc28: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28: <bound method> = bound_method %b.ref.loc28, %impl.elem0.loc28
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc28: init %Cpp.long = call %bound_method.loc28(%b.ref.loc28)
// CHECK:STDOUT:   %.loc28.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc28
// CHECK:STDOUT:   %.loc28.2: %Cpp.long = converted %b.ref.loc28, %.loc28.1
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.call.loc29: init bool = call %bound_method.loc29.3(%.loc28.2, %a.ref.loc30)
// CHECK:STDOUT:   %b.ref.loc32: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc34: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc33: %.4d8 = impl_witness_access constants.%EqWith.impl_witness.949, element1 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.2]
// CHECK:STDOUT:   %bound_method.loc33.1: <bound method> = bound_method %b.ref.loc32, %impl.elem1.loc33
// CHECK:STDOUT:   %specific_fn.loc33: <specific function> = specific_function %impl.elem1.loc33, @T.binding.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.1]
// CHECK:STDOUT:   %bound_method.loc33.2: <bound method> = bound_method %b.ref.loc32, %specific_fn.loc33
// CHECK:STDOUT:   %.loc33: %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.1 = specific_constant imports.%Core.NotEqual.a7f, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.1]
// CHECK:STDOUT:   %NotEqual.ref.loc33: %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.1 = name_ref NotEqual, %.loc33 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.loc33: <bound method> = bound_method %b.ref.loc32, %NotEqual.ref.loc33
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc33: <specific function> = specific_function %NotEqual.ref.loc33, @T.binding.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.2]
// CHECK:STDOUT:   %bound_method.loc33.3: <bound method> = bound_method %b.ref.loc32, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc33
// CHECK:STDOUT:   %impl.elem0.loc32: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc32: <bound method> = bound_method %b.ref.loc32, %impl.elem0.loc32
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc32: init %Cpp.long = call %bound_method.loc32(%b.ref.loc32)
// CHECK:STDOUT:   %.loc32.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc32
// CHECK:STDOUT:   %.loc32.2: %Cpp.long = converted %b.ref.loc32, %.loc32.1
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.call.loc33: init bool = call %bound_method.loc33.3(%.loc32.2, %a.ref.loc34)
// CHECK:STDOUT:   %b.ref.loc36: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc38: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc37: %.d93 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element2 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.2]
// CHECK:STDOUT:   %bound_method.loc37.1: <bound method> = bound_method %b.ref.loc36, %impl.elem2.loc37
// CHECK:STDOUT:   %specific_fn.loc37: <specific function> = specific_function %impl.elem2.loc37, @T.binding.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.1]
// CHECK:STDOUT:   %bound_method.loc37.2: <bound method> = bound_method %b.ref.loc36, %specific_fn.loc37
// CHECK:STDOUT:   %.loc37: %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.1 = specific_constant imports.%Core.Greater.696, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.1]
// CHECK:STDOUT:   %Greater.ref.loc37: %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.1 = name_ref Greater, %.loc37 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.loc37: <bound method> = bound_method %b.ref.loc36, %Greater.ref.loc37
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc37: <specific function> = specific_function %Greater.ref.loc37, @T.binding.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.2]
// CHECK:STDOUT:   %bound_method.loc37.3: <bound method> = bound_method %b.ref.loc36, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc37
// CHECK:STDOUT:   %impl.elem0.loc36: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc36: <bound method> = bound_method %b.ref.loc36, %impl.elem0.loc36
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc36: init %Cpp.long = call %bound_method.loc36(%b.ref.loc36)
// CHECK:STDOUT:   %.loc36.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc36
// CHECK:STDOUT:   %.loc36.2: %Cpp.long = converted %b.ref.loc36, %.loc36.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.call.loc37: init bool = call %bound_method.loc37.3(%.loc36.2, %a.ref.loc38)
// CHECK:STDOUT:   %b.ref.loc40: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc42: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc41: %.4a1 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element0 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.2]
// CHECK:STDOUT:   %bound_method.loc41.1: <bound method> = bound_method %b.ref.loc40, %impl.elem0.loc41
// CHECK:STDOUT:   %specific_fn.loc41: <specific function> = specific_function %impl.elem0.loc41, @T.binding.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.1]
// CHECK:STDOUT:   %bound_method.loc41.2: <bound method> = bound_method %b.ref.loc40, %specific_fn.loc41
// CHECK:STDOUT:   %.loc41: %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.1 = specific_constant imports.%Core.Less.b61, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.1]
// CHECK:STDOUT:   %Less.ref.loc41: %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.1 = name_ref Less, %.loc41 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.loc41: <bound method> = bound_method %b.ref.loc40, %Less.ref.loc41
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc41: <specific function> = specific_function %Less.ref.loc41, @T.binding.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.2]
// CHECK:STDOUT:   %bound_method.loc41.3: <bound method> = bound_method %b.ref.loc40, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc41
// CHECK:STDOUT:   %impl.elem0.loc40: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc40: <bound method> = bound_method %b.ref.loc40, %impl.elem0.loc40
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc40: init %Cpp.long = call %bound_method.loc40(%b.ref.loc40)
// CHECK:STDOUT:   %.loc40.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc40
// CHECK:STDOUT:   %.loc40.2: %Cpp.long = converted %b.ref.loc40, %.loc40.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.call.loc41: init bool = call %bound_method.loc41.3(%.loc40.2, %a.ref.loc42)
// CHECK:STDOUT:   %b.ref.loc44: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc46: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc45: %.7fd = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element3 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.2]
// CHECK:STDOUT:   %bound_method.loc45.1: <bound method> = bound_method %b.ref.loc44, %impl.elem3.loc45
// CHECK:STDOUT:   %specific_fn.loc45: <specific function> = specific_function %impl.elem3.loc45, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.1]
// CHECK:STDOUT:   %bound_method.loc45.2: <bound method> = bound_method %b.ref.loc44, %specific_fn.loc45
// CHECK:STDOUT:   %.loc45: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.1 = specific_constant imports.%Core.GreaterOrEquivalent.c46, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc45: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.1 = name_ref GreaterOrEquivalent, %.loc45 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc45: <bound method> = bound_method %b.ref.loc44, %GreaterOrEquivalent.ref.loc45
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc45: <specific function> = specific_function %GreaterOrEquivalent.ref.loc45, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.2]
// CHECK:STDOUT:   %bound_method.loc45.3: <bound method> = bound_method %b.ref.loc44, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc45
// CHECK:STDOUT:   %impl.elem0.loc44: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc44: <bound method> = bound_method %b.ref.loc44, %impl.elem0.loc44
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc44: init %Cpp.long = call %bound_method.loc44(%b.ref.loc44)
// CHECK:STDOUT:   %.loc44.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc44
// CHECK:STDOUT:   %.loc44.2: %Cpp.long = converted %b.ref.loc44, %.loc44.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc45: init bool = call %bound_method.loc45.3(%.loc44.2, %a.ref.loc46)
// CHECK:STDOUT:   %b.ref.loc48: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc50: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc49: %.8f3 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element1 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.2]
// CHECK:STDOUT:   %bound_method.loc49.1: <bound method> = bound_method %b.ref.loc48, %impl.elem1.loc49
// CHECK:STDOUT:   %specific_fn.loc49: <specific function> = specific_function %impl.elem1.loc49, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a789d4.1]
// CHECK:STDOUT:   %bound_method.loc49.2: <bound method> = bound_method %b.ref.loc48, %specific_fn.loc49
// CHECK:STDOUT:   %.loc49: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.1 = specific_constant imports.%Core.LessOrEquivalent.947, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc49: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.1 = name_ref LessOrEquivalent, %.loc49 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc49: <bound method> = bound_method %b.ref.loc48, %LessOrEquivalent.ref.loc49
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc49: <specific function> = specific_function %LessOrEquivalent.ref.loc49, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a789d4.2]
// CHECK:STDOUT:   %bound_method.loc49.3: <bound method> = bound_method %b.ref.loc48, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc49
// CHECK:STDOUT:   %impl.elem0.loc48: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc48: <bound method> = bound_method %b.ref.loc48, %impl.elem0.loc48
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc48: init %Cpp.long = call %bound_method.loc48(%b.ref.loc48)
// CHECK:STDOUT:   %.loc48.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc48
// CHECK:STDOUT:   %.loc48.2: %Cpp.long = converted %b.ref.loc48, %.loc48.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc49: init bool = call %bound_method.loc49.3(%.loc48.2, %a.ref.loc50)
// CHECK:STDOUT:   %int_1.loc52: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc54: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc53: %.389 = impl_witness_access constants.%EqWith.impl_witness.801, element0 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.326379.2]
// CHECK:STDOUT:   %bound_method.loc53.1: <bound method> = bound_method %int_1.loc52, %impl.elem0.loc53 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.bound.62ea05.1]
// CHECK:STDOUT:   %specific_fn.loc53: <specific function> = specific_function %impl.elem0.loc53, @T.binding.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.1]
// CHECK:STDOUT:   %bound_method.loc53.2: <bound method> = bound_method %int_1.loc52, %specific_fn.loc53 [concrete = constants.%bound_method.8a46b0.1]
// CHECK:STDOUT:   %.loc53: %T.binding.as_type.as.EqWith.impl.Equal.type.85dd3c.1 = specific_constant imports.%Core.Equal.784, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.326379.1]
// CHECK:STDOUT:   %Equal.ref.loc53: %T.binding.as_type.as.EqWith.impl.Equal.type.85dd3c.1 = name_ref Equal, %.loc53 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.326379.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.loc53: <bound method> = bound_method %int_1.loc52, %Equal.ref.loc53 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.bound.62ea05.2]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc53: <specific function> = specific_function %Equal.ref.loc53, @T.binding.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.2]
// CHECK:STDOUT:   %bound_method.loc53.3: <bound method> = bound_method %int_1.loc52, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc53 [concrete = constants.%bound_method.8a46b0.2]
// CHECK:STDOUT:   %impl.elem0.loc52: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc52: <bound method> = bound_method %int_1.loc52, %impl.elem0.loc52 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc52: init %Cpp.long = call %bound_method.loc52(%int_1.loc52) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc52.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc52 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc52.2: %Cpp.long = converted %int_1.loc52, %.loc52.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.call.loc53: init bool = call %bound_method.loc53.3(%.loc52.2, %a.ref.loc54)
// CHECK:STDOUT:   %int_1.loc56: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc58: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc57: %.ec0 = impl_witness_access constants.%EqWith.impl_witness.801, element1 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.2]
// CHECK:STDOUT:   %bound_method.loc57.1: <bound method> = bound_method %int_1.loc56, %impl.elem1.loc57 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bound.29032e.1]
// CHECK:STDOUT:   %specific_fn.loc57: <specific function> = specific_function %impl.elem1.loc57, @T.binding.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.ceecb6.1]
// CHECK:STDOUT:   %bound_method.loc57.2: <bound method> = bound_method %int_1.loc56, %specific_fn.loc57 [concrete = constants.%bound_method.082399.1]
// CHECK:STDOUT:   %.loc57: %T.binding.as_type.as.EqWith.impl.NotEqual.type.bb8d74.1 = specific_constant imports.%Core.NotEqual.a7f, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.1]
// CHECK:STDOUT:   %NotEqual.ref.loc57: %T.binding.as_type.as.EqWith.impl.NotEqual.type.bb8d74.1 = name_ref NotEqual, %.loc57 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.loc57: <bound method> = bound_method %int_1.loc56, %NotEqual.ref.loc57 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bound.29032e.2]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc57: <specific function> = specific_function %NotEqual.ref.loc57, @T.binding.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.ceecb6.2]
// CHECK:STDOUT:   %bound_method.loc57.3: <bound method> = bound_method %int_1.loc56, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc57 [concrete = constants.%bound_method.082399.2]
// CHECK:STDOUT:   %impl.elem0.loc56: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc56: <bound method> = bound_method %int_1.loc56, %impl.elem0.loc56 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc56: init %Cpp.long = call %bound_method.loc56(%int_1.loc56) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc56.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc56 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc56.2: %Cpp.long = converted %int_1.loc56, %.loc56.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.call.loc57: init bool = call %bound_method.loc57.3(%.loc56.2, %a.ref.loc58)
// CHECK:STDOUT:   %int_1.loc60: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc62: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc61: %.cfa = impl_witness_access constants.%OrderedWith.impl_witness.252, element2 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.2]
// CHECK:STDOUT:   %bound_method.loc61.1: <bound method> = bound_method %int_1.loc60, %impl.elem2.loc61 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.bound.8ab0dc.1]
// CHECK:STDOUT:   %specific_fn.loc61: <specific function> = specific_function %impl.elem2.loc61, @T.binding.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.1]
// CHECK:STDOUT:   %bound_method.loc61.2: <bound method> = bound_method %int_1.loc60, %specific_fn.loc61 [concrete = constants.%bound_method.ea09f6.1]
// CHECK:STDOUT:   %.loc61: %T.binding.as_type.as.OrderedWith.impl.Greater.type.4ddc57.1 = specific_constant imports.%Core.Greater.696, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.1]
// CHECK:STDOUT:   %Greater.ref.loc61: %T.binding.as_type.as.OrderedWith.impl.Greater.type.4ddc57.1 = name_ref Greater, %.loc61 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.loc61: <bound method> = bound_method %int_1.loc60, %Greater.ref.loc61 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.bound.8ab0dc.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc61: <specific function> = specific_function %Greater.ref.loc61, @T.binding.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.2]
// CHECK:STDOUT:   %bound_method.loc61.3: <bound method> = bound_method %int_1.loc60, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc61 [concrete = constants.%bound_method.ea09f6.2]
// CHECK:STDOUT:   %impl.elem0.loc60: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc60: <bound method> = bound_method %int_1.loc60, %impl.elem0.loc60 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60: init %Cpp.long = call %bound_method.loc60(%int_1.loc60) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc60.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc60.2: %Cpp.long = converted %int_1.loc60, %.loc60.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.call.loc61: init bool = call %bound_method.loc61.3(%.loc60.2, %a.ref.loc62)
// CHECK:STDOUT:   %int_1.loc64: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc66: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc65: %.ea7 = impl_witness_access constants.%OrderedWith.impl_witness.252, element0 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.2]
// CHECK:STDOUT:   %bound_method.loc65.1: <bound method> = bound_method %int_1.loc64, %impl.elem0.loc65 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.bound.a0ee36.1]
// CHECK:STDOUT:   %specific_fn.loc65: <specific function> = specific_function %impl.elem0.loc65, @T.binding.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.1]
// CHECK:STDOUT:   %bound_method.loc65.2: <bound method> = bound_method %int_1.loc64, %specific_fn.loc65 [concrete = constants.%bound_method.d20c63.1]
// CHECK:STDOUT:   %.loc65: %T.binding.as_type.as.OrderedWith.impl.Less.type.285b47.1 = specific_constant imports.%Core.Less.b61, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.1]
// CHECK:STDOUT:   %Less.ref.loc65: %T.binding.as_type.as.OrderedWith.impl.Less.type.285b47.1 = name_ref Less, %.loc65 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.loc65: <bound method> = bound_method %int_1.loc64, %Less.ref.loc65 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.bound.a0ee36.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc65: <specific function> = specific_function %Less.ref.loc65, @T.binding.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.2]
// CHECK:STDOUT:   %bound_method.loc65.3: <bound method> = bound_method %int_1.loc64, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc65 [concrete = constants.%bound_method.d20c63.2]
// CHECK:STDOUT:   %impl.elem0.loc64: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc64: <bound method> = bound_method %int_1.loc64, %impl.elem0.loc64 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc64: init %Cpp.long = call %bound_method.loc64(%int_1.loc64) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc64.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc64 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc64.2: %Cpp.long = converted %int_1.loc64, %.loc64.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.call.loc65: init bool = call %bound_method.loc65.3(%.loc64.2, %a.ref.loc66)
// CHECK:STDOUT:   %int_1.loc68: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc70: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc69: %.8f6 = impl_witness_access constants.%OrderedWith.impl_witness.252, element3 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.2]
// CHECK:STDOUT:   %bound_method.loc69.1: <bound method> = bound_method %int_1.loc68, %impl.elem3.loc69 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.378e29.1]
// CHECK:STDOUT:   %specific_fn.loc69: <specific function> = specific_function %impl.elem3.loc69, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.1]
// CHECK:STDOUT:   %bound_method.loc69.2: <bound method> = bound_method %int_1.loc68, %specific_fn.loc69 [concrete = constants.%bound_method.212360.1]
// CHECK:STDOUT:   %.loc69: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9cc1f0.1 = specific_constant imports.%Core.GreaterOrEquivalent.c46, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc69: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9cc1f0.1 = name_ref GreaterOrEquivalent, %.loc69 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc69: <bound method> = bound_method %int_1.loc68, %GreaterOrEquivalent.ref.loc69 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.378e29.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc69: <specific function> = specific_function %GreaterOrEquivalent.ref.loc69, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.2]
// CHECK:STDOUT:   %bound_method.loc69.3: <bound method> = bound_method %int_1.loc68, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc69 [concrete = constants.%bound_method.212360.2]
// CHECK:STDOUT:   %impl.elem0.loc68: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc68: <bound method> = bound_method %int_1.loc68, %impl.elem0.loc68 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc68: init %Cpp.long = call %bound_method.loc68(%int_1.loc68) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc68.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc68 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc68.2: %Cpp.long = converted %int_1.loc68, %.loc68.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc69: init bool = call %bound_method.loc69.3(%.loc68.2, %a.ref.loc70)
// CHECK:STDOUT:   %int_1.loc72: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc74: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc73: %.2ed = impl_witness_access constants.%OrderedWith.impl_witness.252, element1 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.2]
// CHECK:STDOUT:   %bound_method.loc73.1: <bound method> = bound_method %int_1.loc72, %impl.elem1.loc73 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.5ef5e5.1]
// CHECK:STDOUT:   %specific_fn.loc73: <specific function> = specific_function %impl.elem1.loc73, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.925a78.1]
// CHECK:STDOUT:   %bound_method.loc73.2: <bound method> = bound_method %int_1.loc72, %specific_fn.loc73 [concrete = constants.%bound_method.e9dda0.1]
// CHECK:STDOUT:   %.loc73: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8a3347.1 = specific_constant imports.%Core.LessOrEquivalent.947, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc73: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8a3347.1 = name_ref LessOrEquivalent, %.loc73 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc73: <bound method> = bound_method %int_1.loc72, %LessOrEquivalent.ref.loc73 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.5ef5e5.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc73: <specific function> = specific_function %LessOrEquivalent.ref.loc73, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.925a78.2]
// CHECK:STDOUT:   %bound_method.loc73.3: <bound method> = bound_method %int_1.loc72, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc73 [concrete = constants.%bound_method.e9dda0.2]
// CHECK:STDOUT:   %impl.elem0.loc72: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc72: <bound method> = bound_method %int_1.loc72, %impl.elem0.loc72 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc72: init %Cpp.long = call %bound_method.loc72(%int_1.loc72) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc72.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc72 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc72.2: %Cpp.long = converted %int_1.loc72, %.loc72.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc73: init bool = call %bound_method.loc73.3(%.loc72.2, %a.ref.loc74)
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
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc214, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.1 = import_ref Core//prelude/types/cpp/int, loc1657, unloaded
// CHECK:STDOUT:   %Core.import_ref.412: %Cpp.long.as.Negate.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc1669, loaded [concrete = constants.%Cpp.long.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.cb912f.1, %Core.import_ref.412), @Cpp.long.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.4 = import_ref Core//prelude/types/cpp/int, loc1688, unloaded
// CHECK:STDOUT:   %Core.import_ref.437: %Cpp.long.as.AddWith.impl.Op.type.094 = import_ref Core//prelude/types/cpp/int, loc1703, loaded [concrete = constants.%Cpp.long.as.AddWith.impl.Op.be0]
// CHECK:STDOUT:   %AddWith.impl_witness_table.2f5 = impl_witness_table (%Core.import_ref.cb912f.4, %Core.import_ref.437), @Cpp.long.as.AddWith.impl.8b9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.7 = import_ref Core//prelude/types/cpp/int, loc1722, unloaded
// CHECK:STDOUT:   %Core.import_ref.51f: %Cpp.long.as.SubWith.impl.Op.type.882 = import_ref Core//prelude/types/cpp/int, loc1737, loaded [concrete = constants.%Cpp.long.as.SubWith.impl.Op.37d]
// CHECK:STDOUT:   %SubWith.impl_witness_table.e2e = impl_witness_table (%Core.import_ref.cb912f.7, %Core.import_ref.51f), @Cpp.long.as.SubWith.impl.214 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.10 = import_ref Core//prelude/types/cpp/int, loc1756, unloaded
// CHECK:STDOUT:   %Core.import_ref.ec4: %Cpp.long.as.MulWith.impl.Op.type.31a = import_ref Core//prelude/types/cpp/int, loc1771, loaded [concrete = constants.%Cpp.long.as.MulWith.impl.Op.4e5]
// CHECK:STDOUT:   %MulWith.impl_witness_table.20e = impl_witness_table (%Core.import_ref.cb912f.10, %Core.import_ref.ec4), @Cpp.long.as.MulWith.impl.ee0 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.13 = import_ref Core//prelude/types/cpp/int, loc1790, unloaded
// CHECK:STDOUT:   %Core.import_ref.d85: %Cpp.long.as.DivWith.impl.Op.type.5e7 = import_ref Core//prelude/types/cpp/int, loc1805, loaded [concrete = constants.%Cpp.long.as.DivWith.impl.Op.33f]
// CHECK:STDOUT:   %DivWith.impl_witness_table.1ae = impl_witness_table (%Core.import_ref.cb912f.13, %Core.import_ref.d85), @Cpp.long.as.DivWith.impl.cb9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb912f.16 = import_ref Core//prelude/types/cpp/int, loc1824, unloaded
// CHECK:STDOUT:   %Core.import_ref.394: %Cpp.long.as.ModWith.impl.Op.type.aa5 = import_ref Core//prelude/types/cpp/int, loc1839, loaded [concrete = constants.%Cpp.long.as.ModWith.impl.Op.06e]
// CHECK:STDOUT:   %ModWith.impl_witness_table.ed4 = impl_witness_table (%Core.import_ref.cb912f.16, %Core.import_ref.394), @Cpp.long.as.ModWith.impl.2d7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ArithmeticHomogeneousLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.68c = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc16: type = splice_block %long.ref.loc16 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc16: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc19: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19.2: %Cpp.long = converted %int_1.loc19, %.loc19.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %x: %Cpp.long = value_binding x, %.loc19.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.68c = value_binding_pattern y [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc28: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc25: type = splice_block %long.ref.loc25 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc25: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc28: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28: <bound method> = bound_method %int_1.loc28, %impl.elem0.loc28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28: init %Cpp.long = call %bound_method.loc28(%int_1.loc28) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc28.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc28.2: %Cpp.long = converted %int_1.loc28, %.loc28.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %y: %Cpp.long = value_binding y, %.loc28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc39: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc38: %.22b = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Cpp.long.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc38: <bound method> = bound_method %x.ref.loc39, %impl.elem1.loc38
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op.call: init %Cpp.long = call %bound_method.loc38(%x.ref.loc39)
// CHECK:STDOUT:   %.loc35: type = splice_block %long.ref.loc35 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc35: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc38.1: %Cpp.long = value_of_initializer %Cpp.long.as.Negate.impl.Op.call
// CHECK:STDOUT:   %.loc38.2: %Cpp.long = converted %Cpp.long.as.Negate.impl.Op.call, %.loc38.1
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc49: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc51: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc50: %.165 = impl_witness_access constants.%AddWith.impl_witness.e9f, element1 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.be0]
// CHECK:STDOUT:   %bound_method.loc50: <bound method> = bound_method %x.ref.loc49, %impl.elem1.loc50
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.call: init %Cpp.long = call %bound_method.loc50(%x.ref.loc49, %y.ref.loc51)
// CHECK:STDOUT:   %.loc46: type = splice_block %long.ref.loc46 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc46: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc50.1: %Cpp.long = value_of_initializer %Cpp.long.as.AddWith.impl.Op.call
// CHECK:STDOUT:   %.loc50.2: %Cpp.long = converted %Cpp.long.as.AddWith.impl.Op.call, %.loc50.1
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc50.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.68c = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc61: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc63: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc62: %.571 = impl_witness_access constants.%SubWith.impl_witness.8a6, element1 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.37d]
// CHECK:STDOUT:   %bound_method.loc62: <bound method> = bound_method %x.ref.loc61, %impl.elem1.loc62
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.call: init %Cpp.long = call %bound_method.loc62(%x.ref.loc61, %y.ref.loc63)
// CHECK:STDOUT:   %.loc58: type = splice_block %long.ref.loc58 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc58: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc62.1: %Cpp.long = value_of_initializer %Cpp.long.as.SubWith.impl.Op.call
// CHECK:STDOUT:   %.loc62.2: %Cpp.long = converted %Cpp.long.as.SubWith.impl.Op.call, %.loc62.1
// CHECK:STDOUT:   %c: %Cpp.long = value_binding c, %.loc62.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.68c = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc73: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc75: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc74: %.7d0 = impl_witness_access constants.%MulWith.impl_witness.87c, element1 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.4e5]
// CHECK:STDOUT:   %bound_method.loc74: <bound method> = bound_method %x.ref.loc73, %impl.elem1.loc74
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.call: init %Cpp.long = call %bound_method.loc74(%x.ref.loc73, %y.ref.loc75)
// CHECK:STDOUT:   %.loc70: type = splice_block %long.ref.loc70 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc69: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc70: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc74.1: %Cpp.long = value_of_initializer %Cpp.long.as.MulWith.impl.Op.call
// CHECK:STDOUT:   %.loc74.2: %Cpp.long = converted %Cpp.long.as.MulWith.impl.Op.call, %.loc74.1
// CHECK:STDOUT:   %d: %Cpp.long = value_binding d, %.loc74.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.68c = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc85: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc87: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc86: %.aa3 = impl_witness_access constants.%DivWith.impl_witness.4ab, element1 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.33f]
// CHECK:STDOUT:   %bound_method.loc86: <bound method> = bound_method %x.ref.loc85, %impl.elem1.loc86
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.call: init %Cpp.long = call %bound_method.loc86(%x.ref.loc85, %y.ref.loc87)
// CHECK:STDOUT:   %.loc82: type = splice_block %long.ref.loc82 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc81: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc82: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc86.1: %Cpp.long = value_of_initializer %Cpp.long.as.DivWith.impl.Op.call
// CHECK:STDOUT:   %.loc86.2: %Cpp.long = converted %Cpp.long.as.DivWith.impl.Op.call, %.loc86.1
// CHECK:STDOUT:   %e: %Cpp.long = value_binding e, %.loc86.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.68c = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc97: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc99: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc98: %.c19 = impl_witness_access constants.%ModWith.impl_witness.c73, element1 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.06e]
// CHECK:STDOUT:   %bound_method.loc98: <bound method> = bound_method %x.ref.loc97, %impl.elem1.loc98
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.call: init %Cpp.long = call %bound_method.loc98(%x.ref.loc97, %y.ref.loc99)
// CHECK:STDOUT:   %.loc94: type = splice_block %long.ref.loc94 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc93: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc94: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc98.1: %Cpp.long = value_of_initializer %Cpp.long.as.ModWith.impl.Op.call
// CHECK:STDOUT:   %.loc98.2: %Cpp.long = converted %Cpp.long.as.ModWith.impl.Op.call, %.loc98.1
// CHECK:STDOUT:   %f: %Cpp.long = value_binding f, %.loc98.2
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
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.type.55d: type = fn_type @T.binding.as_type.as.ImplicitAs.impl.Convert, @T.binding.as_type.as.ImplicitAs.impl(%T.035) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.19b: %T.binding.as_type.as.ImplicitAs.impl.Convert.type.55d = struct_value () [symbolic]
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
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.045: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.045: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.045, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.df00b6.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.8b7759.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.df00b6.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.8b7759.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.b12: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long)> [concrete]
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
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.887: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.052: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.887, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.864a67.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.31eaa0.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.864a67.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.31eaa0.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.59a: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long)> [concrete]
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
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.ddd: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.8d7: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.ddd, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.a37438.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.b067dc.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.a37438.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.b067dc.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.e6d: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long)> [concrete]
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
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.829: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.1d5: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.829, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.19b8da.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.b85335.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.19b8da.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.b85335.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.2a3: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long)> [concrete]
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
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.c4a: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.c4a, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.51594a.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.da4005.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.51594a.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.da4005.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.efd: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long)> [concrete]
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
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.e39: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.f3c: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.e39, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.1245aa.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.500716.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.1245aa.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.500716.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.b27: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long)> [concrete]
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
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.823: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.2c9: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.823, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.b7967f.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.b7967f.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.431: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long)> [concrete]
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
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.820: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.504: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.820, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.04506a.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.04506a.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.bed: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long)> [concrete]
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
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.d0a: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.20f: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.d0a, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0946f9.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0946f9.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.222: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long)> [concrete]
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
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.6ca: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.19a: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.6ca, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1ec453.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1ec453.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.3fe) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc214, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.d0e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1) = import_ref Core//prelude/types/cpp/int, loc4551, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.07a = impl_witness_table (%Core.import_ref.d0e), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f81: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.2)]
// CHECK:STDOUT:   %Core.import_ref.915: %Cpp.long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc119, loaded [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.572 = impl_witness_table (%Core.import_ref.915), @Cpp.long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.bcb: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.1) = import_ref Core//prelude/types/cpp/int, loc4589, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c39240.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.f4b = impl_witness_table (%Core.import_ref.bcb), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.739: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c39240.2)]
// CHECK:STDOUT:   %Core.import_ref.c5b: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.1) = import_ref Core//prelude/types/cpp/int, loc4627, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.421374.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.8cc = impl_witness_table (%Core.import_ref.c5b), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3b8: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.421374.2)]
// CHECK:STDOUT:   %Core.import_ref.84d: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.1) = import_ref Core//prelude/types/cpp/int, loc4665, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.6227df.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.6f7 = impl_witness_table (%Core.import_ref.84d), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f47: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.6227df.2)]
// CHECK:STDOUT:   %Core.import_ref.e98: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.1) = import_ref Core//prelude/types/cpp/int, loc4703, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.666e79.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.a2a = impl_witness_table (%Core.import_ref.e98), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b37: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.666e79.2)]
// CHECK:STDOUT:   %Core.import_ref.325: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.1) = import_ref Core//prelude/types/cpp/int, loc4931, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.d99897.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.cc9 = impl_witness_table (%Core.import_ref.325), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fe8: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.d99897.2)]
// CHECK:STDOUT:   %Core.import_ref.01a: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.1) = import_ref Core//prelude/types/cpp/int, loc4969, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.54c = impl_witness_table (%Core.import_ref.01a), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.5c2: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.2)]
// CHECK:STDOUT:   %Core.import_ref.c8f: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.1) = import_ref Core//prelude/types/cpp/int, loc5007, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.a39 = impl_witness_table (%Core.import_ref.c8f), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.caf: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.2)]
// CHECK:STDOUT:   %Core.import_ref.bcc: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.1) = import_ref Core//prelude/types/cpp/int, loc5045, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.74d = impl_witness_table (%Core.import_ref.bcc), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.030: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.2)]
// CHECK:STDOUT:   %Core.import_ref.5f5: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.1) = import_ref Core//prelude/types/cpp/int, loc5083, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.328772.1)]
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
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc12: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc12: init %Cpp.long = converted %int_1.loc19, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc12
// CHECK:STDOUT:   %.loc16: type = splice_block %long.ref.loc16 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc16: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.68c = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var %b.var_patt
// CHECK:STDOUT:   %int_1.loc28: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc21: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %int_1.loc28, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long = call %bound_method.loc21(%int_1.loc28) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc21: init %Cpp.long = converted %int_1.loc28, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %b.var, %.loc21
// CHECK:STDOUT:   %.loc25: type = splice_block %long.ref.loc25 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc25: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = ref_binding b, %b.var
// CHECK:STDOUT:   %a.ref.loc30: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc32: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc31: %.045 = impl_witness_access constants.%AddAssignWith.impl_witness.fb1, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.8b7759.2]
// CHECK:STDOUT:   %bound_method.loc31.1: <bound method> = bound_method %a.ref.loc30, %impl.elem0.loc31
// CHECK:STDOUT:   %ImplicitAs.facet.loc31: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc31.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc31 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.df00b6.1]
// CHECK:STDOUT:   %bound_method.loc31.2: <bound method> = bound_method %a.ref.loc30, %specific_fn.loc31
// CHECK:STDOUT:   %.loc32: %Cpp.long = acquire_value %b.ref.loc32
// CHECK:STDOUT:   %.loc31.2: %Cpp.long.as.AddAssignWith.impl.Op.type.defb93.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.8b7759.1]
// CHECK:STDOUT:   %Op.ref.loc31: %Cpp.long.as.AddAssignWith.impl.Op.type.defb93.1 = name_ref Op, %.loc31.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.8b7759.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc30, %Op.ref.loc31
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc31, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.df00b6.2]
// CHECK:STDOUT:   %bound_method.loc31.3: <bound method> = bound_method %a.ref.loc30, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc31.3(%a.ref.loc30, %.loc32)
// CHECK:STDOUT:   %a.ref.loc34: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc36: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc35: %.052 = impl_witness_access constants.%SubAssignWith.impl_witness.631, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.31eaa0.2]
// CHECK:STDOUT:   %bound_method.loc35.1: <bound method> = bound_method %a.ref.loc34, %impl.elem0.loc35
// CHECK:STDOUT:   %ImplicitAs.facet.loc35: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc35.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc35 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.864a67.1]
// CHECK:STDOUT:   %bound_method.loc35.2: <bound method> = bound_method %a.ref.loc34, %specific_fn.loc35
// CHECK:STDOUT:   %.loc36: %Cpp.long = acquire_value %b.ref.loc36
// CHECK:STDOUT:   %.loc35.2: %Cpp.long.as.SubAssignWith.impl.Op.type.7e399e.1 = specific_constant imports.%Core.Op.739, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.31eaa0.1]
// CHECK:STDOUT:   %Op.ref.loc35: %Cpp.long.as.SubAssignWith.impl.Op.type.7e399e.1 = name_ref Op, %.loc35.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.31eaa0.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc34, %Op.ref.loc35
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc35, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.864a67.2]
// CHECK:STDOUT:   %bound_method.loc35.3: <bound method> = bound_method %a.ref.loc34, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc35.3(%a.ref.loc34, %.loc36)
// CHECK:STDOUT:   %a.ref.loc38: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc40: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc39: %.8d7 = impl_witness_access constants.%MulAssignWith.impl_witness.0b4, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.b067dc.2]
// CHECK:STDOUT:   %bound_method.loc39.1: <bound method> = bound_method %a.ref.loc38, %impl.elem0.loc39
// CHECK:STDOUT:   %ImplicitAs.facet.loc39: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc39.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc39 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc39: <specific function> = specific_function %impl.elem0.loc39, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.a37438.1]
// CHECK:STDOUT:   %bound_method.loc39.2: <bound method> = bound_method %a.ref.loc38, %specific_fn.loc39
// CHECK:STDOUT:   %.loc40: %Cpp.long = acquire_value %b.ref.loc40
// CHECK:STDOUT:   %.loc39.2: %Cpp.long.as.MulAssignWith.impl.Op.type.c3673e.1 = specific_constant imports.%Core.Op.3b8, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.b067dc.1]
// CHECK:STDOUT:   %Op.ref.loc39: %Cpp.long.as.MulAssignWith.impl.Op.type.c3673e.1 = name_ref Op, %.loc39.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.b067dc.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc38, %Op.ref.loc39
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc39, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.a37438.2]
// CHECK:STDOUT:   %bound_method.loc39.3: <bound method> = bound_method %a.ref.loc38, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc39.3(%a.ref.loc38, %.loc40)
// CHECK:STDOUT:   %a.ref.loc42: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc44: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc43: %.1d5 = impl_witness_access constants.%DivAssignWith.impl_witness.b4e, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.b85335.2]
// CHECK:STDOUT:   %bound_method.loc43.1: <bound method> = bound_method %a.ref.loc42, %impl.elem0.loc43
// CHECK:STDOUT:   %ImplicitAs.facet.loc43: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc43.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc43 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc43: <specific function> = specific_function %impl.elem0.loc43, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.19b8da.1]
// CHECK:STDOUT:   %bound_method.loc43.2: <bound method> = bound_method %a.ref.loc42, %specific_fn.loc43
// CHECK:STDOUT:   %.loc44: %Cpp.long = acquire_value %b.ref.loc44
// CHECK:STDOUT:   %.loc43.2: %Cpp.long.as.DivAssignWith.impl.Op.type.97cfb4.1 = specific_constant imports.%Core.Op.f47, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.b85335.1]
// CHECK:STDOUT:   %Op.ref.loc43: %Cpp.long.as.DivAssignWith.impl.Op.type.97cfb4.1 = name_ref Op, %.loc43.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.b85335.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc42, %Op.ref.loc43
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc43, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.19b8da.2]
// CHECK:STDOUT:   %bound_method.loc43.3: <bound method> = bound_method %a.ref.loc42, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc43.3(%a.ref.loc42, %.loc44)
// CHECK:STDOUT:   %a.ref.loc46: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc48: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc47: %.474 = impl_witness_access constants.%ModAssignWith.impl_witness.faa, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.da4005.2]
// CHECK:STDOUT:   %bound_method.loc47.1: <bound method> = bound_method %a.ref.loc46, %impl.elem0.loc47
// CHECK:STDOUT:   %ImplicitAs.facet.loc47: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc47.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc47 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc47: <specific function> = specific_function %impl.elem0.loc47, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.51594a.1]
// CHECK:STDOUT:   %bound_method.loc47.2: <bound method> = bound_method %a.ref.loc46, %specific_fn.loc47
// CHECK:STDOUT:   %.loc48: %Cpp.long = acquire_value %b.ref.loc48
// CHECK:STDOUT:   %.loc47.2: %Cpp.long.as.ModAssignWith.impl.Op.type.24a130.1 = specific_constant imports.%Core.Op.b37, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.da4005.1]
// CHECK:STDOUT:   %Op.ref.loc47: %Cpp.long.as.ModAssignWith.impl.Op.type.24a130.1 = name_ref Op, %.loc47.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.da4005.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc46, %Op.ref.loc47
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc47, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.51594a.2]
// CHECK:STDOUT:   %bound_method.loc47.3: <bound method> = bound_method %a.ref.loc46, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc47.3(%a.ref.loc46, %.loc48)
// CHECK:STDOUT:   %a.ref.loc50: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc52: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc51: %.f3c = impl_witness_access constants.%BitAndAssignWith.impl_witness.a49, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.500716.2]
// CHECK:STDOUT:   %bound_method.loc51.1: <bound method> = bound_method %a.ref.loc50, %impl.elem0.loc51
// CHECK:STDOUT:   %ImplicitAs.facet.loc51: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc51.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc51 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc51: <specific function> = specific_function %impl.elem0.loc51, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.1245aa.1]
// CHECK:STDOUT:   %bound_method.loc51.2: <bound method> = bound_method %a.ref.loc50, %specific_fn.loc51
// CHECK:STDOUT:   %.loc52: %Cpp.long = acquire_value %b.ref.loc52
// CHECK:STDOUT:   %.loc51.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.bdaab3.1 = specific_constant imports.%Core.Op.fe8, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.500716.1]
// CHECK:STDOUT:   %Op.ref.loc51: %Cpp.long.as.BitAndAssignWith.impl.Op.type.bdaab3.1 = name_ref Op, %.loc51.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.500716.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc50, %Op.ref.loc51
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc51, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.1245aa.2]
// CHECK:STDOUT:   %bound_method.loc51.3: <bound method> = bound_method %a.ref.loc50, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc51.3(%a.ref.loc50, %.loc52)
// CHECK:STDOUT:   %a.ref.loc54: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc56: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc55: %.2c9 = impl_witness_access constants.%BitOrAssignWith.impl_witness.ab4, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.2]
// CHECK:STDOUT:   %bound_method.loc55.1: <bound method> = bound_method %a.ref.loc54, %impl.elem0.loc55
// CHECK:STDOUT:   %ImplicitAs.facet.loc55: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc55.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc55 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc55: <specific function> = specific_function %impl.elem0.loc55, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.b7967f.1]
// CHECK:STDOUT:   %bound_method.loc55.2: <bound method> = bound_method %a.ref.loc54, %specific_fn.loc55
// CHECK:STDOUT:   %.loc56: %Cpp.long = acquire_value %b.ref.loc56
// CHECK:STDOUT:   %.loc55.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a2b279.1 = specific_constant imports.%Core.Op.5c2, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.1]
// CHECK:STDOUT:   %Op.ref.loc55: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a2b279.1 = name_ref Op, %.loc55.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc54, %Op.ref.loc55
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc55, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.b7967f.2]
// CHECK:STDOUT:   %bound_method.loc55.3: <bound method> = bound_method %a.ref.loc54, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc55.3(%a.ref.loc54, %.loc56)
// CHECK:STDOUT:   %a.ref.loc58: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc60: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc59: %.504 = impl_witness_access constants.%BitXorAssignWith.impl_witness.4d7, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.2]
// CHECK:STDOUT:   %bound_method.loc59.1: <bound method> = bound_method %a.ref.loc58, %impl.elem0.loc59
// CHECK:STDOUT:   %ImplicitAs.facet.loc59: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc59.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc59 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc59: <specific function> = specific_function %impl.elem0.loc59, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.04506a.1]
// CHECK:STDOUT:   %bound_method.loc59.2: <bound method> = bound_method %a.ref.loc58, %specific_fn.loc59
// CHECK:STDOUT:   %.loc60: %Cpp.long = acquire_value %b.ref.loc60
// CHECK:STDOUT:   %.loc59.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6f67bc.1 = specific_constant imports.%Core.Op.caf, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.1]
// CHECK:STDOUT:   %Op.ref.loc59: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6f67bc.1 = name_ref Op, %.loc59.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc58, %Op.ref.loc59
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc59, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.04506a.2]
// CHECK:STDOUT:   %bound_method.loc59.3: <bound method> = bound_method %a.ref.loc58, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc59.3(%a.ref.loc58, %.loc60)
// CHECK:STDOUT:   %a.ref.loc62: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc64: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc63: %.20f = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.550, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.2]
// CHECK:STDOUT:   %bound_method.loc63.1: <bound method> = bound_method %a.ref.loc62, %impl.elem0.loc63
// CHECK:STDOUT:   %ImplicitAs.facet.loc63: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc63.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc63 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc63: <specific function> = specific_function %impl.elem0.loc63, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0946f9.1]
// CHECK:STDOUT:   %bound_method.loc63.2: <bound method> = bound_method %a.ref.loc62, %specific_fn.loc63
// CHECK:STDOUT:   %.loc64: %Cpp.long = acquire_value %b.ref.loc64
// CHECK:STDOUT:   %.loc63.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.e1513f.1 = specific_constant imports.%Core.Op.030, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.1]
// CHECK:STDOUT:   %Op.ref.loc63: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.e1513f.1 = name_ref Op, %.loc63.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc62, %Op.ref.loc63
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc63, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0946f9.2]
// CHECK:STDOUT:   %bound_method.loc63.3: <bound method> = bound_method %a.ref.loc62, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc63.3(%a.ref.loc62, %.loc64)
// CHECK:STDOUT:   %a.ref.loc66: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc68: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc67: %.19a = impl_witness_access constants.%RightShiftAssignWith.impl_witness.df5, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.2]
// CHECK:STDOUT:   %bound_method.loc67.1: <bound method> = bound_method %a.ref.loc66, %impl.elem0.loc67
// CHECK:STDOUT:   %ImplicitAs.facet.loc67: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc67.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc67 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc67: <specific function> = specific_function %impl.elem0.loc67, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1ec453.1]
// CHECK:STDOUT:   %bound_method.loc67.2: <bound method> = bound_method %a.ref.loc66, %specific_fn.loc67
// CHECK:STDOUT:   %.loc68: %Cpp.long = acquire_value %b.ref.loc68
// CHECK:STDOUT:   %.loc67.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.095737.1 = specific_constant imports.%Core.Op.3b5, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.1]
// CHECK:STDOUT:   %Op.ref.loc67: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.095737.1 = name_ref Op, %.loc67.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc66, %Op.ref.loc67
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc67, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1ec453.2]
// CHECK:STDOUT:   %bound_method.loc67.3: <bound method> = bound_method %a.ref.loc66, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc67.3(%a.ref.loc66, %.loc68)
// CHECK:STDOUT:   %Destroy.Op.bound.loc21: <bound method> = bound_method %b.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc21: init %empty_tuple.type = call %Destroy.Op.bound.loc21(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc12: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc12: init %empty_tuple.type = call %Destroy.Op.bound.loc12(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long) = "no_op";
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
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.6a4: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.3d2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.6a4, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d19046.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9f0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.bb8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9f0, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d19046.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.ab0: type = facet_type <@SubAssignWith, @SubAssignWith(%i32)> [concrete]
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
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.463: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i32, %SubAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.432: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.463, %SubAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.dd31df.1: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.22e054.2, @Cpp.long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn.dd31df.2: <specific function> = specific_function %Cpp.long.as.SubAssignWith.impl.Op.22e054.1, @Cpp.long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.743: type = facet_type <@MulAssignWith, @MulAssignWith(%i32)> [concrete]
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
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.7fa: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i32, %MulAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.fbe: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.7fa, %MulAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c127b2.1: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.2, @Cpp.long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c127b2.2: <specific function> = specific_function %Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.1, @Cpp.long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.786: type = facet_type <@DivAssignWith, @DivAssignWith(%i32)> [concrete]
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
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.578: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i32, %DivAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.2e1: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.578, %DivAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ee65e7.1: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.2f8f46.2, @Cpp.long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ee65e7.2: <specific function> = specific_function %Cpp.long.as.DivAssignWith.impl.Op.2f8f46.1, @Cpp.long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.e49: type = facet_type <@ModAssignWith, @ModAssignWith(%i32)> [concrete]
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
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.de5: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i32, %ModAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.c03: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.de5, %ModAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.73e9bb.1: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.f69ad5.2, @Cpp.long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn.73e9bb.2: <specific function> = specific_function %Cpp.long.as.ModAssignWith.impl.Op.f69ad5.1, @Cpp.long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.d65: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i32)> [concrete]
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
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.f71: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i32, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.a81: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.f71, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.560221.1: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.2, @Cpp.long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.560221.2: <specific function> = specific_function %Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.1, @Cpp.long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.6e9: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i32)> [concrete]
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
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.b79: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i32, %BitOrAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.1f8: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.b79, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.665975.1: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.73e223.2, @Cpp.long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.665975.2: <specific function> = specific_function %Cpp.long.as.BitOrAssignWith.impl.Op.73e223.1, @Cpp.long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.ea8: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i32)> [concrete]
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
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.291: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i32, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.9ab: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.291, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.4cd193.1: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.616e38.2, @Cpp.long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.4cd193.2: <specific function> = specific_function %Cpp.long.as.BitXorAssignWith.impl.Op.616e38.1, @Cpp.long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.92b: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i32)> [concrete]
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
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.706: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i32, %LeftShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.424: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.706, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.3bd66d.1: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.2, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.3bd66d.2: <specific function> = specific_function %Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.8ae: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i32)> [concrete]
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
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.e38: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i32, %RightShiftAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.eaa: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.e38, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c3db.1: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.2, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c3db.2: <specific function> = specific_function %Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc214, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.d0e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1) = import_ref Core//prelude/types/cpp/int, loc4551, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.07a = impl_witness_table (%Core.import_ref.d0e), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f81: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc300, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ad = impl_witness_table (%Core.import_ref.4fa), @i32.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.bcb: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.1) = import_ref Core//prelude/types/cpp/int, loc4589, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c39240.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.f4b = impl_witness_table (%Core.import_ref.bcb), @Cpp.long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.739: @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long.as.SubAssignWith.impl.Op.type.1f2d1c.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.SubAssignWith.impl.%Cpp.long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long.as.SubAssignWith.impl.Op.c39240.2)]
// CHECK:STDOUT:   %Core.import_ref.c5b: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.1) = import_ref Core//prelude/types/cpp/int, loc4627, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long.as.MulAssignWith.impl.Op.421374.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.8cc = impl_witness_table (%Core.import_ref.c5b), @Cpp.long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3b8: @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long.as.MulAssignWith.impl.Op.type.afa7f7.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.MulAssignWith.impl.%Cpp.long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long.as.MulAssignWith.impl.Op.421374.2)]
// CHECK:STDOUT:   %Core.import_ref.84d: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.1) = import_ref Core//prelude/types/cpp/int, loc4665, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long.as.DivAssignWith.impl.Op.6227df.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.6f7 = impl_witness_table (%Core.import_ref.84d), @Cpp.long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f47: @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long.as.DivAssignWith.impl.Op.type.f3536d.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.DivAssignWith.impl.%Cpp.long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long.as.DivAssignWith.impl.Op.6227df.2)]
// CHECK:STDOUT:   %Core.import_ref.e98: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.1) = import_ref Core//prelude/types/cpp/int, loc4703, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long.as.ModAssignWith.impl.Op.666e79.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.a2a = impl_witness_table (%Core.import_ref.e98), @Cpp.long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b37: @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long.as.ModAssignWith.impl.Op.type.2d6972.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.ModAssignWith.impl.%Cpp.long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long.as.ModAssignWith.impl.Op.666e79.2)]
// CHECK:STDOUT:   %Core.import_ref.325: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.1) = import_ref Core//prelude/types/cpp/int, loc4931, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.d99897.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.cc9 = impl_witness_table (%Core.import_ref.325), @Cpp.long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.fe8: @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long.as.BitAndAssignWith.impl.Op.type.442dcf.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitAndAssignWith.impl.%Cpp.long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitAndAssignWith.impl.Op.d99897.2)]
// CHECK:STDOUT:   %Core.import_ref.01a: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.1) = import_ref Core//prelude/types/cpp/int, loc4969, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.54c = impl_witness_table (%Core.import_ref.01a), @Cpp.long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.5c2: @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long.as.BitOrAssignWith.impl.Op.type.219f14.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitOrAssignWith.impl.%Cpp.long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitOrAssignWith.impl.Op.e5afa1.2)]
// CHECK:STDOUT:   %Core.import_ref.c8f: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.1) = import_ref Core//prelude/types/cpp/int, loc5007, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.a39 = impl_witness_table (%Core.import_ref.c8f), @Cpp.long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.caf: @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long.as.BitXorAssignWith.impl.Op.type.8b07a5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.BitXorAssignWith.impl.%Cpp.long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long.as.BitXorAssignWith.impl.Op.3c4d0d.2)]
// CHECK:STDOUT:   %Core.import_ref.bcc: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.1) = import_ref Core//prelude/types/cpp/int, loc5045, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.74d = impl_witness_table (%Core.import_ref.bcc), @Cpp.long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.030: @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long.as.LeftShiftAssignWith.impl.Op.type.21420b.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.LeftShiftAssignWith.impl.%Cpp.long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e204a3.2)]
// CHECK:STDOUT:   %Core.import_ref.5f5: @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long.as.RightShiftAssignWith.impl.Op.type.f850d3.1) = import_ref Core//prelude/types/cpp/int, loc5083, loaded [symbolic = @Cpp.long.as.RightShiftAssignWith.impl.%Cpp.long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.328772.1)]
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
// CHECK:STDOUT:   %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc19: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %int_1.loc26, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long = call %bound_method.loc19(%int_1.loc26) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc19: init %Cpp.long = converted %int_1.loc26, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc19
// CHECK:STDOUT:   %.loc23: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc28: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc30: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc29.1: %.3d2 = impl_witness_access constants.%AddAssignWith.impl_witness.ca1, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.2]
// CHECK:STDOUT:   %bound_method.loc29.1: <bound method> = bound_method %a.ref.loc28, %impl.elem0.loc29.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc29: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc29.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc29 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.1]
// CHECK:STDOUT:   %bound_method.loc29.2: <bound method> = bound_method %a.ref.loc28, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29.2: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Op.ref.loc29: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = name_ref Op, %.loc29.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc28, %Op.ref.loc29
// CHECK:STDOUT:   %impl.elem0.loc29.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29.3: <bound method> = bound_method %b.ref.loc30, %impl.elem0.loc29.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29: init %Cpp.long = call %bound_method.loc29.3(%b.ref.loc30)
// CHECK:STDOUT:   %.loc29.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29
// CHECK:STDOUT:   %.loc29.4: %Cpp.long = converted %b.ref.loc30, %.loc29.3
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc29, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.2]
// CHECK:STDOUT:   %bound_method.loc29.4: <bound method> = bound_method %a.ref.loc28, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc30: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc30: <bound method> = bound_method %b.ref.loc30, %impl.elem0.loc30
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc30: init %Cpp.long = call %bound_method.loc30(%b.ref.loc30)
// CHECK:STDOUT:   %.loc30.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc30
// CHECK:STDOUT:   %.loc30.2: %Cpp.long = converted %b.ref.loc30, %.loc30.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc29.4(%a.ref.loc28, %.loc30.2)
// CHECK:STDOUT:   %a.ref.loc32: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc34: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc33.1: %.432 = impl_witness_access constants.%SubAssignWith.impl_witness.200, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.22e054.2]
// CHECK:STDOUT:   %bound_method.loc33.1: <bound method> = bound_method %a.ref.loc32, %impl.elem0.loc33.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc33: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc33.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc33 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc33: <specific function> = specific_function %impl.elem0.loc33.1, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.dd31df.1]
// CHECK:STDOUT:   %bound_method.loc33.2: <bound method> = bound_method %a.ref.loc32, %specific_fn.loc33
// CHECK:STDOUT:   %.loc33.2: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cce2.1 = specific_constant imports.%Core.Op.739, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.22e054.1]
// CHECK:STDOUT:   %Op.ref.loc33: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cce2.1 = name_ref Op, %.loc33.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.22e054.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc32, %Op.ref.loc33
// CHECK:STDOUT:   %impl.elem0.loc33.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc33.3: <bound method> = bound_method %b.ref.loc34, %impl.elem0.loc33.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc33: init %Cpp.long = call %bound_method.loc33.3(%b.ref.loc34)
// CHECK:STDOUT:   %.loc33.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc33
// CHECK:STDOUT:   %.loc33.4: %Cpp.long = converted %b.ref.loc34, %.loc33.3
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc33, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.dd31df.2]
// CHECK:STDOUT:   %bound_method.loc33.4: <bound method> = bound_method %a.ref.loc32, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc34: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc34: <bound method> = bound_method %b.ref.loc34, %impl.elem0.loc34
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc34: init %Cpp.long = call %bound_method.loc34(%b.ref.loc34)
// CHECK:STDOUT:   %.loc34.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc34
// CHECK:STDOUT:   %.loc34.2: %Cpp.long = converted %b.ref.loc34, %.loc34.1
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc33.4(%a.ref.loc32, %.loc34.2)
// CHECK:STDOUT:   %a.ref.loc36: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc38: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc37.1: %.fbe = impl_witness_access constants.%MulAssignWith.impl_witness.c16, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.2]
// CHECK:STDOUT:   %bound_method.loc37.1: <bound method> = bound_method %a.ref.loc36, %impl.elem0.loc37.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc37: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc37.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc37 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc37: <specific function> = specific_function %impl.elem0.loc37.1, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c127b2.1]
// CHECK:STDOUT:   %bound_method.loc37.2: <bound method> = bound_method %a.ref.loc36, %specific_fn.loc37
// CHECK:STDOUT:   %.loc37.2: %Cpp.long.as.MulAssignWith.impl.Op.type.d2f0ca.1 = specific_constant imports.%Core.Op.3b8, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.1]
// CHECK:STDOUT:   %Op.ref.loc37: %Cpp.long.as.MulAssignWith.impl.Op.type.d2f0ca.1 = name_ref Op, %.loc37.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc36, %Op.ref.loc37
// CHECK:STDOUT:   %impl.elem0.loc37.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc37.3: <bound method> = bound_method %b.ref.loc38, %impl.elem0.loc37.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc37: init %Cpp.long = call %bound_method.loc37.3(%b.ref.loc38)
// CHECK:STDOUT:   %.loc37.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc37
// CHECK:STDOUT:   %.loc37.4: %Cpp.long = converted %b.ref.loc38, %.loc37.3
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc37, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c127b2.2]
// CHECK:STDOUT:   %bound_method.loc37.4: <bound method> = bound_method %a.ref.loc36, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc38: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc38: <bound method> = bound_method %b.ref.loc38, %impl.elem0.loc38
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc38: init %Cpp.long = call %bound_method.loc38(%b.ref.loc38)
// CHECK:STDOUT:   %.loc38.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc38
// CHECK:STDOUT:   %.loc38.2: %Cpp.long = converted %b.ref.loc38, %.loc38.1
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc37.4(%a.ref.loc36, %.loc38.2)
// CHECK:STDOUT:   %a.ref.loc40: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc42: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc41.1: %.2e1 = impl_witness_access constants.%DivAssignWith.impl_witness.b4b, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2f8f46.2]
// CHECK:STDOUT:   %bound_method.loc41.1: <bound method> = bound_method %a.ref.loc40, %impl.elem0.loc41.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc41: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc41.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc41 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc41: <specific function> = specific_function %impl.elem0.loc41.1, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ee65e7.1]
// CHECK:STDOUT:   %bound_method.loc41.2: <bound method> = bound_method %a.ref.loc40, %specific_fn.loc41
// CHECK:STDOUT:   %.loc41.2: %Cpp.long.as.DivAssignWith.impl.Op.type.16271d.1 = specific_constant imports.%Core.Op.f47, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2f8f46.1]
// CHECK:STDOUT:   %Op.ref.loc41: %Cpp.long.as.DivAssignWith.impl.Op.type.16271d.1 = name_ref Op, %.loc41.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2f8f46.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc40, %Op.ref.loc41
// CHECK:STDOUT:   %impl.elem0.loc41.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc41.3: <bound method> = bound_method %b.ref.loc42, %impl.elem0.loc41.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc41: init %Cpp.long = call %bound_method.loc41.3(%b.ref.loc42)
// CHECK:STDOUT:   %.loc41.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc41
// CHECK:STDOUT:   %.loc41.4: %Cpp.long = converted %b.ref.loc42, %.loc41.3
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc41, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ee65e7.2]
// CHECK:STDOUT:   %bound_method.loc41.4: <bound method> = bound_method %a.ref.loc40, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc42: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc42: <bound method> = bound_method %b.ref.loc42, %impl.elem0.loc42
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc42: init %Cpp.long = call %bound_method.loc42(%b.ref.loc42)
// CHECK:STDOUT:   %.loc42.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc42
// CHECK:STDOUT:   %.loc42.2: %Cpp.long = converted %b.ref.loc42, %.loc42.1
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc41.4(%a.ref.loc40, %.loc42.2)
// CHECK:STDOUT:   %a.ref.loc44: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc46: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc45.1: %.c03 = impl_witness_access constants.%ModAssignWith.impl_witness.a13, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.f69ad5.2]
// CHECK:STDOUT:   %bound_method.loc45.1: <bound method> = bound_method %a.ref.loc44, %impl.elem0.loc45.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc45: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc45.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc45 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc45: <specific function> = specific_function %impl.elem0.loc45.1, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.73e9bb.1]
// CHECK:STDOUT:   %bound_method.loc45.2: <bound method> = bound_method %a.ref.loc44, %specific_fn.loc45
// CHECK:STDOUT:   %.loc45.2: %Cpp.long.as.ModAssignWith.impl.Op.type.e2a2e6.1 = specific_constant imports.%Core.Op.b37, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.f69ad5.1]
// CHECK:STDOUT:   %Op.ref.loc45: %Cpp.long.as.ModAssignWith.impl.Op.type.e2a2e6.1 = name_ref Op, %.loc45.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.f69ad5.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc44, %Op.ref.loc45
// CHECK:STDOUT:   %impl.elem0.loc45.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc45.3: <bound method> = bound_method %b.ref.loc46, %impl.elem0.loc45.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc45: init %Cpp.long = call %bound_method.loc45.3(%b.ref.loc46)
// CHECK:STDOUT:   %.loc45.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc45
// CHECK:STDOUT:   %.loc45.4: %Cpp.long = converted %b.ref.loc46, %.loc45.3
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc45, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.73e9bb.2]
// CHECK:STDOUT:   %bound_method.loc45.4: <bound method> = bound_method %a.ref.loc44, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc46: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc46: <bound method> = bound_method %b.ref.loc46, %impl.elem0.loc46
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc46: init %Cpp.long = call %bound_method.loc46(%b.ref.loc46)
// CHECK:STDOUT:   %.loc46.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc46
// CHECK:STDOUT:   %.loc46.2: %Cpp.long = converted %b.ref.loc46, %.loc46.1
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc45.4(%a.ref.loc44, %.loc46.2)
// CHECK:STDOUT:   %a.ref.loc48: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc50: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc49.1: %.a81 = impl_witness_access constants.%BitAndAssignWith.impl_witness.349, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.2]
// CHECK:STDOUT:   %bound_method.loc49.1: <bound method> = bound_method %a.ref.loc48, %impl.elem0.loc49.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc49: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc49.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc49 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc49: <specific function> = specific_function %impl.elem0.loc49.1, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.560221.1]
// CHECK:STDOUT:   %bound_method.loc49.2: <bound method> = bound_method %a.ref.loc48, %specific_fn.loc49
// CHECK:STDOUT:   %.loc49.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.aa6f07.1 = specific_constant imports.%Core.Op.fe8, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.1]
// CHECK:STDOUT:   %Op.ref.loc49: %Cpp.long.as.BitAndAssignWith.impl.Op.type.aa6f07.1 = name_ref Op, %.loc49.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc48, %Op.ref.loc49
// CHECK:STDOUT:   %impl.elem0.loc49.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc49.3: <bound method> = bound_method %b.ref.loc50, %impl.elem0.loc49.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc49: init %Cpp.long = call %bound_method.loc49.3(%b.ref.loc50)
// CHECK:STDOUT:   %.loc49.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc49
// CHECK:STDOUT:   %.loc49.4: %Cpp.long = converted %b.ref.loc50, %.loc49.3
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc49, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.560221.2]
// CHECK:STDOUT:   %bound_method.loc49.4: <bound method> = bound_method %a.ref.loc48, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc50: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc50: <bound method> = bound_method %b.ref.loc50, %impl.elem0.loc50
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc50: init %Cpp.long = call %bound_method.loc50(%b.ref.loc50)
// CHECK:STDOUT:   %.loc50.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc50
// CHECK:STDOUT:   %.loc50.2: %Cpp.long = converted %b.ref.loc50, %.loc50.1
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc49.4(%a.ref.loc48, %.loc50.2)
// CHECK:STDOUT:   %a.ref.loc52: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc54: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc53.1: %.1f8 = impl_witness_access constants.%BitOrAssignWith.impl_witness.119, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.73e223.2]
// CHECK:STDOUT:   %bound_method.loc53.1: <bound method> = bound_method %a.ref.loc52, %impl.elem0.loc53.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc53: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc53.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc53 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc53: <specific function> = specific_function %impl.elem0.loc53.1, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.665975.1]
// CHECK:STDOUT:   %bound_method.loc53.2: <bound method> = bound_method %a.ref.loc52, %specific_fn.loc53
// CHECK:STDOUT:   %.loc53.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fac636.1 = specific_constant imports.%Core.Op.5c2, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.73e223.1]
// CHECK:STDOUT:   %Op.ref.loc53: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fac636.1 = name_ref Op, %.loc53.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.73e223.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc52, %Op.ref.loc53
// CHECK:STDOUT:   %impl.elem0.loc53.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc53.3: <bound method> = bound_method %b.ref.loc54, %impl.elem0.loc53.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc53: init %Cpp.long = call %bound_method.loc53.3(%b.ref.loc54)
// CHECK:STDOUT:   %.loc53.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc53
// CHECK:STDOUT:   %.loc53.4: %Cpp.long = converted %b.ref.loc54, %.loc53.3
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc53, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.665975.2]
// CHECK:STDOUT:   %bound_method.loc53.4: <bound method> = bound_method %a.ref.loc52, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc54: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc54: <bound method> = bound_method %b.ref.loc54, %impl.elem0.loc54
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc54: init %Cpp.long = call %bound_method.loc54(%b.ref.loc54)
// CHECK:STDOUT:   %.loc54.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc54
// CHECK:STDOUT:   %.loc54.2: %Cpp.long = converted %b.ref.loc54, %.loc54.1
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc53.4(%a.ref.loc52, %.loc54.2)
// CHECK:STDOUT:   %a.ref.loc56: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc58: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc57.1: %.9ab = impl_witness_access constants.%BitXorAssignWith.impl_witness.5aa, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.616e38.2]
// CHECK:STDOUT:   %bound_method.loc57.1: <bound method> = bound_method %a.ref.loc56, %impl.elem0.loc57.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc57: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc57.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc57 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc57: <specific function> = specific_function %impl.elem0.loc57.1, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.4cd193.1]
// CHECK:STDOUT:   %bound_method.loc57.2: <bound method> = bound_method %a.ref.loc56, %specific_fn.loc57
// CHECK:STDOUT:   %.loc57.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.53faad.1 = specific_constant imports.%Core.Op.caf, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.616e38.1]
// CHECK:STDOUT:   %Op.ref.loc57: %Cpp.long.as.BitXorAssignWith.impl.Op.type.53faad.1 = name_ref Op, %.loc57.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.616e38.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc56, %Op.ref.loc57
// CHECK:STDOUT:   %impl.elem0.loc57.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc57.3: <bound method> = bound_method %b.ref.loc58, %impl.elem0.loc57.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc57: init %Cpp.long = call %bound_method.loc57.3(%b.ref.loc58)
// CHECK:STDOUT:   %.loc57.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc57
// CHECK:STDOUT:   %.loc57.4: %Cpp.long = converted %b.ref.loc58, %.loc57.3
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc57, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.4cd193.2]
// CHECK:STDOUT:   %bound_method.loc57.4: <bound method> = bound_method %a.ref.loc56, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc58: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc58: <bound method> = bound_method %b.ref.loc58, %impl.elem0.loc58
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc58: init %Cpp.long = call %bound_method.loc58(%b.ref.loc58)
// CHECK:STDOUT:   %.loc58.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc58
// CHECK:STDOUT:   %.loc58.2: %Cpp.long = converted %b.ref.loc58, %.loc58.1
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc57.4(%a.ref.loc56, %.loc58.2)
// CHECK:STDOUT:   %a.ref.loc60: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc62: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc61.1: %.424 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.199, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.2]
// CHECK:STDOUT:   %bound_method.loc61.1: <bound method> = bound_method %a.ref.loc60, %impl.elem0.loc61.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc61: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc61.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc61 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc61: <specific function> = specific_function %impl.elem0.loc61.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.3bd66d.1]
// CHECK:STDOUT:   %bound_method.loc61.2: <bound method> = bound_method %a.ref.loc60, %specific_fn.loc61
// CHECK:STDOUT:   %.loc61.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.426a0e.1 = specific_constant imports.%Core.Op.030, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.1]
// CHECK:STDOUT:   %Op.ref.loc61: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.426a0e.1 = name_ref Op, %.loc61.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc60, %Op.ref.loc61
// CHECK:STDOUT:   %impl.elem0.loc61.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc61.3: <bound method> = bound_method %b.ref.loc62, %impl.elem0.loc61.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc61: init %Cpp.long = call %bound_method.loc61.3(%b.ref.loc62)
// CHECK:STDOUT:   %.loc61.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc61
// CHECK:STDOUT:   %.loc61.4: %Cpp.long = converted %b.ref.loc62, %.loc61.3
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc61, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.3bd66d.2]
// CHECK:STDOUT:   %bound_method.loc61.4: <bound method> = bound_method %a.ref.loc60, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc62: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc62: <bound method> = bound_method %b.ref.loc62, %impl.elem0.loc62
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc62: init %Cpp.long = call %bound_method.loc62(%b.ref.loc62)
// CHECK:STDOUT:   %.loc62.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc62
// CHECK:STDOUT:   %.loc62.2: %Cpp.long = converted %b.ref.loc62, %.loc62.1
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc61.4(%a.ref.loc60, %.loc62.2)
// CHECK:STDOUT:   %a.ref.loc64: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc66: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc65.1: %.eaa = impl_witness_access constants.%RightShiftAssignWith.impl_witness.e4e, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.2]
// CHECK:STDOUT:   %bound_method.loc65.1: <bound method> = bound_method %a.ref.loc64, %impl.elem0.loc65.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc65: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc65.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc65 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc65: <specific function> = specific_function %impl.elem0.loc65.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c3db.1]
// CHECK:STDOUT:   %bound_method.loc65.2: <bound method> = bound_method %a.ref.loc64, %specific_fn.loc65
// CHECK:STDOUT:   %.loc65.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2dfa9.1 = specific_constant imports.%Core.Op.3b5, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.1]
// CHECK:STDOUT:   %Op.ref.loc65: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2dfa9.1 = name_ref Op, %.loc65.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc64, %Op.ref.loc65
// CHECK:STDOUT:   %impl.elem0.loc65.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc65.3: <bound method> = bound_method %b.ref.loc66, %impl.elem0.loc65.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc65: init %Cpp.long = call %bound_method.loc65.3(%b.ref.loc66)
// CHECK:STDOUT:   %.loc65.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc65
// CHECK:STDOUT:   %.loc65.4: %Cpp.long = converted %b.ref.loc66, %.loc65.3
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc65, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c3db.2]
// CHECK:STDOUT:   %bound_method.loc65.4: <bound method> = bound_method %a.ref.loc64, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc66: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc66: <bound method> = bound_method %b.ref.loc66, %impl.elem0.loc66
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc66: init %Cpp.long = call %bound_method.loc66(%b.ref.loc66)
// CHECK:STDOUT:   %.loc66.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc66
// CHECK:STDOUT:   %.loc66.2: %Cpp.long = converted %b.ref.loc66, %.loc66.1
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc65.4(%a.ref.loc64, %.loc66.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_and_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
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
// CHECK:STDOUT:   %AddAssignWith.type.761: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
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
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.e1c: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.2e2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.e1c, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3a3075.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.5fb128.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3a3075.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.5fb128.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc214, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.d0e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1) = import_ref Core//prelude/types/cpp/int, loc4551, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.1)]
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
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc12: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc12: init %Cpp.long = converted %int_1.loc19, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc12
// CHECK:STDOUT:   %.loc16: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc22.1: %.2e2 = impl_witness_access constants.%AddAssignWith.impl_witness.623, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.5fb128.2]
// CHECK:STDOUT:   %bound_method.loc22.1: <bound method> = bound_method %a.ref, %impl.elem0.loc22.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc22.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc22.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3a3075.1]
// CHECK:STDOUT:   %bound_method.loc22.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc22.2: %Cpp.long.as.AddAssignWith.impl.Op.type.a474b7.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.5fb128.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.a474b7.1 = name_ref Op, %.loc22.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.5fb128.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc22.2: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc22.3: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc22.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long = call %bound_method.loc22.3(%int_1.loc23) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc22.4: %Cpp.long = converted %int_1.loc23, %.loc22.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3a3075.2]
// CHECK:STDOUT:   %bound_method.loc22.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc23: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %Cpp.long = call %bound_method.loc23(%int_1.loc23) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc23.2: %Cpp.long = converted %int_1.loc23, %.loc23.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc22.4(%a.ref, %.loc23.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long) = "no_op";
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
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.6a4: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.3d2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.6a4, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.1: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d19046.2, @Cpp.long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9f0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long, %ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %.bb8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9f0, %ImplicitAs.facet.174 [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert: %i32.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.2: <specific function> = specific_function %Cpp.long.as.AddAssignWith.impl.Op.d19046.1, @Cpp.long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.174) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b38 = import_ref Core//prelude/types/cpp/int, loc214, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc141, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b2d.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl.b2d [concrete]
// CHECK:STDOUT:   %Core.import_ref.d0e: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.1) = import_ref Core//prelude/types/cpp/int, loc4551, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.07a = impl_witness_table (%Core.import_ref.d0e), @Cpp.long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.f81: @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long.as.AddAssignWith.impl.Op.type.9f3dfb.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long.as.AddAssignWith.impl.%Cpp.long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long.as.AddAssignWith.impl.Op.78f138.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: %i32.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc300, loaded [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc12: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long = call %bound_method.loc12(%int_1.loc19) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc12: init %Cpp.long = converted %int_1.loc19, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc12
// CHECK:STDOUT:   %.loc16: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc26: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc26: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc26.1: <bound method> = bound_method %int_1.loc26, %impl.elem0.loc26 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem0.loc26, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26.2: <bound method> = bound_method %int_1.loc26, %specific_fn.loc26 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26: init %i32 = call %bound_method.loc26.2(%int_1.loc26) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc26.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc26.2: %i32 = converted %int_1.loc26, %.loc26.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %b: %i32 = value_binding b, %.loc26.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc29.1: %.3d2 = impl_witness_access constants.%AddAssignWith.impl_witness.ca1, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.2]
// CHECK:STDOUT:   %bound_method.loc29.1: <bound method> = bound_method %a.ref, %impl.elem0.loc29.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc29.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.1]
// CHECK:STDOUT:   %bound_method.loc29.2: <bound method> = bound_method %a.ref, %specific_fn.loc29
// CHECK:STDOUT:   %.loc29.2: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = name_ref Op, %.loc29.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc29.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29.3: <bound method> = bound_method %b.ref, %impl.elem0.loc29.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc29: init %Cpp.long = call %bound_method.loc29.3(%b.ref)
// CHECK:STDOUT:   %.loc29.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc29
// CHECK:STDOUT:   %.loc29.4: %Cpp.long = converted %b.ref, %.loc29.3
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.2]
// CHECK:STDOUT:   %bound_method.loc29.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc30: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc30: <bound method> = bound_method %b.ref, %impl.elem0.loc30
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc30: init %Cpp.long = call %bound_method.loc30(%b.ref)
// CHECK:STDOUT:   %.loc30.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc30
// CHECK:STDOUT:   %.loc30.2: %Cpp.long = converted %b.ref, %.loc30.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc29.4(%a.ref, %.loc30.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- increment_decrement_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = constants.%Cpp.long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b8a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc214, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.903 = impl_witness_table (%Core.import_ref.b8a), @Core.IntLiteral.as.ImplicitAs.impl.052 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9d7: %Cpp.long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc5324, loaded [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.9d7), @Cpp.long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d5d: %Cpp.long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc5294, loaded [concrete = constants.%Cpp.long.as.Dec.impl.Op]
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
// CHECK:STDOUT:   %impl.elem0.loc12: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %int_1, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc12(%int_1) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc12: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc12
// CHECK:STDOUT:   %.loc16: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc22: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc21: %.314 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %a.ref.loc22, %impl.elem0.loc21
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc21(%a.ref.loc22)
// CHECK:STDOUT:   %a.ref.loc25: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc24: %.253 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %a.ref.loc25, %impl.elem0.loc24
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc24(%a.ref.loc25)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.long) = "no_op";
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.702: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.71f = import_ref Core//prelude/types/cpp/int, loc355, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.14b]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5b5 = impl_witness_table (%Core.import_ref.702), @Core.IntLiteral.as.ImplicitAs.impl.541 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULong.cpp_overload_set.value: %PassULong.cpp_overload_set.type = cpp_overload_set_value @PassULong.cpp_overload_set [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULong__carbon_thunk.decl.108234.1: %PassULong__carbon_thunk.type.3fc2d8.1 = fn_decl @PassULong__carbon_thunk.1 [concrete = constants.%PassULong__carbon_thunk.ff747d.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.741: @Core.IntLiteral.as.ImplicitAs.impl.c85.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6) = import_ref Core//prelude/types/uint, loc145, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.c85.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.46e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.899 = impl_witness_table (%Core.import_ref.741), @Core.IntLiteral.as.ImplicitAs.impl.c85 [concrete]
// CHECK:STDOUT:   %UIntResult.decl: type = class_decl @UIntResult [concrete = constants.%UIntResult] {} {}
// CHECK:STDOUT:   %PassULong__carbon_thunk.decl.108234.2: %PassULong__carbon_thunk.type.3fc2d8.2 = fn_decl @PassULong__carbon_thunk.2 [concrete = constants.%PassULong__carbon_thunk.ff747d.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.190: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc384, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.f58 = impl_witness_table (%Core.import_ref.190), @Core.IntLiteral.as.As.impl.8c9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.f5e: %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc414, loaded [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.50d = impl_witness_table (%Core.import_ref.f5e), @Cpp.unsigned_long.as.ImplicitAs.impl.78e [concrete]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/types/float, loc145, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31a = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.5b7 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc18: type = splice_block %unsigned_long.ref.loc18 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc18: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc21: %.541 = impl_witness_access constants.%ImplicitAs.impl_witness.a19, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.14b]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.92f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.unsigned_long = call %bound_method.loc21(%int_1.loc21) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %.loc21.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %.loc21.2: %Cpp.unsigned_long = converted %int_1.loc21, %.loc21.1 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %cpp_unsigned_long: %Cpp.unsigned_long = value_binding cpp_unsigned_long, %.loc21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_result.patt: %pattern_type.6f2 = value_binding_pattern cpp_unsigned_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc32: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc35: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc36.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc36: %ptr.f5e = addr_of %.loc36.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc36: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.1(%cpp_unsigned_long.ref.loc35, %addr.loc36)
// CHECK:STDOUT:   %.loc36.2: init %ULongResult to %.loc36.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc36
// CHECK:STDOUT:   %.loc28: type = splice_block %ULongResult.ref.loc28 [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref.loc28: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc36.3: ref %ULongResult = temporary %.loc36.1, %.loc36.2
// CHECK:STDOUT:   %.loc36.4: %ULongResult = acquire_value %.loc36.3
// CHECK:STDOUT:   %cpp_unsigned_long_result: %ULongResult = value_binding cpp_unsigned_long_result, %.loc36.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong.patt: %pattern_type.5b7 = value_binding_pattern cpp_compat_ulong [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc47: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc44: type = splice_block %ULong32.ref [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.4b4 [concrete = imports.%CppCompat.4b4]
// CHECK:STDOUT:     %ULong32.ref: type = name_ref ULong32, imports.%Core.ULong32 [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_ulong: %Cpp.unsigned_long = value_binding cpp_compat_ulong, %cpp_unsigned_long.ref.loc47
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_result.patt: %pattern_type.6f2 = value_binding_pattern cpp_compat_ulong_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc58: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong.ref: %Cpp.unsigned_long = name_ref cpp_compat_ulong, %cpp_compat_ulong
// CHECK:STDOUT:   %.loc62.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc62: %ptr.f5e = addr_of %.loc62.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc62: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.1(%cpp_compat_ulong.ref, %addr.loc62)
// CHECK:STDOUT:   %.loc62.2: init %ULongResult to %.loc62.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc62
// CHECK:STDOUT:   %.loc54: type = splice_block %ULongResult.ref.loc54 [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc53: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref.loc54: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc62.3: ref %ULongResult = temporary %.loc62.1, %.loc62.2
// CHECK:STDOUT:   %.loc62.4: %ULongResult = acquire_value %.loc62.3
// CHECK:STDOUT:   %cpp_compat_ulong_result: %ULongResult = value_binding cpp_compat_ulong_result, %.loc62.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u32.patt: %pattern_type.4a9 = value_binding_pattern carbon_u32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc69: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u32: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %impl.elem0.loc69: %.b47 = impl_witness_access constants.%ImplicitAs.impl_witness.29e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.90e]
// CHECK:STDOUT:   %bound_method.loc69.1: <bound method> = bound_method %int_1.loc69, %impl.elem0.loc69 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.412]
// CHECK:STDOUT:   %specific_fn.loc69: <specific function> = specific_function %impl.elem0.loc69, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc69.2: <bound method> = bound_method %int_1.loc69, %specific_fn.loc69 [concrete = constants.%bound_method.21d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc69: init %u32 = call %bound_method.loc69.2(%int_1.loc69) [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc69.1: %u32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc69 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc69.2: %u32 = converted %int_1.loc69, %.loc69.1 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %carbon_u32: %u32 = value_binding carbon_u32, %.loc69.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u32_result.patt: %pattern_type.dc7 = value_binding_pattern carbon_u32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc79: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc80: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u32.ref: %u32 = name_ref carbon_u32, %carbon_u32
// CHECK:STDOUT:   %.loc84.1: ref %UIntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc84: %ptr.059 = addr_of %.loc84.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc84: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.2(%carbon_u32.ref, %addr.loc84)
// CHECK:STDOUT:   %.loc84.2: init %UIntResult to %.loc84.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc84
// CHECK:STDOUT:   %.loc76: type = splice_block %UIntResult.ref [concrete = constants.%UIntResult] {
// CHECK:STDOUT:     %Cpp.ref.loc75: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %UIntResult.ref: type = name_ref UIntResult, imports.%UIntResult.decl [concrete = constants.%UIntResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc84.3: ref %UIntResult = temporary %.loc84.1, %.loc84.2
// CHECK:STDOUT:   %.loc84.4: %UIntResult = acquire_value %.loc84.3
// CHECK:STDOUT:   %carbon_u32_result: %UIntResult = value_binding carbon_u32_result, %.loc84.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc104.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc104: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc104.1: <bound method> = bound_method %float, %impl.elem0.loc104 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc104: <specific function> = specific_function %impl.elem0.loc104, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc104.2: <bound method> = bound_method %float, %specific_fn.loc104 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc104.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc104.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc104.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc104.4: init %f32.97e to %.loc104.3 = in_place_init %.loc104.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc104.5: init %array_type to %a.var = array_init (%.loc104.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc86: init %array_type = converted %.loc104.1, %.loc104.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc86
// CHECK:STDOUT:   %.loc99: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc94: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc96: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc97: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc95.1: %.59b = impl_witness_access constants.%As.impl_witness.d5f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc95.1: <bound method> = bound_method %int_1.loc94, %impl.elem0.loc95.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc95.1(%int_1.loc94) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %.loc95.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %.loc95.2: %Cpp.unsigned_long = converted %int_1.loc94, %.loc95.1 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %impl.elem0.loc95.2: %.0db = impl_witness_access constants.%ImplicitAs.impl_witness.a13, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc95.2: <bound method> = bound_method %.loc95.2, %impl.elem0.loc95.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc95.2(%.loc95.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc95.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc95.4: Core.IntLiteral = converted %.loc95.2, %.loc95.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc95.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   %UIntResult.cpp_destructor.bound: <bound method> = bound_method %.loc84.3, constants.%UIntResult.cpp_destructor
// CHECK:STDOUT:   %UIntResult.cpp_destructor.call: init %empty_tuple.type = call %UIntResult.cpp_destructor.bound(%.loc84.3)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound.loc62: <bound method> = bound_method %.loc62.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call.loc62: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound.loc62(%.loc62.3)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound.loc36: <bound method> = bound_method %.loc36.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call.loc36: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound.loc36(%.loc36.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_unsigned_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long: type = class_type @ULong32 [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long = constants.%Cpp.unsigned_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.702: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc355, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5b5 = impl_witness_table (%Core.import_ref.702), @Core.IntLiteral.as.ImplicitAs.impl.541 [concrete]
// CHECK:STDOUT:   %Core.import_ref.1e0: %Cpp.unsigned_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc141, loaded [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %impl.elem0.loc12: %.541 = impl_witness_access constants.%ImplicitAs.impl_witness.a19, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %int_1, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc12(%int_1) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %.loc12: init %Cpp.unsigned_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   assign %a.var, %.loc12
// CHECK:STDOUT:   %.loc16: type = splice_block %unsigned_long.ref.loc16 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc16: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.unsigned_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.5b7 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.5b7 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.unsigned_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.unsigned_long = name_ref a, %a
// CHECK:STDOUT:   %.loc29: %Cpp.unsigned_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc29: %.f45 = impl_witness_access constants.%Copy.impl_witness.ae3, element0 [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc29: <bound method> = bound_method %.loc29, %impl.elem0.loc29
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long = call %bound_method.loc29(%.loc29)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc26: type = splice_block %unsigned_long.ref.loc26 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc26: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc21: <bound method> = bound_method %b.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc21: init %empty_tuple.type = call %Destroy.Op.bound.loc21(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc12: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc12: init %empty_tuple.type = call %Destroy.Op.bound.loc12(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.unsigned_long) = "no_op";
// CHECK:STDOUT:
