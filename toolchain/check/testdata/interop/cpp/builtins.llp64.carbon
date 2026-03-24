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
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc22_25: type = splice_block %long_long.ref.loc22 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long_long.ref.loc22: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc22: %.a64 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc22_38.1: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_38.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.288]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc22_38.2(%int_1.loc22) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc22_38.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc22_38.2: %i64 = converted %int_1.loc22, %.loc22_38.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %cpp_long_long: %i64 = value_binding cpp_long_long, %.loc22_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long_long.patt: %pattern_type.95b = value_binding_pattern carbon_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref: %i64 = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %i64.loc23: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long_long: %i64 = value_binding carbon_long_long, %cpp_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc25_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc25_55: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc25_55.1: <bound method> = bound_method %float, %impl.elem0.loc25_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc25_55: <specific function> = specific_function %impl.elem0.loc25_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_55.2: <bound method> = bound_method %float, %specific_fn.loc25_55 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc25_55.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc25_55.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc25_55.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc25_55.4: init %f32.97e to %.loc25_55.3 = in_place_init %.loc25_55.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc25_55.5: init %array_type to %a.var = array_init (%.loc25_55.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc25_3: init %array_type = converted %.loc25_55.1, %.loc25_55.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc25_3
// CHECK:STDOUT:   %.loc25_46: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc25: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc25: type = name_ref long_long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc25_30.1: %.fc7 = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:     %bound_method.loc25_30.1: <bound method> = bound_method %int_1.loc25, %impl.elem0.loc25_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc25_30.1: <specific function> = specific_function %impl.elem0.loc25_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_30.2: <bound method> = bound_method %int_1.loc25, %specific_fn.loc25_30.1 [concrete = constants.%bound_method.41b]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc25_30.2(%int_1.loc25) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc25_30.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %.loc25_30.2: %i64 = converted %int_1.loc25, %.loc25_30.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:     %impl.elem0.loc25_30.2: %.495 = impl_witness_access constants.%ImplicitAs.impl_witness.e86, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.535]
// CHECK:STDOUT:     %bound_method.loc25_30.3: <bound method> = bound_method %.loc25_30.2, %impl.elem0.loc25_30.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc25_30.2: <specific function> = specific_function %impl.elem0.loc25_30.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_30.4: <bound method> = bound_method %.loc25_30.2, %specific_fn.loc25_30.2 [concrete = constants.%bound_method.d1e]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc25_30.4(%.loc25_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc25_30.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc25_30.4: Core.IntLiteral = converted %.loc25_30.2, %.loc25_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc25_30.4, %f32 [concrete = constants.%array_type]
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
// CHECK:STDOUT:   %int_1.loc37: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc37_34: type = splice_block %unsigned_long_long.ref.loc37 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long_long.ref.loc37: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc37: %.f17 = impl_witness_access constants.%ImplicitAs.impl_witness.bb3, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.042]
// CHECK:STDOUT:   %bound_method.loc37_56.1: <bound method> = bound_method %int_1.loc37, %impl.elem0.loc37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc37: <specific function> = specific_function %impl.elem0.loc37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_56.2: <bound method> = bound_method %int_1.loc37, %specific_fn.loc37 [concrete = constants.%bound_method.fec]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc37_56.2(%int_1.loc37) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc37_56.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %.loc37_56.2: %u64 = converted %int_1.loc37, %.loc37_56.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:   %cpp_unsigned_long_long: %u64 = value_binding cpp_unsigned_long_long, %.loc37_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long_long.patt: %pattern_type.157 = value_binding_pattern carbon_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref: %u64 = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %u64.loc38: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long_long: %u64 = value_binding carbon_unsigned_long_long, %cpp_unsigned_long_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc40_64.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc40_64: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc40_64.1: <bound method> = bound_method %float, %impl.elem0.loc40_64 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc40_64: <specific function> = specific_function %impl.elem0.loc40_64, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc40_64.2: <bound method> = bound_method %float, %specific_fn.loc40_64 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc40_64.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc40_64.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc40_64.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc40_64.4: init %f32.97e to %.loc40_64.3 = in_place_init %.loc40_64.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc40_64.5: init %array_type to %a.var = array_init (%.loc40_64.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc40_3: init %array_type = converted %.loc40_64.1, %.loc40_64.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc40_3
// CHECK:STDOUT:   %.loc40_55: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc40: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc40: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc40: type = name_ref unsigned_long_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:     %impl.elem0.loc40_30.1: %.904 = impl_witness_access constants.%As.impl_witness.f54, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.715]
// CHECK:STDOUT:     %bound_method.loc40_30.1: <bound method> = bound_method %int_1.loc40, %impl.elem0.loc40_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc40_30.1: <specific function> = specific_function %impl.elem0.loc40_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc40_30.2: <bound method> = bound_method %int_1.loc40, %specific_fn.loc40_30.1 [concrete = constants.%bound_method.9cf]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc40_30.2(%int_1.loc40) [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc40_30.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %.loc40_30.2: %u64 = converted %int_1.loc40, %.loc40_30.1 [concrete = constants.%int_1.f23]
// CHECK:STDOUT:     %impl.elem0.loc40_30.2: %.74a = impl_witness_access constants.%ImplicitAs.impl_witness.2cc, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.309]
// CHECK:STDOUT:     %bound_method.loc40_30.3: <bound method> = bound_method %.loc40_30.2, %impl.elem0.loc40_30.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc40_30.2: <specific function> = specific_function %impl.elem0.loc40_30.2, @UInt.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc40_30.4: <bound method> = bound_method %.loc40_30.2, %specific_fn.loc40_30.2 [concrete = constants.%bound_method.f18]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc40_30.4(%.loc40_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc40_30.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc40_30.4: Core.IntLiteral = converted %.loc40_30.2, %.loc40_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc40_30.4, %f32 [concrete = constants.%array_type]
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
// CHECK:STDOUT:   %int_1.loc59: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc59_20: type = splice_block %long.ref.loc59 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc59: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc59: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc59: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc59: <bound method> = bound_method %int_1.loc59, %impl.elem0.loc59 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc59(%int_1.loc59) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc59_28.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc59_28.2: %Cpp.long = converted %int_1.loc59, %.loc59_28.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %cpp_long: %Cpp.long = value_binding cpp_long, %.loc59_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_result.patt: %pattern_type.cd9 = value_binding_pattern cpp_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc60_48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc60: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long.ref.loc60: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc60_69.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc60: %ptr.305 = addr_of %.loc60_69.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc60: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.1(%cpp_long.ref.loc60, %addr.loc60)
// CHECK:STDOUT:   %.loc60_69.2: init %LongResult to %.loc60_69.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc60
// CHECK:STDOUT:   %.loc60_34: type = splice_block %LongResult.ref.loc60 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc60_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc60: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc60_69.3: ref %LongResult = temporary %.loc60_69.1, %.loc60_69.2
// CHECK:STDOUT:   %.loc60_69.4: %LongResult = acquire_value %.loc60_69.3
// CHECK:STDOUT:   %cpp_long_result: %LongResult = value_binding cpp_long_result, %.loc60_69.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long.patt: %pattern_type.68c = value_binding_pattern cpp_compat_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc62: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %.loc62: type = splice_block %Long32.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.4b4 [concrete = imports.%CppCompat.4b4]
// CHECK:STDOUT:     %Long32.ref: type = name_ref Long32, imports.%Core.Long32 [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long: %Cpp.long = value_binding cpp_compat_long, %cpp_long.ref.loc62
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_result.patt: %pattern_type.cd9 = value_binding_pattern cpp_compat_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc63_55: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc63: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long.ref: %Cpp.long = name_ref cpp_compat_long, %cpp_compat_long
// CHECK:STDOUT:   %.loc63_83.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc63: %ptr.305 = addr_of %.loc63_83.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc63: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.1(%cpp_compat_long.ref, %addr.loc63)
// CHECK:STDOUT:   %.loc63_83.2: init %LongResult to %.loc63_83.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc63
// CHECK:STDOUT:   %.loc63_41: type = splice_block %LongResult.ref.loc63 [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc63_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref.loc63: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc63_83.3: ref %LongResult = temporary %.loc63_83.1, %.loc63_83.2
// CHECK:STDOUT:   %.loc63_83.4: %LongResult = acquire_value %.loc63_83.3
// CHECK:STDOUT:   %cpp_compat_long_result: %LongResult = value_binding cpp_compat_long_result, %.loc63_83.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32.patt: %pattern_type.7ce = value_binding_pattern carbon_i32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc65: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i32.loc65_37: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc65_34.1: %i32 = as_compatible %cpp_long.ref.loc65
// CHECK:STDOUT:   %.loc65_34.2: %i32 = converted %cpp_long.ref.loc65, %.loc65_34.1
// CHECK:STDOUT:   %i32.loc65_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %carbon_i32: %i32 = value_binding carbon_i32, %.loc65_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i32_result.patt: %pattern_type.454 = value_binding_pattern carbon_i32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc66_49: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLong.ref.loc66: %PassLong.cpp_overload_set.type = name_ref PassLong, imports.%PassLong.cpp_overload_set.value [concrete = constants.%PassLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i32.ref: %i32 = name_ref carbon_i32, %carbon_i32
// CHECK:STDOUT:   %.loc66_72.1: ref %IntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc66: %ptr.3cb = addr_of %.loc66_72.1
// CHECK:STDOUT:   %PassLong__carbon_thunk.call.loc66: init %empty_tuple.type = call imports.%PassLong__carbon_thunk.decl.d16229.2(%carbon_i32.ref, %addr.loc66)
// CHECK:STDOUT:   %.loc66_72.2: init %IntResult to %.loc66_72.1 = mark_in_place_init %PassLong__carbon_thunk.call.loc66
// CHECK:STDOUT:   %.loc66_36: type = splice_block %IntResult.ref [concrete = constants.%IntResult] {
// CHECK:STDOUT:     %Cpp.ref.loc66_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntResult.ref: type = name_ref IntResult, imports.%IntResult.decl [concrete = constants.%IntResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc66_72.3: ref %IntResult = temporary %.loc66_72.1, %.loc66_72.2
// CHECK:STDOUT:   %.loc66_72.4: %IntResult = acquire_value %.loc66_72.3
// CHECK:STDOUT:   %carbon_i32_result: %IntResult = value_binding carbon_i32_result, %.loc66_72.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.95b = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref.loc68: %Cpp.long = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc68: %.12d = impl_witness_access constants.%ImplicitAs.impl_witness.bbc, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bbf]
// CHECK:STDOUT:   %bound_method.loc68: <bound method> = bound_method %cpp_long.ref.loc68, %impl.elem0.loc68
// CHECK:STDOUT:   %Cpp.long.as.ImplicitAs.impl.Convert.call.loc68: init %i64 = call %bound_method.loc68(%cpp_long.ref.loc68)
// CHECK:STDOUT:   %.loc68_32.1: %i64 = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc68
// CHECK:STDOUT:   %.loc68_32.2: %i64 = converted %cpp_long.ref.loc68, %.loc68_32.1
// CHECK:STDOUT:   %carbon_i64: %i64 = value_binding carbon_i64, %.loc68_32.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc70_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc70_50: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc70_50.1: <bound method> = bound_method %float, %impl.elem0.loc70_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc70_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc70_50.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc70_50.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc70_50.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc70_50.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc70_50.4: init %f32.97e to %.loc70_50.3 = in_place_init %.loc70_50.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc70_50.5: init %array_type to %a.var = array_init (%.loc70_50.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc70_3: init %array_type = converted %.loc70_50.1, %.loc70_50.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc70_3
// CHECK:STDOUT:   %.loc70_41: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc70: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc70: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc70: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:     %impl.elem0.loc70_30.1: %.a3d = impl_witness_access constants.%As.impl_witness.60c, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc70_30.1: <bound method> = bound_method %int_1.loc70, %impl.elem0.loc70_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long = call %bound_method.loc70_30.1(%int_1.loc70) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %.loc70_30.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %.loc70_30.2: %Cpp.long = converted %int_1.loc70, %.loc70_30.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:     %impl.elem0.loc70_30.2: %.7f4 = impl_witness_access constants.%ImplicitAs.impl_witness.e4d, element0 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.d49]
// CHECK:STDOUT:     %bound_method.loc70_30.2: <bound method> = bound_method %.loc70_30.2, %impl.elem0.loc70_30.2 [concrete = constants.%Cpp.long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long.as.ImplicitAs.impl.Convert.call.loc70: init Core.IntLiteral = call %bound_method.loc70_30.2(%.loc70_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc70_30.3: Core.IntLiteral = value_of_initializer %Cpp.long.as.ImplicitAs.impl.Convert.call.loc70 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc70_30.4: Core.IntLiteral = converted %.loc70_30.2, %.loc70_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc70_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   %IntResult.cpp_destructor.bound: <bound method> = bound_method %.loc66_72.3, constants.%IntResult.cpp_destructor
// CHECK:STDOUT:   %IntResult.cpp_destructor.call: init %empty_tuple.type = call %IntResult.cpp_destructor.bound(%.loc66_72.3)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound.loc63: <bound method> = bound_method %.loc63_83.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call.loc63: init %empty_tuple.type = call %LongResult.cpp_destructor.bound.loc63(%.loc63_83.3)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound.loc60: <bound method> = bound_method %.loc60_69.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call.loc60: init %empty_tuple.type = call %LongResult.cpp_destructor.bound.loc60(%.loc60_69.3)
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
// CHECK:STDOUT:   %impl.elem0.loc100: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc100: <bound method> = bound_method %int_1, %impl.elem0.loc100 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc100(%int_1) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc100_3: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc100_3
// CHECK:STDOUT:   %.loc100_13: type = splice_block %long.ref.loc100 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc100: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc100: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.68c = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %.loc101_28: %Cpp.long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc101: %.539 = impl_witness_access constants.%Copy.impl_witness.e75, element0 [concrete = constants.%Cpp.long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc101: <bound method> = bound_method %.loc101_28, %impl.elem0.loc101
// CHECK:STDOUT:   %Cpp.long.as.Copy.impl.Op.call: init %Cpp.long = call %bound_method.loc101(%.loc101_28)
// CHECK:STDOUT:   assign %b.var, %Cpp.long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc101_20: type = splice_block %long.ref.loc101 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc101: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc101: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc101: <bound method> = bound_method %b.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc101: init %empty_tuple.type = call %Destroy.Op.bound.loc101(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc100: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc100: init %empty_tuple.type = call %Destroy.Op.bound.loc100(%a.var)
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
// CHECK:STDOUT:   %int_1.loc113: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc113_13: type = splice_block %long.ref.loc113 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc113: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc113: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc113: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc113: <bound method> = bound_method %int_1.loc113, %impl.elem0.loc113 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc113: init %Cpp.long = call %bound_method.loc113(%int_1.loc113) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc113_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc113 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc113_21.2: %Cpp.long = converted %int_1.loc113, %.loc113_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc113_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc114: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc114_13: type = splice_block %long.ref.loc114 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc114: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc114: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc114: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc114: <bound method> = bound_method %int_1.loc114, %impl.elem0.loc114 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc114: init %Cpp.long = call %bound_method.loc114(%int_1.loc114) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc114_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc114 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc114_21.2: %Cpp.long = converted %int_1.loc114, %.loc114_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc114_21.2
// CHECK:STDOUT:   %a.ref.loc115: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc115: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc115: %.909 = impl_witness_access constants.%EqWith.impl_witness.6db, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.1b7]
// CHECK:STDOUT:   %bound_method.loc115: <bound method> = bound_method %a.ref.loc115, %impl.elem0.loc115
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc115(%a.ref.loc115, %b.ref.loc115)
// CHECK:STDOUT:   %a.ref.loc116: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc116: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc116: %.163 = impl_witness_access constants.%EqWith.impl_witness.6db, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.5de]
// CHECK:STDOUT:   %bound_method.loc116: <bound method> = bound_method %a.ref.loc116, %impl.elem1.loc116
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc116(%a.ref.loc116, %b.ref.loc116)
// CHECK:STDOUT:   %a.ref.loc117: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc117: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.fdc = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.cc7]
// CHECK:STDOUT:   %bound_method.loc117: <bound method> = bound_method %a.ref.loc117, %impl.elem2
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc117(%a.ref.loc117, %b.ref.loc117)
// CHECK:STDOUT:   %a.ref.loc118: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc118: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc118: %.c90 = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.911]
// CHECK:STDOUT:   %bound_method.loc118: <bound method> = bound_method %a.ref.loc118, %impl.elem0.loc118
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc118(%a.ref.loc118, %b.ref.loc118)
// CHECK:STDOUT:   %a.ref.loc119: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc119: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.a9b = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.337]
// CHECK:STDOUT:   %bound_method.loc119: <bound method> = bound_method %a.ref.loc119, %impl.elem3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc119(%a.ref.loc119, %b.ref.loc119)
// CHECK:STDOUT:   %a.ref.loc120: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc120: %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc120: %.fad = impl_witness_access constants.%OrderedWith.impl_witness.2b3, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.187]
// CHECK:STDOUT:   %bound_method.loc120: <bound method> = bound_method %a.ref.loc120, %impl.elem1.loc120
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc120(%a.ref.loc120, %b.ref.loc120)
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
// CHECK:STDOUT:   %Core.import_ref.c23: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.4aa4c3.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.7e103a.1)]
// CHECK:STDOUT:   %Core.import_ref.df7: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long.as.OrderedWith.impl.Greater.type.662b24.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long.as.OrderedWith.impl.Greater.02ce94.1)]
// CHECK:STDOUT:   %Core.import_ref.a61: @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.74811b.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long.as.OrderedWith.impl.15a.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.dd28ba.1)]
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
// CHECK:STDOUT:   %int_1.loc133: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc133_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc133: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc133: <bound method> = bound_method %int_1.loc133, %impl.elem0.loc133 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc133: init %Cpp.long = call %bound_method.loc133(%int_1.loc133) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc133_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc133 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc133_21.2: %Cpp.long = converted %int_1.loc133, %.loc133_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc133_21.2
// CHECK:STDOUT:   %a.ref.loc134: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc134: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc134_5.1: %.09a = impl_witness_access constants.%EqWith.impl_witness.15a, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.2]
// CHECK:STDOUT:   %bound_method.loc134_5.1: <bound method> = bound_method %a.ref.loc134, %impl.elem0.loc134_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc134: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc134_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc134 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc134: <specific function> = specific_function %impl.elem0.loc134_5.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.1]
// CHECK:STDOUT:   %bound_method.loc134_5.2: <bound method> = bound_method %a.ref.loc134, %specific_fn.loc134
// CHECK:STDOUT:   %.loc134_5.2: %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.1 = specific_constant imports.%Core.Equal.0ad, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.1]
// CHECK:STDOUT:   %Equal.ref.loc134: %Cpp.long.as.EqWith.impl.Equal.type.c1ed1a.1 = name_ref Equal, %.loc134_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.0f3e3b.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc134: <bound method> = bound_method %a.ref.loc134, %Equal.ref.loc134
// CHECK:STDOUT:   %impl.elem0.loc134_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc134_5.3: <bound method> = bound_method %b.ref.loc134, %impl.elem0.loc134_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc134_5: init %Cpp.long = call %bound_method.loc134_5.3(%b.ref.loc134)
// CHECK:STDOUT:   %.loc134_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc134_5
// CHECK:STDOUT:   %.loc134_5.4: %Cpp.long = converted %b.ref.loc134, %.loc134_5.3
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc134: <specific function> = specific_function %Equal.ref.loc134, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.b56f6d.2]
// CHECK:STDOUT:   %bound_method.loc134_5.4: <bound method> = bound_method %a.ref.loc134, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc134
// CHECK:STDOUT:   %impl.elem0.loc134_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc134_8: <bound method> = bound_method %b.ref.loc134, %impl.elem0.loc134_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc134_8: init %Cpp.long = call %bound_method.loc134_8(%b.ref.loc134)
// CHECK:STDOUT:   %.loc134_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc134_8
// CHECK:STDOUT:   %.loc134_8.2: %Cpp.long = converted %b.ref.loc134, %.loc134_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc134: init bool = call %bound_method.loc134_5.4(%a.ref.loc134, %.loc134_8.2)
// CHECK:STDOUT:   %a.ref.loc135: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc135: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc135: %.1c7 = impl_witness_access constants.%EqWith.impl_witness.15a, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.2]
// CHECK:STDOUT:   %bound_method.loc135_5.1: <bound method> = bound_method %a.ref.loc135, %impl.elem1.loc135
// CHECK:STDOUT:   %ImplicitAs.facet.loc135: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc135_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc135 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc135: <specific function> = specific_function %impl.elem1.loc135, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.1]
// CHECK:STDOUT:   %bound_method.loc135_5.2: <bound method> = bound_method %a.ref.loc135, %specific_fn.loc135
// CHECK:STDOUT:   %.loc135_5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.1 = specific_constant imports.%Core.NotEqual.b1f, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.1]
// CHECK:STDOUT:   %NotEqual.ref.loc135: %Cpp.long.as.EqWith.impl.NotEqual.type.b6d1d8.1 = name_ref NotEqual, %.loc135_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.6a28a6.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc135: <bound method> = bound_method %a.ref.loc135, %NotEqual.ref.loc135
// CHECK:STDOUT:   %impl.elem0.loc135_5: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc135_5.3: <bound method> = bound_method %b.ref.loc135, %impl.elem0.loc135_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc135_5: init %Cpp.long = call %bound_method.loc135_5.3(%b.ref.loc135)
// CHECK:STDOUT:   %.loc135_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc135_5
// CHECK:STDOUT:   %.loc135_5.4: %Cpp.long = converted %b.ref.loc135, %.loc135_5.3
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc135: <specific function> = specific_function %NotEqual.ref.loc135, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.25d32e.2]
// CHECK:STDOUT:   %bound_method.loc135_5.4: <bound method> = bound_method %a.ref.loc135, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc135
// CHECK:STDOUT:   %impl.elem0.loc135_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc135_8: <bound method> = bound_method %b.ref.loc135, %impl.elem0.loc135_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc135_8: init %Cpp.long = call %bound_method.loc135_8(%b.ref.loc135)
// CHECK:STDOUT:   %.loc135_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc135_8
// CHECK:STDOUT:   %.loc135_8.2: %Cpp.long = converted %b.ref.loc135, %.loc135_8.1
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc135: init bool = call %bound_method.loc135_5.4(%a.ref.loc135, %.loc135_8.2)
// CHECK:STDOUT:   %a.ref.loc136: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc136: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc136: %.883 = impl_witness_access constants.%OrderedWith.impl_witness.576, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.2]
// CHECK:STDOUT:   %bound_method.loc136_5.1: <bound method> = bound_method %a.ref.loc136, %impl.elem2.loc136
// CHECK:STDOUT:   %ImplicitAs.facet.loc136: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc136_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc136 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc136: <specific function> = specific_function %impl.elem2.loc136, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.1]
// CHECK:STDOUT:   %bound_method.loc136_5.2: <bound method> = bound_method %a.ref.loc136, %specific_fn.loc136
// CHECK:STDOUT:   %.loc136_5.2: %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.1 = specific_constant imports.%Core.Greater.9da, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.1]
// CHECK:STDOUT:   %Greater.ref.loc136: %Cpp.long.as.OrderedWith.impl.Greater.type.a1c5d5.1 = name_ref Greater, %.loc136_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.9898ff.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc136: <bound method> = bound_method %a.ref.loc136, %Greater.ref.loc136
// CHECK:STDOUT:   %impl.elem0.loc136_5: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc136_5.3: <bound method> = bound_method %b.ref.loc136, %impl.elem0.loc136_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc136_5: init %Cpp.long = call %bound_method.loc136_5.3(%b.ref.loc136)
// CHECK:STDOUT:   %.loc136_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc136_5
// CHECK:STDOUT:   %.loc136_5.4: %Cpp.long = converted %b.ref.loc136, %.loc136_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc136: <specific function> = specific_function %Greater.ref.loc136, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.760220.2]
// CHECK:STDOUT:   %bound_method.loc136_5.4: <bound method> = bound_method %a.ref.loc136, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc136
// CHECK:STDOUT:   %impl.elem0.loc136_7: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc136_7: <bound method> = bound_method %b.ref.loc136, %impl.elem0.loc136_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc136_7: init %Cpp.long = call %bound_method.loc136_7(%b.ref.loc136)
// CHECK:STDOUT:   %.loc136_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc136_7
// CHECK:STDOUT:   %.loc136_7.2: %Cpp.long = converted %b.ref.loc136, %.loc136_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc136: init bool = call %bound_method.loc136_5.4(%a.ref.loc136, %.loc136_7.2)
// CHECK:STDOUT:   %a.ref.loc137: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc137: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc137_5.1: %.481 = impl_witness_access constants.%OrderedWith.impl_witness.576, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.2]
// CHECK:STDOUT:   %bound_method.loc137_5.1: <bound method> = bound_method %a.ref.loc137, %impl.elem0.loc137_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc137: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc137_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc137 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc137: <specific function> = specific_function %impl.elem0.loc137_5.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.1]
// CHECK:STDOUT:   %bound_method.loc137_5.2: <bound method> = bound_method %a.ref.loc137, %specific_fn.loc137
// CHECK:STDOUT:   %.loc137_5.2: %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.1 = specific_constant imports.%Core.Less.c86, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.1]
// CHECK:STDOUT:   %Less.ref.loc137: %Cpp.long.as.OrderedWith.impl.Less.type.8fda0e.1 = name_ref Less, %.loc137_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.5299ce.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc137: <bound method> = bound_method %a.ref.loc137, %Less.ref.loc137
// CHECK:STDOUT:   %impl.elem0.loc137_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc137_5.3: <bound method> = bound_method %b.ref.loc137, %impl.elem0.loc137_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc137_5: init %Cpp.long = call %bound_method.loc137_5.3(%b.ref.loc137)
// CHECK:STDOUT:   %.loc137_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc137_5
// CHECK:STDOUT:   %.loc137_5.4: %Cpp.long = converted %b.ref.loc137, %.loc137_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc137: <specific function> = specific_function %Less.ref.loc137, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.89e78a.2]
// CHECK:STDOUT:   %bound_method.loc137_5.4: <bound method> = bound_method %a.ref.loc137, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc137
// CHECK:STDOUT:   %impl.elem0.loc137_7: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc137_7: <bound method> = bound_method %b.ref.loc137, %impl.elem0.loc137_7
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc137_7: init %Cpp.long = call %bound_method.loc137_7(%b.ref.loc137)
// CHECK:STDOUT:   %.loc137_7.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc137_7
// CHECK:STDOUT:   %.loc137_7.2: %Cpp.long = converted %b.ref.loc137, %.loc137_7.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc137: init bool = call %bound_method.loc137_5.4(%a.ref.loc137, %.loc137_7.2)
// CHECK:STDOUT:   %a.ref.loc138: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc138: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc138: %.753 = impl_witness_access constants.%OrderedWith.impl_witness.576, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.2]
// CHECK:STDOUT:   %bound_method.loc138_5.1: <bound method> = bound_method %a.ref.loc138, %impl.elem3.loc138
// CHECK:STDOUT:   %ImplicitAs.facet.loc138: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc138_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc138 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc138: <specific function> = specific_function %impl.elem3.loc138, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.1]
// CHECK:STDOUT:   %bound_method.loc138_5.2: <bound method> = bound_method %a.ref.loc138, %specific_fn.loc138
// CHECK:STDOUT:   %.loc138_5.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.1 = specific_constant imports.%Core.GreaterOrEquivalent.76e, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc138: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.062a19.1 = name_ref GreaterOrEquivalent, %.loc138_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.c90ad0.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc138: <bound method> = bound_method %a.ref.loc138, %GreaterOrEquivalent.ref.loc138
// CHECK:STDOUT:   %impl.elem0.loc138_5: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc138_5.3: <bound method> = bound_method %b.ref.loc138, %impl.elem0.loc138_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc138_5: init %Cpp.long = call %bound_method.loc138_5.3(%b.ref.loc138)
// CHECK:STDOUT:   %.loc138_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc138_5
// CHECK:STDOUT:   %.loc138_5.4: %Cpp.long = converted %b.ref.loc138, %.loc138_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc138: <specific function> = specific_function %GreaterOrEquivalent.ref.loc138, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.e67d4e.2]
// CHECK:STDOUT:   %bound_method.loc138_5.4: <bound method> = bound_method %a.ref.loc138, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc138
// CHECK:STDOUT:   %impl.elem0.loc138_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc138_8: <bound method> = bound_method %b.ref.loc138, %impl.elem0.loc138_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc138_8: init %Cpp.long = call %bound_method.loc138_8(%b.ref.loc138)
// CHECK:STDOUT:   %.loc138_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc138_8
// CHECK:STDOUT:   %.loc138_8.2: %Cpp.long = converted %b.ref.loc138, %.loc138_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc138: init bool = call %bound_method.loc138_5.4(%a.ref.loc138, %.loc138_8.2)
// CHECK:STDOUT:   %a.ref.loc139: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc139: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc139: %.14a = impl_witness_access constants.%OrderedWith.impl_witness.576, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.2]
// CHECK:STDOUT:   %bound_method.loc139_5.1: <bound method> = bound_method %a.ref.loc139, %impl.elem1.loc139
// CHECK:STDOUT:   %ImplicitAs.facet.loc139: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc139_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc139 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc139: <specific function> = specific_function %impl.elem1.loc139, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9548d5.1]
// CHECK:STDOUT:   %bound_method.loc139_5.2: <bound method> = bound_method %a.ref.loc139, %specific_fn.loc139
// CHECK:STDOUT:   %.loc139_5.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.1 = specific_constant imports.%Core.LessOrEquivalent.4a6, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc139: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.11ca57.1 = name_ref LessOrEquivalent, %.loc139_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1eddf1.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc139: <bound method> = bound_method %a.ref.loc139, %LessOrEquivalent.ref.loc139
// CHECK:STDOUT:   %impl.elem0.loc139_5: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc139_5.3: <bound method> = bound_method %b.ref.loc139, %impl.elem0.loc139_5
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc139_5: init %Cpp.long = call %bound_method.loc139_5.3(%b.ref.loc139)
// CHECK:STDOUT:   %.loc139_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc139_5
// CHECK:STDOUT:   %.loc139_5.4: %Cpp.long = converted %b.ref.loc139, %.loc139_5.3
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc139: <specific function> = specific_function %LessOrEquivalent.ref.loc139, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.9548d5.2]
// CHECK:STDOUT:   %bound_method.loc139_5.4: <bound method> = bound_method %a.ref.loc139, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc139
// CHECK:STDOUT:   %impl.elem0.loc139_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc139_8: <bound method> = bound_method %b.ref.loc139, %impl.elem0.loc139_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc139_8: init %Cpp.long = call %bound_method.loc139_8(%b.ref.loc139)
// CHECK:STDOUT:   %.loc139_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc139_8
// CHECK:STDOUT:   %.loc139_8.2: %Cpp.long = converted %b.ref.loc139, %.loc139_8.1
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc139: init bool = call %bound_method.loc139_5.4(%a.ref.loc139, %.loc139_8.2)
// CHECK:STDOUT:   %a.ref.loc141: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc141: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc141_5.1: %.a03 = impl_witness_access constants.%EqWith.impl_witness.8e5, element0 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.6ed862.2]
// CHECK:STDOUT:   %bound_method.loc141_5.1: <bound method> = bound_method %a.ref.loc141, %impl.elem0.loc141_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc141: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc141_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc141 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc141: <specific function> = specific_function %impl.elem0.loc141_5.1, @Cpp.long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.334ae9.1]
// CHECK:STDOUT:   %bound_method.loc141_5.2: <bound method> = bound_method %a.ref.loc141, %specific_fn.loc141
// CHECK:STDOUT:   %.loc141_5.2: %Cpp.long.as.EqWith.impl.Equal.type.416d09.1 = specific_constant imports.%Core.Equal.0ad, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.6ed862.1]
// CHECK:STDOUT:   %Equal.ref.loc141: %Cpp.long.as.EqWith.impl.Equal.type.416d09.1 = name_ref Equal, %.loc141_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.6ed862.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.bound.loc141: <bound method> = bound_method %a.ref.loc141, %Equal.ref.loc141
// CHECK:STDOUT:   %impl.elem0.loc141_5.2: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc141_5.3: <bound method> = bound_method %int_1.loc141, %impl.elem0.loc141_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc141_5: init %Cpp.long = call %bound_method.loc141_5.3(%int_1.loc141) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc141_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc141_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc141_5.4: %Cpp.long = converted %int_1.loc141, %.loc141_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc141: <specific function> = specific_function %Equal.ref.loc141, @Cpp.long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.Equal.specific_fn.334ae9.2]
// CHECK:STDOUT:   %bound_method.loc141_5.4: <bound method> = bound_method %a.ref.loc141, %Cpp.long.as.EqWith.impl.Equal.specific_fn.loc141
// CHECK:STDOUT:   %impl.elem0.loc141_8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc141_8: <bound method> = bound_method %int_1.loc141, %impl.elem0.loc141_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc141_8: init %Cpp.long = call %bound_method.loc141_8(%int_1.loc141) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc141_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc141_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc141_8.2: %Cpp.long = converted %int_1.loc141, %.loc141_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.Equal.call.loc141: init bool = call %bound_method.loc141_5.4(%a.ref.loc141, %.loc141_8.2)
// CHECK:STDOUT:   %a.ref.loc142: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc142: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc142: %.9a1 = impl_witness_access constants.%EqWith.impl_witness.8e5, element1 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.f686b3.2]
// CHECK:STDOUT:   %bound_method.loc142_5.1: <bound method> = bound_method %a.ref.loc142, %impl.elem1.loc142
// CHECK:STDOUT:   %ImplicitAs.facet.loc142: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc142_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc142 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc142: <specific function> = specific_function %impl.elem1.loc142, @Cpp.long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5aa006.1]
// CHECK:STDOUT:   %bound_method.loc142_5.2: <bound method> = bound_method %a.ref.loc142, %specific_fn.loc142
// CHECK:STDOUT:   %.loc142_5.2: %Cpp.long.as.EqWith.impl.NotEqual.type.de968c.1 = specific_constant imports.%Core.NotEqual.b1f, @Cpp.long.as.EqWith.impl.f13(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.f686b3.1]
// CHECK:STDOUT:   %NotEqual.ref.loc142: %Cpp.long.as.EqWith.impl.NotEqual.type.de968c.1 = name_ref NotEqual, %.loc142_5.2 [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.f686b3.1]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.bound.loc142: <bound method> = bound_method %a.ref.loc142, %NotEqual.ref.loc142
// CHECK:STDOUT:   %impl.elem0.loc142_5: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc142_5.3: <bound method> = bound_method %int_1.loc142, %impl.elem0.loc142_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc142_5: init %Cpp.long = call %bound_method.loc142_5.3(%int_1.loc142) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc142_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc142_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc142_5.4: %Cpp.long = converted %int_1.loc142, %.loc142_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc142: <specific function> = specific_function %NotEqual.ref.loc142, @Cpp.long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.EqWith.impl.NotEqual.specific_fn.5aa006.2]
// CHECK:STDOUT:   %bound_method.loc142_5.4: <bound method> = bound_method %a.ref.loc142, %Cpp.long.as.EqWith.impl.NotEqual.specific_fn.loc142
// CHECK:STDOUT:   %impl.elem0.loc142_8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc142_8: <bound method> = bound_method %int_1.loc142, %impl.elem0.loc142_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc142_8: init %Cpp.long = call %bound_method.loc142_8(%int_1.loc142) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc142_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc142_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc142_8.2: %Cpp.long = converted %int_1.loc142, %.loc142_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.EqWith.impl.NotEqual.call.loc142: init bool = call %bound_method.loc142_5.4(%a.ref.loc142, %.loc142_8.2)
// CHECK:STDOUT:   %a.ref.loc143: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc143: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc143: %.5af = impl_witness_access constants.%OrderedWith.impl_witness.52f, element2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.74a807.2]
// CHECK:STDOUT:   %bound_method.loc143_5.1: <bound method> = bound_method %a.ref.loc143, %impl.elem2.loc143
// CHECK:STDOUT:   %ImplicitAs.facet.loc143: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc143_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc143 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc143: <specific function> = specific_function %impl.elem2.loc143, @Cpp.long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.b2149b.1]
// CHECK:STDOUT:   %bound_method.loc143_5.2: <bound method> = bound_method %a.ref.loc143, %specific_fn.loc143
// CHECK:STDOUT:   %.loc143_5.2: %Cpp.long.as.OrderedWith.impl.Greater.type.3fd1ac.1 = specific_constant imports.%Core.Greater.9da, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.74a807.1]
// CHECK:STDOUT:   %Greater.ref.loc143: %Cpp.long.as.OrderedWith.impl.Greater.type.3fd1ac.1 = name_ref Greater, %.loc143_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.74a807.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.bound.loc143: <bound method> = bound_method %a.ref.loc143, %Greater.ref.loc143
// CHECK:STDOUT:   %impl.elem0.loc143_5: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc143_5.3: <bound method> = bound_method %int_1.loc143, %impl.elem0.loc143_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc143_5: init %Cpp.long = call %bound_method.loc143_5.3(%int_1.loc143) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc143_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc143_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc143_5.4: %Cpp.long = converted %int_1.loc143, %.loc143_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc143: <specific function> = specific_function %Greater.ref.loc143, @Cpp.long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Greater.specific_fn.b2149b.2]
// CHECK:STDOUT:   %bound_method.loc143_5.4: <bound method> = bound_method %a.ref.loc143, %Cpp.long.as.OrderedWith.impl.Greater.specific_fn.loc143
// CHECK:STDOUT:   %impl.elem0.loc143_7: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc143_7: <bound method> = bound_method %int_1.loc143, %impl.elem0.loc143_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc143_7: init %Cpp.long = call %bound_method.loc143_7(%int_1.loc143) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc143_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc143_7 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc143_7.2: %Cpp.long = converted %int_1.loc143, %.loc143_7.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Greater.call.loc143: init bool = call %bound_method.loc143_5.4(%a.ref.loc143, %.loc143_7.2)
// CHECK:STDOUT:   %a.ref.loc144: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc144: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc144_5.1: %.6b7 = impl_witness_access constants.%OrderedWith.impl_witness.52f, element0 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.0de767.2]
// CHECK:STDOUT:   %bound_method.loc144_5.1: <bound method> = bound_method %a.ref.loc144, %impl.elem0.loc144_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc144: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc144_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc144 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc144: <specific function> = specific_function %impl.elem0.loc144_5.1, @Cpp.long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.b6fc5e.1]
// CHECK:STDOUT:   %bound_method.loc144_5.2: <bound method> = bound_method %a.ref.loc144, %specific_fn.loc144
// CHECK:STDOUT:   %.loc144_5.2: %Cpp.long.as.OrderedWith.impl.Less.type.8ced31.1 = specific_constant imports.%Core.Less.c86, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.0de767.1]
// CHECK:STDOUT:   %Less.ref.loc144: %Cpp.long.as.OrderedWith.impl.Less.type.8ced31.1 = name_ref Less, %.loc144_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.0de767.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.bound.loc144: <bound method> = bound_method %a.ref.loc144, %Less.ref.loc144
// CHECK:STDOUT:   %impl.elem0.loc144_5.2: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc144_5.3: <bound method> = bound_method %int_1.loc144, %impl.elem0.loc144_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc144_5: init %Cpp.long = call %bound_method.loc144_5.3(%int_1.loc144) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc144_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc144_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc144_5.4: %Cpp.long = converted %int_1.loc144, %.loc144_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc144: <specific function> = specific_function %Less.ref.loc144, @Cpp.long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.Less.specific_fn.b6fc5e.2]
// CHECK:STDOUT:   %bound_method.loc144_5.4: <bound method> = bound_method %a.ref.loc144, %Cpp.long.as.OrderedWith.impl.Less.specific_fn.loc144
// CHECK:STDOUT:   %impl.elem0.loc144_7: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc144_7: <bound method> = bound_method %int_1.loc144, %impl.elem0.loc144_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc144_7: init %Cpp.long = call %bound_method.loc144_7(%int_1.loc144) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc144_7.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc144_7 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc144_7.2: %Cpp.long = converted %int_1.loc144, %.loc144_7.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.Less.call.loc144: init bool = call %bound_method.loc144_5.4(%a.ref.loc144, %.loc144_7.2)
// CHECK:STDOUT:   %a.ref.loc145: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc145: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc145: %.1f9 = impl_witness_access constants.%OrderedWith.impl_witness.52f, element3 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.2]
// CHECK:STDOUT:   %bound_method.loc145_5.1: <bound method> = bound_method %a.ref.loc145, %impl.elem3.loc145
// CHECK:STDOUT:   %ImplicitAs.facet.loc145: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc145_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc145 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc145: <specific function> = specific_function %impl.elem3.loc145, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f90679.1]
// CHECK:STDOUT:   %bound_method.loc145_5.2: <bound method> = bound_method %a.ref.loc145, %specific_fn.loc145
// CHECK:STDOUT:   %.loc145_5.2: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2273d7.1 = specific_constant imports.%Core.GreaterOrEquivalent.76e, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc145: %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.type.2273d7.1 = name_ref GreaterOrEquivalent, %.loc145_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.abd013.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc145: <bound method> = bound_method %a.ref.loc145, %GreaterOrEquivalent.ref.loc145
// CHECK:STDOUT:   %impl.elem0.loc145_5: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc145_5.3: <bound method> = bound_method %int_1.loc145, %impl.elem0.loc145_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc145_5: init %Cpp.long = call %bound_method.loc145_5.3(%int_1.loc145) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc145_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc145_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc145_5.4: %Cpp.long = converted %int_1.loc145, %.loc145_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc145: <specific function> = specific_function %GreaterOrEquivalent.ref.loc145, @Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.f90679.2]
// CHECK:STDOUT:   %bound_method.loc145_5.4: <bound method> = bound_method %a.ref.loc145, %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc145
// CHECK:STDOUT:   %impl.elem0.loc145_8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc145_8: <bound method> = bound_method %int_1.loc145, %impl.elem0.loc145_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc145_8: init %Cpp.long = call %bound_method.loc145_8(%int_1.loc145) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc145_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc145_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc145_8.2: %Cpp.long = converted %int_1.loc145, %.loc145_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc145: init bool = call %bound_method.loc145_5.4(%a.ref.loc145, %.loc145_8.2)
// CHECK:STDOUT:   %a.ref.loc146: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc146: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc146: %.76b = impl_witness_access constants.%OrderedWith.impl_witness.52f, element1 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.2]
// CHECK:STDOUT:   %bound_method.loc146_5.1: <bound method> = bound_method %a.ref.loc146, %impl.elem1.loc146
// CHECK:STDOUT:   %ImplicitAs.facet.loc146: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %.loc146_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet.loc146 [concrete = constants.%ImplicitAs.facet.eed]
// CHECK:STDOUT:   %specific_fn.loc146: <specific function> = specific_function %impl.elem1.loc146, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.c5c24a.1]
// CHECK:STDOUT:   %bound_method.loc146_5.2: <bound method> = bound_method %a.ref.loc146, %specific_fn.loc146
// CHECK:STDOUT:   %.loc146_5.2: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.28524d.1 = specific_constant imports.%Core.LessOrEquivalent.4a6, @Cpp.long.as.OrderedWith.impl.15a(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc146: %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.type.28524d.1 = name_ref LessOrEquivalent, %.loc146_5.2 [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.9833d9.1]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.bound.loc146: <bound method> = bound_method %a.ref.loc146, %LessOrEquivalent.ref.loc146
// CHECK:STDOUT:   %impl.elem0.loc146_5: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc146_5.3: <bound method> = bound_method %int_1.loc146, %impl.elem0.loc146_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc146_5: init %Cpp.long = call %bound_method.loc146_5.3(%int_1.loc146) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc146_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc146_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc146_5.4: %Cpp.long = converted %int_1.loc146, %.loc146_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc146: <specific function> = specific_function %LessOrEquivalent.ref.loc146, @Cpp.long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.c5c24a.2]
// CHECK:STDOUT:   %bound_method.loc146_5.4: <bound method> = bound_method %a.ref.loc146, %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc146
// CHECK:STDOUT:   %impl.elem0.loc146_8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc146_8: <bound method> = bound_method %int_1.loc146, %impl.elem0.loc146_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc146_8: init %Cpp.long = call %bound_method.loc146_8(%int_1.loc146) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc146_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc146_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc146_8.2: %Cpp.long = converted %int_1.loc146, %.loc146_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.OrderedWith.impl.LessOrEquivalent.call.loc146: init bool = call %bound_method.loc146_5.4(%a.ref.loc146, %.loc146_8.2)
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
// CHECK:STDOUT:   %int_1.loc159: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc159_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc159: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc159: <bound method> = bound_method %int_1.loc159, %impl.elem0.loc159 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc159: init %Cpp.long = call %bound_method.loc159(%int_1.loc159) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc159_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc159 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc159_21.2: %Cpp.long = converted %int_1.loc159, %.loc159_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc159_21.2
// CHECK:STDOUT:   %b.ref.loc160: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc160: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc160_5: %.566 = impl_witness_access constants.%EqWith.impl_witness.949, element0 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.2]
// CHECK:STDOUT:   %bound_method.loc160_5.1: <bound method> = bound_method %b.ref.loc160, %impl.elem0.loc160_5
// CHECK:STDOUT:   %specific_fn.loc160: <specific function> = specific_function %impl.elem0.loc160_5, @T.binding.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.1]
// CHECK:STDOUT:   %bound_method.loc160_5.2: <bound method> = bound_method %b.ref.loc160, %specific_fn.loc160
// CHECK:STDOUT:   %.loc160_5: %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.1 = specific_constant imports.%Core.Equal.784, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.1]
// CHECK:STDOUT:   %Equal.ref.loc160: %T.binding.as_type.as.EqWith.impl.Equal.type.d5eb8f.1 = name_ref Equal, %.loc160_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.b65948.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.loc160: <bound method> = bound_method %b.ref.loc160, %Equal.ref.loc160
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc160: <specific function> = specific_function %Equal.ref.loc160, @T.binding.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.73f7cb.2]
// CHECK:STDOUT:   %bound_method.loc160_5.3: <bound method> = bound_method %b.ref.loc160, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc160
// CHECK:STDOUT:   %impl.elem0.loc160_3: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc160_3: <bound method> = bound_method %b.ref.loc160, %impl.elem0.loc160_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc160: init %Cpp.long = call %bound_method.loc160_3(%b.ref.loc160)
// CHECK:STDOUT:   %.loc160_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc160
// CHECK:STDOUT:   %.loc160_3.2: %Cpp.long = converted %b.ref.loc160, %.loc160_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.call.loc160: init bool = call %bound_method.loc160_5.3(%.loc160_3.2, %a.ref.loc160)
// CHECK:STDOUT:   %b.ref.loc161: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc161: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc161: %.4d8 = impl_witness_access constants.%EqWith.impl_witness.949, element1 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.2]
// CHECK:STDOUT:   %bound_method.loc161_5.1: <bound method> = bound_method %b.ref.loc161, %impl.elem1.loc161
// CHECK:STDOUT:   %specific_fn.loc161: <specific function> = specific_function %impl.elem1.loc161, @T.binding.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.1]
// CHECK:STDOUT:   %bound_method.loc161_5.2: <bound method> = bound_method %b.ref.loc161, %specific_fn.loc161
// CHECK:STDOUT:   %.loc161_5: %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.1 = specific_constant imports.%Core.NotEqual.a7f, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.1]
// CHECK:STDOUT:   %NotEqual.ref.loc161: %T.binding.as_type.as.EqWith.impl.NotEqual.type.0a8f8f.1 = name_ref NotEqual, %.loc161_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.37228f.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.loc161: <bound method> = bound_method %b.ref.loc161, %NotEqual.ref.loc161
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc161: <specific function> = specific_function %NotEqual.ref.loc161, @T.binding.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.91cc57.2]
// CHECK:STDOUT:   %bound_method.loc161_5.3: <bound method> = bound_method %b.ref.loc161, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc161
// CHECK:STDOUT:   %impl.elem0.loc161: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc161_3: <bound method> = bound_method %b.ref.loc161, %impl.elem0.loc161
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc161: init %Cpp.long = call %bound_method.loc161_3(%b.ref.loc161)
// CHECK:STDOUT:   %.loc161_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc161
// CHECK:STDOUT:   %.loc161_3.2: %Cpp.long = converted %b.ref.loc161, %.loc161_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.call.loc161: init bool = call %bound_method.loc161_5.3(%.loc161_3.2, %a.ref.loc161)
// CHECK:STDOUT:   %b.ref.loc162: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc162: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc162: %.d93 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element2 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.2]
// CHECK:STDOUT:   %bound_method.loc162_5.1: <bound method> = bound_method %b.ref.loc162, %impl.elem2.loc162
// CHECK:STDOUT:   %specific_fn.loc162: <specific function> = specific_function %impl.elem2.loc162, @T.binding.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.1]
// CHECK:STDOUT:   %bound_method.loc162_5.2: <bound method> = bound_method %b.ref.loc162, %specific_fn.loc162
// CHECK:STDOUT:   %.loc162_5: %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.1 = specific_constant imports.%Core.Greater.696, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.1]
// CHECK:STDOUT:   %Greater.ref.loc162: %T.binding.as_type.as.OrderedWith.impl.Greater.type.c0ea4d.1 = name_ref Greater, %.loc162_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.5cc4da.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.loc162: <bound method> = bound_method %b.ref.loc162, %Greater.ref.loc162
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc162: <specific function> = specific_function %Greater.ref.loc162, @T.binding.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.6e99ac.2]
// CHECK:STDOUT:   %bound_method.loc162_5.3: <bound method> = bound_method %b.ref.loc162, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc162
// CHECK:STDOUT:   %impl.elem0.loc162: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc162_3: <bound method> = bound_method %b.ref.loc162, %impl.elem0.loc162
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc162: init %Cpp.long = call %bound_method.loc162_3(%b.ref.loc162)
// CHECK:STDOUT:   %.loc162_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc162
// CHECK:STDOUT:   %.loc162_3.2: %Cpp.long = converted %b.ref.loc162, %.loc162_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.call.loc162: init bool = call %bound_method.loc162_5.3(%.loc162_3.2, %a.ref.loc162)
// CHECK:STDOUT:   %b.ref.loc163: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc163: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc163_5: %.4a1 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element0 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.2]
// CHECK:STDOUT:   %bound_method.loc163_5.1: <bound method> = bound_method %b.ref.loc163, %impl.elem0.loc163_5
// CHECK:STDOUT:   %specific_fn.loc163: <specific function> = specific_function %impl.elem0.loc163_5, @T.binding.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.1]
// CHECK:STDOUT:   %bound_method.loc163_5.2: <bound method> = bound_method %b.ref.loc163, %specific_fn.loc163
// CHECK:STDOUT:   %.loc163_5: %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.1 = specific_constant imports.%Core.Less.b61, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.1]
// CHECK:STDOUT:   %Less.ref.loc163: %T.binding.as_type.as.OrderedWith.impl.Less.type.c1d567.1 = name_ref Less, %.loc163_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.f24104.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.loc163: <bound method> = bound_method %b.ref.loc163, %Less.ref.loc163
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc163: <specific function> = specific_function %Less.ref.loc163, @T.binding.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.548b71.2]
// CHECK:STDOUT:   %bound_method.loc163_5.3: <bound method> = bound_method %b.ref.loc163, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc163
// CHECK:STDOUT:   %impl.elem0.loc163_3: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc163_3: <bound method> = bound_method %b.ref.loc163, %impl.elem0.loc163_3
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc163: init %Cpp.long = call %bound_method.loc163_3(%b.ref.loc163)
// CHECK:STDOUT:   %.loc163_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc163
// CHECK:STDOUT:   %.loc163_3.2: %Cpp.long = converted %b.ref.loc163, %.loc163_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.call.loc163: init bool = call %bound_method.loc163_5.3(%.loc163_3.2, %a.ref.loc163)
// CHECK:STDOUT:   %b.ref.loc164: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc164: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc164: %.7fd = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element3 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.2]
// CHECK:STDOUT:   %bound_method.loc164_5.1: <bound method> = bound_method %b.ref.loc164, %impl.elem3.loc164
// CHECK:STDOUT:   %specific_fn.loc164: <specific function> = specific_function %impl.elem3.loc164, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.1]
// CHECK:STDOUT:   %bound_method.loc164_5.2: <bound method> = bound_method %b.ref.loc164, %specific_fn.loc164
// CHECK:STDOUT:   %.loc164_5: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.1 = specific_constant imports.%Core.GreaterOrEquivalent.c46, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc164: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9e0984.1 = name_ref GreaterOrEquivalent, %.loc164_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.6ad500.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc164: <bound method> = bound_method %b.ref.loc164, %GreaterOrEquivalent.ref.loc164
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc164: <specific function> = specific_function %GreaterOrEquivalent.ref.loc164, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.445813.2]
// CHECK:STDOUT:   %bound_method.loc164_5.3: <bound method> = bound_method %b.ref.loc164, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc164
// CHECK:STDOUT:   %impl.elem0.loc164: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc164_3: <bound method> = bound_method %b.ref.loc164, %impl.elem0.loc164
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc164: init %Cpp.long = call %bound_method.loc164_3(%b.ref.loc164)
// CHECK:STDOUT:   %.loc164_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc164
// CHECK:STDOUT:   %.loc164_3.2: %Cpp.long = converted %b.ref.loc164, %.loc164_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc164: init bool = call %bound_method.loc164_5.3(%.loc164_3.2, %a.ref.loc164)
// CHECK:STDOUT:   %b.ref.loc165: %i32 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc165: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc165: %.8f3 = impl_witness_access constants.%OrderedWith.impl_witness.f9b, element1 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.2]
// CHECK:STDOUT:   %bound_method.loc165_5.1: <bound method> = bound_method %b.ref.loc165, %impl.elem1.loc165
// CHECK:STDOUT:   %specific_fn.loc165: <specific function> = specific_function %impl.elem1.loc165, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a789d4.1]
// CHECK:STDOUT:   %bound_method.loc165_5.2: <bound method> = bound_method %b.ref.loc165, %specific_fn.loc165
// CHECK:STDOUT:   %.loc165_5: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.1 = specific_constant imports.%Core.LessOrEquivalent.947, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc165: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1492cb.1 = name_ref LessOrEquivalent, %.loc165_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.48db04.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc165: <bound method> = bound_method %b.ref.loc165, %LessOrEquivalent.ref.loc165
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc165: <specific function> = specific_function %LessOrEquivalent.ref.loc165, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.174) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.a789d4.2]
// CHECK:STDOUT:   %bound_method.loc165_5.3: <bound method> = bound_method %b.ref.loc165, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc165
// CHECK:STDOUT:   %impl.elem0.loc165: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc165_3: <bound method> = bound_method %b.ref.loc165, %impl.elem0.loc165
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc165: init %Cpp.long = call %bound_method.loc165_3(%b.ref.loc165)
// CHECK:STDOUT:   %.loc165_3.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc165
// CHECK:STDOUT:   %.loc165_3.2: %Cpp.long = converted %b.ref.loc165, %.loc165_3.1
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc165: init bool = call %bound_method.loc165_5.3(%.loc165_3.2, %a.ref.loc165)
// CHECK:STDOUT:   %int_1.loc167: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc167: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc167_5: %.389 = impl_witness_access constants.%EqWith.impl_witness.801, element0 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.326379.2]
// CHECK:STDOUT:   %bound_method.loc167_5.1: <bound method> = bound_method %int_1.loc167, %impl.elem0.loc167_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.bound.62ea05.1]
// CHECK:STDOUT:   %specific_fn.loc167: <specific function> = specific_function %impl.elem0.loc167_5, @T.binding.as_type.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.1]
// CHECK:STDOUT:   %bound_method.loc167_5.2: <bound method> = bound_method %int_1.loc167, %specific_fn.loc167 [concrete = constants.%bound_method.8a46b0.1]
// CHECK:STDOUT:   %.loc167_5: %T.binding.as_type.as.EqWith.impl.Equal.type.85dd3c.1 = specific_constant imports.%Core.Equal.784, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.326379.1]
// CHECK:STDOUT:   %Equal.ref.loc167: %T.binding.as_type.as.EqWith.impl.Equal.type.85dd3c.1 = name_ref Equal, %.loc167_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.326379.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.bound.loc167: <bound method> = bound_method %int_1.loc167, %Equal.ref.loc167 [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.bound.62ea05.2]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc167: <specific function> = specific_function %Equal.ref.loc167, @T.binding.as_type.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.Equal.specific_fn.ad9d43.2]
// CHECK:STDOUT:   %bound_method.loc167_5.3: <bound method> = bound_method %int_1.loc167, %T.binding.as_type.as.EqWith.impl.Equal.specific_fn.loc167 [concrete = constants.%bound_method.8a46b0.2]
// CHECK:STDOUT:   %impl.elem0.loc167_3: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc167_3: <bound method> = bound_method %int_1.loc167, %impl.elem0.loc167_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc167: init %Cpp.long = call %bound_method.loc167_3(%int_1.loc167) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc167_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc167 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc167_3.2: %Cpp.long = converted %int_1.loc167, %.loc167_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.Equal.call.loc167: init bool = call %bound_method.loc167_5.3(%.loc167_3.2, %a.ref.loc167)
// CHECK:STDOUT:   %int_1.loc168: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc168: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc168: %.ec0 = impl_witness_access constants.%EqWith.impl_witness.801, element1 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.2]
// CHECK:STDOUT:   %bound_method.loc168_5.1: <bound method> = bound_method %int_1.loc168, %impl.elem1.loc168 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bound.29032e.1]
// CHECK:STDOUT:   %specific_fn.loc168: <specific function> = specific_function %impl.elem1.loc168, @T.binding.as_type.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.ceecb6.1]
// CHECK:STDOUT:   %bound_method.loc168_5.2: <bound method> = bound_method %int_1.loc168, %specific_fn.loc168 [concrete = constants.%bound_method.082399.1]
// CHECK:STDOUT:   %.loc168_5: %T.binding.as_type.as.EqWith.impl.NotEqual.type.bb8d74.1 = specific_constant imports.%Core.NotEqual.a7f, @T.binding.as_type.as.EqWith.impl.8ff(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.1]
// CHECK:STDOUT:   %NotEqual.ref.loc168: %T.binding.as_type.as.EqWith.impl.NotEqual.type.bb8d74.1 = name_ref NotEqual, %.loc168_5 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.ffd0ee.1]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.bound.loc168: <bound method> = bound_method %int_1.loc168, %NotEqual.ref.loc168 [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.bound.29032e.2]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc168: <specific function> = specific_function %NotEqual.ref.loc168, @T.binding.as_type.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.ceecb6.2]
// CHECK:STDOUT:   %bound_method.loc168_5.3: <bound method> = bound_method %int_1.loc168, %T.binding.as_type.as.EqWith.impl.NotEqual.specific_fn.loc168 [concrete = constants.%bound_method.082399.2]
// CHECK:STDOUT:   %impl.elem0.loc168: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc168_3: <bound method> = bound_method %int_1.loc168, %impl.elem0.loc168 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc168: init %Cpp.long = call %bound_method.loc168_3(%int_1.loc168) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc168_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc168 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc168_3.2: %Cpp.long = converted %int_1.loc168, %.loc168_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.EqWith.impl.NotEqual.call.loc168: init bool = call %bound_method.loc168_5.3(%.loc168_3.2, %a.ref.loc168)
// CHECK:STDOUT:   %int_1.loc169: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc169: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc169: %.cfa = impl_witness_access constants.%OrderedWith.impl_witness.252, element2 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.2]
// CHECK:STDOUT:   %bound_method.loc169_5.1: <bound method> = bound_method %int_1.loc169, %impl.elem2.loc169 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.bound.8ab0dc.1]
// CHECK:STDOUT:   %specific_fn.loc169: <specific function> = specific_function %impl.elem2.loc169, @T.binding.as_type.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.1]
// CHECK:STDOUT:   %bound_method.loc169_5.2: <bound method> = bound_method %int_1.loc169, %specific_fn.loc169 [concrete = constants.%bound_method.ea09f6.1]
// CHECK:STDOUT:   %.loc169_5: %T.binding.as_type.as.OrderedWith.impl.Greater.type.4ddc57.1 = specific_constant imports.%Core.Greater.696, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.1]
// CHECK:STDOUT:   %Greater.ref.loc169: %T.binding.as_type.as.OrderedWith.impl.Greater.type.4ddc57.1 = name_ref Greater, %.loc169_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.4a0e77.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.bound.loc169: <bound method> = bound_method %int_1.loc169, %Greater.ref.loc169 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.bound.8ab0dc.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc169: <specific function> = specific_function %Greater.ref.loc169, @T.binding.as_type.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.87e25a.2]
// CHECK:STDOUT:   %bound_method.loc169_5.3: <bound method> = bound_method %int_1.loc169, %T.binding.as_type.as.OrderedWith.impl.Greater.specific_fn.loc169 [concrete = constants.%bound_method.ea09f6.2]
// CHECK:STDOUT:   %impl.elem0.loc169: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc169_3: <bound method> = bound_method %int_1.loc169, %impl.elem0.loc169 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc169: init %Cpp.long = call %bound_method.loc169_3(%int_1.loc169) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc169_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc169 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc169_3.2: %Cpp.long = converted %int_1.loc169, %.loc169_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Greater.call.loc169: init bool = call %bound_method.loc169_5.3(%.loc169_3.2, %a.ref.loc169)
// CHECK:STDOUT:   %int_1.loc170: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc170: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc170_5: %.ea7 = impl_witness_access constants.%OrderedWith.impl_witness.252, element0 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.2]
// CHECK:STDOUT:   %bound_method.loc170_5.1: <bound method> = bound_method %int_1.loc170, %impl.elem0.loc170_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.bound.a0ee36.1]
// CHECK:STDOUT:   %specific_fn.loc170: <specific function> = specific_function %impl.elem0.loc170_5, @T.binding.as_type.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.1]
// CHECK:STDOUT:   %bound_method.loc170_5.2: <bound method> = bound_method %int_1.loc170, %specific_fn.loc170 [concrete = constants.%bound_method.d20c63.1]
// CHECK:STDOUT:   %.loc170_5: %T.binding.as_type.as.OrderedWith.impl.Less.type.285b47.1 = specific_constant imports.%Core.Less.b61, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.1]
// CHECK:STDOUT:   %Less.ref.loc170: %T.binding.as_type.as.OrderedWith.impl.Less.type.285b47.1 = name_ref Less, %.loc170_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.11dfa8.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.bound.loc170: <bound method> = bound_method %int_1.loc170, %Less.ref.loc170 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.bound.a0ee36.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc170: <specific function> = specific_function %Less.ref.loc170, @T.binding.as_type.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.47804e.2]
// CHECK:STDOUT:   %bound_method.loc170_5.3: <bound method> = bound_method %int_1.loc170, %T.binding.as_type.as.OrderedWith.impl.Less.specific_fn.loc170 [concrete = constants.%bound_method.d20c63.2]
// CHECK:STDOUT:   %impl.elem0.loc170_3: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc170_3: <bound method> = bound_method %int_1.loc170, %impl.elem0.loc170_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc170: init %Cpp.long = call %bound_method.loc170_3(%int_1.loc170) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc170_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc170 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc170_3.2: %Cpp.long = converted %int_1.loc170, %.loc170_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.Less.call.loc170: init bool = call %bound_method.loc170_5.3(%.loc170_3.2, %a.ref.loc170)
// CHECK:STDOUT:   %int_1.loc171: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc171: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc171: %.8f6 = impl_witness_access constants.%OrderedWith.impl_witness.252, element3 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.2]
// CHECK:STDOUT:   %bound_method.loc171_5.1: <bound method> = bound_method %int_1.loc171, %impl.elem3.loc171 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.378e29.1]
// CHECK:STDOUT:   %specific_fn.loc171: <specific function> = specific_function %impl.elem3.loc171, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.1]
// CHECK:STDOUT:   %bound_method.loc171_5.2: <bound method> = bound_method %int_1.loc171, %specific_fn.loc171 [concrete = constants.%bound_method.212360.1]
// CHECK:STDOUT:   %.loc171_5: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9cc1f0.1 = specific_constant imports.%Core.GreaterOrEquivalent.c46, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc171: %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.9cc1f0.1 = name_ref GreaterOrEquivalent, %.loc171_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.8659c0.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc171: <bound method> = bound_method %int_1.loc171, %GreaterOrEquivalent.ref.loc171 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.378e29.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc171: <specific function> = specific_function %GreaterOrEquivalent.ref.loc171, @T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.3899ef.2]
// CHECK:STDOUT:   %bound_method.loc171_5.3: <bound method> = bound_method %int_1.loc171, %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc171 [concrete = constants.%bound_method.212360.2]
// CHECK:STDOUT:   %impl.elem0.loc171: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc171_3: <bound method> = bound_method %int_1.loc171, %impl.elem0.loc171 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc171: init %Cpp.long = call %bound_method.loc171_3(%int_1.loc171) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc171_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc171 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc171_3.2: %Cpp.long = converted %int_1.loc171, %.loc171_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc171: init bool = call %bound_method.loc171_5.3(%.loc171_3.2, %a.ref.loc171)
// CHECK:STDOUT:   %int_1.loc172: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc172: %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc172: %.2ed = impl_witness_access constants.%OrderedWith.impl_witness.252, element1 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.2]
// CHECK:STDOUT:   %bound_method.loc172_5.1: <bound method> = bound_method %int_1.loc172, %impl.elem1.loc172 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.5ef5e5.1]
// CHECK:STDOUT:   %specific_fn.loc172: <specific function> = specific_function %impl.elem1.loc172, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.925a78.1]
// CHECK:STDOUT:   %bound_method.loc172_5.2: <bound method> = bound_method %int_1.loc172, %specific_fn.loc172 [concrete = constants.%bound_method.e9dda0.1]
// CHECK:STDOUT:   %.loc172_5: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8a3347.1 = specific_constant imports.%Core.LessOrEquivalent.947, @T.binding.as_type.as.OrderedWith.impl.891(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc172: %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.type.8a3347.1 = name_ref LessOrEquivalent, %.loc172_5 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.70a674.1]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc172: <bound method> = bound_method %int_1.loc172, %LessOrEquivalent.ref.loc172 [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.5ef5e5.2]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc172: <specific function> = specific_function %LessOrEquivalent.ref.loc172, @T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.eed) [concrete = constants.%T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.925a78.2]
// CHECK:STDOUT:   %bound_method.loc172_5.3: <bound method> = bound_method %int_1.loc172, %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc172 [concrete = constants.%bound_method.e9dda0.2]
// CHECK:STDOUT:   %impl.elem0.loc172: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc172_3: <bound method> = bound_method %int_1.loc172, %impl.elem0.loc172 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc172: init %Cpp.long = call %bound_method.loc172_3(%int_1.loc172) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc172_3.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc172 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc172_3.2: %Cpp.long = converted %int_1.loc172, %.loc172_3.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %T.binding.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc172: init bool = call %bound_method.loc172_5.3(%.loc172_3.2, %a.ref.loc172)
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
// CHECK:STDOUT:   %int_1.loc241: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc241_13: type = splice_block %long.ref.loc241 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc241: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc241: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc241: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc241: <bound method> = bound_method %int_1.loc241, %impl.elem0.loc241 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc241: init %Cpp.long = call %bound_method.loc241(%int_1.loc241) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc241_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc241 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc241_21.2: %Cpp.long = converted %int_1.loc241, %.loc241_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %x: %Cpp.long = value_binding x, %.loc241_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.68c = value_binding_pattern y [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc242: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc242_13: type = splice_block %long.ref.loc242 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc242: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc242: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc242: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc242: <bound method> = bound_method %int_1.loc242, %impl.elem0.loc242 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc242: init %Cpp.long = call %bound_method.loc242(%int_1.loc242) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc242_21.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc242 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc242_21.2: %Cpp.long = converted %int_1.loc242, %.loc242_21.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %y: %Cpp.long = value_binding y, %.loc242_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.68c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc244: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %impl.elem1.loc244: %.22b = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Cpp.long.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc244: <bound method> = bound_method %x.ref.loc244, %impl.elem1.loc244
// CHECK:STDOUT:   %Cpp.long.as.Negate.impl.Op.call: init %Cpp.long = call %bound_method.loc244(%x.ref.loc244)
// CHECK:STDOUT:   %.loc244_20: type = splice_block %long.ref.loc244 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc244: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc244: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc244_28.1: %Cpp.long = value_of_initializer %Cpp.long.as.Negate.impl.Op.call
// CHECK:STDOUT:   %.loc244_28.2: %Cpp.long = converted %Cpp.long.as.Negate.impl.Op.call, %.loc244_28.1
// CHECK:STDOUT:   %a: %Cpp.long = value_binding a, %.loc244_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc245: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc245: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc245: %.165 = impl_witness_access constants.%AddWith.impl_witness.e9f, element1 [concrete = constants.%Cpp.long.as.AddWith.impl.Op.be0]
// CHECK:STDOUT:   %bound_method.loc245: <bound method> = bound_method %x.ref.loc245, %impl.elem1.loc245
// CHECK:STDOUT:   %Cpp.long.as.AddWith.impl.Op.call: init %Cpp.long = call %bound_method.loc245(%x.ref.loc245, %y.ref.loc245)
// CHECK:STDOUT:   %.loc245_20: type = splice_block %long.ref.loc245 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc245: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc245: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc245_30.1: %Cpp.long = value_of_initializer %Cpp.long.as.AddWith.impl.Op.call
// CHECK:STDOUT:   %.loc245_30.2: %Cpp.long = converted %Cpp.long.as.AddWith.impl.Op.call, %.loc245_30.1
// CHECK:STDOUT:   %b: %Cpp.long = value_binding b, %.loc245_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.68c = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc246: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc246: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc246: %.571 = impl_witness_access constants.%SubWith.impl_witness.8a6, element1 [concrete = constants.%Cpp.long.as.SubWith.impl.Op.37d]
// CHECK:STDOUT:   %bound_method.loc246: <bound method> = bound_method %x.ref.loc246, %impl.elem1.loc246
// CHECK:STDOUT:   %Cpp.long.as.SubWith.impl.Op.call: init %Cpp.long = call %bound_method.loc246(%x.ref.loc246, %y.ref.loc246)
// CHECK:STDOUT:   %.loc246_20: type = splice_block %long.ref.loc246 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc246: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc246: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc246_30.1: %Cpp.long = value_of_initializer %Cpp.long.as.SubWith.impl.Op.call
// CHECK:STDOUT:   %.loc246_30.2: %Cpp.long = converted %Cpp.long.as.SubWith.impl.Op.call, %.loc246_30.1
// CHECK:STDOUT:   %c: %Cpp.long = value_binding c, %.loc246_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.68c = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc247: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc247: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc247: %.7d0 = impl_witness_access constants.%MulWith.impl_witness.87c, element1 [concrete = constants.%Cpp.long.as.MulWith.impl.Op.4e5]
// CHECK:STDOUT:   %bound_method.loc247: <bound method> = bound_method %x.ref.loc247, %impl.elem1.loc247
// CHECK:STDOUT:   %Cpp.long.as.MulWith.impl.Op.call: init %Cpp.long = call %bound_method.loc247(%x.ref.loc247, %y.ref.loc247)
// CHECK:STDOUT:   %.loc247_20: type = splice_block %long.ref.loc247 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc247: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc247: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc247_30.1: %Cpp.long = value_of_initializer %Cpp.long.as.MulWith.impl.Op.call
// CHECK:STDOUT:   %.loc247_30.2: %Cpp.long = converted %Cpp.long.as.MulWith.impl.Op.call, %.loc247_30.1
// CHECK:STDOUT:   %d: %Cpp.long = value_binding d, %.loc247_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.68c = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc248: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc248: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc248: %.aa3 = impl_witness_access constants.%DivWith.impl_witness.4ab, element1 [concrete = constants.%Cpp.long.as.DivWith.impl.Op.33f]
// CHECK:STDOUT:   %bound_method.loc248: <bound method> = bound_method %x.ref.loc248, %impl.elem1.loc248
// CHECK:STDOUT:   %Cpp.long.as.DivWith.impl.Op.call: init %Cpp.long = call %bound_method.loc248(%x.ref.loc248, %y.ref.loc248)
// CHECK:STDOUT:   %.loc248_20: type = splice_block %long.ref.loc248 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc248: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc248: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc248_30.1: %Cpp.long = value_of_initializer %Cpp.long.as.DivWith.impl.Op.call
// CHECK:STDOUT:   %.loc248_30.2: %Cpp.long = converted %Cpp.long.as.DivWith.impl.Op.call, %.loc248_30.1
// CHECK:STDOUT:   %e: %Cpp.long = value_binding e, %.loc248_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.68c = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref.loc249: %Cpp.long = name_ref x, %x
// CHECK:STDOUT:   %y.ref.loc249: %Cpp.long = name_ref y, %y
// CHECK:STDOUT:   %impl.elem1.loc249: %.c19 = impl_witness_access constants.%ModWith.impl_witness.c73, element1 [concrete = constants.%Cpp.long.as.ModWith.impl.Op.06e]
// CHECK:STDOUT:   %bound_method.loc249: <bound method> = bound_method %x.ref.loc249, %impl.elem1.loc249
// CHECK:STDOUT:   %Cpp.long.as.ModWith.impl.Op.call: init %Cpp.long = call %bound_method.loc249(%x.ref.loc249, %y.ref.loc249)
// CHECK:STDOUT:   %.loc249_20: type = splice_block %long.ref.loc249 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc249: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc249: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc249_30.1: %Cpp.long = value_of_initializer %Cpp.long.as.ModWith.impl.Op.call
// CHECK:STDOUT:   %.loc249_30.2: %Cpp.long = converted %Cpp.long.as.ModWith.impl.Op.call, %.loc249_30.1
// CHECK:STDOUT:   %f: %Cpp.long = value_binding f, %.loc249_30.2
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
// CHECK:STDOUT:   %int_1.loc508: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc508: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc508: <bound method> = bound_method %int_1.loc508, %impl.elem0.loc508 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc508: init %Cpp.long = call %bound_method.loc508(%int_1.loc508) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc508_3: init %Cpp.long = converted %int_1.loc508, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc508 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc508_3
// CHECK:STDOUT:   %.loc508_13: type = splice_block %long.ref.loc508 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc508: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc508: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.68c = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.68c = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long = var %b.var_patt
// CHECK:STDOUT:   %int_1.loc509: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc509: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc509: <bound method> = bound_method %int_1.loc509, %impl.elem0.loc509 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc509: init %Cpp.long = call %bound_method.loc509(%int_1.loc509) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc509_3: init %Cpp.long = converted %int_1.loc509, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc509 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %b.var, %.loc509_3
// CHECK:STDOUT:   %.loc509_13: type = splice_block %long.ref.loc509 [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref.loc509: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc509: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long = ref_binding b, %b.var
// CHECK:STDOUT:   %a.ref.loc511: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc511: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc511: %.045 = impl_witness_access constants.%AddAssignWith.impl_witness.fb1, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.8b7759.2]
// CHECK:STDOUT:   %bound_method.loc511_5.1: <bound method> = bound_method %a.ref.loc511, %impl.elem0.loc511
// CHECK:STDOUT:   %ImplicitAs.facet.loc511: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc511_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc511 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc511: <specific function> = specific_function %impl.elem0.loc511, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.df00b6.1]
// CHECK:STDOUT:   %bound_method.loc511_5.2: <bound method> = bound_method %a.ref.loc511, %specific_fn.loc511
// CHECK:STDOUT:   %.loc511_8: %Cpp.long = acquire_value %b.ref.loc511
// CHECK:STDOUT:   %.loc511_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.defb93.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.8b7759.1]
// CHECK:STDOUT:   %Op.ref.loc511: %Cpp.long.as.AddAssignWith.impl.Op.type.defb93.1 = name_ref Op, %.loc511_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.8b7759.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc511, %Op.ref.loc511
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc511, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.df00b6.2]
// CHECK:STDOUT:   %bound_method.loc511_5.3: <bound method> = bound_method %a.ref.loc511, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc511_5.3(%a.ref.loc511, %.loc511_8)
// CHECK:STDOUT:   %a.ref.loc512: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc512: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc512: %.052 = impl_witness_access constants.%SubAssignWith.impl_witness.631, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.31eaa0.2]
// CHECK:STDOUT:   %bound_method.loc512_5.1: <bound method> = bound_method %a.ref.loc512, %impl.elem0.loc512
// CHECK:STDOUT:   %ImplicitAs.facet.loc512: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc512_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc512 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc512: <specific function> = specific_function %impl.elem0.loc512, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.864a67.1]
// CHECK:STDOUT:   %bound_method.loc512_5.2: <bound method> = bound_method %a.ref.loc512, %specific_fn.loc512
// CHECK:STDOUT:   %.loc512_8: %Cpp.long = acquire_value %b.ref.loc512
// CHECK:STDOUT:   %.loc512_5.2: %Cpp.long.as.SubAssignWith.impl.Op.type.7e399e.1 = specific_constant imports.%Core.Op.739, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.31eaa0.1]
// CHECK:STDOUT:   %Op.ref.loc512: %Cpp.long.as.SubAssignWith.impl.Op.type.7e399e.1 = name_ref Op, %.loc512_5.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.31eaa0.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc512, %Op.ref.loc512
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc512, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.864a67.2]
// CHECK:STDOUT:   %bound_method.loc512_5.3: <bound method> = bound_method %a.ref.loc512, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc512_5.3(%a.ref.loc512, %.loc512_8)
// CHECK:STDOUT:   %a.ref.loc513: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc513: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc513: %.8d7 = impl_witness_access constants.%MulAssignWith.impl_witness.0b4, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.b067dc.2]
// CHECK:STDOUT:   %bound_method.loc513_5.1: <bound method> = bound_method %a.ref.loc513, %impl.elem0.loc513
// CHECK:STDOUT:   %ImplicitAs.facet.loc513: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc513_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc513 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc513: <specific function> = specific_function %impl.elem0.loc513, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.a37438.1]
// CHECK:STDOUT:   %bound_method.loc513_5.2: <bound method> = bound_method %a.ref.loc513, %specific_fn.loc513
// CHECK:STDOUT:   %.loc513_8: %Cpp.long = acquire_value %b.ref.loc513
// CHECK:STDOUT:   %.loc513_5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.c3673e.1 = specific_constant imports.%Core.Op.3b8, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.b067dc.1]
// CHECK:STDOUT:   %Op.ref.loc513: %Cpp.long.as.MulAssignWith.impl.Op.type.c3673e.1 = name_ref Op, %.loc513_5.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.b067dc.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc513, %Op.ref.loc513
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc513, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.a37438.2]
// CHECK:STDOUT:   %bound_method.loc513_5.3: <bound method> = bound_method %a.ref.loc513, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc513_5.3(%a.ref.loc513, %.loc513_8)
// CHECK:STDOUT:   %a.ref.loc514: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc514: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc514: %.1d5 = impl_witness_access constants.%DivAssignWith.impl_witness.b4e, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.b85335.2]
// CHECK:STDOUT:   %bound_method.loc514_5.1: <bound method> = bound_method %a.ref.loc514, %impl.elem0.loc514
// CHECK:STDOUT:   %ImplicitAs.facet.loc514: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc514_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc514 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc514: <specific function> = specific_function %impl.elem0.loc514, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.19b8da.1]
// CHECK:STDOUT:   %bound_method.loc514_5.2: <bound method> = bound_method %a.ref.loc514, %specific_fn.loc514
// CHECK:STDOUT:   %.loc514_8: %Cpp.long = acquire_value %b.ref.loc514
// CHECK:STDOUT:   %.loc514_5.2: %Cpp.long.as.DivAssignWith.impl.Op.type.97cfb4.1 = specific_constant imports.%Core.Op.f47, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.b85335.1]
// CHECK:STDOUT:   %Op.ref.loc514: %Cpp.long.as.DivAssignWith.impl.Op.type.97cfb4.1 = name_ref Op, %.loc514_5.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.b85335.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc514, %Op.ref.loc514
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc514, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.19b8da.2]
// CHECK:STDOUT:   %bound_method.loc514_5.3: <bound method> = bound_method %a.ref.loc514, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc514_5.3(%a.ref.loc514, %.loc514_8)
// CHECK:STDOUT:   %a.ref.loc515: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc515: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc515: %.474 = impl_witness_access constants.%ModAssignWith.impl_witness.faa, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.da4005.2]
// CHECK:STDOUT:   %bound_method.loc515_5.1: <bound method> = bound_method %a.ref.loc515, %impl.elem0.loc515
// CHECK:STDOUT:   %ImplicitAs.facet.loc515: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc515_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc515 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc515: <specific function> = specific_function %impl.elem0.loc515, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.51594a.1]
// CHECK:STDOUT:   %bound_method.loc515_5.2: <bound method> = bound_method %a.ref.loc515, %specific_fn.loc515
// CHECK:STDOUT:   %.loc515_8: %Cpp.long = acquire_value %b.ref.loc515
// CHECK:STDOUT:   %.loc515_5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.24a130.1 = specific_constant imports.%Core.Op.b37, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.da4005.1]
// CHECK:STDOUT:   %Op.ref.loc515: %Cpp.long.as.ModAssignWith.impl.Op.type.24a130.1 = name_ref Op, %.loc515_5.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.da4005.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc515, %Op.ref.loc515
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc515, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.51594a.2]
// CHECK:STDOUT:   %bound_method.loc515_5.3: <bound method> = bound_method %a.ref.loc515, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc515_5.3(%a.ref.loc515, %.loc515_8)
// CHECK:STDOUT:   %a.ref.loc517: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc517: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc517: %.f3c = impl_witness_access constants.%BitAndAssignWith.impl_witness.a49, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.500716.2]
// CHECK:STDOUT:   %bound_method.loc517_5.1: <bound method> = bound_method %a.ref.loc517, %impl.elem0.loc517
// CHECK:STDOUT:   %ImplicitAs.facet.loc517: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc517_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc517 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc517: <specific function> = specific_function %impl.elem0.loc517, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.1245aa.1]
// CHECK:STDOUT:   %bound_method.loc517_5.2: <bound method> = bound_method %a.ref.loc517, %specific_fn.loc517
// CHECK:STDOUT:   %.loc517_8: %Cpp.long = acquire_value %b.ref.loc517
// CHECK:STDOUT:   %.loc517_5.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.bdaab3.1 = specific_constant imports.%Core.Op.fe8, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.500716.1]
// CHECK:STDOUT:   %Op.ref.loc517: %Cpp.long.as.BitAndAssignWith.impl.Op.type.bdaab3.1 = name_ref Op, %.loc517_5.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.500716.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc517, %Op.ref.loc517
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc517, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.1245aa.2]
// CHECK:STDOUT:   %bound_method.loc517_5.3: <bound method> = bound_method %a.ref.loc517, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc517_5.3(%a.ref.loc517, %.loc517_8)
// CHECK:STDOUT:   %a.ref.loc518: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc518: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc518: %.2c9 = impl_witness_access constants.%BitOrAssignWith.impl_witness.ab4, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.2]
// CHECK:STDOUT:   %bound_method.loc518_5.1: <bound method> = bound_method %a.ref.loc518, %impl.elem0.loc518
// CHECK:STDOUT:   %ImplicitAs.facet.loc518: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc518_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc518 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc518: <specific function> = specific_function %impl.elem0.loc518, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.b7967f.1]
// CHECK:STDOUT:   %bound_method.loc518_5.2: <bound method> = bound_method %a.ref.loc518, %specific_fn.loc518
// CHECK:STDOUT:   %.loc518_8: %Cpp.long = acquire_value %b.ref.loc518
// CHECK:STDOUT:   %.loc518_5.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a2b279.1 = specific_constant imports.%Core.Op.5c2, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.1]
// CHECK:STDOUT:   %Op.ref.loc518: %Cpp.long.as.BitOrAssignWith.impl.Op.type.a2b279.1 = name_ref Op, %.loc518_5.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.9b8237.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc518, %Op.ref.loc518
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc518, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.b7967f.2]
// CHECK:STDOUT:   %bound_method.loc518_5.3: <bound method> = bound_method %a.ref.loc518, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc518_5.3(%a.ref.loc518, %.loc518_8)
// CHECK:STDOUT:   %a.ref.loc519: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc519: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc519: %.504 = impl_witness_access constants.%BitXorAssignWith.impl_witness.4d7, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.2]
// CHECK:STDOUT:   %bound_method.loc519_5.1: <bound method> = bound_method %a.ref.loc519, %impl.elem0.loc519
// CHECK:STDOUT:   %ImplicitAs.facet.loc519: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc519_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc519 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc519: <specific function> = specific_function %impl.elem0.loc519, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.04506a.1]
// CHECK:STDOUT:   %bound_method.loc519_5.2: <bound method> = bound_method %a.ref.loc519, %specific_fn.loc519
// CHECK:STDOUT:   %.loc519_8: %Cpp.long = acquire_value %b.ref.loc519
// CHECK:STDOUT:   %.loc519_5.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6f67bc.1 = specific_constant imports.%Core.Op.caf, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.1]
// CHECK:STDOUT:   %Op.ref.loc519: %Cpp.long.as.BitXorAssignWith.impl.Op.type.6f67bc.1 = name_ref Op, %.loc519_5.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.d7e378.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc519, %Op.ref.loc519
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc519, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.04506a.2]
// CHECK:STDOUT:   %bound_method.loc519_5.3: <bound method> = bound_method %a.ref.loc519, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc519_5.3(%a.ref.loc519, %.loc519_8)
// CHECK:STDOUT:   %a.ref.loc520: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc520: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc520: %.20f = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.550, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.2]
// CHECK:STDOUT:   %bound_method.loc520_5.1: <bound method> = bound_method %a.ref.loc520, %impl.elem0.loc520
// CHECK:STDOUT:   %ImplicitAs.facet.loc520: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc520_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc520 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc520: <specific function> = specific_function %impl.elem0.loc520, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0946f9.1]
// CHECK:STDOUT:   %bound_method.loc520_5.2: <bound method> = bound_method %a.ref.loc520, %specific_fn.loc520
// CHECK:STDOUT:   %.loc520_9: %Cpp.long = acquire_value %b.ref.loc520
// CHECK:STDOUT:   %.loc520_5.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.e1513f.1 = specific_constant imports.%Core.Op.030, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.1]
// CHECK:STDOUT:   %Op.ref.loc520: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.e1513f.1 = name_ref Op, %.loc520_5.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.216e26.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc520, %Op.ref.loc520
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc520, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.0946f9.2]
// CHECK:STDOUT:   %bound_method.loc520_5.3: <bound method> = bound_method %a.ref.loc520, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc520_5.3(%a.ref.loc520, %.loc520_9)
// CHECK:STDOUT:   %a.ref.loc521: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc521: ref %Cpp.long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc521: %.19a = impl_witness_access constants.%RightShiftAssignWith.impl_witness.df5, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.2]
// CHECK:STDOUT:   %bound_method.loc521_5.1: <bound method> = bound_method %a.ref.loc521, %impl.elem0.loc521
// CHECK:STDOUT:   %ImplicitAs.facet.loc521: %ImplicitAs.type.819 = facet_value constants.%Cpp.long, (constants.%ImplicitAs.impl_witness.bd4) [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %.loc521_5.1: %ImplicitAs.type.819 = converted constants.%Cpp.long, %ImplicitAs.facet.loc521 [concrete = constants.%ImplicitAs.facet.3fe]
// CHECK:STDOUT:   %specific_fn.loc521: <specific function> = specific_function %impl.elem0.loc521, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1ec453.1]
// CHECK:STDOUT:   %bound_method.loc521_5.2: <bound method> = bound_method %a.ref.loc521, %specific_fn.loc521
// CHECK:STDOUT:   %.loc521_9: %Cpp.long = acquire_value %b.ref.loc521
// CHECK:STDOUT:   %.loc521_5.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.095737.1 = specific_constant imports.%Core.Op.3b5, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.1]
// CHECK:STDOUT:   %Op.ref.loc521: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.095737.1 = name_ref Op, %.loc521_5.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.860e6c.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc521, %Op.ref.loc521
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc521, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.3fe) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.1ec453.2]
// CHECK:STDOUT:   %bound_method.loc521_5.3: <bound method> = bound_method %a.ref.loc521, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc521_5.3(%a.ref.loc521, %.loc521_9)
// CHECK:STDOUT:   %Destroy.Op.bound.loc509: <bound method> = bound_method %b.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc509: init %empty_tuple.type = call %Destroy.Op.bound.loc509(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc508: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc508: init %empty_tuple.type = call %Destroy.Op.bound.loc508(%a.var)
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
// CHECK:STDOUT:   %int_1.loc534: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc534: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc534: <bound method> = bound_method %int_1.loc534, %impl.elem0.loc534 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc534: init %Cpp.long = call %bound_method.loc534(%int_1.loc534) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc534_3: init %Cpp.long = converted %int_1.loc534, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc534 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc534_3
// CHECK:STDOUT:   %.loc534_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc535: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc535: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc535_5.1: %.3d2 = impl_witness_access constants.%AddAssignWith.impl_witness.ca1, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.2]
// CHECK:STDOUT:   %bound_method.loc535_5.1: <bound method> = bound_method %a.ref.loc535, %impl.elem0.loc535_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc535: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc535_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc535 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc535: <specific function> = specific_function %impl.elem0.loc535_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.1]
// CHECK:STDOUT:   %bound_method.loc535_5.2: <bound method> = bound_method %a.ref.loc535, %specific_fn.loc535
// CHECK:STDOUT:   %.loc535_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Op.ref.loc535: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = name_ref Op, %.loc535_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc535, %Op.ref.loc535
// CHECK:STDOUT:   %impl.elem0.loc535_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc535_5.3: <bound method> = bound_method %b.ref.loc535, %impl.elem0.loc535_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc535_5: init %Cpp.long = call %bound_method.loc535_5.3(%b.ref.loc535)
// CHECK:STDOUT:   %.loc535_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc535_5
// CHECK:STDOUT:   %.loc535_5.4: %Cpp.long = converted %b.ref.loc535, %.loc535_5.3
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc535, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.2]
// CHECK:STDOUT:   %bound_method.loc535_5.4: <bound method> = bound_method %a.ref.loc535, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc535_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc535_8: <bound method> = bound_method %b.ref.loc535, %impl.elem0.loc535_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc535_8: init %Cpp.long = call %bound_method.loc535_8(%b.ref.loc535)
// CHECK:STDOUT:   %.loc535_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc535_8
// CHECK:STDOUT:   %.loc535_8.2: %Cpp.long = converted %b.ref.loc535, %.loc535_8.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc535_5.4(%a.ref.loc535, %.loc535_8.2)
// CHECK:STDOUT:   %a.ref.loc536: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc536: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc536_5.1: %.432 = impl_witness_access constants.%SubAssignWith.impl_witness.200, element0 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.22e054.2]
// CHECK:STDOUT:   %bound_method.loc536_5.1: <bound method> = bound_method %a.ref.loc536, %impl.elem0.loc536_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc536: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc536_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc536 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc536: <specific function> = specific_function %impl.elem0.loc536_5.1, @Cpp.long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.dd31df.1]
// CHECK:STDOUT:   %bound_method.loc536_5.2: <bound method> = bound_method %a.ref.loc536, %specific_fn.loc536
// CHECK:STDOUT:   %.loc536_5.2: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cce2.1 = specific_constant imports.%Core.Op.739, @Cpp.long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.22e054.1]
// CHECK:STDOUT:   %Op.ref.loc536: %Cpp.long.as.SubAssignWith.impl.Op.type.f5cce2.1 = name_ref Op, %.loc536_5.2 [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.22e054.1]
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc536, %Op.ref.loc536
// CHECK:STDOUT:   %impl.elem0.loc536_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc536_5.3: <bound method> = bound_method %b.ref.loc536, %impl.elem0.loc536_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc536_5: init %Cpp.long = call %bound_method.loc536_5.3(%b.ref.loc536)
// CHECK:STDOUT:   %.loc536_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc536_5
// CHECK:STDOUT:   %.loc536_5.4: %Cpp.long = converted %b.ref.loc536, %.loc536_5.3
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc536, @Cpp.long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.SubAssignWith.impl.Op.specific_fn.dd31df.2]
// CHECK:STDOUT:   %bound_method.loc536_5.4: <bound method> = bound_method %a.ref.loc536, %Cpp.long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc536_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc536_8: <bound method> = bound_method %b.ref.loc536, %impl.elem0.loc536_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc536_8: init %Cpp.long = call %bound_method.loc536_8(%b.ref.loc536)
// CHECK:STDOUT:   %.loc536_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc536_8
// CHECK:STDOUT:   %.loc536_8.2: %Cpp.long = converted %b.ref.loc536, %.loc536_8.1
// CHECK:STDOUT:   %Cpp.long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc536_5.4(%a.ref.loc536, %.loc536_8.2)
// CHECK:STDOUT:   %a.ref.loc537: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc537: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc537_5.1: %.fbe = impl_witness_access constants.%MulAssignWith.impl_witness.c16, element0 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.2]
// CHECK:STDOUT:   %bound_method.loc537_5.1: <bound method> = bound_method %a.ref.loc537, %impl.elem0.loc537_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc537: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc537_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc537 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc537: <specific function> = specific_function %impl.elem0.loc537_5.1, @Cpp.long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c127b2.1]
// CHECK:STDOUT:   %bound_method.loc537_5.2: <bound method> = bound_method %a.ref.loc537, %specific_fn.loc537
// CHECK:STDOUT:   %.loc537_5.2: %Cpp.long.as.MulAssignWith.impl.Op.type.d2f0ca.1 = specific_constant imports.%Core.Op.3b8, @Cpp.long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.1]
// CHECK:STDOUT:   %Op.ref.loc537: %Cpp.long.as.MulAssignWith.impl.Op.type.d2f0ca.1 = name_ref Op, %.loc537_5.2 [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.8b6fd2.1]
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc537, %Op.ref.loc537
// CHECK:STDOUT:   %impl.elem0.loc537_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc537_5.3: <bound method> = bound_method %b.ref.loc537, %impl.elem0.loc537_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc537_5: init %Cpp.long = call %bound_method.loc537_5.3(%b.ref.loc537)
// CHECK:STDOUT:   %.loc537_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc537_5
// CHECK:STDOUT:   %.loc537_5.4: %Cpp.long = converted %b.ref.loc537, %.loc537_5.3
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc537, @Cpp.long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.MulAssignWith.impl.Op.specific_fn.c127b2.2]
// CHECK:STDOUT:   %bound_method.loc537_5.4: <bound method> = bound_method %a.ref.loc537, %Cpp.long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc537_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc537_8: <bound method> = bound_method %b.ref.loc537, %impl.elem0.loc537_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc537_8: init %Cpp.long = call %bound_method.loc537_8(%b.ref.loc537)
// CHECK:STDOUT:   %.loc537_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc537_8
// CHECK:STDOUT:   %.loc537_8.2: %Cpp.long = converted %b.ref.loc537, %.loc537_8.1
// CHECK:STDOUT:   %Cpp.long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc537_5.4(%a.ref.loc537, %.loc537_8.2)
// CHECK:STDOUT:   %a.ref.loc538: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc538: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc538_5.1: %.2e1 = impl_witness_access constants.%DivAssignWith.impl_witness.b4b, element0 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2f8f46.2]
// CHECK:STDOUT:   %bound_method.loc538_5.1: <bound method> = bound_method %a.ref.loc538, %impl.elem0.loc538_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc538: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc538_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc538 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc538: <specific function> = specific_function %impl.elem0.loc538_5.1, @Cpp.long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ee65e7.1]
// CHECK:STDOUT:   %bound_method.loc538_5.2: <bound method> = bound_method %a.ref.loc538, %specific_fn.loc538
// CHECK:STDOUT:   %.loc538_5.2: %Cpp.long.as.DivAssignWith.impl.Op.type.16271d.1 = specific_constant imports.%Core.Op.f47, @Cpp.long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2f8f46.1]
// CHECK:STDOUT:   %Op.ref.loc538: %Cpp.long.as.DivAssignWith.impl.Op.type.16271d.1 = name_ref Op, %.loc538_5.2 [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.2f8f46.1]
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc538, %Op.ref.loc538
// CHECK:STDOUT:   %impl.elem0.loc538_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc538_5.3: <bound method> = bound_method %b.ref.loc538, %impl.elem0.loc538_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc538_5: init %Cpp.long = call %bound_method.loc538_5.3(%b.ref.loc538)
// CHECK:STDOUT:   %.loc538_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc538_5
// CHECK:STDOUT:   %.loc538_5.4: %Cpp.long = converted %b.ref.loc538, %.loc538_5.3
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc538, @Cpp.long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.DivAssignWith.impl.Op.specific_fn.ee65e7.2]
// CHECK:STDOUT:   %bound_method.loc538_5.4: <bound method> = bound_method %a.ref.loc538, %Cpp.long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc538_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc538_8: <bound method> = bound_method %b.ref.loc538, %impl.elem0.loc538_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc538_8: init %Cpp.long = call %bound_method.loc538_8(%b.ref.loc538)
// CHECK:STDOUT:   %.loc538_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc538_8
// CHECK:STDOUT:   %.loc538_8.2: %Cpp.long = converted %b.ref.loc538, %.loc538_8.1
// CHECK:STDOUT:   %Cpp.long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc538_5.4(%a.ref.loc538, %.loc538_8.2)
// CHECK:STDOUT:   %a.ref.loc539: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc539: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc539_5.1: %.c03 = impl_witness_access constants.%ModAssignWith.impl_witness.a13, element0 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.f69ad5.2]
// CHECK:STDOUT:   %bound_method.loc539_5.1: <bound method> = bound_method %a.ref.loc539, %impl.elem0.loc539_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc539: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc539_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc539 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc539: <specific function> = specific_function %impl.elem0.loc539_5.1, @Cpp.long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.73e9bb.1]
// CHECK:STDOUT:   %bound_method.loc539_5.2: <bound method> = bound_method %a.ref.loc539, %specific_fn.loc539
// CHECK:STDOUT:   %.loc539_5.2: %Cpp.long.as.ModAssignWith.impl.Op.type.e2a2e6.1 = specific_constant imports.%Core.Op.b37, @Cpp.long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.f69ad5.1]
// CHECK:STDOUT:   %Op.ref.loc539: %Cpp.long.as.ModAssignWith.impl.Op.type.e2a2e6.1 = name_ref Op, %.loc539_5.2 [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.f69ad5.1]
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc539, %Op.ref.loc539
// CHECK:STDOUT:   %impl.elem0.loc539_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc539_5.3: <bound method> = bound_method %b.ref.loc539, %impl.elem0.loc539_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc539_5: init %Cpp.long = call %bound_method.loc539_5.3(%b.ref.loc539)
// CHECK:STDOUT:   %.loc539_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc539_5
// CHECK:STDOUT:   %.loc539_5.4: %Cpp.long = converted %b.ref.loc539, %.loc539_5.3
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc539, @Cpp.long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.ModAssignWith.impl.Op.specific_fn.73e9bb.2]
// CHECK:STDOUT:   %bound_method.loc539_5.4: <bound method> = bound_method %a.ref.loc539, %Cpp.long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc539_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc539_8: <bound method> = bound_method %b.ref.loc539, %impl.elem0.loc539_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc539_8: init %Cpp.long = call %bound_method.loc539_8(%b.ref.loc539)
// CHECK:STDOUT:   %.loc539_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc539_8
// CHECK:STDOUT:   %.loc539_8.2: %Cpp.long = converted %b.ref.loc539, %.loc539_8.1
// CHECK:STDOUT:   %Cpp.long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc539_5.4(%a.ref.loc539, %.loc539_8.2)
// CHECK:STDOUT:   %a.ref.loc541: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc541: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc541_5.1: %.a81 = impl_witness_access constants.%BitAndAssignWith.impl_witness.349, element0 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.2]
// CHECK:STDOUT:   %bound_method.loc541_5.1: <bound method> = bound_method %a.ref.loc541, %impl.elem0.loc541_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc541: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc541_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc541 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc541: <specific function> = specific_function %impl.elem0.loc541_5.1, @Cpp.long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.560221.1]
// CHECK:STDOUT:   %bound_method.loc541_5.2: <bound method> = bound_method %a.ref.loc541, %specific_fn.loc541
// CHECK:STDOUT:   %.loc541_5.2: %Cpp.long.as.BitAndAssignWith.impl.Op.type.aa6f07.1 = specific_constant imports.%Core.Op.fe8, @Cpp.long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.1]
// CHECK:STDOUT:   %Op.ref.loc541: %Cpp.long.as.BitAndAssignWith.impl.Op.type.aa6f07.1 = name_ref Op, %.loc541_5.2 [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.5f705c.1]
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc541, %Op.ref.loc541
// CHECK:STDOUT:   %impl.elem0.loc541_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc541_5.3: <bound method> = bound_method %b.ref.loc541, %impl.elem0.loc541_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc541_5: init %Cpp.long = call %bound_method.loc541_5.3(%b.ref.loc541)
// CHECK:STDOUT:   %.loc541_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc541_5
// CHECK:STDOUT:   %.loc541_5.4: %Cpp.long = converted %b.ref.loc541, %.loc541_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc541, @Cpp.long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn.560221.2]
// CHECK:STDOUT:   %bound_method.loc541_5.4: <bound method> = bound_method %a.ref.loc541, %Cpp.long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc541_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc541_8: <bound method> = bound_method %b.ref.loc541, %impl.elem0.loc541_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc541_8: init %Cpp.long = call %bound_method.loc541_8(%b.ref.loc541)
// CHECK:STDOUT:   %.loc541_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc541_8
// CHECK:STDOUT:   %.loc541_8.2: %Cpp.long = converted %b.ref.loc541, %.loc541_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc541_5.4(%a.ref.loc541, %.loc541_8.2)
// CHECK:STDOUT:   %a.ref.loc542: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc542: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc542_5.1: %.1f8 = impl_witness_access constants.%BitOrAssignWith.impl_witness.119, element0 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.73e223.2]
// CHECK:STDOUT:   %bound_method.loc542_5.1: <bound method> = bound_method %a.ref.loc542, %impl.elem0.loc542_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc542: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc542_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc542 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc542: <specific function> = specific_function %impl.elem0.loc542_5.1, @Cpp.long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.665975.1]
// CHECK:STDOUT:   %bound_method.loc542_5.2: <bound method> = bound_method %a.ref.loc542, %specific_fn.loc542
// CHECK:STDOUT:   %.loc542_5.2: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fac636.1 = specific_constant imports.%Core.Op.5c2, @Cpp.long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.73e223.1]
// CHECK:STDOUT:   %Op.ref.loc542: %Cpp.long.as.BitOrAssignWith.impl.Op.type.fac636.1 = name_ref Op, %.loc542_5.2 [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.73e223.1]
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc542, %Op.ref.loc542
// CHECK:STDOUT:   %impl.elem0.loc542_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc542_5.3: <bound method> = bound_method %b.ref.loc542, %impl.elem0.loc542_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc542_5: init %Cpp.long = call %bound_method.loc542_5.3(%b.ref.loc542)
// CHECK:STDOUT:   %.loc542_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc542_5
// CHECK:STDOUT:   %.loc542_5.4: %Cpp.long = converted %b.ref.loc542, %.loc542_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc542, @Cpp.long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn.665975.2]
// CHECK:STDOUT:   %bound_method.loc542_5.4: <bound method> = bound_method %a.ref.loc542, %Cpp.long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc542_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc542_8: <bound method> = bound_method %b.ref.loc542, %impl.elem0.loc542_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc542_8: init %Cpp.long = call %bound_method.loc542_8(%b.ref.loc542)
// CHECK:STDOUT:   %.loc542_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc542_8
// CHECK:STDOUT:   %.loc542_8.2: %Cpp.long = converted %b.ref.loc542, %.loc542_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc542_5.4(%a.ref.loc542, %.loc542_8.2)
// CHECK:STDOUT:   %a.ref.loc543: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc543: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc543_5.1: %.9ab = impl_witness_access constants.%BitXorAssignWith.impl_witness.5aa, element0 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.616e38.2]
// CHECK:STDOUT:   %bound_method.loc543_5.1: <bound method> = bound_method %a.ref.loc543, %impl.elem0.loc543_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc543: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc543_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc543 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc543: <specific function> = specific_function %impl.elem0.loc543_5.1, @Cpp.long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.4cd193.1]
// CHECK:STDOUT:   %bound_method.loc543_5.2: <bound method> = bound_method %a.ref.loc543, %specific_fn.loc543
// CHECK:STDOUT:   %.loc543_5.2: %Cpp.long.as.BitXorAssignWith.impl.Op.type.53faad.1 = specific_constant imports.%Core.Op.caf, @Cpp.long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.616e38.1]
// CHECK:STDOUT:   %Op.ref.loc543: %Cpp.long.as.BitXorAssignWith.impl.Op.type.53faad.1 = name_ref Op, %.loc543_5.2 [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.616e38.1]
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc543, %Op.ref.loc543
// CHECK:STDOUT:   %impl.elem0.loc543_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc543_5.3: <bound method> = bound_method %b.ref.loc543, %impl.elem0.loc543_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc543_5: init %Cpp.long = call %bound_method.loc543_5.3(%b.ref.loc543)
// CHECK:STDOUT:   %.loc543_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc543_5
// CHECK:STDOUT:   %.loc543_5.4: %Cpp.long = converted %b.ref.loc543, %.loc543_5.3
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc543, @Cpp.long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn.4cd193.2]
// CHECK:STDOUT:   %bound_method.loc543_5.4: <bound method> = bound_method %a.ref.loc543, %Cpp.long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc543_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc543_8: <bound method> = bound_method %b.ref.loc543, %impl.elem0.loc543_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc543_8: init %Cpp.long = call %bound_method.loc543_8(%b.ref.loc543)
// CHECK:STDOUT:   %.loc543_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc543_8
// CHECK:STDOUT:   %.loc543_8.2: %Cpp.long = converted %b.ref.loc543, %.loc543_8.1
// CHECK:STDOUT:   %Cpp.long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc543_5.4(%a.ref.loc543, %.loc543_8.2)
// CHECK:STDOUT:   %a.ref.loc544: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc544: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc544_5.1: %.424 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.199, element0 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.2]
// CHECK:STDOUT:   %bound_method.loc544_5.1: <bound method> = bound_method %a.ref.loc544, %impl.elem0.loc544_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc544: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc544_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc544 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc544: <specific function> = specific_function %impl.elem0.loc544_5.1, @Cpp.long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.3bd66d.1]
// CHECK:STDOUT:   %bound_method.loc544_5.2: <bound method> = bound_method %a.ref.loc544, %specific_fn.loc544
// CHECK:STDOUT:   %.loc544_5.2: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.426a0e.1 = specific_constant imports.%Core.Op.030, @Cpp.long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.1]
// CHECK:STDOUT:   %Op.ref.loc544: %Cpp.long.as.LeftShiftAssignWith.impl.Op.type.426a0e.1 = name_ref Op, %.loc544_5.2 [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.e25b25.1]
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc544, %Op.ref.loc544
// CHECK:STDOUT:   %impl.elem0.loc544_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc544_5.3: <bound method> = bound_method %b.ref.loc544, %impl.elem0.loc544_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc544_5: init %Cpp.long = call %bound_method.loc544_5.3(%b.ref.loc544)
// CHECK:STDOUT:   %.loc544_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc544_5
// CHECK:STDOUT:   %.loc544_5.4: %Cpp.long = converted %b.ref.loc544, %.loc544_5.3
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc544, @Cpp.long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn.3bd66d.2]
// CHECK:STDOUT:   %bound_method.loc544_5.4: <bound method> = bound_method %a.ref.loc544, %Cpp.long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc544_9: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc544_9: <bound method> = bound_method %b.ref.loc544, %impl.elem0.loc544_9
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc544_9: init %Cpp.long = call %bound_method.loc544_9(%b.ref.loc544)
// CHECK:STDOUT:   %.loc544_9.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc544_9
// CHECK:STDOUT:   %.loc544_9.2: %Cpp.long = converted %b.ref.loc544, %.loc544_9.1
// CHECK:STDOUT:   %Cpp.long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc544_5.4(%a.ref.loc544, %.loc544_9.2)
// CHECK:STDOUT:   %a.ref.loc545: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc545: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc545_5.1: %.eaa = impl_witness_access constants.%RightShiftAssignWith.impl_witness.e4e, element0 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.2]
// CHECK:STDOUT:   %bound_method.loc545_5.1: <bound method> = bound_method %a.ref.loc545, %impl.elem0.loc545_5.1
// CHECK:STDOUT:   %ImplicitAs.facet.loc545: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc545_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet.loc545 [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc545: <specific function> = specific_function %impl.elem0.loc545_5.1, @Cpp.long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c3db.1]
// CHECK:STDOUT:   %bound_method.loc545_5.2: <bound method> = bound_method %a.ref.loc545, %specific_fn.loc545
// CHECK:STDOUT:   %.loc545_5.2: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2dfa9.1 = specific_constant imports.%Core.Op.3b5, @Cpp.long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.1]
// CHECK:STDOUT:   %Op.ref.loc545: %Cpp.long.as.RightShiftAssignWith.impl.Op.type.f2dfa9.1 = name_ref Op, %.loc545_5.2 [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.56db42.1]
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc545, %Op.ref.loc545
// CHECK:STDOUT:   %impl.elem0.loc545_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc545_5.3: <bound method> = bound_method %b.ref.loc545, %impl.elem0.loc545_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc545_5: init %Cpp.long = call %bound_method.loc545_5.3(%b.ref.loc545)
// CHECK:STDOUT:   %.loc545_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc545_5
// CHECK:STDOUT:   %.loc545_5.4: %Cpp.long = converted %b.ref.loc545, %.loc545_5.3
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc545, @Cpp.long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn.11c3db.2]
// CHECK:STDOUT:   %bound_method.loc545_5.4: <bound method> = bound_method %a.ref.loc545, %Cpp.long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc545_9: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc545_9: <bound method> = bound_method %b.ref.loc545, %impl.elem0.loc545_9
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc545_9: init %Cpp.long = call %bound_method.loc545_9(%b.ref.loc545)
// CHECK:STDOUT:   %.loc545_9.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc545_9
// CHECK:STDOUT:   %.loc545_9.2: %Cpp.long = converted %b.ref.loc545, %.loc545_9.1
// CHECK:STDOUT:   %Cpp.long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc545_5.4(%a.ref.loc545, %.loc545_9.2)
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
// CHECK:STDOUT:   %int_1.loc557: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc557: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc557: <bound method> = bound_method %int_1.loc557, %impl.elem0.loc557 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc557: init %Cpp.long = call %bound_method.loc557(%int_1.loc557) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc557_3: init %Cpp.long = converted %int_1.loc557, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc557 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc557_3
// CHECK:STDOUT:   %.loc557_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc558: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc558_5.1: %.2e2 = impl_witness_access constants.%AddAssignWith.impl_witness.623, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.5fb128.2]
// CHECK:STDOUT:   %bound_method.loc558_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc558_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.2ce) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc558_5.1: %ImplicitAs.type.819 = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc558_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3a3075.1]
// CHECK:STDOUT:   %bound_method.loc558_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc558_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.a474b7.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.5fb128.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.a474b7.1 = name_ref Op, %.loc558_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.5fb128.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc558_5.2: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc558_5.3: <bound method> = bound_method %int_1.loc558, %impl.elem0.loc558_5.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc558_5: init %Cpp.long = call %bound_method.loc558_5.3(%int_1.loc558) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc558_5.3: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc558_5 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc558_5.4: %Cpp.long = converted %int_1.loc558, %.loc558_5.3 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.3a3075.2]
// CHECK:STDOUT:   %bound_method.loc558_5.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc558_8: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc558_8: <bound method> = bound_method %int_1.loc558, %impl.elem0.loc558_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc558_8: init %Cpp.long = call %bound_method.loc558_8(%int_1.loc558) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc558_8.1: %Cpp.long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc558_8 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc558_8.2: %Cpp.long = converted %int_1.loc558, %.loc558_8.1 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc558_5.4(%a.ref, %.loc558_8.2)
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
// CHECK:STDOUT:   %int_1.loc570: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc570: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f2a]
// CHECK:STDOUT:   %bound_method.loc570: <bound method> = bound_method %int_1.loc570, %impl.elem0.loc570 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cc8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc570: init %Cpp.long = call %bound_method.loc570(%int_1.loc570) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc570_3: init %Cpp.long = converted %int_1.loc570, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc570 [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc570_3
// CHECK:STDOUT:   %.loc570_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc571: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc571: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc571_16.1: <bound method> = bound_method %int_1.loc571, %impl.elem0.loc571 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc571: <specific function> = specific_function %impl.elem0.loc571, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc571_16.2: <bound method> = bound_method %int_1.loc571, %specific_fn.loc571 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc571: init %i32 = call %bound_method.loc571_16.2(%int_1.loc571) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc571_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc571 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc571_16.2: %i32 = converted %int_1.loc571, %.loc571_16.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %b: %i32 = value_binding b, %.loc571_16.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc572_5.1: %.3d2 = impl_witness_access constants.%AddAssignWith.impl_witness.ca1, element0 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.2]
// CHECK:STDOUT:   %bound_method.loc572_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc572_5.1
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.819 = facet_value constants.%i32, (constants.%ImplicitAs.impl_witness.0fc) [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %.loc572_5.1: %ImplicitAs.type.819 = converted constants.%i32, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet.174]
// CHECK:STDOUT:   %specific_fn.loc572: <specific function> = specific_function %impl.elem0.loc572_5.1, @Cpp.long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.1]
// CHECK:STDOUT:   %bound_method.loc572_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc572
// CHECK:STDOUT:   %.loc572_5.2: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = specific_constant imports.%Core.Op.f81, @Cpp.long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long.as.AddAssignWith.impl.Op.type.8bd196.1 = name_ref Op, %.loc572_5.2 [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.d19046.1]
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %impl.elem0.loc572_5.2: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc572_5.3: <bound method> = bound_method %b.ref, %impl.elem0.loc572_5.2
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc572_5: init %Cpp.long = call %bound_method.loc572_5.3(%b.ref)
// CHECK:STDOUT:   %.loc572_5.3: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc572_5
// CHECK:STDOUT:   %.loc572_5.4: %Cpp.long = converted %b.ref, %.loc572_5.3
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.174) [concrete = constants.%Cpp.long.as.AddAssignWith.impl.Op.specific_fn.2a1dd4.2]
// CHECK:STDOUT:   %bound_method.loc572_5.4: <bound method> = bound_method %a.ref, %Cpp.long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc572_8: %.bb8 = impl_witness_access constants.%ImplicitAs.impl_witness.0fc, element0 [concrete = constants.%i32.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc572_8: <bound method> = bound_method %b.ref, %impl.elem0.loc572_8
// CHECK:STDOUT:   %i32.as.ImplicitAs.impl.Convert.call.loc572_8: init %Cpp.long = call %bound_method.loc572_8(%b.ref)
// CHECK:STDOUT:   %.loc572_8.1: %Cpp.long = value_of_initializer %i32.as.ImplicitAs.impl.Convert.call.loc572_8
// CHECK:STDOUT:   %.loc572_8.2: %Cpp.long = converted %b.ref, %.loc572_8.1
// CHECK:STDOUT:   %Cpp.long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc572_5.4(%a.ref, %.loc572_8.2)
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
// CHECK:STDOUT:   %impl.elem0.loc614: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc614: <bound method> = bound_method %int_1, %impl.elem0.loc614 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long = call %bound_method.loc614(%int_1) [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   %.loc614_3: init %Cpp.long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5a4]
// CHECK:STDOUT:   assign %a.var, %.loc614_3
// CHECK:STDOUT:   %.loc614_13: type = splice_block %long.ref [concrete = constants.%Cpp.long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref: type = name_ref long, constants.%Cpp.long [concrete = constants.%Cpp.long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc615: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc615: %.314 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc615: <bound method> = bound_method %a.ref.loc615, %impl.elem0.loc615
// CHECK:STDOUT:   %Cpp.long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc615(%a.ref.loc615)
// CHECK:STDOUT:   %a.ref.loc616: ref %Cpp.long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc616: %.253 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc616: <bound method> = bound_method %a.ref.loc616, %impl.elem0.loc616
// CHECK:STDOUT:   %Cpp.long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc616(%a.ref.loc616)
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
// CHECK:STDOUT:   %int_1.loc635: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc635_29: type = splice_block %unsigned_long.ref.loc635 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc635: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc635: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc635: %.541 = impl_witness_access constants.%ImplicitAs.impl_witness.a19, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.14b]
// CHECK:STDOUT:   %bound_method.loc635: <bound method> = bound_method %int_1.loc635, %impl.elem0.loc635 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.92f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc635: init %Cpp.unsigned_long = call %bound_method.loc635(%int_1.loc635) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %.loc635_46.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc635 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %.loc635_46.2: %Cpp.unsigned_long = converted %int_1.loc635, %.loc635_46.1 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %cpp_unsigned_long: %Cpp.unsigned_long = value_binding cpp_unsigned_long, %.loc635_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_result.patt: %pattern_type.6f2 = value_binding_pattern cpp_unsigned_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc636_58: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc636: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc636: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc636_89.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc636: %ptr.f5e = addr_of %.loc636_89.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc636: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.1(%cpp_unsigned_long.ref.loc636, %addr.loc636)
// CHECK:STDOUT:   %.loc636_89.2: init %ULongResult to %.loc636_89.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc636
// CHECK:STDOUT:   %.loc636_43: type = splice_block %ULongResult.ref.loc636 [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc636_40: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref.loc636: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc636_89.3: ref %ULongResult = temporary %.loc636_89.1, %.loc636_89.2
// CHECK:STDOUT:   %.loc636_89.4: %ULongResult = acquire_value %.loc636_89.3
// CHECK:STDOUT:   %cpp_unsigned_long_result: %ULongResult = value_binding cpp_unsigned_long_result, %.loc636_89.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong.patt: %pattern_type.5b7 = value_binding_pattern cpp_compat_ulong [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref.loc638: %Cpp.unsigned_long = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %.loc638: type = splice_block %ULong32.ref [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.4b4 [concrete = imports.%CppCompat.4b4]
// CHECK:STDOUT:     %ULong32.ref: type = name_ref ULong32, imports.%Core.ULong32 [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_ulong: %Cpp.unsigned_long = value_binding cpp_compat_ulong, %cpp_unsigned_long.ref.loc638
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_result.patt: %pattern_type.6f2 = value_binding_pattern cpp_compat_ulong_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc639_57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc639: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong.ref: %Cpp.unsigned_long = name_ref cpp_compat_ulong, %cpp_compat_ulong
// CHECK:STDOUT:   %.loc639_87.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc639: %ptr.f5e = addr_of %.loc639_87.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc639: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.1(%cpp_compat_ulong.ref, %addr.loc639)
// CHECK:STDOUT:   %.loc639_87.2: init %ULongResult to %.loc639_87.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc639
// CHECK:STDOUT:   %.loc639_42: type = splice_block %ULongResult.ref.loc639 [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc639_39: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref.loc639: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc639_87.3: ref %ULongResult = temporary %.loc639_87.1, %.loc639_87.2
// CHECK:STDOUT:   %.loc639_87.4: %ULongResult = acquire_value %.loc639_87.3
// CHECK:STDOUT:   %cpp_compat_ulong_result: %ULongResult = value_binding cpp_compat_ulong_result, %.loc639_87.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u32.patt: %pattern_type.4a9 = value_binding_pattern carbon_u32 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc641: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u32: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %impl.elem0.loc641: %.b47 = impl_witness_access constants.%ImplicitAs.impl_witness.29e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.90e]
// CHECK:STDOUT:   %bound_method.loc641_25.1: <bound method> = bound_method %int_1.loc641, %impl.elem0.loc641 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.412]
// CHECK:STDOUT:   %specific_fn.loc641: <specific function> = specific_function %impl.elem0.loc641, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc641_25.2: <bound method> = bound_method %int_1.loc641, %specific_fn.loc641 [concrete = constants.%bound_method.21d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc641: init %u32 = call %bound_method.loc641_25.2(%int_1.loc641) [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc641_25.1: %u32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc641 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc641_25.2: %u32 = converted %int_1.loc641, %.loc641_25.1 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %carbon_u32: %u32 = value_binding carbon_u32, %.loc641_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u32_result.patt: %pattern_type.dc7 = value_binding_pattern carbon_u32_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc642_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULong.ref.loc642: %PassULong.cpp_overload_set.type = name_ref PassULong, imports.%PassULong.cpp_overload_set.value [concrete = constants.%PassULong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u32.ref: %u32 = name_ref carbon_u32, %carbon_u32
// CHECK:STDOUT:   %.loc642_74.1: ref %UIntResult = temporary_storage
// CHECK:STDOUT:   %addr.loc642: %ptr.059 = addr_of %.loc642_74.1
// CHECK:STDOUT:   %PassULong__carbon_thunk.call.loc642: init %empty_tuple.type = call imports.%PassULong__carbon_thunk.decl.108234.2(%carbon_u32.ref, %addr.loc642)
// CHECK:STDOUT:   %.loc642_74.2: init %UIntResult to %.loc642_74.1 = mark_in_place_init %PassULong__carbon_thunk.call.loc642
// CHECK:STDOUT:   %.loc642_36: type = splice_block %UIntResult.ref [concrete = constants.%UIntResult] {
// CHECK:STDOUT:     %Cpp.ref.loc642_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %UIntResult.ref: type = name_ref UIntResult, imports.%UIntResult.decl [concrete = constants.%UIntResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc642_74.3: ref %UIntResult = temporary %.loc642_74.1, %.loc642_74.2
// CHECK:STDOUT:   %.loc642_74.4: %UIntResult = acquire_value %.loc642_74.3
// CHECK:STDOUT:   %carbon_u32_result: %UIntResult = value_binding carbon_u32_result, %.loc642_74.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc644_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc644_59: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc644_59.1: <bound method> = bound_method %float, %impl.elem0.loc644_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc644: <specific function> = specific_function %impl.elem0.loc644_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc644_59.2: <bound method> = bound_method %float, %specific_fn.loc644 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc644_59.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc644_59.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc644_59.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc644_59.4: init %f32.97e to %.loc644_59.3 = in_place_init %.loc644_59.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc644_59.5: init %array_type to %a.var = array_init (%.loc644_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc644_3: init %array_type = converted %.loc644_59.1, %.loc644_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc644_3
// CHECK:STDOUT:   %.loc644_50: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %int_1.loc644: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc644: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc644: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:     %impl.elem0.loc644_30.1: %.59b = impl_witness_access constants.%As.impl_witness.d5f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc644_30.1: <bound method> = bound_method %int_1.loc644, %impl.elem0.loc644_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc644_30.1(%int_1.loc644) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %.loc644_30.1: %Cpp.unsigned_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %.loc644_30.2: %Cpp.unsigned_long = converted %int_1.loc644, %.loc644_30.1 [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:     %impl.elem0.loc644_30.2: %.0db = impl_witness_access constants.%ImplicitAs.impl_witness.a13, element0 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc644_30.2: <bound method> = bound_method %.loc644_30.2, %impl.elem0.loc644_30.2 [concrete = constants.%Cpp.unsigned_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc644_30.2(%.loc644_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc644_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc644_30.4: Core.IntLiteral = converted %.loc644_30.2, %.loc644_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc644_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   %UIntResult.cpp_destructor.bound: <bound method> = bound_method %.loc642_74.3, constants.%UIntResult.cpp_destructor
// CHECK:STDOUT:   %UIntResult.cpp_destructor.call: init %empty_tuple.type = call %UIntResult.cpp_destructor.bound(%.loc642_74.3)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound.loc639: <bound method> = bound_method %.loc639_87.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call.loc639: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound.loc639(%.loc639_87.3)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound.loc636: <bound method> = bound_method %.loc636_89.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call.loc636: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound.loc636(%.loc636_89.3)
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
// CHECK:STDOUT:   %impl.elem0.loc681: %.541 = impl_witness_access constants.%ImplicitAs.impl_witness.a19, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc681: <bound method> = bound_method %int_1, %impl.elem0.loc681 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long = call %bound_method.loc681(%int_1) [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   %.loc681_3: init %Cpp.unsigned_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0ab]
// CHECK:STDOUT:   assign %a.var, %.loc681_3
// CHECK:STDOUT:   %.loc681_13: type = splice_block %unsigned_long.ref.loc681 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc681: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc681: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.unsigned_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.5b7 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.5b7 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.unsigned_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.unsigned_long = name_ref a, %a
// CHECK:STDOUT:   %.loc682_37: %Cpp.unsigned_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc682: %.f45 = impl_witness_access constants.%Copy.impl_witness.ae3, element0 [concrete = constants.%Cpp.unsigned_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc682: <bound method> = bound_method %.loc682_37, %impl.elem0.loc682
// CHECK:STDOUT:   %Cpp.unsigned_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long = call %bound_method.loc682(%.loc682_37)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc682_20: type = splice_block %unsigned_long.ref.loc682 [concrete = constants.%Cpp.unsigned_long] {
// CHECK:STDOUT:     %Cpp.ref.loc682: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc682: type = name_ref unsigned_long, constants.%Cpp.unsigned_long [concrete = constants.%Cpp.unsigned_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc682: <bound method> = bound_method %b.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc682: init %empty_tuple.type = call %Destroy.Op.bound.loc682(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc681: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc681: init %empty_tuple.type = call %Destroy.Op.bound.loc681(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Cpp.unsigned_long) = "no_op";
// CHECK:STDOUT:
