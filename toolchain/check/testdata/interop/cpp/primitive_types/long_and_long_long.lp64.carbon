// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/primitive_types/long_and_long_long.lp64.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/primitive_types/long_and_long_long.lp64.carbon

// --- long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  let cpp_long: Cpp.long = 1;
  let unused carbon_long: i64 = cpp_long;

  var unused a: array(f32, 1 as Cpp.long) = (1.0,);
  //@dump-sem-ir-end
}

// --- unsigned_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  //@dump-sem-ir-begin
  let cpp_unsigned_long: Cpp.unsigned_long = 1;
  let unused carbon_unsigned_long: u64 = cpp_unsigned_long;

  var unused a: array(f32, 1 as Cpp.unsigned_long) = (1.0,);
  //@dump-sem-ir-end
}

// --- long_long.carbon

library "[[@TEST_NAME]]";

import Cpp inline
'''
class LongLongResult {};
auto PassLongLong(long long x) -> LongLongResult;

class LongResult {};
auto PassLongLong(long x) -> LongResult;
''';

fn F() {
  //@dump-sem-ir-begin
  let cpp_long_long: Cpp.long_long = 1;
  let unused cpp_long_long_result: Cpp.LongLongResult = Cpp.PassLongLong(cpp_long_long);

  let cpp_compat_long_long: Core.CppCompat.LongLong64 = cpp_long_long;
  let unused cpp_compat_long_long_result: Cpp.LongLongResult = Cpp.PassLongLong(cpp_compat_long_long);

  let carbon_i64: i64 = cpp_long_long as i64;
  let unused carbon_i64_result: Cpp.LongResult = Cpp.PassLongLong(carbon_i64);

  var unused a: array(f32, 1 as Cpp.long_long) = (1.0,);
  //@dump-sem-ir-end
}


// --- fail_implicit_conversion_long_long_to_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  let cpp_long_long: Cpp.long_long = 1;
  // CHECK:STDERR: fail_implicit_conversion_long_long_to_i64.carbon:[[@LINE+7]]:32: error: cannot implicitly convert expression of type `Cpp.long_long` to `i64` [ConversionFailure]
  // CHECK:STDERR:   let unused carbon_i64: i64 = cpp_long_long;
  // CHECK:STDERR:                                ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_implicit_conversion_long_long_to_i64.carbon:[[@LINE+4]]:32: note: type `Cpp.long_long` does not implement interface `Core.ImplicitAs(i64)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused carbon_i64: i64 = cpp_long_long;
  // CHECK:STDERR:                                ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused carbon_i64: i64 = cpp_long_long;
}

// --- copy_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CopyLongLong() {
  //@dump-sem-ir-begin
  var a: Cpp.long_long = 1;
  var unused b: Cpp.long_long = a;
  //@dump-sem-ir-end
}

// --- comparisons_homogeneous_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHomogeneousLongLong() {
  //@dump-sem-ir-begin
  let a: Cpp.long_long = 1;
  let b: Cpp.long_long = 1;
  a == b;
  a != b;
  a > b;
  a < b;
  a >= b;
  a <= b;
  //@dump-sem-ir-end
}

// --- comparisons_heterogeneous_long_long_left_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongLongLeftSide() {
  let b: i64 = 1;
  //@dump-sem-ir-begin
  let a: Cpp.long_long = 1;
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

// --- comparisons_heterogeneous_long_long_right_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongLongRightSide() {
  let b: i64 = 1;
  //@dump-sem-ir-begin
  let a: Cpp.long_long = 1;
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

// --- fail_todo_comparisons_heterogeneous_long_long_and_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongLongAndI32() {
  let a: Cpp.long_long = 1;
  let b: i32 = 1;
  // CHECK:STDERR: fail_todo_comparisons_heterogeneous_long_long_and_i32.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.EqWith(i32)` in type `Cpp.long_long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a == b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  a == b;
  // CHECK:STDERR: fail_todo_comparisons_heterogeneous_long_long_and_i32.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.EqWith(Cpp.long_long)` in type `i32` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   b == a;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  b == a;
}

// --- comparisons_heterogeneous_long_long_and_i128.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongLongAndI128() {
  let a: Cpp.long_long = 1;
  let b: i128 = 1;
  a == b;
  b == a;
}

// --- fail_todo_comparisons_heterogeneous_long_long_and_i256.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn ComparisonsHeterogeneousLongLongAndI256() {
  let a: Cpp.long_long = 1;
  let b: i256 = 1;
  // CHECK:STDERR: fail_todo_comparisons_heterogeneous_long_long_and_i256.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.EqWith(i256)` in type `Cpp.long_long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a == b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  a == b;
  // CHECK:STDERR: fail_todo_comparisons_heterogeneous_long_long_and_i256.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.EqWith(Cpp.long_long)` in type `i256` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   b == a;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  b == a;
}

// --- arithmetic_homogeneous_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn ArithmeticHomogeneousLongLong() {
  let x: Cpp.long_long = 1;
  let y: Cpp.long_long = 1;

  AssertSameType(-x, x);
  AssertSameType(x + y, x);
  AssertSameType(x - y, x);
  AssertSameType(x * y, x);
  AssertSameType(x / y, x);
  AssertSameType(x % y, x);
}

// --- arithmetic_heterogeneous_long_long_left_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongLongLeftSide() {
  let b: i64 = 1;
  let x: Cpp.long_long = 1;

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

// --- arithmetic_heterogeneous_long_long_right_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongLongRightSide() {
  let b: i64 = 1;
  let x: Cpp.long_long = 1;
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

// --- fail_todo_arithmetic_heterogeneous_long_long_and_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongLongAndI32() {
  let x: Cpp.long_long = 1;
  let y: i32 = 1;
  // CHECK:STDERR: fail_todo_arithmetic_heterogeneous_long_long_and_i32.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.AddWith(i32)` in type `Cpp.long_long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(x + y, x);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(x + y, x);

  // CHECK:STDERR: fail_todo_arithmetic_heterogeneous_long_long_and_i32.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.AddWith(Cpp.long_long)` in type `i32` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(y + x, x);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(y + x, x);
}

// --- arithmetic_heterogeneous_long_long_and_i128.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn ArithmeticHeterogeneousLongLongAndI128() {
  let x: Cpp.long_long = 1;
  let y: i128 = 1;
  AssertSameType(x + y, y);
  AssertSameType(y + x, y);
}

// --- bitwise_homogeneous_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn BitWiseHomogeneousLongLong() {
  let a: Cpp.long_long = 1;
  let b: Cpp.long_long = 1;

  AssertSameType(^a, a);
  AssertSameType(a & b, a);
  AssertSameType(a | b, a);
  AssertSameType(a ^ b, a);
  AssertSameType(a << b, a);
  AssertSameType(a >> b, a);
}

// --- bitwise_heterogeneous_long_long_left_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongLongLeftSide() {
  let b: i64 = 1;
  let a: Cpp.long_long = 1;

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

// --- bitwise_heterogeneous_long_long_right_side.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongLongRightSide() {
  let b: i64 = 1;
  let a: Cpp.long_long = 1;

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

// --- fail_todo_bitwise_heterogeneous_long_long_and_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongLongAndI32() {
  let a: Cpp.long_long = 1;
  let b: i32 = 1;

  // CHECK:STDERR: fail_todo_bitwise_heterogeneous_long_long_and_i32.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.BitAndWith(i32)` in type `Cpp.long_long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(a & b, a);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(a & b, a);
  // CHECK:STDERR: fail_todo_bitwise_heterogeneous_long_long_and_i32.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.BitAndWith(Cpp.long_long)` in type `i32` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   AssertSameType(b & a, a);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  AssertSameType(b & a, a);
}

// --- bitwise_heterogeneous_long_long_and_i128.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn AssertSameType[T:! type](unused a: T, unused b: T) {}

fn BitWiseHeterogeneousLongLongAndI128() {
  let a: Cpp.long_long = 1;
  let b: i128 = 1;

  AssertSameType(a & b, b);
  AssertSameType(b & a, b);
}

// --- compound_assignment_homogeneous_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentHomogeneousLongLong() {
  //@dump-sem-ir-begin
  var a: Cpp.long_long = 1;
  var b: Cpp.long_long = 1;

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

// --- compound_assignment_hereogeneous_long_long_and_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentLongLongAndI64() {
  let b: i64 = 1;
  //@dump-sem-ir-begin
  var a: Cpp.long_long = 1;
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

// --- compound_assignment_heterogeneous_long_long_and_int_literal.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentLongLongAndIntLiteral() {
  //@dump-sem-ir-begin
  var a: Cpp.long_long = 1;
  a += 1;
  //@dump-sem-ir-end
}

// --- compound_assignment_heterogeneous_long_long_and_runtime_i64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentLongLongAndRuntimeI64() {
  //@dump-sem-ir-begin
  var a: Cpp.long_long = 1;
  let b: i64 = 1;
  a += b;
  //@dump-sem-ir-end
}

// --- fail_todo_compound_assignment_heterogeneous_long_long_and_i32.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentHeterogeneousLongLongAndI32() {
  var a: Cpp.long_long = 1;
  // CHECK:STDERR: fail_todo_compound_assignment_heterogeneous_long_long_and_i32.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.AddAssignWith(i32)` in type `Cpp.long_long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a += (1 as i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  a += (1 as i32);
}

// --- fail_compound_assignment_heterogeneous_long_long_and_i128.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CompoundAssignmentHeterogeneousLongLongAndI128() {
  var a: Cpp.long_long = 1;
  // CHECK:STDERR: fail_compound_assignment_heterogeneous_long_long_and_i128.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.AddAssignWith(i128)` in type `Cpp.long_long` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a += (1 as i128);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  a += (1 as i128);
}

// --- increment_decrement_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn IncrementDecrementLongLong() {
  //@dump-sem-ir-begin
  var a: Cpp.long_long = 1;
  ++a;
  --a;
  //@dump-sem-ir-end
}

// --- unsigned_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp inline
'''
class ULongLongResult {};
auto PassULongLong(unsigned long long x) -> ULongLongResult;

class ULongResult {};
auto PassULongLong(unsigned long x) -> ULongResult;
''';

fn F() {
  //@dump-sem-ir-begin
  let cpp_unsigned_long_long: Cpp.unsigned_long_long = 1;
  let unused cpp_unsigned_long_long_result: Cpp.ULongLongResult = Cpp.PassULongLong(cpp_unsigned_long_long);

  let cpp_compat_ulong_long: Core.CppCompat.ULongLong64 = cpp_unsigned_long_long;
  let unused cpp_compat_ulong_long_result: Cpp.ULongLongResult = Cpp.PassULongLong(cpp_compat_ulong_long);

  let carbon_u64: u64 = 1;
  let unused carbon_u64_result: Cpp.ULongResult = Cpp.PassULongLong(carbon_u64);

  var unused a: array(f32, 1 as Cpp.unsigned_long_long) = (1.0,);
  //@dump-sem-ir-end
}

// --- fail_pass_ulonglong_to_u64.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {
  let cpp_unsigned_long_long: Cpp.unsigned_long_long = 1;

  // For portability reasons, this is not allowed even though `unsigned long
  // long` happens to fit in `u64` on this target.
  // CHECK:STDERR: fail_pass_ulonglong_to_u64.carbon:[[@LINE+11]]:25: error: cannot implicitly convert expression of type `Cpp.unsigned_long_long` to `u64` [ConversionFailure]
  // CHECK:STDERR:   let carbon_u64: u64 = cpp_unsigned_long_long;
  // CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_pass_ulonglong_to_u64.carbon:[[@LINE+8]]:25: note: type `Cpp.unsigned_long_long` does not implement interface `Core.ImplicitAs(u64)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let carbon_u64: u64 = cpp_unsigned_long_long;
  // CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_pass_ulonglong_to_u64.carbon:[[@LINE+4]]:7: warning: binding `carbon_u64` unused [UnusedBinding]
  // CHECK:STDERR:   let carbon_u64: u64 = cpp_unsigned_long_long;
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let carbon_u64: u64 = cpp_unsigned_long_long;
}

// --- copy_unsigned_long_long.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn CopyUnsignedLongLong() {
  //@dump-sem-ir-begin
  var a: Cpp.unsigned_long_long = 1;
  var unused b: Cpp.unsigned_long_long = a;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.f57: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.221: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.48d: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.0f9: %Int.as.ImplicitAs.impl.Convert.type.48d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.091: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6e8: %ImplicitAs.type.221 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.091) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.6e8) [concrete]
// CHECK:STDOUT:   %.f34: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a04, %ImplicitAs.facet.6e8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.c5e: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d98: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.998: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c5d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.6ee: %Core.IntLiteral.as.As.impl.Convert.type.c5d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.2e1: <witness> = impl_witness imports.%As.impl_witness_table.1c5, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.1a1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3d9: %Core.IntLiteral.as.As.impl.Convert.type.1a1 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.998 = facet_value Core.IntLiteral, (%As.impl_witness.2e1) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2b4: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i64, %As.facet) [concrete]
// CHECK:STDOUT:   %.01b: type = fn_type_with_self_type %As.WithSelf.Convert.type.2b4, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.3d9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.3d9, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.991: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b8b, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.ad9: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.be5: %Int.as.ImplicitAs.impl.Convert.type.ad9 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cee: %ImplicitAs.type.7cb = facet_value %i64, (%ImplicitAs.impl_witness.a75) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.004: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.cee) [concrete]
// CHECK:STDOUT:   %.779: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.004, %ImplicitAs.facet.cee [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.d98, %Int.as.ImplicitAs.impl.Convert.be5 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.be5, @Int.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.e21: <bound method> = bound_method %int_1.d98, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.c73 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ed3: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.c73)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.303: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.393, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.65e: %ImplicitAs.type.ed3 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.303) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.2e4: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.c73, %ImplicitAs.facet.65e) [concrete]
// CHECK:STDOUT:   %.17b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.2e4, %ImplicitAs.facet.65e [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3e4: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.943: %f32.c73 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.943) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long = @F.%i64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.32e: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.48d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.0f9)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b8b = impl_witness_table (%Core.import_ref.32e), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.64b: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c5d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.6ee)]
// CHECK:STDOUT:   %As.impl_witness_table.1c5 = impl_witness_table (%Core.import_ref.64b), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.393 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long.patt: %pattern_type.f57 = value_binding_pattern cpp_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_20: type = splice_block %long.ref.loc8 [concrete = constants.%i64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %long.ref.loc8: type = name_ref long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.f34 = impl_witness_access constants.%ImplicitAs.impl_witness.091, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b]
// CHECK:STDOUT:   %bound_method.loc8_28.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_28.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.c5e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i64 = call %bound_method.loc8_28.2(%int_1.loc8) [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc8_28.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc8_28.2: %i64 = converted %int_1.loc8, %.loc8_28.1 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %cpp_long: %i64 = value_binding cpp_long, %.loc8_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_long.patt: %pattern_type.f57 = value_binding_pattern carbon_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long.ref: %i64 = name_ref cpp_long, %cpp_long
// CHECK:STDOUT:   %i64.loc9: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_long: %i64 = value_binding carbon_long, %cpp_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_50.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_50: %.17b = impl_witness_access constants.%ImplicitAs.impl_witness.303, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
// CHECK:STDOUT:   %bound_method.loc11_50.1: <bound method> = bound_method %float, %impl.elem0.loc11_50 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_50: <specific function> = specific_function %impl.elem0.loc11_50, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.2: <bound method> = bound_method %float, %specific_fn.loc11_50 [concrete = constants.%bound_method.3e4]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.c73 = call %bound_method.loc11_50.2(%float) [concrete = constants.%float.943]
// CHECK:STDOUT:   %.loc11_50.2: init %f32.c73 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.943]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_50.3: ref %f32.c73 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_50.4: init %f32.c73 to %.loc11_50.3 = in_place_init %.loc11_50.2 [concrete = constants.%float.943]
// CHECK:STDOUT:   %.loc11_50.5: init %array_type to %a.var = array_init (%.loc11_50.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_50.1, %.loc11_50.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_41: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.c73 [concrete = constants.%f32.c73]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long.ref.loc11: type = name_ref long, %i64.2 [concrete = constants.%i64]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.01b = impl_witness_access constants.%As.impl_witness.2e1, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.3d9]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.991]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.d98]
// CHECK:STDOUT:     %.loc11_30.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.d98]
// CHECK:STDOUT:     %.loc11_30.2: %i64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.779 = impl_witness_access constants.%ImplicitAs.impl_witness.a75, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.be5]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.e21]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.2(%self.param: ref %f32.c73) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.3(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsigned_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.b35: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.14f: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.b68(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.170: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.3ec: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.3d3(%From.fe9) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.06e: %UInt.as.ImplicitAs.impl.Convert.type.3ec = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9c4: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.42f, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b18: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.633: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b18 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d29: %ImplicitAs.type.14f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.9c4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8cc: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.d29) [concrete]
// CHECK:STDOUT:   %.93e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8cc, %ImplicitAs.facet.d29 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.633 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.633, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.7e7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.01f: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.72f: type = facet_type <@As, @As(%u64)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.fb3: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.f86: %Core.IntLiteral.as.As.impl.Convert.type.fb3 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.8fb: <witness> = impl_witness imports.%As.impl_witness_table.8fa, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.49f: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.990: %Core.IntLiteral.as.As.impl.Convert.type.49f = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.72f = facet_value Core.IntLiteral, (%As.impl_witness.8fb) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.8dd: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u64, %As.facet) [concrete]
// CHECK:STDOUT:   %.04a: type = fn_type_with_self_type %As.WithSelf.Convert.type.8dd, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.990 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.990, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.318: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b02: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a83, @UInt.as.ImplicitAs.impl.3d3(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.cd8: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.3d3(%int_64) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.7d0: %UInt.as.ImplicitAs.impl.Convert.type.cd8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e41: %ImplicitAs.type.7cb = facet_value %u64, (%ImplicitAs.impl_witness.b02) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.e41) [concrete]
// CHECK:STDOUT:   %.637: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da8, %ImplicitAs.facet.e41 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.01f, %UInt.as.ImplicitAs.impl.Convert.7d0 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.7d0, @UInt.as.ImplicitAs.impl.Convert.1(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.b37: <bound method> = bound_method %int_1.01f, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.c73 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ed3: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.c73)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.303: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.393, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.65e: %ImplicitAs.type.ed3 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.303) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.2e4: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.c73, %ImplicitAs.facet.65e) [concrete]
// CHECK:STDOUT:   %.17b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.2e4, %ImplicitAs.facet.65e [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3e4: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.943: %f32.c73 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.943) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long = @F.%u64.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.aa5: @Core.IntLiteral.as.ImplicitAs.impl.b68.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b68.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.170)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.42f = impl_witness_table (%Core.import_ref.aa5), @Core.IntLiteral.as.ImplicitAs.impl.b68 [concrete]
// CHECK:STDOUT:   %Core.import_ref.baa: @UInt.as.ImplicitAs.impl.3d3.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.3ec) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.3d3.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.06e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a83 = impl_witness_table (%Core.import_ref.baa), @UInt.as.ImplicitAs.impl.3d3 [concrete]
// CHECK:STDOUT:   %Core.import_ref.59a: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.fb3) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.f86)]
// CHECK:STDOUT:   %As.impl_witness_table.8fa = impl_witness_table (%Core.import_ref.59a), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.393 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long.patt: %pattern_type.b35 = value_binding_pattern cpp_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_29: type = splice_block %unsigned_long.ref.loc8 [concrete = constants.%u64] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %unsigned_long.ref.loc8: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.93e = impl_witness_access constants.%ImplicitAs.impl_witness.9c4, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.633]
// CHECK:STDOUT:   %bound_method.loc8_46.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_46.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.7e7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u64 = call %bound_method.loc8_46.2(%int_1.loc8) [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc8_46.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc8_46.2: %u64 = converted %int_1.loc8, %.loc8_46.1 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %cpp_unsigned_long: %u64 = value_binding cpp_unsigned_long, %.loc8_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_unsigned_long.patt: %pattern_type.b35 = value_binding_pattern carbon_unsigned_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long.ref: %u64 = name_ref cpp_unsigned_long, %cpp_unsigned_long
// CHECK:STDOUT:   %u64.loc9: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %carbon_unsigned_long: %u64 = value_binding carbon_unsigned_long, %cpp_unsigned_long.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc11_59.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc11_59: %.17b = impl_witness_access constants.%ImplicitAs.impl_witness.303, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
// CHECK:STDOUT:   %bound_method.loc11_59.1: <bound method> = bound_method %float, %impl.elem0.loc11_59 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_59: <specific function> = specific_function %impl.elem0.loc11_59, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_59.2: <bound method> = bound_method %float, %specific_fn.loc11_59 [concrete = constants.%bound_method.3e4]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.c73 = call %bound_method.loc11_59.2(%float) [concrete = constants.%float.943]
// CHECK:STDOUT:   %.loc11_59.2: init %f32.c73 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.943]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_59.3: ref %f32.c73 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_59.4: init %f32.c73 to %.loc11_59.3 = in_place_init %.loc11_59.2 [concrete = constants.%float.943]
// CHECK:STDOUT:   %.loc11_59.5: init %array_type to %a.var = array_init (%.loc11_59.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type = converted %.loc11_59.1, %.loc11_59.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_50: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.c73 [concrete = constants.%f32.c73]
// CHECK:STDOUT:     %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long.ref.loc11: type = name_ref unsigned_long, %u64.2 [concrete = constants.%u64]
// CHECK:STDOUT:     %impl.elem0.loc11_30.1: %.04a = impl_witness_access constants.%As.impl_witness.8fb, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.990]
// CHECK:STDOUT:     %bound_method.loc11_30.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.1: <specific function> = specific_function %impl.elem0.loc11_30.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_30.1 [concrete = constants.%bound_method.318]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u64 = call %bound_method.loc11_30.2(%int_1.loc11) [concrete = constants.%int_1.01f]
// CHECK:STDOUT:     %.loc11_30.1: %u64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.01f]
// CHECK:STDOUT:     %.loc11_30.2: %u64 = converted %int_1.loc11, %.loc11_30.1 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:     %impl.elem0.loc11_30.2: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.b02, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.7d0]
// CHECK:STDOUT:     %bound_method.loc11_30.3: <bound method> = bound_method %.loc11_30.2, %impl.elem0.loc11_30.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_30.2: <specific function> = specific_function %impl.elem0.loc11_30.2, @UInt.as.ImplicitAs.impl.Convert.1(constants.%int_64) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_30.4: <bound method> = bound_method %.loc11_30.2, %specific_fn.loc11_30.2 [concrete = constants.%bound_method.b37]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_30.4(%.loc11_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc11_30.4: Core.IntLiteral = converted %.loc11_30.2, %.loc11_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.2(%self.param: ref %f32.c73) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.3(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.660: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2ed: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.660) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.062: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %.ca2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.062, %ImplicitAs.facet.2ed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %LongLongResult: type = class_type @LongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.357: type = pattern_type %LongLongResult [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.662: type = ptr_type %LongLongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.4c6a2a.1: type = fn_type @PassLongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.8eb04c.1: %PassLongLong__carbon_thunk.type.4c6a2a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f57: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %LongResult: type = class_type @LongResult [concrete]
// CHECK:STDOUT:   %pattern_type.ae7: type = pattern_type %LongResult [concrete]
// CHECK:STDOUT:   %ptr.aec: type = ptr_type %LongResult [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.type.4c6a2a.2: type = fn_type @PassLongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.8eb04c.2: %PassLongLong__carbon_thunk.type.4c6a2a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.d39: type = facet_type <@As, @As(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.cc2: <witness> = impl_witness imports.%As.impl_witness_table.742 [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d39 = facet_value Core.IntLiteral, (%As.impl_witness.cc2) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.ee9: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.8af: type = fn_type_with_self_type %As.WithSelf.Convert.type.ee9, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f86: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.da7 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.474: %ImplicitAs.type.7cb = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.f86) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.aef: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.474) [concrete]
// CHECK:STDOUT:   %.c52: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.aef, %ImplicitAs.facet.474 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.b19, %Cpp.long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.c73 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ed3: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.c73)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.303: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.393, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.65e: %ImplicitAs.type.ed3 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.303) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.2e4: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.c73, %ImplicitAs.facet.65e) [concrete]
// CHECK:STDOUT:   %.17b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.2e4, %ImplicitAs.facet.65e [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.943: %f32.c73 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.943) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc24_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor.type: type = fn_type @LongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongResult.cpp_destructor: %LongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.type: type = fn_type @LongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %LongLongResult.cpp_destructor: %LongLongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .CppCompat = %CppCompat.f0dbf9.1
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     .LongLongResult = %LongLongResult.decl
// CHECK:STDOUT:     .PassLongLong = %PassLongLong.cpp_overload_set.value
// CHECK:STDOUT:     .LongResult = %LongResult.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.f0dbf9.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .LongLong64 = %Core.LongLong64
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.LongLong64: type = import_ref Core//prelude/types/cpp/int, LongLong64, loaded [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d6d = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %LongLongResult.decl: type = class_decl @LongLongResult [concrete = constants.%LongLongResult] {} {}
// CHECK:STDOUT:   %PassLongLong.cpp_overload_set.value: %PassLongLong.cpp_overload_set.type = cpp_overload_set_value @PassLongLong.cpp_overload_set [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.204cd1.1: %PassLongLong__carbon_thunk.type.4c6a2a.1 = fn_decl @PassLongLong__carbon_thunk.1 [concrete = constants.%PassLongLong__carbon_thunk.8eb04c.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %LongResult.decl: type = class_decl @LongResult [concrete = constants.%LongResult] {} {}
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.decl.204cd1.2: %PassLongLong__carbon_thunk.type.4c6a2a.2 = fn_decl @PassLongLong__carbon_thunk.2 [concrete = constants.%PassLongLong__carbon_thunk.8eb04c.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.742 = impl_witness_table (%Core.import_ref.06c923.2), @Core.IntLiteral.as.As.impl.084 [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Cpp.long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.da7 = impl_witness_table (%Core.import_ref.06c923.3), @Cpp.long_long.as.ImplicitAs.impl.c90 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.393 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long.patt: %pattern_type.d58 = value_binding_pattern cpp_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_25: type = splice_block %long_long.ref.loc15 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc15: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc15_38.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc15_38.2: %Cpp.long_long = converted %int_1.loc15, %.loc15_38.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %cpp_long_long: %Cpp.long_long = value_binding cpp_long_long, %.loc15_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_long_long_result.patt: %pattern_type.357 = value_binding_pattern cpp_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_57: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc16: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_long_long.ref.loc16: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc16_87.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.662 = addr_of %.loc16_87.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.204cd1.1(%cpp_long_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_87.2: init %LongLongResult to %.loc16_87.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_39: type = splice_block %LongLongResult.ref.loc16 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc16: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_87.3: ref %LongLongResult = temporary %.loc16_87.1, %.loc16_87.2
// CHECK:STDOUT:   %.loc16_87.4: %LongLongResult = acquire_value %.loc16_87.3
// CHECK:STDOUT:   %cpp_long_long_result: %LongLongResult = value_binding cpp_long_long_result, %.loc16_87.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_long.patt: %pattern_type.d58 = value_binding_pattern cpp_compat_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc18: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %.loc18: type = splice_block %LongLong64.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.f0dbf9.1 [concrete = imports.%CppCompat.f0dbf9.1]
// CHECK:STDOUT:     %LongLong64.ref: type = name_ref LongLong64, imports.%Core.LongLong64 [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_long_long: %Cpp.long_long = value_binding cpp_compat_long_long, %cpp_long_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_long_long_result.patt: %pattern_type.357 = value_binding_pattern cpp_compat_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_64: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc19: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_long_long.ref: %Cpp.long_long = name_ref cpp_compat_long_long, %cpp_compat_long_long
// CHECK:STDOUT:   %.loc19_101.1: ref %LongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.662 = addr_of %.loc19_101.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.204cd1.1(%cpp_compat_long_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_101.2: init %LongLongResult to %.loc19_101.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_46: type = splice_block %LongLongResult.ref.loc19 [concrete = constants.%LongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongLongResult.ref.loc19: type = name_ref LongLongResult, imports.%LongLongResult.decl [concrete = constants.%LongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_101.3: ref %LongLongResult = temporary %.loc19_101.1, %.loc19_101.2
// CHECK:STDOUT:   %.loc19_101.4: %LongLongResult = acquire_value %.loc19_101.3
// CHECK:STDOUT:   %cpp_compat_long_long_result: %LongLongResult = value_binding cpp_compat_long_long_result, %.loc19_101.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64.patt: %pattern_type.f57 = value_binding_pattern carbon_i64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_long_long.ref.loc21: %Cpp.long_long = name_ref cpp_long_long, %cpp_long_long
// CHECK:STDOUT:   %i64.loc21_42: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %.loc21_39.1: %i64 = as_compatible %cpp_long_long.ref.loc21
// CHECK:STDOUT:   %.loc21_39.2: %i64 = converted %cpp_long_long.ref.loc21, %.loc21_39.1
// CHECK:STDOUT:   %i64.loc21_19: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %carbon_i64: %i64 = value_binding carbon_i64, %.loc21_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_i64_result.patt: %pattern_type.ae7 = value_binding_pattern carbon_i64_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassLongLong.ref.loc22: %PassLongLong.cpp_overload_set.type = name_ref PassLongLong, imports.%PassLongLong.cpp_overload_set.value [concrete = constants.%PassLongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_i64.ref: %i64 = name_ref carbon_i64, %carbon_i64
// CHECK:STDOUT:   %.loc22_77.1: ref %LongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.aec = addr_of %.loc22_77.1
// CHECK:STDOUT:   %PassLongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassLongLong__carbon_thunk.decl.204cd1.2(%carbon_i64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_77.2: init %LongResult to %.loc22_77.1 = mark_in_place_init %PassLongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_36: type = splice_block %LongResult.ref [concrete = constants.%LongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %LongResult.ref: type = name_ref LongResult, imports.%LongResult.decl [concrete = constants.%LongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_77.3: ref %LongResult = temporary %.loc22_77.1, %.loc22_77.2
// CHECK:STDOUT:   %.loc22_77.4: %LongResult = acquire_value %.loc22_77.3
// CHECK:STDOUT:   %carbon_i64_result: %LongResult = value_binding carbon_i64_result, %.loc22_77.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_55.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_55: %.17b = impl_witness_access constants.%ImplicitAs.impl_witness.303, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
// CHECK:STDOUT:   %bound_method.loc24_55.1: <bound method> = bound_method %float, %impl.elem0.loc24_55 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc24_55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_55.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.c73 = call %bound_method.loc24_55.2(%float) [concrete = constants.%float.943]
// CHECK:STDOUT:   %.loc24_55.2: init %f32.c73 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.943]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_55.3: ref %f32.c73 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_55.4: init %f32.c73 to %.loc24_55.3 = in_place_init %.loc24_55.2 [concrete = constants.%float.943]
// CHECK:STDOUT:   %.loc24_55.5: init %array_type to %a.var = array_init (%.loc24_55.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_55.1, %.loc24_55.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_46: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.c73 [concrete = constants.%f32.c73]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc24: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.8af = impl_witness_access constants.%As.impl_witness.cc2, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.b19]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.c52 = impl_witness_access constants.%ImplicitAs.impl_witness.f86, element0 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   %LongResult.cpp_destructor.bound: <bound method> = bound_method %.loc22_77.3, constants.%LongResult.cpp_destructor
// CHECK:STDOUT:   %LongResult.cpp_destructor.call: init %empty_tuple.type = call %LongResult.cpp_destructor.bound(%.loc22_77.3)
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_101.3, constants.%LongLongResult.cpp_destructor
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.call.loc19: init %empty_tuple.type = call %LongLongResult.cpp_destructor.bound.loc19(%.loc19_101.3)
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.bound.loc16: <bound method> = bound_method %.loc16_87.3, constants.%LongLongResult.cpp_destructor
// CHECK:STDOUT:   %LongLongResult.cpp_destructor.call.loc16: init %empty_tuple.type = call %LongLongResult.cpp_destructor.bound.loc16(%.loc16_87.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.2(%self.param: ref %f32.c73) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.3(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.660: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.660) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.062: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ca2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.062, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5c7: <witness> = impl_witness imports.%Copy.impl_witness_table.a14 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.5c7) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.4bd: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.3c3: type = fn_type_with_self_type %Copy.WithSelf.Op.type.4bd, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op: %Cpp.long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d6d = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.a14 = impl_witness_table (%Core.import_ref.9c3), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d58 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.d58 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_33: %Cpp.long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.3c3 = impl_witness_access constants.%Copy.impl_witness.5c7, element0 [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_33, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.call: init %Cpp.long_long = call %bound_method.loc9(%.loc9_33)
// CHECK:STDOUT:   assign %b.var, %Cpp.long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_homogeneous_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.660: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.660) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.062: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ca2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.062, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.9fc: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.77a: <witness> = impl_witness imports.%EqWith.impl_witness_table.ed1 [concrete]
// CHECK:STDOUT:   %EqWith.facet.757: %EqWith.type.9fc = facet_value %Cpp.long_long, (%EqWith.impl_witness.77a) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.140: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.757) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.eb1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.757) [concrete]
// CHECK:STDOUT:   %.ac4: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.140, %EqWith.facet.757 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.78c: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.1ca: %Cpp.long_long.as.EqWith.impl.Equal.type.78c = struct_value () [concrete]
// CHECK:STDOUT:   %.e72: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.eb1, %EqWith.facet.757 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.a6a: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.type.456: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.4b7: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.cc0 [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.cd7: %OrderedWith.type.456 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.4b7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.ddc: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.cd7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.d85: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.cd7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.232: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.cd7) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.1ff: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.cd7) [concrete]
// CHECK:STDOUT:   %.4cd: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.232, %OrderedWith.facet.cd7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.725: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.7c8: %Cpp.long_long.as.OrderedWith.impl.Greater.type.725 = struct_value () [concrete]
// CHECK:STDOUT:   %.c03: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.ddc, %OrderedWith.facet.cd7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.335: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.606: %Cpp.long_long.as.OrderedWith.impl.Less.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %.ce5: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.1ff, %OrderedWith.facet.cd7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = struct_value () [concrete]
// CHECK:STDOUT:   %.663: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.d85, %OrderedWith.facet.cd7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.42c: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0c8: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.42c = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d6d = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.a87: %Cpp.long_long.as.EqWith.impl.Equal.type.78c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %Core.import_ref.adc9: %Cpp.long_long.as.EqWith.impl.NotEqual.type.8e5 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %EqWith.impl_witness_table.ed1 = impl_witness_table (%Core.import_ref.a87, %Core.import_ref.adc9), @Cpp.long_long.as.EqWith.impl.eaa [concrete]
// CHECK:STDOUT:   %Core.import_ref.0e8: %Cpp.long_long.as.OrderedWith.impl.Less.type.335 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %Core.import_ref.757: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.42c = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %Core.import_ref.c32c: %Cpp.long_long.as.OrderedWith.impl.Greater.type.725 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %Core.import_ref.381: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2d9 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.cc0 = impl_witness_table (%Core.import_ref.0e8, %Core.import_ref.757, %Core.import_ref.c32c, %Core.import_ref.381), @Cpp.long_long.as.OrderedWith.impl.278 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHomogeneousLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc8: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_26.2: %Cpp.long_long = converted %int_1.loc8, %.loc8_26.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc8_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d58 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %b: %Cpp.long_long = value_binding b, %.loc9_26.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10: %.ac4 = impl_witness_access constants.%EqWith.impl_witness.77a, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.1ca]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call: init bool = call %bound_method.loc10(%a.ref.loc10, %b.ref.loc10)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.e72 = impl_witness_access constants.%EqWith.impl_witness.77a, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.a6a]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call: init bool = call %bound_method.loc11(%a.ref.loc11, %b.ref.loc11)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2: %.4cd = impl_witness_access constants.%OrderedWith.impl_witness.4b7, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.7c8]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem2
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call: init bool = call %bound_method.loc12(%a.ref.loc12, %b.ref.loc12)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.c03 = impl_witness_access constants.%OrderedWith.impl_witness.4b7, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.606]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc13(%a.ref.loc13, %b.ref.loc13)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3: %.ce5 = impl_witness_access constants.%OrderedWith.impl_witness.4b7, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.a09]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.ref.loc14, %impl.elem3
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call: init bool = call %bound_method.loc14(%a.ref.loc14, %b.ref.loc14)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.663 = impl_witness_access constants.%OrderedWith.impl_witness.4b7, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0c8]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call: init bool = call %bound_method.loc15(%a.ref.loc15, %b.ref.loc15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_heterogeneous_long_long_left_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.660: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2ed: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.660) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.062: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %.ca2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.062, %ImplicitAs.facet.2ed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.d67: type = facet_type <@EqWith, @EqWith(%i64)> [concrete]
// CHECK:STDOUT:   %T.bf2: %ImplicitAs.type.0ec = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.600(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.4b5f60.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.600(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.4b5f60.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.600(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.ec6d64.1: %Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.600(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.ec6d64.2: %Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.aee: type = facet_type <@EqWith, @EqWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.305: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a87 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7a4: %ImplicitAs.type.0ec = facet_value %i64, (%ImplicitAs.impl_witness.305) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.621: <witness> = impl_witness imports.%EqWith.impl_witness_table.efb, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.73aeb2.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.01ccf4.1: %Cpp.long_long.as.EqWith.impl.Equal.type.73aeb2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.7f7ea7.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.ac575d.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.7f7ea7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.73aeb2.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.01ccf4.2: %Cpp.long_long.as.EqWith.impl.Equal.type.73aeb2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.7f7ea7.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.ac575d.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.7f7ea7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.016: %EqWith.type.d67 = facet_value %Cpp.long_long, (%EqWith.impl_witness.621) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.e1d: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%i64, %EqWith.facet.016) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.0d6: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%i64, %EqWith.facet.016) [concrete]
// CHECK:STDOUT:   %.5d2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.e1d, %EqWith.facet.016 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.2dd31b.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.01ccf4.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.2dd31b.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.01ccf4.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3ae: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %.25e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3ae, %ImplicitAs.facet.7a4 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c98: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.0d6, %EqWith.facet.016 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.38af67.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.ac575d.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.38af67.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.ac575d.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.af1: type = facet_type <@OrderedWith, @OrderedWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c812d5.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c812d5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.692f01.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.692f01.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.33bdbc.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.33bdbc.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.0057df.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.deb(%T.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.0057df.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.975: type = facet_type <@OrderedWith, @OrderedWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.d8f: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.cdc, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.c8dbf2.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.844627.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.c8dbf2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.128dfd.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ba53b2.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.128dfd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.88b872.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.b57b33.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.88b872.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.78fc74.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.4798a8.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.78fc74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.c8dbf2.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.844627.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.c8dbf2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.128dfd.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ba53b2.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.128dfd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.88b872.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.b57b33.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.88b872.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.78fc74.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.4798a8.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.78fc74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.519: %OrderedWith.type.af1 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.d8f) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.f1a: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.519) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.e05: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.519) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.005: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.519) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.86e: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%i64, %OrderedWith.facet.519) [concrete]
// CHECK:STDOUT:   %.105: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.005, %OrderedWith.facet.519 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.625b09.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.b57b33.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.625b09.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.b57b33.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %.054: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.f1a, %OrderedWith.facet.519 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.53698d.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.844627.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.53698d.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.844627.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %.bb8: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.86e, %OrderedWith.facet.519 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1115b0.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.4798a8.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1115b0.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.4798a8.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %.6e5: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.e05, %OrderedWith.facet.519 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3cc3f1.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ba53b2.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3cc3f1.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ba53b2.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.fa6: <witness> = impl_witness imports.%EqWith.impl_witness_table.efb, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.520dd9.1: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.2, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.f6cc6a.1: %Cpp.long_long.as.EqWith.impl.Equal.type.520dd9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.dc208e.1: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.2, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.06262e.1: %Cpp.long_long.as.EqWith.impl.NotEqual.type.dc208e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.type.520dd9.2: type = fn_type @Cpp.long_long.as.EqWith.impl.Equal.1, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.f6cc6a.2: %Cpp.long_long.as.EqWith.impl.Equal.type.520dd9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.type.dc208e.2: type = fn_type @Cpp.long_long.as.EqWith.impl.NotEqual.1, @Cpp.long_long.as.EqWith.impl.600(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.06262e.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.dc208e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.0eb: %EqWith.type.aee = facet_value %Cpp.long_long, (%EqWith.impl_witness.fa6) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.97a: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.0eb) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.f3e: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(Core.IntLiteral, %EqWith.facet.0eb) [concrete]
// CHECK:STDOUT:   %.590: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.97a, %EqWith.facet.0eb [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.3ab10a.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.f6cc6a.2, @Cpp.long_long.as.EqWith.impl.Equal.1(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.3ab10a.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.Equal.f6cc6a.1, @Cpp.long_long.as.EqWith.impl.Equal.2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %.97f: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.f3e, %EqWith.facet.0eb [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.8d8af1.1: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.06262e.2, @Cpp.long_long.as.EqWith.impl.NotEqual.1(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.8d8af1.2: <specific function> = specific_function %Cpp.long_long.as.EqWith.impl.NotEqual.06262e.1, @Cpp.long_long.as.EqWith.impl.NotEqual.2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.784: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.cdc, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.29a8c5.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.7bb755.1: %Cpp.long_long.as.OrderedWith.impl.Less.type.29a8c5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.c460d8.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0b5eb3.1: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.c460d8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.f8fc10.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.a43ce7.1: %Cpp.long_long.as.OrderedWith.impl.Greater.type.f8fc10.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.6f3433.1: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2b5b9d.1: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.6f3433.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.type.29a8c5.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Less.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.7bb755.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.29a8c5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.c460d8.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0b5eb3.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.c460d8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.type.f8fc10.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.Greater.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.a43ce7.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.f8fc10.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.6f3433.2: type = fn_type @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1, @Cpp.long_long.as.OrderedWith.impl.deb(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2b5b9d.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.6f3433.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.84c: %OrderedWith.type.975 = facet_value %Cpp.long_long, (%OrderedWith.impl_witness.784) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.05e: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.84c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.437: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.84c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.ce0: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.84c) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.b92: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(Core.IntLiteral, %OrderedWith.facet.84c) [concrete]
// CHECK:STDOUT:   %.671: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.ce0, %OrderedWith.facet.84c [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.50a98d.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.a43ce7.2, @Cpp.long_long.as.OrderedWith.impl.Greater.1(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.50a98d.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Greater.a43ce7.1, @Cpp.long_long.as.OrderedWith.impl.Greater.2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %.168: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.05e, %OrderedWith.facet.84c [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.ac6947.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.7bb755.2, @Cpp.long_long.as.OrderedWith.impl.Less.1(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.ac6947.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.Less.7bb755.1, @Cpp.long_long.as.OrderedWith.impl.Less.2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %.db2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.b92, %OrderedWith.facet.84c [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.32ac6e.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2b5b9d.2, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.32ac6e.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2b5b9d.1, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %.ada: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.437, %OrderedWith.facet.84c [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.640035.1: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0b5eb3.2, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.640035.2: <specific function> = specific_function %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0b5eb3.1, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d6d = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7e1: @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.Equal.type.2 (%Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.Equal.2 (constants.%Cpp.long_long.as.EqWith.impl.Equal.ec6d64.1)]
// CHECK:STDOUT:   %Core.import_ref.ba2: @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.NotEqual.type.2 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.NotEqual.2 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.4b5f60.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.efb = impl_witness_table (%Core.import_ref.7e1, %Core.import_ref.ba2), @Cpp.long_long.as.EqWith.impl.600 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.422: @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.NotEqual.type.1 (%Cpp.long_long.as.EqWith.impl.NotEqual.type.b639b1.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.NotEqual.1 (constants.%Cpp.long_long.as.EqWith.impl.NotEqual.4b5f60.2)]
// CHECK:STDOUT:   %Core.Equal.cce: @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.Equal.type.1 (%Cpp.long_long.as.EqWith.impl.Equal.type.f2c091.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @Cpp.long_long.as.EqWith.impl.600.%Cpp.long_long.as.EqWith.impl.Equal.1 (constants.%Cpp.long_long.as.EqWith.impl.Equal.ec6d64.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a87 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.020: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Less.type.2 (%Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Less.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.0057df.1)]
// CHECK:STDOUT:   %Core.import_ref.152: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.33bdbc.1)]
// CHECK:STDOUT:   %Core.import_ref.ba3: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Greater.type.2 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Greater.2 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.692f01.1)]
// CHECK:STDOUT:   %Core.import_ref.cea: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c812d5.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.cdc = impl_witness_table (%Core.import_ref.020, %Core.import_ref.152, %Core.import_ref.ba3, %Core.import_ref.cea), @Cpp.long_long.as.OrderedWith.impl.deb [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.7ba: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.362d6d.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.c812d5.2)]
// CHECK:STDOUT:   %Core.Greater.91c: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Greater.type.1 (%Cpp.long_long.as.OrderedWith.impl.Greater.type.9c33c6.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Greater.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Greater.692f01.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.211: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.1 (%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.7ff5fb.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.33bdbc.2)]
// CHECK:STDOUT:   %Core.Less.868: @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Less.type.1 (%Cpp.long_long.as.OrderedWith.impl.Less.type.a2518e.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @Cpp.long_long.as.OrderedWith.impl.deb.%Cpp.long_long.as.OrderedWith.impl.Less.1 (constants.%Cpp.long_long.as.OrderedWith.impl.Less.0057df.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongLeftSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc9_26.2
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.5d2 = impl_witness_access constants.%EqWith.impl_witness.621, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.01ccf4.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.2dd31b.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.EqWith.impl.Equal.type.73aeb2.1 = specific_constant imports.%Core.Equal.cce, @Cpp.long_long.as.EqWith.impl.600(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.01ccf4.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %Cpp.long_long.as.EqWith.impl.Equal.type.73aeb2.1 = name_ref Equal, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.01ccf4.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %a.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.2dd31b.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc11: %.c98 = impl_witness_access constants.%EqWith.impl_witness.621, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.ac575d.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.38af67.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.7f7ea7.1 = specific_constant imports.%Core.NotEqual.422, @Cpp.long_long.as.EqWith.impl.600(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.ac575d.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %Cpp.long_long.as.EqWith.impl.NotEqual.type.7f7ea7.1 = name_ref NotEqual, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.ac575d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %a.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.38af67.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem2.loc12: %.105 = impl_witness_access constants.%OrderedWith.impl_witness.d8f, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.b57b33.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.625b09.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.88b872.1 = specific_constant imports.%Core.Greater.91c, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.b57b33.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %Cpp.long_long.as.OrderedWith.impl.Greater.type.88b872.1 = name_ref Greater, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.b57b33.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %a.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.625b09.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_7: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_7(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_7.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_7.2)
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.054 = impl_witness_access constants.%OrderedWith.impl_witness.d8f, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.844627.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.53698d.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.c8dbf2.1 = specific_constant imports.%Core.Less.868, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.844627.1]
// CHECK:STDOUT:   %Less.ref.loc13: %Cpp.long_long.as.OrderedWith.impl.Less.type.c8dbf2.1 = name_ref Less, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.844627.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %a.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.53698d.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_7: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_7: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_7
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_7(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_7.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_7.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_7.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_7.2)
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem3.loc14: %.bb8 = impl_witness_access constants.%OrderedWith.impl_witness.d8f, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.4798a8.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1115b0.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.78fc74.1 = specific_constant imports.%Core.GreaterOrEquivalent.7ba, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.4798a8.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.78fc74.1 = name_ref GreaterOrEquivalent, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.4798a8.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %a.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.1115b0.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1.loc15: %.6e5 = impl_witness_access constants.%OrderedWith.impl_witness.d8f, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ba53b2.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3cc3f1.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.128dfd.1 = specific_constant imports.%Core.LessOrEquivalent.211, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ba53b2.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.128dfd.1 = name_ref LessOrEquivalent, %.loc15_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.ba53b2.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %a.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.3cc3f1.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_8: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long_long = call %bound_method.loc15_8(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_8.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8.2)
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.590 = impl_witness_access constants.%EqWith.impl_witness.fa6, element0 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.f6cc6a.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.660) [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Cpp.long_long.as.EqWith.impl.Equal.1(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.3ab10a.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.EqWith.impl.Equal.type.520dd9.1 = specific_constant imports.%Core.Equal.cce, @Cpp.long_long.as.EqWith.impl.600(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.f6cc6a.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %Cpp.long_long.as.EqWith.impl.Equal.type.520dd9.1 = name_ref Equal, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.f6cc6a.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %a.ref.loc17, %Equal.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @Cpp.long_long.as.EqWith.impl.Equal.2(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.EqWith.impl.Equal.specific_fn.3ab10a.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.EqWith.impl.Equal.specific_fn.loc17
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_8(%int_1.loc17) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc18: %.97f = impl_witness_access constants.%EqWith.impl_witness.fa6, element1 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.06262e.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem1.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.660) [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.1(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.8d8af1.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.EqWith.impl.NotEqual.type.dc208e.1 = specific_constant imports.%Core.NotEqual.422, @Cpp.long_long.as.EqWith.impl.600(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.06262e.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %Cpp.long_long.as.EqWith.impl.NotEqual.type.dc208e.1 = name_ref NotEqual, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.06262e.1]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %a.ref.loc18, %NotEqual.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @Cpp.long_long.as.EqWith.impl.NotEqual.2(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.8d8af1.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.EqWith.impl.NotEqual.specific_fn.loc18
// CHECK:STDOUT:   %impl.elem0.loc18: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_8(%int_1.loc18) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem2.loc19: %.671 = impl_witness_access constants.%OrderedWith.impl_witness.784, element2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.a43ce7.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem2.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.660) [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.1(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.50a98d.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.OrderedWith.impl.Greater.type.f8fc10.1 = specific_constant imports.%Core.Greater.91c, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.a43ce7.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %Cpp.long_long.as.OrderedWith.impl.Greater.type.f8fc10.1 = name_ref Greater, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.a43ce7.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %a.ref.loc19, %Greater.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @Cpp.long_long.as.OrderedWith.impl.Greater.2(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.50a98d.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.OrderedWith.impl.Greater.specific_fn.loc19
// CHECK:STDOUT:   %impl.elem0.loc19: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_7: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_7(%int_1.loc19) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_7.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_7.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_7.2)
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.168 = impl_witness_access constants.%OrderedWith.impl_witness.784, element0 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.7bb755.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.660) [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Cpp.long_long.as.OrderedWith.impl.Less.1(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.ac6947.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.OrderedWith.impl.Less.type.29a8c5.1 = specific_constant imports.%Core.Less.868, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.7bb755.1]
// CHECK:STDOUT:   %Less.ref.loc20: %Cpp.long_long.as.OrderedWith.impl.Less.type.29a8c5.1 = name_ref Less, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.7bb755.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %a.ref.loc20, %Less.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @Cpp.long_long.as.OrderedWith.impl.Less.2(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.ac6947.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.OrderedWith.impl.Less.specific_fn.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_7: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_7: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_7(%int_1.loc20) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_7.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_7.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_7.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_7.2)
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem3.loc21: %.db2 = impl_witness_access constants.%OrderedWith.impl_witness.784, element3 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2b5b9d.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem3.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.660) [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.1(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.32ac6e.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.6f3433.1 = specific_constant imports.%Core.GreaterOrEquivalent.7ba, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2b5b9d.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.type.6f3433.1 = name_ref GreaterOrEquivalent, %.loc21_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2b5b9d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %a.ref.loc21, %GreaterOrEquivalent.ref.loc21
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.2(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.32ac6e.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21
// CHECK:STDOUT:   %impl.elem0.loc21: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_8: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long_long = call %bound_method.loc21_8(%int_1.loc21) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_8.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_8.2)
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem1.loc22: %.ada = impl_witness_access constants.%OrderedWith.impl_witness.784, element1 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0b5eb3.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %a.ref.loc22, %impl.elem1.loc22
// CHECK:STDOUT:   %ImplicitAs.facet.loc22: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.660) [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %.loc22_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet.loc22 [concrete = constants.%ImplicitAs.facet.2ed]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.1(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.640035.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %a.ref.loc22, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_5.2: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.c460d8.1 = specific_constant imports.%Core.LessOrEquivalent.211, @Cpp.long_long.as.OrderedWith.impl.deb(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0b5eb3.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.type.c460d8.1 = name_ref LessOrEquivalent, %.loc22_5.2 [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.0b5eb3.1]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %a.ref.loc22, %LessOrEquivalent.ref.loc22
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.2(constants.%ImplicitAs.facet.2ed) [concrete = constants.%Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.640035.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %a.ref.loc22, %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22
// CHECK:STDOUT:   %impl.elem0.loc22: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_8: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long_long = call %bound_method.loc22_8(%int_1.loc22) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc22_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc22_8.2: %Cpp.long_long = converted %int_1.loc22, %.loc22_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.OrderedWith.impl.LessOrEquivalent.call.loc22: init bool = call %bound_method.loc22_5.3(%a.ref.loc22, %.loc22_8.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- comparisons_heterogeneous_long_long_right_side.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.660: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2ed: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.660) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.062: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %.ca2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.062, %ImplicitAs.facet.2ed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %EqWith.type.9fc: type = facet_type <@EqWith, @EqWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.bf2: %ImplicitAs.type.0ec = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.925fdb.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.6d3(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.b16a99.1: %T.as_type.as.EqWith.impl.NotEqual.type.925fdb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.925fdb.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.6d3(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.b16a99.2: %T.as_type.as.EqWith.impl.NotEqual.type.925fdb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.3cb0e1.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.6d3(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.a1333e.1: %T.as_type.as.EqWith.impl.Equal.type.3cb0e1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.3cb0e1.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.6d3(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.a1333e.2: %T.as_type.as.EqWith.impl.Equal.type.3cb0e1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.305: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a87 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7a4: %ImplicitAs.type.0ec = facet_value %i64, (%ImplicitAs.impl_witness.305) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.084: <witness> = impl_witness imports.%EqWith.impl_witness_table.db1, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.6d33d6.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.817082.1: %T.as_type.as.EqWith.impl.Equal.type.6d33d6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.2c42f8.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.de7c60.1: %T.as_type.as.EqWith.impl.NotEqual.type.2c42f8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.6d33d6.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.817082.2: %T.as_type.as.EqWith.impl.Equal.type.6d33d6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.2c42f8.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.de7c60.2: %T.as_type.as.EqWith.impl.NotEqual.type.2c42f8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.306: %EqWith.type.9fc = facet_value %i64, (%EqWith.impl_witness.084) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.5b2: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.306) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.86d: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.306) [concrete]
// CHECK:STDOUT:   %.183: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.5b2, %EqWith.facet.306 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.044d32.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.817082.2, @T.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.044d32.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.817082.1, @T.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3ae: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %.25e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3ae, %ImplicitAs.facet.7a4 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %.eca: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.86d, %EqWith.facet.306 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.b44c79.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.de7c60.2, @T.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.b44c79.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.de7c60.1, @T.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %OrderedWith.type.456: type = facet_type <@OrderedWith, @OrderedWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.df5290.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.df5290.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.878ae3.1: %T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.878ae3.2: %T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.2cb445.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.2cb445.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.2daa9a.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.c37ef5.1: %T.as_type.as.OrderedWith.impl.Less.type.2daa9a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.2daa9a.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.df2(%T.bf2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.c37ef5.2: %T.as_type.as.OrderedWith.impl.Less.type.2daa9a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.7e3: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.8b4, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.7b405f.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.c5f236.1: %T.as_type.as.OrderedWith.impl.Less.type.7b405f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.577251.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e5cfba.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.577251.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.085dd9.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.749015.1: %T.as_type.as.OrderedWith.impl.Greater.type.085dd9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.0f0b65.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f86b3e.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.0f0b65.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.7b405f.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.c5f236.2: %T.as_type.as.OrderedWith.impl.Less.type.7b405f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.577251.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e5cfba.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.577251.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.085dd9.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.749015.2: %T.as_type.as.OrderedWith.impl.Greater.type.085dd9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.0f0b65.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f86b3e.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.0f0b65.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.72a: %OrderedWith.type.456 = facet_value %i64, (%OrderedWith.impl_witness.7e3) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.606: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.72a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.2a1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.72a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.f32: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.72a) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.15d: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.72a) [concrete]
// CHECK:STDOUT:   %.af0: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.f32, %OrderedWith.facet.72a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.cbc6c2.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.749015.2, @T.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.cbc6c2.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.749015.1, @T.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %.901: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.606, %OrderedWith.facet.72a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.a8a2ad.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.c5f236.2, @T.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.a8a2ad.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.c5f236.1, @T.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %.96b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.15d, %OrderedWith.facet.72a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.8a93e9.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f86b3e.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.8a93e9.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f86b3e.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %.829: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.2a1, %OrderedWith.facet.72a [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ab9ff7.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e5cfba.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ab9ff7.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.e5cfba.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness.cb6: <witness> = impl_witness imports.%EqWith.impl_witness_table.db1, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.8e6244.1: type = fn_type @T.as_type.as.EqWith.impl.Equal.4, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.66756b.1: %T.as_type.as.EqWith.impl.Equal.type.8e6244.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.fb7a86.1: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.4, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.672038.1: %T.as_type.as.EqWith.impl.NotEqual.type.fb7a86.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.type.8e6244.2: type = fn_type @T.as_type.as.EqWith.impl.Equal.3, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.66756b.2: %T.as_type.as.EqWith.impl.Equal.type.8e6244.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.type.fb7a86.2: type = fn_type @T.as_type.as.EqWith.impl.NotEqual.3, @T.as_type.as.EqWith.impl.6d3(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.672038.2: %T.as_type.as.EqWith.impl.NotEqual.type.fb7a86.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.facet.83c: %EqWith.type.9fc = facet_value Core.IntLiteral, (%EqWith.impl_witness.cb6) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.5b1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.83c) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.a9f: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Cpp.long_long, %EqWith.facet.83c) [concrete]
// CHECK:STDOUT:   %.658: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.5b1, %EqWith.facet.83c [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.2fcdd5.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.66756b.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.7834c5.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.66756b.2, @T.as_type.as.EqWith.impl.Equal.3(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.7c9d36.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.7834c5.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.2fcdd5.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.66756b.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.7834c5.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.Equal.66756b.1, @T.as_type.as.EqWith.impl.Equal.4(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.7c9d36.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.Equal.specific_fn.7834c5.2 [concrete]
// CHECK:STDOUT:   %.1f9: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.a9f, %EqWith.facet.83c [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.a9911b.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.672038.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.598adf.1: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.672038.2, @T.as_type.as.EqWith.impl.NotEqual.3(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.1f073e.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.598adf.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.a9911b.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.672038.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.598adf.2: <specific function> = specific_function %T.as_type.as.EqWith.impl.NotEqual.672038.1, @T.as_type.as.EqWith.impl.NotEqual.4(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.1f073e.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.598adf.2 [concrete]
// CHECK:STDOUT:   %OrderedWith.impl_witness.ccd: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.8b4, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.28a50f.1: type = fn_type @T.as_type.as.OrderedWith.impl.Less.4, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.09733d.1: %T.as_type.as.OrderedWith.impl.Less.type.28a50f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1522a4.1: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.b7cd44.1: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1522a4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.e2356e.1: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.4, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.a80774.1: %T.as_type.as.OrderedWith.impl.Greater.type.e2356e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.09194f.1: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.ae8de7.1: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.09194f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.type.28a50f.2: type = fn_type @T.as_type.as.OrderedWith.impl.Less.3, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.09733d.2: %T.as_type.as.OrderedWith.impl.Less.type.28a50f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1522a4.2: type = fn_type @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.b7cd44.2: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1522a4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.type.e2356e.2: type = fn_type @T.as_type.as.OrderedWith.impl.Greater.3, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.a80774.2: %T.as_type.as.OrderedWith.impl.Greater.type.e2356e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.09194f.2: type = fn_type @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3, @T.as_type.as.OrderedWith.impl.df2(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.ae8de7.2: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.09194f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.54d: %OrderedWith.type.456 = facet_value Core.IntLiteral, (%OrderedWith.impl_witness.ccd) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.a56: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.54d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.21f: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.54d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.22c: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.54d) [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.6ad: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Cpp.long_long, %OrderedWith.facet.54d) [concrete]
// CHECK:STDOUT:   %.1e0: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.22c, %OrderedWith.facet.54d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.ba4552.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.a80774.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.3eccd1.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.a80774.2, @T.as_type.as.OrderedWith.impl.Greater.3(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.560bca.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.3eccd1.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.ba4552.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.a80774.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.3eccd1.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Greater.a80774.1, @T.as_type.as.OrderedWith.impl.Greater.4(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.560bca.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.3eccd1.2 [concrete]
// CHECK:STDOUT:   %.4c9: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.a56, %OrderedWith.facet.54d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.069497.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.09733d.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.0beaf8.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.09733d.2, @T.as_type.as.OrderedWith.impl.Less.3(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.103f49.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.0beaf8.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.069497.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.09733d.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.0beaf8.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.Less.09733d.1, @T.as_type.as.OrderedWith.impl.Less.4(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.103f49.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.Less.specific_fn.0beaf8.2 [concrete]
// CHECK:STDOUT:   %.b09: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.6ad, %OrderedWith.facet.54d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.22317d.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.ae8de7.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.371f57.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.ae8de7.2, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.ffa540.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.371f57.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.22317d.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.ae8de7.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.371f57.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.ae8de7.1, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.ffa540.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.371f57.2 [concrete]
// CHECK:STDOUT:   %.cab: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.21f, %OrderedWith.facet.54d [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.2eaa72.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.b7cd44.2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.05fce2.1: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.b7cd44.2, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.83e687.1: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.05fce2.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.2eaa72.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.b7cd44.1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.05fce2.2: <specific function> = specific_function %T.as_type.as.OrderedWith.impl.LessOrEquivalent.b7cd44.1, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(%ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %bound_method.83e687.2: <bound method> = bound_method %int_1.5b8, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.05fce2.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d6d = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c48: @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.Equal.type.2 (%T.as_type.as.EqWith.impl.Equal.type.3cb0e1.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.Equal.2 (constants.%T.as_type.as.EqWith.impl.Equal.a1333e.1)]
// CHECK:STDOUT:   %Core.import_ref.98d: @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.NotEqual.type.2 (%T.as_type.as.EqWith.impl.NotEqual.type.925fdb.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.NotEqual.2 (constants.%T.as_type.as.EqWith.impl.NotEqual.b16a99.1)]
// CHECK:STDOUT:   %EqWith.impl_witness_table.db1 = impl_witness_table (%Core.import_ref.c48, %Core.import_ref.98d), @T.as_type.as.EqWith.impl.6d3 [concrete]
// CHECK:STDOUT:   %Core.NotEqual.0b2: @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.NotEqual.type.1 (%T.as_type.as.EqWith.impl.NotEqual.type.925fdb.2) = import_ref Core//prelude/types/cpp/int, NotEqual, loaded [symbolic = @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.NotEqual.1 (constants.%T.as_type.as.EqWith.impl.NotEqual.b16a99.2)]
// CHECK:STDOUT:   %Core.Equal.4f5: @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.Equal.type.1 (%T.as_type.as.EqWith.impl.Equal.type.3cb0e1.2) = import_ref Core//prelude/types/cpp/int, Equal, loaded [symbolic = @T.as_type.as.EqWith.impl.6d3.%T.as_type.as.EqWith.impl.Equal.1 (constants.%T.as_type.as.EqWith.impl.Equal.a1333e.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a87 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.97f: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Less.type.2 (%T.as_type.as.OrderedWith.impl.Less.type.2daa9a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Less.2 (constants.%T.as_type.as.OrderedWith.impl.Less.c37ef5.1)]
// CHECK:STDOUT:   %Core.import_ref.409: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2cb445.1)]
// CHECK:STDOUT:   %Core.import_ref.56a: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Greater.type.2 (%T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Greater.2 (constants.%T.as_type.as.OrderedWith.impl.Greater.878ae3.1)]
// CHECK:STDOUT:   %Core.import_ref.b00: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.2 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.2 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.df5290.1)]
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.8b4 = impl_witness_table (%Core.import_ref.97f, %Core.import_ref.409, %Core.import_ref.56a, %Core.import_ref.b00), @T.as_type.as.OrderedWith.impl.df2 [concrete]
// CHECK:STDOUT:   %Core.GreaterOrEquivalent.8e2: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.d6724c.2) = import_ref Core//prelude/types/cpp/int, GreaterOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.df5290.2)]
// CHECK:STDOUT:   %Core.Greater.c24: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Greater.type.1 (%T.as_type.as.OrderedWith.impl.Greater.type.4fb98a.2) = import_ref Core//prelude/types/cpp/int, Greater, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Greater.1 (constants.%T.as_type.as.OrderedWith.impl.Greater.878ae3.2)]
// CHECK:STDOUT:   %Core.LessOrEquivalent.d90: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1 (%T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.3d985f.2) = import_ref Core//prelude/types/cpp/int, LessOrEquivalent, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.1 (constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.2cb445.2)]
// CHECK:STDOUT:   %Core.Less.2d9: @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Less.type.1 (%T.as_type.as.OrderedWith.impl.Less.type.2daa9a.2) = import_ref Core//prelude/types/cpp/int, Less, loaded [symbolic = @T.as_type.as.OrderedWith.impl.df2.%T.as_type.as.OrderedWith.impl.Less.1 (constants.%T.as_type.as.OrderedWith.impl.Less.c37ef5.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ComparisonsHeterogeneousLongLongRightSide() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc9: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_26.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_26.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %a: %Cpp.long_long = value_binding a, %.loc9_26.2
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc10: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.183 = impl_witness_access constants.%EqWith.impl_witness.084, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.817082.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @T.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.044d32.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %b.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5: %T.as_type.as.EqWith.impl.Equal.type.6d33d6.1 = specific_constant imports.%Core.Equal.4f5, @T.as_type.as.EqWith.impl.6d3(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.817082.1]
// CHECK:STDOUT:   %Equal.ref.loc10: %T.as_type.as.EqWith.impl.Equal.type.6d33d6.1 = name_ref Equal, %.loc10_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.817082.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc10: <bound method> = bound_method %b.ref.loc10, %Equal.ref.loc10
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10: <specific function> = specific_function %Equal.ref.loc10, @T.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.044d32.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %b.ref.loc10, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc10
// CHECK:STDOUT:   %impl.elem0.loc10_3: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_3(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_3.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc10: init bool = call %bound_method.loc10_5.3(%.loc10_3.2, %a.ref.loc10)
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc11: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc11: %.eca = impl_witness_access constants.%EqWith.impl_witness.084, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.de7c60.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.ref.loc11, %impl.elem1.loc11
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem1.loc11, @T.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.b44c79.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5: %T.as_type.as.EqWith.impl.NotEqual.type.2c42f8.1 = specific_constant imports.%Core.NotEqual.0b2, @T.as_type.as.EqWith.impl.6d3(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.de7c60.1]
// CHECK:STDOUT:   %NotEqual.ref.loc11: %T.as_type.as.EqWith.impl.NotEqual.type.2c42f8.1 = name_ref NotEqual, %.loc11_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.de7c60.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc11: <bound method> = bound_method %b.ref.loc11, %NotEqual.ref.loc11
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11: <specific function> = specific_function %NotEqual.ref.loc11, @T.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.b44c79.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %b.ref.loc11, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc11
// CHECK:STDOUT:   %impl.elem0.loc11: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_3: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_3(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_3.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_3.1
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc11: init bool = call %bound_method.loc11_5.3(%.loc11_3.2, %a.ref.loc11)
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc12: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc12: %.af0 = impl_witness_access constants.%OrderedWith.impl_witness.7e3, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.749015.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %b.ref.loc12, %impl.elem2.loc12
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem2.loc12, @T.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.cbc6c2.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %b.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5: %T.as_type.as.OrderedWith.impl.Greater.type.085dd9.1 = specific_constant imports.%Core.Greater.c24, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.749015.1]
// CHECK:STDOUT:   %Greater.ref.loc12: %T.as_type.as.OrderedWith.impl.Greater.type.085dd9.1 = name_ref Greater, %.loc12_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.749015.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc12: <bound method> = bound_method %b.ref.loc12, %Greater.ref.loc12
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12: <specific function> = specific_function %Greater.ref.loc12, @T.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.cbc6c2.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %b.ref.loc12, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc12
// CHECK:STDOUT:   %impl.elem0.loc12: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_3: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_3(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_3.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc12: init bool = call %bound_method.loc12_5.3(%.loc12_3.2, %a.ref.loc12)
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc13: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.901 = impl_witness_access constants.%OrderedWith.impl_witness.7e3, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.c5f236.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @T.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.a8a2ad.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %b.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5: %T.as_type.as.OrderedWith.impl.Less.type.7b405f.1 = specific_constant imports.%Core.Less.2d9, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.c5f236.1]
// CHECK:STDOUT:   %Less.ref.loc13: %T.as_type.as.OrderedWith.impl.Less.type.7b405f.1 = name_ref Less, %.loc13_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.c5f236.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc13: <bound method> = bound_method %b.ref.loc13, %Less.ref.loc13
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13: <specific function> = specific_function %Less.ref.loc13, @T.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.a8a2ad.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %b.ref.loc13, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_3
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_3(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_3.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc13: init bool = call %bound_method.loc13_5.3(%.loc13_3.2, %a.ref.loc13)
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc14: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc14: %.96b = impl_witness_access constants.%OrderedWith.impl_witness.7e3, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f86b3e.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %b.ref.loc14, %impl.elem3.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem3.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.8a93e9.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %b.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.0f0b65.1 = specific_constant imports.%Core.GreaterOrEquivalent.8e2, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f86b3e.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc14: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.0f0b65.1 = name_ref GreaterOrEquivalent, %.loc14_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.f86b3e.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc14: <bound method> = bound_method %b.ref.loc14, %GreaterOrEquivalent.ref.loc14
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14: <specific function> = specific_function %GreaterOrEquivalent.ref.loc14, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.8a93e9.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %b.ref.loc14, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc14
// CHECK:STDOUT:   %impl.elem0.loc14: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_3(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_3.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc14: init bool = call %bound_method.loc14_5.3(%.loc14_3.2, %a.ref.loc14)
// CHECK:STDOUT:   %b.ref.loc15: %i64 = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc15: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc15: %.829 = impl_witness_access constants.%OrderedWith.impl_witness.7e3, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.e5cfba.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %b.ref.loc15, %impl.elem1.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem1.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ab9ff7.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %b.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.577251.1 = specific_constant imports.%Core.LessOrEquivalent.d90, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.e5cfba.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc15: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.577251.1 = name_ref LessOrEquivalent, %.loc15_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.e5cfba.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc15: <bound method> = bound_method %b.ref.loc15, %LessOrEquivalent.ref.loc15
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15: <specific function> = specific_function %LessOrEquivalent.ref.loc15, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.7a4) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.ab9ff7.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %b.ref.loc15, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.long_long = call %bound_method.loc15_3(%b.ref.loc15)
// CHECK:STDOUT:   %.loc15_3.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc15
// CHECK:STDOUT:   %.loc15_3.2: %Cpp.long_long = converted %b.ref.loc15, %.loc15_3.1
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc15: init bool = call %bound_method.loc15_5.3(%.loc15_3.2, %a.ref.loc15)
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc17: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.658 = impl_witness_access constants.%EqWith.impl_witness.cb6, element0 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.66756b.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.2fcdd5.1]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @T.as_type.as.EqWith.impl.Equal.3(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.7834c5.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_1.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.7c9d36.1]
// CHECK:STDOUT:   %.loc17_5: %T.as_type.as.EqWith.impl.Equal.type.8e6244.1 = specific_constant imports.%Core.Equal.4f5, @T.as_type.as.EqWith.impl.6d3(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.66756b.1]
// CHECK:STDOUT:   %Equal.ref.loc17: %T.as_type.as.EqWith.impl.Equal.type.8e6244.1 = name_ref Equal, %.loc17_5 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.66756b.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.bound.loc17: <bound method> = bound_method %int_1.loc17, %Equal.ref.loc17 [concrete = constants.%T.as_type.as.EqWith.impl.Equal.bound.2fcdd5.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17: <specific function> = specific_function %Equal.ref.loc17, @T.as_type.as.EqWith.impl.Equal.4(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.EqWith.impl.Equal.specific_fn.7834c5.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %int_1.loc17, %T.as_type.as.EqWith.impl.Equal.specific_fn.loc17 [concrete = constants.%bound_method.7c9d36.2]
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_3(%int_1.loc17) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc17_3.2: %Cpp.long_long = converted %int_1.loc17, %.loc17_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call.loc17: init bool = call %bound_method.loc17_5.3(%.loc17_3.2, %a.ref.loc17)
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc18: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc18: %.1f9 = impl_witness_access constants.%EqWith.impl_witness.cb6, element1 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.672038.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_1.loc18, %impl.elem1.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.a9911b.1]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem1.loc18, @T.as_type.as.EqWith.impl.NotEqual.3(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.598adf.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.1f073e.1]
// CHECK:STDOUT:   %.loc18_5: %T.as_type.as.EqWith.impl.NotEqual.type.fb7a86.1 = specific_constant imports.%Core.NotEqual.0b2, @T.as_type.as.EqWith.impl.6d3(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.672038.1]
// CHECK:STDOUT:   %NotEqual.ref.loc18: %T.as_type.as.EqWith.impl.NotEqual.type.fb7a86.1 = name_ref NotEqual, %.loc18_5 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.672038.1]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.bound.loc18: <bound method> = bound_method %int_1.loc18, %NotEqual.ref.loc18 [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.bound.a9911b.2]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18: <specific function> = specific_function %NotEqual.ref.loc18, @T.as_type.as.EqWith.impl.NotEqual.4(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.EqWith.impl.NotEqual.specific_fn.598adf.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %int_1.loc18, %T.as_type.as.EqWith.impl.NotEqual.specific_fn.loc18 [concrete = constants.%bound_method.1f073e.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_3(%int_1.loc18) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc18_3.2: %Cpp.long_long = converted %int_1.loc18, %.loc18_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call.loc18: init bool = call %bound_method.loc18_5.3(%.loc18_3.2, %a.ref.loc18)
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc19: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem2.loc19: %.1e0 = impl_witness_access constants.%OrderedWith.impl_witness.ccd, element2 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.a80774.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %int_1.loc19, %impl.elem2.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.ba4552.1]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem2.loc19, @T.as_type.as.OrderedWith.impl.Greater.3(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.3eccd1.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.560bca.1]
// CHECK:STDOUT:   %.loc19_5: %T.as_type.as.OrderedWith.impl.Greater.type.e2356e.1 = specific_constant imports.%Core.Greater.c24, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.a80774.1]
// CHECK:STDOUT:   %Greater.ref.loc19: %T.as_type.as.OrderedWith.impl.Greater.type.e2356e.1 = name_ref Greater, %.loc19_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.a80774.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.bound.loc19: <bound method> = bound_method %int_1.loc19, %Greater.ref.loc19 [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.bound.ba4552.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19: <specific function> = specific_function %Greater.ref.loc19, @T.as_type.as.OrderedWith.impl.Greater.4(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Greater.specific_fn.3eccd1.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %int_1.loc19, %T.as_type.as.OrderedWith.impl.Greater.specific_fn.loc19 [concrete = constants.%bound_method.560bca.2]
// CHECK:STDOUT:   %impl.elem0.loc19: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_3(%int_1.loc19) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc19_3.2: %Cpp.long_long = converted %int_1.loc19, %.loc19_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call.loc19: init bool = call %bound_method.loc19_5.3(%.loc19_3.2, %a.ref.loc19)
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc20: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.4c9 = impl_witness_access constants.%OrderedWith.impl_witness.ccd, element0 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.09733d.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.069497.1]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @T.as_type.as.OrderedWith.impl.Less.3(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.0beaf8.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %int_1.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.103f49.1]
// CHECK:STDOUT:   %.loc20_5: %T.as_type.as.OrderedWith.impl.Less.type.28a50f.1 = specific_constant imports.%Core.Less.2d9, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.09733d.1]
// CHECK:STDOUT:   %Less.ref.loc20: %T.as_type.as.OrderedWith.impl.Less.type.28a50f.1 = name_ref Less, %.loc20_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.09733d.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.bound.loc20: <bound method> = bound_method %int_1.loc20, %Less.ref.loc20 [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.bound.069497.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20: <specific function> = specific_function %Less.ref.loc20, @T.as_type.as.OrderedWith.impl.Less.4(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.Less.specific_fn.0beaf8.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %int_1.loc20, %T.as_type.as.OrderedWith.impl.Less.specific_fn.loc20 [concrete = constants.%bound_method.103f49.2]
// CHECK:STDOUT:   %impl.elem0.loc20_3: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc20_3: <bound method> = bound_method %int_1.loc20, %impl.elem0.loc20_3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_3(%int_1.loc20) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc20_3.2: %Cpp.long_long = converted %int_1.loc20, %.loc20_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call.loc20: init bool = call %bound_method.loc20_5.3(%.loc20_3.2, %a.ref.loc20)
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc21: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem3.loc21: %.b09 = impl_witness_access constants.%OrderedWith.impl_witness.ccd, element3 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.ae8de7.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %int_1.loc21, %impl.elem3.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.22317d.1]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem3.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.3(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.371f57.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.ffa540.1]
// CHECK:STDOUT:   %.loc21_5: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.09194f.1 = specific_constant imports.%Core.GreaterOrEquivalent.8e2, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.ae8de7.1]
// CHECK:STDOUT:   %GreaterOrEquivalent.ref.loc21: %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.type.09194f.1 = name_ref GreaterOrEquivalent, %.loc21_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.ae8de7.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.loc21: <bound method> = bound_method %int_1.loc21, %GreaterOrEquivalent.ref.loc21 [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.bound.22317d.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21: <specific function> = specific_function %GreaterOrEquivalent.ref.loc21, @T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.4(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.371f57.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %int_1.loc21, %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.specific_fn.loc21 [concrete = constants.%bound_method.ffa540.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %Cpp.long_long = call %bound_method.loc21_3(%int_1.loc21) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc21_3.2: %Cpp.long_long = converted %int_1.loc21, %.loc21_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call.loc21: init bool = call %bound_method.loc21_5.3(%.loc21_3.2, %a.ref.loc21)
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %a.ref.loc22: %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1.loc22: %.cab = impl_witness_access constants.%OrderedWith.impl_witness.ccd, element1 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.b7cd44.2]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_1.loc22, %impl.elem1.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.2eaa72.1]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem1.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.3(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.05fce2.1]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.83e687.1]
// CHECK:STDOUT:   %.loc22_5: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1522a4.1 = specific_constant imports.%Core.LessOrEquivalent.d90, @T.as_type.as.OrderedWith.impl.df2(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.b7cd44.1]
// CHECK:STDOUT:   %LessOrEquivalent.ref.loc22: %T.as_type.as.OrderedWith.impl.LessOrEquivalent.type.1522a4.1 = name_ref LessOrEquivalent, %.loc22_5 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.b7cd44.1]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.loc22: <bound method> = bound_method %int_1.loc22, %LessOrEquivalent.ref.loc22 [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.bound.2eaa72.2]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22: <specific function> = specific_function %LessOrEquivalent.ref.loc22, @T.as_type.as.OrderedWith.impl.LessOrEquivalent.4(constants.%ImplicitAs.facet.2ed) [concrete = constants.%T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.05fce2.2]
// CHECK:STDOUT:   %bound_method.loc22_5.3: <bound method> = bound_method %int_1.loc22, %T.as_type.as.OrderedWith.impl.LessOrEquivalent.specific_fn.loc22 [concrete = constants.%bound_method.83e687.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc22_3: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %Cpp.long_long = call %bound_method.loc22_3(%int_1.loc22) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc22_3.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc22_3.2: %Cpp.long_long = converted %int_1.loc22, %.loc22_3.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call.loc22: init bool = call %bound_method.loc22_5.3(%.loc22_3.2, %a.ref.loc22)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_homogeneous_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.type.6c4: type = fn_type @T.as_type.as.ImplicitAs.impl.Convert, @T.as_type.as.ImplicitAs.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.ImplicitAs.impl.Convert.61d: %T.as_type.as.ImplicitAs.impl.Convert.type.6c4 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.660: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2ed: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.660) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.062: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %.ca2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.062, %ImplicitAs.facet.2ed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.166: type = facet_type <@AddAssignWith, @AddAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %U.bf2: %ImplicitAs.type.0ec = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.5c7: <witness> = impl_witness imports.%Copy.impl_witness_table.a14 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.long_long, (%Copy.impl_witness.5c7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3f3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5b9, @T.as_type.as.ImplicitAs.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7bc: %ImplicitAs.type.0ec = facet_value %Cpp.long_long, (%ImplicitAs.impl_witness.3f3) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.82d: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.442, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.7da4ae.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.57b7c2.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.7da4ae.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.7da4ae.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.57b7c2.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.7da4ae.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.af3: %AddAssignWith.type.166 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.82d) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.aa9: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Cpp.long_long, %AddAssignWith.facet.af3) [concrete]
// CHECK:STDOUT:   %.a99: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.aa9, %AddAssignWith.facet.af3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.51a919.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.57b7c2.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.51a919.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.57b7c2.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.81c: type = facet_type <@SubAssignWith, @SubAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.67476b.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.67476b.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.d2c: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.425, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.c6ad2e.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.2d41e3.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.c6ad2e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.c6ad2e.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.2d41e3.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.c6ad2e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.3d7: %SubAssignWith.type.81c = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.d2c) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.5cd: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Cpp.long_long, %SubAssignWith.facet.3d7) [concrete]
// CHECK:STDOUT:   %.17f: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.5cd, %SubAssignWith.facet.3d7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.6e9b6c.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.2d41e3.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.6e9b6c.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.2d41e3.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.65c: type = facet_type <@MulAssignWith, @MulAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.96f: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.769, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.892850.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.46f6d9.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.892850.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.892850.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.46f6d9.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.892850.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.1c1: %MulAssignWith.type.65c = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.96f) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.433: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Cpp.long_long, %MulAssignWith.facet.1c1) [concrete]
// CHECK:STDOUT:   %.50e: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.433, %MulAssignWith.facet.1c1 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.762651.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.46f6d9.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.762651.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.46f6d9.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.d82: type = facet_type <@DivAssignWith, @DivAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.053c65.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.053c65.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.0c4: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.538, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.90284d.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.1dcb1e.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.90284d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.90284d.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.1dcb1e.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.90284d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.653: %DivAssignWith.type.d82 = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.0c4) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.6ce: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Cpp.long_long, %DivAssignWith.facet.653) [concrete]
// CHECK:STDOUT:   %.1d3: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.6ce, %DivAssignWith.facet.653 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.20bed4.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.1dcb1e.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.20bed4.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.1dcb1e.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.3e2: type = facet_type <@ModAssignWith, @ModAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.ed9: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.cdc, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.030e50.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.1c5c20.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.030e50.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.030e50.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.1c5c20.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.030e50.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.761: %ModAssignWith.type.3e2 = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.ed9) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.3f1: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Cpp.long_long, %ModAssignWith.facet.761) [concrete]
// CHECK:STDOUT:   %.e86: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.3f1, %ModAssignWith.facet.761 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.0d98f1.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.1c5c20.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.0d98f1.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.1c5c20.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.2f8: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.8e7: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.d41, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.f75b5c.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.da411d.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.f75b5c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.f75b5c.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.da411d.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.f75b5c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.6e7: %BitAndAssignWith.type.2f8 = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.8e7) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.f7e: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Cpp.long_long, %BitAndAssignWith.facet.6e7) [concrete]
// CHECK:STDOUT:   %.922: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.f7e, %BitAndAssignWith.facet.6e7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.dc94b5.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.da411d.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.dc94b5.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.da411d.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.eac: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.d84: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.446, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.503598.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.1137bd.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.503598.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.503598.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.1137bd.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.503598.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.bf3: %BitOrAssignWith.type.eac = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.d84) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.5b1: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Cpp.long_long, %BitOrAssignWith.facet.bf3) [concrete]
// CHECK:STDOUT:   %.cc3: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.5b1, %BitOrAssignWith.facet.bf3 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.9908b9.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.1137bd.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.9908b9.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.1137bd.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.434: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.462: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.8e4, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.3d8271.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.84559f.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.3d8271.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.3d8271.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.84559f.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.3d8271.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.562: %BitXorAssignWith.type.434 = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.462) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.0f1: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Cpp.long_long, %BitXorAssignWith.facet.562) [concrete]
// CHECK:STDOUT:   %.cae: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.0f1, %BitXorAssignWith.facet.562 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.1bb816.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.84559f.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.1bb816.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.84559f.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.509: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.812: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.dd0, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.25e1dd.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e964c.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.25e1dd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.25e1dd.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e964c.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.25e1dd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.033: %LeftShiftAssignWith.type.509 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.812) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.5dd: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Cpp.long_long, %LeftShiftAssignWith.facet.033) [concrete]
// CHECK:STDOUT:   %.c95: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.5dd, %LeftShiftAssignWith.facet.033 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.ebef73.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e964c.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.ebef73.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e964c.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.a59: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.dab: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.00b, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.5579b8.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.02ad61.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.5579b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.5579b8.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.02ad61.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.5579b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.347: %RightShiftAssignWith.type.a59 = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.dab) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.82e: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Cpp.long_long, %RightShiftAssignWith.facet.347) [concrete]
// CHECK:STDOUT:   %.888: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.82e, %RightShiftAssignWith.facet.347 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.fa67f5.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.02ad61.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.fa67f5.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.02ad61.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.7bc) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Copy.impl.Op: %Cpp.long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.051: @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert.type (%T.as_type.as.ImplicitAs.impl.Convert.type.6c4) = import_ref Core//prelude/types/cpp/int, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.ImplicitAs.impl.%T.as_type.as.ImplicitAs.impl.Convert (constants.%T.as_type.as.ImplicitAs.impl.Convert.61d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5b9 = impl_witness_table (%Core.import_ref.051), @T.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d6d = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.634: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.442 = impl_witness_table (%Core.import_ref.634), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c11: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2)]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.a14 = impl_witness_table (%Core.import_ref.9c3), @Cpp.long_long.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.14b: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.67476b.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.425 = impl_witness_table (%Core.import_ref.14b), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3dd: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.67476b.2)]
// CHECK:STDOUT:   %Core.import_ref.1e7: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.769 = impl_witness_table (%Core.import_ref.1e7), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1a0: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.2)]
// CHECK:STDOUT:   %Core.import_ref.90c: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.053c65.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.538 = impl_witness_table (%Core.import_ref.90c), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.4db: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.053c65.2)]
// CHECK:STDOUT:   %Core.import_ref.620: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.cdc = impl_witness_table (%Core.import_ref.620), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.978: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.2)]
// CHECK:STDOUT:   %Core.import_ref.f6c: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.d41 = impl_witness_table (%Core.import_ref.f6c), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.96f: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.446 = impl_witness_table (%Core.import_ref.4fa), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.152: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.2)]
// CHECK:STDOUT:   %Core.import_ref.0a0: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.8e4 = impl_witness_table (%Core.import_ref.0a0), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.993: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.2)]
// CHECK:STDOUT:   %Core.import_ref.463: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.dd0 = impl_witness_table (%Core.import_ref.463), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3c4: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.2)]
// CHECK:STDOUT:   %Core.import_ref.6d5: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.00b = impl_witness_table (%Core.import_ref.6d5), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.77b: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentHomogeneousLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref.loc8 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc8: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d58 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.d58 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.long_long = var %b.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long_long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %b.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref.loc9 [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref.loc9: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11: %.a99 = impl_witness_access constants.%AddAssignWith.impl_witness.82d, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.57b7c2.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.3f3) [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.51a919.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_8: %Cpp.long_long = acquire_value %b.ref.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.7da4ae.1 = specific_constant imports.%Core.Op.c11, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.57b7c2.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.AddAssignWith.impl.Op.type.7da4ae.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.57b7c2.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.51a919.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12: %.17f = impl_witness_access constants.%SubAssignWith.impl_witness.d2c, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.2d41e3.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.3f3) [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.6e9b6c.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_8: %Cpp.long_long = acquire_value %b.ref.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.c6ad2e.1 = specific_constant imports.%Core.Op.3dd, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.2d41e3.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.SubAssignWith.impl.Op.type.c6ad2e.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.2d41e3.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.6e9b6c.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13: %.50e = impl_witness_access constants.%MulAssignWith.impl_witness.96f, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.46f6d9.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.3f3) [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.762651.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_8: %Cpp.long_long = acquire_value %b.ref.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.892850.1 = specific_constant imports.%Core.Op.1a0, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.46f6d9.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.MulAssignWith.impl.Op.type.892850.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.46f6d9.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.762651.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14: %.1d3 = impl_witness_access constants.%DivAssignWith.impl_witness.0c4, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.1dcb1e.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.3f3) [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.20bed4.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_8: %Cpp.long_long = acquire_value %b.ref.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.90284d.1 = specific_constant imports.%Core.Op.4db, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.1dcb1e.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.DivAssignWith.impl.Op.type.90284d.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.1dcb1e.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.20bed4.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8)
// CHECK:STDOUT:   %a.ref.loc15: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc15: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc15: %.e86 = impl_witness_access constants.%ModAssignWith.impl_witness.ed9, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.1c5c20.2]
// CHECK:STDOUT:   %bound_method.loc15_5.1: <bound method> = bound_method %a.ref.loc15, %impl.elem0.loc15
// CHECK:STDOUT:   %ImplicitAs.facet.loc15: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.3f3) [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %.loc15_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc15 [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.0d98f1.1]
// CHECK:STDOUT:   %bound_method.loc15_5.2: <bound method> = bound_method %a.ref.loc15, %specific_fn.loc15
// CHECK:STDOUT:   %.loc15_8: %Cpp.long_long = acquire_value %b.ref.loc15
// CHECK:STDOUT:   %.loc15_5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.030e50.1 = specific_constant imports.%Core.Op.978, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.1c5c20.1]
// CHECK:STDOUT:   %Op.ref.loc15: %Cpp.long_long.as.ModAssignWith.impl.Op.type.030e50.1 = name_ref Op, %.loc15_5.2 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.1c5c20.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc15, %Op.ref.loc15
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc15, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.0d98f1.2]
// CHECK:STDOUT:   %bound_method.loc15_5.3: <bound method> = bound_method %a.ref.loc15, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_5.3(%a.ref.loc15, %.loc15_8)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17: %.922 = impl_witness_access constants.%BitAndAssignWith.impl_witness.8e7, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.da411d.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.3f3) [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.dc94b5.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_8: %Cpp.long_long = acquire_value %b.ref.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.f75b5c.1 = specific_constant imports.%Core.Op.96f, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.da411d.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.f75b5c.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.da411d.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.dc94b5.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18: %.cc3 = impl_witness_access constants.%BitOrAssignWith.impl_witness.d84, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1137bd.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.3f3) [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.9908b9.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_8: %Cpp.long_long = acquire_value %b.ref.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.503598.1 = specific_constant imports.%Core.Op.152, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1137bd.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.503598.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1137bd.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.9908b9.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19: %.cae = impl_witness_access constants.%BitXorAssignWith.impl_witness.462, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.84559f.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.3f3) [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.1bb816.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_8: %Cpp.long_long = acquire_value %b.ref.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.3d8271.1 = specific_constant imports.%Core.Op.993, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.84559f.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.3d8271.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.84559f.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.1bb816.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_8)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20: %.c95 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.812, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e964c.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.3f3) [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.ebef73.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_9: %Cpp.long_long = acquire_value %b.ref.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.25e1dd.1 = specific_constant imports.%Core.Op.3c4, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e964c.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.25e1dd.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e964c.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.ebef73.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9)
// CHECK:STDOUT:   %a.ref.loc21: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc21: ref %Cpp.long_long = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc21: %.888 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.dab, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.02ad61.2]
// CHECK:STDOUT:   %bound_method.loc21_5.1: <bound method> = bound_method %a.ref.loc21, %impl.elem0.loc21
// CHECK:STDOUT:   %ImplicitAs.facet.loc21: %ImplicitAs.type.0ec = facet_value constants.%Cpp.long_long, (constants.%ImplicitAs.impl_witness.3f3) [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %.loc21_5.1: %ImplicitAs.type.0ec = converted constants.%Cpp.long_long, %ImplicitAs.facet.loc21 [concrete = constants.%ImplicitAs.facet.7bc]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.fa67f5.1]
// CHECK:STDOUT:   %bound_method.loc21_5.2: <bound method> = bound_method %a.ref.loc21, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_9: %Cpp.long_long = acquire_value %b.ref.loc21
// CHECK:STDOUT:   %.loc21_5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.5579b8.1 = specific_constant imports.%Core.Op.77b, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.02ad61.1]
// CHECK:STDOUT:   %Op.ref.loc21: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.5579b8.1 = name_ref Op, %.loc21_5.2 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.02ad61.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc21, %Op.ref.loc21
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc21, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7bc) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.fa67f5.2]
// CHECK:STDOUT:   %bound_method.loc21_5.3: <bound method> = bound_method %a.ref.loc21, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc21_5.3(%a.ref.loc21, %.loc21_9)
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_hereogeneous_long_long_and_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.660: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2ed: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.660) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.062: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %.ca2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.062, %ImplicitAs.facet.2ed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.d51: type = facet_type <@AddAssignWith, @AddAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %U.bf2: %ImplicitAs.type.0ec = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.305: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a87 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7a4: %ImplicitAs.type.0ec = facet_value %i64, (%ImplicitAs.impl_witness.305) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.610: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.442, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.3d0: %AddAssignWith.type.d51 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.610) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.55a: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet.3d0) [concrete]
// CHECK:STDOUT:   %.4c9: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.55a, %AddAssignWith.facet.3d0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.57a765.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.57a765.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3ae: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %.25e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3ae, %ImplicitAs.facet.7a4 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.01e: type = facet_type <@SubAssignWith, @SubAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.67476b.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.67476b.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.2a8: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table.425, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.798be9.1: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.2, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.1509c5.1: %Cpp.long_long.as.SubAssignWith.impl.Op.type.798be9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.type.798be9.2: type = fn_type @Cpp.long_long.as.SubAssignWith.impl.Op.1, @Cpp.long_long.as.SubAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.1509c5.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.798be9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.b78: %SubAssignWith.type.01e = facet_value %Cpp.long_long, (%SubAssignWith.impl_witness.2a8) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.926: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i64, %SubAssignWith.facet.b78) [concrete]
// CHECK:STDOUT:   %.aed: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.926, %SubAssignWith.facet.b78 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.7824e6.1: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.1509c5.2, @Cpp.long_long.as.SubAssignWith.impl.Op.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.7824e6.2: <specific function> = specific_function %Cpp.long_long.as.SubAssignWith.impl.Op.1509c5.1, @Cpp.long_long.as.SubAssignWith.impl.Op.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.813: type = facet_type <@MulAssignWith, @MulAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.3bc: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table.769, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.26ce70.1: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.2, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.5d705c.1: %Cpp.long_long.as.MulAssignWith.impl.Op.type.26ce70.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.type.26ce70.2: type = fn_type @Cpp.long_long.as.MulAssignWith.impl.Op.1, @Cpp.long_long.as.MulAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.5d705c.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.26ce70.2 = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.f52: %MulAssignWith.type.813 = facet_value %Cpp.long_long, (%MulAssignWith.impl_witness.3bc) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.221: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i64, %MulAssignWith.facet.f52) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.221, %MulAssignWith.facet.f52 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.89951d.1: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.5d705c.2, @Cpp.long_long.as.MulAssignWith.impl.Op.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.89951d.2: <specific function> = specific_function %Cpp.long_long.as.MulAssignWith.impl.Op.5d705c.1, @Cpp.long_long.as.MulAssignWith.impl.Op.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.4ac: type = facet_type <@DivAssignWith, @DivAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.053c65.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.053c65.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.f1b: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table.538, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.5f499c.1: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.2, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.98d348.1: %Cpp.long_long.as.DivAssignWith.impl.Op.type.5f499c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.type.5f499c.2: type = fn_type @Cpp.long_long.as.DivAssignWith.impl.Op.1, @Cpp.long_long.as.DivAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.98d348.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.5f499c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.5d0: %DivAssignWith.type.4ac = facet_value %Cpp.long_long, (%DivAssignWith.impl_witness.f1b) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.ec0: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i64, %DivAssignWith.facet.5d0) [concrete]
// CHECK:STDOUT:   %.2a8: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.ec0, %DivAssignWith.facet.5d0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.2d709b.1: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.98d348.2, @Cpp.long_long.as.DivAssignWith.impl.Op.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.2d709b.2: <specific function> = specific_function %Cpp.long_long.as.DivAssignWith.impl.Op.98d348.1, @Cpp.long_long.as.DivAssignWith.impl.Op.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.6a8: type = facet_type <@ModAssignWith, @ModAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.563: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table.cdc, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.89fbed.1: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.2, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.6bf2ce.1: %Cpp.long_long.as.ModAssignWith.impl.Op.type.89fbed.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.type.89fbed.2: type = fn_type @Cpp.long_long.as.ModAssignWith.impl.Op.1, @Cpp.long_long.as.ModAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.6bf2ce.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.89fbed.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.0e4: %ModAssignWith.type.6a8 = facet_value %Cpp.long_long, (%ModAssignWith.impl_witness.563) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.a6d: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i64, %ModAssignWith.facet.0e4) [concrete]
// CHECK:STDOUT:   %.7bd: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.a6d, %ModAssignWith.facet.0e4 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.b20423.1: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.6bf2ce.2, @Cpp.long_long.as.ModAssignWith.impl.Op.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.b20423.2: <specific function> = specific_function %Cpp.long_long.as.ModAssignWith.impl.Op.6bf2ce.1, @Cpp.long_long.as.ModAssignWith.impl.Op.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.817: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.7e3: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table.d41, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.212afa.1: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.2, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.678907.1: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.212afa.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.212afa.2: type = fn_type @Cpp.long_long.as.BitAndAssignWith.impl.Op.1, @Cpp.long_long.as.BitAndAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.678907.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.212afa.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.8ef: %BitAndAssignWith.type.817 = facet_value %Cpp.long_long, (%BitAndAssignWith.impl_witness.7e3) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.89a: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i64, %BitAndAssignWith.facet.8ef) [concrete]
// CHECK:STDOUT:   %.920: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.89a, %BitAndAssignWith.facet.8ef [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.4fc9be.1: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.678907.2, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.4fc9be.2: <specific function> = specific_function %Cpp.long_long.as.BitAndAssignWith.impl.Op.678907.1, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.9cc: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.0f6: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table.446, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2aebbe.1: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.2, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bbf3fb.1: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2aebbe.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2aebbe.2: type = fn_type @Cpp.long_long.as.BitOrAssignWith.impl.Op.1, @Cpp.long_long.as.BitOrAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bbf3fb.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2aebbe.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.b74: %BitOrAssignWith.type.9cc = facet_value %Cpp.long_long, (%BitOrAssignWith.impl_witness.0f6) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.db9: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i64, %BitOrAssignWith.facet.b74) [concrete]
// CHECK:STDOUT:   %.ab1: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.db9, %BitOrAssignWith.facet.b74 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.80ce92.1: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.bbf3fb.2, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.80ce92.2: <specific function> = specific_function %Cpp.long_long.as.BitOrAssignWith.impl.Op.bbf3fb.1, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.801: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.400: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table.8e4, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c459c7.1: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.2, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.9c6ff1.1: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c459c7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c459c7.2: type = fn_type @Cpp.long_long.as.BitXorAssignWith.impl.Op.1, @Cpp.long_long.as.BitXorAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.9c6ff1.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c459c7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.f31: %BitXorAssignWith.type.801 = facet_value %Cpp.long_long, (%BitXorAssignWith.impl_witness.400) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.559: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i64, %BitXorAssignWith.facet.f31) [concrete]
// CHECK:STDOUT:   %.931: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.559, %BitXorAssignWith.facet.f31 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.c20ea8.1: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.9c6ff1.2, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.c20ea8.2: <specific function> = specific_function %Cpp.long_long.as.BitXorAssignWith.impl.Op.9c6ff1.1, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.d52: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.afc: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table.dd0, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.8c1aba.1: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e7679.1: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.8c1aba.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.8c1aba.2: type = fn_type @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1, @Cpp.long_long.as.LeftShiftAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e7679.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.8c1aba.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.41f: %LeftShiftAssignWith.type.d52 = facet_value %Cpp.long_long, (%LeftShiftAssignWith.impl_witness.afc) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.8f0: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i64, %LeftShiftAssignWith.facet.41f) [concrete]
// CHECK:STDOUT:   %.7e2: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.8f0, %LeftShiftAssignWith.facet.41f [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.b21942.1: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e7679.2, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.b21942.2: <specific function> = specific_function %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e7679.1, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.7d8: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.68a: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table.00b, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.959149.1: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.f75d42.1: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.959149.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.959149.2: type = fn_type @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1, @Cpp.long_long.as.RightShiftAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.f75d42.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.959149.2 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.e02: %RightShiftAssignWith.type.7d8 = facet_value %Cpp.long_long, (%RightShiftAssignWith.impl_witness.68a) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.263: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i64, %RightShiftAssignWith.facet.e02) [concrete]
// CHECK:STDOUT:   %.128: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.263, %RightShiftAssignWith.facet.e02 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.eff961.1: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.f75d42.2, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.eff961.2: <specific function> = specific_function %Cpp.long_long.as.RightShiftAssignWith.impl.Op.f75d42.1, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d6d = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.634: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.442 = impl_witness_table (%Core.import_ref.634), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c11: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a87 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.14b: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.2 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.67476b.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table.425 = impl_witness_table (%Core.import_ref.14b), @Cpp.long_long.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3dd: @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.type.1 (%Cpp.long_long.as.SubAssignWith.impl.Op.type.aafcc5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.SubAssignWith.impl.%Cpp.long_long.as.SubAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.SubAssignWith.impl.Op.67476b.2)]
// CHECK:STDOUT:   %Core.import_ref.1e7: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.2 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table.769 = impl_witness_table (%Core.import_ref.1e7), @Cpp.long_long.as.MulAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1a0: @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.type.1 (%Cpp.long_long.as.MulAssignWith.impl.Op.type.987caa.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.MulAssignWith.impl.%Cpp.long_long.as.MulAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.MulAssignWith.impl.Op.59c70e.2)]
// CHECK:STDOUT:   %Core.import_ref.90c: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.2 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.053c65.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table.538 = impl_witness_table (%Core.import_ref.90c), @Cpp.long_long.as.DivAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.4db: @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.type.1 (%Cpp.long_long.as.DivAssignWith.impl.Op.type.7d9a4f.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.DivAssignWith.impl.%Cpp.long_long.as.DivAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.DivAssignWith.impl.Op.053c65.2)]
// CHECK:STDOUT:   %Core.import_ref.620: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.2 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table.cdc = impl_witness_table (%Core.import_ref.620), @Cpp.long_long.as.ModAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.978: @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.type.1 (%Cpp.long_long.as.ModAssignWith.impl.Op.type.aee511.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.ModAssignWith.impl.%Cpp.long_long.as.ModAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.ModAssignWith.impl.Op.0a710f.2)]
// CHECK:STDOUT:   %Core.import_ref.f6c: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table.d41 = impl_witness_table (%Core.import_ref.f6c), @Cpp.long_long.as.BitAndAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.96f: @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitAndAssignWith.impl.Op.type.cd65f6.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitAndAssignWith.impl.%Cpp.long_long.as.BitAndAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.3a2cbb.2)]
// CHECK:STDOUT:   %Core.import_ref.4fa: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table.446 = impl_witness_table (%Core.import_ref.4fa), @Cpp.long_long.as.BitOrAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.152: @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitOrAssignWith.impl.Op.type.a360c5.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitOrAssignWith.impl.%Cpp.long_long.as.BitOrAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.65c84e.2)]
// CHECK:STDOUT:   %Core.import_ref.0a0: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.2 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table.8e4 = impl_witness_table (%Core.import_ref.0a0), @Cpp.long_long.as.BitXorAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.993: @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.1 (%Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c7963a.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.BitXorAssignWith.impl.%Cpp.long_long.as.BitXorAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.3b70cd.2)]
// CHECK:STDOUT:   %Core.import_ref.463: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table.dd0 = impl_witness_table (%Core.import_ref.463), @Cpp.long_long.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.3c4: @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.d78f49.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.LeftShiftAssignWith.impl.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.688a59.2)]
// CHECK:STDOUT:   %Core.import_ref.6d5: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table.00b = impl_witness_table (%Core.import_ref.6d5), @Cpp.long_long.as.RightShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.77b: @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.1 (%Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.2a7c37.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.RightShiftAssignWith.impl.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.9744e0.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_3: init %Cpp.long_long = converted %int_1.loc9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc10: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.4c9 = impl_witness_access constants.%AddAssignWith.impl_witness.610, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc10: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc10 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.57a765.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.1 = specific_constant imports.%Core.Op.c11, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.1]
// CHECK:STDOUT:   %Op.ref.loc10: %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc10, %Op.ref.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc10, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.57a765.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref.loc10, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref.loc10, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc10: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref.loc10)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref.loc10, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref.loc10, %.loc10_8.2)
// CHECK:STDOUT:   %a.ref.loc11: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc11: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc11_5: %.aed = impl_witness_access constants.%SubAssignWith.impl_witness.2a8, element0 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.1509c5.2]
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc11: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc11_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc11 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_5, @Cpp.long_long.as.SubAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.7824e6.1]
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %a.ref.loc11, %specific_fn.loc11
// CHECK:STDOUT:   %.loc11_5.2: %Cpp.long_long.as.SubAssignWith.impl.Op.type.798be9.1 = specific_constant imports.%Core.Op.3dd, @Cpp.long_long.as.SubAssignWith.impl(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.1509c5.1]
// CHECK:STDOUT:   %Op.ref.loc11: %Cpp.long_long.as.SubAssignWith.impl.Op.type.798be9.1 = name_ref Op, %.loc11_5.2 [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.1509c5.1]
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc11, %Op.ref.loc11
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc11, @Cpp.long_long.as.SubAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn.7824e6.2]
// CHECK:STDOUT:   %bound_method.loc11_5.3: <bound method> = bound_method %a.ref.loc11, %Cpp.long_long.as.SubAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc11_8: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc11_8: <bound method> = bound_method %b.ref.loc11, %impl.elem0.loc11_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc11: init %Cpp.long_long = call %bound_method.loc11_8(%b.ref.loc11)
// CHECK:STDOUT:   %.loc11_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %.loc11_8.2: %Cpp.long_long = converted %b.ref.loc11, %.loc11_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_5.3(%a.ref.loc11, %.loc11_8.2)
// CHECK:STDOUT:   %a.ref.loc12: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc12: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc12_5: %.9c3 = impl_witness_access constants.%MulAssignWith.impl_witness.3bc, element0 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.5d705c.2]
// CHECK:STDOUT:   %bound_method.loc12_5.1: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc12: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc12_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc12 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_5, @Cpp.long_long.as.MulAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.89951d.1]
// CHECK:STDOUT:   %bound_method.loc12_5.2: <bound method> = bound_method %a.ref.loc12, %specific_fn.loc12
// CHECK:STDOUT:   %.loc12_5.2: %Cpp.long_long.as.MulAssignWith.impl.Op.type.26ce70.1 = specific_constant imports.%Core.Op.1a0, @Cpp.long_long.as.MulAssignWith.impl(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.5d705c.1]
// CHECK:STDOUT:   %Op.ref.loc12: %Cpp.long_long.as.MulAssignWith.impl.Op.type.26ce70.1 = name_ref Op, %.loc12_5.2 [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.5d705c.1]
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc12, %Op.ref.loc12
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc12, @Cpp.long_long.as.MulAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn.89951d.2]
// CHECK:STDOUT:   %bound_method.loc12_5.3: <bound method> = bound_method %a.ref.loc12, %Cpp.long_long.as.MulAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc12_8: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc12_8: <bound method> = bound_method %b.ref.loc12, %impl.elem0.loc12_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc12: init %Cpp.long_long = call %bound_method.loc12_8(%b.ref.loc12)
// CHECK:STDOUT:   %.loc12_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %.loc12_8.2: %Cpp.long_long = converted %b.ref.loc12, %.loc12_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc12_5.3(%a.ref.loc12, %.loc12_8.2)
// CHECK:STDOUT:   %a.ref.loc13: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc13: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc13_5: %.2a8 = impl_witness_access constants.%DivAssignWith.impl_witness.f1b, element0 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.98d348.2]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %a.ref.loc13, %impl.elem0.loc13_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc13: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc13_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc13 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_5, @Cpp.long_long.as.DivAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.2d709b.1]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %a.ref.loc13, %specific_fn.loc13
// CHECK:STDOUT:   %.loc13_5.2: %Cpp.long_long.as.DivAssignWith.impl.Op.type.5f499c.1 = specific_constant imports.%Core.Op.4db, @Cpp.long_long.as.DivAssignWith.impl(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.98d348.1]
// CHECK:STDOUT:   %Op.ref.loc13: %Cpp.long_long.as.DivAssignWith.impl.Op.type.5f499c.1 = name_ref Op, %.loc13_5.2 [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.98d348.1]
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc13, %Op.ref.loc13
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc13, @Cpp.long_long.as.DivAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn.2d709b.2]
// CHECK:STDOUT:   %bound_method.loc13_5.3: <bound method> = bound_method %a.ref.loc13, %Cpp.long_long.as.DivAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc13_8: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc13_8: <bound method> = bound_method %b.ref.loc13, %impl.elem0.loc13_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc13: init %Cpp.long_long = call %bound_method.loc13_8(%b.ref.loc13)
// CHECK:STDOUT:   %.loc13_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %.loc13_8.2: %Cpp.long_long = converted %b.ref.loc13, %.loc13_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_5.3(%a.ref.loc13, %.loc13_8.2)
// CHECK:STDOUT:   %a.ref.loc14: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc14: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc14_5: %.7bd = impl_witness_access constants.%ModAssignWith.impl_witness.563, element0 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.6bf2ce.2]
// CHECK:STDOUT:   %bound_method.loc14_5.1: <bound method> = bound_method %a.ref.loc14, %impl.elem0.loc14_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc14: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc14_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc14 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_5, @Cpp.long_long.as.ModAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.b20423.1]
// CHECK:STDOUT:   %bound_method.loc14_5.2: <bound method> = bound_method %a.ref.loc14, %specific_fn.loc14
// CHECK:STDOUT:   %.loc14_5.2: %Cpp.long_long.as.ModAssignWith.impl.Op.type.89fbed.1 = specific_constant imports.%Core.Op.978, @Cpp.long_long.as.ModAssignWith.impl(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.6bf2ce.1]
// CHECK:STDOUT:   %Op.ref.loc14: %Cpp.long_long.as.ModAssignWith.impl.Op.type.89fbed.1 = name_ref Op, %.loc14_5.2 [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.6bf2ce.1]
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc14, %Op.ref.loc14
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc14, @Cpp.long_long.as.ModAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn.b20423.2]
// CHECK:STDOUT:   %bound_method.loc14_5.3: <bound method> = bound_method %a.ref.loc14, %Cpp.long_long.as.ModAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc14_8: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc14_8: <bound method> = bound_method %b.ref.loc14, %impl.elem0.loc14_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc14: init %Cpp.long_long = call %bound_method.loc14_8(%b.ref.loc14)
// CHECK:STDOUT:   %.loc14_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %.loc14_8.2: %Cpp.long_long = converted %b.ref.loc14, %.loc14_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_5.3(%a.ref.loc14, %.loc14_8.2)
// CHECK:STDOUT:   %a.ref.loc16: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc16: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc16_5: %.920 = impl_witness_access constants.%BitAndAssignWith.impl_witness.7e3, element0 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.678907.2]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %a.ref.loc16, %impl.elem0.loc16_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc16: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc16_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc16 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16_5, @Cpp.long_long.as.BitAndAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.4fc9be.1]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %a.ref.loc16, %specific_fn.loc16
// CHECK:STDOUT:   %.loc16_5.2: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.212afa.1 = specific_constant imports.%Core.Op.96f, @Cpp.long_long.as.BitAndAssignWith.impl(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.678907.1]
// CHECK:STDOUT:   %Op.ref.loc16: %Cpp.long_long.as.BitAndAssignWith.impl.Op.type.212afa.1 = name_ref Op, %.loc16_5.2 [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.678907.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc16, %Op.ref.loc16
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc16, @Cpp.long_long.as.BitAndAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn.4fc9be.2]
// CHECK:STDOUT:   %bound_method.loc16_5.3: <bound method> = bound_method %a.ref.loc16, %Cpp.long_long.as.BitAndAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc16_8: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc16_8: <bound method> = bound_method %b.ref.loc16, %impl.elem0.loc16_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc16: init %Cpp.long_long = call %bound_method.loc16_8(%b.ref.loc16)
// CHECK:STDOUT:   %.loc16_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc16
// CHECK:STDOUT:   %.loc16_8.2: %Cpp.long_long = converted %b.ref.loc16, %.loc16_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_5.3(%a.ref.loc16, %.loc16_8.2)
// CHECK:STDOUT:   %a.ref.loc17: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc17: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc17_5: %.ab1 = impl_witness_access constants.%BitOrAssignWith.impl_witness.0f6, element0 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.bbf3fb.2]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %a.ref.loc17, %impl.elem0.loc17_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc17: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc17_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc17 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17_5, @Cpp.long_long.as.BitOrAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.80ce92.1]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %a.ref.loc17, %specific_fn.loc17
// CHECK:STDOUT:   %.loc17_5.2: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2aebbe.1 = specific_constant imports.%Core.Op.152, @Cpp.long_long.as.BitOrAssignWith.impl(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.bbf3fb.1]
// CHECK:STDOUT:   %Op.ref.loc17: %Cpp.long_long.as.BitOrAssignWith.impl.Op.type.2aebbe.1 = name_ref Op, %.loc17_5.2 [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.bbf3fb.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc17, %Op.ref.loc17
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc17, @Cpp.long_long.as.BitOrAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn.80ce92.2]
// CHECK:STDOUT:   %bound_method.loc17_5.3: <bound method> = bound_method %a.ref.loc17, %Cpp.long_long.as.BitOrAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc17_8: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc17_8: <bound method> = bound_method %b.ref.loc17, %impl.elem0.loc17_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc17: init %Cpp.long_long = call %bound_method.loc17_8(%b.ref.loc17)
// CHECK:STDOUT:   %.loc17_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc17
// CHECK:STDOUT:   %.loc17_8.2: %Cpp.long_long = converted %b.ref.loc17, %.loc17_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc17_5.3(%a.ref.loc17, %.loc17_8.2)
// CHECK:STDOUT:   %a.ref.loc18: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc18: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.931 = impl_witness_access constants.%BitXorAssignWith.impl_witness.400, element0 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.9c6ff1.2]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %a.ref.loc18, %impl.elem0.loc18_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc18: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc18_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc18 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18_5, @Cpp.long_long.as.BitXorAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.c20ea8.1]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %a.ref.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_5.2: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c459c7.1 = specific_constant imports.%Core.Op.993, @Cpp.long_long.as.BitXorAssignWith.impl(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.9c6ff1.1]
// CHECK:STDOUT:   %Op.ref.loc18: %Cpp.long_long.as.BitXorAssignWith.impl.Op.type.c459c7.1 = name_ref Op, %.loc18_5.2 [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.9c6ff1.1]
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc18, %Op.ref.loc18
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @Cpp.long_long.as.BitXorAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn.c20ea8.2]
// CHECK:STDOUT:   %bound_method.loc18_5.3: <bound method> = bound_method %a.ref.loc18, %Cpp.long_long.as.BitXorAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc18_8: <bound method> = bound_method %b.ref.loc18, %impl.elem0.loc18_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc18: init %Cpp.long_long = call %bound_method.loc18_8(%b.ref.loc18)
// CHECK:STDOUT:   %.loc18_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc18
// CHECK:STDOUT:   %.loc18_8.2: %Cpp.long_long = converted %b.ref.loc18, %.loc18_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc18_5.3(%a.ref.loc18, %.loc18_8.2)
// CHECK:STDOUT:   %a.ref.loc19: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc19: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc19_5: %.7e2 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.afc, element0 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e7679.2]
// CHECK:STDOUT:   %bound_method.loc19_5.1: <bound method> = bound_method %a.ref.loc19, %impl.elem0.loc19_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc19: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc19_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc19 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19_5, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.b21942.1]
// CHECK:STDOUT:   %bound_method.loc19_5.2: <bound method> = bound_method %a.ref.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_5.2: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.8c1aba.1 = specific_constant imports.%Core.Op.3c4, @Cpp.long_long.as.LeftShiftAssignWith.impl(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e7679.1]
// CHECK:STDOUT:   %Op.ref.loc19: %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.type.8c1aba.1 = name_ref Op, %.loc19_5.2 [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.4e7679.1]
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc19, %Op.ref.loc19
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc19, @Cpp.long_long.as.LeftShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn.b21942.2]
// CHECK:STDOUT:   %bound_method.loc19_5.3: <bound method> = bound_method %a.ref.loc19, %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc19_9: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %b.ref.loc19, %impl.elem0.loc19_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc19: init %Cpp.long_long = call %bound_method.loc19_9(%b.ref.loc19)
// CHECK:STDOUT:   %.loc19_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc19
// CHECK:STDOUT:   %.loc19_9.2: %Cpp.long_long = converted %b.ref.loc19, %.loc19_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.3(%a.ref.loc19, %.loc19_9.2)
// CHECK:STDOUT:   %a.ref.loc20: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref.loc20: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc20_5: %.128 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.68a, element0 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.f75d42.2]
// CHECK:STDOUT:   %bound_method.loc20_5.1: <bound method> = bound_method %a.ref.loc20, %impl.elem0.loc20_5
// CHECK:STDOUT:   %ImplicitAs.facet.loc20: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc20_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet.loc20 [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20_5, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.eff961.1]
// CHECK:STDOUT:   %bound_method.loc20_5.2: <bound method> = bound_method %a.ref.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_5.2: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.959149.1 = specific_constant imports.%Core.Op.77b, @Cpp.long_long.as.RightShiftAssignWith.impl(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.f75d42.1]
// CHECK:STDOUT:   %Op.ref.loc20: %Cpp.long_long.as.RightShiftAssignWith.impl.Op.type.959149.1 = name_ref Op, %.loc20_5.2 [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.f75d42.1]
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref.loc20, %Op.ref.loc20
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc20, @Cpp.long_long.as.RightShiftAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn.eff961.2]
// CHECK:STDOUT:   %bound_method.loc20_5.3: <bound method> = bound_method %a.ref.loc20, %Cpp.long_long.as.RightShiftAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc20_9: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20_9: <bound method> = bound_method %b.ref.loc20, %impl.elem0.loc20_9
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call.loc20: init %Cpp.long_long = call %bound_method.loc20_9(%b.ref.loc20)
// CHECK:STDOUT:   %.loc20_9.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call.loc20
// CHECK:STDOUT:   %.loc20_9.2: %Cpp.long_long = converted %b.ref.loc20, %.loc20_9.1
// CHECK:STDOUT:   %Cpp.long_long.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc20_5.3(%a.ref.loc20, %.loc20_9.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_long_and_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.660: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.660) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.062: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ca2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.062, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.5ab: type = facet_type <@AddAssignWith, @AddAssignWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %U.bf2: %ImplicitAs.type.0ec = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.b05: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.442, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3d164b.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.7e3843.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3d164b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.3d164b.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.7e3843.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3d164b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.6b7: %AddAssignWith.type.5ab = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.b05) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.ca5: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(Core.IntLiteral, %AddAssignWith.facet.6b7) [concrete]
// CHECK:STDOUT:   %.eab: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.ca5, %AddAssignWith.facet.6b7 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.d7abb9.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.7e3843.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.d7abb9.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.7e3843.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc8_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d6d = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.634: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.442 = impl_witness_table (%Core.import_ref.634), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c11: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndIntLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9_5: %.eab = impl_witness_access constants.%AddAssignWith.impl_witness.b05, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.7e3843.2]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc9_5
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.660) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_5.1: %ImplicitAs.type.0ec = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc9_5, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.d7abb9.1]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %.loc9_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3d164b.1 = specific_constant imports.%Core.Op.c11, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.7e3843.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.3d164b.1 = name_ref Op, %.loc9_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.7e3843.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.d7abb9.2]
// CHECK:STDOUT:   %bound_method.loc9_5.3: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc9_8: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc9_8: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %Cpp.long_long = call %bound_method.loc9_8(%int_1.loc9) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_8.1: %Cpp.long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc9_8.2: %Cpp.long_long = converted %int_1.loc9, %.loc9_8.1 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_5.3(%a.ref, %.loc9_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.3(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_assignment_heterogeneous_long_long_and_runtime_i64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.221: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.660: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2ed: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.660) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.062: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.2ed) [concrete]
// CHECK:STDOUT:   %.ca2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.062, %ImplicitAs.facet.2ed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %pattern_type.f57: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.091: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6e8: %ImplicitAs.type.221 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.091) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.6e8) [concrete]
// CHECK:STDOUT:   %.f34: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a04, %ImplicitAs.facet.6e8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.89f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d98: %i64 = int_value 1 [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.d51: type = facet_type <@AddAssignWith, @AddAssignWith(%i64)> [concrete]
// CHECK:STDOUT:   %U.bf2: %ImplicitAs.type.0ec = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%U.bf2) [symbolic]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.305: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a87 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7a4: %ImplicitAs.type.0ec = facet_value %i64, (%ImplicitAs.impl_witness.305) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.610: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table.442, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.1: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.2, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.1: %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.2: type = fn_type @Cpp.long_long.as.AddAssignWith.impl.Op.1, @Cpp.long_long.as.AddAssignWith.impl(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.3d0: %AddAssignWith.type.d51 = facet_value %Cpp.long_long, (%AddAssignWith.impl_witness.610) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.55a: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i64, %AddAssignWith.facet.3d0) [concrete]
// CHECK:STDOUT:   %.4c9: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.55a, %AddAssignWith.facet.3d0 [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.57a765.1: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.2, @Cpp.long_long.as.AddAssignWith.impl.Op.1(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.57a765.2: <specific function> = specific_function %Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.1, @Cpp.long_long.as.AddAssignWith.impl.Op.2(%ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3ae: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %.25e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3ae, %ImplicitAs.facet.7a4 [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.type: type = fn_type @i64.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert: %i64.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc8_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d6d = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.634: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.2 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.1) = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.2 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table.442 = impl_witness_table (%Core.import_ref.634), @Cpp.long_long.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.c11: @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.type.1 (%Cpp.long_long.as.AddAssignWith.impl.Op.type.524923.2) = import_ref Core//prelude/types/cpp/int, Op, loaded [symbolic = @Cpp.long_long.as.AddAssignWith.impl.%Cpp.long_long.as.AddAssignWith.impl.Op.1 (constants.%Cpp.long_long.as.AddAssignWith.impl.Op.8f0746.2)]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i64.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a87 = impl_witness_table (%Core.import_ref.06c923.2), @i64.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CompoundAssignmentLongLongAndRuntimeI64() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %Cpp.long_long = call %bound_method.loc8(%int_1.loc8) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1.loc8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.f57 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0.loc9: %.f34 = impl_witness_access constants.%ImplicitAs.impl_witness.091, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a1b]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.89f]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i64 = call %bound_method.loc9_16.2(%int_1.loc9) [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc9_16.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %.loc9_16.2: %i64 = converted %int_1.loc9, %.loc9_16.1 [concrete = constants.%int_1.d98]
// CHECK:STDOUT:   %b: %i64 = value_binding b, %.loc9_16.2
// CHECK:STDOUT:   %a.ref: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i64 = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0.loc10_5: %.4c9 = impl_witness_access constants.%AddAssignWith.impl_witness.610, element0 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.2]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %a.ref, %impl.elem0.loc10_5
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ec = facet_value constants.%i64, (constants.%ImplicitAs.impl_witness.305) [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %.loc10_5.1: %ImplicitAs.type.0ec = converted constants.%i64, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet.7a4]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_5, @Cpp.long_long.as.AddAssignWith.impl.Op.1(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.57a765.1]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %a.ref, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.1 = specific_constant imports.%Core.Op.c11, @Cpp.long_long.as.AddAssignWith.impl(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.1]
// CHECK:STDOUT:   %Op.ref: %Cpp.long_long.as.AddAssignWith.impl.Op.type.6f7c53.1 = name_ref Op, %.loc10_5.2 [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.0167bf.1]
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %a.ref, %Op.ref
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Cpp.long_long.as.AddAssignWith.impl.Op.2(constants.%ImplicitAs.facet.7a4) [concrete = constants.%Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn.57a765.2]
// CHECK:STDOUT:   %bound_method.loc10_5.3: <bound method> = bound_method %a.ref, %Cpp.long_long.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %impl.elem0.loc10_8: %.25e = impl_witness_access constants.%ImplicitAs.impl_witness.305, element0 [concrete = constants.%i64.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc10_8: <bound method> = bound_method %b.ref, %impl.elem0.loc10_8
// CHECK:STDOUT:   %i64.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc10_8(%b.ref)
// CHECK:STDOUT:   %.loc10_8.1: %Cpp.long_long = value_of_initializer %i64.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_8.2: %Cpp.long_long = converted %b.ref, %.loc10_8.1
// CHECK:STDOUT:   %Cpp.long_long.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_5.3(%a.ref, %.loc10_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.3(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- increment_decrement_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.long_long: type = class_type @LongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Cpp.long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ec: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.long_long)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.660: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ec = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.660) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.062: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ca2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.062, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.b19: %Cpp.long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness imports.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Cpp.long_long, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.929: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet) [concrete]
// CHECK:STDOUT:   %.ac5: type = fn_type_with_self_type %Inc.WithSelf.Op.type.929, %Inc.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.type: type = fn_type @Cpp.long_long.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op: %Cpp.long_long.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness imports.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %Cpp.long_long, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.2cb: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.a4b: type = fn_type_with_self_type %Dec.WithSelf.Op.type.2cb, %Dec.facet [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op.type: type = fn_type @Cpp.long_long.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op: %Cpp.long_long.as.Dec.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc8_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .long_long = constants.%Cpp.long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d6d = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.e33 [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.1: %Cpp.long_long.as.Inc.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %Inc.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.1), @Cpp.long_long.as.Inc.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.2: %Cpp.long_long.as.Dec.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
// CHECK:STDOUT:   %Dec.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.2), @Cpp.long_long.as.Dec.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IncrementDecrementLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ca2 = impl_witness_access constants.%ImplicitAs.impl_witness.660, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.b19]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %long_long.ref [concrete = constants.%Cpp.long_long] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %long_long.ref: type = name_ref long_long, constants.%Cpp.long_long [concrete = constants.%Cpp.long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.long_long = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc9: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc9: %.ac5 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %a.ref.loc9, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.long_long.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9(%a.ref.loc9)
// CHECK:STDOUT:   %a.ref.loc10: ref %Cpp.long_long = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc10: %.a4b = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%Cpp.long_long.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %Cpp.long_long.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10(%a.ref.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %i64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.3(%self.param: ref %Cpp.long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsigned_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long: type = class_type @ULongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.4cd: type = pattern_type %Cpp.unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1c8: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.14f: type = facet_type <@ImplicitAs, @ImplicitAs(%u64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fa4: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bd1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e8c: %ImplicitAs.type.1c8 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fa4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e2: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet.e8c) [concrete]
// CHECK:STDOUT:   %.ee9: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e2, %ImplicitAs.facet.e8c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce [concrete]
// CHECK:STDOUT:   %int_1.877: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongLongResult: type = class_type @ULongLongResult [concrete]
// CHECK:STDOUT:   %pattern_type.97e: type = pattern_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.type: type = cpp_overload_set_type @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.130: type = ptr_type %ULongLongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.76ef5b.1: type = fn_type @PassULongLong__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.030145.1: %PassULongLong__carbon_thunk.type.76ef5b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.b35: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b68(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.170: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9c4: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.42f, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b18: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.633: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b18 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d29: %ImplicitAs.type.14f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.9c4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8cc: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u64, %ImplicitAs.facet.d29) [concrete]
// CHECK:STDOUT:   %.93e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8cc, %ImplicitAs.facet.d29 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eb0: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.633 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.633, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.7e7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.01f: %u64 = int_value 1 [concrete]
// CHECK:STDOUT:   %ULongResult: type = class_type @ULongResult [concrete]
// CHECK:STDOUT:   %pattern_type.97f: type = pattern_type %ULongResult [concrete]
// CHECK:STDOUT:   %ptr.e40: type = ptr_type %ULongResult [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.type.76ef5b.2: type = fn_type @PassULongLong__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.030145.2: %PassULongLong__carbon_thunk.type.76ef5b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.7f6: type = facet_type <@As, @As(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.f99: <witness> = impl_witness imports.%As.impl_witness_table.14e [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.7f6 = facet_value Core.IntLiteral, (%As.impl_witness.f99) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.a98: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Cpp.unsigned_long_long, %As.facet) [concrete]
// CHECK:STDOUT:   %.912: type = fn_type_with_self_type %As.WithSelf.Convert.type.a98, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.83c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.2bc [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fdc: %ImplicitAs.type.7cb = facet_value %Cpp.unsigned_long_long, (%ImplicitAs.impl_witness.83c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a28: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.fdc) [concrete]
// CHECK:STDOUT:   %.27b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a28, %ImplicitAs.facet.fdc [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type: type = fn_type @Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.877, %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.c73 [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ed3: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.c73)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.303: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.393, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.65e: %ImplicitAs.type.ed3 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.303) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.2e4: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.c73, %ImplicitAs.facet.65e) [concrete]
// CHECK:STDOUT:   %.17b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.2e4, %ImplicitAs.facet.65e [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3e4: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.943: %f32.c73 = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.943) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc24_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor.type: type = fn_type @ULongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongResult.cpp_destructor: %ULongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.type: type = fn_type @ULongLongResult.cpp_destructor [concrete]
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor: %ULongLongResult.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .CppCompat = %CppCompat.f0dbf9.1
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long_long = constants.%Cpp.unsigned_long_long
// CHECK:STDOUT:     .ULongLongResult = %ULongLongResult.decl
// CHECK:STDOUT:     .PassULongLong = %PassULongLong.cpp_overload_set.value
// CHECK:STDOUT:     .ULongResult = %ULongResult.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.f0dbf9.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .ULongLong64 = %Core.ULongLong64
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ULongLong64: type = import_ref Core//prelude/types/cpp/int, ULongLong64, loaded [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ae6 = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bd1 = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.ImplicitAs.impl.f5b [concrete]
// CHECK:STDOUT:   %ULongLongResult.decl: type = class_decl @ULongLongResult [concrete = constants.%ULongLongResult] {} {}
// CHECK:STDOUT:   %PassULongLong.cpp_overload_set.value: %PassULongLong.cpp_overload_set.type = cpp_overload_set_value @PassULongLong.cpp_overload_set [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.1: %PassULongLong__carbon_thunk.type.76ef5b.1 = fn_decl @PassULongLong__carbon_thunk.1 [concrete = constants.%PassULongLong__carbon_thunk.030145.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.aa5: @Core.IntLiteral.as.ImplicitAs.impl.b68.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b68.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.170)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.42f = impl_witness_table (%Core.import_ref.aa5), @Core.IntLiteral.as.ImplicitAs.impl.b68 [concrete]
// CHECK:STDOUT:   %ULongResult.decl: type = class_decl @ULongResult [concrete = constants.%ULongResult] {} {}
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.decl.3ca866.2: %PassULongLong__carbon_thunk.type.76ef5b.2 = fn_decl @PassULongLong__carbon_thunk.2 [concrete = constants.%PassULongLong__carbon_thunk.030145.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/types/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table.14e = impl_witness_table (%Core.import_ref.06c923.2), @Core.IntLiteral.as.As.impl.f3d [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.2bc = impl_witness_table (%Core.import_ref.06c923.3), @Cpp.unsigned_long_long.as.ImplicitAs.impl.437 [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.393 = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long.patt: %pattern_type.4cd = value_binding_pattern cpp_unsigned_long_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_34: type = splice_block %unsigned_long_long.ref.loc15 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc15: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.ee9 = impl_witness_access constants.%ImplicitAs.impl_witness.fa4, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3ce]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %Cpp.unsigned_long_long = call %bound_method.loc15(%int_1.loc15) [concrete = constants.%int_1.877]
// CHECK:STDOUT:   %.loc15_56.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_1.877]
// CHECK:STDOUT:   %.loc15_56.2: %Cpp.unsigned_long_long = converted %int_1.loc15, %.loc15_56.1 [concrete = constants.%int_1.877]
// CHECK:STDOUT:   %cpp_unsigned_long_long: %Cpp.unsigned_long_long = value_binding cpp_unsigned_long_long, %.loc15_56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_unsigned_long_long_result.patt: %pattern_type.97e = value_binding_pattern cpp_unsigned_long_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_67: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc16: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc16: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc16_107.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc16: %ptr.130 = addr_of %.loc16_107.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.3ca866.1(%cpp_unsigned_long_long.ref.loc16, %addr.loc16)
// CHECK:STDOUT:   %.loc16_107.2: init %ULongLongResult to %.loc16_107.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc16
// CHECK:STDOUT:   %.loc16_48: type = splice_block %ULongLongResult.ref.loc16 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc16_45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc16: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_107.3: ref %ULongLongResult = temporary %.loc16_107.1, %.loc16_107.2
// CHECK:STDOUT:   %.loc16_107.4: %ULongLongResult = acquire_value %.loc16_107.3
// CHECK:STDOUT:   %cpp_unsigned_long_long_result: %ULongLongResult = value_binding cpp_unsigned_long_long_result, %.loc16_107.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_long.patt: %pattern_type.4cd = value_binding_pattern cpp_compat_ulong_long [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_unsigned_long_long.ref.loc18: %Cpp.unsigned_long_long = name_ref cpp_unsigned_long_long, %cpp_unsigned_long_long
// CHECK:STDOUT:   %.loc18: type = splice_block %ULongLong64.ref [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CppCompat.ref: <namespace> = name_ref CppCompat, imports.%CppCompat.f0dbf9.1 [concrete = imports.%CppCompat.f0dbf9.1]
// CHECK:STDOUT:     %ULongLong64.ref: type = name_ref ULongLong64, imports.%Core.ULongLong64 [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_compat_ulong_long: %Cpp.unsigned_long_long = value_binding cpp_compat_ulong_long, %cpp_unsigned_long_long.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cpp_compat_ulong_long_result.patt: %pattern_type.97e = value_binding_pattern cpp_compat_ulong_long_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_66: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc19: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %cpp_compat_ulong_long.ref: %Cpp.unsigned_long_long = name_ref cpp_compat_ulong_long, %cpp_compat_ulong_long
// CHECK:STDOUT:   %.loc19_105.1: ref %ULongLongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc19: %ptr.130 = addr_of %.loc19_105.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.3ca866.1(%cpp_compat_ulong_long.ref, %addr.loc19)
// CHECK:STDOUT:   %.loc19_105.2: init %ULongLongResult to %.loc19_105.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_47: type = splice_block %ULongLongResult.ref.loc19 [concrete = constants.%ULongLongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc19_44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongLongResult.ref.loc19: type = name_ref ULongLongResult, imports.%ULongLongResult.decl [concrete = constants.%ULongLongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_105.3: ref %ULongLongResult = temporary %.loc19_105.1, %.loc19_105.2
// CHECK:STDOUT:   %.loc19_105.4: %ULongLongResult = acquire_value %.loc19_105.3
// CHECK:STDOUT:   %cpp_compat_ulong_long_result: %ULongLongResult = value_binding cpp_compat_ulong_long_result, %.loc19_105.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64.patt: %pattern_type.b35 = value_binding_pattern carbon_u64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %u64: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %impl.elem0.loc21: %.93e = impl_witness_access constants.%ImplicitAs.impl_witness.9c4, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.633]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eb0]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.7e7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %u64 = call %bound_method.loc21_25.2(%int_1.loc21) [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc21_25.1: %u64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %.loc21_25.2: %u64 = converted %int_1.loc21, %.loc21_25.1 [concrete = constants.%int_1.01f]
// CHECK:STDOUT:   %carbon_u64: %u64 = value_binding carbon_u64, %.loc21_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %carbon_u64_result.patt: %pattern_type.97f = value_binding_pattern carbon_u64_result [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_51: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassULongLong.ref.loc22: %PassULongLong.cpp_overload_set.type = name_ref PassULongLong, imports.%PassULongLong.cpp_overload_set.value [concrete = constants.%PassULongLong.cpp_overload_set.value]
// CHECK:STDOUT:   %carbon_u64.ref: %u64 = name_ref carbon_u64, %carbon_u64
// CHECK:STDOUT:   %.loc22_79.1: ref %ULongResult = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.e40 = addr_of %.loc22_79.1
// CHECK:STDOUT:   %PassULongLong__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%PassULongLong__carbon_thunk.decl.3ca866.2(%carbon_u64.ref, %addr.loc22)
// CHECK:STDOUT:   %.loc22_79.2: init %ULongResult to %.loc22_79.1 = mark_in_place_init %PassULongLong__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_36: type = splice_block %ULongResult.ref [concrete = constants.%ULongResult] {
// CHECK:STDOUT:     %Cpp.ref.loc22_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ULongResult.ref: type = name_ref ULongResult, imports.%ULongResult.decl [concrete = constants.%ULongResult]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_79.3: ref %ULongResult = temporary %.loc22_79.1, %.loc22_79.2
// CHECK:STDOUT:   %.loc22_79.4: %ULongResult = acquire_value %.loc22_79.3
// CHECK:STDOUT:   %carbon_u64_result: %ULongResult = value_binding carbon_u64_result, %.loc22_79.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.18b = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.18b = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc24_64.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc24_64: %.17b = impl_witness_access constants.%ImplicitAs.impl_witness.303, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
// CHECK:STDOUT:   %bound_method.loc24_64.1: <bound method> = bound_method %float, %impl.elem0.loc24_64 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24_64, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_64.2: <bound method> = bound_method %float, %specific_fn.loc24 [concrete = constants.%bound_method.3e4]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.c73 = call %bound_method.loc24_64.2(%float) [concrete = constants.%float.943]
// CHECK:STDOUT:   %.loc24_64.2: init %f32.c73 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.943]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_64.3: ref %f32.c73 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc24_64.4: init %f32.c73 to %.loc24_64.3 = in_place_init %.loc24_64.2 [concrete = constants.%float.943]
// CHECK:STDOUT:   %.loc24_64.5: init %array_type to %a.var = array_init (%.loc24_64.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_3: init %array_type = converted %.loc24_64.1, %.loc24_64.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc24_3
// CHECK:STDOUT:   %.loc24_55: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32: type = type_literal constants.%f32.c73 [concrete = constants.%f32.c73]
// CHECK:STDOUT:     %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc24: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:     %impl.elem0.loc24_30.1: %.912 = impl_witness_access constants.%As.impl_witness.f99, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_30.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc24_30.1(%int_1.loc24) [concrete = constants.%int_1.877]
// CHECK:STDOUT:     %.loc24_30.1: %Cpp.unsigned_long_long = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.877]
// CHECK:STDOUT:     %.loc24_30.2: %Cpp.unsigned_long_long = converted %int_1.loc24, %.loc24_30.1 [concrete = constants.%int_1.877]
// CHECK:STDOUT:     %impl.elem0.loc24_30.2: %.27b = impl_witness_access constants.%ImplicitAs.impl_witness.83c, element0 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc24_30.2: <bound method> = bound_method %.loc24_30.2, %impl.elem0.loc24_30.2 [concrete = constants.%Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc24_30.2(%.loc24_30.2) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.3: Core.IntLiteral = value_of_initializer %Cpp.unsigned_long_long.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc24_30.4: Core.IntLiteral = converted %.loc24_30.2, %.loc24_30.3 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc24_30.4, %f32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   %ULongResult.cpp_destructor.bound: <bound method> = bound_method %.loc22_79.3, constants.%ULongResult.cpp_destructor
// CHECK:STDOUT:   %ULongResult.cpp_destructor.call: init %empty_tuple.type = call %ULongResult.cpp_destructor.bound(%.loc22_79.3)
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_105.3, constants.%ULongLongResult.cpp_destructor
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.call.loc19: init %empty_tuple.type = call %ULongLongResult.cpp_destructor.bound.loc19(%.loc19_105.3)
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.bound.loc16: <bound method> = bound_method %.loc16_107.3, constants.%ULongLongResult.cpp_destructor
// CHECK:STDOUT:   %ULongLongResult.cpp_destructor.call.loc16: init %empty_tuple.type = call %ULongLongResult.cpp_destructor.bound.loc16(%.loc16_107.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.2(%self.param: ref %f32.c73) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.3(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_unsigned_long_long.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long: type = class_type @ULongLong64 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %u64.builtin: type = int_type unsigned, %int_64 [concrete]
// CHECK:STDOUT:   %pattern_type.4cd: type = pattern_type %Cpp.unsigned_long_long [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1c8: type = facet_type <@ImplicitAs, @ImplicitAs(%Cpp.unsigned_long_long)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fa4: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bd1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.1c8 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fa4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6e2: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Cpp.unsigned_long_long, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ee9: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6e2, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.877: %Cpp.unsigned_long_long = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.15f: <witness> = impl_witness imports.%Copy.impl_witness_table.217 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Cpp.unsigned_long_long, (%Copy.impl_witness.15f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a3f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.294: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a3f, %Copy.facet [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op.type: type = fn_type @Cpp.unsigned_long_long.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op: %Cpp.unsigned_long_long.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .unsigned_long_long = constants.%Cpp.unsigned_long_long
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bd1 = impl_witness_table (%Core.import_ref.06c), @Core.IntLiteral.as.ImplicitAs.impl.f5b [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Cpp.unsigned_long_long.as.Copy.impl.Op.type = import_ref Core//prelude/types/cpp/int, loc{{\d+_\d+}}, loaded [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.217 = impl_witness_table (%Core.import_ref.9c3), @Cpp.unsigned_long_long.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyUnsignedLongLong() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.4cd = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.4cd = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Cpp.unsigned_long_long = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.ee9 = impl_witness_access constants.%ImplicitAs.impl_witness.fa4, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %int_1, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %Cpp.unsigned_long_long = call %bound_method.loc8(%int_1) [concrete = constants.%int_1.877]
// CHECK:STDOUT:   %.loc8_3: init %Cpp.unsigned_long_long = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.877]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %unsigned_long_long.ref.loc8 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc8: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Cpp.unsigned_long_long = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.4cd = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.4cd = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Cpp.unsigned_long_long = var %b.var_patt
// CHECK:STDOUT:   %a.ref: ref %Cpp.unsigned_long_long = name_ref a, %a
// CHECK:STDOUT:   %.loc9_42: %Cpp.unsigned_long_long = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc9: %.294 = impl_witness_access constants.%Copy.impl_witness.15f, element0 [concrete = constants.%Cpp.unsigned_long_long.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_42, %impl.elem0.loc9
// CHECK:STDOUT:   %Cpp.unsigned_long_long.as.Copy.impl.Op.call: init %Cpp.unsigned_long_long = call %bound_method.loc9(%.loc9_42)
// CHECK:STDOUT:   assign %b.var, %Cpp.unsigned_long_long.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %unsigned_long_long.ref.loc9 [concrete = constants.%Cpp.unsigned_long_long] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %unsigned_long_long.ref.loc9: type = name_ref unsigned_long_long, constants.%Cpp.unsigned_long_long [concrete = constants.%Cpp.unsigned_long_long]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Cpp.unsigned_long_long = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %u64.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %u64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %Cpp.unsigned_long_long) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
